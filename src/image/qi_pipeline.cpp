// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/qi_pipeline.cpp - see include/strata/image/qi_pipeline.hpp.

#include "strata/image/qi_pipeline.hpp"

#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strata::image {

namespace gpu = strata::gpu;

namespace {

constexpr uint64_t kMiB = 1ull << 20;
constexpr uint64_t kMargin = 512 * kMiB;      // the driver's own and the copies' room, kept free
constexpr uint64_t kScratch = 384 * kMiB;     // the products' dequantization scratch (the DiT's and the encoder's)

uint64_t clampu(uint64_t v, uint64_t lo, uint64_t hi) { return std::max(lo, std::min(v, hi)); }

}  // namespace

bool QiPipeline::open(const std::string& dit, const std::string& vae, const std::string& text, std::string& err) {
    s_ = core::Runtime::get().create_stream();
    dit_path_ = dit;
    vae_path_ = vae;
    text_path_ = text;
    size_t free = 0, total = 0;
    gpu::mem_info(&free, &total);
    te_size_ = std::make_unique<QiText>();
    dit_size_ = std::make_unique<QiDit>();
    vae_size_ = std::make_unique<QiVae>();
    if (!(te_size_->open(text, s_, err, true) && dit_size_->open(dit, s_, err, true) &&
          vae_size_->open(vae, s_, err, true)))
        return false;
    // the budget a request plans with: what was free before the parts, less a margin, within the small-card limit the
    // engine takes too
    auto_budget_ = free > kMargin ? (uint64_t) free - kMargin : 0;
    if (const char* lim = std::getenv("STRATA_VRAM_LIMIT_MIB")) {
        const uint64_t l = std::strtoull(lim, nullptr, 10) * kMiB;
        if (l > 0) auto_budget_ = std::min(auto_budget_, l > kMargin ? l - kMargin : 0);
    }
    return true;
}

void QiPipeline::ensure_open() {
    if (dit_ && vae_ && te_) return;
    std::string err;
    te_ = std::make_unique<QiText>();
    dit_ = std::make_unique<QiDit>();
    vae_ = std::make_unique<QiVae>();
    if (!(te_->open(text_path_, s_, err) && dit_->open(dit_path_, s_, err) && vae_->open(vae_path_, s_, err))) {
        te_.reset();
        dit_.reset();
        vae_.reset();
        throw std::runtime_error(err);
    }
    dit_->set_block_budget(dit_->block_min());   // nothing resident until a request plans its budget
    te_->set_layer_budget(te_->layer_min());
    dit_set_ = dit_->block_min();
    te_set_ = te_->layer_min();
    mlp_set_ = -1;
}

uint64_t QiPipeline::held() const {
    if (!(dit_ && vae_ && te_)) return 0;
    return dit_->weight_bytes() + vae_->weight_bytes() + kScratch + dit_->blocks_vram() + te_->layers_vram() +
           dit_->activations_held() + vae_->scratch_held();
}

void QiPipeline::fit(uint64_t reserve) {
    size_t free = 0, total = 0;
    gpu::mem_info(&free, &total);
    const uint64_t have = held() + (uint64_t) free;
    set_budget(have > reserve ? have - reserve : 0);
}

QiNeed QiPipeline::need(int64_t n_text, int64_t width, int64_t height) {
    const QiDit& dit = *dit_size_;
    const QiText& te = *te_size_;
    const QiVae& vae = *vae_size_;
    const int64_t h = height / 16, w = width / 16, T = n_text + h * w;
    const uint64_t fixed = dit.weight_bytes() + vae.weight_bytes() + kScratch;
    QiNeed n;
    n.least = fixed + std::max(dit.activation_bytes(T, 512), vae.scratch_min()) + dit.block_min() + te.layer_min();
    // the DiT's blocks resident (used 8 times an image), the encoder streamed (once a prompt), the VAE in one piece
    n.full = fixed + dit.block_bytes() + te.layer_min() + std::max(dit.activation_bytes(T), vae.scratch_bytes(h, w));
    n.held = held();
    size_t free = 0, total = 0;
    gpu::mem_info(&free, &total);
    n.free = free;
    return n;
}

void QiPipeline::set_budget(uint64_t bytes) {
    budget_ = bytes;
    if (!(dit_ && vae_ && te_)) return;   // released: the next request plans with it
    const QiPlan p = plan(0, 0, 0);   // shrink at once what no longer fits (a grown budget is taken at the next request)
    (void) p;
}

void QiPipeline::release(bool all) {
    if (!(dit_ && vae_ && te_)) return;
    if (all) {   // every buffer the parts hold goes with them; the next request opens them again
        gpu::stream_sync(s_);
        te_.reset();
        dit_.reset();
        vae_.reset();
        return;
    }
    dit_->drop_blocks();
    te_->drop_layers();
    dit_->drop_activations();
    vae_->drop_scratch();
    dit_set_ = te_set_ = 0;
    mlp_set_ = -1;
}

QiPlan QiPipeline::plan(int64_t T, int64_t h, int64_t w) {
    QiPlan p;
    p.budget = budget_ > 0 ? budget_ : auto_budget_;
    p.fixed = dit_->weight_bytes() + vae_->weight_bytes() + kScratch;
    if (p.budget <= p.fixed) throw std::runtime_error("image engine: the VRAM budget (" + std::to_string(p.budget >> 20) +
                                                      " MiB) is below the weights that always stay");
    const uint64_t avail = p.budget - p.fixed, dit_min = dit_->block_min(), te_min = te_->layer_min();
    // the DiT's activations, the MLP in bands of rows when the whole takes more than a third of what is there
    int64_t mlp = 0;
    p.activations = T > 0 ? dit_->activation_bytes(T) : 0;
    for (int64_t rows : {4096, 2048, 1024, 512}) {
        if (T == 0 || p.activations * 3 <= avail) break;
        mlp = rows;
        p.activations = dit_->activation_bytes(T, rows);
    }
    // the room after the layers: the DiT's activations, and later the VAE's least (they take it in turn)
    const uint64_t work = std::max(p.activations, vae_->scratch_min());
    if (avail < work + dit_min + te_min)
        throw std::runtime_error("image engine: the VRAM budget (" + std::to_string(p.budget >> 20) +
                                 " MiB) is too small for this image");
    p.blocks = clampu(avail - work - te_min, dit_min, dit_->block_bytes());
    p.layers = clampu(avail - work - p.blocks, te_min, te_->layer_bytes());
    p.mlp_rows = mlp;
    if (p.blocks != dit_set_) {
        dit_->set_block_budget(p.blocks);
        dit_set_ = p.blocks;
    }
    if (p.layers != te_set_) {
        te_->set_layer_budget(p.layers);
        te_set_ = p.layers;
    }
    if (T > 0 && mlp != mlp_set_) {
        dit_->set_mlp_rows(mlp);
        mlp_set_ = mlp;
    }
    if (T == 0) dit_->drop_activations();   // a budget change between requests: the activations come back on use
    p.resident_blocks = dit_->resident_blocks();
    p.resident_layers = te_->resident_layers();
    // the VAE: what the layers leave, the DiT's activations' share included (freed if it needs it), in tiles when even
    // that is short
    const uint64_t left = avail - p.blocks - p.layers;
    p.vae = left;
    p.vae_tiled = h > 0 && vae_->scratch_bytes(h, w) > left;
    return p;
}

std::vector<uint8_t> QiPipeline::generate(const QiRequest& r, QiTimes* times, QiPlan* plan_out,
                                          const std::function<bool(int, int)>& step) {
    using clock = std::chrono::steady_clock;
    if (r.width % 16 != 0 || r.height % 16 != 0 || r.width <= 0 || r.height <= 0)
        throw std::invalid_argument("image engine: the size must be a multiple of 16");
    if ((int64_t) r.ids.size() <= r.drop) throw std::invalid_argument("image engine: no prompt after the system turn");
    const int64_t h = r.height / 16, w = r.width / 16, n_text = (int64_t) r.ids.size() - r.drop;
    ensure_open();
    QiPlan p = plan(n_text + h * w, h, w);
    const auto t0 = clock::now();
    if (r.ids != last_ids_) {
        last_states_ = te_->encode(r.ids);
        last_ids_ = r.ids;
    } else {
        p.text_cached = true;
    }
    const int64_t C = te_->hidden();
    dit_->set_context(last_states_.data() + r.drop * C, n_text);
    const auto t1 = clock::now();
    const int64_t n = dit_->channels() * h * w, HW = r.width * r.height;
    const auto noise = philox_randn(r.seed, (uint64_t) n);
    float *x = nullptr, *rgba = nullptr;
    if (!gpu::alloc_device(&x, (size_t) n * 4) || !gpu::alloc_device(&rgba, (size_t) (4 * HW) * 4))
        throw std::runtime_error(std::string("image engine: GPU memory: ") + gpu::last_error());
    std::vector<float> out((size_t) (4 * HW));
    try {
        gpu::copy(x, noise.data(), (size_t) n * 4);
        dit_->sample(x, h, w, r.sigmas.empty() ? turbo_sigmas() : r.sigmas, step);
        const auto t2 = clock::now();
        // the VAE beside the DiT's activations when it fits there; else they give way to it
        uint64_t vae_room = p.vae > p.activations ? p.vae - p.activations : 0;
        if (vae_->scratch_bytes(h, w) > vae_room) {
            dit_->drop_activations();
            vae_room = p.vae;
        }
        if (vae_->scratch_bytes(h, w) > vae_room && te_->layers_vram() > 0) {
            // the encoder's layers too (the next prompt streams them back; the same prompt's states are kept)
            te_->drop_layers();
            te_set_ = 0;
            vae_room += p.layers;
            p.vae_tiled = vae_->scratch_bytes(h, w) > vae_room;
        }
        vae_->decode(x, h, w, rgba, vae_room);
        if (p.vae_tiled) vae_->drop_scratch();   // a tile's buffers: not worth keeping beside the layers
        gpu::copy(out.data(), rgba, out.size() * 4);
        const auto t3 = clock::now();
        if (times != nullptr) {
            times->text = std::chrono::duration<double>(t1 - t0).count();
            times->sampling = std::chrono::duration<double>(t2 - t1).count();
            times->decode = std::chrono::duration<double>(t3 - t2).count();
        }
    } catch (...) {
        gpu::free(x);
        gpu::free(rgba);
        throw;
    }
    gpu::free(x);
    gpu::free(rgba);
    if (plan_out != nullptr) *plan_out = p;
    std::vector<uint8_t> img((size_t) (4 * HW));
    for (int64_t q = 0; q < HW; ++q)
        for (int c = 0; c < 4; ++c)
            img[(size_t) (q * 4 + c)] = (uint8_t) std::lround(std::clamp(out[(size_t) (c * HW + q)], 0.0f, 1.0f) * 255.0f);
    return img;
}

}  // namespace strata::image
