// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/qi_pipeline.cpp - see include/strata/image/qi_pipeline.hpp.

#include "strata/image/qi_pipeline.hpp"

#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/image/qi_kernels.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>

namespace strata::image {

namespace gpu = strata::gpu;

namespace {

constexpr uint64_t kMiB = 1ull << 20;
constexpr uint64_t kMargin = 512 * kMiB;      // the driver's own and the copies' room, kept free
constexpr uint64_t kScratch = 384 * kMiB;     // the products' dequantization scratch (the DiT's and the encoder's)

uint64_t clampu(uint64_t v, uint64_t lo, uint64_t hi) { return std::max(lo, std::min(v, hi)); }

// RGBA8 [H][W][4] of `f` at W x H, nearest (as the references are resized)
std::vector<uint8_t> resized(const QiRef& f, int64_t W, int64_t H) {
    std::vector<uint8_t> out((size_t) (W * H * 4));
    for (int64_t y = 0; y < H; ++y)
        for (int64_t x = 0; x < W; ++x)
            std::copy_n(f.rgba.data() + ((y * f.height / H) * f.width + x * f.width / W) * 4, 4,
                        out.data() + (y * W + x) * 4);
    return out;
}

}  // namespace

void qi_ref_size(int64_t ref_w, int64_t ref_h, int64_t& w, int64_t& h) {
    // diffusers' calculate_dimensions(1024 * 1024, ratio): the width the root of the area times the ratio, the height the
    // width over it, each rounded to a multiple of 32 as Python's round does (half to even: nearbyint)
    const double ratio = (double) ref_w / (double) ref_h;
    const double vw = std::sqrt(1024.0 * 1024.0 * ratio), vh = vw / ratio;
    w = std::max<int64_t>(32, (int64_t) std::nearbyint(vw / 32) * 32);
    h = std::max<int64_t>(32, (int64_t) std::nearbyint(vh / 32) * 32);
}

uint64_t vram_cap(uint64_t total) {
    uint64_t cap = total > kMargin ? total - kMargin : 0;
    if (const char* lim = std::getenv("STRATA_VRAM_LIMIT_MIB")) {
        const uint64_t l = std::strtoull(lim, nullptr, 10) * kMiB;
        if (l > 0) cap = std::min(cap, l > kMargin ? l - kMargin : 0);
    }
    return cap;
}

bool QiPipeline::open(const std::string& dit, const std::string& vae, const std::string& text, std::string& err,
                      const std::string& vision) {
    s_ = core::Runtime::get().create_stream();
    dit_path_ = dit;
    vae_path_ = vae;
    text_path_ = text;
    vision_path_ = vision;
    if (!vision.empty()) {
        vision_size_ = std::make_unique<QiVision>();
        if (!vision_size_->open(vision, s_, err, true)) return false;
    }
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
    auto_budget_ = vram_cap(free);
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

QiNeed QiPipeline::need(int64_t n_text, int64_t width, int64_t height,
                        const std::vector<std::pair<int64_t, int64_t>>& refs, bool mask, int64_t n_neg) {
    const QiDit& dit = *dit_size_;
    const QiText& te = *te_size_;
    const QiVae& vae = *vae_size_;
    const int64_t h = height / 16, w = width / 16;
    int64_t T = n_text + h * w, te_T = n_text + 16;   // the encoder's tokens: the system turn's too
    uint64_t edit = 0;   // an edit's reference work before the sampling: the vision encoder and the VAE's encoder
    for (const auto& [rw0, rh0] : refs) {
        int64_t rw = 0, rh = 0;
        qi_ref_size(rw0, rh0, rw, rh);
        T += (rh / 16) * (rw / 16) - 1;   // the pad's place: the reference's latent tokens
        te_T += (rh / 32) * (rw / 32) - 1;   // and in the encoder its image tokens
        const uint64_t v = vision_size_ ? vision_size_->weight_bytes() + vision_size_->scratch_bytes((rh / 16) * (rw / 16)) : 0;
        edit = std::max(edit, v + vae.encode_scratch_bytes(rh, rw) + (uint64_t) (4 * rh * rw) * 4);
    }
    // a mask: the first reference encoded again at the image's size; the base latent, the noise and the mask kept
    if (mask) edit = std::max(edit, vae.encode_scratch_bytes(height, width) + (uint64_t) (4 * height * width) * 4);
    const uint64_t inpaint = mask ? (uint64_t) ((2 * dit.channels() + 1) * h * w) * 4 : 0;
    // guidance: the negative prompt's prefix beside the first (the same references), and its sequence
    // its activations: both sequences side by side after the first step (forward_pair)
    const int64_t neg_pre = n_neg > 0 ? n_neg + (T - h * w - n_text) : 0,
                  T_act = n_neg > 0 ? qi_attn_rows(T) + qi_attn_rows(neg_pre + h * w) : T;
    const uint64_t fixed = dit.weight_bytes() + vae.weight_bytes() + kScratch + dit.prefix_row_bytes(T - h * w) +
                           dit.prefix_row_bytes(neg_pre) + inpaint;
    // the prefix cache: in VRAM in the plan's room, the rest in pinned RAM (least: all of it there)
    const uint64_t kv = dit.prefix_kv_bytes(T - h * w) + dit.prefix_kv_bytes(neg_pre), kv_all = kv * (uint64_t) dit.blocks();
    QiNeed n;
    if (n_neg > 0) te_T = std::max(te_T, te_T - n_text + n_neg);
    T = T_act;
    edit = std::max(edit, te.activation_bytes(te_T));   // the encoder's own buffers, before the DiT's
    n.least = fixed + std::max({dit.activation_bytes(T, 512), vae.scratch_min(), edit}) + dit.block_min() +
              te.layer_min();
    // the DiT's blocks resident (used 8 times an image), the encoder streamed (once a prompt), the VAE in one piece
    n.full = fixed + kv_all + dit.block_bytes() + te.layer_min() +
             std::max({dit.activation_bytes(T), vae.scratch_bytes(h, w), edit});
    n.held = held();
    size_t free = 0, total = 0;
    gpu::mem_info(&free, &total);
    n.free = free;
    n.cap = vram_cap(total);
    const uint64_t room = n.cap > n.least ? n.cap - n.least : 0;
    n.host = kv == 0 ? 0 : kv_all - std::min<uint64_t>(room / kv, (uint64_t) dit.blocks()) * kv;
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

QiPlan QiPipeline::plan(int64_t T, int64_t h, int64_t w, int64_t n_ids, int64_t n_neg_pre) {
    QiPlan p;
    p.budget = budget_ > 0 ? budget_ : auto_budget_;
    // the prefix's rows stay through the sampling and the decode (an edit's: its references' tokens too)
    const int64_t n_pre = T > 0 ? T - h * w : 0;
    p.fixed = dit_->weight_bytes() + vae_->weight_bytes() + kScratch + dit_->prefix_row_bytes(n_pre) +
              dit_->prefix_row_bytes(n_neg_pre);
    // the activations: guidance's both sequences side by side after the first step (forward_pair)
    if (T > 0 && n_neg_pre > 0) T = qi_attn_rows(T) + qi_attn_rows(n_neg_pre + h * w);
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
    // the encoder's buffers too: they come and go before the DiT's (generate frees them after the encode)
    const uint64_t work = std::max({p.activations, vae_->scratch_min(), te_->activation_bytes(n_ids)});
    if (avail < work + dit_min + te_min)
        throw std::runtime_error("image engine: the VRAM budget (" + std::to_string(p.budget >> 20) +
                                 " MiB) is too small for this image");
    // the prefix cache first: a block's that is not in VRAM is copied in at every step with the compute waiting (the
    // blocks' weights stream beside it); then the DiT's blocks (8 or more uses an image), then the encoder's layers
    const uint64_t kv = dit_->prefix_kv_bytes(n_pre) + dit_->prefix_kv_bytes(n_neg_pre),
                   spare = avail - work - dit_min - te_min;
    p.resident_kv = kv > 0 ? (int) std::min<uint64_t>(spare / kv, (uint64_t) dit_->blocks()) : dit_->blocks();
    p.prefix_kv = (uint64_t) p.resident_kv * kv;
    dit_->set_prefix_vram_blocks(p.resident_kv);
    p.blocks = clampu(avail - work - te_min - p.prefix_kv, dit_min, dit_->block_bytes());
    p.layers = clampu(avail - work - p.blocks - p.prefix_kv, te_min, te_->layer_bytes());
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
    const uint64_t left = avail - p.blocks - p.layers - p.prefix_kv;
    p.vae = left;
    p.vae_tiled = h > 0 && vae_->scratch_bytes(h, w) > left;
    return p;
}

void QiPipeline::edit_refs(const QiRequest& r, std::vector<int32_t>& ids, std::vector<QiVisionOut>& vouts,
                           std::vector<QiText::Image>& te_images, std::vector<QiDit::Ref>& dit_refs,
                           std::vector<float*>& ref_z, float** base_z, std::vector<int32_t>& neg_ids,
                           std::vector<QiText::Image>& neg_te, std::vector<QiDit::Ref>& neg_refs) {
    if (vision_path_.empty()) throw std::invalid_argument("image engine: an edit needs the vision encoder (--vision)");
    if (std::count(r.ids.begin(), r.ids.end(), kImagePad) != (int64_t) r.refs.size())
        throw std::invalid_argument("image engine: the prompt's image pads and the reference images differ in number");
    // room for the vision encoder and the VAE's encoder: the cache gives way when the budget is short of them
    std::vector<std::pair<int64_t, int64_t>> sizes;
    sizes.reserve(r.refs.size());
    for (const QiRef& f : r.refs) sizes.emplace_back(f.width, f.height);
    const bool mask = !r.mask.rgba.empty();
    const QiNeed nd = need(0, r.width, r.height, sizes, mask);
    const uint64_t edit = nd.least - need(0, r.width, r.height).least;
    if (held() + edit > budget()) release(false);
    ensure_open();
    vision_ = std::make_unique<QiVision>();
    std::string err;
    if (!vision_->open(vision_path_, s_, err)) {
        vision_.reset();
        throw std::runtime_error(err);
    }
    std::vector<std::pair<int64_t, int64_t>> grids;   // each reference's worked size
    for (const QiRef& f : r.refs) {
        if (f.width <= 0 || f.height <= 0 || f.rgba.size() != (size_t) (f.width * f.height * 4))
            throw std::invalid_argument("image engine: a reference image is empty");
        int64_t rw = 0, rh = 0;
        qi_ref_size(f.width, f.height, rw, rh);
        // resized as stable-diffusion.cpp resizes it (nearest), 0 .. 1; the vision encoder's input over white
        std::vector<float> rgba((size_t) (4 * rh * rw)), rgb((size_t) (3 * rh * rw));
        for (int64_t y = 0; y < rh; ++y)
            for (int64_t x = 0; x < rw; ++x) {
                const uint8_t* px = f.rgba.data() + ((y * f.height / rh) * f.width + x * f.width / rw) * 4;
                const float a = (float) px[3] / 255.f;
                for (int c = 0; c < 4; ++c) rgba[(size_t) ((c * rh + y) * rw + x)] = (float) px[c] / 255.f;
                for (int c = 0; c < 3; ++c)
                    rgb[(size_t) ((c * rh + y) * rw + x)] = 2.f * ((float) px[c] / 255.f * a + 1.f - a) - 1.f;
            }
        vouts.push_back(vision_->encode(rgb.data(), rw, rh));
        float *d = nullptr, *z = nullptr;
        if (!gpu::alloc_device(&z, (size_t) (64 * (rh / 16) * (rw / 16)) * 4))
            throw std::runtime_error(std::string("image engine: GPU memory: ") + gpu::last_error());
        ref_z.push_back(z);   // the caller frees it, whatever happens below
        if (!gpu::alloc_device(&d, rgba.size() * 4))
            throw std::runtime_error(std::string("image engine: GPU memory: ") + gpu::last_error());
        try {
            gpu::copy(d, rgba.data(), rgba.size() * 4);
            vae_->encode(d, rh, rw, z);
        } catch (...) {
            gpu::free(d);
            throw;
        }
        gpu::free(d);
        grids.emplace_back(rw, rh);
    }
    if (mask) {   // the first reference at the image's size: the latent the part the mask keeps follows
        const std::vector<uint8_t> px = resized(r.refs[0], r.width, r.height);
        const int64_t HW = r.width * r.height;
        std::vector<float> rgba((size_t) (4 * HW));
        for (int64_t q = 0; q < HW; ++q)
            for (int c = 0; c < 4; ++c) rgba[(size_t) (c * HW + q)] = (float) px[(size_t) (q * 4 + c)] / 255.f;
        float* d = nullptr;
        if (!gpu::alloc_device(&d, rgba.size() * 4) ||
            !gpu::alloc_device(base_z, (size_t) (64 * (r.height / 16) * (r.width / 16)) * 4))
            throw std::runtime_error(std::string("image engine: GPU memory: ") + gpu::last_error());
        gpu::copy(d, rgba.data(), rgba.size() * 4);
        vae_->encode(d, r.height, r.width, *base_z);
        gpu::free(d);
    }
    vision_.reset();   // its weights go: an edit opens it again
    vae_->drop_scratch();
    // the ids with each pad widened to its reference's tokens: the prompt's, and guidance's negative prompt's
    auto widen = [&](const std::vector<int32_t>& src, std::vector<int32_t>& out, std::vector<QiText::Image>& te,
                     std::vector<QiDit::Ref>& refs) {
        size_t k = 0;
        for (const int32_t id : src) {
            if (id != kImagePad) {
                out.push_back(id);
                continue;
            }
            if (k >= vouts.size())
                throw std::invalid_argument("image engine: the negative prompt's image pads and the references differ");
            const QiVisionOut& v = vouts[k];
            QiText::Image im;
            im.index = (int64_t) out.size();
            im.rows = v.rows;
            im.grid_h = v.grid_h;
            im.grid_w = v.grid_w;
            im.main = v.main.data();
            for (const auto& d : v.deepstack) im.deepstack.push_back(d.data());
            te.push_back(im);
            QiDit::Ref f;
            f.slot = im.index;
            f.slots = v.rows;
            f.w = grids[k].first / 16;
            f.h = grids[k].second / 16;
            f.latent = ref_z[k];
            refs.push_back(f);
            out.insert(out.end(), (size_t) v.rows, kImagePad);
            ++k;
        }
    };
    widen(r.ids, ids, te_images, dit_refs);
    if (r.cfg > 1.0f && !r.neg_ids.empty()) widen(r.neg_ids, neg_ids, neg_te, neg_refs);
}

std::vector<uint8_t> QiPipeline::generate(const QiRequest& r, QiTimes* times, QiPlan* plan_out,
                                          const std::function<bool(int, int)>& step,
                                          const std::function<void(const char*)>& stage) {
    using clock = std::chrono::steady_clock;
    if (r.width % 16 != 0 || r.height % 16 != 0 || r.width <= 0 || r.height <= 0)
        throw std::invalid_argument("image engine: the size must be a multiple of 16");
    if ((int64_t) r.ids.size() <= r.drop) throw std::invalid_argument("image engine: no prompt after the system turn");
    const int64_t h = r.height / 16, w = r.width / 16;
    const auto t0 = clock::now();
    ensure_open();
    // an edit: its references through the vision encoder and the VAE's encoder, their pads widened to their tokens
    std::vector<int32_t> ids;
    std::vector<QiVisionOut> vouts;
    std::vector<QiText::Image> te_images;
    std::vector<QiDit::Ref> dit_refs;
    std::vector<float*> ref_z;
    struct FreeZ {
        std::vector<float*>& z;
        ~FreeZ() {
            for (float* p : z) gpu::free(p);
        }
    } free_z{ref_z};
    const bool mask = !r.mask.rgba.empty();
    if (mask && (r.refs.empty() || r.mask.rgba.size() != (size_t) (r.mask.width * r.mask.height * 4)))
        throw std::invalid_argument("image engine: a mask needs its reference image, and it must not be empty");
    float* base_z = nullptr;
    const bool guided = r.cfg > 1.0f && !r.neg_ids.empty();
    std::vector<int32_t> neg_ids;   // an edit's negative prompt, its pads widened
    std::vector<QiText::Image> neg_te;
    std::vector<QiDit::Ref> neg_refs;
    if (!r.refs.empty()) {
        if (stage) stage("refs");
        edit_refs(r, ids, vouts, te_images, dit_refs, ref_z, &base_z, neg_ids, neg_te, neg_refs);
        if (base_z != nullptr) ref_z.push_back(base_z);
    }
    if (stage) stage("text");
    const std::vector<int32_t>& all_ids = r.refs.empty() ? r.ids : ids;
    const int64_t n_ctx = (int64_t) all_ids.size() - r.drop;
    int64_t n_pre = n_ctx;
    for (const QiDit::Ref& f : dit_refs) n_pre += f.h * f.w - f.slots;
    const std::vector<int32_t>& neg_all = r.refs.empty() ? r.neg_ids : neg_ids;
    const int64_t n_neg_ctx = guided ? (int64_t) neg_all.size() - r.drop : 0;
    if (guided && n_neg_ctx <= 0) throw std::invalid_argument("image engine: no negative prompt after the system turn");
    const int64_t n_neg_pre = guided ? n_neg_ctx + (n_pre - n_ctx) : 0;
    QiPlan p = plan(n_pre + h * w, h, w, (int64_t) std::max(all_ids.size(), neg_all.size()), n_neg_pre);
    if (!r.refs.empty()) {   // the references' states are not kept: another edit has other images
        last_states_ = te_->encode(all_ids, te_images);
        te_->drop_activations();
        last_ids_.clear();
    } else if (r.ids != last_ids_) {
        last_states_ = te_->encode(r.ids);
        te_->drop_activations();
        last_ids_ = r.ids;
    } else {
        p.text_cached = true;
    }
    const int64_t C = te_->hidden();
    for (QiDit::Ref& f : dit_refs) f.slot -= r.drop;
    dit_->set_context(last_states_.data() + r.drop * C, n_ctx, dit_refs);
    if (guided) {   // the negative prompt's states (kept for the same ids again when there are no references)
        if (!r.refs.empty()) {
            last_neg_states_ = te_->encode(neg_all, neg_te);
            last_neg_ids_.clear();
        } else if (r.neg_ids != last_neg_ids_) {
            last_neg_states_ = te_->encode(r.neg_ids);
            last_neg_ids_ = r.neg_ids;
        }
        te_->drop_activations();
        for (QiDit::Ref& f : neg_refs) f.slot -= r.drop;
        dit_->set_context(last_neg_states_.data() + r.drop * C, n_neg_ctx, neg_refs, true);
    } else {
        dit_->drop_second_context();
    }
    const auto t1 = clock::now();
    const int64_t n = dit_->channels() * h * w, HW = r.width * r.height;
    const auto noise = philox_randn(r.seed, (uint64_t) n);
    float *x = nullptr, *rgba = nullptr;
    if (!gpu::alloc_device(&x, (size_t) n * 4) || !gpu::alloc_device(&rgba, (size_t) (4 * HW) * 4)) {
        const std::string why = gpu::last_error();
        gpu::free(x);   // the first one, when the second was refused
        throw std::runtime_error("image engine: GPU memory: " + why);
    }
    std::vector<float> out((size_t) (4 * HW));
    // a mask: 1 for each latent token with a pixel of the mask less than opaque (the part that changes), the noise
    // the kept part's latent follows
    std::function<void(float*, float)> keep;
    std::vector<uint8_t> mask_px;
    if (mask) {
        mask_px = resized(r.mask, r.width, r.height);
        std::vector<float> m((size_t) (h * w), 0.f);
        for (int64_t y = 0; y < r.height; ++y)
            for (int64_t xx = 0; xx < r.width; ++xx)
                if (mask_px[(size_t) ((y * r.width + xx) * 4 + 3)] < 255) m[(size_t) ((y / 16) * w + xx / 16)] = 1.f;
        float *nz = nullptr, *mk = nullptr;
        for (auto [p, bytes] : {std::pair<float**, size_t>{&nz, (size_t) n * 4}, {&mk, m.size() * 4}}) {
            if (!gpu::alloc_device(p, bytes))
                throw std::runtime_error(std::string("image engine: GPU memory: ") + gpu::last_error());
            ref_z.push_back(*p);   // freed with the references' latents
        }
        gpu::copy(nz, noise.data(), (size_t) n * 4);
        gpu::copy(mk, m.data(), m.size() * 4);
        const int64_t C = dit_->channels();
        keep = [this, base_z, nz, mk, C, h, w](float* xs, float sigma) {
            qi_inpaint_blend(xs, base_z, nz, mk, sigma, C, h * w, s_);
        };
    }
    try {
        gpu::copy(x, noise.data(), (size_t) n * 4);
        if (stage) stage("sampling");
        dit_->sample(x, h, w, r.sigmas.empty() ? turbo_sigmas() : r.sigmas, step, keep, guided ? r.cfg : 1.0f);
        dit_->drop_host_prefix();
        const auto t2 = clock::now();
        if (stage) stage("decode");
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
        if (vae_->scratch_bytes(h, w) > vae_room && dit_set_ > dit_->block_min()) {
            // and as many of the DiT's resident blocks as the VAE still lacks: this image's sampling is done with
            // them, and the next one streams them back in (a budget a little short after a shrink tiled the VAE)
            const uint64_t give = std::min<uint64_t>(vae_->scratch_bytes(h, w) - vae_room, dit_set_ - dit_->block_min());
            dit_set_ -= give;
            dit_->set_block_budget(dit_set_);
            vae_room += give;
            p.vae_tiled = vae_->scratch_bytes(h, w) > vae_room;
        }
        vae_->decode(x, h, w, rgba, vae_room);
        // a tile's buffers are not worth keeping beside the layers, and buffers past the plan's share for the VAE (the
        // room the encoder's layers or the DiT's blocks gave it) are that room: the next request takes it back for
        // them (RTX 4070, 1920x1088: the second image's blocks were refused beside the first one's VAE buffers)
        if (p.vae_tiled || vae_->scratch_bytes(h, w) > p.vae) vae_->drop_scratch();
        gpu::copy(out.data(), rgba, out.size() * 4);
        const auto t3 = clock::now();
        if (times != nullptr) {
            times->text = std::chrono::duration<double>(t1 - t0).count();
            times->sampling = std::chrono::duration<double>(t2 - t1).count();
            times->decode = std::chrono::duration<double>(t3 - t2).count();
        }
    } catch (...) {
        dit_->drop_host_prefix();
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
    if (mask) {   // the kept part is the reference's own pixels (not through the VAE), blended by the mask's alpha
        const std::vector<uint8_t> base = resized(r.refs[0], r.width, r.height);
        for (int64_t q = 0; q < HW; ++q) {
            const int a = mask_px[(size_t) (q * 4 + 3)];
            for (int c = 0; c < 4; ++c) {
                const size_t i = (size_t) (q * 4 + c);
                img[i] = (uint8_t) ((a * base[i] + (255 - a) * img[i] + 127) / 255);
            }
        }
    }
    return img;
}

}  // namespace strata::image
