// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/hunyuanimage3_pipeline.cpp - see include/strata/image/hunyuanimage3_pipeline.hpp.
//
// The walk of one image, in the order the reference runs it
// (records/2026-10-23-hunyuanimage3-plan/CONFORM-2026-10-24.md, and `golden-cpu/` for the numbers every step here
// is held to):
//
//   ids -> trim at the last `<img>` -> the block's 2D rope positions
//   wte(text rows), and each step: the timestep/guidance/timestep_r rows over them, and patch_embed(latent) for
//   the block
//   32 MoE decoder layers over the whole sequence                (HunyuanImage3Dit, on the card)
//   final_layer(the block's rows, time_embed_2(t)) -> the velocity, and x += (sigma_next - sigma) * v
//   the VAE's decode of the last latent                          (HunyuanImage3Vae, on the card)
//   [-1,1] -> [0,1] -> RGBA8
#include "strata/image/hunyuanimage3_pipeline.hpp"

#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/image/hunyuanimage3_host.hpp"
#include "strata/image/qi_dit.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <stdexcept>

namespace strata::image {

namespace gpu = strata::gpu;

namespace {

// The Instruct-Distil's own sampling: 8 steps at shift 3, its guidance 2.5 (the golden's manifest agrees, and
// `probe sigmas` holds the schedule to the reference's FlowMatchDiscreteScheduler).  A request that gives sigmas
// or a guidance of its own overrides these; the steps the progress reports are the sigmas' less the terminal one.
constexpr int64_t kDistilSteps = 8;
constexpr float kDistilShift = 3.0f, kDistilGuidance = 2.5f;

// The latent's channels and the VAE's spatial downsample: an image of W x H is W/16 x H/16 tokens of 32 channels.
constexpr int64_t kLatentChannels = 32, kDownsample = 16;

uint64_t clampu(uint64_t v, uint64_t lo, uint64_t hi) { return std::max(lo, std::min(v, hi)); }

}  // namespace

bool HunyuanImage3Pipeline::open(const std::string& dit, const std::string& vae, std::string& err,
                                 int64_t max_layers) {
    s_ = core::Runtime::get().create_stream();
    dit_path_ = dit;
    vae_path_ = vae;
    max_layers_ = max_layers;
    size_t free = 0, total = 0;
    gpu::mem_info(&free, &total);
    dit_size_ = std::make_unique<HunyuanImage3Dit>();
    vae_size_ = std::make_unique<HunyuanImage3Vae>();
    around_size_ = std::make_unique<HunyuanImage3EntryExit>();
    if (!(dit_size_->open(dit, s_, err, max_layers, true) && vae_size_->open(vae, s_, err, true) &&
          around_size_->open(dit, s_, err, true)))
        return false;
    auto_budget_ = vram_cap(free);
    return true;
}

void HunyuanImage3Pipeline::ensure_open() {
    if (dit_ && vae_ && around_) return;
    std::string err;
    dit_ = std::make_unique<HunyuanImage3Dit>();
    vae_ = std::make_unique<HunyuanImage3Vae>();
    around_ = std::make_unique<HunyuanImage3EntryExit>();
    if (!dit_->open(dit_path_, s_, err, max_layers_)) {
        dit_.reset();
        vae_.reset();
        around_.reset();
        throw std::runtime_error(err);
    }
    if (!vae_->open(vae_path_, s_, err)) {
        dit_.reset();
        vae_.reset();
        around_.reset();
        throw std::runtime_error(err);
    }
    if (!around_->open(dit_path_, s_, err) || around_->hidden() != dit_->dims().hidden) {
        dit_.reset();
        vae_.reset();
        around_.reset();
        throw std::runtime_error(err.empty() ? "image engine: the entry and exit's hidden width is not the decoder's"
                                             : err);
    }
    dit_->set_bank_budget(dit_->bank_min());   // nothing resident until a request plans its budget
    banks_set_ = dit_->bank_min();
    text_ids_.clear();
    text_rows_.clear();
}

void HunyuanImage3Pipeline::set_budget(uint64_t bytes) {
    budget_ = bytes;
    if (dit_ == nullptr) return;   // released: the next request plans with it
    const QiPlan p = plan(dit_->tokens(), 0, 0);
    (void) p;   // a shrink takes effect at once; a grown budget is taken at the next request
}

void HunyuanImage3Pipeline::release(bool all) {
    if (dit_ == nullptr && vae_ == nullptr) return;
    if (all) {   // the parts go with it: the next request opens them again, dense products re-dequantized
        if (dit_) dit_->drop_sequence();
        if (vae_) vae_->drop_scratch();
        if (around_) around_->drop_grid();
        if (s_ != nullptr) gpu::stream_sync(s_);
        dit_.reset();
        vae_.reset();
        around_.reset();
        banks_set_ = 0;
        text_ids_.clear();
        text_rows_.clear();
        return;
    }
    dit_->drop_sequence();
    vae_->drop_scratch();
    if (around_) around_->drop_grid();
    dit_->set_bank_budget(dit_->bank_min());   // the streamed regions stay, the resident prefix gives its VRAM back
    banks_set_ = dit_->bank_min();
}

void HunyuanImage3Pipeline::fit(uint64_t reserve) {
    size_t free = 0, total = 0;
    gpu::mem_info(&free, &total);
    const uint64_t have = held() + (uint64_t) free;
    set_budget(have > reserve ? have - reserve : 0);
}

uint64_t HunyuanImage3Pipeline::held() const {
    if (dit_ == nullptr || vae_ == nullptr) return 0;
    return dit_->held() + vae_->weight_bytes() + vae_->scratch_held() + (around_ ? around_->held() : 0);
}

bool HunyuanImage3Pipeline::warm_up_request(QiRequest& r) const {
    // the smallest sequence the model's own markers make: `<boi>`, the size and ratio tokens, the three the step
    // writes over, and a 4 x 4 image block (64 x 64 pixels).  Nothing here is a picture; it is the graph's build.
    r.ids.clear();
    r.ids.push_back(kHunyuanImage3Boi);
    r.ids.push_back(kHunyuanImage3ImgSize256);   // the block is 4 x 4; the size token is the smallest the model has
    r.ids.push_back(kHunyuanImage3ImgRatio0);
    r.ids.push_back(kHunyuanImage3Timestep);
    r.ids.push_back(kHunyuanImage3Guidance);
    r.ids.push_back(kHunyuanImage3TimestepR);
    for (int64_t i = 0; i < (int64_t) (4 * 4); ++i) r.ids.push_back(kHunyuanImage3Image);
    r.drop = 0;
    r.width = r.height = 64;
    r.seed = 0;
    r.sigmas = {1.0f, 0.0f};   // one step
    r.guidance = 0.0f;
    return true;
}

bool HunyuanImage3Pipeline::read_sequence(const QiRequest& r, std::vector<int32_t>& ids, int64_t& T, int64_t& start,
                                          std::vector<int32_t>& y, std::vector<int32_t>& x, std::string& err) const {
    if (r.width % kDownsample != 0 || r.height % kDownsample != 0 || r.width <= 0 || r.height <= 0) {
        err = "image engine: the size must be a positive multiple of 16";
        return false;
    }
    if (r.ids.empty()) {
        err = "image engine: the prompt is empty";
        return false;
    }
    ids = r.ids;
    const int64_t keep = hunyuanimage3_trim(ids.data(), (int64_t) ids.size(), err);
    if (keep < 0) return false;
    ids.resize((size_t) keep);
    T = (int64_t) ids.size();
    const int64_t th = r.height / kDownsample, tw = r.width / kDownsample;
    if (!hunyuanimage3_positions(ids.data(), T, th, tw, y, x, start, err)) {
        err = err + ": the prompt's image block is not the " + std::to_string(th * tw) + " tokens of a " +
              std::to_string(r.width) + " x " + std::to_string(r.height) + " image";
        return false;
    }
    if (T <= start) {
        err = "image engine: the sequence holds no text before the image block";
        return false;
    }
    return true;
}

QiNeed HunyuanImage3Pipeline::need(int64_t n_text, int64_t width, int64_t height,
                                   const std::vector<std::pair<int64_t, int64_t>>&, bool, int64_t) {
    const int64_t th = height / kDownsample, tw = width / kDownsample;
    if (th <= 0 || tw <= 0 || n_text < th * tw + 1) {
        throw std::invalid_argument("image engine: the ids do not hold the image block and some text");
    }
    const uint64_t fixed = dit_size_->dense_bytes() + vae_size_->weight_bytes() + around_size_->weight_bytes();
    const uint64_t act = dit_size_->activation_bytes(n_text) + around_size_->scratch_bytes(th, tw);
    const uint64_t vae = vae_size_->scratch_bytes(th, tw);
    QiNeed n;
    // least: every bank streamed (two slots of the largest), the DiT's sequence buffers and the VAE's taken in
    // turn - the decode runs after the walk gives them back.  full: every bank resident as well, both buffers kept.
    n.least = fixed + dit_size_->bank_min() + std::max(act, vae);
    n.full = fixed + dit_size_->bank_bytes() + act + vae;
    n.held = held();
    size_t free = 0, total = 0;
    gpu::mem_info(&free, &total);
    n.free = free;
    n.cap = vram_cap(total);
    const uint64_t room = n.cap > fixed + std::max(act, vae) ? n.cap - fixed - std::max(act, vae) : 0;
    n.host = dit_size_->bank_bytes() - std::min(dit_size_->bank_bytes(), room);
    return n;
}

QiPlan HunyuanImage3Pipeline::plan(int64_t T, int64_t th, int64_t tw) {
    QiPlan p;
    p.budget = budget_ > 0 ? budget_ : auto_budget_;
    p.fixed = dit_->dense_bytes() + vae_->weight_bytes() + around_->weight_bytes();
    if (p.budget <= p.fixed)
        throw std::runtime_error("image engine: the VRAM budget (" + std::to_string(p.budget >> 20) +
                                 " MiB) is below the weights that always stay (" +
                                 std::to_string(p.fixed >> 20) + " MiB)");
    const uint64_t avail = p.budget - p.fixed;
    // the DiT's buffers for the sequence, and the VAE's for the picture: one at a time (generate frees the
    // sequence's before the decode), so the plan keeps the larger of the two out of the banks' way
    p.activations = T > 0 ? dit_->activation_bytes(T) : 0;
    p.activations += around_->scratch_bytes(th, tw);   // the entry and the exit's grid, alive over the same walk
    p.vae = th > 0 ? vae_->scratch_bytes(th, tw) : 0;
    const uint64_t work = std::max(p.activations, p.vae);
    if (avail < work + dit_->bank_min())
        throw std::runtime_error("image engine: the VRAM budget (" + std::to_string(p.budget >> 20) +
                                 " MiB) is too small for this image: its walk needs " +
                                 std::to_string((work + dit_->bank_min()) >> 20) + " MiB beside the weights");
    p.blocks = clampu(avail - work, dit_->bank_min(), dit_->bank_bytes());
    p.resident_blocks = 0;   // this model's streamed units are its layers' banks
    p.layers = dit_->bank_bytes();
    p.resident_layers = 0;   // set when the budget is applied
    p.streamed = 0;          // as above: the bytes a pass reads for the banks that are not resident
    p.vae_tiled = false;     // the DCAE VAE is not tiled: its group norms take the whole image
    if (p.blocks != banks_set_) {
        dit_->set_bank_budget(p.blocks);
        banks_set_ = p.blocks;
    }
    p.resident_layers = dit_->resident_banks();
    p.streamed = dit_->bank_bytes() - dit_->bank_vram();
    return p;
}

std::vector<uint8_t> HunyuanImage3Pipeline::generate(const QiRequest& r, QiTimes* times, QiPlan* plan_out,
                                                     const std::function<bool(int, int)>& step,
                                                     const std::function<void(const char*)>& stage) {
    using clock = std::chrono::steady_clock;
    if (!r.refs.empty() || !r.mask.rgba.empty())
        throw std::invalid_argument("image engine: HunyuanImage 3.0 does not take reference images yet");
    if (!r.neg_ids.empty() || r.cfg > 1.0f)
        throw std::invalid_argument("image engine: HunyuanImage 3.0 samples once: no negative prompt, no cfg");
    if (r.drop != 0)
        throw std::invalid_argument("image engine: HunyuanImage 3.0 reads its whole sequence: drop must be 0");
    const auto t0 = clock::now();
    std::string err;
    std::vector<int32_t> ids;
    int64_t T = 0, start = 0;
    std::vector<int32_t> y, x;
    if (!read_sequence(r, ids, T, start, y, x, err)) throw std::invalid_argument(err);
    ensure_open();
    const int64_t hidden = dit_->dims().hidden;
    QiPlan p = plan(T, r.height / kDownsample, r.width / kDownsample);
    if (!dit_->set_sequence(T, (T - start), y.data(), x.data(), err))
        throw std::runtime_error("image engine: " + err);

    // ---- the text rows: `wte` over the ids before the block, gathered once for the sequence and kept for the same
    // ids coming again.  The walk rewrites every row of the sequence, text included, so the cached rows go back into
    // the buffer at each step - the host reference re-gathers them per step (`hidden_in_host`), which is why the two
    // paths agree at one step and only there when this copy is missed.
    if (stage) stage("text");
    std::vector<float> hid((size_t) (T * hidden));
    if (ids == text_ids_) {
        p.text_cached = true;
    } else {
        if (!hunyuanimage3_wte_host(*dit_->file(), ids.data(), start, hid.data(), err))
            throw std::runtime_error("image engine: " + err);
        text_ids_ = ids;
        text_rows_.assign(hid.begin(), hid.begin() + start * hidden);
    }
    auto restore_text_rows = [&] {
        std::copy(text_rows_.begin(), text_rows_.end(), hid.begin());
    };

    // ---- the schedule: the request's sigmas if it gives them, else its step count through the model's shift,
    // else the model's own 8 steps (`--steps`, or `STEPS:` on an IMG line, or neither: the Distil's default)
    std::vector<float> sigmas = r.sigmas;
    if (sigmas.empty()) {
        const int64_t wanted = r.steps > 0 ? r.steps : kDistilSteps;
        sigmas.resize((size_t) (wanted + 1));
        hunyuanimage3_sigmas(wanted, kDistilShift, sigmas.data());
    }
    const int64_t steps = (int64_t) sigmas.size() - 1;
    const float guidance_scale = r.guidance > 0.0f ? r.guidance : kDistilGuidance;
    const float guidance = 1000.0f * guidance_scale;   // the number the guidance token embeds
    const int64_t th = r.height / kDownsample, tw = r.width / kDownsample;
    const int64_t n_img = th * tw, n_lat = kLatentChannels * n_img;
    float* img = hid.data() + (size_t) (start * hidden);   // the block's rows, where patch_embed writes
    if (!around_->set_grid(th, tw, err)) throw std::runtime_error("image engine: " + err);

    std::vector<float> lat((size_t) n_lat), vel((size_t) n_lat);
    if (!init_latent_.empty()) {
        if ((int64_t) init_latent_.size() != n_lat)
            throw std::invalid_argument("image engine: the starting latent is not this size's [32][" +
                                        std::to_string(th) + "][" + std::to_string(tw) + "]");
        std::copy(init_latent_.begin(), init_latent_.end(), lat.begin());
    } else {
        const auto noise = philox_randn(r.seed, (uint64_t) n_lat);
        std::copy(noise.begin(), noise.end(), lat.begin());
    }

    const auto t1 = clock::now();
    if (stage) stage("sampling");
    float *dz = nullptr, *drgb = nullptr;
    try {
        for (int64_t s = 0; s < steps; ++s) {
            const float sigma = sigmas[(size_t) s], sigma_next = sigmas[(size_t) (s + 1)];
            const float t = sigma * 1000.0f, t_r = sigma_next * 1000.0f;
            restore_text_rows();
            // STRATA_HUNYUANIMAGE3_HOST_ENDS=patch|final|both: a check's - the host FP32 pieces in place of the
            // card's, so one run at a time can tell either piece's cost in the picture (see the pipeline header)
            static const std::string kHostEnds = [] {
                const char* v = std::getenv("STRATA_HUNYUANIMAGE3_HOST_ENDS");
                return std::string(v != nullptr ? v : "");
            }();
            const bool host_patch = kHostEnds == "patch" || kHostEnds == "both";
            const bool host_final = kHostEnds == "final" || kHostEnds == "both";
            if (!hunyuanimage3_meta_rows_host(*dit_->file(), ids.data(), start, t, t_r, guidance, hid.data(), err) ||
                (host_patch ? !hunyuanimage3_patch_embed_host(*dit_->file(), lat.data(), th, tw, t, img, err)
                            : !around_->patch_embed(lat.data(), th, tw, t, img, err)))
                throw std::runtime_error("image engine: " + err);
            if (!dit_->forward(hid.data(), hid.data(), dit_->layers(), {}, err))
                throw std::runtime_error("image engine: the DiT: " + err);
            if (host_final ? !hunyuanimage3_final_layer_host(*dit_->file(), img, th, tw, t, vel.data(), err)
                           : !around_->final_layer(img, th, tw, t, vel.data(), err))
                throw std::runtime_error("image engine: " + err);
            hunyuanimage3_euler_step(lat.data(), vel.data(), n_lat, sigma, sigma_next);
            if (step && !step((int) (s + 1), (int) steps)) throw QiStopped();
        }
        if (times != nullptr) times->sampling = std::chrono::duration<double>(clock::now() - t1).count();

        // ---- the VAE.  The sequence's buffers are the DiT's last hold on the card, and the decode is the bigger
        // worker at a full image's size: give them back first, then what the banks leave is the VAE's.  The banks'
        // resident prefix is this model's cache between requests, but they are 43 GB and the card is not, so a
        // decode that needs the room takes it from them and the next image's walk streams the banks in again (as
        // QiPipeline frees the blocks and the encoder's layers the VAE lacks room beside).
        dit_->drop_sequence();
        around_->drop_grid();   // the entry and the exit's buffers too: the decode is the bigger worker
        uint64_t vae_room = 0;
        const uint64_t vae_need = vae_->scratch_bytes(th, tw);
        if (vae_need <= p.budget - p.fixed)
            vae_room = p.budget - p.fixed - std::min(p.budget - p.fixed, dit_->bank_vram());
        if (vae_need > vae_room && dit_->bank_vram() > dit_->bank_min()) {
            dit_->set_bank_budget(dit_->bank_min());   // every resident bank gives way to the decode
            banks_set_ = dit_->bank_min();
            vae_room = p.budget > p.fixed + dit_->bank_vram() ? p.budget - p.fixed - dit_->bank_vram() : 0;
        }
        if (vae_need > vae_room)
            throw std::runtime_error("image engine: the VAE's decode of a " + std::to_string(r.width) + " x " +
                                     std::to_string(r.height) + " image needs " +
                                     std::to_string(vae_need >> 20) + " MiB beside the weights and the streamed "
                                     "slots, and the budget leaves " + std::to_string(vae_room >> 20) +
                                     " (the VAE's tiled decode is not built)");
        if (stage) stage("decode");
        const auto t2 = clock::now();
        const int64_t HW = r.width * r.height;
        if (!gpu::alloc_device(&dz, (size_t) n_lat * sizeof(float)) ||
            !gpu::alloc_device(&drgb, (size_t) (3 * HW) * sizeof(float))) {
            if (dz != nullptr) gpu::free(dz);
            if (drgb != nullptr) gpu::free(drgb);
            throw std::runtime_error(std::string("image engine: GPU memory: ") + gpu::last_error());
        }
        gpu::copy(dz, lat.data(), (size_t) n_lat * sizeof(float));
        vae_->decode(dz, th, tw, drgb);
        std::vector<float> rgb((size_t) (3 * HW));
        gpu::copy(rgb.data(), drgb, rgb.size() * sizeof(float));
        gpu::free(dz);
        gpu::free(drgb);
        dz = drgb = nullptr;
        if (times != nullptr) times->decode = std::chrono::duration<double>(clock::now() - t2).count();

        // the reference pipeline's postprocess: [-1,1] -> [0,1], the three planes into RGBA8 rows
        std::vector<uint8_t> out((size_t) (4 * HW));
        for (int64_t q = 0; q < HW; ++q) {
            for (int c = 0; c < 3; ++c) {
                const float v = std::clamp(rgb[(size_t) (c * HW + q)] / 2.0f + 0.5f, 0.0f, 1.0f);
                out[(size_t) (q * 4 + c)] = (uint8_t) std::lround(v * 255.0f);
            }
            out[(size_t) (q * 4 + 3)] = 255;
        }
        if (times != nullptr) times->text = std::chrono::duration<double>(t1 - t0).count();
        if (plan_out != nullptr) {
            p.budget = budget();   // what the next request plans with (the banks may have given room to the VAE)
            *plan_out = p;
        }
        return out;
    } catch (...) {
        if (dz != nullptr) gpu::free(dz);
        if (drgb != nullptr) gpu::free(drgb);
        dit_->drop_sequence();   // a stopped or failed request gives its buffers back
        around_->drop_grid();
        throw;
    }
}

}  // namespace strata::image
