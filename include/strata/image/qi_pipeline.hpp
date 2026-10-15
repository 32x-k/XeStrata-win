// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/qi_pipeline.hpp - Qwen-Image-2.1's text to image on the GPU: the text encoder's states of the
// prompt's ids, the DiT's flow sampling from the seed's noise, and the VAE's image, within a VRAM budget.
//
// The budget is split per request (plan): the weights that always stay (the DiT's small ones, the VAE's), the
// products' scratch and the DiT's activations come first (the MLP in bands of rows when the whole would take too
// much); then the DiT's blocks get what remains, all of them when they fit (they are used 8 times a request), past
// the text encoder's two slots; then the text encoder's layers.  The VAE decodes in what is left, the DiT's
// activations freed when it needs them, in tiles when even that is too little.  What a request leaves in VRAM (the
// layers' resident prefixes, the activations) stays for the next one: the cache.  The same prompt's text states are
// kept too.
#pragma once

#include "strata/image/qi_dit.hpp"
#include "strata/image/qi_text.hpp"
#include "strata/image/qi_vae.hpp"
#include "strata/image/qi_vision.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace strata::image {

/// An edit's reference image: RGBA8 [height][width][4] as read (qi_image_io).
struct QiRef {
    std::vector<uint8_t> rgba;
    int64_t width = 0, height = 0;
};

struct QiRequest {
    std::vector<int32_t> ids;   // the prompt in the model's chat form, tokenized; an edit's one <|image_pad|> per ref
    int64_t drop = 0;           // the leading ids whose states the DiT does not take (the system turn)
    int64_t width = 1024, height = 1024;   // multiples of 16
    uint64_t seed = 0;
    std::vector<float> sigmas;  // empty: Turbo's 8 steps
    std::vector<QiRef> refs;    // an edit's reference images, in the order of their pads
};

/// The size an edit's reference image is worked at: the target's pixel count, the reference's shape, sides rounded to
/// multiples of 32 (as stable-diffusion.cpp sizes it for Qwen-Image-2.1).
void qi_ref_size(int64_t ref_w, int64_t ref_h, int64_t width, int64_t height, int64_t& w, int64_t& h);

/// The token id of <|image_pad|> in Qwen3-VL's vocabulary.
constexpr int32_t kImagePad = 151655;

struct QiTimes {
    double text = 0, sampling = 0, decode = 0;
};

/// How a request's budget was split (bytes; the report).
struct QiPlan {
    uint64_t budget = 0, fixed = 0, activations = 0, blocks = 0, layers = 0, vae = 0;
    int resident_blocks = 0, resident_layers = 0;
    int64_t mlp_rows = 0;   // 0: all
    bool vae_tiled = false, text_cached = false;
};

/// The VRAM an image needs (bytes): `least`, the budget it runs in at all (the layers streamed, the MLP in bands, the
/// VAE in tiles); `full`, the budget with the DiT's blocks resident and the VAE in one piece (the encoder, run once a
/// prompt, streamed); `held`, what the engine holds now; `free`, the device's free VRAM.
struct QiNeed {
    uint64_t least = 0, full = 0, held = 0, free = 0;
};

class QiPipeline {
public:
    /// Reads the parts' files for their sizes; the parts go on the GPU with the first request (ensure_open).
    bool open(const std::string& dit, const std::string& vae, const std::string& text, std::string& err,
              const std::string& vision = "");

    /// The VRAM the engine may take (0: the free VRAM at open, less a margin, within STRATA_VRAM_LIMIT_MIB).
    void set_budget(uint64_t bytes);
    /// The budget in force: the one set, else the free VRAM's at open.
    uint64_t budget() const { return budget_ > 0 ? budget_ : auto_budget_; }
    /// Frees what the engine holds on the GPU beyond its fixed weights (the cache): the next request loads again.
    /// `all`: the fixed weights and the products' scratch too (the parts are opened again on the next request).
    void release(bool all = false);
    /// The budget set to what the engine holds and the VRAM free now, less `reserve` (another program's room): taken
    /// before a request when the GPU's memory is shared.
    void fit(uint64_t reserve);
    /// What the engine holds on the GPU now (an estimate of its own buffers).
    uint64_t held() const;

    /// What an image of n_text prompt ids (after the system turn) at width x height needs (from the parts' sizes:
    /// nothing put on the GPU); refs: an edit's reference images' sizes (width, height as read).
    QiNeed need(int64_t n_text, int64_t width, int64_t height,
                const std::vector<std::pair<int64_t, int64_t>>& refs = {});

    /// RGBA8 [height][width][4] of the request.  `step`: the DiT's progress (QiDit::sample), false to stop.
    std::vector<uint8_t> generate(const QiRequest& r, QiTimes* times = nullptr, QiPlan* plan = nullptr,
                                  const std::function<bool(int, int)>& step = {});

private:
    QiPlan plan(int64_t T, int64_t h, int64_t w);

    void ensure_open();
    void edit_refs(const QiRequest& r, std::vector<int32_t>& ids, std::vector<QiVisionOut>& vouts,
                   std::vector<QiText::Image>& te_images, std::vector<QiDit::Ref>& dit_refs, std::vector<float*>& ref_z);

    void* s_ = nullptr;
    std::string dit_path_, vae_path_, text_path_, vision_path_;
    std::unique_ptr<QiVision> vision_;           // an edit's, opened for it and let go after it
    std::unique_ptr<QiVision> vision_size_;
    std::unique_ptr<QiText> te_;
    std::unique_ptr<QiDit> dit_;
    std::unique_ptr<QiVae> vae_;
    // the parts read for their sizes only (open, dry): what an image needs is known with nothing on the GPU
    std::unique_ptr<QiText> te_size_;
    std::unique_ptr<QiDit> dit_size_;
    std::unique_ptr<QiVae> vae_size_;
    uint64_t budget_ = 0, auto_budget_ = 0;
    uint64_t dit_set_ = 0, te_set_ = 0;
    int64_t mlp_set_ = -1;
    std::vector<int32_t> last_ids_;   // the last prompt, and its text states (host)
    std::vector<float> last_states_;
};

}  // namespace strata::image
