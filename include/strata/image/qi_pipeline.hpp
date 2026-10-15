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

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace strata::image {

struct QiRequest {
    std::vector<int32_t> ids;   // the prompt in the model's chat form, tokenized
    int64_t drop = 0;           // the leading ids whose states the DiT does not take (the system turn)
    int64_t width = 1024, height = 1024;   // multiples of 16
    uint64_t seed = 0;
    std::vector<float> sigmas;  // empty: Turbo's 8 steps
};

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

class QiPipeline {
public:
    bool open(const std::string& dit, const std::string& vae, const std::string& text, std::string& err);

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

    /// RGBA8 [height][width][4] of the request.  `step`: the DiT's progress (QiDit::sample), false to stop.
    std::vector<uint8_t> generate(const QiRequest& r, QiTimes* times = nullptr, QiPlan* plan = nullptr,
                                  const std::function<bool(int, int)>& step = {});

private:
    QiPlan plan(int64_t T, int64_t h, int64_t w);

    void ensure_open();

    void* s_ = nullptr;
    std::string dit_path_, vae_path_, text_path_;
    std::unique_ptr<QiText> te_;
    std::unique_ptr<QiDit> dit_;
    std::unique_ptr<QiVae> vae_;
    uint64_t budget_ = 0, auto_budget_ = 0;
    uint64_t dit_set_ = 0, te_set_ = 0;
    int64_t mlp_set_ = -1;
    std::vector<int32_t> last_ids_;   // the last prompt, and its text states (host)
    std::vector<float> last_states_;
};

}  // namespace strata::image
