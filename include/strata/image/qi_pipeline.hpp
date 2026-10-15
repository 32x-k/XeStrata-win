// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/qi_pipeline.hpp - Qwen-Image-2.1's text to image on the GPU: the text encoder's states of the
// prompt's ids, the DiT's flow sampling from the seed's noise, and the VAE's image.
#pragma once

#include "strata/image/qi_dit.hpp"
#include "strata/image/qi_text.hpp"
#include "strata/image/qi_vae.hpp"

#include <cstdint>
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

class QiPipeline {
public:
    bool open(const std::string& dit, const std::string& vae, const std::string& text, std::string& err);

    /// RGBA8 [height][width][4] of the request.
    std::vector<uint8_t> generate(const QiRequest& r, QiTimes* times = nullptr);

private:
    void* s_ = nullptr;
    QiText te_;
    QiDit dit_;
    QiVae vae_;
};

}  // namespace strata::image
