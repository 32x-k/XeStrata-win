// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/qi_pipeline.cpp - see include/strata/image/qi_pipeline.hpp.

#include "strata/image/qi_pipeline.hpp"

#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace strata::image {

namespace gpu = strata::gpu;

bool QiPipeline::open(const std::string& dit, const std::string& vae, const std::string& text, std::string& err) {
    s_ = core::Runtime::get().create_stream();
    return te_.open(text, s_, err) && dit_.open(dit, s_, err) && vae_.open(vae, s_, err);
}

std::vector<uint8_t> QiPipeline::generate(const QiRequest& r, QiTimes* times) {
    using clock = std::chrono::steady_clock;
    if (r.width % 16 != 0 || r.height % 16 != 0 || r.width <= 0 || r.height <= 0)
        throw std::invalid_argument("image engine: the size must be a multiple of 16");
    if ((int64_t) r.ids.size() <= r.drop) throw std::invalid_argument("image engine: no prompt after the system turn");
    const auto t0 = clock::now();
    const auto states = te_.encode(r.ids);
    const int64_t C = te_.hidden(), n_text = (int64_t) r.ids.size() - r.drop;
    dit_.set_context(states.data() + r.drop * C, n_text);
    const auto t1 = clock::now();
    const int64_t h = r.height / 16, w = r.width / 16, n = dit_.channels() * h * w;
    const auto noise = philox_randn(r.seed, (uint64_t) n);
    float *x = nullptr, *rgba = nullptr;
    const int64_t HW = r.width * r.height;
    if (!gpu::alloc_device(&x, (size_t) n * 4) || !gpu::alloc_device(&rgba, (size_t) (4 * HW) * 4))
        throw std::runtime_error(std::string("image engine: GPU memory: ") + gpu::last_error());
    std::vector<float> out((size_t) (4 * HW));
    try {
        gpu::copy(x, noise.data(), (size_t) n * 4);
        dit_.sample(x, h, w, r.sigmas.empty() ? turbo_sigmas() : r.sigmas);
        const auto t2 = clock::now();
        vae_.decode(x, h, w, rgba);
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
    std::vector<uint8_t> img((size_t) (4 * HW));
    for (int64_t p = 0; p < HW; ++p)
        for (int c = 0; c < 4; ++c)
            img[(size_t) (p * 4 + c)] = (uint8_t) std::lround(std::clamp(out[(size_t) (c * HW + p)], 0.0f, 1.0f) * 255.0f);
    return img;
}

}  // namespace strata::image
