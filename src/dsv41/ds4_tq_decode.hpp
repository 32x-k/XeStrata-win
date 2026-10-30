// SPDX-FileCopyrightText: 2026 Peasant Smith and the Project Maya contributors
// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
#pragma once
#include "ds4_iq_types.hpp"
#include "ds4_sycl.hpp"
#include "ggml-common-tq.h"
namespace strata::ds4::gguf::iq::tq::sycl_decode {
struct Decoder {
    int K;
    bool computed;

    float value(const uint8_t* row, int index) const {
        const uint8_t* b = row + (size_t) (index / 128) * (2 + 4 * K);
        const int step = (index % 128) / 4, component = index % 4;
        const int pos = (31 - step) * K;
        uint32_t state = 0;
        for (int bit = 0; bit < 15; ++bit) {
            const int at = (pos + bit) % (32 * K);
            state |= ((uint32_t) b[2 + at / 8] >> (at % 8) & 1u) << bit;
        }
        uint16_t v;
        if (computed) {
            const uint32_t w = component < 2 ? ((state * 89226354u + 64248484u) & 0x8FFF8FFFu) ^ 0x3B603B60u
                                             : ((state * 0x9E3779B1u + 0x7F4A7C15u) & 0x8FFF8FFFu) ^ 0x3B603B60u;
            v = (uint16_t) (w >> (16 * (component % 2)));
        } else {
            const uint32_t x = state * 0x9e3779b1u;
            const uint32_t p = component < 2 ? x >> 21 : (x >> 10) & 2047u;
            v = tq_lut_f16[2 * p + component % 2];
        }
        const uint16_t scale = (uint16_t) ((unsigned) b[0] | (unsigned) b[1] << 8);
        float weight = sycl_detail::f16_value(v);
        if (!computed) {
            // The CPU codebook rounds fp16 * 2^13 half away from zero, then divides by 2^13.
            const float scaled = weight * 8192.0f;
            const int rounded = (int) (scaled < 0.0f ? scaled - 0.5f : scaled + 0.5f);
            weight = (float) rounded / 8192.0f;
        }
        return sycl_detail::f16_value(scale) * weight;
    }
    float dot_q8(const uint8_t* row, const int8_t* x, const float* xd, int k) const {
        float sum = 0.0f;
        for (int b = 0; b < k / 32; ++b) {
            float part = 0.0f;
            for (int j = 0; j < 32; ++j) part = sycl::fma(value(row, b * 32 + j), (float) x[b * 32 + j], part);
            sum = sycl::fma(xd[b], part, sum);
        }
        return sum;
    }
};
inline Decoder decoder(int type) { return {tq_kbits(type), type == kTQK5X || type == kTQK4X}; }
inline float value(int type, const uint8_t* row, int index) { return decoder(type).value(row, index); }
}  // namespace strata::ds4::gguf::iq::tq::sycl_decode
