// SPDX-FileCopyrightText: 2026 Peasant Smith and the Project Maya contributors
// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// Strata. Portable access to the TQW repacked trellis format. Its 16-row tiles and two stream halves are
// storage dimensions, independent of the device's sub-group size. The regular engine reads the original rows.
#pragma once
#include "ds4_sycl.hpp"
#include <stdexcept>

namespace strata::ds4::tqw {
using half8_t = sycl::marray<uint16_t, 8>;  // FP16 bit patterns; no fp16 device aspect required.
struct Job {
    const uint32_t* words;
    const uint16_t* scales;
    const half8_t* xb;
    float* y;
    int rows, in, ldy;
};
inline void step_words(const uint32_t* lut, uint32_t state, uint32_t& w0, uint32_t& w1, int code) {
    if (code == 1) {
        w0 = ((state * 89226354u + 64248484u) & 0x8FFF8FFFu) ^ 0x3B603B60u;
        w1 = ((state * 0x9E3779B1u + 0x7F4A7C15u) & 0x8FFF8FFFu) ^ 0x3B603B60u;
    } else {
        const uint32_t x = state * 0x9e3779b1u;
        w0 = lut[x >> 21]; w1 = lut[(x >> 10) & 2047u];
    }
}
inline void prep_b(const float* x, int ldx, int nb, int in, half8_t* out, sycl::queue& s) {
    sycl_detail::parallel(s, (size_t) (in / 128) * 16 * nb, [=](size_t id) {
        const int n = (int) (id % nb), kg = (int) ((id / nb) % 2);
        const int m = (int) ((id / (2 * nb)) % 8), blk = (int) (id / (16 * nb));
        half8_t hi, lo;
        for (int j = 0; j < 8; ++j) {
            const float v = x[(size_t) n * ldx + blk * 128 + 4 * (16 * kg + 2 * m) + j];
            hi[j] = sycl_detail::f16_bits(v);
            lo[j] = sycl_detail::f16_bits(v - sycl_detail::f16_value(hi[j]));
        }
        out[2 * id] = hi; out[2 * id + 1] = lo;
    });
}
template <int K, int NB, int T, int EXACT, int CODE>
inline void gemv(const Job* jobs, const uint32_t* lut, int n_jobs, int max_rows, sycl::queue& s) {
    static_assert(K >= 4 && K <= 7 && NB > 0 && T > 0);
    sycl_detail::parallel(s, (size_t) n_jobs * max_rows * NB, [=](size_t id) {
        const int row = (int) (id % max_rows), token = (int) ((id / max_rows) % NB);
        const Job job = jobs[id / ((size_t) max_rows * NB)];
        if (row >= job.rows) return;
        const int tile = row / 16, r = row % 16, pairs = job.in / 256;
        float sum = 0.0f;
        for (int blk = 0; blk < job.in / 128; ++blk) {
            const int pair = blk / 2, bk = blk % 2;
            float part = 0.0f;
            for (int step = 0; step < 32; ++step) {
                const int kg = step / 16, u = step % 16;
                uint32_t state = 0;
                for (int bit = 0; bit < 15; ++bit) {
                    const int pos = (15 - u) * K + bit;
                    const int lane = r + 16 * (pos < 16 * K ? kg : 1 - kg);
                    const int at = bk * 16 * K + pos % (16 * K);
                    const uint32_t w = job.words[(((size_t) tile * pairs + pair) * K + at / 32) * 32 + lane];
                    state |= ((w >> (at % 32)) & 1u) << bit;
                }
                uint32_t w0, w1; step_words(lut, state, w0, w1, CODE);
                const size_t at = ((((size_t) blk * 8 + u / 2) * 2 + kg) * NB + token) * 2;
                for (int c = 0; c < 4; ++c) {
                    const float w = sycl_detail::f16_value((uint16_t) ((c < 2 ? w0 : w1) >> (16 * (c % 2))));
                    const int j = (u % 2) * 4 + c;
                    part = sycl::fma(w, sycl_detail::f16_value(job.xb[at][j]), part);
                    if constexpr (EXACT) part = sycl::fma(w, sycl_detail::f16_value(job.xb[at + 1][j]), part);
                }
            }
            const float d = sycl_detail::f16_value(job.scales[(((size_t) tile * pairs + pair) * 16 + r) * 2 + bk]);
            sum = sycl::fma(d, part, sum);
        }
        job.y[(size_t) token * job.ldy + row] = sum;
    });
}
}  // namespace strata::ds4::tqw
