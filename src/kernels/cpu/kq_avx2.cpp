// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/cpu/kq_avx2.cpp - the routing-aware prefetch's BF16 router dot product, four tokens per pass over the
// weights.  Compiled with -mavx2 -mfma; called only where cpu_avx2_ok().
#include "strata/kernels/cpu/kq_avx2.hpp"

#include <immintrin.h>

#include <cstddef>

namespace strata::kernels::cpu {
namespace {

inline float hsum8(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

}  // namespace

void bf16_rows_dot_multi(const uint16_t* w, int rows, int cols, const float* x, int nt, float* out) {
    for (int t0 = 0; t0 < nt; t0 += 4) {
        const int m = nt - t0 < 4 ? nt - t0 : 4;
        const float* x0 = x + (size_t) t0 * cols;
        const float* x1 = x0 + cols;
        const float* x2 = x1 + cols;
        const float* x3 = x2 + cols;
        for (int r = 0; r < rows; ++r) {
            const uint16_t* wr = w + (size_t) r * (size_t) cols;
            __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps(),
                   a3 = _mm256_setzero_ps();
            for (int c = 0; c + 8 <= cols; c += 8) {
                const __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(wr + c));
                const __m256 wf = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16));
                a0 = _mm256_fmadd_ps(wf, _mm256_loadu_ps(x0 + c), a0);
                if (m > 1) a1 = _mm256_fmadd_ps(wf, _mm256_loadu_ps(x1 + c), a1);
                if (m > 2) a2 = _mm256_fmadd_ps(wf, _mm256_loadu_ps(x2 + c), a2);
                if (m > 3) a3 = _mm256_fmadd_ps(wf, _mm256_loadu_ps(x3 + c), a3);
            }
            out[(size_t) t0 * rows + r] = hsum8(a0);
            if (m > 1) out[(size_t) (t0 + 1) * rows + r] = hsum8(a1);
            if (m > 2) out[(size_t) (t0 + 2) * rows + r] = hsum8(a2);
            if (m > 3) out[(size_t) (t0 + 3) * rows + r] = hsum8(a3);
        }
    }
}

}  // namespace strata::kernels::cpu
