// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/dit_rows.hpp - the parts Mage-Flow's DiT (mf_dit.cpp) and MOSS-Image's (moss_dit.cpp) share: the blocks'
// bf16 weights quantized on the host (Q8_0, or W8A8's rotated int8 rows), and the row-wise kernels that write a
// product's FP16 input with each row's power-of-two scale, scale its output back and add it through a gate.
//
// The kernels' helpers are in a namespace of their own, not an anonymous one: SYCL names a lambda's kernel by its
// mangled type, and an anonymous namespace mangles the same in every file (_GLOBAL__N_1).
#pragma once

#include "strata/core/per_device.hpp"
#include "strata/core/runtime.hpp"
#include "strata/image/mf_dit.hpp"
#include "strata/image/qi_kernels.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

namespace strata::image::dit_rows {

namespace syclex = sycl::ext::oneapi::experimental;
using strata::kernels::f16_from_f32;

inline sycl::queue& q_of(void* s) { return core::Runtime::get().stream(s); }

inline float bf16_value(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// x rounded to the nearest bf16 (ties to even), as a float
inline float round_bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    u += 0x7FFFu + ((u >> 16) & 1u);
    u &= 0xFFFF0000u;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// row of K bf16 values -> Q8_0 (ggml's reference rounding): blocks of an FP16 scale and 32 int8 values
inline void quantize_row(const uint16_t* src, int64_t K, uint8_t* dst) {
    for (int64_t b = 0; b < K / 32; ++b) {
        float v[32], amax = 0.f;
        for (int j = 0; j < 32; ++j) {
            v[j] = bf16_value(src[b * 32 + j]);
            amax = std::max(amax, std::fabs(v[j]));
        }
        const float d = amax / 127.f, id = d != 0.f ? 1.f / d : 0.f;
        const uint16_t dh = f16_from_f32(d);
        int8_t* qs = reinterpret_cast<int8_t*>(dst + b * 34 + 2);
        for (int j = 0; j < 32; ++j) qs[j] = (int8_t) std::lround(v[j] * id);
        std::memcpy(dst + b * 34, &dh, 2);
    }
}

// row of K bf16 values -> int8 values and their scale as qi_rows_i8 makes them: each group of kQiRotate times the
// normalized Hadamard matrix (Sylvester's, the same butterflies), then one scale for the row
inline void quantize_row_w8(const uint16_t* src, int64_t K, int8_t* dst, float* scale) {
    std::vector<float> v((size_t) K);
    float amax = 0.f;
    for (int64_t g = 0; g < K; g += kQiRotate) {
        float* x = v.data() + g;
        for (int64_t j = 0; j < kQiRotate; ++j) x[j] = bf16_value(src[g + j]);
        for (int64_t m = 1; m < kQiRotate; m <<= 1)
            for (int64_t i = 0; i < kQiRotate; i += 2 * m)
                for (int64_t j = i; j < i + m; ++j) {
                    const float a = x[j], b = x[j + m];
                    x[j] = a + b;
                    x[j + m] = a - b;
                }
        for (int64_t j = 0; j < kQiRotate; ++j) {
            x[j] *= 1.f / 16.f;
            amax = std::max(amax, std::fabs(x[j]));
        }
    }
    const float d = amax / 127.f, id = amax > 0.f ? 1.f / d : 0.f;
    for (int64_t j = 0; j < K; ++j) dst[j] = (int8_t) std::lround(v[(size_t) j] * id);
    *scale = d;
}

// the rows of a [n][k] bf16 matrix quantized on the host's threads: Q8_0, or with w8 as quantize_row_w8 (the n rows'
// int8 values, then their n scales)
inline void quantize_matrix(const uint16_t* src, int64_t n, int64_t k, uint8_t* dst, bool w8) {
    const uint64_t row = w8 ? (uint64_t) k : (uint64_t) (k / 32 * 34);
    auto* scales = reinterpret_cast<float*>(dst + (uint64_t) n * row);
    const int nt = (int) std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    std::vector<std::thread> th;
    th.reserve((size_t) nt);
    for (int t = 0; t < nt; ++t)
        th.emplace_back([=] {
            for (int64_t r = t; r < n; r += nt) {
                if (w8) quantize_row_w8(src + r * k, k, reinterpret_cast<int8_t*>(dst + r * row), scales + r);
                else quantize_row(src + r * k, k, dst + r * row);
            }
        });
    for (auto& x : th) x.join();
}

// Mage-Flow's activations pass FP16's range (its model ran in bf16: the image stream's normed rows up to 1.9e5, the
// attention's v up to 6.1e5, a text MLP's output 5.7e6 at the first steps of a 1024 x 1024 Turbo run).  So a product's
// FP16 input holds each row divided by a power of two (rs, at least 1) that brings its largest value under kRowLim, and
// the product's output rows are multiplied back by it; the attention's v is divided by one power of two a head over a
// context's rows (its output is a weighted sum of them), and the output multiplied back.  Powers of two leave the
// rounding as it is; FP16 keeps 3 more bits than bf16.
constexpr float kRowLim = 16384.f;

inline float pow2_scale(float amax) {
    return amax > kRowLim ? sycl::exp2(sycl::ceil(sycl::log2(amax / kRowLim))) : 1.f;
}

inline bool has_sg32(sycl::queue& q) {
    static core::PerDevice<bool> sg32;
    return sg32.get(q.get_device(), [&q] {
        const auto sizes = q.get_device().get_info<sycl::info::device::sub_group_sizes>();
        return std::find(sizes.begin(), sizes.end(), (size_t) 32) != sizes.end();
    });
}

// A row-wise kernel writing FP16 with each row's scale: Op (LnMod, Gelu, Heads below) gives a row's statistics and its
// values; a sub-group of 32 a row, 8 rows a work-group (or, without sub-groups of 32, a work-item a row).
constexpr int64_t kSG = 32, kRows = 8;

// the row's scale is taken from a bound of |op.value| where the op has a cheaper one than the value itself
template<class Op, class St> float bound(const Op& op, int64_t t, int64_t c, const St& st) {
    if constexpr (requires { op.bound(t, c); }) return op.bound(t, c);
    else return sycl::fabs(op.value(t, c, st));
}

template<class Op> struct Rows16 {
    Op op;
    int64_t T, C;
    uint16_t* y16;
    float* rs;
    void operator()(sycl::nd_item<1> it) const {
        const auto sg = it.get_sub_group();
        const int64_t t = static_cast<int64_t>(it.get_group_linear_id()) * kRows + sg.get_group_linear_id();
        const int64_t lane = sg.get_local_linear_id();
        if (t >= T) return;
        const auto sum = [&](float v) { return sycl::reduce_over_group(sg, v, sycl::plus<float>()); };
        const auto st = op.stats(t, lane, kSG, sum);
        float m = 0.f;
        for (int64_t c = lane; c < C; c += kSG) m = sycl::fmax(m, bound(op, t, c, st));
        const float sc = pow2_scale(sycl::reduce_over_group(sg, m, sycl::maximum<float>()));
        const float inv = 1.f / sc;
        for (int64_t c = lane; c < C; c += kSG) y16[t * C + c] = f16_from_f32(op.value(t, c, st) * inv);
        if (lane == 0) rs[t] = sc;
    }
    auto get(syclex::properties_tag) const { return syclex::properties{syclex::sub_group_size<kSG>}; }
};

template<class Op> inline void rows16(const Op& op, int64_t T, int64_t C, uint16_t* y16, float* rs, void* s) {
    if (T <= 0) return;
    sycl::queue& q = q_of(s);
    if (has_sg32(q)) {
        q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) ((T + kRows - 1) / kRows * kRows * kSG)),
                                         sycl::range<1>((size_t) (kRows * kSG))),
                       Rows16<Op>{op, T, C, y16, rs});
        return;
    }
    q.parallel_for(sycl::range<1>((size_t) T), [=](sycl::id<1> id) {
        const auto t = static_cast<int64_t>(id[0]);
        const auto st = op.stats(t, 0, 1, [](float v) { return v; });
        float m = 0.f;
        for (int64_t c = 0; c < C; ++c) m = sycl::fmax(m, bound(op, t, c, st));
        const float sc = pow2_scale(m), inv = 1.f / sc;
        for (int64_t c = 0; c < C; ++c) y16[t * C + c] = f16_from_f32(op.value(t, c, st) * inv);
        rs[t] = sc;
    });
}

struct Stats {
    float mean = 0.f, inv = 1.f;
};

// the attention's output o16 [T][C] times its heads' scales hs (head_dim channels a head)
struct Heads {
    const uint16_t* o16;
    const float* hs;
    int64_t C, hd;
    template<class R> Stats stats(int64_t, int64_t, int64_t, R) const { return {}; }
    float value(int64_t t, int64_t c, const Stats&) const { return kernels::f32_from_f16(o16[t * C + c]) * hs[c / hd]; }
};

// The per-head kernels below: a sub-group of 32 a row's head (its values read side by side, kSG apart), kRows heads a
// work-group; a work-item a row's head without sub-groups of 32.

// part[r][h] = the largest |v| of row r's head h, rows of a ([na][C]) then b ([nb][C])
struct HeadMax {
    MfOut a, b;
    int64_t na, nb, heads, hd;
    float* part;
    void operator()(sycl::nd_item<1> it) const {
        const auto sg = it.get_sub_group();
        const int64_t i = static_cast<int64_t>(it.get_group_linear_id()) * kRows + sg.get_group_linear_id();
        if (i >= (na + nb) * heads) return;
        const int64_t r = i / heads, h = i - r * heads;
        float m = 0.f;
        for (int64_t d = sg.get_local_linear_id(); d < hd; d += kSG)
            m = sycl::fmax(m, sycl::fabs(r < na ? a.at(r, h * hd + d) : b.at(r - na, h * hd + d)));
        m = sycl::reduce_over_group(sg, m, sycl::maximum<float>());
        if (sg.get_local_linear_id() == 0) part[i] = m;
    }
    auto get(syclex::properties_tag) const { return syclex::properties{syclex::sub_group_size<kSG>}; }
};

inline void head_max(const MfOut& a, int64_t na, const MfOut& b, int64_t nb, int64_t heads, int64_t hd, float* part,
              void* s) {
    const int64_t n = (na + nb) * heads;
    if (n <= 0) return;
    sycl::queue& q = q_of(s);
    if (has_sg32(q)) {
        q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) ((n + kRows - 1) / kRows * kRows * kSG)),
                                         sycl::range<1>((size_t) (kRows * kSG))),
                       HeadMax{a, b, na, nb, heads, hd, part});
        return;
    }
    q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t r = i / heads, h = i - r * heads;
        float m = 0.f;
        for (int64_t d = 0; d < hd; ++d)
            m = sycl::fmax(m, sycl::fabs(r < na ? a.at(r, h * hd + d) : b.at(r - na, h * hd + d)));
        part[i] = m;
    });
}

// hs[h] = the scale of head h over the rows' part: a work-group a head, its items striding the rows
inline void head_scale(const float* part, int64_t rows, int64_t heads, float* hs, void* s) {
    constexpr size_t kWG = 256;
    q_of(s).parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) heads * kWG), sycl::range<1>(kWG)),
                         [=](sycl::nd_item<1> it) {
        const auto h = static_cast<int64_t>(it.get_group_linear_id());
        float m = 0.f;
        for (auto r = static_cast<int64_t>(it.get_local_linear_id()); r < rows; r += (int64_t) kWG)
            m = sycl::fmax(m, part[r * heads + h]);
        m = sycl::reduce_over_group(it.get_group(), m, sycl::maximum<float>());
        if (it.get_local_linear_id() == 0) hs[h] = pow2_scale(m);
    });
}

// y[t] = y[t] * rs[t] + b (rs null: 1; b null: none)
inline void rows_back(float* y, const float* rs, const float* b, int64_t T, int64_t N, void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) (T * N)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t t = i / N;
        float v = y[i];
        if (rs != nullptr) v *= rs[t];
        if (b != nullptr) v += b[i - t * N];
        y[i] = v;
    });
}

// x[t] += gate * h[t] (per channel; gate null: 1)
inline void gate_add(float* x, const MfOut& h, const float* gate, int64_t T, int64_t C, void* s) {
    if (T <= 0) return;
    q_of(s).parallel_for(sycl::range<2>((size_t) T, (size_t) C), [=](sycl::id<2> id) {
        const auto t = static_cast<int64_t>(id[0]), c = static_cast<int64_t>(id[1]);
        x[t * C + c] += (gate != nullptr ? gate[c] : 1.f) * h.at(t, c);
    });
}

}  // namespace strata::image::dit_rows
