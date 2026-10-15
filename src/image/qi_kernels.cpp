// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/qi_kernels.cpp - see include/strata/image/qi_kernels.hpp.

#include "strata/image/qi_kernels.hpp"

#include "strata/core/runtime.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <sycl/sycl.hpp>

#include <cmath>
#include <stdexcept>

namespace strata::image {

namespace {

using strata::kernels::f16_from_f32;
using strata::kernels::f32_from_f16;

sycl::queue& queue_for(void* stream) { return core::Runtime::get().stream(stream); }

void finish(void* stream, const sycl::event& e, const char* what) {
    if (!stream) core::Runtime::get().wait(e, what);
}

constexpr float kEps = 1e-6f;

}  // namespace

void qi_ln_mod16(const float* x, const float* mod, int64_t n_text, int64_t T, int64_t C, uint16_t* y16, void* stream) {
    if (T <= 0) return;
    constexpr size_t local = 256;
    const auto e = queue_for(stream).parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) T * local), sycl::range<1>(local)), [=](sycl::nd_item<1> it) {
            const int64_t t = static_cast<int64_t>(it.get_group_linear_id()), lane = static_cast<int64_t>(it.get_local_linear_id());
            const float* r = x + t * C;
            float s = 0.f;
            for (int64_t c = lane; c < C; c += local) s += r[c];
            const float mean = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>()) / (float) C;
            float v = 0.f;
            for (int64_t c = lane; c < C; c += local) {
                const float d = r[c] - mean;
                v += d * d;
            }
            const float inv = 1.f / sycl::sqrt(sycl::reduce_over_group(it.get_group(), v, sycl::plus<float>()) /
                                               (float) C + kEps);
            const float* m = mod + (t < n_text ? C : 0);
            for (int64_t c = lane; c < C; c += local) y16[t * C + c] = f16_from_f32((r[c] - mean) * inv * (1.f + m[c]));
        });
    finish(stream, e, "qi_ln_mod16");
}

void qi_gate_add(float* x, const float* h, const float* mod, int64_t n_text, int64_t T, int64_t C, void* stream) {
    if (T <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) (T * C)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t t = i / C, c = i - t * C;
        x[i] += h[i] * sycl::tanh(mod[(t < n_text ? C : 0) + c]);
    });
    finish(stream, e, "qi_gate_add");
}

void qi_qk_norm_rope16(const float* q, const float* k, const float* v, const float* wq, const float* wk,
                       const float* cos, const float* sin, int64_t T, int64_t heads, int64_t head_dim,
                       uint16_t* q16, uint16_t* k16, uint16_t* v16, void* stream) {
    if (T <= 0) return;
    if (head_dim % 2 != 0 || head_dim > 256) throw std::invalid_argument("qi_qk_norm_rope16: head_dim");
    const size_t local = (size_t) head_dim;
    const int64_t half = head_dim / 2;
    const auto e = queue_for(stream).parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) (T * heads) * local), sycl::range<1>(local)),
        [=](sycl::nd_item<1> it) {
            const int64_t g = static_cast<int64_t>(it.get_group_linear_id()), d = static_cast<int64_t>(it.get_local_linear_id());
            const int64_t t = g / heads, i = g * head_dim + d;
            const int64_t j = d / 2;
            const float c = cos[t * half + j], s = sin[t * half + j];
            const bool odd = (d & 1) != 0;
            for (int which = 0; which < 2; ++which) {
                const float* src = which == 0 ? q : k;
                const float* w = which == 0 ? wq : wk;
                const float val = src[i];
                const float ss = sycl::reduce_over_group(it.get_group(), val * val, sycl::plus<float>());
                const float n = val / sycl::sqrt(ss / (float) head_dim + kEps) * w[d];
                const float other = sycl::permute_group_by_xor(it.get_sub_group(), n, 1);
                // the pair (a, b) = (even, odd): a' = a c - b s, b' = a s + b c
                const float out = odd ? other * s + n * c : n * c - other * s;
                (which == 0 ? q16 : k16)[i] = f16_from_f32(out);
            }
            v16[i] = f16_from_f32(v[i]);
        });
    finish(stream, e, "qi_qk_norm_rope16");
}

void qi_attention_plain(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t T, int64_t heads,
                        int64_t head_dim, int64_t n_text, float* out, void* stream) {
    if (T <= 0) return;
    if (head_dim != 128) throw std::invalid_argument("qi_attention_plain: head_dim 128");
    constexpr int64_t D = 128, TILE = 128;
    const int64_t row = heads * D;
    const float scale = 1.f / sycl::sqrt((float) D);
    const auto e = queue_for(stream).submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> qs(D, h), ps(TILE, h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (T * heads * D)), sycl::range<1>(D)),
                       [=](sycl::nd_item<1> it) {
            const int64_t g = static_cast<int64_t>(it.get_group_linear_id()), lane = static_cast<int64_t>(it.get_local_linear_id());
            const int64_t t = g / heads, hh = g - t * heads;
            qs[lane] = f32_from_f16(q16[t * row + hh * D + lane]) * scale;
            sycl::group_barrier(it.get_group());
            const int64_t keys = t < n_text ? t + 1 : T;
            float m = -INFINITY, l = 0.f, acc = 0.f;
            for (int64_t j0 = 0; j0 < keys; j0 += TILE) {
                const int64_t j = j0 + lane;
                float s = -INFINITY;
                if (j < keys) {
                    const uint16_t* kr = k16 + j * row + hh * D;
                    s = 0.f;
                    for (int64_t d = 0; d < D; ++d) s += qs[d] * f32_from_f16(kr[d]);
                }
                const float tm = sycl::reduce_over_group(it.get_group(), s, sycl::maximum<float>());
                const float nm = sycl::fmax(m, tm);
                const float p = j < keys ? sycl::exp(s - nm) : 0.f;
                const float corr = sycl::exp(m - nm);
                l = l * corr + sycl::reduce_over_group(it.get_group(), p, sycl::plus<float>());
                ps[lane] = p;
                sycl::group_barrier(it.get_group());
                acc *= corr;
                const int64_t n = sycl::min(TILE, keys - j0);
                for (int64_t jj = 0; jj < n; ++jj) acc += ps[jj] * f32_from_f16(v16[(j0 + jj) * row + hh * D + lane]);
                sycl::group_barrier(it.get_group());
                m = nm;
            }
            out[t * row + hh * D + lane] = acc / l;
        });
    });
    finish(stream, e, "qi_attention_plain");
}

void qi_swiglu16(const float* gu, int64_t T, int64_t inter, uint16_t* h16, void* stream) {
    if (T <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) (T * inter)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t t = i / inter, c = i - t * inter;
        const float g = gu[t * 2 * inter + c], u = gu[t * 2 * inter + inter + c];
        h16[i] = f16_from_f32(u * g / (1.f + sycl::exp(-g)));
    });
    finish(stream, e, "qi_swiglu16");
}

void qi_to_f16(const float* x, int64_t n, uint16_t* y16, void* stream) {
    if (n <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) {
        y16[i] = f16_from_f32(x[i]);
    });
    finish(stream, e, "qi_to_f16");
}

void qi_rms_norm16(const float* x, const float* w, int64_t rows, int64_t C, uint16_t* y16, void* stream) {
    if (rows <= 0) return;
    constexpr size_t local = 256;
    const auto e = queue_for(stream).parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) rows * local), sycl::range<1>(local)), [=](sycl::nd_item<1> it) {
            const auto r = static_cast<int64_t>(it.get_group_linear_id());
            const auto lane = static_cast<int64_t>(it.get_local_linear_id());
            const float* xr = x + r * C;
            float s = 0.f;
            for (int64_t c = lane; c < C; c += local) s += xr[c] * xr[c];
            const float inv = 1.f / sycl::sqrt(sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>()) /
                                               (float) C + kEps);
            for (int64_t c = lane; c < C; c += local) y16[r * C + c] = f16_from_f32(xr[c] * inv * w[c]);
        });
    finish(stream, e, "qi_rms_norm16");
}

void qi_gelu16(const float* x, int64_t n, uint16_t* y16, void* stream) {
    if (n <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) {
        const float v = x[i];
        y16[i] = f16_from_f32(0.5f * v * (1.f + sycl::tanh(0.7978845608028654f * (v + 0.044715f * v * v * v))));
    });
    finish(stream, e, "qi_gelu16");
}

void qi_silu(float* x, int64_t n, uint16_t* y16, void* stream) {
    if (n <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) {
        const float v = x[i] / (1.f + sycl::exp(-x[i]));
        x[i] = v;
        if (y16) y16[i] = f16_from_f32(v);
    });
    finish(stream, e, "qi_silu");
}

void qi_latent_tokens16(const float* latent, int64_t C, int64_t HW, uint16_t* tokens16, void* stream) {
    if (HW <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) (C * HW)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t p = i / C, c = i - p * C;
        tokens16[i] = f16_from_f32(latent[c * HW + p]);
    });
    finish(stream, e, "qi_latent_tokens16");
}

void qi_tokens_latent(const float* tokens, int64_t C, int64_t HW, float* latent, void* stream) {
    if (HW <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) (C * HW)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t c = i / HW, p = i - c * HW;
        latent[i] = tokens[p * C + c];
    });
    finish(stream, e, "qi_tokens_latent");
}

void qi_axpy(float* x, const float* v, float a, int64_t n, void* stream) {
    if (n <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { x[i] += a * v[i]; });
    finish(stream, e, "qi_axpy");
}

}  // namespace strata::image
