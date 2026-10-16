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

namespace {

// Tiles of BQ queries of one head a work-group: the keys and values come into local memory BK rows at a time and serve
// every query of the tile (the work-group reads K and V once per tile of queries, not once per query).  Each query
// has 8 work-items: 4 of its tile's scores each, and 16 neighbouring of its 128 output channels each (local memory
// read 8 halves at a time); a query's 8 work-items are
// neighbours in a sub-group (any size a multiple of 8), so its row's max and sum go through sub-group shuffles.
template<int64_t BQ>
void attention_tiles(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t T, int64_t heads,
                     int64_t n_text, const int32_t* key_lim, float* out, void* stream) {
    constexpr int64_t D = 128, BK = 32, LANES = 8, WG = BQ * LANES;
    constexpr int64_t S = D + 8;                        // the rows' stride in local memory: neighbouring rows in other banks
    constexpr int SC = (int) (BK / LANES), OC = (int) (D / LANES);
    using H8 = sycl::vec<sycl::half, 8>;
    const int64_t row = heads * D, tiles = (T + BQ - 1) / BQ;
    const float scale = 1.f / sycl::sqrt((float) D);
    const auto e = queue_for(stream).submit([&](sycl::handler& h) {
        sycl::local_accessor<sycl::half, 1> qs(BQ * S, h), ks(BK * S, h), vs(BK * S, h);
        sycl::local_accessor<float, 1> ps(BQ * BK, h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (tiles * heads * WG)), sycl::range<1>((size_t) WG)),
                       [=](sycl::nd_item<1> it) {
            const auto g = static_cast<int64_t>(it.get_group_linear_id());
            const auto tid = static_cast<int64_t>(it.get_local_linear_id());
            const auto sg = it.get_sub_group();
            const int64_t hh = g % heads, q0 = (g / heads) * BQ;
            const int64_t qi = tid / LANES, lane = tid % LANES, t = q0 + qi;
            const sycl::half* qp = qs.get_multi_ptr<sycl::access::decorated::no>().get();
            const sycl::half* kp = ks.get_multi_ptr<sycl::access::decorated::no>().get();
            const sycl::half* vp = vs.get_multi_ptr<sycl::access::decorated::no>().get();
            for (int64_t i = tid; i < BQ * D; i += WG) {
                const int64_t r = i / D, d = i % D;
                const uint16_t bits = q0 + r < T ? q16[(q0 + r) * row + hh * D + d] : (uint16_t) 0;
                qs[r * S + d] = sycl::bit_cast<sycl::half>(bits);
            }
            // the keys a query sees: a prefix row its limit (a text row its own prefix), the image rows every row
            const int64_t keys = t < n_text ? (key_lim ? key_lim[t] : t + 1) : T;
            const int64_t lr = sycl::min(q0 + BQ, T) - 1;   // the tile's furthest key: its last row's (they never fall)
            const int64_t last = lr < n_text ? (key_lim ? key_lim[lr] : lr + 1) : T;
            float m = -INFINITY, l = 0.f, acc[OC];
            for (int c = 0; c < OC; ++c) acc[c] = 0.f;
            for (int64_t j0 = 0; j0 < last; j0 += BK) {
                sycl::group_barrier(it.get_group());
                for (int64_t i = tid; i < BK * D; i += WG) {
                    const int64_t r = i / D, d = i % D, j = j0 + r;
                    const uint16_t kb = j < T ? k16[j * row + hh * D + d] : (uint16_t) 0;
                    const uint16_t vb = j < T ? v16[j * row + hh * D + d] : (uint16_t) 0;
                    ks[r * S + d] = sycl::bit_cast<sycl::half>(kb);
                    vs[r * S + d] = sycl::bit_cast<sycl::half>(vb);
                }
                sycl::group_barrier(it.get_group());
                float sc[SC];
                for (int r = 0; r < SC; ++r) sc[r] = 0.f;
                for (int64_t d = 0; d < D; d += 8) {   // a Q value serves the work-item's keys
                    const H8 qv = *reinterpret_cast<const H8*>(qp + qi * S + d);
                    for (int r = 0; r < SC; ++r) {
                        const H8 kv = *reinterpret_cast<const H8*>(kp + (lane + LANES * r) * S + d);
                        for (int x = 0; x < 8; ++x) sc[r] += (float) qv[x] * (float) kv[x];
                    }
                }
                float tm = -INFINITY;
                for (int r = 0; r < SC; ++r) {
                    sc[r] = j0 + lane + LANES * r < keys ? sc[r] * scale : -INFINITY;
                    tm = sycl::fmax(tm, sc[r]);
                }
                for (int x = 1; x < LANES; x <<= 1) tm = sycl::fmax(tm, sycl::permute_group_by_xor(sg, tm, x));
                const float nm = sycl::fmax(m, tm);
                const float corr = nm == -INFINITY ? 1.f : sycl::exp(m - nm);
                float ls = 0.f;
                for (int r = 0; r < SC; ++r) {
                    const float pr = sc[r] == -INFINITY ? 0.f : sycl::exp(sc[r] - nm);
                    ps[qi * BK + lane + LANES * r] = pr;
                    ls += pr;
                }
                for (int x = 1; x < LANES; x <<= 1) ls += sycl::permute_group_by_xor(sg, ls, x);
                l = l * corr + ls;
                m = nm;
                for (int c = 0; c < OC; ++c) acc[c] *= corr;
                sycl::group_barrier(it.get_group());
                for (int64_t kj = 0; kj < BK; ++kj) {
                    const float pr = ps[qi * BK + kj];
                    for (int c = 0; c < OC; c += 8) {
                        const H8 vv = *reinterpret_cast<const H8*>(vp + kj * S + lane * OC + c);
                        for (int x = 0; x < 8; ++x) acc[c + x] += pr * (float) vv[x];
                    }
                }
            }
            if (t < T)
                for (int c = 0; c < OC; ++c) out[t * row + hh * D + lane * OC + c] = acc[c] / l;
        });
    });
    finish(stream, e, "qi_attention_plain");
}

}  // namespace

// 64 queries a work-group where the device takes 512 work-items and their local memory (B70 without its matrix
// engines: 56 ms a 1024 x 1024 block's attention, 68 ms with 32), else 32
void qi_attention_plain(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t T, int64_t heads,
                        int64_t head_dim, int64_t n_text, float* out, void* stream, const int32_t* key_lim) {
    if (T <= 0) return;
    if (head_dim != 128) throw std::invalid_argument("qi_attention_plain: head_dim 128");
    const sycl::device dev = queue_for(stream).get_device();
    const size_t wg = dev.get_info<sycl::info::device::max_work_group_size>();
    const size_t slm = dev.get_info<sycl::info::device::local_mem_size>();
    constexpr size_t kBig = (64 + 2 * 32) * (128 + 8) * 2 + 64 * 32 * 4;   // attention_tiles<64>'s local memory
    if (wg >= 512 && slm >= kBig) attention_tiles<64>(q16, k16, v16, T, heads, n_text, key_lim, out, stream);
    else attention_tiles<32>(q16, k16, v16, T, heads, n_text, key_lim, out, stream);
}

void qi_text_qk_rope16(const float* q, const float* k, const float* v, const float* wq, const float* wk, int64_t T,
                       int64_t q_heads, int64_t kv_heads, int64_t head_dim, float theta, uint16_t* q16, uint16_t* k16,
                       uint16_t* v16, void* stream, const int32_t* pos3, int sec_h, int sec_w) {
    if (T <= 0) return;
    if (head_dim % 2 != 0 || head_dim > 256) throw std::invalid_argument("qi_text_qk_rope16: head_dim");
    const size_t local = (size_t) head_dim;
    const int64_t heads = q_heads + kv_heads, half = head_dim / 2;
    const auto e = queue_for(stream).submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> row(local, h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (T * heads) * local), sycl::range<1>(local)),
                       [=](sycl::nd_item<1> it) {
            const auto g = static_cast<int64_t>(it.get_group_linear_id());
            const auto d = static_cast<int64_t>(it.get_local_linear_id());
            const int64_t t = g / heads, hh = g - t * heads;
            const bool isq = hh < q_heads;
            const int64_t i = isq ? (t * q_heads + hh) * head_dim + d : (t * kv_heads + hh - q_heads) * head_dim + d;
            const float val = isq ? q[i] : k[i];
            const float ss = sycl::reduce_over_group(it.get_group(), val * val, sycl::plus<float>());
            const float n = val / sycl::sqrt(ss / (float) head_dim + kEps) * (isq ? wq : wk)[d];
            row[d] = n;
            sycl::group_barrier(it.get_group());
            const int64_t j = d < half ? d : d - half;
            float pos = (float) t;
            if (pos3 != nullptr)
                pos = (float) (j % 3 == 1 && j < 3 * (int64_t) sec_h ? pos3[T + t]
                               : j % 3 == 2 && j < 3 * (int64_t) sec_w ? pos3[2 * T + t] : pos3[t]);
            const float ang = pos * sycl::pow(theta, -2.0f * (float) j / (float) head_dim);
            const float c = sycl::cos(ang), s = sycl::sin(ang);
            const float out = d < half ? n * c - row[d + half] * s : n * c + row[d - half] * s;
            (isq ? q16 : k16)[i] = f16_from_f32(out);
            if (!isq) v16[i] = f16_from_f32(v[i]);
        });
    });
    finish(stream, e, "qi_text_qk_rope16");
}

void qi_attention_causal(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t T, int64_t q_heads,
                         int64_t kv_heads, int64_t head_dim, float* out, void* stream) {
    if (T <= 0) return;
    if (head_dim != 128) throw std::invalid_argument("qi_attention_causal: head_dim 128");
    constexpr int64_t D = 128, TILE = 128;
    const int64_t qrow = q_heads * D, krow = kv_heads * D, group = q_heads / kv_heads;
    const float scale = 1.f / sycl::sqrt((float) D);
    const auto e = queue_for(stream).submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> qs(D, h), ps(TILE, h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (T * q_heads * D)), sycl::range<1>(D)),
                       [=](sycl::nd_item<1> it) {
            const auto g = static_cast<int64_t>(it.get_group_linear_id());
            const auto lane = static_cast<int64_t>(it.get_local_linear_id());
            const int64_t t = g / q_heads, hh = g - t * q_heads, kh = hh / group;
            qs[lane] = f32_from_f16(q16[t * qrow + hh * D + lane]) * scale;
            sycl::group_barrier(it.get_group());
            const int64_t keys = t + 1;
            float m = -INFINITY, l = 0.f, acc = 0.f;
            for (int64_t j0 = 0; j0 < keys; j0 += TILE) {
                const int64_t j = j0 + lane;
                float s = -INFINITY;
                if (j < keys) {
                    const uint16_t* kr = k16 + j * krow + kh * D;
                    s = 0.f;
                    for (int64_t d = 0; d < D; ++d) s += qs[d] * f32_from_f16(kr[d]);
                }
                const float nm = sycl::fmax(m, sycl::reduce_over_group(it.get_group(), s, sycl::maximum<float>()));
                const float p = j < keys ? sycl::exp(s - nm) : 0.f;
                const float corr = sycl::exp(m - nm);
                l = l * corr + sycl::reduce_over_group(it.get_group(), p, sycl::plus<float>());
                ps[lane] = p;
                sycl::group_barrier(it.get_group());
                acc *= corr;
                const int64_t n = sycl::min(TILE, keys - j0);
                for (int64_t jj = 0; jj < n; ++jj) acc += ps[jj] * f32_from_f16(v16[(j0 + jj) * krow + kh * D + lane]);
                sycl::group_barrier(it.get_group());
                m = nm;
            }
            out[t * qrow + hh * D + lane] = acc / l;
        });
    });
    finish(stream, e, "qi_attention_causal");
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

void qi_bias_add(float* y, const float* b, int64_t T, int64_t N, void* stream) {
    if (T <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) (T * N)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        y[i] += b[i % N];
    });
    finish(stream, e, "qi_bias_add");
}

void qi_layer_norm16(const float* x, const float* w, const float* b, int64_t T, int64_t C, float eps, uint16_t* y16,
                     void* stream) {
    if (T <= 0) return;
    constexpr size_t L = 256;
    const auto e = queue_for(stream).parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) T * L), sycl::range<1>(L)), [=](sycl::nd_item<1> it) {
            const auto t = static_cast<int64_t>(it.get_group_linear_id());
            const auto l = static_cast<int64_t>(it.get_local_linear_id());
            const float* r = x + t * C;
            float s = 0.f, s2 = 0.f;
            for (int64_t c = l; c < C; c += (int64_t) L) {
                s += r[c];
                s2 += r[c] * r[c];
            }
            s = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());
            s2 = sycl::reduce_over_group(it.get_group(), s2, sycl::plus<float>());
            const float mean = s / (float) C;
            const float inv = 1.f / sycl::sqrt(sycl::fmax(s2 / (float) C - mean * mean, 0.f) + eps);
            for (int64_t c = l; c < C; c += (int64_t) L) y16[t * C + c] = f16_from_f32((r[c] - mean) * inv * w[c] + b[c]);
        });
    finish(stream, e, "qi_layer_norm16");
}

void qi_gelu_erf16(const float* x, int64_t n, uint16_t* y16, void* stream) {
    if (n <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) {
        const float v = x[i];
        y16[i] = f16_from_f32(0.5f * v * (1.f + sycl::erf(v * 0.70710678118654752f)));
    });
    finish(stream, e, "qi_gelu_erf16");
}

void qi_vit_qkv_rope16(const float* qkv, const int32_t* yx, int64_t T, int64_t heads, int64_t hd, float theta,
                       float q_scale, uint16_t* q16, uint16_t* k16, uint16_t* v16, void* stream) {
    if (T <= 0) return;
    constexpr int64_t D = 128;
    const int64_t C = heads * hd, half = hd / 2, quarter = hd / 4;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) (T * heads * D)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t t = i / (heads * D), h = (i / D) % heads, d = i % D;
        const int64_t o = t * heads * D + h * D + d;
        if (d >= hd) {
            q16[o] = k16[o] = v16[o] = 0;
            return;
        }
        const float* row = qkv + t * 3 * C + h * hd;
        const int64_t j = d < half ? d : d - half;              // the pair's frequency index
        const float pos = (float) (j < quarter ? yx[2 * t] : yx[2 * t + 1]);
        const int64_t f = j < quarter ? j : j - quarter;
        const float ang = pos * sycl::pow(theta, -2.0f * (float) f / (float) half);
        const float c = sycl::cos(ang), s = sycl::sin(ang);
        auto rot = [&](const float* r) {
            return d < half ? r[d] * c - r[d + half] * s : r[d] * c + r[d - half] * s;
        };
        q16[o] = f16_from_f32(rot(row) * q_scale);
        k16[o] = f16_from_f32(rot(row + C));
        v16[o] = f16_from_f32(row[2 * C + d]);
    });
    finish(stream, e, "qi_vit_qkv_rope16");
}

void qi_unpad_heads16(const uint16_t* in16, int64_t T, int64_t heads, int64_t hd, uint16_t* out16, void* stream) {
    if (T <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) (T * heads * hd)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t t = i / (heads * hd), r = i % (heads * hd), h = r / hd, d = r % hd;
        out16[i] = in16[(t * heads + h) * 128 + d];
    });
    finish(stream, e, "qi_unpad_heads16");
}

}  // namespace strata::image
