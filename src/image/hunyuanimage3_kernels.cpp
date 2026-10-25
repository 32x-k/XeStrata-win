// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/hunyuanimage3_kernels.cpp - see include/strata/image/hunyuanimage3_kernels.hpp.
//
// The shapes are the model's (head_dim 128, hidden 4096, an intermediate of 3072 = 12 groups of 256) but nothing
// here is tuned: these are the plain forms, held to the host's numbers first (strata-image-hunyuanimage3-probe's
// `gpu` and `gpulayer`), and tile/local-memory versions come once the values agree.

#include "strata/image/hunyuanimage3_kernels.hpp"

#include "strata/core/runtime.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <sycl/sycl.hpp>

#include <stdexcept>

namespace strata::image {

// The kernels' helpers in a namespace of this file's own: SYCL names a lambda's kernel by its mangled type, and
// an anonymous namespace mangles the same in every file.
namespace hunyuanimage3_detail {

using strata::kernels::f16_from_f32;
using strata::kernels::f32_from_bf16;
using strata::kernels::f32_from_f16;

sycl::queue& queue_for(void* stream) { return core::Runtime::get().stream(stream); }

// The GroupNorm(32) statistics (hunyuanimage3_group_part and -stats): a work-group a chunk of kGnChunk positions and
// one of the 32 groups, the chunk sums reduced over at most kGnMaxChunk of them a work-group.  The VAE's 32 groups
// and the entry and exit's are the same reduction; only eps differs.
constexpr int64_t kGnGroups = 32;      // every GroupNorm here: nn.GroupNorm(32, C)
constexpr int64_t kGnChunk = 256;      // positions a chunk
constexpr int64_t kGnMaxChunk = 256;   // chunks a reducer's work-group

// The router: its dot products in partial sums of this many terms (FP32 throughout, the reference's own type, and
// no FP64, which not every GPU has), and the most experts a token takes.
constexpr int64_t kRouterChunk = 64;
constexpr int64_t kRouterMaxTopk = 32;

void finish(void* stream, const sycl::event& e, const char* what) {
    if (!stream) core::Runtime::get().wait(e, what);
}

// An FP8 e4m3 byte as a float: 4 exponent bits with a bias of 7, 3 mantissa bits, no inf and no NaN in the
// files.  e=0 is a subnormal m/8 * 2^-6 (the dequantization was calibrated against the BF16 weights,
// CALIB-2026-10-23.md).
float e4m3(uint8_t b) {
    const int s = (b >> 7) & 1, e = (b >> 3) & 0xF, m = b & 7;
    const float v = e == 0 ? ((float) m / 8.0f) * 0.015625f          // 2^-6
                           : (1.0f + (float) m / 8.0f) * sycl::pow(2.0f, (float) (e - 7));
    return s ? -v : v;
}

// One group of 256 values times the normalized regular Hadamard, in place: four butterflies on the group's
// base-4 digits (strides 64, 16, 4, 1).  H4 = [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]].
void hadamard256(float* t) {
    constexpr int h4[4][4] = {{1, 1, 1, -1}, {1, 1, -1, 1}, {1, -1, 1, 1}, {-1, 1, 1, 1}};
    float scale = 1.0f;
    for (int n = 256; n > 1; n /= 4) scale *= 0.5f;                 // 1/sqrt(256) = 1/16
    for (int stride = 64; stride >= 1; stride /= 4) {
        const int step = stride * 4;
        for (int base = 0; base < 256; base += step)
            for (int off = 0; off < stride; ++off) {
                float v[4];
                for (int i = 0; i < 4; ++i) v[i] = t[base + off + i * stride];
                for (int i = 0; i < 4; ++i) {
                    float sum = 0.f;
                    for (int j = 0; j < 4; ++j) sum += (float) h4[i][j] * v[j];
                    t[base + off + i * stride] = sum;
                }
            }
    }
    for (int i = 0; i < 256; ++i) t[i] *= scale;
}

}  // namespace hunyuanimage3_detail

using hunyuanimage3_detail::e4m3;
using hunyuanimage3_detail::f16_from_f32;
using hunyuanimage3_detail::f32_from_bf16;
using hunyuanimage3_detail::f32_from_f16;
using hunyuanimage3_detail::finish;
using hunyuanimage3_detail::hadamard256;
using hunyuanimage3_detail::kGnChunk;
using hunyuanimage3_detail::kGnGroups;
using hunyuanimage3_detail::kGnMaxChunk;
using hunyuanimage3_detail::kRouterChunk;
using hunyuanimage3_detail::kRouterMaxTopk;
using hunyuanimage3_detail::queue_for;

void hunyuanimage3_rms_norm16(const float* x, const float* w, int64_t rows, int64_t C, float eps, uint16_t* y16,
                    void* stream) {
    if (rows <= 0) return;
    constexpr size_t local = 256;
    const auto e = queue_for(stream).parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) rows * local), sycl::range<1>(local)), [=](sycl::nd_item<1> it) {
            const auto r = static_cast<int64_t>(it.get_group_linear_id());
            const auto lane = static_cast<int64_t>(it.get_local_linear_id());
            const float* xr = x + r * C;
            float s = 0.f;
            for (int64_t c = lane; c < C; c += (int64_t) local) s += xr[c] * xr[c];
            const float inv = 1.f / sycl::sqrt(sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>()) /
                                               (float) C + eps);
            uint16_t* yr = y16 + r * C;
            for (int64_t c = lane; c < C; c += (int64_t) local) yr[c] = f16_from_f32(xr[c] * inv * w[c]);
        });
    finish(stream, e, "hunyuanimage3_rms_norm16");
}

void hunyuanimage3_rms_norm32(const float* x, const float* w, int64_t rows, int64_t C, float eps, float* y32, void* stream) {
    if (rows <= 0) return;
    constexpr size_t local = 256;
    const auto e = queue_for(stream).parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) rows * local), sycl::range<1>(local)), [=](sycl::nd_item<1> it) {
            const auto r = static_cast<int64_t>(it.get_group_linear_id());
            const auto lane = static_cast<int64_t>(it.get_local_linear_id());
            const float* xr = x + r * C;
            float s = 0.f;
            for (int64_t c = lane; c < C; c += (int64_t) local) s += xr[c] * xr[c];
            const float inv = 1.f / sycl::sqrt(sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>()) /
                                               (float) C + eps);
            float* yr = y32 + r * C;
            for (int64_t c = lane; c < C; c += (int64_t) local) yr[c] = xr[c] * inv * w[c];
        });
    finish(stream, e, "hunyuanimage3_rms_norm32");
}

void hunyuanimage3_rotate16_from32(const float* x32, int64_t rows, int64_t K, int64_t block, uint16_t* out16, void* stream) {
    if (rows <= 0 || block <= 1) return;
    if (block != 256) throw std::runtime_error("hunyuanimage3_rotate16_from32: only the 256-wide group is implemented");
    const int64_t groups = rows * (K / 256);
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) groups), [=](sycl::id<1> i) {
        const int64_t g = (int64_t) i[0];
        const int64_t r = g / (K / 256), off = (g % (K / 256)) * 256;
        const float* src = x32 + r * K + off;
        uint16_t* dst = out16 + r * K + off;
        float t[256];
        for (int k = 0; k < 256; ++k) t[k] = src[k];
        hadamard256(t);
        for (int k = 0; k < 256; ++k) dst[k] = f16_from_f32(t[k]);   // the one rounding, after the spread
    });
    finish(stream, e, "hunyuanimage3_rotate16_from32");
}

void hunyuanimage3_rotate16(const uint16_t* x16, int64_t rows, int64_t K, int64_t block, uint16_t* out16, void* stream) {
    if (rows <= 0 || block <= 1) return;
    if (block != 256) throw std::runtime_error("hunyuanimage3_rotate16: only the 256-wide group is implemented");
    const int64_t groups = rows * (K / 256);
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) groups), [=](sycl::id<1> i) {
        const int64_t g = (int64_t) i[0];
        const int64_t r = g / (K / 256), off = (g % (K / 256)) * 256;
        const uint16_t* src = x16 + r * K + off;
        uint16_t* dst = out16 + r * K + off;
        float t[256];
        for (int k = 0; k < 256; ++k) t[k] = f32_from_f16(src[k]);
        hadamard256(t);
        for (int k = 0; k < 256; ++k) dst[k] = f16_from_f32(t[k]);
    });
    finish(stream, e, "hunyuanimage3_rotate16");
}

void hunyuanimage3_w4a8_dequant16(const int8_t* codes, const uint8_t* s_rel, const float* s_channel, const float* codebook,
                        int64_t r0, int64_t r1, int64_t N, int64_t K, int64_t group, uint16_t* out16,
                        void* stream) {
    if (r1 <= r0) return;
    (void) N;
    const int64_t rows = r1 - r0;
    const auto e = queue_for(stream).parallel_for(
        sycl::range<2>((size_t) rows, (size_t) (K / 2)), [=](sycl::id<2> i) {
            const int64_t r = (int64_t) i[0], c = (int64_t) i[1];
            const uint8_t byte = (uint8_t) codes[r * (K / 2) + c];
            // the even column (2c) in the low nibble, the odd (2c+1) in the high: measured against the BF16
            // weights (the other order gives cos 0.02)
            const int lo = byte & 0xF, hi = byte >> 4;
            const float sc = s_channel[r];
            const uint8_t* const sr = s_rel + r * (K / group);
            out16[r * K + 2 * c] = f16_from_f32(codebook[lo] * e4m3(sr[(2 * c) / group]) * sc);
            out16[r * K + 2 * c + 1] = f16_from_f32(codebook[hi] * e4m3(sr[(2 * c + 1) / group]) * sc);
        });
    finish(stream, e, "hunyuanimage3_w4a8_dequant16");
}

void hunyuanimage3_int8_dequant16(const int8_t* w, const float* row_scale, int64_t r0, int64_t r1, int64_t N, int64_t K,
                        uint16_t* out16, void* stream) {
    if (r1 <= r0) return;
    (void) N;
    const int64_t rows = r1 - r0;
    const auto e = queue_for(stream).parallel_for(
        sycl::range<2>((size_t) rows, (size_t) K), [=](sycl::id<2> i) {
            const int64_t r = (int64_t) i[0], c = (int64_t) i[1];
            out16[r * K + c] = f16_from_f32((float) w[r * K + c] * row_scale[r]);
        });
    finish(stream, e, "hunyuanimage3_int8_dequant16");
}

void hunyuanimage3_rope_angles(const int32_t* pos_y, const int32_t* pos_x, int64_t T, int64_t head_dim, float theta,
                     float* ang, void* stream) {
    if (T <= 0) return;
    const int64_t half = head_dim / 2;
    const auto e = queue_for(stream).parallel_for(sycl::range<2>((size_t) T, (size_t) half), [=](sycl::id<2> i) {
        const int64_t t = (int64_t) i[0], j = (int64_t) i[1];
        // FP32 as the reference's `1.0 / (base ** (arange(0, n, 2).float() / n))` (and no FP64, which not every
        // GPU has)
        const float pos = (float) ((j % 2) ? pos_x[t] : pos_y[t]);
        ang[t * half + j] = pos * (1.0f / sycl::pow(theta, (float) (2 * j) / (float) head_dim));
    });
    finish(stream, e, "hunyuanimage3_rope_angles");
}

void hunyuanimage3_qk_rope_norm16(const float* qkv, const float* wq, const float* wk, const float* ang, int64_t T,
                        int64_t q_heads, int64_t kv_heads, int64_t head_dim, float eps, uint16_t* q16,
                        uint16_t* k16, uint16_t* v16, void* stream) {
    if (T <= 0) return;
    const int64_t groups = q_heads / kv_heads;      // the reference's num_key_value_groups
    const int64_t row = q_heads + 2 * kv_heads;     // one token's (q head | k head | v head) jobs
    constexpr size_t local = 128;                   // one work-group a (token, head): head_dim 128 channels
    const auto e = queue_for(stream).parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) T * row * local), sycl::range<1>(local)), [=](sycl::nd_item<1> it) {
            const int64_t job = (int64_t) it.get_group_linear_id();
            const int64_t t = job / row, which = job % row;
            const int64_t lane = (int64_t) it.get_local_linear_id();
            // the file's order: [kv head][q head of that group, then k, then v][head_dim]
            int64_t src = 0;
            const float* w = nullptr;
            uint16_t* dst = nullptr;
            int64_t dst_head = 0;
            if (which < q_heads) {
                const int64_t kv = which / groups, g = which % groups;
                src = (t * kv_heads + kv) * (groups + 2) + g;
                w = wq;
                dst = q16;
                dst_head = which;
            } else if (which < q_heads + kv_heads) {
                const int64_t kv = which - q_heads;
                src = (t * kv_heads + kv) * (groups + 2) + groups;
                w = wk;
                dst = k16;
                dst_head = kv * groups;   // written to each of the q heads this KV head serves, below
            } else {
                const int64_t kv = which - q_heads - kv_heads;
                src = (t * kv_heads + kv) * (groups + 2) + groups + 1;
                dst = v16;
                dst_head = kv * groups;
            }
            const float* x = qkv + src * head_dim;
            if (which >= q_heads + kv_heads) {            // v: only the repeat
                for (int64_t r = 0; r < groups; ++r) dst[((t * q_heads) + dst_head + r) * head_dim + lane] =
                    f16_from_f32(x[lane]);
                return;
            }
            float s = 0.f;
            for (int64_t c = lane; c < head_dim; c += (int64_t) local) s += x[c] * x[c];
            const float inv = 1.f / sycl::sqrt(sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>()) /
                                               (float) head_dim + eps);
            // The rope before the norm, as the reference does.  The rotation is orthogonal on each pair, so the
            // head's sum of squares - the norm's denominator - is the same before and after it: reading it off
            // `x` is the same number, to rounding.
            const int64_t half = head_dim / 2;
            const int64_t j = lane < half ? lane : lane - half;                // the pair's low column
            const float a = x[lane], partner = x[lane < half ? lane + half : j];
            // (the partner is the OTHER half of the pair: for the upper column that is `x[j]`, not `x[j + half]`, which
            // is the column itself - written the second way, this kernel turned every head's upper 64 columns into
            // a*(cos + sin) and cost the layer 6% of q, 63% of k: `probe stages`, B70 and the CPU golden, 2026-10-24)
            const float c0 = sycl::cos(ang[t * (head_dim / 2) + j]), s0 = sycl::sin(ang[t * (head_dim / 2) + j]);
            // rotate_half, the SAME angle for both halves, then the norm (the reference's order)
            const float rot = lane < half ? a * c0 - partner * s0 : a * c0 + partner * s0;
            const float val = rot * inv * w[lane];
            const int64_t reps = which < q_heads ? 1 : groups;   // q has its own head; k serves `groups`
            for (int64_t r = 0; r < reps; ++r)
                dst[((t * q_heads) + dst_head + r) * head_dim + lane] = f16_from_f32(val);
        });
    finish(stream, e, "hunyuanimage3_qk_rope_norm16");
}

void hunyuanimage3_swiglu16_rotated(const float* gu, int64_t T, int64_t inter, uint16_t* out16, void* stream) {
    if (T <= 0) return;
    if (inter % 256 != 0) throw std::runtime_error("hunyuanimage3_swiglu16_rotated: the intermediate is not in 256 groups");
    const int64_t groups = T * (inter / 256);
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) groups), [=](sycl::id<1> i) {
        const int64_t t = (int64_t) i[0] / (inter / 256), off = ((int64_t) i[0] % (inter / 256)) * 256;
        const float* g = gu + t * 2 * inter + off;
        float v[256];
        for (int k = 0; k < 256; ++k) {
            const float gate = g[k], up = g[inter + k];
            v[k] = gate * (up / (1.f + sycl::exp(-up)));     // x1 * silu(x2), the SECOND half silu'd
        }
        hadamard256(v);
        uint16_t* o = out16 + t * inter + off;
        for (int k = 0; k < 256; ++k) o[k] = f16_from_f32(v[k]);
    });
    finish(stream, e, "hunyuanimage3_swiglu16_rotated");
}

void hunyuanimage3_gather16(const uint16_t* x16, const int32_t* rows, int64_t n_rows, int64_t K, uint16_t* out16,
                  void* stream) {
    if (n_rows <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<2>((size_t) n_rows, (size_t) K), [=](sycl::id<2> i) {
        const int64_t r = (int64_t) i[0], c = (int64_t) i[1];
        out16[r * K + c] = x16[(int64_t) rows[r] * K + c];
    });
    finish(stream, e, "hunyuanimage3_gather16");
}

void hunyuanimage3_scatter_scale_add(float* x, const float* y, const float* w, const int32_t* rows, int64_t n_rows,
                           int64_t C, void* stream) {
    if (n_rows <= 0) return;
    const auto e = queue_for(stream).parallel_for(
        sycl::range<2>((size_t) n_rows, (size_t) C), [=](sycl::id<2> i) {
            const int64_t r = (int64_t) i[0], c = (int64_t) i[1];
            sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device> target(
                x[(int64_t) rows[r] * C + c]);
            target += w[r] * y[r * C + c];
        });
    finish(stream, e, "hunyuanimage3_scatter_scale_add");
}

void hunyuanimage3_add(float* x, const float* v, int64_t n, void* stream) {
    if (n <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) {
        x[i[0]] += v[i[0]];
    });
    finish(stream, e, "hunyuanimage3_add");
}

void hunyuanimage3_embed_bf16_16(const uint16_t* table_bf16, const int32_t* ids, int64_t rows, int64_t C, uint16_t* out16,
                       void* stream) {
    if (rows <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<2>((size_t) rows, (size_t) C), [=](sycl::id<2> i) {
        const int64_t r = (int64_t) i[0], c = (int64_t) i[1];
        out16[r * C + c] = f16_from_f32(f32_from_bf16(table_bf16[(int64_t) ids[r] * C + c]));
    });
    finish(stream, e, "hunyuanimage3_embed_bf16_16");
}

void hunyuanimage3_copy16(const uint16_t* src16, uint16_t* dst16, int64_t n, void* stream) {
    if (n <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) {
        dst16[i[0]] = src16[i[0]];
    });
    finish(stream, e, "hunyuanimage3_copy16");
}

void hunyuanimage3_router_logits(const float* x32, const float* w, int64_t T, int64_t C, int64_t E, float* logits,
                       void* stream) {
    if (T <= 0) return;
    const auto e = queue_for(stream).parallel_for(
        sycl::range<2>((size_t) T, (size_t) E), [=](sycl::id<2> i) {
            const int64_t t = (int64_t) i[0], ex = (int64_t) i[1];
            const float* xr = x32 + t * C;
            const float* wr = w + ex * C;
            float sum = 0.f;
            for (int64_t c0 = 0; c0 < C; c0 += kRouterChunk) {
                float part = 0.f;
                const int64_t c1 = c0 + kRouterChunk < C ? c0 + kRouterChunk : C;
                for (int64_t c = c0; c < c1; ++c) part += xr[c] * wr[c];
                sum += part;
            }
            logits[t * E + ex] = sum;
        });
    finish(stream, e, "hunyuanimage3_router_logits");
}

void hunyuanimage3_router_topk(const float* logits, int64_t T, int64_t E, int64_t K, int32_t* idx, float* wt,
                               void* stream) {
    if (T <= 0) return;
    if (K <= 0 || K > kRouterMaxTopk || K > E)
        throw std::runtime_error("hunyuanimage3_router_topk: " + std::to_string(K) + " of " + std::to_string(E) +
                                 " experts");
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) T), [=](sycl::id<1> id) {
        const int64_t t = (int64_t) id[0];
        const float* lg = logits + t * E;
        int32_t pick[kRouterMaxTopk];
        for (int64_t j = 0; j < K; ++j) {
            int32_t best = -1;
            for (int64_t ex = 0; ex < E; ++ex) {
                bool taken = false;
                for (int64_t i = 0; i < j; ++i) taken = taken || pick[i] == (int32_t) ex;
                if (!taken && (best < 0 || lg[ex] > lg[best])) best = (int32_t) ex;   // the lower expert on a tie
            }
            pick[j] = best;
        }
        // softmax over the K picked (norm_topk_prob: the full softmax's denominator cancels), from the largest
        const float m = lg[pick[0]];
        float s = 0.f;
        for (int64_t j = 0; j < K; ++j) s += sycl::exp(lg[pick[j]] - m);
        for (int64_t j = 0; j < K; ++j) {
            idx[t * K + j] = pick[j];
            wt[t * K + j] = sycl::exp(lg[pick[j]] - m) / s;
        }
    });
    finish(stream, e, "hunyuanimage3_router_topk");
}

void hunyuanimage3_route_lists(const int32_t* idx, const float* wt, int64_t T, int64_t K, int64_t E, int32_t* bounds,
                               int32_t* rows, float* w, void* stream) {
    if (T <= 0) return;
    sycl::queue& q = queue_for(stream);
    const int64_t n = T * K;
    // an expert a work-item: its count, then (after the prefix) its rows in token order
    q.parallel_for(sycl::range<1>((size_t) E), [=](sycl::id<1> id) {
        const int32_t ex = (int32_t) id[0];
        int32_t c = 0;
        for (int64_t i = 0; i < n; ++i) c += idx[i] == ex ? 1 : 0;
        bounds[ex + 1] = c;
    });
    q.single_task([=]() {
        bounds[0] = 0;
        for (int64_t ex = 0; ex < E; ++ex) bounds[ex + 1] += bounds[ex];
    });
    const auto e = q.parallel_for(sycl::range<1>((size_t) E), [=](sycl::id<1> id) {
        const int32_t ex = (int32_t) id[0];
        int32_t at = bounds[ex];
        for (int64_t i = 0; i < n; ++i)
            if (idx[i] == ex) {
                rows[at] = (int32_t) (i / K);
                w[at] = wt[i];
                ++at;
            }
    });
    finish(stream, e, "hunyuanimage3_route_lists");
}

void hunyuanimage3_grid_in16(const float* x, int64_t C, int64_t P, uint16_t* out16, void* stream, uint16_t* lo16) {
    if (C <= 0 || P <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) (P * C)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t p = i / C, c = i - p * C;
        const float v = x[c * P + p];
        out16[i] = f16_from_f32(v);
        if (lo16 != nullptr) lo16[i] = f16_from_f32(v - f32_from_f16(out16[i]));
    });
    finish(stream, e, "hunyuanimage3_grid_in16");
}

void hunyuanimage3_to16(const float* x, int64_t n, uint16_t* y16, void* stream, uint16_t* lo16) {
    if (n <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const float v = x[id[0]];
        y16[id[0]] = f16_from_f32(v);
        if (lo16 != nullptr) lo16[id[0]] = f16_from_f32(v - f32_from_f16(y16[id[0]]));
    });
    finish(stream, e, "hunyuanimage3_to16");
}

void hunyuanimage3_im2col16(const uint16_t* in16, int64_t H, int64_t W, int64_t C, int64_t p0, int64_t rows,
                            uint16_t* col16, void* stream) {
    if (rows <= 0) return;
    if (C % 8 != 0) throw std::runtime_error("hunyuanimage3_im2col16: C is not a multiple of 8");
    const int64_t C8 = C / 8;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) (rows * 9 * C8)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t c8 = (i % C8) * 8, rk = i / C8, r = rk / 9, tap = rk - r * 9;
        const int64_t p = p0 + r, x = p % W, y = p / W;
        const int64_t yy = y + tap / 3 - 1, xx = x + tap % 3 - 1;
        using V = sycl::vec<uint16_t, 8>;
        V v(0);
        if (yy >= 0 && yy < H && xx >= 0 && xx < W)
            v = *reinterpret_cast<const V*>(in16 + (yy * W + xx) * C + c8);
        *reinterpret_cast<V*>(col16 + (r * 9 + tap) * C + c8) = v;
    });
    finish(stream, e, "hunyuanimage3_im2col16");
}

void hunyuanimage3_bias_add32(float* y, const float* b, int64_t rows, int64_t N, void* stream) {
    if (rows <= 0 || N <= 0) return;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) (rows * N)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        y[i] += b[i % N];
    });
    finish(stream, e, "hunyuanimage3_bias_add32");
}

int64_t hunyuanimage3_group_ws_floats(int64_t P) {
    const int64_t B = (P + kGnChunk - 1) / kGnChunk, G = (B + kGnMaxChunk - 1) / kGnMaxChunk;
    return 2 * kGnGroups + (B + G) * kGnGroups;
}

void hunyuanimage3_group_part(const float* x, const float* mean, int64_t P, int64_t C, bool sq, float* part,
                              void* stream) {
    if (P <= 0) return;
    const int64_t cg = C / kGnGroups, B = (P + kGnChunk - 1) / kGnChunk;
    const auto e = queue_for(stream).parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) (B * kGnGroups * 32)), sycl::range<1>(32)), [=](sycl::nd_item<1> it) {
            const auto lane = static_cast<int64_t>(it.get_local_linear_id());
            const auto blk = static_cast<int64_t>(it.get_group_linear_id());
            const int64_t b = blk / kGnGroups, g = blk - b * kGnGroups;
            const int64_t p1 = std::min<int64_t>(P, (b + 1) * kGnChunk);
            const float m = sq ? mean[g] : 0.f;
            float a = 0.f, comp = 0.f;
            for (int64_t p = b * kGnChunk + lane; p < p1; p += 32) {
                const float* r = x + p * C + g * cg;
                for (int64_t c = 0; c < cg; ++c) {
                    const float d = r[c] - m;
                    const float y = (sq ? d * d : d) - comp, t = a + y;
                    comp = (t - a) - y;
                    a = t;
                }
            }
            a = sycl::reduce_over_group(it.get_group(), a, sycl::plus<float>());
            if (lane == 0) part[blk] = a;
        });
    finish(stream, e, "hunyuanimage3_group_part");
}

void hunyuanimage3_group_stats(const float* part, int64_t B, int64_t P, int64_t CG, float eps, bool sq,
                               float* stats, float* mid, void* stream) {
    const int64_t G = (B + kGnMaxChunk - 1) / kGnMaxChunk;
    if (G > 1) {
        const auto e = queue_for(stream).parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) (G * kGnGroups * 32)), sycl::range<1>(32)),
            [=](sycl::nd_item<1> it) {
                const auto lane = static_cast<int64_t>(it.get_local_linear_id());
                const auto blk = static_cast<int64_t>(it.get_group_linear_id());
                const int64_t gi = blk / kGnGroups, g = blk - gi * kGnGroups;
                const int64_t b1 = std::min<int64_t>(B, (gi + 1) * kGnMaxChunk);
                float a = 0.f, comp = 0.f;
                for (int64_t b = gi * kGnMaxChunk + lane; b < b1; b += 32) {
                    const float y = part[b * kGnGroups + g] - comp, t = a + y;
                    comp = (t - a) - y;
                    a = t;
                }
                a = sycl::reduce_over_group(it.get_group(), a, sycl::plus<float>());
                if (lane == 0) mid[gi * kGnGroups + g] = a;
            });
        finish(stream, e, "hunyuanimage3_group_stats (the reducers' level)");
        queue_for(stream).wait();   // the last level reads every group's entry
    }
    // one work-item a group, serial Kahan over the (at most kGnMaxChunk) entries of its slice, or over the B chunk
    // sums themselves when they fit one reducer
    const int64_t MG = G > 1 ? G : B;
    const float* src = G > 1 ? mid : part;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) kGnGroups), [=](sycl::id<1> id) {
        const auto g = static_cast<int64_t>(id[0]);
        float a = 0.f, comp = 0.f;
        for (int64_t i = 0; i < MG; ++i) {
            const float y = src[i * kGnGroups + g] - comp, t = a + y;
            comp = (t - a) - y;
            a = t;
        }
        const float n = (float) P * (float) CG;   // FP32 here too (no device FP64): the sums are Kahan-folded
        stats[g] = sq ? 1.0f / sycl::sqrt(a / n + eps) : a / n;   // and a 1e-7 share of a ratio is far under the
    });                                                          // FP16 floor the convolutions set
    finish(stream, e, "hunyuanimage3_group_stats");
}

void hunyuanimage3_group_apply16(const float* x, const float* mean, const float* inv, const float* w,
                                 const float* b, const float* scale, const float* shift, int64_t P, int64_t C,
                                 bool silu, uint16_t* y16, void* stream, uint16_t* lo16) {
    if (P <= 0) return;
    const int64_t cg = C / kGnGroups;
    const auto e = queue_for(stream).parallel_for(sycl::range<1>((size_t) (P * C)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t c = i % C, g = c / cg;
        float v = (x[i] - mean[g]) * inv[g] * w[c] + b[c];
        if (scale != nullptr) v = v * (1.f + scale[c]) + shift[c];
        if (silu) v = v / (1.f + sycl::exp(-v));
        y16[i] = f16_from_f32(v);
        if (lo16 != nullptr) lo16[i] = f16_from_f32(v - f32_from_f16(y16[i]));
    });
    finish(stream, e, "hunyuanimage3_group_apply16");
}

void hunyuanimage3_scale_rows16(const float* x, const float* rs, int64_t n_rows, int64_t C, uint16_t* y16, void* stream) {
    if (n_rows <= 0) return;
    const auto e = queue_for(stream).parallel_for(
        sycl::range<2>((size_t) n_rows, (size_t) C), [=](sycl::id<2> i) {
            const int64_t r = (int64_t) i[0], c = (int64_t) i[1];
            const float s = rs != nullptr ? rs[r] : 1.f;
            y16[r * C + c] = f16_from_f32(x[r * C + c] / s);
        });
    finish(stream, e, "hunyuanimage3_scale_rows16");
}

}  // namespace strata::image
