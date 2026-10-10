// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/prompt_attn.cpp - see include/strata/kernels/prompt_attn.hpp.
#include "strata/kernels/prompt_attn.hpp"
#include "strata/core/runtime.hpp"
#include "device_target.hpp"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdlib>
#include <limits>

namespace strata::kernels {
namespace {

constexpr int NT = 256;   // work-items a work-group

// the device's own conversion (f16_bits.hpp's exact one branches and loops: in the inner loops it took most of the
// time); the pools hold no subnormals worth the difference
[[maybe_unused]] inline float h2f(uint16_t bits) { return (float) sycl::bit_cast<sycl::half>(bits); }

// BQ queries of one KV head: R = BQ x G rows (query, head), Q in local memory as FP32 (scaled), the step's BK cells of
// K and V as FP16, the step's scores (then probabilities) per row.  Local memory is read 4 values at a time (Q and
// the probabilities as float4, K as 4 halves), the rows padded to keep those reads aligned and K's rows off one bank.
// Scores: work-item t takes cell t % BK of rows t / BK, + NT / BK, ...: neighbouring work-items read neighbouring K
// rows and the same Q row.  Output: work-item t owns column d = t % HD of rows t / HD, + NT / HD, ... in registers.
template <int HD, int G, int BQ, int BK>
struct Kernel {
    static constexpr int R = BQ * G, RS = NT / HD, OPT = R / RS, QS = HD + 4, KS = HD + 4;
    static constexpr int SG = NT / BK, SR = (R + SG - 1) / SG;   // the scores: row groups, rows a work-item
    static_assert(NT % HD == 0 && R % RS == 0 && HD % 8 == 0 && BK % 4 == 0, "the work splits evenly");
    using f4 = sycl::vec<float, 4>;
    using h4 = sycl::vec<uint16_t, 4>;
    using h8 = sycl::vec<uint16_t, 8>;
    const float* q;
    const uint16_t* kp;
    const uint16_t* vp;
    const int32_t* table;
    float* attn;
    int64_t pos0, T;
    int n_head, n_kv, page_size;
    float scale;
    sycl::local_accessor<float, 1> qs, sc, mrow, lrow, arow;
    sycl::local_accessor<uint16_t, 1> ks, vs;

    void operator()(sycl::nd_item<2> it) const {
        constexpr float NEG = -std::numeric_limits<float>::infinity();
        const int t = (int) it.get_local_id(1), h = (int) it.get_group(0);
        const int64_t qbase = (int64_t) it.get_group(1) * BQ;
        const int d = t % HD, r0 = t / HD;
        float* Q = qs.get_multi_ptr<sycl::access::decorated::no>().get();
        float* S = sc.get_multi_ptr<sycl::access::decorated::no>().get();
        float* M = mrow.get_multi_ptr<sycl::access::decorated::no>().get();
        float* L = lrow.get_multi_ptr<sycl::access::decorated::no>().get();
        float* A = arow.get_multi_ptr<sycl::access::decorated::no>().get();
        uint16_t* K = ks.get_multi_ptr<sycl::access::decorated::no>().get();
        uint16_t* V = vs.get_multi_ptr<sycl::access::decorated::no>().get();
        for (int i = t; i < R * HD; i += NT) {
            const int r = i / HD, dd = i - r * HD;
            const int64_t tq = qbase + r / G;
            Q[r * QS + dd] = tq < T ? q[(tq * n_head + (int64_t) h * G + r % G) * HD + dd] * scale : 0.0f;
        }
        if (t < R) {
            M[t] = NEG;
            L[t] = 0.0f;
        }
        float o[OPT];
        for (int k = 0; k < OPT; ++k) o[k] = 0.0f;
        const int64_t n_cells = pos0 + sycl::min<int64_t>(qbase + BQ, T);   // through the block's last query
        const int j = t % BK, rg = t / BK;
        for (int64_t c0 = 0; c0 < n_cells; c0 += BK) {
            sycl::group_barrier(it.get_group());   // the last step's K, V and probabilities are read
            for (int i = t; i < BK * HD / 8; i += NT) {   // 8 halves a work-item
                const int jj = i / (HD / 8), dd = (i - jj * (HD / 8)) * 8;
                const int64_t c = c0 + jj;
                h8 kv(0), vv(0);
                if (c < n_cells) {
                    const int64_t page = table[c / page_size];
                    const int64_t row = ((page * n_kv + h) * page_size + c % page_size) * HD + dd;
                    kv = *reinterpret_cast<const h8*>(kp + row);
                    vv = *reinterpret_cast<const h8*>(vp + row);
                }
                *reinterpret_cast<h4*>(K + jj * KS + dd) = kv.template swizzle<0, 1, 2, 3>();
                *reinterpret_cast<h4*>(K + jj * KS + dd + 4) = kv.template swizzle<4, 5, 6, 7>();
                *reinterpret_cast<h8*>(V + jj * HD + dd) = vv;
            }
            sycl::group_barrier(it.get_group());
            {   // scores, masked past each row's own position
                const int64_t c = c0 + j;
                float acc[SR];
                for (int k = 0; k < SR; ++k) acc[k] = 0.0f;
                for (int dd = 0; dd < HD; dd += 4) {
                    const h4 kh = *reinterpret_cast<const h4*>(K + j * KS + dd);
                    const float k0 = h2f(kh[0]), k1 = h2f(kh[1]), k2 = h2f(kh[2]), k3 = h2f(kh[3]);
                    for (int k = 0; k < SR; ++k) {
                        const int r = rg + k * SG;
                        if (SR * SG > R && r >= R) break;
                        const f4 qv = *reinterpret_cast<const f4*>(Q + r * QS + dd);
                        acc[k] = sycl::fma(qv[0], k0, sycl::fma(qv[1], k1, sycl::fma(qv[2], k2, sycl::fma(qv[3], k3, acc[k]))));
                    }
                }
                for (int k = 0; k < SR; ++k) {
                    const int r = rg + k * SG;
                    if (r >= R) break;
                    S[r * BK + j] = c <= pos0 + qbase + r / G && c < n_cells ? acc[k] : NEG;
                }
            }
            sycl::group_barrier(it.get_group());
            if (t < R) {   // the row's online softmax: its new maximum, the old sum's factor, the step's probabilities
                float mx = M[t];
                for (int jj = 0; jj < BK; ++jj) mx = sycl::fmax(mx, S[t * BK + jj]);
                const float a = M[t] == NEG ? 0.0f : sycl::exp(M[t] - mx);
                float sum = 0.0f;
                for (int jj = 0; jj < BK; ++jj) {
                    const float p = S[t * BK + jj] == NEG ? 0.0f : sycl::exp(S[t * BK + jj] - mx);
                    S[t * BK + jj] = p;
                    sum += p;
                }
                L[t] = L[t] * a + sum;
                M[t] = mx;
                A[t] = a;
            }
            sycl::group_barrier(it.get_group());
            for (int k = 0; k < OPT; ++k) o[k] *= A[r0 + k * RS];
            for (int jj = 0; jj < BK; jj += 4) {
                const float v0 = h2f(V[jj * HD + d]), v1 = h2f(V[(jj + 1) * HD + d]), v2 = h2f(V[(jj + 2) * HD + d]),
                            v3 = h2f(V[(jj + 3) * HD + d]);
                for (int k = 0; k < OPT; ++k) {
                    const f4 p = *reinterpret_cast<const f4*>(S + (r0 + k * RS) * BK + jj);
                    o[k] = sycl::fma(p[0], v0, sycl::fma(p[1], v1, sycl::fma(p[2], v2, sycl::fma(p[3], v3, o[k]))));
                }
            }
        }
        for (int k = 0; k < OPT; ++k) {
            const int r = r0 + k * RS;
            const int64_t tq = qbase + r / G;
            if (tq < T) attn[(tq * n_head + (int64_t) h * G + r % G) * HD + d] = o[k] / L[r];
        }
    }
};

// RDNA3 and RDNA4 (GFX11, GFX12): Q.K^T and P.V through the FP16 WMMA instructions (16 x 16 x 16, FP32
// accumulators; the layouts as iq_mmq.cpp's int8 ones: lane l holds row l % 16 of A and column l % 16 of B and of the
// accumulator; GFX12 8 of the 16 K values a lane (l / 16 picks which) and accumulator rows 8 (l / 16) + e, GFX11 all
// 16 and rows 2 e + l / 16).  BQ queries of one KV head, R = BQ x G rows in 16-row tiles; 32 cells a step: K, then V
// transposed, through one local buffer; the scores to FP32 local memory, the softmax by a work-item a row, the
// probabilities to FP16 for P.V.  The output tiles stay in the waves' accumulators across the steps, rescaled by
// each row's factor where the layout puts the row.
#if defined(__SYCL_DEVICE_ONLY__) && defined(__AMDGCN__) && (defined(__GFX11__) || defined(__GFX12__))
#define STRATA_PA_WMMA 1
#else
#define STRATA_PA_WMMA 0
#endif

template <int HD, int G, int BQ>
struct WmmaKernel {
    static constexpr int BK = 32, W = NT / 32, R = BQ * G, RT = R / 16, DT = HD / 16, NO = RT * DT / W;
    static constexpr int QS = HD + 8, KS = HD + 8, VS = BK + 8, PS = BK + 8;   // local row strides (halves)
    static constexpr int KV = BK * KS > HD * VS ? BK * KS : HD * VS;
    static_assert(R % 16 == 0 && (RT * DT) % W == 0 && HD % 16 == 0, "whole tiles, shared evenly by the waves");
    using h8 = sycl::vec<uint16_t, 8>;
    const float* q;
    const uint16_t* kp;
    const uint16_t* vp;
    const int32_t* table;
    float* attn;
    int64_t pos0, T;
    int n_head, n_kv, page_size;
    float scale;
    sycl::local_accessor<uint16_t, 1> qs, kv, ps;
    sycl::local_accessor<float, 1> sc, mrow, lrow, arow;

    void operator()(sycl::nd_item<2> it) const {
#if STRATA_PA_WMMA
        using hv8 = _Float16 __attribute__((ext_vector_type(8)));
        using fv8 = float __attribute__((ext_vector_type(8)));
#if defined(__GFX12__)
        using frag = hv8;
        constexpr int KOFF = 8, KN = 8;   // a lane's K values: 8 of the 16, the half l / 16 picks
        auto row_of = [](int hi, int e) { return 8 * hi + e; };
#else
        using frag = _Float16 __attribute__((ext_vector_type(16)));
        constexpr int KOFF = 0, KN = 16;
        auto row_of = [](int hi, int e) { return 2 * e + hi; };
#endif
        auto mma = [](frag a, frag b, fv8 c) {
#if defined(__GFX12__)
            return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
#else
            return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#endif
        };
        constexpr float NEG = -std::numeric_limits<float>::infinity();
        const int t = (int) it.get_local_id(1), h = (int) it.get_group(0);
        const int wave = t / 32, lane = t % 32, lo = lane & 15, hi = lane >> 4;
        const int64_t qbase = (int64_t) it.get_group(1) * BQ;
        uint16_t* Q = qs.get_multi_ptr<sycl::access::decorated::no>().get();
        uint16_t* KVb = kv.get_multi_ptr<sycl::access::decorated::no>().get();
        uint16_t* P = ps.get_multi_ptr<sycl::access::decorated::no>().get();
        float* S = sc.get_multi_ptr<sycl::access::decorated::no>().get();
        float* M = mrow.get_multi_ptr<sycl::access::decorated::no>().get();
        float* L = lrow.get_multi_ptr<sycl::access::decorated::no>().get();
        float* A = arow.get_multi_ptr<sycl::access::decorated::no>().get();
        auto frag_at = [](const uint16_t* p) { return *reinterpret_cast<const frag*>(p); };
        for (int i = t; i < R * HD; i += NT) {
            const int r = i / HD, dd = i - r * HD;
            const int64_t tq = qbase + r / G;
            const float v = tq < T ? q[(tq * n_head + (int64_t) h * G + r % G) * HD + dd] * scale : 0.0f;
            Q[r * QS + dd] = sycl::bit_cast<uint16_t>(sycl::half(v));
        }
        if (t < R) {
            M[t] = NEG;
            L[t] = 0.0f;
        }
        fv8 o[NO];
        for (int k = 0; k < NO; ++k) o[k] = fv8{0, 0, 0, 0, 0, 0, 0, 0};
        const int64_t n_cells = pos0 + sycl::min<int64_t>(qbase + BQ, T);   // through the block's last query
        for (int64_t c0 = 0; c0 < n_cells; c0 += BK) {
            sycl::group_barrier(it.get_group());   // the last step's V and probabilities are read
            for (int i = t; i < BK * HD / 8; i += NT) {   // K: [cell][dim]
                const int j = i / (HD / 8), dd = (i - j * (HD / 8)) * 8;
                const int64_t c = c0 + j;
                h8 v(0);
                if (c < n_cells) {
                    const int64_t page = table[c / page_size];
                    v = *reinterpret_cast<const h8*>(kp + ((page * n_kv + h) * page_size + c % page_size) * HD + dd);
                }
                *reinterpret_cast<h8*>(KVb + j * KS + dd) = v;
            }
            sycl::group_barrier(it.get_group());
            for (int tile = wave; tile < RT * (BK / 16); tile += W) {   // S = Q . K^T, a 16 x 16 tile a wave
                const int rt = tile / (BK / 16), ct = tile % (BK / 16);
                fv8 acc = fv8{0, 0, 0, 0, 0, 0, 0, 0};
                for (int k0 = 0; k0 < HD; k0 += 16)
                    acc = mma(frag_at(Q + (rt * 16 + lo) * QS + k0 + hi * KOFF),
                              frag_at(KVb + (ct * 16 + lo) * KS + k0 + hi * KOFF), acc);
                for (int e = 0; e < 8; ++e) S[(rt * 16 + row_of(hi, e)) * BK + ct * 16 + lo] = acc[e];
            }
            sycl::group_barrier(it.get_group());
            for (int i = t; i < BK * HD / 8; i += NT) {   // V transposed: [dim][cell]
                const int j = i / (HD / 8), dd = (i - j * (HD / 8)) * 8;
                const int64_t c = c0 + j;
                h8 v(0);
                if (c < n_cells) {
                    const int64_t page = table[c / page_size];
                    v = *reinterpret_cast<const h8*>(vp + ((page * n_kv + h) * page_size + c % page_size) * HD + dd);
                }
                for (int e = 0; e < 8; ++e) KVb[(dd + e) * VS + j] = v[e];
            }
            if (t < R) {   // the row's online softmax: its new maximum, the old sum's factor, the step's probabilities
                const int64_t qpos = pos0 + qbase + t / G;
                float mx = M[t];
                for (int j = 0; j < BK; ++j)
                    if (c0 + j <= qpos && c0 + j < n_cells) mx = sycl::fmax(mx, S[t * BK + j]);
                const float a = M[t] == NEG ? 0.0f : sycl::exp(M[t] - mx);
                float sum = 0.0f;
                for (int j = 0; j < BK; ++j) {
                    const float p = c0 + j <= qpos && c0 + j < n_cells ? sycl::exp(S[t * BK + j] - mx) : 0.0f;
                    const sycl::half ph(p);
                    P[t * PS + j] = sycl::bit_cast<uint16_t>(ph);
                    sum += (float) ph;
                }
                L[t] = L[t] * a + sum;
                M[t] = mx;
                A[t] = a;
            }
            sycl::group_barrier(it.get_group());
            for (int k = 0; k < NO; ++k) {   // O += P . V on the wave's own output tiles
                const int tile = wave + k * W, rt = tile / DT, dt = tile % DT;
                for (int e = 0; e < 8; ++e) o[k][e] *= A[rt * 16 + row_of(hi, e)];
                for (int k0 = 0; k0 < BK; k0 += 16)
                    o[k] = mma(frag_at(P + (rt * 16 + lo) * PS + k0 + hi * KOFF),
                               frag_at(KVb + (dt * 16 + lo) * VS + k0 + hi * KOFF), o[k]);
            }
        }
        (void) KN;
        for (int k = 0; k < NO; ++k) {
            const int tile = wave + k * W, rt = tile / DT, dt = tile % DT;
            for (int e = 0; e < 8; ++e) {
                const int r = rt * 16 + row_of(hi, e);
                const int64_t tq = qbase + r / G;
                if (tq < T) attn[(tq * n_head + (int64_t) h * G + r % G) * HD + dt * 16 + lo] = o[k][e] / L[r];
            }
        }
#else
        (void) it;
#endif
    }
    auto get(sycl::ext::oneapi::experimental::properties_tag) const {
        return sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::sub_group_size<STRATA_SUB_GROUP(32)>};
    }
};

// GFX11 and GFX12, as SYCL reports the architecture (iq_mmq.cpp's amd_wmma: the values carry the version)
bool amd_wmma(const sycl::device& d) {
    namespace syclex = sycl::ext::oneapi::experimental;
    if (d.get_backend() != sycl::backend::ext_oneapi_hip) return false;
    const auto a = static_cast<uint64_t>(d.get_info<syclex::info::device::architecture>());
    return a >= static_cast<uint64_t>(syclex::architecture::amd_gpu_gfx1100) && a < 0x0200000000130000ull;
}

template <int HD, int G, int BQ>
void launch_wmma(sycl::queue& q, const float* qv, const uint16_t* kp, const uint16_t* vp, const int32_t* table,
                 int64_t pos0, int64_t T, const QsaShapes& s, float* attn) {
    using KT = WmmaKernel<HD, G, BQ>;
    const size_t blocks = (size_t) ((T + BQ - 1) / BQ);
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint16_t, 1> qs(sycl::range<1>((size_t) KT::R * KT::QS), h), kv(sycl::range<1>((size_t) KT::KV), h),
            ps(sycl::range<1>((size_t) KT::R * KT::PS), h);
        sycl::local_accessor<float, 1> sc(sycl::range<1>((size_t) KT::R * KT::BK), h), mrow(sycl::range<1>(KT::R), h),
            lrow(sycl::range<1>(KT::R), h), arow(sycl::range<1>(KT::R), h);
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t) s.n_head_kv, blocks * NT), sycl::range<2>(1, NT)),
                       KT{qv, kp, vp, table, attn, pos0, T, (int) s.n_head, (int) s.n_head_kv, (int) s.page_size,
                          1.0f / std::sqrt((float) HD), qs, kv, ps, sc, mrow, lrow, arow});
    });
}

template <int HD, int G, int BQ, int BK>
void launch(sycl::queue& q, const float* qv, const uint16_t* kp, const uint16_t* vp, const int32_t* table, int64_t pos0,
            int64_t T, const QsaShapes& s, float* attn) {
    using KT = Kernel<HD, G, BQ, BK>;
    const size_t blocks = (size_t) ((T + BQ - 1) / BQ);
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> qs(sycl::range<1>((size_t) KT::R * KT::QS), h), sc(sycl::range<1>((size_t) KT::R * BK), h),
            mrow(sycl::range<1>(KT::R), h), lrow(sycl::range<1>(KT::R), h), arow(sycl::range<1>(KT::R), h);
        sycl::local_accessor<uint16_t, 1> ks(sycl::range<1>((size_t) BK * KT::KS), h), vs(sycl::range<1>((size_t) BK * HD), h);
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t) s.n_head_kv, blocks * NT), sycl::range<2>(1, NT)),
                       KT{qv, kp, vp, table, attn, pos0, T, (int) s.n_head, (int) s.n_head_kv, (int) s.page_size,
                          1.0f / std::sqrt((float) HD), qs, sc, mrow, lrow, arow, ks, vs});
    });
}

}  // namespace

bool prompt_attn(const float* q, const uint16_t* k_pool, const uint16_t* v_pool, const int32_t* page_table,
                 int64_t pos0, int64_t T, const QsaShapes& s, float* attn, void* stream) {
    if (T <= 0) return true;
    if (s.n_head_kv <= 0 || s.n_head % s.n_head_kv != 0 || s.page_size <= 0) return false;
    const int64_t g = s.n_head / s.n_head_kv;
    sycl::queue& queue = core::Runtime::get().stream(stream);
    static const bool no_mma = [] {   // STRATA_PROMPT_ATTN_MMA=0: the FMA kernel on every GPU
        const char* v = std::getenv("STRATA_PROMPT_ATTN_MMA");
        return v != nullptr && std::strtol(v, nullptr, 10) == 0;
    }();
    bool mma = false;
    if (!no_mma && amd_wmma(queue.get_device())) {
        // local memory: Q, K or V, the scores and probabilities, 44 to 50 KiB (RDNA's work-groups have 64)
        mma = true;
        if (s.head_dim == 128 && g == 10) launch_wmma<128, 10, 8>(queue, q, k_pool, v_pool, page_table, pos0, T, s, attn);
        else if (s.head_dim == 256 && g == 8) launch_wmma<256, 8, 4>(queue, q, k_pool, v_pool, page_table, pos0, T, s, attn);
        else mma = false;
    }
    // local memory: BQ x G rows of FP32 Q and BK cells of K and V, 37 to 42 KiB: within the 48 KiB every GPU gives
    if (!mma) {
        if (s.head_dim == 128 && g == 10) launch<128, 10, 4, 32>(queue, q, k_pool, v_pool, page_table, pos0, T, s, attn);
        else if (s.head_dim == 256 && g == 8) launch<256, 8, 2, 16>(queue, q, k_pool, v_pool, page_table, pos0, T, s, attn);
        else if (s.head_dim == 256 && g == 12) launch<256, 12, 2, 16>(queue, q, k_pool, v_pool, page_table, pos0, T, s, attn);
        else return false;
    }
    if (stream == nullptr) queue.wait_and_throw();
    return true;
}

}  // namespace strata::kernels
