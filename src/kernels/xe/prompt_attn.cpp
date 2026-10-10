// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/prompt_attn.cpp - see include/strata/kernels/prompt_attn.hpp.
#include "strata/kernels/prompt_attn.hpp"
#include "strata/core/runtime.hpp"

#include <sycl/sycl.hpp>

#include <cmath>
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
    // local memory: BQ x G rows of FP32 Q and BK cells of K and V, 37 to 42 KiB: within the 48 KiB every GPU gives
    if (s.head_dim == 128 && g == 10) launch<128, 10, 4, 32>(queue, q, k_pool, v_pool, page_table, pos0, T, s, attn);
    else if (s.head_dim == 256 && g == 8) launch<256, 8, 2, 16>(queue, q, k_pool, v_pool, page_table, pos0, T, s, attn);
    else if (s.head_dim == 256 && g == 12) launch<256, 12, 2, 16>(queue, q, k_pool, v_pool, page_table, pos0, T, s, attn);
    else return false;
    if (stream == nullptr) queue.wait_and_throw();
    return true;
}

}  // namespace strata::kernels
