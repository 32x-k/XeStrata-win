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
inline float h2f(uint16_t bits) { return (float) sycl::bit_cast<sycl::half>(bits); }

// BQ queries of one KV head: R = BQ x G rows (query, head), Q in local memory as FP32 (scaled), the step's BK cells of
// K and V as FP16, the step's scores (then probabilities) per row.  Work-item t owns column d = t % HD of rows t / HD,
// + NT / HD, ... of the output in registers.  The rows of Q and K are a word longer than HD in local memory: the
// scores' neighbouring work-items read neighbouring rows, which would all fall in one bank at a stride of HD.
template <int HD, int G, int BQ, int BK>
struct Kernel {
    static constexpr int R = BQ * G, RS = NT / HD, OPT = R / RS, QS = HD + 1, KS = HD + 2;
    static_assert(NT % HD == 0 && R % RS == 0, "the output columns and rows split evenly");
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
            M[t] = -std::numeric_limits<float>::infinity();
            L[t] = 0.0f;
        }
        float o[OPT];
        for (int k = 0; k < OPT; ++k) o[k] = 0.0f;
        const int64_t n_cells = pos0 + sycl::min<int64_t>(qbase + BQ, T);   // through the block's last query
        for (int64_t c0 = 0; c0 < n_cells; c0 += BK) {
            sycl::group_barrier(it.get_group());   // the last step's K, V and probabilities are read
            for (int i = t; i < BK * HD; i += NT) {
                const int j = i / HD, dd = i - j * HD;
                const int64_t c = c0 + j;
                uint16_t kv = 0, vv = 0;
                if (c < n_cells) {
                    const int64_t page = table[c / page_size];
                    const int64_t row = (page * n_kv + h) * page_size + c % page_size;
                    kv = kp[row * HD + dd];
                    vv = vp[row * HD + dd];
                }
                K[j * KS + dd] = kv;
                V[i] = vv;
            }
            sycl::group_barrier(it.get_group());
            for (int i = t; i < R * BK; i += NT) {   // scores, masked past each row's own position
                const int r = i / BK, j = i - r * BK;
                const int64_t c = c0 + j, qpos = pos0 + qbase + r / G;
                float s = -std::numeric_limits<float>::infinity();
                if (c <= qpos && c < n_cells) {
                    s = 0.0f;
                    for (int dd = 0; dd < HD; ++dd) s = sycl::fma(Q[r * QS + dd], h2f(K[j * KS + dd]), s);
                }
                S[i] = s;
            }
            sycl::group_barrier(it.get_group());
            if (t < R) {   // the row's online softmax: its new maximum, the old sum's factor, the step's probabilities
                float mx = M[t];
                for (int j = 0; j < BK; ++j) mx = sycl::fmax(mx, S[t * BK + j]);
                const float a = M[t] == -std::numeric_limits<float>::infinity() ? 0.0f : sycl::exp(M[t] - mx);
                float sum = 0.0f;
                for (int j = 0; j < BK; ++j) {
                    const float p = S[t * BK + j] == -std::numeric_limits<float>::infinity() ? 0.0f
                                                                                              : sycl::exp(S[t * BK + j] - mx);
                    S[t * BK + j] = p;
                    sum += p;
                }
                L[t] = L[t] * a + sum;
                M[t] = mx;
                A[t] = a;
            }
            sycl::group_barrier(it.get_group());
            for (int k = 0; k < OPT; ++k) {
                const int r = r0 + k * RS;
                float acc = o[k] * A[r];
                for (int j = 0; j < BK; ++j) acc = sycl::fma(S[r * BK + j], h2f(V[j * HD + d]), acc);
                o[k] = acc;
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
