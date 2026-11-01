// SPDX-FileCopyrightText: 2026 Peasant Smith and the Project Maya contributors
// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/main/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
#pragma once

#include "ds4_finish.hpp"
#include "ds4_sycl.hpp"
#include <cfloat>

namespace strata::ds4::compress_detail {

// Upstream ds4_kernels.cu compress_kernel: one group processes the window in
// token order. Use finish's 256-item order, rather than upstream compress's 512,
// to preserve the existing SYCL result. This is an arithmetic-order constraint,
// not a measured device optimum; the same reported capabilities as finish gate it.
template <int SG>
void compress_kernel(const float* kv, const float* score, float* kv_state, float* score_state,
                     int ratio, int p0, int T, int d, const float* w, float eps, float* latent, sycl::queue& q) {
    constexpr int threads = finish_detail::kThreads;
    sycl_detail::submit(q, [&](sycl::handler& handler) {
        sycl::local_accessor<float, 1> red(sycl::range<1>(threads / 32 + 1), handler);
        sycl::local_accessor<uint32_t, 1> exchange(sycl::range<1>(SG == 16 ? threads : 1), handler);
        auto body = [=](sycl::nd_item<1> item) {
            const int tid = int(item.get_local_linear_id());
            const int count = (d + threads - 1) / threads;
            int g = 0;
            for (int t = 0; t < T; ++t) {
                const int slot = (p0 + t) % ratio;
                for (int e = tid; e < d; e += threads) {
                    kv_state[size_t(slot) * d + e] = kv[size_t(t) * d + e];
                    score_state[size_t(slot) * d + e] = score[size_t(t) * d + e];
                }
                item.barrier(sycl::access::fence_space::global_and_local);
                if ((p0 + t + 1) % ratio) continue;

                float r[finish_detail::kValues], ss = 0.f;
                for (int j = 0; j < count; ++j) {
                    const int e = tid + j * threads;
                    float m = -FLT_MAX, den = 0.f, v = 0.f;
                    if (e < d) {
                        for (int s = 0; s < ratio; ++s) m = sycl::fmax(m, score_state[size_t(s) * d + e]);
                        for (int s = 0; s < ratio; ++s) den += sycl::exp(score_state[size_t(s) * d + e] - m);
                        for (int s = 0; s < ratio; ++s) v += kv_state[size_t(s) * d + e] *
                            (sycl::exp(score_state[size_t(s) * d + e] - m) / den);
                    }
                    r[j] = v;
                }
                // Keep finish_kernel's register iteration and two-level XOR sum.
                for (int j = 0; j < count; ++j) ss += r[j] * r[j];
                if (w) {
                    const float total = finish_detail::block_sum<SG>(item, ss, red, exchange);
                    const float inv = sycl::rsqrt(total / float(d) + eps);
                    for (int j = 0; j < count; ++j) {
                        const int e = tid + j * threads;
                        if (e < d) r[j] = r[j] * inv * w[e];
                    }
                }
                for (int j = 0; j < count; ++j) {
                    const int e = tid + j * threads;
                    if (e < d) latent[size_t(g) * d + e] = r[j];
                }
                ++g;
                item.barrier(sycl::access::fence_space::global_and_local);
            }
        };
        handler.parallel_for(sycl::nd_range<1>(threads, threads),
                             finish_detail::BoundBody<SG, decltype(body)>{body});
    });
}

inline bool compress(const float* kv, const float* score, float* kv_state, float* score_state,
                     int ratio, int p0, int T, int d, const float* w, float eps, float* latent, sycl::queue& q) {
    if (T <= 0) return true;
    if (ratio <= 0 || d <= 0 || d > finish_detail::kValues * finish_detail::kThreads) return false;
    const int sg = finish_detail::subgroup(finish_detail::caps(q), true);
    if (!sg) return false;
    sycl_detail::require_in_order(q);
    if (sg == 32) compress_kernel<32>(kv, score, kv_state, score_state, ratio, p0, T, d, w, eps, latent, q);
    else compress_kernel<16>(kv, score, kv_state, score_state, ratio, p0, T, d, w, eps, latent, q);
    return true;
}

}  // namespace strata::ds4::compress_detail
