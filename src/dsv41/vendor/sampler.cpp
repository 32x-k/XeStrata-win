// SPDX-FileCopyrightText: 2026 Peasant Smith and the Project Maya contributors
// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT

// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// Keep the upstream double-precision sampling chain on the host. SYCL transfers are ordered on the supplied queue;
// this path also supports devices without fp64, and waits before its temporary host buffers leave scope.
#include "strata/dsv41/sampler.hpp"
#include "../ds4_sycl.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
namespace strata::dsv41k {
namespace {
inline uint32_t host_umulhi(uint32_t a, uint32_t b) { return (uint32_t) (((uint64_t) a * (uint64_t) b) >> 32); }
inline float host_philox_uniform(uint64_t seed, uint64_t counter) {
    uint32_t c0 = (uint32_t) counter, c1 = (uint32_t) (counter >> 32), c2 = (uint32_t) seed, c3 = (uint32_t) (seed >> 32);
    for (int i = 0; i < 10; ++i) {
        const uint32_t hi0 = host_umulhi(0x9E3779B9u, c0), hi1 = host_umulhi(0xBB67AE85u, c2);
        const uint32_t lo0 = 0x9E3779B9u * c0, lo1 = 0xBB67AE85u * c2;
        const uint32_t n0 = hi1 ^ c1 ^ (uint32_t) i, n1 = lo1, n2 = hi0 ^ c3 ^ 0u, n3 = lo0;
        c0 = n0; c1 = n1; c2 = n2; c3 = n3;
    }
    return (float) (c0 >> 8) * (1.0f / 16777216.0f);
}
inline float host_penalised(float logit, int count, const SamplerParams& p) {
    if (count <= 0) return logit;
    if (logit <= 0.0f) logit *= p.penalty_repeat;
    else logit /= p.penalty_repeat;
    logit -= (float) count * p.penalty_freq + (count > 0 ? 1.0f : 0.0f) * p.penalty_present;
    return logit;
}

// MODE 0 draw, 1 draft (kept list into ql), 2 verify (draft[t] against ql[t])
template <int MODE>
void host_sample(const float* dlogits, int n_tokens, int n_vocab, const int* dhistory, int history_len, const SamplerParams& p,
                 int* dout, SpecList* dql, const int* ddraft, sycl::queue& stream) {
    strata::ds4::sycl_detail::require_in_order(stream);
    const bool use_history = dhistory && history_len > 0 && p.penalty_last_n > 0;
    std::vector<float> hlogits((size_t) n_tokens * n_vocab), hbias(p.logit_bias ? (size_t) n_vocab : 0);
    std::vector<int> hhistory(use_history ? (size_t) n_tokens * history_len : 0), hout((size_t) n_tokens), hdraft;
    std::vector<SpecList> hq(dql ? (size_t) n_tokens : 0);
    stream.memcpy(hlogits.data(), dlogits, hlogits.size() * sizeof(float)).wait_and_throw();
    if (use_history) stream.memcpy(hhistory.data(), dhistory, hhistory.size() * sizeof(int)).wait_and_throw();
    if (p.logit_bias) stream.memcpy(hbias.data(), p.logit_bias, hbias.size() * sizeof(float)).wait_and_throw();
    if constexpr (MODE == 2) {
        hdraft.resize((size_t) n_tokens);
        stream.memcpy(hq.data(), dql, hq.size() * sizeof(SpecList)).wait_and_throw();
        stream.memcpy(hdraft.data(), ddraft, hdraft.size() * sizeof(int)).wait_and_throw();
    }
    const float* logits = hlogits.data();
    const int* history = use_history ? hhistory.data() : nullptr;
    int* out = hout.data();
    SpecList* ql = hq.data();
    const int* draft = hdraft.data();
    const float* bias = p.logit_bias ? hbias.data() : nullptr;
    const float inv_t = p.temperature > 0.0f ? 1.0f / p.temperature : 0.0f;
    const int KMAX = 64;
    int k = (p.top_k > 0 && p.top_k < KMAX) ? p.top_k : KMAX;
    if (k > n_vocab) k = n_vocab;
    std::vector<float> sc((size_t) n_vocab);
    std::vector<int> idx((size_t) n_vocab);
    for (int t = 0; t < n_tokens; ++t) {
        const float* l = logits + (size_t) t * n_vocab;
        const int* hrow = history ? history + (size_t) t * history_len : nullptr;
        int hlen = 0;
        if (hrow) {
            hlen = p.penalty_last_n < history_len ? p.penalty_last_n : history_len;
            if (hlen < 0) hlen = 0;
            hrow += history_len - hlen;
        }
        for (int v = 0; v < n_vocab; ++v) {
            int cnt = 0;
            if (hlen > 0)
                for (int i = 0; i < hlen; ++i) cnt += hrow[i] == v;
            sc[(size_t) v] = host_penalised(bias ? l[v] + bias[v] : l[v], cnt, p);
        }
        // the ordered key: NaN below everything, -0 as +0 (the kernel's ordered())
        auto less_than = [&](int a, int b) {   // a ranks before b
            const float x = sc[(size_t) a], y = sc[(size_t) b];
            const bool nx = x != x, ny = y != y;
            if (nx != ny) return ny;
            if (nx && ny) return a < b;
            if (x != y) return x > y;
            return a < b;
        };
        for (int v = 0; v < n_vocab; ++v) idx[(size_t) v] = v;
        std::nth_element(idx.begin(), idx.begin() + (k - 1), idx.end(), less_than);
        std::sort(idx.begin(), idx.begin() + k, less_than);
        if (p.greedy || p.temperature <= 0.0f) { out[t] = idx[0]; continue; }
        int sel_ids[64];
        float sel_logit[64];
        for (int i = 0; i < k; ++i) { sel_ids[i] = idx[(size_t) i]; sel_logit[i] = sc[(size_t) idx[(size_t) i]]; }
        int n_keep = k;
        float mx = sel_logit[0];
        for (int i = 1; i < k; ++i) mx = std::max(mx, sel_logit[i]);
        if (p.top_p < 1.0f) {
            double sum = 0.0;
            for (int i = 0; i < k; ++i) sum += std::exp((double) sel_logit[i] - (double) mx);
            double cum = 0.0;
            int cut = k;
            for (int i = 0; i < k; ++i) {
                cum += std::exp((double) sel_logit[i] - (double) mx) / sum;
                if (cum >= (double) p.top_p) { cut = i + 1; break; }
            }
            if (cut < p.min_keep) cut = p.min_keep < k ? p.min_keep : k;
            n_keep = cut;
        }
        if (p.min_p > 0.0f) {
            const float thresh = sel_logit[0] + std::log(p.min_p);
            for (int i = 0; i < n_keep; ++i)
                if (sel_logit[i] < thresh) { n_keep = i; break; }
        }
        n_keep = std::max(1, n_keep);
        auto scaled = [&](int i) { return sel_logit[i] * inv_t; };
        float smx = scaled(0);
        for (int i = 1; i < n_keep; ++i) smx = std::max(smx, scaled(i));
        double sum = 0.0;
        for (int i = 0; i < n_keep; ++i) sum += std::exp((double) scaled(i) - (double) smx);
        double w[64];
        for (int i = 0; i < n_keep; ++i) w[i] = std::exp((double) scaled(i) - (double) smx) / sum;
        auto draw = [&](float u) -> int {
            double cum = 0.0;
            int pick = sel_ids[n_keep - 1];
            for (int i = 0; i < n_keep; ++i) {
                cum += w[i];
                if ((double) u < cum) { pick = sel_ids[i]; break; }
            }
            return pick;
        };
        if (MODE == 0) {
            out[t] = draw(host_philox_uniform(p.seed, p.counter + (uint64_t) t));
        } else if (MODE == 1) {
            SpecList* q = ql + t;
            q->n = n_keep;
            for (int i = 0; i < n_keep; ++i) { q->ids[i] = sel_ids[i]; q->p[i] = w[i]; }
            out[t] = draw(host_philox_uniform(p.seed ^ 0x5bd1e9955bd1e995ull, p.counter + (uint64_t) t));
        } else {
            const SpecList* q = ql + t;
            const int d = draft[t];
            const int qn = q->n < KMAX ? q->n : KMAX;
            auto qof = [&](int id) -> double {
                for (int j = 0; j < qn; ++j)
                    if (q->ids[j] == id) return q->p[j];
                return 0.0;
            };
            double pd = 0.0;
            for (int i = 0; i < n_keep; ++i)
                if (sel_ids[i] == d) { pd = w[i]; break; }
            const double qd = qof(d);
            int pick = -1;
            if (qd <= 0.0) {
                pick = draw(host_philox_uniform(p.seed, p.counter + (uint64_t) t));
            } else if ((double) host_philox_uniform(p.seed ^ 0x9e3779b97f4a7c15ull, p.counter + (uint64_t) t) * qd < pd) {
                pick = d;
            } else {
                double r[64], z = 0.0;
                for (int i = 0; i < n_keep; ++i) {
                    const double x = w[i] - qof(sel_ids[i]);
                    r[i] = x > 0.0 ? x : 0.0;
                    z += r[i];
                }
                const float u = host_philox_uniform(p.seed ^ 0xc2b2ae3d27d4eb4full, p.counter + (uint64_t) t);
                if (z > 0.0) {
                    double cum = 0.0;
                    for (int i = 0; i < n_keep; ++i) {
                        if (r[i] <= 0.0) continue;
                        pick = sel_ids[i];
                        cum += r[i] / z;
                        if ((double) u < cum) break;
                    }
                } else {
                    pick = draw(u);
                }
            }
            out[t] = pick;
        }
    }
    stream.memcpy(dout, hout.data(), hout.size() * sizeof(int)).wait_and_throw();
    if constexpr (MODE == 1) stream.memcpy(dql, hq.data(), hq.size() * sizeof(SpecList)).wait_and_throw();
}

void spec_check(const SamplerParams& p, int n_vocab) {
    if (p.greedy || p.temperature <= 0.0f || p.penalty_last_n > 0 || n_vocab <= 0)
        throw std::invalid_argument("dsv41: speculative sampling requires sampling without penalties");
}
}  // namespace
void sample_tokens(const float* logits, int n_tokens, int n_vocab, const int* history, int history_len,
                   const SamplerParams& p, int* out, sycl::queue& stream) {
    if (n_tokens <= 0 || n_vocab <= 0) return;
    if (p.penalty_last_n > 0 && (!history || history_len <= 0))
        throw std::invalid_argument("dsv41: sampling penalties require history");
    host_sample<0>(logits, n_tokens, n_vocab, history, history_len, p, out, nullptr, nullptr, stream);
}
void sample_draft(const float* logits, int n_tokens, int n_vocab, const SamplerParams& p, SpecList* q, int* out,
                  sycl::queue& stream) {
    if (n_tokens <= 0) return;
    spec_check(p, n_vocab);
    host_sample<1>(logits, n_tokens, n_vocab, nullptr, 0, p, out, q, nullptr, stream);
}
void sample_verify(const float* logits, int n_tokens, int n_vocab, const SamplerParams& p, const SpecList* q,
                   const int* draft, int* out, sycl::queue& stream) {
    if (n_tokens <= 0) return;
    spec_check(p, n_vocab);
    host_sample<2>(logits, n_tokens, n_vocab, nullptr, 0, p, out, const_cast<SpecList*>(q), draft, stream);
}
}  // namespace strata::dsv41k
