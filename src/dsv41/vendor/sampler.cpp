// SPDX-FileCopyrightText: 2026 Peasant Smith and the Project Maya contributors
// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT

// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// Select candidates cooperatively on the device, then keep the upstream double-precision sampling chain on the
// host. Only the shortlist crosses back; this also supports devices without fp64 and keeps the host draw unchanged.
#include "strata/dsv41/sampler.hpp"
#include "strata/core/readback.hpp"
#include "../ds4_sycl.hpp"
#include <algorithm>
#include <cmath>
#include <exception>
#include <new>
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

constexpr int kMax = 64;
struct Selected {
    int ids[kMax];
    float logits[kMax];
};

// The ordered score and inverse ID form a total order: NaN last, signed zero tied, then the smallest ID.
uint32_t ordered_score(float value) {
    uint32_t b = sycl::bit_cast<uint32_t>(value);
    if ((b & 0x7fffffffu) > 0x7f800000u) return 0;
    if (b == 0x80000000u) b = 0;
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

struct SelectionShape {
    size_t local = 0, bins = 0, capacity = 0, bit_words = 0;
    unsigned shift = 0;
};

SelectionShape selection_shape(const sycl::queue& q, int vocab, bool history) {
    const auto device = q.get_device();
    const size_t limit = std::min({size_t(vocab), device.get_info<sycl::info::device::max_work_group_size>(),
                                  device.get_info<sycl::info::device::max_work_item_sizes<1>>()[0]});
    const size_t memory = device.get_info<sycl::info::device::local_mem_size>();
    SelectionShape shape;
    // Two histogram bins and two candidate words per work-item. The byte-radix fallback needs 256 bins and
    // the API permits 64 candidates even when the device's work-group is smaller than that.
    for (size_t local = 1; local <= limit; local *= 2) {
        const size_t bins = std::max<size_t>(256, 2 * local), cap = std::max<size_t>(kMax, local);
        const size_t bytes = (bins + 2 * cap + 4 + 1) * sizeof(uint32_t);
        if (bytes > memory) break;
        shape = {local, bins, cap, 0, 0};
        if (local > limit / 2) break;
    }
    if (!shape.local) return shape;  // The existing host selection also works with very little local memory.
    unsigned bits = 0;
    for (size_t bins = shape.bins; bins > 1; bins >>= 1) ++bits;
    shape.shift = 32 - bits;
    const size_t bitmap = (size_t(vocab) + 31) / 32;
    const size_t base = (shape.bins + 2 * shape.capacity + 4) * sizeof(uint32_t);
    if (history && bitmap <= (memory - base) / sizeof(uint32_t)) shape.bit_words = bitmap;
    return shape;
}

struct SelectionMemory {
    sycl::queue& q;
    Selected* data = nullptr;
    ~SelectionMemory() {
        // This guard is declared after the host vectors: even a failed transfer must finish using them first.
        try { q.wait(); if (data) sycl::free(data, q); }
        catch (...) { std::terminate(); }
    }
    SelectionMemory(const SelectionMemory&) = delete;
    SelectionMemory& operator=(const SelectionMemory&) = delete;
    explicit SelectionMemory(sycl::queue& queue) : q(queue) {}
};

sycl::event select_candidates(const float* logits, int tokens, int vocab, const int* history, int history_len,
                             const SamplerParams& p, int k, Selected* out, int* greedy_out,
                             SelectionShape shape, sycl::queue& q) {
    const bool greedy = p.greedy || p.temperature <= 0.f;
    return q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<uint32_t, 1> hist(shape.bins, cgh), keys(shape.capacity, cgh), state(4, cgh);
        sycl::local_accessor<int, 1> ids(shape.capacity, cgh);
        sycl::local_accessor<uint32_t, 1> bits(std::max<size_t>(1, shape.bit_words), cgh);
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(size_t(tokens) * shape.local),
                                          sycl::range<1>(shape.local)), [=](sycl::nd_item<1> it) {
            const size_t t = it.get_group_linear_id(), tid = it.get_local_linear_id();
            const auto group = it.get_group();
            if (!greedy)
                for (size_t i = tid; i < kMax; i += shape.local) { out[t].ids[i] = 0; out[t].logits[i] = 0.f; }
            const float* row = logits + t * size_t(vocab);
            const int hlen = history ? sycl::max(0, sycl::min(history_len, p.penalty_last_n)) : 0;
            const int* hrow = hlen ? history + t * size_t(history_len) + (history_len - hlen) : nullptr;
            if (shape.bit_words) {
                for (size_t b = tid; b < shape.bit_words; b += shape.local) bits[b] = 0;
                sycl::group_barrier(group);
                for (size_t i = tid; i < size_t(hlen); i += shape.local) {
                    const int v = hrow[i];
                    if (v >= 0 && v < vocab)
                        sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                            sycl::access::address_space::local_space>(bits[size_t(v) >> 5]).fetch_or(1u << (v & 31));
                }
                sycl::group_barrier(group);
            }
            auto score = [&](int v) {
                int count = 0;
                if (hlen && (!shape.bit_words || (bits[size_t(v) >> 5] & (1u << (v & 31)))))
                    for (int i = 0; i < hlen; ++i) count += hrow[i] == v;
                return host_penalised(p.logit_bias ? row[v] + p.logit_bias[v] : row[v], count, p);
            };
            auto add = [&](uint32_t& value) {
                return sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                    sycl::access::address_space::local_space>(value).fetch_add(1u);
            };
            if (greedy) {
                uint64_t best = 0;
                for (size_t v = tid; v < size_t(vocab); v += shape.local) {
                    const uint64_t key = (uint64_t(ordered_score(score(int(v)))) << 32) | ~uint32_t(v);
                    best = sycl::max(best, key);
                }
                best = sycl::reduce_over_group(group, best, sycl::maximum<uint64_t>());
                if (tid == 0) greedy_out[t] = int(~uint32_t(best));
                return;
            }
            for (size_t b = tid; b < shape.bins; b += shape.local) hist[b] = 0;
            if (tid == 0) state[0] = 0;
            sycl::group_barrier(group);
            for (size_t v = tid; v < size_t(vocab); v += shape.local)
                add(hist[ordered_score(score(int(v))) >> shape.shift]);
            sycl::group_barrier(group);
            if (tid == 0) {
                uint32_t count = 0;
                size_t bin = shape.bins;
                do { --bin; count += hist[bin]; } while (count < uint32_t(k) && bin);
                state[1] = uint32_t(bin);
                state[2] = count;
            }
            sycl::group_barrier(group);
            if (state[2] <= shape.capacity) {
                const uint32_t bin = state[1];
                for (size_t v = tid; v < size_t(vocab); v += shape.local) {
                    const uint32_t key = ordered_score(score(int(v)));
                    if ((key >> shape.shift) >= bin) {
                        const uint32_t at = add(state[0]);
                        keys[at] = key; ids[at] = int(v);
                    }
                }
                sycl::group_barrier(group);
            } else {
                // Find the k-th score one byte at a time, then retain the lowest IDs among its ties.
                if (tid == 0) { state[1] = 0; state[2] = uint32_t(k); }
                sycl::group_barrier(group);
                uint32_t mask = 0;
                for (int pass = 0; pass < 4; ++pass) {
                    const unsigned shift = unsigned(24 - 8 * pass);
                    for (size_t b = tid; b < 256; b += shape.local) hist[b] = 0;
                    sycl::group_barrier(group);
                    const uint32_t prefix = state[1];
                    for (size_t v = tid; v < size_t(vocab); v += shape.local) {
                        const uint32_t key = ordered_score(score(int(v)));
                        if ((key & mask) == prefix) add(hist[(key >> shift) & 255u]);
                    }
                    sycl::group_barrier(group);
                    if (tid == 0) {
                        uint32_t above = 0;
                        int digit = 255;
                        while (digit && above + hist[size_t(digit)] < state[2]) above += hist[size_t(digit--)];
                        state[2] -= above;
                        state[1] = prefix | (uint32_t(digit) << shift);
                    }
                    mask |= 255u << shift;
                    sycl::group_barrier(group);
                }
                const uint32_t threshold = state[1], need = state[2];
                for (size_t v = tid; v < size_t(vocab); v += shape.local) {
                    const uint32_t key = ordered_score(score(int(v)));
                    if (key > threshold) {
                        const uint32_t at = add(state[0]);
                        keys[at] = key; ids[at] = int(v);
                    }
                }
                sycl::group_barrier(group);
                const uint32_t above = state[0];
                const size_t chunk = (size_t(vocab) + shape.local - 1) / shape.local;
                const size_t lo = tid * chunk, hi = sycl::min(size_t(vocab), lo + chunk);
                uint32_t mine = 0;
                for (size_t v = lo; v < hi; ++v) mine += ordered_score(score(int(v))) == threshold;
                uint32_t rank = sycl::exclusive_scan_over_group(group, mine, sycl::plus<uint32_t>());
                for (size_t v = lo; v < hi && rank < need; ++v) {
                    if (ordered_score(score(int(v))) == threshold) {
                        const size_t at = size_t(above) + rank++;
                        keys[at] = threshold; ids[at] = int(v);
                    }
                }
                sycl::group_barrier(group);
                if (tid == 0) state[0] = uint32_t(k);
                sycl::group_barrier(group);
            }
            const size_t count = state[0];
            for (size_t i = tid; i < count; i += shape.local) {
                const uint32_t key = keys[i];
                const int id = ids[i];
                int rank = 0;
                for (size_t j = 0; j < count && rank < k; ++j)
                    rank += keys[j] > key || (keys[j] == key && ids[j] < id);
                if (rank < k) { out[t].ids[rank] = id; out[t].logits[rank] = score(id); }
            }
        });
    });
}

// MODE 0 draw, 1 draft (kept list into ql), 2 verify (draft[t] against ql[t])
template <int MODE>
void host_sample(const float* dlogits, int n_tokens, int n_vocab, const int* dhistory, int history_len, const SamplerParams& p,
                 int* dout, SpecList* dql, const int* ddraft, sycl::queue& stream) {
    strata::ds4::sycl_detail::require_in_order(stream);
    const bool use_history = dhistory && history_len > 0 && p.penalty_last_n > 0;
    const auto shape = selection_shape(stream, n_vocab, use_history);
    const bool device_select = shape.local != 0;
    if (device_select && (p.greedy || p.temperature <= 0.f)) {
        select_candidates(dlogits, n_tokens, n_vocab, use_history ? dhistory : nullptr, history_len, p, 1,
                          nullptr, dout, shape, stream).wait_and_throw();
        return;
    }
    const int KMAX = kMax;
    int k = (p.top_k > 0 && p.top_k < KMAX) ? p.top_k : KMAX;
    if (k > n_vocab) k = n_vocab;
    std::vector<float> hlogits(device_select ? 0 : (size_t) n_tokens * n_vocab);
    std::vector<float> hbias(!device_select && p.logit_bias ? (size_t) n_vocab : 0);
    std::vector<int> hhistory(!device_select && use_history ? (size_t) n_tokens * history_len : 0);
    std::vector<int> hout((size_t) n_tokens), hdraft;
    std::vector<SpecList> hq(dql ? (size_t) n_tokens : 0);
    std::vector<Selected> selected(device_select ? (size_t) n_tokens : 0);
    if constexpr (MODE == 2) hdraft.resize((size_t) n_tokens);
    SelectionMemory memory(stream);
    if (device_select) {
        memory.data = sycl::malloc_device<Selected>(selected.size(), stream);
        if (!memory.data) throw std::bad_alloc();
        select_candidates(dlogits, n_tokens, n_vocab, use_history ? dhistory : nullptr, history_len, p, k,
                          memory.data, nullptr, shape, stream);
    }
    const core::ReadbackSpan copies[] = {
        {selected.data(), memory.data, selected.size() * sizeof(Selected)},
        {hlogits.data(), dlogits, hlogits.size() * sizeof(float)},
        {hhistory.data(), dhistory, hhistory.size() * sizeof(int)},
        {hbias.data(), p.logit_bias, hbias.size() * sizeof(float)},
        {hq.data(), dql, MODE == 2 ? hq.size() * sizeof(SpecList) : 0},
        {hdraft.data(), ddraft, hdraft.size() * sizeof(int)},
    };
    core::readback(stream, copies);
    const float* logits = hlogits.data();
    const int* history = use_history ? hhistory.data() : nullptr;
    int* out = hout.data();
    SpecList* ql = hq.data();
    const int* draft = hdraft.data();
    const float* bias = p.logit_bias ? hbias.data() : nullptr;
    const float inv_t = p.temperature > 0.0f ? 1.0f / p.temperature : 0.0f;
    std::vector<float> sc(device_select ? 0 : (size_t) n_vocab);
    std::vector<int> idx(device_select ? 0 : (size_t) n_vocab);
    for (int t = 0; t < n_tokens; ++t) {
        int sel_ids[64];
        float sel_logit[64];
        if (device_select) {
            if (p.greedy || p.temperature <= 0.0f) { out[t] = selected[(size_t) t].ids[0]; continue; }
            std::copy_n(selected[(size_t) t].ids, k, sel_ids);
            std::copy_n(selected[(size_t) t].logits, k, sel_logit);
        } else {
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
        for (int i = 0; i < k; ++i) { sel_ids[i] = idx[(size_t) i]; sel_logit[i] = sc[(size_t) idx[(size_t) i]]; }
        }
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
