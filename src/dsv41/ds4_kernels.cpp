// SPDX-FileCopyrightText: 2026 Peasant Smith and the Project Maya contributors
// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/main/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
#include "ds4_kernels.cuh"
#include "ds4_numerics.cuh"
#include "ds4_q6k.cuh"
#include "ds4_mailbox.hpp"
#include "ds4_finish.hpp"
#include "ds4_compress.hpp"
#include "ds4_hc.hpp"
#include "strata/core/per_device.hpp"
#include "strata/core/readback.hpp"
#if defined(SYCL_EXT_ONEAPI_SUB_GROUP_MASK) && SYCL_EXT_ONEAPI_SUB_GROUP_MASK >= 2
#include <sycl/ext/oneapi/sub_group_mask.hpp>
#endif
#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

namespace strata::ds4 {
namespace {
using sycl_detail::parallel;
using sycl_detail::bf16_value;
using sycl_detail::f16_value;

struct MailboxRegistry {
    std::mutex mutex;
    std::unordered_map<const sycl::queue*, std::weak_ptr<MailboxControl>> queues;
};
MailboxRegistry& mailbox_registry() {
    static MailboxRegistry registry;
    return registry;
}
std::shared_ptr<MailboxControl> mailbox_for(const sycl::queue& queue) {
    auto& registry = mailbox_registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    const auto found = registry.queues.find(&queue);
    return found == registry.queues.end() ? nullptr : found->second.lock();
}

float pow2_ceil(float x) {
    const uint32_t b = sycl::bit_cast<uint32_t>(x);
    const int e = int((b >> 23) & 255u) + ((b & 0x7fffffu) != 0u);
    return sycl::bit_cast<float>(uint32_t(e) << 23);
}
float e2m1_round(float v) {
    const float a = sycl::fabs(v);
    const float r = a <= .25f ? 0.f : a < .75f ? .5f : a <= 1.25f ? 1.f : a < 1.75f ? 1.5f :
                    a <= 2.5f ? 2.f : a < 3.5f ? 3.f : a <= 5.f ? 4.f : 6.f;
    return sycl::copysign(r, v);
}
uint8_t e2m1_code(float q) {
    const float a = sycl::fabs(q);
    const uint32_t m = a <= 1.5f ? uint32_t(a * 2.f) : a <= 2.f ? 4u : a <= 3.f ? 5u : a <= 4.f ? 6u : 7u;
    return uint8_t(m | ((sycl::bit_cast<uint32_t>(q) >> 31) << 3));
}
uint8_t e4m3_code(float v, int k = 0) {
    const uint32_t b = sycl::bit_cast<uint32_t>(v), sg = (b >> 24) & 128u, a = b & 0x7fffffffu;
    if (!a) return uint8_t(sg);
    const int e = int(a >> 23) - 127 - k;
    if (e >= -6) return uint8_t(sg | (uint32_t(e + 7) << 3) | ((a >> 20) & 7u));
    return uint8_t(sg | (((a & 0x7fffffu) | 0x800000u) >> (14 - e)));
}
float e2m1_value(uint8_t q) {
    constexpr float values[] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    return q & 8 ? -values[q & 7] : values[q & 7];
}
float kv_value(const void* data, int fmt, int d, int e) {
    const auto* b = static_cast<const uint8_t*>(data);
    if (fmt == kKvBf16) return bf16_value(static_cast<const uint16_t*>(data)[e]);
    if (fmt == kKvFp8) return e4m3_value(b[e]) * sycl::bit_cast<float>(uint32_t(b[d + e / 32]) << 23);
    const float q = e2m1_value(uint8_t(b[e / 2] >> (4 * (e & 1))));
    const float scale = fmt == kKvFp4 ? e4m3_value(b[d / 2 + e / 16]) :
                        sycl::bit_cast<float>(uint32_t(b[d / 2 + e / 32]) << 23);
    return q * scale;
}
void rows_at(const Rows& r, int t, int& n, int& k) {
    if (!r.gen) { n = r.n[t]; k = r.k[t]; return; }
    const int nn = int((r.p0 + t + 1) / r.R), cb = sycl::min(r.ccap, (nn + r.blk - 1) / r.blk);
    switch (r.gen) {
    case 1: n = nn; k = sycl::min(r.kcap, nn); break;
    case 2: n = nn; k = cb; break;
    case 3: n = cb * r.blk; k = sycl::min(r.kcap, nn); break;
    default: n = (nn + r.blk - 1) / r.blk; k = r.kcap; break;
    }
}
struct GemvCaps {
    size_t max_group = 0;
    std::vector<size_t> sub_groups;
};
size_t gemv_group_size(const sycl::queue& q, int k) {
    const auto device = q.get_device();
    static core::PerDevice<GemvCaps> cache;
    const auto& caps = cache.get(device, [&] {
        return GemvCaps{std::min(device.get_info<sycl::info::device::max_work_group_size>(),
                                device.get_info<sycl::info::device::max_work_item_sizes<1>>()[0]),
                        device.get_info<sycl::info::device::sub_group_sizes>()};
    });
    const size_t limit = std::min(size_t(std::max(k, 1)), caps.max_group);
    size_t local = 1;
    for (const auto width : caps.sub_groups)
        if (width <= limit) local = std::max(local, width);
    return local;
}
float gemv_weight(float v) { return v; }
float gemv_weight(uint16_t v) { return f16_value(v); }
template <class Weight>
void gemv_rows(const Weight* W, const float* x, size_t ldx, int n, int k, int T, float alpha,
               float* y, size_t ldy, sycl::queue& s) {
    sycl_detail::require_in_order(s);
    if (T <= 0 || n <= 0) return;
    // One output uses a reported sub-group's extent, rather than the device's
    // entire work-group. A group reduction also works if the compiler chooses
    // a narrower sub-group, and the shape is independent of the token count.
    const size_t local = gemv_group_size(s, k), columns = size_t(std::max(k, 0));
    sycl_detail::launch(s, sycl::nd_range<1>(sycl::range<1>(size_t(T) * n * local), sycl::range<1>(local)),
                   [=](sycl::nd_item<1> item) {
        const size_t i = item.get_group_linear_id(), t = i / n, r = i % n;
        float sum = 0.f;
        for (size_t j = item.get_local_linear_id(); j < columns; j += local)
            sum += gemv_weight(W[r * columns + j]) * x[t * ldx + j];
        sum = sycl::reduce_over_group(item.get_group(), sum, sycl::plus<float>());
        if (item.get_local_linear_id() == 0) y[t * ldy + r] = alpha * sum;
    });
}
float route_score(float l) { return sycl::sqrt(l > 20.f ? l : sycl::log1p(sycl::exp(l))); }
void route_row(const float* logits, const float* bias, int n, int k, float scale, int* ids, float* weights) {
    float sum = 0.f;
    for (int r = 0; r < k; ++r) {
        float best = -FLT_MAX;
        int ix = -1;
        for (int i = 0; i < n; ++i) {
            bool used = false;
            for (int j = 0; j < r; ++j) used |= ids[j] == i;
            const float v = route_score(logits[i]) + bias[i];
            if (!used && (v > best || (v == best && (ix < 0 || i < ix)))) { best = v; ix = i; }
        }
        ids[r] = ix;
        if (weights) {
            weights[r] = ix < 0 ? 0.f : route_score(logits[ix]);
            sum += weights[r];
        }
    }
    if (weights) for (int r = 0; r < k; ++r) weights[r] = weights[r] / (sum + 1e-20f) * scale;
}
struct RouteCaps { size_t max_group, local_bytes; };
template <class F>
bool route_groups(sycl::queue& s, int T, int n, F f) {
    sycl_detail::require_in_order(s);
    if (T <= 0) return true;
    if (n <= 0) return false;
    const auto device = s.get_device();
    static core::PerDevice<RouteCaps> cache;
    const auto& caps = cache.get(device, [&] {
        return RouteCaps{std::min(device.get_info<sycl::info::device::max_work_group_size>(),
                                 device.get_info<sycl::info::device::max_work_item_sizes<1>>()[0]),
                         size_t(device.get_info<sycl::info::device::local_mem_size>())};
    });
    const size_t bytes = size_t(n) * 2 * sizeof(float);
    if (bytes >= caps.local_bytes) return false;
    // Cache both scores and reserve a score/ID pair per item for the reductions.
    // A small device can use fewer items, or the serial path if the row will not fit.
    const size_t local = std::min({size_t(n), caps.max_group,
                                   (caps.local_bytes - bytes) / (sizeof(float) + sizeof(int))});
    if (!local) return false;
    sycl_detail::submit(s, [&](sycl::handler& h) {
        sycl::local_accessor<float, 1> scores(size_t(n), h), biased(size_t(n), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(size_t(T) * local), sycl::range<1>(local)),
                       [=](sycl::nd_item<1> item) { f(item, scores, biased); });
    });
    return true;
}
void route_group(sycl::nd_item<1> item, sycl::local_accessor<float, 1> scores,
                 sycl::local_accessor<float, 1> biased, const float* logits, const float* bias,
                 int n, int k, float scale, int* ids, float* weights) {
    const size_t lane = item.get_local_linear_id(), local = item.get_local_range(0);
    const auto group = item.get_group();
    for (size_t i = lane; i < size_t(n); i += local) {
        scores[i] = route_score(logits[i]);
        biased[i] = scores[i] + bias[i];
    }
    sycl::group_barrier(group);
    float sum = 0.f;
    for (int r = 0; r < k; ++r) {
        float best = -FLT_MAX;
        int ix = std::numeric_limits<int>::max();
        for (size_t i = lane; i < size_t(n); i += local) {
            const float v = biased[i];
            if (v > best || (v == best && int(i) < ix)) { best = v; ix = int(i); }
        }
        const float top = sycl::reduce_over_group(group, best, sycl::maximum<float>());
        ix = sycl::reduce_over_group(group, best == top ? ix : std::numeric_limits<int>::max(), sycl::minimum<int>());
        if (lane == 0) {
            const bool valid = ix < n;
            ids[r] = valid ? ix : -1;
            if (weights) { weights[r] = valid ? scores[ix] : 0.f; sum += weights[r]; }
            if (valid) biased[ix] = -std::numeric_limits<float>::infinity();
        }
        sycl::group_barrier(group);
    }
    if (lane == 0 && weights)
        for (int r = 0; r < k; ++r) weights[r] = weights[r] / (sum + 1e-20f) * scale;
}
struct AttnCounts { int first, window, selected; };
AttnCounts attn_counts(const AttnRows& a, int t) {
    if (!a.gen) return {a.win0[t], a.nwin[t], a.nsel[t]};
    const long long ta = a.t0 + t, pa = a.p0 + ta;
    const long long first = sycl::max(static_cast<long long>(a.win) - a.p0, ta + 1);
    return {int(first), int(a.win + ta - first + 1),
            a.R > 0 ? int(sycl::min(static_cast<long long>(a.kcap), (pa + 1) / a.R)) : 0};
}
struct AttnCaps { size_t max_group, local_bytes, subgroup, units; };
struct AttnTile { size_t local = 0, heads = 0, rows = 0; };
AttnTile attn_tile(const sycl::queue& q, int heads, int d, int T, int requested_heads) {
    const auto device = q.get_device();
    static core::PerDevice<AttnCaps> cache;
    const auto& caps = cache.get(device, [&] {
        AttnCaps c{std::min(device.get_info<sycl::info::device::max_work_group_size>(),
                            device.get_info<sycl::info::device::max_work_item_sizes<1>>()[0]),
                   size_t(device.get_info<sycl::info::device::local_mem_size>()), 1,
                   size_t(device.get_info<sycl::info::device::max_compute_units>())};
        for (const auto width : device.get_info<sycl::info::device::sub_group_sizes>())
            if (width <= c.max_group) c.subgroup = std::max(c.subgroup, width);
        return c;
    });
    const size_t dim = size_t(d), units = std::max(size_t(1), caps.units);
    const size_t head_limit = std::min(size_t(heads), requested_heads > 0 ? size_t(requested_heads) :
                                      std::max(size_t(1), size_t(T) * heads / units));
    AttnTile best;
    for (size_t g = 1; g <= head_limit; ++g) {
        size_t local = std::min(g * dim, caps.max_group);
        if (local >= caps.subgroup) local -= local % caps.subgroup;
        if (!local) continue;
        // Queries and outputs stay local across all KV tiles. Leave two floats
        // per item for implementations that need local scratch for reductions.
        const size_t fixed = (2 * g * dim + 3 * g + 2 * local) * sizeof(float);
        if (fixed >= caps.local_bytes) continue;
        const size_t rows = std::min(std::max(size_t(1), local / caps.subgroup),
                                      (caps.local_bytes - fixed) / ((dim + g) * sizeof(float)));
        if (!rows) continue;
        // Automatic grouping maximizes head/row pairs in the available tile,
        // while head_limit leaves work groups for the reported compute units.
        if (requested_heads > 0 || g * rows > best.heads * best.rows) best = {local, g, rows};
    }
    return best;
}
bool unpack_selected_parts(const void* ckv, const int* sel, int sel_stride, const AttnRows& a, int T, int d,
                           uint16_t* decoded, size_t decoded_stride, sycl::queue& s);
// Upstream unpack_sel_kernel: each query's selected packed rows decoded once into the scratch (rows decoded_stride apart).
void unpack_selected(const void* ckv, const int* sel, int sel_stride, const AttnRows& a, int T, int d,
                     uint16_t* decoded, size_t decoded_stride, sycl::queue& s) {
    if (unpack_selected_parts(ckv, sel, sel_stride, a, T, d, decoded, decoded_stride, s)) return;
    const size_t dim = size_t(d), row_bytes = kv_row_bytes(kKvFp4, d);
    parallel(s, size_t(T) * sel_stride * dim, [=](size_t i) {
        const size_t row = i / dim, t = row / sel_stride, j = row % sel_stride, e = i % dim;
        const int selected = attn_counts(a, int(t)).selected;
        if (selected <= 0 || j >= size_t(selected)) return;
        const auto* source = static_cast<const uint8_t*>(ckv) + size_t(sel[t * sel_stride + j]) * row_bytes;
        // E2M1 times E4M3 has at most eight significant bits, exact in BF16.
        decoded[(t * decoded_stride + j) * dim + e] = bf16_rn(kv_value(source, kKvFp4, d, int(e)));
    });
}
bool sparse_attn_tiled(const float* q, const uint16_t* win, const void* ckv, const int* sel, int sel_stride,
                       const AttnRows& a, const float* sink, int heads, int d, int T, float scale,
                       float* out, sycl::queue& s, int ckv_fmt, int heads_per_block, const AttnScratch* scratch) {
    const auto tile = attn_tile(s, heads, d, T, heads_per_block);
    if (!tile.local) return false;
    const size_t dim = size_t(d), G = tile.heads, R = tile.rows, local = tile.local;
    const size_t head_tiles = (size_t(heads) + G - 1) / G, row_bytes = kv_row_bytes(ckv_fmt, d);
    uint16_t* decoded = nullptr;
    size_t decoded_stride = 0;
    // Scratch rows have the public 512-component model layout. Predecode when
    // requested, or when multiple head blocks would repeat the packed decode.
    if (ckv_fmt == kKvFp4 && d == 512 && sel_stride > 0 && scratch && scratch->rows &&
        T <= scratch->max_t && sel_stride <= scratch->cap &&
        (heads_per_block == -1 || (heads_per_block == 0 && head_tiles > 1))) {
        decoded = scratch->rows;
        decoded_stride = size_t(scratch->cap);
        unpack_selected(ckv, sel, sel_stride, a, T, d, decoded, decoded_stride, s);
    }
    sycl_detail::submit(s, [&](sycl::handler& h) {
        sycl::local_accessor<float, 1> query(G * dim, h), accum(G * dim, h), values(R * dim, h);
        sycl::local_accessor<float, 1> scores(G * R, h), maximum(G, h), denominator(G, h), old_scale(G, h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(size_t(T) * head_tiles * local), sycl::range<1>(local)),
                       [=](sycl::nd_item<1> item) {
            const size_t group = item.get_group_linear_id(), lane = item.get_local_linear_id();
            const size_t t = group / head_tiles, hb = (group % head_tiles) * G;
            const size_t active_heads = sycl::min(G, size_t(heads) - hb);
            const auto ar = attn_counts(a, int(t));
            const int total = ar.window + ar.selected;
            const size_t base = (t * heads + hb) * dim;
            if (total <= 0) {
                for (size_t i = lane; i < active_heads * dim; i += local) out[base + i] = 0.f;
                return;
            }
            for (size_t i = lane; i < G * dim; i += local) {
                query[i] = i < active_heads * dim ? q[base + i] : 0.f;
                accum[i] = 0.f;
            }
            for (size_t g = lane; g < G; g += local) {
                maximum[g] = g < active_heads ? sink[hb + g] : 0.f;
                denominator[g] = 1.f;
            }
            const auto sg = item.get_sub_group();
            const size_t sl = sg.get_local_linear_id(), sw = sg.get_local_linear_range();
            const size_t si = sg.get_group_linear_id(), sn = sg.get_group_linear_range();
            for (int start = 0; start < total; start += int(R)) {
                const size_t nr = sycl::min(R, size_t(total - start));
                for (size_t i = lane; i < nr * dim; i += local) {
                    const int j = start + int(i / dim), e = int(i % dim);
                    const int fmt = j < ar.window || decoded ? kKvBf16 : ckv_fmt;
                    const void* row = j < ar.window ? static_cast<const void*>(win + size_t(ar.first + j) * dim) :
                        decoded ? static_cast<const void*>(decoded + (t * decoded_stride + size_t(j - ar.window)) * dim) :
                        static_cast<const void*>(static_cast<const uint8_t*>(ckv) +
                            size_t(sel[t * sel_stride + j - ar.window]) * row_bytes);
                    values[i] = kv_value(row, fmt, d, e);
                }
                sycl::group_barrier(item.get_group());
                // The actual sub-groups cover pairs independently of the host's
                // shape hint; neither their width nor their count is assumed.
                for (size_t pair = si; pair < G * nr; pair += sn) {
                    const size_t g = pair / nr, r = pair % nr;
                    float dot = 0.f;
                    for (size_t e = sl; e < dim; e += sw) dot += query[g * dim + e] * values[r * dim + e];
                    dot = sycl::reduce_over_group(sg, dot, sycl::plus<float>());
                    if (sl == 0) scores[g * R + r] = dot * scale;
                }
                sycl::group_barrier(item.get_group());
                for (size_t g = si; g < G; g += sn) {
                    float top = -std::numeric_limits<float>::infinity();
                    for (size_t r = sl; r < nr; r += sw) top = sycl::fmax(top, scores[g * R + r]);
                    top = sycl::reduce_over_group(sg, top, sycl::maximum<float>());
                    const float next = sycl::fmax(maximum[g], top), keep = sycl::exp(maximum[g] - next);
                    float sum = 0.f;
                    for (size_t r = sl; r < nr; r += sw) {
                        const float weight = sycl::exp(scores[g * R + r] - next);
                        scores[g * R + r] = weight;
                        sum += weight;
                    }
                    sum = sycl::reduce_over_group(sg, sum, sycl::plus<float>());
                    if (sl == 0) {
                        maximum[g] = next;
                        denominator[g] = denominator[g] * keep + sum;
                        old_scale[g] = keep;
                    }
                }
                sycl::group_barrier(item.get_group());
                for (size_t i = lane; i < G * dim; i += local) {
                    const size_t g = i / dim, e = i % dim;
                    float value = accum[i] * old_scale[g];
                    for (size_t r = 0; r < nr; ++r) value = sycl::fma(scores[g * R + r], values[r * dim + e], value);
                    accum[i] = value;
                }
                sycl::group_barrier(item.get_group());
            }
            for (size_t i = lane; i < active_heads * dim; i += local) out[base + i] = accum[i] / denominator[i / dim];
        });
    });
    return true;
}
uint32_t ord_key(float v) {
    const uint32_t u = sycl::bit_cast<uint32_t>(v);
    return u ^ ((u & 0x80000000u) ? 0xffffffffu : 0x80000000u);
}
float iq4nl_value(const uint8_t* b, int e) {
    constexpr int8_t values[] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
    const uint8_t q = b[2 + (e & 15)];
    return f16_value(uint16_t(b[0] | uint16_t(b[1]) << 8)) * float(values[(e & 16) ? q >> 4 : q & 15]);
}
float engram_value(const uint8_t* row, const uint8_t* scales, int e, int fmt) {
    if (fmt == 1) return q6k_value(row + size_t(e / 256) * kQ6kBytes, e % 256);
    if (fmt == 2) return iq4nl_value(row + size_t(e / 32) * 18, e % 32);
    const uint8_t b = scales[e / 32];
    const float scale = b == 255 ? sycl::bit_cast<float>(0x7fc00000u) :
                        b == 0 ? 5.877471754111438e-39f : sycl::bit_cast<float>(uint32_t(b) << 23);
    return e4m3_value(row[e]) * scale;
}

struct GroupCaps {
    size_t max_group = 0, local = 0;
    std::vector<size_t> subgroups;
};
const GroupCaps& group_caps(const sycl::queue& q) {
    const auto device = q.get_device();
    static core::PerDevice<GroupCaps> cache;
    return cache.get(device, [&] {
        return GroupCaps{std::min(device.get_info<sycl::info::device::max_work_group_size>(),
                                  device.get_info<sycl::info::device::max_work_item_sizes<1>>()[0]),
                         size_t(device.get_info<sycl::info::device::local_mem_size>()),
                         device.get_info<sycl::info::device::sub_group_sizes>()};
    });
}
// Whether a kernel can release host-visible data to the host: atomics on host allocations and a
// system-scope fence ahead of the flag's store.
bool system_publish(const sycl::queue& q) {
    const auto device = q.get_device();
    static core::PerDevice<bool> cache;
    return cache.get(device, [&] {
        const auto fences = device.get_info<sycl::info::device::atomic_fence_scope_capabilities>();
        return device.has(sycl::aspect::usm_atomic_host_allocations) &&
               std::find(fences.begin(), fences.end(), sycl::memory_scope::system) != fences.end();
    });
}
bool has_subgroup(const GroupCaps& caps, size_t width) {
    return std::find(caps.subgroups.begin(), caps.subgroups.end(), width) != caps.subgroups.end();
}
template <int SG>
float subgroup_sum(sycl::sub_group sg, float v) {
#pragma unroll
    for (int o = SG / 2; o > 0; o >>= 1) v += sycl::permute_group_by_xor(sg, v, o);
    return v;
}
template <int SG>
float subgroup_max(sycl::sub_group sg, float v) {
#pragma unroll
    for (int o = SG / 2; o > 0; o >>= 1) v = sycl::fmax(v, sycl::permute_group_by_xor(sg, v, o));
    return v;
}

// Upstream sparse_attn_kernel<kContig>: a work-group a (head, query), its rows in tiles of 64, a subgroup a row's
// dot (each lane d / SG components), the running max and denominator on subgroup 0, two output values a work-item,
// the sink added at the end. With Contig the selected rows are the query's rows of the decoded scratch (rows apart
// sel_stride), otherwise BF16 rows indexed by sel. Upstream's 256-item group (d = 512, two values a work-item) is
// taken only where the device reports it and the subgroup width.
template <bool Contig, int SG>
void sparse_rows(const float* q, const uint16_t* win, const uint16_t* rows_base, const int* sel, int sel_stride,
                 const AttnRows& a, const float* sink, int heads, int T, float scale, float* out, sycl::queue& s) {
    constexpr int WG = 256, Tile = 64, D = 512, Per = D / SG;
    sycl_detail::submit(s, [&](sycl::handler& h) {
        sycl::local_accessor<float, 1> qs(sycl::range<1>(D), h), sc(sycl::range<1>(Tile), h), st(sycl::range<1>(3), h);
        // A tile's rows as element offsets: a window row from win, a selected row from rows_base (negated, less one).
        sycl::local_accessor<long long, 1> rows(sycl::range<1>(Tile), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(size_t(T) * heads * WG), sycl::range<1>(WG)),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(STRATA_SUB_GROUP(SG))]] {
            const int g = int(it.get_group_linear_id()), t = g / heads, hh = g % heads;
            const int tid = int(it.get_local_linear_id());
            const auto c = attn_counts(a, t);
            auto row = [&](int j) { const long long o = rows[j]; return o >= 0 ? win + o : rows_base + (-1 - o); };
            const int* sl = sel + size_t(t) * sel_stride;
            for (int i = tid; i < D; i += WG) qs[i] = q[(size_t(t) * heads + hh) * D + i];
            if (tid == 0) { st[0] = -1e30f; st[2] = 0.f; }
            it.barrier(sycl::access::fence_space::local_space);
            const auto sg = it.get_sub_group();
            const int lane = int(sg.get_local_linear_id()), wp = tid / SG, nw = WG / SG;
            float o0 = 0.f, o1 = 0.f;
            const int total = c.window + c.selected;
            for (int t0 = 0; t0 < total; t0 += Tile) {
                const int nt = sycl::min(Tile, total - t0);
                if (tid < nt) {
                    const int j = t0 + tid;
                    rows[tid] = j < c.window ? (static_cast<long long>(c.first) + j) * D :
                        -1 - static_cast<long long>(Contig ? size_t(t) * sel_stride + size_t(j - c.window) :
                                                    size_t(sl[j - c.window])) * D;
                }
                it.barrier(sycl::access::fence_space::local_space);
                for (int jj = wp; jj < nt; jj += nw) {
                    const uint16_t* rr = row(jj) + lane * Per;
                    float dot = 0.f;
#pragma unroll
                    for (int x = 0; x < Per; ++x) dot += qs[lane * Per + x] * bf16_value(rr[x]);
                    dot = subgroup_sum<SG>(sg, dot);
                    if (lane == 0) sc[jj] = dot * scale;
                }
                it.barrier(sycl::access::fence_space::local_space);
                if (wp == 0) {
                    float m = -std::numeric_limits<float>::infinity();
                    for (int j = lane; j < nt; j += SG) m = sycl::fmax(m, sc[j]);
                    m = subgroup_max<SG>(sg, m);
                    const float m_old = st[0], m_new = sycl::fmax(m_old, m);
                    float sum = 0.f;
                    for (int j = lane; j < nt; j += SG) {
                        const float p = sycl::exp(sc[j] - m_new);
                        sc[j] = p;
                        sum += p;
                    }
                    // Every lane read st[0] before this convergent reduction, so lane 0 may update it without a
                    // subgroup barrier: on the HIP backend that barrier is the work-group's, and in this branch it
                    // would pair with the other subgroups' barrier below, which then read the old values.
                    sum = subgroup_sum<SG>(sg, sum);
                    if (lane == 0) {
                        st[1] = sycl::exp(m_old - m_new);
                        st[2] = st[2] * st[1] + sum;
                        st[0] = m_new;
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);
                const float cs = st[1];
                o0 *= cs;
                o1 *= cs;
                const int col = tid * 2;
                for (int j = 0; j < nt; ++j) {
                    const uint16_t* rr = row(j);
                    o0 += sc[j] * bf16_value(rr[col]);
                    o1 += sc[j] * bf16_value(rr[col + 1]);
                }
                it.barrier(sycl::access::fence_space::local_space);
            }
            const float den = st[2] + sycl::exp(sink[hh] - st[0]);
            float* ov = out + (size_t(t) * heads + hh) * D;
            ov[size_t(tid) * 2] = o0 / den;
            ov[size_t(tid) * 2 + 1] = o1 / den;
        });
    });
}
// The subgroup width for upstream's warp-a-row GEMVs (eight rows a group), or 0.
int gemv_warp_width(const sycl::queue& s) {
    const auto& caps = group_caps(s);
    for (const int sg : {32, 16}) if (caps.max_group >= 8 * size_t(sg) && has_subgroup(caps, size_t(sg))) return sg;
    return 0;
}
// The vector loads need 16-byte aligned rows: `per` values a 16-byte load.
template <class Weight>
bool gemv_warp_ready(const sycl::queue& s, const Weight* W, const float* x, size_t ldx, int k, int per) {
    return gemv_warp_width(s) && k > 0 && k % per == 0 && ldx % 4 == 0 &&
           reinterpret_cast<uintptr_t>(W) % 16 == 0 && reinterpret_cast<uintptr_t>(x) % 16 == 0;
}
template <class T, size_t N>
T load_aligned(const void* p) {
    T value;
    __builtin_memcpy(&value, __builtin_assume_aligned(p, N), sizeof(value));
    return value;
}
// Upstream gemv_f16_rows_kernel (gemv_f16_kernel's sums for each activation): a subgroup a row, eight rows a
// group, each lane eight F16 weights and the activations' float4s a step, two activations a group.
template <int SG>
void gemv_f16_warp(const uint16_t* W, const float* x, size_t ldx, int n, int k, int T, float alpha,
                   float* y, size_t ldy, sycl::queue& s) {
    constexpr int Rows = 8, RowsT = 2, WG = Rows * SG;
    const size_t gx = (size_t(n) + Rows - 1) / Rows, gy = (size_t(T) + RowsT - 1) / RowsT;
    sycl_detail::launch(s, sycl::nd_range<1>(sycl::range<1>(gx * gy * WG), sycl::range<1>(WG)),
                   [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(STRATA_SUB_GROUP(SG))]] {
        const size_t g = it.get_group_linear_id();
        const int tid = int(it.get_local_linear_id()), lane = tid % SG;
        const int row = int(g % gx) * Rows + tid / SG, t0 = int(g / gx) * RowsT, nt = sycl::min(RowsT, T - t0);
        if (row >= n) return;
        const uint16_t* w = W + size_t(row) * k;
        float acc[RowsT] = {};
        for (int c = lane * 8; c < k; c += SG * 8) {
            const auto q = load_aligned<sycl::vec<uint16_t, 8>, 16>(w + c);
            float xv[RowsT][8] = {};
#pragma unroll
            for (int t = 0; t < RowsT; ++t)
                if (t < nt) {
                    const float* xp = x + size_t(t0 + t) * ldx + c;
                    const auto a = load_aligned<sycl::float4, 16>(xp), b = load_aligned<sycl::float4, 16>(xp + 4);
                    xv[t][0] = a.x(); xv[t][1] = a.y(); xv[t][2] = a.z(); xv[t][3] = a.w();
                    xv[t][4] = b.x(); xv[t][5] = b.y(); xv[t][6] = b.z(); xv[t][7] = b.w();
                }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float f0 = f16_value(q[2 * i]), f1 = f16_value(q[2 * i + 1]);
#pragma unroll
                for (int t = 0; t < RowsT; ++t)
                    if (t < nt) acc[t] += f0 * xv[t][size_t(2) * i] + f1 * xv[t][size_t(2) * i + 1];
            }
        }
#pragma unroll
        for (int t = 0; t < RowsT; ++t)
            if (t < nt) {
                const float a = subgroup_sum<SG>(it.get_sub_group(), acc[t]);
                if (lane == 0) y[size_t(t0 + t) * ldy + row] = alpha * a;
            }
    });
}
// Upstream gemv_f32_kernel (the router): a subgroup a row, eight rows a group, the lanes over float4s.
template <int SG>
void gemv_f32_warp(const float* W, const float* x, int n, int k, int T, float* y, size_t ldy, sycl::queue& s) {
    constexpr int Rows = 8, WG = Rows * SG;
    const size_t gx = (size_t(n) + Rows - 1) / Rows;
    sycl_detail::launch(s, sycl::nd_range<1>(sycl::range<1>(gx * size_t(T) * WG), sycl::range<1>(WG)),
                   [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(STRATA_SUB_GROUP(SG))]] {
        const size_t g = it.get_group_linear_id();
        const int tid = int(it.get_local_linear_id()), lane = tid % SG;
        const int row = int(g % gx) * Rows + tid / SG, t = int(g / gx);
        if (row >= n) return;
        const float* w = W + size_t(row) * k;
        const float* xv = x + size_t(t) * k;
        float acc = 0.f;
        for (int c = lane; c < k / 4; c += SG) {
            const auto a = load_aligned<sycl::float4, 16>(w + size_t(4) * c), b = load_aligned<sycl::float4, 16>(xv + size_t(4) * c);
            acc = sycl::fma(a.x(), b.x(), acc);
            acc = sycl::fma(a.y(), b.y(), acc);
            acc = sycl::fma(a.z(), b.z(), acc);
            acc = sycl::fma(a.w(), b.w(), acc);
        }
        acc = subgroup_sum<SG>(it.get_sub_group(), acc);
        if (lane == 0) y[size_t(t) * ldy + row] = acc;
    });
}
// Upstream unpack_sel_kernel's shape for 512-value rows: sixteen work-items a selected row, each 32 values from one
// 16-byte load of codes and their two scales, stored 16 bytes at a time - kv_value's values, the same BF16 bits.
// Takes upstream's 256-item group where the device reports it, and 16-byte aligned rows.
bool unpack_selected_parts(const void* ckv, const int* sel, int sel_stride, const AttnRows& a, int T, int d,
                           uint16_t* decoded, size_t decoded_stride, sycl::queue& s) {
    constexpr int Parts = 16, Rows = 16, WG = Parts * Rows, Dim = 512;
    if (d != Dim || sel_stride <= 0 || group_caps(s).max_group < size_t(WG) ||
        reinterpret_cast<uintptr_t>(ckv) % 16 || reinterpret_cast<uintptr_t>(decoded) % 16) return false;
    const size_t row_bytes = kv_row_bytes(kKvFp4, Dim), groups = (size_t(sel_stride) + Rows - 1) / Rows;
    sycl_detail::launch(s, sycl::nd_range<2>(sycl::range<2>(size_t(T), groups * WG), sycl::range<2>(1, WG)),
                        [=](sycl::nd_item<2> it) {
        const int t = int(it.get_group(0)), item = int(it.get_local_id(1));
        const int j = int(it.get_group(1)) * Rows + item / Parts, part = item % Parts;
        // attn_counts' selected rows, a.nsel read at constant indices: indexing the kernel's copy of the rows
        // by t would put it in private memory.
        int selected = 0;
        if (a.gen) {
            if (a.R > 0) selected = int(sycl::min(static_cast<long long>(a.kcap), (a.p0 + a.t0 + t + 1) / a.R));
        } else {
#pragma unroll
            for (int i = 0; i < kWin; ++i)
                if (i == t) selected = a.nsel[i];
        }
        if (j >= selected) return;
        const auto* row = static_cast<const uint8_t*>(ckv) + size_t(sel[size_t(t) * sel_stride + j]) * row_bytes;
        const auto codes = load_aligned<sycl::vec<uint32_t, 4>, 16>(row + size_t(part) * 16);
        const float k0 = e4m3_value(row[Dim / 2 + part * 2]), k1 = e4m3_value(row[Dim / 2 + part * 2 + 1]);
        sycl::vec<uint32_t, 16> out;
#pragma unroll
        for (int i = 0; i < 16; ++i) {   // values 2i and 2i + 1: the low and the high nibble of byte i
            const uint32_t b = codes[i >> 2] >> (8 * (i & 3));
            const float k = i < 8 ? k0 : k1;
            out[i] = uint32_t(bf16_rn(e2m1_value(uint8_t(b & 15u)) * k)) |
                     uint32_t(bf16_rn(e2m1_value(uint8_t((b >> 4) & 15u)) * k)) << 16;
        }
        __builtin_memcpy(__builtin_assume_aligned(decoded + (size_t(t) * decoded_stride + j) * Dim + size_t(part) * 32, 16),
                         &out, sizeof(out));
    });
    return true;
}
// The subgroup width the rows kernel can take on this device, or 0.
int sparse_rows_width(const sycl::queue& s) {
    const auto& caps = group_caps(s);
    if (caps.max_group < 256 || caps.local < 579 * sizeof(float) + 64 * sizeof(long long)) return 0;
    return has_subgroup(caps, 32) ? 32 : has_subgroup(caps, 16) ? 16 : 0;
}

// Upstream index_score_kernel<128, KF>: a work-item a position, the token's queries and head weights in local
// memory, each key row decoded once for every head. Upstream's 256-item launch bound is used only where the
// device reports that group size and the local memory.
template <int KF>
bool index_score_group(const float* q, const float* w, const uint8_t* kc, int heads, int T, int n, const int* cand,
                       int cand_stride, int blk, const Rows& lim, float* out, long long out_stride, sycl::queue& s) {
    constexpr int HD = 128, WG = 256;
    const auto& caps = group_caps(s);
    const size_t qn = size_t(heads) * HD + size_t(heads);
    if (caps.max_group < size_t(WG) || caps.local < qn * sizeof(float)) return false;
    const size_t groups = (size_t(n) + WG - 1) / WG;
    sycl_detail::submit(s, [&](sycl::handler& h) {
        sycl::local_accessor<float, 1> qs(sycl::range<1>(qn), h);
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>(size_t(T), groups * WG), sycl::range<2>(1, WG)),
                       [=](sycl::nd_item<2> item) {
            const int t = int(item.get_group(0)), tid = int(item.get_local_id(1));
            for (int i = tid; i < heads * HD; i += WG) qs[i] = q[size_t(t) * heads * HD + i];
            for (int i = tid; i < heads; i += WG) qs[heads * HD + i] = w[size_t(t) * 64 + i];
            item.barrier(sycl::access::fence_space::local_space);
            const int i = int(item.get_group(1)) * WG + tid;
            if (i >= n) return;
            float* o = out + static_cast<long long>(t) * out_stride;
            int ln, lk;
            rows_at(lim, t, ln, lk);
            int p = i;
            if (cand) {
                if (i >= lk * blk) { o[i] = -std::numeric_limits<float>::infinity(); return; }
                p = cand[size_t(t) * cand_stride + i / blk] * blk + i % blk;
            }
            if (p >= ln) { o[i] = -std::numeric_limits<float>::infinity(); return; }
            const uint8_t* kr = kc + size_t(p) * kv_row_bytes(KF, HD);
            float acc[32];
#pragma unroll
            for (int hh = 0; hh < 32; ++hh) acc[hh] = 0.f;
#pragma unroll 2
            for (int c = 0; c < HD; c += 8) {
                float kv[8];
                if constexpr (KF == kKvFp4p) {
                    const float ks = sycl::bit_cast<float>(uint32_t(kr[HD / 2 + c / 32]) << 23);
#pragma unroll
                    for (int x = 0; x < 8; ++x) kv[x] = e2m1_value(uint8_t(kr[(c + x) / 2] >> (4 * (x & 1)))) * ks;
                } else {
                    const auto* b = reinterpret_cast<const uint16_t*>(kr) + c;
#pragma unroll
                    for (int x = 0; x < 8; ++x) kv[x] = bf16_value(b[x]);
                }
#pragma unroll
                for (int hh = 0; hh < 32; ++hh) {
                    if (hh >= heads) break;
                    float query[8];
#pragma unroll
                    for (int x = 0; x < 8; ++x) query[x] = qs[hh * HD + c + x];
#pragma unroll
                    for (int x = 0; x < 8; ++x) acc[hh] += query[x] * kv[x];
                }
            }
            float sum = 0.f;
#pragma unroll
            for (int hh = 0; hh < 32; ++hh) {
                if (hh >= heads) break;
                sum += sycl::fmax(acc[hh], 0.f) * qs[heads * HD + hh];
            }
            o[i] = sum;
        });
    });
    return true;
}

// Upstream topk_kernel's ballot/popcount and warp-count prefix, generalized to
// reported subgroups. The old WG scan path remains the capability fallback.
#if defined(SYCL_EXT_ONEAPI_SUB_GROUP_MASK) && SYCL_EXT_ONEAPI_SUB_GROUP_MASK >= 2
template <int SG>
bool topk_ballot_group(const float* vals, long long stride, int T, const Rows& r, const int* cand, int cand_stride,
                       int blk, int* out, int out_stride, size_t wg, const GroupCaps& caps, sycl::queue& s) {
    if (wg % SG || !has_subgroup(caps, SG)) return false;
    const size_t count_groups = wg / SG;
    const size_t local_bytes = 256 * sizeof(unsigned) + 5 * sizeof(uint32_t) + 2 * count_groups * sizeof(int);
    if (caps.local < local_bytes) return false;
    sycl_detail::submit(s, [&](sycl::handler& h) {
        sycl::local_accessor<unsigned, 1> hist(sycl::range<1>(256), h);
        sycl::local_accessor<uint32_t, 1> state(sycl::range<1>(5), h); // prefix, mask, remaining, gt, eq
        sycl::local_accessor<int, 1> offsets(sycl::range<1>(2 * count_groups), h);
        h.parallel_for(sycl::nd_range<1>(size_t(T) * wg, wg),
                       [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(STRATA_SUB_GROUP(SG))]] {
            const auto sg = item.get_sub_group();
            const int lane = int(sg.get_local_id()[0]), width = int(sg.get_local_range()[0]);
            const int wid = int(sg.get_group_id()[0]), groups = int(sg.get_group_range()[0]);
            // Assign consecutive logical positions from subgroup IDs explicitly;
            // do not assume a physical mapping from work-group local IDs.
            const int tid = wid * width + lane, step = int(wg), t = int(item.get_group_linear_id());
            int n, k;
            rows_at(r, t, n, k);
            if (n <= 0 || k <= 0) return;
            const float* v = vals + static_cast<long long>(t) * stride;
            int* dst = out + size_t(t) * out_stride;
            const int* cd = cand ? cand + size_t(t) * cand_stride : nullptr;
            if (k >= n) {
                for (int i = tid; i < n; i += step) dst[i] = cd ? cd[i / blk] * blk + i % blk : i;
                return;
            }
            if (tid == 0) {
                state[0] = 0u; state[1] = 0u; state[2] = uint32_t(k); state[3] = 0u; state[4] = 0u;
            }
            item.barrier(sycl::access::fence_space::local_space);
            // Preserve all four upstream radix passes and the integer threshold.
            for (int shift = 24; shift >= 0; shift -= 8) {
                for (int i = tid; i < 256; i += step) hist[i] = 0u;
                item.barrier(sycl::access::fence_space::local_space);
                const uint32_t prefix = state[0], mask = state[1];
                for (int i = tid; i < n; i += step) {
                    const uint32_t u = ord_key(v[i]);
                    if ((u & mask) == prefix)
                        sycl::atomic_ref<unsigned, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                                         sycl::access::address_space::local_space>(hist[(u >> shift) & 255u])
                            .fetch_add(1u);
                }
                item.barrier(sycl::access::fence_space::local_space);
                if (tid == 0) {
                    const int rem = int(state[2]);
                    int cum = 0;
                    for (int digit = 255; digit >= 0; --digit) {
                        const int count = int(hist[digit]);
                        if (cum + count >= rem) {
                            state[0] = prefix | (uint32_t(digit) << shift);
                            state[1] = mask | (255u << shift);
                            state[2] = uint32_t(rem - cum);
                            break;
                        }
                        cum += count;
                    }
                }
                item.barrier(sycl::access::fence_space::local_space);
            }
            const uint32_t threshold = state[0], below = (uint32_t(1) << lane) - 1u;
            const int remaining = int(state[2]), n_gt = k - remaining;
            for (int i0 = 0; i0 < n; i0 += step) {
                const int i = i0 + tid;
                const uint32_t u = i < n ? ord_key(v[i]) : 0u;
                const bool gt = i < n && u > threshold, eq = i < n && u == threshold;
                uint32_t bg = 0, be = 0;
                sycl::ext::oneapi::group_ballot(sg, gt).extract_bits(bg);
                sycl::ext::oneapi::group_ballot(sg, eq).extract_bits(be);
                if (lane == 0) {
                    offsets[wid] = int(sycl::popcount(bg));
                    offsets[groups + wid] = int(sycl::popcount(be));
                }
                item.barrier(sycl::access::fence_space::local_space);
                if (wid == 0) {
                    int carry_gt = int(state[3]), carry_eq = int(state[4]);
                    // At WG1024, SG32 takes one chunk and SG16 takes four.
                    // All members of this subgroup execute every shuffle.
                    for (int first = 0; first < groups; first += width) {
                        const int j = first + lane;
                        const int g0 = j < groups ? offsets[j] : 0;
                        const int e0 = j < groups ? offsets[groups + j] : 0;
                        int gs = g0, es = e0;
                        for (int delta = 1; delta < width; delta <<= 1) {
                            const int a = sycl::shift_group_right(sg, gs, uint32_t(delta));
                            const int b = sycl::shift_group_right(sg, es, uint32_t(delta));
                            if (lane >= delta) { gs += a; es += b; }
                        }
                        if (j < groups) {
                            offsets[j] = carry_gt + gs - g0;
                            offsets[groups + j] = carry_eq + es - e0;
                        }
                        carry_gt += sycl::select_from_group(sg, gs, sycl::id<1>(size_t(width - 1)));
                        carry_eq += sycl::select_from_group(sg, es, sycl::id<1>(size_t(width - 1)));
                    }
                    if (lane == 0) { state[3] = uint32_t(carry_gt); state[4] = uint32_t(carry_eq); }
                }
                // No conditional subgroup barrier: all work-items reach each
                // work-group barrier, including the one before scratch reuse.
                item.barrier(sycl::access::fence_space::local_space);
                int slot = -1;
                if (gt) slot = offsets[wid] + int(sycl::popcount(bg & below));
                else if (eq) {
                    const int equal_rank = offsets[groups + wid] + int(sycl::popcount(be & below));
                    if (equal_rank < remaining) slot = n_gt + equal_rank;
                }
                if (slot >= 0) dst[slot] = cd ? cd[i / blk] * blk + i % blk : i;
                item.barrier(sycl::access::fence_space::local_space);
            }
        });
    });
    return true;
}
#endif


// Upstream topk_kernel: a work-group a row, the k-th largest key by four 8-bit radix passes over a local
// histogram, then the picks in index order (strictly greater first, then the lowest equal positions). The ranks
// are integer scans over the work-group, so the result does not depend on the group or sub-group size; the size
// is upstream's 1024, or the device's largest group where that is smaller.
bool topk_group(const float* vals, long long stride, int T, const Rows& r, const int* cand, int cand_stride, int blk,
                int* out, int out_stride, sycl::queue& s) {
    const auto& caps = group_caps(s);
    const size_t wg = std::min<size_t>(1024, caps.max_group);
    if (wg < 32 || caps.local < 256 * sizeof(unsigned) + 4 * sizeof(uint32_t)) return false;
#if defined(SYCL_EXT_ONEAPI_SUB_GROUP_MASK) && SYCL_EXT_ONEAPI_SUB_GROUP_MASK >= 2
    if (topk_ballot_group<32>(vals, stride, T, r, cand, cand_stride, blk, out, out_stride, wg, caps, s)) return true;
    if (topk_ballot_group<16>(vals, stride, T, r, cand, cand_stride, blk, out, out_stride, wg, caps, s)) return true;
#endif
    sycl_detail::submit(s, [&](sycl::handler& h) {
        sycl::local_accessor<unsigned, 1> hist(sycl::range<1>(256), h);
        sycl::local_accessor<uint32_t, 1> state(sycl::range<1>(3), h);   // prefix, mask, remaining
        h.parallel_for(sycl::nd_range<1>(size_t(T) * wg, wg), [=](sycl::nd_item<1> item) {
            const auto group = item.get_group();
            const int t = int(item.get_group_linear_id()), tid = int(item.get_local_linear_id()), step = int(wg);
            int n, k;
            rows_at(r, t, n, k);
            if (n <= 0 || k <= 0) return;
            const float* v = vals + static_cast<long long>(t) * stride;
            int* dst = out + size_t(t) * out_stride;
            const int* cd = cand ? cand + size_t(t) * cand_stride : nullptr;
            if (k >= n) {
                for (int i = tid; i < n; i += step) dst[i] = cd ? cd[i / blk] * blk + i % blk : i;
                return;
            }
            if (tid == 0) { state[0] = 0u; state[1] = 0u; state[2] = uint32_t(k); }
            item.barrier(sycl::access::fence_space::local_space);
            for (int shift = 24; shift >= 0; shift -= 8) {
                for (int i = tid; i < 256; i += step) hist[i] = 0u;
                item.barrier(sycl::access::fence_space::local_space);
                const uint32_t prefix = state[0], mask = state[1];
                for (int i = tid; i < n; i += step) {
                    const uint32_t u = ord_key(v[i]);
                    if ((u & mask) == prefix)
                        sycl::atomic_ref<unsigned, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                                         sycl::access::address_space::local_space>(hist[(u >> shift) & 255u])
                            .fetch_add(1u);
                }
                item.barrier(sycl::access::fence_space::local_space);
                if (tid == 0) {
                    const int rem = int(state[2]);
                    int cum = 0;
                    for (int digit = 255; digit >= 0; --digit) {
                        const int count = int(hist[digit]);
                        if (cum + count >= rem) {
                            state[0] = prefix | (uint32_t(digit) << shift);
                            state[1] = mask | (255u << shift);
                            state[2] = uint32_t(rem - cum);
                            break;
                        }
                        cum += count;
                    }
                }
                item.barrier(sycl::access::fence_space::local_space);
            }
            const uint32_t thr = state[0];
            const int remaining = int(state[2]), n_gt = k - remaining;
            int cnt_gt = 0, cnt_eq = 0;
            for (int i0 = 0; i0 < n; i0 += step) {
                const int i = i0 + tid;
                const uint32_t u = i < n ? ord_key(v[i]) : 0u;
                const int gt = i < n && u > thr, eq = i < n && u == thr;
                const int rank_gt = sycl::exclusive_scan_over_group(group, gt, sycl::plus<int>());
                const int rank_eq = sycl::exclusive_scan_over_group(group, eq, sycl::plus<int>());
                int slot = -1;
                if (gt) slot = cnt_gt + rank_gt;
                else if (eq && cnt_eq + rank_eq < remaining) slot = n_gt + cnt_eq + rank_eq;
                if (slot >= 0) dst[slot] = cd ? cd[i / blk] * blk + i % blk : i;
                cnt_gt += sycl::reduce_over_group(group, gt, sycl::plus<int>());
                cnt_eq += sycl::reduce_over_group(group, eq, sycl::plus<int>());
            }
        });
    });
    return true;
}
}

void bind_mailbox(sycl::queue& queue, const std::shared_ptr<MailboxControl>& control) {
    auto& registry = mailbox_registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    registry.queues[&queue] = control;
}
void unbind_mailbox(const sycl::queue* queue) noexcept {
    try {
        auto& registry = mailbox_registry();
        std::lock_guard<std::mutex> lock(registry.mutex);
        registry.queues.erase(queue);
    } catch (...) { std::fputs("ds4: mailbox registry cleanup failed\n", stderr); }
}

void embed(const uint16_t* rows, int T, float* h, int dim, int hc, sycl::queue& s) {
    parallel(s, size_t(T) * dim, [=](size_t i) {
        const size_t t = i / dim, e = i % dim;
        const float v = bf16_value(rows[i]);
        for (int j = 0; j < hc; ++j) h[(t * hc + j) * dim + e] = v;
    });
}

void finish(float* x, int n_vec, int len, long long stride, const Finish& f, sycl::queue& s) {
    if (n_vec <= 0) return;
    if (len <= 0 || f.rd < 0 || f.rd > len || (f.rd & 1) || (f.rd && ((len - f.rd) & 1)) || f.vpp <= 0)
        throw std::invalid_argument("ds4 finish: invalid vector or rotary dimensions");
    if (f.pdst && (f.quant < 1 || f.quant > 3 || len % 32))
        throw std::invalid_argument("ds4 finish: packed rows require whole quantization blocks");
    if (finish_detail::finish(x, n_vec, len, stride, f, s)) return;
    parallel(s, size_t(n_vec), [=](size_t v) {
        float* row = x + static_cast<long long>(v) * stride;
        float ss = 0.f;
        if (f.w) for (int e = 0; e < len; ++e) ss += row[e] * row[e];
        const float inv = f.w ? sycl::rsqrt(ss / float(len) + f.eps) : 1.f;
        const int g = f.quant == 2 ? 16 : 32;
        const int r0 = len - f.rd, pos = f.pos + int(v / f.vpp) * f.pos_step;
        uint8_t* packed = f.pdst ? f.pdst + static_cast<long long>(v) * f.pdst_stride : nullptr;
        uint16_t* dst = f.dst ? f.dst + static_cast<long long>(v) * f.dst_stride : nullptr;
        // These 32 values are a quantization block, independent of the device's subgroup width.
        for (int start = 0; start < len; start += g) {
            float vals[32];
            const int count = sycl::min(g, len - start);
            for (int j = 0; j < count; ++j) vals[j] = f.w ? row[start + j] * inv * f.w[start + j] : row[start + j];
            for (int j = 0; j + 1 < count; j += 2) {
                const int e = start + j;
                if (e < r0) continue;
                const float ang = float(pos) * f.freqs[(e - r0) / 2];
                const float c = sycl::cos(ang), sn = (f.inverse ? -1.f : 1.f) * sycl::sin(ang);
                const float a = vals[j], b = vals[j + 1];
                vals[j] = a * c - b * sn;
                vals[j + 1] = a * sn + b * c;
            }
            float amax = 0.f, scale = 1.f;
            if (f.quant) {
                for (int j = 0; j < count; ++j) amax = sycl::fmax(amax, sycl::fabs(vals[j]));
                if (f.quant == 1) scale = pow2_ceil(sycl::fmax(amax, 1e-4f) / 448.f);
                else if (f.quant == 2) scale = e4m3_round(sycl::fmax(amax, 6.f * .001953125f) / 6.f);
                else scale = pow2_ceil(sycl::fmax(amax, 6.f * 1.1754943508222875e-38f) / 6.f);
            }
            for (int j = 0; j < count; ++j) {
                const float q = f.quant == 1 ? e4m3_round(sycl::clamp(vals[j] / scale, -448.f, 448.f)) :
                                f.quant ? e2m1_round(sycl::clamp(vals[j] / scale, -6.f, 6.f)) : vals[j];
                vals[j] = q * scale;
                const int e = start + j;
                if (packed) {
                    if (f.quant == 1) packed[e] = e4m3_code(q);
                    else if (!(e & 1)) packed[e / 2] = e2m1_code(q);
                    else packed[e / 2] |= uint8_t(e2m1_code(q) << 4);
                }
                if (dst) dst[e] = bf16_rn(vals[j]);
                else if (!packed) row[e] = vals[j];
            }
            if (packed) packed[(f.quant == 1 ? len : len / 2) + start / g] =
                f.quant == 2 ? e4m3_code(scale) : uint8_t(sycl::bit_cast<uint32_t>(scale) >> 23);
        }
    });
}

void hc_mix(const float* h, const float* fn, int hcdim, int n_mix, int T, float eps, float* mixes, sycl::queue& s) {
    if (hc_detail::mix(h, fn, hcdim, n_mix, T, eps, mixes, s)) return;
    parallel(s, size_t(T) * n_mix, [=](size_t i) {
        const int t = int(i / n_mix), m = int(i % n_mix);
        float acc = 0.f, ss = 0.f;
        for (int k = 0; k < hcdim; ++k) {
            const float v = h[size_t(t) * hcdim + k];
            acc += v * fn[size_t(m) * hcdim + k]; ss += v * v;
        }
        mixes[size_t(t) * 32 + m] = acc * sycl::rsqrt(ss / float(hcdim) + eps);
    });
}
void hc_split(const float* mixes, const float* scale, const float* base, int hc, int iters, float eps, int T,
              float* pre, float* post, float* comb, sycl::queue& s) {
    if (hc <= 0 || hc > 8 || 2 * hc + hc * hc > 32) throw std::invalid_argument("ds4 hyper-connections: invalid mix shape");
    const hc_detail::Split args{scale, base, hc, iters, eps, pre, post, comb};
    if (hc_detail::split(mixes, T, args, s)) return;
    parallel(s, size_t(T), [=](size_t t) {
        const float* mx = mixes + t * 32;
        float c[8][8];
        for (int j = 0; j < hc; ++j) {
            pre[t * 8 + j] = 1.f / (1.f + sycl::exp(-(mx[j] * scale[0] + base[j]))) + eps;
            post[t * 8 + j] = 2.f / (1.f + sycl::exp(-(mx[hc + j] * scale[1] + base[hc + j])));
            float m = -FLT_MAX;
            for (int k = 0; k < hc; ++k) {
                c[j][k] = mx[2 * hc + j * hc + k] * scale[2] + base[2 * hc + j * hc + k];
                m = sycl::fmax(m, c[j][k]);
            }
            float sum = 0.f;
            for (int k = 0; k < hc; ++k) { c[j][k] = sycl::exp(c[j][k] - m); sum += c[j][k]; }
            for (int k = 0; k < hc; ++k) c[j][k] = c[j][k] / sum + eps;
        }
        for (int it = 0; it < sycl::max(iters, 1); ++it) {
            if (it) for (int j = 0; j < hc; ++j) {
                float sum = 0.f;
                for (int k = 0; k < hc; ++k) sum += c[j][k];
                for (int k = 0; k < hc; ++k) c[j][k] /= sum + eps;
            }
            for (int k = 0; k < hc; ++k) {
                float sum = 0.f;
                for (int j = 0; j < hc; ++j) sum += c[j][k];
                for (int j = 0; j < hc; ++j) c[j][k] /= sum + eps;
            }
        }
        for (int j = 0; j < hc; ++j) for (int k = 0; k < hc; ++k) comb[t * 64 + size_t(j) * hc + k] = c[j][k];
    });
}
void hc_mix_split(const float* h, const float* fn, int hcdim, int n_mix, int T, float eps, float* mixes,
                  const float* scale, const float* base, int hc, int iters, float hc_eps, float* pre,
                  float* post, float* comb, sycl::queue& s) {
    hc_mix(h, fn, hcdim, n_mix, T, eps, mixes, s);
    hc_split(mixes, scale, base, hc, iters, hc_eps, T, pre, post, comb, s);
}
void hc_mix_split(const float* h, const float* fn, int hcdim, int n_mix, int T, float eps, float* mixes, unsigned* cnt,
                  const float* scale, const float* base, int hc, int iters, float hc_eps, float* pre,
                  float* post, float* comb, sycl::queue& s) {
    const hc_detail::Split args{scale, base, hc, iters, hc_eps, pre, post, comb};
    if (hc_detail::mix(h, fn, hcdim, n_mix, T, eps, mixes, s, cnt, &args)) return;
    hc_mix_split(h, fn, hcdim, n_mix, T, eps, mixes, scale, base, hc, iters, hc_eps, pre, post, comb, s);
    if (cnt) s.memset(cnt, 0, size_t(T) * sizeof(unsigned));
}
void hc_pre_norm(const float* h, const float* mix, const float* w, int dim, int hc, int T, float eps, float* x, sycl::queue& s) {
    if (hc_detail::pre_norm(h, mix, w, dim, hc, T, eps, x, s)) return;
    parallel(s, size_t(T) * dim, [=](size_t i) {
        const size_t t = i / dim, e = i % dim;
        float v = 0.f;
        for (int j = 0; j < hc; ++j) v += mix[t * 8 + j] * h[(t * hc + j) * dim + e];
        x[i] = v;
    });
    if (w) { Finish f; f.w = w; f.eps = eps; finish(x, T, dim, dim, f, s); }
}
void hc_post(const float* y, const float* res, const float* post, const float* comb, int dim, int hc, int T,
             float* out, sycl::queue& s) {
    parallel(s, size_t(T) * hc * dim, [=](size_t i) {
        const size_t e = i % dim, k = (i / dim) % hc, t = i / (size_t(hc) * dim);
        float v = post[t * 8 + k] * y[t * dim + e];
        for (int j = 0; j < hc; ++j) v += comb[t * 64 + size_t(j) * hc + k] * res[(t * hc + j) * dim + e];
        out[i] = v;
    });
}
void hc_post_inplace(const float* y, float* h, const float* post, const float* comb, int dim, int hc, int T, sycl::queue& s) {
    if (hc > 8) throw std::invalid_argument("ds4 hyper-connections: too many streams");
    parallel(s, size_t(T) * dim, [=](size_t i) {
        const size_t t = i / dim, e = i % dim;
        float vals[8];
        for (int j = 0; j < hc; ++j) vals[j] = h[(t * hc + j) * dim + e];
        for (int k = 0; k < hc; ++k) {
            float v = post[t * 8 + k] * y[i];
            for (int j = 0; j < hc; ++j) v += comb[t * 64 + size_t(j) * hc + k] * vals[j];
            h[(t * hc + k) * dim + e] = v;
        }
    });
}
void hc_mix_scale(const float* h, int hcdim, int n_mix, int T, float eps, float* mixes, sycl::queue& s) {
    parallel(s, size_t(T), [=](size_t t) {
        float ss = 0.f;
        for (int k = 0; k < hcdim; ++k) { const float v = h[t * hcdim + k]; ss += v * v; }
        const float inv = sycl::rsqrt(ss / float(hcdim) + eps);
        for (int k = 0; k < n_mix; ++k) mixes[t * 32 + k] *= inv;
    });
}
void gemv_f16_rows(const uint16_t* W, const float* x, size_t ldx, int n, int k, int T, float alpha,
                   float* y, size_t ldy, sycl::queue& s) {
    if (T > 0 && n > 0 && gemv_warp_ready(s, W, x, ldx, k, 8)) {
        sycl_detail::require_in_order(s);
        if (gemv_warp_width(s) == 32) gemv_f16_warp<32>(W, x, ldx, n, k, T, alpha, y, ldy, s);
        else gemv_f16_warp<16>(W, x, ldx, n, k, T, alpha, y, ldy, s);
        return;
    }
    gemv_rows(W, x, ldx, n, k, T, alpha, y, ldy, s);
}
void gemv_f16(const uint16_t* W, const float* x, int n, int k, int T, float alpha, float* y, sycl::queue& s) {
    gemv_f16_rows(W, x, k, n, k, T, alpha, y, n, s);
}
void gemv_f32(const float* W, const float* x, int n, int k, int T, float* y, int ldy, sycl::queue& s) {
    if (T > 0 && n > 0 && gemv_warp_ready(s, W, x, size_t(k), k, 4)) {
        sycl_detail::require_in_order(s);
        if (gemv_warp_width(s) == 32) gemv_f32_warp<32>(W, x, n, k, T, y, size_t(ldy), s);
        else gemv_f32_warp<16>(W, x, n, k, T, y, size_t(ldy), s);
        return;
    }
    gemv_rows(W, x, size_t(k), n, k, T, 1.f, y, size_t(ldy), s);
}
void route(const float* logits, const float* bias, int n, int k, int T, float scale, int* idx, float* w,
           sycl::queue& s, const float* bias_vl, const VlBits* vl) {
    if (k < 0 || k > kPicks || k > n) throw std::invalid_argument("ds4 route: invalid pick count");
    const VlBits bits = vl ? *vl : VlBits{};
    const bool launched = route_groups(s, T, n, [=](sycl::nd_item<1> item, auto scores, auto biased) {
        const size_t t = item.get_group_linear_id();
        const float* b = bias_vl && ((bits.w[t >> 5] >> (t & 31)) & 1u) ? bias_vl : bias;
        route_group(item, scores, biased, logits + t * 1024, b, n, k, scale, idx + t * kPicks, w + t * kPicks);
    });
    if (!launched) parallel(s, size_t(T), [=](size_t t) {
        const float* b = bias_vl && ((bits.w[t >> 5] >> (t & 31)) & 1u) ? bias_vl : bias;
        route_row(logits + t * 1024, b, n, k, scale, idx + t * kPicks, w + t * kPicks);
    });
}
void swiglu(const float* g, const float* u, int n, float lim, float* h, sycl::queue& s) {
    parallel(s, size_t(n), [=](size_t i) {
        const float gv = lim > 0.f ? sycl::fmin(g[i], lim) : g[i];
        const float uv = lim > 0.f ? sycl::clamp(u[i], -lim, lim) : u[i];
        h[i] = gv / (1.f + sycl::exp(-gv)) * uv;
    });
}

void compress_win(const float* kv, const float* score, float* kv_state, float* score_state, int ratio, int p0, int T,
                  int d, const float* w, float eps, float* latent, sycl::queue& s) {
    if (compress_detail::compress(kv, score, kv_state, score_state, ratio, p0, T, d, w, eps, latent, s)) return;
    int g = 0;
    for (int t = 0; t < T; ++t) {
        const int slot = (p0 + t) % ratio;
        parallel(s, size_t(d), [=](size_t e) {
            kv_state[size_t(slot) * d + e] = kv[size_t(t) * d + e];
            score_state[size_t(slot) * d + e] = score[size_t(t) * d + e];
        });
        if ((p0 + t + 1) % ratio) continue;
        float* row = latent + size_t(g++) * d;
        parallel(s, size_t(d), [=](size_t e) {
            float m = -FLT_MAX, den = 0.f, v = 0.f;
            for (int j = 0; j < ratio; ++j) m = sycl::fmax(m, score_state[size_t(j) * d + e]);
            for (int j = 0; j < ratio; ++j) den += sycl::exp(score_state[size_t(j) * d + e] - m);
            for (int j = 0; j < ratio; ++j) v += kv_state[size_t(j) * d + e] *
                (sycl::exp(score_state[size_t(j) * d + e] - m) / den);
            row[e] = v;
        });
        Finish f; f.w = w; f.eps = eps;
        finish(row, 1, d, d, f, s);
    }
}
void index_score(const float* q, const float* w, const void* k, int heads, int hd, int T, int n,
                 const int* cand, int cand_stride, int blk, const Rows& lim, float* out, long long out_stride,
                 sycl::queue& s, int kfmt) {
    sycl_detail::require_in_order(s);
    if (T <= 0 || n <= 0) return;
    if (hd == 128 && heads > 0 && heads <= 32) {
        const auto* kc = static_cast<const uint8_t*>(k);
        if (kfmt == kKvFp4p &&
            index_score_group<kKvFp4p>(q, w, kc, heads, T, n, cand, cand_stride, blk, lim, out, out_stride, s)) return;
        if (kfmt == kKvBf16 &&
            index_score_group<kKvBf16>(q, w, kc, heads, T, n, cand, cand_stride, blk, lim, out, out_stride, s)) return;
    }
    parallel(s, size_t(T) * n, [=](size_t z) {
        const int t = int(z / n), i = int(z % n);
        int ln, lk; rows_at(lim, t, ln, lk);
        int p = i;
        if (cand) p = i < lk * blk ? cand[size_t(t) * cand_stride + i / blk] * blk + i % blk : -1;
        float result = -std::numeric_limits<float>::infinity();
        if (p >= 0 && p < ln) {
            const uint8_t* kr = static_cast<const uint8_t*>(k) + size_t(p) * kv_row_bytes(kfmt, hd);
            result = 0.f;
            for (int h = 0; h < heads; ++h) {
                float dot = 0.f;
                for (int e = 0; e < hd; ++e) dot += q[(size_t(t) * heads + h) * hd + e] * kv_value(kr, kfmt, hd, e);
                result += sycl::fmax(dot, 0.f) * w[size_t(t) * 64 + h];
            }
        }
        out[static_cast<long long>(t) * out_stride + i] = result;
    });
}
void topk(const float* vals, long long stride, int T, const Rows& r, const int* cand, int cand_stride, int blk,
          int* out, int out_stride, sycl::queue& s) {
    sycl_detail::require_in_order(s);
    if (T <= 0) return;
    if (topk_group(vals, stride, T, r, cand, cand_stride, blk, out, out_stride, s)) return;
    parallel(s, size_t(T), [=](size_t t) {
        int n, k; rows_at(r, int(t), n, k);
        if (n <= 0 || k <= 0) return;
        const float* v = vals + static_cast<long long>(t) * stride;
        int* dst = out + t * out_stride;
        const int* cd = cand ? cand + t * cand_stride : nullptr;
        if (k >= n) {
            for (int i = 0; i < n; ++i) dst[i] = cd ? cd[i / blk] * blk + i % blk : i;
            return;
        }
        uint32_t prefix = 0, mask = 0;
        int remaining = k;
        // Radix selection preserves upstream's ordering: strictly greater values in
        // position order, followed by the lowest positions equal to the threshold.
        for (int shift = 24; shift >= 0; shift -= 8) {
            unsigned hist[256] = {};
            for (int i = 0; i < n; ++i) {
                const uint32_t u = ord_key(v[i]);
                if ((u & mask) == prefix) ++hist[(u >> shift) & 255u];
            }
            int cum = 0;
            for (int digit = 255; digit >= 0; --digit) {
                if (cum + int(hist[digit]) >= remaining) {
                    prefix |= uint32_t(digit) << shift; mask |= 255u << shift;
                    remaining -= cum; break;
                }
                cum += int(hist[digit]);
            }
        }
        int greater = 0, equal = k - remaining;
        for (int i = 0; i < n; ++i) {
            const uint32_t u = ord_key(v[i]);
            const int p = cd ? cd[i / blk] * blk + i % blk : i;
            if (u > prefix) dst[greater++] = p;
            else if (u == prefix && remaining > 0) { dst[equal++] = p; --remaining; }
        }
    });
}
void cand_blocks(const float* score, long long stride, int T, const Rows& r, int blk, float* bm, long long bm_stride,
                 int* out, int out_stride, sycl::queue& s) {
    Rows rb = r;
    int max_blocks = 0;
    if (r.gen) { rb.gen = 4; rb.blk = blk; max_blocks = (int((r.p0 + T) / r.R) + blk - 1) / blk; }
    else for (int t = 0; t < T; ++t) {
        rb.n[t] = (r.n[t] + blk - 1) / blk;
        max_blocks = std::max(max_blocks, rb.n[t]);
    }
    parallel(s, size_t(T) * max_blocks, [=](size_t z) {
        const int t = int(z / max_blocks), b = int(z % max_blocks);
        int n, k; rows_at(r, t, n, k);
        if (b >= (n + blk - 1) / blk) return;
        float m = -std::numeric_limits<float>::infinity();
        for (int j = b * blk; j < sycl::min(n, (b + 1) * blk); ++j)
            m = sycl::fmax(m, score[static_cast<long long>(t) * stride + j]);
        if (b == (n - 1) / blk) m = std::numeric_limits<float>::infinity();
        bm[static_cast<long long>(t) * bm_stride + b] = m;
    });
    topk(bm, bm_stride, T, rb, nullptr, 0, 1, out, out_stride, s);
}
void sparse_attn(const float* q, const uint16_t* win, const void* ckv, const int* sel, int sel_stride,
                 const AttnRows& a, const float* sink, int heads, int d, int T, float scale, float* out,
                 sycl::queue& s, int ckv_fmt, int heads_per_block, const AttnScratch* scr) {
    sycl_detail::require_in_order(s);
    if (T <= 0 || heads <= 0 || d <= 0) return;
    // Upstream's BF16 rows and few-query packed rows (decoded once a query into the scratch) take sparse_attn_kernel.
    if (const int width = d == 512 ? sparse_rows_width(s) : 0) {
        const uint16_t* rows = nullptr;
        bool contig = false;
        if (ckv_fmt == kKvBf16) rows = static_cast<const uint16_t*>(ckv);
        else if (ckv_fmt == kKvFp4 && scr && scr->rows && sel_stride > 0 && T <= scr->max_t &&
                 sel_stride <= scr->cap && (heads_per_block == -1 || (heads_per_block == 0 && T <= kAttnScratchT))) {
            unpack_selected(ckv, sel, sel_stride, a, T, d, scr->rows, size_t(scr->cap), s);
            rows = scr->rows;
            contig = true;
        }
        if (rows) {
            const int stride = contig ? scr->cap : sel_stride;
            if (width == 32 && contig) sparse_rows<true, 32>(q, win, rows, sel, stride, a, sink, heads, T, scale, out, s);
            else if (width == 32) sparse_rows<false, 32>(q, win, rows, sel, stride, a, sink, heads, T, scale, out, s);
            else if (contig) sparse_rows<true, 16>(q, win, rows, sel, stride, a, sink, heads, T, scale, out, s);
            else sparse_rows<false, 16>(q, win, rows, sel, stride, a, sink, heads, T, scale, out, s);
            return;
        }
    }
    if (sparse_attn_tiled(q, win, ckv, sel, sel_stride, a, sink, heads, d, T, scale,
                          out, s, ckv_fmt, heads_per_block, scr)) return;
    // One work group per query/head. The reduction spans the group regardless of
    // subgroup width; each item owns a strided subset of output components.
    const auto device = s.get_device();
    const size_t local = std::min({size_t(d), device.get_info<sycl::info::device::max_work_group_size>(),
                                  device.get_info<sycl::info::device::max_work_item_sizes<1>>()[0]});
    sycl_detail::launch(s, sycl::nd_range<1>(sycl::range<1>(size_t(T) * heads * local), sycl::range<1>(local)),
                   [=](sycl::nd_item<1> item) {
        const size_t group = item.get_group_linear_id(), lane = item.get_local_linear_id();
        const int h = int(group % heads), t = int(group / heads);
        const auto ar = attn_counts(a, t);
        const int w0 = ar.first, nw = ar.window, ns = ar.selected;
        const float* qr = q + group * d;
        float* result = out + group * d;
        for (size_t e = lane; e < size_t(d); e += local) result[e] = 0.f;
        float m = sink[h], den = 1.f;
        for (int j = 0; j < nw + ns; ++j) {
            const int fmt = j < nw ? kKvBf16 : ckv_fmt;
            const void* row = j < nw ? static_cast<const void*>(win + size_t(w0 + j) * d) :
                static_cast<const void*>(static_cast<const uint8_t*>(ckv) +
                    size_t(sel[size_t(t) * sel_stride + j - nw]) * kv_row_bytes(ckv_fmt, d));
            float dot = 0.f;
            for (size_t e = lane; e < size_t(d); e += local) dot += qr[e] * kv_value(row, fmt, d, int(e));
            dot = sycl::reduce_over_group(item.get_group(), dot, sycl::plus<float>()) * scale;
            const float next = sycl::fmax(m, dot), old_scale = sycl::exp(m - next), weight = sycl::exp(dot - next);
            for (size_t e = lane; e < size_t(d); e += local)
                result[e] = result[e] * old_scale + weight * kv_value(row, fmt, d, int(e));
            den = den * old_scale + weight; m = next;
        }
        for (size_t e = lane; e < size_t(d); e += local) result[e] /= den;
    });
}
void ring_to_ext(const uint8_t* ring, uint16_t* ext, int64_t p0, int n, int d, sycl::queue& s) {
    parallel(s, size_t(n) * d, [=](size_t z) {
        const int r = int(z / d), e = int(z % d);
        const int64_t p = p0 - n + r;
        ext[z] = p < 0 ? 0 : bf16_rn(kv_value(ring + size_t(p % n) * kv_row_bytes(kKvFp8, d), kKvFp8, d, e));
    });
}
void ext_to_ring(const uint16_t* ext, uint8_t* ring, int64_t p0, int n, int T, int d, sycl::queue& s, uint8_t* backup) {
    if (d % 32) throw std::invalid_argument("ds4 ring: incomplete FP8 block");
    if (finish_detail::ext_to_ring(ext, ring, p0, n, T, d, s, backup)) return;
    const size_t rb = kv_row_bytes(kKvFp8, d);
    parallel(s, size_t(T), [=](size_t t) {
        if (T - int(t) > n) return;
        uint8_t* row = ring + size_t((p0 + t) % n) * rb;
        if (backup) for (size_t j = 0; j < rb; ++j) backup[t * rb + j] = row[j];
        for (int start = 0; start < d; start += 32) {
            uint32_t a = 0;
            for (int j = 0; j < 32; ++j) a = sycl::max(a,
                sycl::bit_cast<uint32_t>(bf16_value(ext[(n + t) * d + start + j])) & 0x7fffffffu);
            const int k = a == 0 ? 0 : int(a >> 23) - 135 + ((a & 0x7fffffu) > 0x600000u);
            for (int j = 0; j < 32; ++j) row[start + j] = e4m3_code(bf16_value(ext[(n + t) * d + start + j]), k);
            row[d + start / 32] = uint8_t(k + 127);
        }
    });
}
void ring_restore(uint8_t* ring, const uint8_t* backup, int64_t p0, int n, int keep, int T, int d, sycl::queue& s) {
    const size_t rb = kv_row_bytes(kKvFp8, d);
    if (T <= keep) return;
    parallel(s, size_t(T - keep) * rb, [=](size_t z) {
        const size_t t = keep + z / rb, e = z % rb;
        ring[size_t((p0 + t) % n) * rb + e] = backup[t * rb + e];
    });
}
void ring_restore_batch(const RingBatch& b, int64_t p0, int n, int keep, int T, int d, sycl::queue& s) {
    for (int i = 0; i < b.n; ++i) ring_restore(b.ring[i], b.bak[i], p0, n, keep, T, d, s);
}
void engram_dequant(const uint8_t* rows, const uint8_t* scales, int n, int hd, float* out, sycl::queue& s, int fmt) {
    parallel(s, size_t(n) * hd, [=](size_t z) {
        const size_t r = z / hd;
        out[z] = engram_value(rows + r * hd, fmt ? nullptr : scales + r * (hd / 32), int(z % hd), fmt);
    });
}
void engram_dequant_tok(const uint8_t* src, size_t per, int cols, int T, int hd, float* out, sycl::queue& s, int fmt) {
    parallel(s, size_t(T) * cols * hd, [=](size_t z) {
        const size_t r = z / hd, t = r / cols, c = r % cols;
        out[z] = engram_value(src + t * per + c * hd, fmt ? nullptr : src + t * per + size_t(cols) * hd + c * (hd / 32),
                              int(z % hd), fmt);
    });
}
void engram_gate(float* h, const float* kv, const float* w, int hc, int dim, int T, float eps, sycl::queue& s) {
    parallel(s, size_t(T) * hc, [=](size_t z) {
        const size_t t = z / hc, k = z % hc;
        float* row = h + z * dim;
        const float* key = kv + (t * (hc + 1) + k) * dim;
        const float* value = kv + (t * (hc + 1) + hc) * dim;
        float sh = 0.f, sk = 0.f, sd = 0.f;
        for (int e = 0; e < dim; ++e) { sh += row[e] * row[e]; sk += key[e] * key[e]; sd += row[e] * w[k * dim + e] * key[e]; }
        const float dot = sd * sycl::rsqrt(sh / float(dim) + eps) * sycl::rsqrt(sk / float(dim) + eps) * sycl::rsqrt(float(dim));
        const float sq = sycl::copysign(sycl::sqrt(sycl::fmax(sycl::fabs(dot), 1e-6f)), dot);
        const float gate = 1.f / (1.f + sycl::exp(-sq));
        for (int e = 0; e < dim; ++e) row[e] += gate * value[e];
    });
}
void gather_rows(const float* x, const Gather& g, int n, int cols, float* out, sycl::queue& s) {
    parallel(s, size_t(n) * cols, [=](size_t z) { out[z] = x[size_t(g.src[z / cols]) * cols + z % cols]; });
}
void combine(const float* ed, const float* cpu, const float* sh, const Combine& c, int T, int dim, float* out, sycl::queue& s) {
    parallel(s, size_t(T) * dim, [=](size_t z) {
        const size_t t = z / dim, e = z % dim;
        float v = 0.f;
        for (int j = 0; j < c.n[t]; ++j) v += c.w[t][j] * ed[size_t(c.row[t][j]) * dim + e];
        if (cpu && c.cpu[t]) v += cpu[z];
        if (sh) v += sh[z];
        out[z] = v;
    });
}

void route_publish(const float* logits, const float* bias, int n, int k, int T, float scale,
                   const unsigned long long* vram, const float* x, int dim, int* idx, float* w,
                   unsigned long long* plan_ptr, Req* req, unsigned seq, sycl::queue& s,
                   const float* bias_vl, unsigned vl, int na0, int na1) {
    if (T > kWin || dim > kMaxDim || k < 0 || k > kPicks || k > n)
        throw std::invalid_argument("ds4 route_publish: invalid request dimensions");
    const auto publish = [=](size_t t) {
        const int* ids = idx + t * kPicks;
        const float* weights = w + t * kPicks;
        bool cpu = false;
        for (int i = 0; i < k; ++i) {
            plan_ptr[t * kPicks + i] = vram[ids[i]];
            cpu |= vram[ids[i]] == 0;
            if (req) { req->ids[t][i] = ids[i]; req->w[t][i] = weights[i]; }
        }
        return cpu;
    };
    const bool launched = route_groups(s, T, n, [=](sycl::nd_item<1> item, auto scores, auto biased) {
        const size_t t = item.get_group_linear_id(), lane = item.get_local_linear_id();
        const float* b = bias_vl && ((vl >> t) & 1u) ? bias_vl : bias;
        route_group(item, scores, biased, logits + t * 1024, b, n, k, scale, idx + t * kPicks, w + t * kPicks);
        bool cpu = lane == 0 ? publish(t) : false;
        cpu = sycl::group_broadcast(item.get_group(), cpu, 0);
        if (req && cpu)
            for (size_t e = lane; e < size_t(std::max(dim, 0)); e += item.get_local_range(0))
                req->x[t][e] = x[t * dim + e];
    });
    if (!launched) parallel(s, size_t(T), [=](size_t t) {
        const float* b = bias_vl && ((vl >> t) & 1u) ? bias_vl : bias;
        route_row(logits + t * 1024, b, n, k, scale, idx + t * kPicks, w + t * kPicks);
        const bool cpu = publish(t);
        if (req && cpu) for (int e = 0; e < dim; ++e) req->x[t][e] = x[t * dim + e];
    });
    if (req && system_publish(s)) {
        // Upstream publishes the request from the device. Where the device reports system-scope fences and
        // atomics, a one-item kernel behind the routing kernel writes the counts and releases the sequence to
        // the host; this spares a host task's round trip.
        sycl_detail::single(s, [=] {
            req->na[0] = na0; req->na[1] = na1; req->T = T;
            // The system-scope fence orders the request ahead of the flag for the host; the flag itself is an
            // atomic on the host allocation (the device need not report system-scope atomic operations).
            sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
            sycl::atomic_ref<unsigned, sycl::memory_order::relaxed, sycl::memory_scope::device,
                             sycl::access::address_space::generic_space>(req->seq).store(seq);
        });
        return;
    }
    if (req) s.submit([=](sycl::handler& h) {
        // Completion of the preceding kernel makes its mailbox writes visible to
        // this host task. The release store publishes the entire request at once.
        h.host_task([=] {
            req->na[0] = na0; req->na[1] = na1; req->T = T;
            std::atomic_ref<unsigned>(req->seq).store(seq, std::memory_order_release);
        });
    });
}
void route_ahead(const float* logits, const float* bias, int n, int m, int T, Req* req, int off, sycl::queue& s) {
    if (T > kWin || off < 0 || off + m > kAhead) throw std::invalid_argument("ds4 route_ahead: invalid request dimensions");
    const bool launched = route_groups(s, T, n, [=](sycl::nd_item<1> item, auto scores, auto biased) {
        const size_t t = item.get_group_linear_id();
        route_group(item, scores, biased, logits + t * 1024, bias, n, m, 1.f, req->ahead[t] + off, nullptr);
    });
    if (!launched) parallel(s, size_t(T), [=](size_t t) {
        route_row(logits + t * 1024, bias, n, m, 1.f, req->ahead[t] + off, nullptr);
    });
}
void zc_stage(unsigned long long* plan, const int* idx, const unsigned long long* zc_tab, int k,
              uint8_t* stage, size_t stride, size_t bytes, sycl::queue& s) {
    parallel(s, size_t(k) * bytes, [=](size_t z) {
        const size_t i = z / bytes, e = z % bytes;
        if (!plan[i] && zc_tab[idx[i]]) stage[i * stride + e] = reinterpret_cast<const uint8_t*>(zc_tab[idx[i]])[e];
    });
    parallel(s, size_t(k), [=](size_t i) {
        if (!plan[i] && zc_tab[idx[i]]) plan[i] = reinterpret_cast<unsigned long long>(stage + i * stride);
    });
}
void wait_seq(const unsigned* seq_ptr, unsigned seq, sycl::queue& s) {
    sycl_detail::require_in_order(s);
    // The mailbox is host-accessible USM. Waiting on the host avoids requiring
    // system-scope atomics or forward progress between GPU work groups.
    const auto control = mailbox_for(s);
    s.submit([=](sycl::handler& h) { h.host_task([=] { wait_mailbox(seq_ptr, seq, false, control.get()); }); });
}
void moe_sort(const int* idx, int T, int k, const int* vs, int n_vs, int* bounds, int* row_tok, int* pos, sycl::queue& s) {
    parallel(s, size_t(n_vs), [=](size_t slot) {
        int count = 0;
        for (int t = 0; t < T; ++t) for (int i = 0; i < k; ++i) count += vs[idx[size_t(t) * kPicks + i]] == int(slot);
        bounds[slot + 1] = count;
    });
    sycl_detail::single(s, [=] { bounds[0] = 0; for (int i = 0; i < n_vs; ++i) bounds[i + 1] += bounds[i]; });
    parallel(s, size_t(n_vs), [=](size_t slot) {
        int r = bounds[slot];
        for (int t = 0; t < T; ++t) for (int i = 0; i < k; ++i) if (vs[idx[size_t(t) * kPicks + i]] == int(slot)) {
            row_tok[r] = t; pos[size_t(t) * kPicks + i] = r++;
        }
    });
}
void moe_combine(float* acc, const float* y, const int* pos, const float* w, int T, int k, int r0, int r1, int dim, sycl::queue& s) {
    parallel(s, size_t(T) * dim, [=](size_t z) {
        const size_t t = z / dim, e = z % dim;
        float v = acc[z];
        for (int i = 0; i < k; ++i) {
            const int r = pos[t * kPicks + i];
            if (r >= r0 && r < r1) v = sycl::fma(w[t * kPicks + i], y[size_t(r - r0) * dim + e], v);
        }
        acc[z] = v;
    });
}
void init_pre(float* pre, int T, sycl::queue& s) {
    parallel(s, size_t(T) * 8, [=](size_t i) { pre[i] = (i % 8) == 0 ? 1.f : 0.f; });
}
void hc_mean(const float* h, int T, int hc, int dim, float* out, long long out_stride, sycl::queue& s) {
    parallel(s, size_t(T) * dim, [=](size_t z) {
        const size_t t = z / dim, e = z % dim;
        float v = 0.f;
        for (int j = 0; j < hc; ++j) v += h[(t * hc + j) * dim + e];
        out[static_cast<long long>(t) * out_stride + e] = v / float(hc);
    });
}
void iota_rows(int* out, int T, int n, int stride, sycl::queue& s) {
    parallel(s, size_t(T) * n, [=](size_t z) { out[(z / n) * stride + z % n] = int(z % n); });
}
void markov_add(float* logits, const uint16_t* E, const uint16_t* H, const int* tok, int V, int R, sycl::queue& s) {
    parallel(s, size_t(V), [=](size_t v) {
        float sum = 0.f;
        for (int i = 0; i < R; ++i) sum += bf16_value(E[size_t(*tok) * R + i]) * f16_value(H[v * R + i]);
        logits[v] += sum;
    });
}
void markov_add_q8(float* logits, const uint16_t* E, const uint8_t* Hq, const int* tok, int V, int R, sycl::queue& s) {
    parallel(s, size_t(V), [=](size_t v) {
        const uint8_t* row = Hq + v * (R / 32) * 34;
        float sum = 0.f;
        for (int i = 0; i < R; ++i) {
            const uint8_t* block = row + size_t(i / 32) * 34;
            const float scale = f16_value(uint16_t(block[0] | uint16_t(block[1]) << 8));
            sum += bf16_value(E[size_t(*tok) * R + i]) * (scale * float(reinterpret_cast<const int8_t*>(block + 2)[i % 32]));
        }
        logits[v] += sum;
    });
}
void markov_share(float* logits, const uint16_t* H, const uint16_t* erow, int vn, int R, sycl::queue& s) {
    parallel(s, size_t(vn), [=](size_t v) {
        float sum = 0.f;
        for (int i = 0; i < R; ++i) sum += bf16_value(erow[i]) * f16_value(H[v * R + i]);
        logits[v] += sum;
    });
}
void argmax_share(const float* row, int vn, int v0, unsigned* slot, unsigned seq, sycl::queue& s) {
    sycl_detail::require_in_order(s);
    sycl_detail::single(s, [=] {
        float best = -std::numeric_limits<float>::infinity();
        int index = 0;
        for (int i = 0; i < vn; ++i) if (row[i] > best) { best = row[i]; index = i; }
        slot[0] = sycl::bit_cast<unsigned>(best); slot[1] = unsigned(v0 + index);
    });
    set_flag(slot + 2, seq, s);
}
void markov_pick(const unsigned* slots, int nm, unsigned seq, int* tok, const uint16_t* E, uint16_t* erow, int R, sycl::queue& s) {
    if (nm != 1) throw std::invalid_argument("ds4: this build supports one GPU; distributed markov heads are unavailable");
    wait_flags(slots + 2, 1, 4, seq, s);
    parallel(s, size_t(R), [=](size_t i) {
        const int index = int(slots[1]);
        if (!i) *tok = index;
        erow[i] = E[size_t(index) * R + i];
    });
}
void confidence(const float* x, const uint16_t* E, const int* tok, const float* w, int T, int dim, int R, float* out, sycl::queue& s) {
    parallel(s, size_t(T), [=](size_t t) {
        float sum = 0.f;
        for (int i = 0; i < dim; ++i) sum += w[i] * x[t * dim + i];
        for (int i = 0; i < R; ++i) sum += w[dim + i] * bf16_value(E[size_t(tok[t]) * R + i]);
        out[t] = 1.f / (1.f + sycl::exp(-sum));
    });
}
void xfer_put(const Xfer& x, unsigned* flag, unsigned seq, unsigned* cnt, sycl::queue& s) {
    sycl_detail::require_in_order(s);
    if (x.n < 0 || x.n > 4) throw std::invalid_argument("xfer_put segment count");
    core::ReadbackSpan copies[4]{};
    for (int i = 0; i < x.n; ++i) copies[i] = {x.dst[i], x.src[i], x.bytes[i]};
    core::readback_async(s, std::span<const core::ReadbackSpan>(copies, size_t(x.n)));
    if (cnt) s.memset(cnt, 0, sizeof(unsigned));
    if (flag) set_flag(flag, seq, s);
}
void win_submit(sycl::queue& s) {
    // SYCL submissions are already enqueued; no vendor-specific flush is needed.
    (void)s;
}
void xfer_get(const Xfer& x, const unsigned* flag, unsigned seq, sycl::queue& s) {
    sycl_detail::require_in_order(s);
    if (flag) wait_flags(flag, 1, 1, seq, s);
    for (int i = 0; i < x.n; ++i) if (x.bytes[i]) core::copy_async(s, x.dst[i], x.src[i], x.bytes[i]);
}
void wait_flags(const unsigned* flags, int n, int stride, unsigned seq, sycl::queue& s) {
    sycl_detail::require_in_order(s);
    if (n <= 0) return;
    const auto control = mailbox_for(s);
    s.submit([=](sycl::handler& h) { h.host_task([=] {
        for (int i = 0; i < n; ++i) wait_mailbox(flags + size_t(i) * stride, seq, true, control.get());
    }); });
}
void set_flag(unsigned* flag, unsigned v, sycl::queue& s) {
    sycl_detail::require_in_order(s);
    s.submit([=](sycl::handler& h) { h.host_task([=] {
        std::atomic_ref<unsigned>(*flag).store(v, std::memory_order_release);
    }); });
}
void allreduce(float* x, int n, const ArGroup& a, int mi, unsigned seq, unsigned* cnt, sycl::queue& s) {
    (void)x; (void)n; (void)mi; (void)seq; (void)cnt; (void)s;
    if (a.nm != 1) throw std::invalid_argument("ds4: this build supports one GPU; all-reduce is unavailable");
}

}  // namespace strata::ds4
