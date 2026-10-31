// SPDX-FileCopyrightText: 2026 Peasant Smith and the Project Maya contributors
// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/main/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
#pragma once

#include "ds4_sycl.hpp"
#include "strata/core/per_device.hpp"
#include "../kernels/xe/device_target.hpp"
#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <vector>

namespace strata::ds4::hc_detail {

struct Caps {
    size_t group = 0, local = 0;
    std::vector<size_t> subgroups;
    bool device_acq_rel = false;
};
inline const Caps& caps(const sycl::queue& q) {
    const auto device = q.get_device();
    static core::PerDevice<Caps> cache;
    return cache.get(device, [&] {
        const auto orders = device.get_info<sycl::info::device::atomic_memory_order_capabilities>();
        const auto scopes = device.get_info<sycl::info::device::atomic_memory_scope_capabilities>();
        return Caps{std::min(device.get_info<sycl::info::device::max_work_group_size>(),
                             device.get_info<sycl::info::device::max_work_item_sizes<1>>()[0]),
                    size_t(device.get_info<sycl::info::device::local_mem_size>()),
                    device.get_info<sycl::info::device::sub_group_sizes>(),
                    std::find(orders.begin(), orders.end(), sycl::memory_order::acq_rel) != orders.end() &&
                    std::find(scopes.begin(), scopes.end(), sycl::memory_scope::device) != scopes.end()};
    });
}
inline int subgroup(const Caps& c, size_t group, bool fused = false, size_t extra_bytes = 0) {
    if (c.group < group) return 0;
    for (const int sg : {32, 16}) {
        const size_t bytes = (group / size_t(sg) + 1 + (fused ? 32 : 0)) * sizeof(float) +
                             (fused ? sizeof(unsigned) : 0) + extra_bytes;
        if (c.local >= bytes && std::find(c.subgroups.begin(), c.subgroups.end(), size_t(sg)) != c.subgroups.end())
            return sg;
    }
    return 0;
}

// The exact upstream launch bounds are also visible to the device compiler.
// They are selected only after reported WG, subgroup and local-memory checks.
template <int SG, int WG, class Body>
struct BoundBody {
    Body body;
    [[sycl::reqd_work_group_size(WG), sycl::reqd_sub_group_size(STRATA_SUB_GROUP(SG))]]
    void operator()(sycl::nd_item<1> item) const { body(item); }
};
template <int SG>
inline float warp_sum(sycl::sub_group group, float value) {
#pragma unroll
    for (int offset = SG / 2; offset > 0; offset >>= 1)
        value += sycl::permute_group_by_xor(group, value, offset);
    return value;
}

// Upstream ds4_kernels.cu block_sum: subgroup XOR, subgroup sums in local
// memory, then subgroup 0. SG32 preserves the original order exactly. SG16
// folds the extra subgroup sums in each lane before its final XOR reduction.
template <int SG, int WG>
inline float block_sum(sycl::nd_item<1> item, float value, sycl::local_accessor<float, 1> red) {
    const size_t tid = item.get_local_linear_id(), lane = tid % SG, warp = tid / SG;
    value = warp_sum<SG>(item.get_sub_group(), value);
    if (lane == 0) red[warp] = value;
    item.barrier(sycl::access::fence_space::local_space);
    if (warp == 0) {
        float sum = lane < WG / SG ? red[lane] : 0.f;
        for (size_t i = lane + SG; i < WG / SG; i += SG) sum += red[i];
        sum = warp_sum<SG>(item.get_sub_group(), sum);
        if (lane == 0) red[WG / SG] = sum;
    }
    item.barrier(sycl::access::fence_space::local_space);
    const float result = red[WG / SG];
    item.barrier(sycl::access::fence_space::local_space);
    return result;
}

struct Split {
    const float* scale = nullptr;
    const float* base = nullptr;
    int hc = 0, iters = 0;
    float eps = 0.f;
    float* pre = nullptr;
    float* post = nullptr;
    float* comb = nullptr;
};
template <int SG>
inline void split_body(sycl::sub_group group, const float* mx, size_t token, const Split& a) {
    const int lane = int(group.get_local_linear_id()), hc = a.hc;
    float* pre = a.pre + token * 8;
    float* post = a.post + token * 8;
    float* comb = a.comb + token * 64;
    if (lane < hc) {
        pre[lane] = 1.f / (1.f + sycl::exp(-(mx[lane] * a.scale[0] + a.base[lane]))) + a.eps;
        post[lane] = 2.f / (1.f + sycl::exp(-(mx[hc + lane] * a.scale[1] + a.base[hc + lane])));
    }
    if (hc == 4) {
        // Upstream hc_split_body: one lane per 4x4 matrix entry. The XOR
        // row/column distances stay 1/2 and 4/8 for both reported SG sizes.
        const bool on = lane < 16;
        float c = on ? mx[8 + lane] * a.scale[2] + a.base[8 + lane] : -FLT_MAX;
        auto row_sum = [&](float v) {
            v += sycl::permute_group_by_xor(group, v, 1);
            return v + sycl::permute_group_by_xor(group, v, 2);
        };
        auto col_sum = [&](float v) {
            v += sycl::permute_group_by_xor(group, v, 4);
            return v + sycl::permute_group_by_xor(group, v, 8);
        };
        float maximum = sycl::fmax(c, sycl::permute_group_by_xor(group, c, 1));
        maximum = sycl::fmax(maximum, sycl::permute_group_by_xor(group, maximum, 2));
        c = on ? sycl::exp(c - maximum) : 0.f;
        c = c / row_sum(c) + a.eps;
        c = on ? c : 0.f;
        c = c / (col_sum(c) + a.eps);
        for (int iteration = 1; iteration < a.iters; ++iteration) {
            c = c / (row_sum(c) + a.eps);
            c = c / (col_sum(c) + a.eps);
        }
        if (on) comb[lane] = c;
        return;
    }
    // Upstream lane-zero fallback for HC other than four.
    if (lane != 0) return;
    float c[8][8];
    for (int j = 0; j < hc; ++j) {
        float maximum = -FLT_MAX;
        for (int k = 0; k < hc; ++k) {
            c[j][k] = mx[2 * hc + j * hc + k] * a.scale[2] + a.base[2 * hc + j * hc + k];
            maximum = sycl::fmax(maximum, c[j][k]);
        }
        float sum = 0.f;
        for (int k = 0; k < hc; ++k) { c[j][k] = sycl::exp(c[j][k] - maximum); sum += c[j][k]; }
        for (int k = 0; k < hc; ++k) c[j][k] = c[j][k] / sum + a.eps;
    }
    for (int k = 0; k < hc; ++k) {
        float sum = 0.f;
        for (int j = 0; j < hc; ++j) sum += c[j][k];
        for (int j = 0; j < hc; ++j) c[j][k] /= sum + a.eps;
    }
    for (int iteration = 1; iteration < a.iters; ++iteration) {
        for (int j = 0; j < hc; ++j) {
            float sum = 0.f;
            for (int k = 0; k < hc; ++k) sum += c[j][k];
            for (int k = 0; k < hc; ++k) c[j][k] /= sum + a.eps;
        }
        for (int k = 0; k < hc; ++k) {
            float sum = 0.f;
            for (int j = 0; j < hc; ++j) sum += c[j][k];
            for (int j = 0; j < hc; ++j) c[j][k] /= sum + a.eps;
        }
    }
    for (int j = 0; j < hc; ++j) for (int k = 0; k < hc; ++k) comb[j * hc + k] = c[j][k];
}

inline sycl::float4 load4(const float* pointer) {
    sycl::float4 result;
    __builtin_memcpy(&result, __builtin_assume_aligned(pointer, 16), sizeof(result));
    return result;
}

// Upstream hc_mix_kernel / hc_mix_split_kernel, one 256-item work-group per
// (mix, token). These are algorithm launch shapes, gated by reported limits.
template <int SG, bool Fused>
void mix_kernel(const float* h, const float* fn, int hcdim, int n_mix, int tokens, float eps,
                float* mixes, unsigned* counters, const Split& split, sycl::queue& q) {
    constexpr int WG = 256;
    q.submit([&](sycl::handler& handler) {
        sycl::local_accessor<float, 1> red(sycl::range<1>(WG / SG + 1), handler);
        sycl::local_accessor<float, 1> smx(sycl::range<1>(Fused ? 32 : 1), handler);
        sycl::local_accessor<unsigned, 1> last(sycl::range<1>(1), handler);
        auto body = [=](sycl::nd_item<1> item) {
            const size_t tid = item.get_local_linear_id(), index = item.get_group_linear_id();
            const size_t token = index / size_t(n_mix), mix = index % size_t(n_mix);
            const float* x = h + token * hcdim;
            const float* w = fn + mix * hcdim;
            float acc = 0.f, ss = 0.f;
            for (size_t k = tid * 4; k < size_t(hcdim); k += WG * 4) {
                const auto hv = load4(x + k), wv = load4(w + k);
                acc += hv.x() * wv.x() + hv.y() * wv.y() + hv.z() * wv.z() + hv.w() * wv.w();
                ss += hv.x() * hv.x() + hv.y() * hv.y() + hv.z() * hv.z() + hv.w() * hv.w();
            }
            acc = block_sum<SG, WG>(item, acc, red);
            ss = block_sum<SG, WG>(item, ss, red);
            if (tid == 0) {
                mixes[token * 32 + mix] = acc * sycl::rsqrt(ss / float(hcdim) + eps);
                if constexpr (Fused) {
                    // The release/acquire RMW chain publishes every mix to the
                    // last group, replacing upstream threadfence + atomicAdd.
                    sycl::atomic_ref<unsigned, sycl::memory_order::acq_rel, sycl::memory_scope::device,
                                     sycl::access::address_space::global_space> count(counters[token]);
                    last[0] = count.fetch_add(1u) == unsigned(n_mix - 1) ? 1u : 0u;
                }
            }
            if constexpr (Fused) {
                item.barrier(sycl::access::fence_space::global_and_local);
                if (!last[0]) return; // Uniform for the entire work-group.
                if (tid < SG) {
                    for (size_t i = tid; i < 32; i += SG)
                        smx[i] = i < size_t(n_mix) ? mixes[token * 32 + i] : 0.f;
                    sycl::group_barrier(item.get_sub_group());
                    split_body<SG>(item.get_sub_group(),
                        smx.template get_multi_ptr<sycl::access::decorated::no>().get(), token, split);
                }
                if (tid == 0) {
                    sycl::atomic_ref<unsigned, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                     sycl::access::address_space::global_space> count(counters[token]);
                    count.store(0u);
                }
            }
        };
        handler.parallel_for(sycl::nd_range<1>(size_t(tokens) * n_mix * WG, WG), BoundBody<SG, WG, decltype(body)>{body});
    });
}
inline bool mix(const float* h, const float* fn, int hcdim, int n_mix, int tokens, float eps,
                float* mixes, sycl::queue& q, unsigned* counters = nullptr, const Split* split = nullptr) {
    sycl_detail::require_in_order(q);
    // Leave invalid split shapes to the existing public validation, even for
    // an empty token batch. A zero mix count must not skip a requested split.
    if (split && (split->hc <= 0 || split->hc > 4 ||
                  2 * split->hc + split->hc * split->hc > n_mix)) return false;
    if (tokens <= 0 || n_mix <= 0) return true;
    if (hcdim <= 0 || hcdim % 4 || n_mix > 32 || (reinterpret_cast<uintptr_t>(h) % 16) ||
        (reinterpret_cast<uintptr_t>(fn) % 16)) return false;
    const bool fused = split != nullptr;
    const auto& c = caps(q);
    // Non-fused instantiations still declare the one-element dummy accessors.
    const int sg = subgroup(c, 256, fused, fused ? 0 : sizeof(float) + sizeof(unsigned));
    if (!sg || (fused && (!counters || !c.device_acq_rel))) return false;
    const Split args = fused ? *split : Split{};
    if (sg == 32) {
        if (fused) mix_kernel<32, true>(h, fn, hcdim, n_mix, tokens, eps, mixes, counters, args, q);
        else mix_kernel<32, false>(h, fn, hcdim, n_mix, tokens, eps, mixes, nullptr, args, q);
    } else {
        if (fused) mix_kernel<16, true>(h, fn, hcdim, n_mix, tokens, eps, mixes, counters, args, q);
        else mix_kernel<16, false>(h, fn, hcdim, n_mix, tokens, eps, mixes, nullptr, args, q);
    }
    return true;
}

template <int SG>
void split_kernel(const float* mixes, int tokens, const Split& args, sycl::queue& q) {
    auto body = [=](sycl::nd_item<1> item) {
        const size_t token = item.get_group_linear_id();
        split_body<SG>(item.get_sub_group(), mixes + token * 32, token, args);
    };
    q.parallel_for(sycl::nd_range<1>(size_t(tokens) * SG, SG), BoundBody<SG, SG, decltype(body)>{body});
}
inline bool split(const float* mixes, int tokens, const Split& args, sycl::queue& q) {
    sycl_detail::require_in_order(q);
    if (tokens <= 0) return true;
    const auto& c = caps(q);
    // One subgroup, no local scratch. The existing API validates the HC shape.
    for (const int sg : {32, 16}) if (c.group >= size_t(sg) &&
        std::find(c.subgroups.begin(), c.subgroups.end(), size_t(sg)) != c.subgroups.end()) {
        if (sg == 32) split_kernel<32>(mixes, tokens, args, q);
        else split_kernel<16>(mixes, tokens, args, q);
        return true;
    }
    return false;
}

// Upstream hc_pre_norm_kernel: one 1024-item group/token, at most eight
// retained values per item. No intermediate x write or second norm launch.
template <int SG>
void pre_norm_kernel(const float* h, const float* mix, const float* w, int dim, int hc, int tokens,
                     float eps, float* x, sycl::queue& q) {
    constexpr int WG = 1024;
    q.submit([&](sycl::handler& handler) {
        sycl::local_accessor<float, 1> red(sycl::range<1>(WG / SG + 1), handler);
        auto body = [=](sycl::nd_item<1> item) {
            const size_t token = item.get_group_linear_id(), tid = item.get_local_linear_id();
            const float* row = h + token * hc * dim;
            const float* pre = mix + token * 8;
            float r[8], ss = 0.f;
            size_t count = 0;
            for (size_t e = tid; e < size_t(dim); e += WG, ++count) {
                float a = 0.f;
                for (int stream = 0; stream < hc; ++stream) a += pre[stream] * row[size_t(stream) * dim + e];
                r[count] = a;
                ss += a * a;
            }
            const float inv = sycl::rsqrt(block_sum<SG, WG>(item, ss, red) / float(dim) + eps);
            count = 0;
            for (size_t e = tid; e < size_t(dim); e += WG, ++count)
                x[token * dim + e] = w ? r[count] * inv * w[e] : r[count];
        };
        handler.parallel_for(sycl::nd_range<1>(size_t(tokens) * WG, WG), BoundBody<SG, WG, decltype(body)>{body});
    });
}
inline bool pre_norm(const float* h, const float* mix, const float* w, int dim, int hc, int tokens,
                     float eps, float* x, sycl::queue& q) {
    sycl_detail::require_in_order(q);
    if (tokens <= 0) return true;
    if (dim <= 0 || dim > 8 * 1024 || hc <= 0 || hc > 8) return false;
    const int sg = subgroup(caps(q), 1024);
    if (!sg) return false;
    if (sg == 32) pre_norm_kernel<32>(h, mix, w, dim, hc, tokens, eps, x, q);
    else pre_norm_kernel<16>(h, mix, w, dim, hc, tokens, eps, x, q);
    return true;
}

} // namespace strata::ds4::hc_detail
