// SPDX-FileCopyrightText: 2026 Peasant Smith and the Project Maya contributors
// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/main/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
#pragma once

#include "ds4_kernels.cuh"
#include "ds4_numerics.cuh"
#include "strata/core/per_device.hpp"
#include "../kernels/xe/device_target.hpp"
#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

namespace strata::ds4::finish_detail {

// These are the upstream algorithm's launch shape and register bound, used
// only when the device reports support. Other shapes keep the scalar path.
constexpr int kThreads = 256;
constexpr int kValues = 32;

// A comma fold emits the upstream register iterations in ascending order.
// Each invocation has a constant index, without a loop-unroll optimization hint.
template <class F, size_t... I>
inline void static_for_impl(F&& fn, std::index_sequence<I...>) {
    (fn(std::integral_constant<int, static_cast<int>(I)>{}), ...);
}
template <int Count, class F>
inline void static_for(F&& fn) {
    static_for_impl(fn, std::make_index_sequence<static_cast<size_t>(Count)>{});
}

struct Caps {
    size_t group = 0, local = 0;
    std::vector<size_t> subgroups;
};
inline const Caps& caps(const sycl::queue& q) {
    const auto device = q.get_device();
    // Function-local initialization is lazy and independent of other translation units.
    static core::PerDevice<Caps> cache; // NOLINT(bugprone-dynamic-static-initializers)
    return cache.get(device, [&] {
        return Caps{std::min(device.get_info<sycl::info::device::max_work_group_size>(),
                             device.get_info<sycl::info::device::max_work_item_sizes<1>>()[0]),
                    size_t(device.get_info<sycl::info::device::local_mem_size>()),
                    device.get_info<sycl::info::device::sub_group_sizes>()};
    });
}
inline int subgroup(const Caps& c, bool norm) {
    if (c.group < kThreads) return 0;
    for (const int sg : {32, 16}) {
        const size_t exchange = sg == 16 ? kThreads : 1;
        const size_t bytes = exchange * sizeof(uint32_t) + (norm ? (kThreads / 32 + 1) * sizeof(float) : 0);
        if (c.local >= bytes && std::find(c.subgroups.begin(), c.subgroups.end(), size_t(sg)) != c.subgroups.end())
            return sg;
    }
    return 0;
}
template <int SG, class Body>
struct BoundBody {
    Body body;
    // Inlined into the kernel as upstream's __global__ body is: left a call (the body is past the inliner's size
    // limit), the kernel copied its arguments to private memory and read Finish from there (R9700, HIP backend).
    [[sycl::reqd_work_group_size(kThreads), sycl::reqd_sub_group_size(STRATA_SUB_GROUP(SG))]] [[gnu::always_inline]]
    void operator()(sycl::nd_item<1> item) const { body(item); }
};

// Preserve the upstream logical 32-lane XOR order on an SG16 device too.
// Every item in the WG calls this helper: the offset-16 exchange uses LDS,
// with a second barrier before any caller can reuse the same scratch.
template <int SG, class T>
inline T xor32(sycl::nd_item<1> item, T value, int offset, sycl::local_accessor<uint32_t, 1> exchange) {
    if constexpr (SG == 16) {
        if (offset == 16) {
            const size_t tid = item.get_local_linear_id();
            exchange[tid] = sycl::bit_cast<uint32_t>(value);
            item.barrier(sycl::access::fence_space::local_space);
            const T other = sycl::bit_cast<T>(exchange[tid ^ 16u]);
            item.barrier(sycl::access::fence_space::local_space);
            return other;
        }
    }
    return sycl::permute_group_by_xor(item.get_sub_group(), value, offset);
}
template <int SG>
inline float warp_sum(sycl::nd_item<1> item, float value, sycl::local_accessor<uint32_t, 1> exchange) {
    static_for<5>([&](auto index) {
        constexpr int offset = 16 >> decltype(index)::value;
        value += xor32<SG>(item, value, offset, exchange);
    });
    return value;
}
template <int SG>
inline float block_sum(sycl::nd_item<1> item, float value, sycl::local_accessor<float, 1> red,
                        sycl::local_accessor<uint32_t, 1> exchange) {
    const size_t tid = item.get_local_linear_id(), lane = tid & 31u, warp = tid >> 5;
    value = warp_sum<SG>(item, value, exchange);
    if (lane == 0) red[warp] = value;
    item.barrier(sycl::access::fence_space::local_space);
    if constexpr (SG == 32) {
        // Upstream block_sum: only logical warp zero reduces the warp sums.
        if (warp == 0) {
            float sum = lane < kThreads / 32 ? red[lane] : 0.f;
            sum = warp_sum<SG>(item, sum, exchange);
            if (lane == 0) red[kThreads / 32] = sum;
        }
    } else {
        // All WG items participate so the SG16 virtual-warp exchange is uniform.
        float sum = tid < kThreads / 32 ? red[tid] : 0.f;
        sum = warp_sum<SG>(item, sum, exchange);
        if (tid == 0) red[kThreads / 32] = sum;
    }
    item.barrier(sycl::access::fence_space::local_space);
    const float result = red[kThreads / 32];
    item.barrier(sycl::access::fence_space::local_space);
    return result;
}
template <int SG>
inline float group_max(sycl::nd_item<1> item, float value, int group,
                        sycl::local_accessor<uint32_t, 1> exchange) {
    for (int offset = group / 2; offset > 0; offset >>= 1)
        value = sycl::fmax(value, xor32<SG>(item, value, offset, exchange));
    return value;
}
inline float pow2_ceil(float value) {
    const uint32_t bits = sycl::bit_cast<uint32_t>(value);
    const int exponent = int((bits >> 23) & 255u) + ((bits & 0x7fffffu) != 0u);
    return sycl::bit_cast<float>(uint32_t(exponent) << 23);
}
inline float e2m1_round(float value) {
    const float a = sycl::fabs(value);
    const float r = a <= .25f ? 0.f : a < .75f ? .5f : a <= 1.25f ? 1.f : a < 1.75f ? 1.5f :
                    a <= 2.5f ? 2.f : a < 3.5f ? 3.f : a <= 5.f ? 4.f : 6.f;
    return sycl::copysign(r, value);
}
inline uint32_t e2m1_code(float value) {
    const float a = sycl::fabs(value);
    const uint32_t m = a <= 1.5f ? uint32_t(a * 2.f) : a <= 2.f ? 4u : a <= 3.f ? 5u : a <= 4.f ? 6u : 7u;
    return m | ((sycl::bit_cast<uint32_t>(value) >> 31) << 3);
}
inline uint8_t e4m3_code(float value, int k = 0) {
    const uint32_t bits = sycl::bit_cast<uint32_t>(value), sign = (bits >> 24) & 128u, a = bits & 0x7fffffffu;
    if (!a) return uint8_t(sign);
    const int exponent = int(a >> 23) - 127 - k;
    if (exponent >= -6) return uint8_t(sign | (uint32_t(exponent + 7) << 3) | ((a >> 20) & 7u));
    if (exponent <= -18) return uint8_t(sign);
    return uint8_t(sign | (((a & 0x7fffffu) | 0x800000u) >> (14 - exponent)));
}
inline float quant_scale(float a, int quant) {
    // The reciprocal products are intentional: this is the upstream order.
    if (quant == 1) return pow2_ceil(sycl::fmax(a, 1e-4f) * (1.f / 448.f));
    if (quant == 2) return e4m3_round(sycl::fmax(a, 6.f * .001953125f) / 6.f);
    return pow2_ceil(sycl::fmax(a, 6.f * 1.1754943508222875e-38f) * (1.f / 6.f));
}
inline float quant_value(float value, float scale, int quant) {
    if (quant == 1) return e4m3_round(sycl::fmin(sycl::fmax(value / scale, -448.f), 448.f));
    return e2m1_round(sycl::fmin(sycl::fmax(value / scale, -6.f), 6.f));
}

// Upstream ds4_kernels.cu finish_kernel<kPack>, with its 256-item strided
// register rows, two-level XOR norm, adjacent RoPE and 16/32-value quantization.
template <int SG, bool Pack>
void finish_kernel(float* x, int vectors, int len, long long stride, Finish f, sycl::queue& q) {
    sycl_detail::submit(q, [&](sycl::handler& handler) {
        sycl::local_accessor<float, 1> red(sycl::range<1>(kThreads / 32 + 1), handler);
        sycl::local_accessor<uint32_t, 1> exchange(sycl::range<1>(SG == 16 ? kThreads : 1), handler);
        auto body = [=](sycl::nd_item<1> item) {
            const int tid = int(item.get_local_linear_id());
            const size_t vector = item.get_group_linear_id();
            float* row = x + static_cast<long long>(vector) * stride;
            float r[kValues], ss = 0.f;
            const int count = (len + kThreads - 1) / kThreads;
            for (int j = 0; j < count; ++j) {
                const int e = tid + j * kThreads;
                r[j] = e < len ? row[e] : 0.f;
                ss += r[j] * r[j];
            }
            if (f.w) {
                const float total = block_sum<SG>(item, ss, red, exchange);
                const float inv = sycl::rsqrt(total / float(len) + f.eps);
                for (int j = 0; j < count; ++j) {
                    const int e = tid + j * kThreads;
                    if (e < len) r[j] = r[j] * inv * f.w[e];
                }
            }
            if (f.rd > 0) {
                const int r0 = len - f.rd;
                const int pos = f.pos + int(vector / size_t(f.vpp)) * f.pos_step;
                for (int j = 0; j < count; ++j) {
                    const int e = tid + j * kThreads;
                    const float other = sycl::permute_group_by_xor(item.get_sub_group(), r[j], 1);
                    if (e >= r0 && e < len) {
                        const float angle = float(pos) * f.freqs[(e - r0) >> 1];
                        const float c = sycl::cos(angle), sn = f.inverse ? -sycl::sin(angle) : sycl::sin(angle);
                        if (((e - r0) & 1) == 0) r[j] = r[j] * c - other * sn;
                        else r[j] = other * sn + r[j] * c;
                    }
                }
            }
            uint8_t* packed = Pack ? f.pdst + static_cast<long long>(vector) * f.pdst_stride : nullptr;
            if (f.quant != 0 || Pack) {
                const int group = f.quant == 2 ? 16 : 32;
                for (int j = 0; j < count; ++j) {
                    const int e = tid + j * kThreads;
                    const float a = group_max<SG>(item, e < len ? sycl::fabs(r[j]) : 0.f, group, exchange);
                    float rounded = 0.f, scale = 1.f;
                    if (e < len) {
                        scale = quant_scale(a, f.quant);
                        rounded = quant_value(r[j], scale, f.quant);
                        r[j] = rounded * scale;
                    }
                    if constexpr (Pack) {
                        if (f.quant == 1) {
                            if (e < len) packed[e] = e4m3_code(rounded);
                        } else {
                            const uint32_t code = e2m1_code(rounded);
                            const uint32_t other = sycl::permute_group_by_xor(item.get_sub_group(), code, 1);
                            if (e < len && !(e & 1)) packed[e >> 1] = uint8_t(code | (other << 4));
                        }
                        if (e < len && e % group == 0)
                            packed[(f.quant == 1 ? len : len / 2) + e / group] = f.quant == 2 ?
                                e4m3_code(scale) : uint8_t(sycl::bit_cast<uint32_t>(scale) >> 23);
                    }
                }
            }
            uint16_t* dst = f.dst ? f.dst + static_cast<long long>(vector) * f.dst_stride : nullptr;
            for (int j = 0; j < count; ++j) {
                const int e = tid + j * kThreads;
                if (e >= len) continue;
                if (dst) dst[e] = bf16_rn(r[j]);
                else if (!Pack) row[e] = r[j];
            }
        };
        handler.parallel_for(sycl::nd_range<1>(size_t(vectors) * kThreads, kThreads), BoundBody<SG, decltype(body)>{body});
    });
}
inline bool finish(float* x, int vectors, int len, long long stride, const Finish& f, sycl::queue& q) {
    sycl_detail::require_in_order(q);
    if (len <= 0 || len > kValues * kThreads) return false;
    const int sg = subgroup(caps(q), true);
    if (!sg) return false;
    if (sg == 32) {
        if (f.pdst) finish_kernel<32, true>(x, vectors, len, stride, f, q);
        else finish_kernel<32, false>(x, vectors, len, stride, f, q);
    } else {
        if (f.pdst) finish_kernel<16, true>(x, vectors, len, stride, f, q);
        else finish_kernel<16, false>(x, vectors, len, stride, f, q);
    }
    return true;
}

// Upstream ds4_kernels.cu ext_to_ring_kernel. Ring inputs are already rounded
// BF16 cache values. Keep the integer-bit scale rule and the backup/overwritten-
// row contract separate from finish_kernel.
template <int SG>
void ext_to_ring_kernel(const uint16_t* ext, uint8_t* ring, int64_t p0, int n, int tokens, int dim,
                        sycl::queue& q, uint8_t* backup) {
    const size_t row_bytes = kv_row_bytes(kKvFp8, dim);
    sycl_detail::submit(q, [&](sycl::handler& handler) {
        sycl::local_accessor<uint32_t, 1> exchange(sycl::range<1>(SG == 16 ? kThreads : 1), handler);
        auto body = [=](sycl::nd_item<1> item) {
            const size_t token = item.get_group_linear_id(), tid = item.get_local_linear_id();
            if (tokens - int(token) > n) return; // Uniform WG skip; a later token overwrites this slot.
            uint8_t* row = ring + size_t((p0 + static_cast<int64_t>(token)) % n) * row_bytes;
            if (backup) {
                for (size_t i = tid; i < row_bytes; i += kThreads) backup[token * row_bytes + i] = row[i];
                item.barrier(sycl::access::fence_space::global_and_local);
            }
            const uint16_t* src = ext + (size_t(n) + token) * dim;
            for (size_t e0 = 0; e0 < size_t(dim); e0 += kThreads) {
                const size_t e = e0 + tid;
                const float value = e < size_t(dim) ? sycl_detail::bf16_value(src[e]) : 0.f;
                uint32_t a = sycl::bit_cast<uint32_t>(value) & 0x7fffffffu;
                static_for<5>([&](auto index) {
                    constexpr int offset = 16 >> decltype(index)::value;
                    a = sycl::max(a, xor32<SG>(item, a, offset, exchange));
                });
                const int k = a == 0 ? 0 : int(a >> 23) - 135 + ((a & 0x7fffffu) > 0x600000u);
                if (e >= size_t(dim)) continue;
                row[e] = e4m3_code(value, k);
                if ((e & 31u) == 0) row[size_t(dim) + e / 32] = uint8_t(k + 127);
            }
        };
        handler.parallel_for(sycl::nd_range<1>(size_t(tokens) * kThreads, kThreads), BoundBody<SG, decltype(body)>{body});
    });
}
inline bool ext_to_ring(const uint16_t* ext, uint8_t* ring, int64_t p0, int n, int tokens, int dim,
                        sycl::queue& q, uint8_t* backup) {
    sycl_detail::require_in_order(q);
    if (tokens == 0) return true;
    if (tokens < 0 || dim <= 0 || n <= 0) return false;
    const int sg = subgroup(caps(q), false);
    if (!sg) return false;
    if (sg == 32) ext_to_ring_kernel<32>(ext, ring, p0, n, tokens, dim, q, backup);
    else ext_to_ring_kernel<16>(ext, ring, p0, n, tokens, dim, q, backup);
    return true;
}

} // namespace strata::ds4::finish_detail
