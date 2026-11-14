// SPDX-FileCopyrightText: 2026 Peasant Smith and the Project Maya contributors
// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/main/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
#pragma once

#include "ds4_iq_decode.hpp"
#include "ds4_quant_sycl.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::ds4::gguf::iq::token_detail {
using namespace sycl_decode;

struct Caps { size_t group = 0, local = 0; std::vector<size_t> subgroups; };
const Caps& caps(const sycl::queue& q);

// Upstream ds4_iq.cu sub_dot: two exact 16-value integer sums, then their FP32 scale.
// Copy the representation at the checked 16-byte alignment instead of aliasing an int8 array as int4 objects.
inline sycl::int4 load_codes(const int8_t* p) {
    sycl::int4 words;
    __builtin_memcpy(&words, __builtin_assume_aligned(p, 16), sizeof(words));
    return words;
}
template <int T>
inline float sub_dot(const Sub& s, const sycl::int4& xa, const sycl::int4& xb) {
#pragma clang fp contract(off)
    using strata::kernels::xe::dp4a;
    int lo = dp4a(s.q[0], xa.x(), 0);
    lo = dp4a(s.q[1], xa.y(), lo);
    lo = dp4a(s.q[2], xa.z(), lo);
    lo = dp4a(s.q[3], xa.w(), lo);
    int hi = dp4a(s.q[4], xb.x(), 0);
    hi = dp4a(s.q[5], xb.y(), hi);
    hi = dp4a(s.q[6], xb.z(), hi);
    hi = dp4a(s.q[7], xb.w(), hi);
    float value = s.d * float(s.s0 * lo + s.s1 * hi);
    if constexpr (Ty<T>::MIN) {
        int sl = dp4a(0x01010101, xa.x(), 0);
        sl = dp4a(0x01010101, xa.y(), sl);
        sl = dp4a(0x01010101, xa.z(), sl);
        sl = dp4a(0x01010101, xa.w(), sl);
        int sh = dp4a(0x01010101, xb.x(), 0);
        sh = dp4a(0x01010101, xb.y(), sh);
        sh = dp4a(0x01010101, xb.z(), sh);
        sh = dp4a(0x01010101, xb.w(), sh);
        value -= s.m0 * float(sl) + s.m1 * float(sh);
    }
    return value;
}

inline float swiglu(float gate, float up, float limit) {
#pragma clang fp contract(off)
    if (limit > 0.0f) {
        gate = sycl::fmin(gate, limit);
        up = sycl::fmin(sycl::fmax(up, -limit), limit);
    }
    return (gate / (1.0f + sycl::exp(-gate))) * up;
}

inline bool group(const Args& a, int grp, const uint8_t*& blob, int& first, int& end) {
    if (!a.grp_ptr) {
        if (!a.plan_ptr[grp]) return false;
        blob = reinterpret_cast<const uint8_t*>(a.plan_ptr[grp]);
        first = grp;
        end = grp + 1;
    } else {
        if (grp >= *a.n_groups) return false;
        blob = reinterpret_cast<const uint8_t*>(a.grp_ptr[grp]);
        first = a.grp_start[grp];
        end = a.grp_start[grp + 1];
    }
    return true;
}

template <int SG, int RW>
inline float lanes_sum(sycl::sub_group sg, float value) {
#pragma unroll
    for (int offset = SG / (2 * RW); offset > 0; offset >>= 1)
        value += sycl::permute_group_by_xor(sg, value, offset);
    return value;
}

// Upstream 870aedd9 ds4_iq.cu gate_up_kernel<T,NE,1>, HIP nonpersistent grid.
// One subgroup owns one output row; a lane retains its quantization sub-block within each block.
template <int T, int NE, int SG>
void gate_up_kernel(const Args& a, int groups, size_t warps, sycl::queue& q) {
    constexpr int LPR = SG, BPS = kQK<T> / 32, STEP = LPR / BPS;   // the 32-value sub-blocks a block
    static_assert(LPR % BPS == 0);
    const size_t threads = warps * SG, tiles = size_t(a.n_ff) / warps;
    const size_t rb = row_bytes_t<T>(a.n_embd);
    sycl_detail::submit(q, [&](sycl::handler& h) {
        sycl::local_accessor<uint32_t, 1> tab(sycl::range<1>(Ty<T>::TAB > 0 ? Ty<T>::TAB : 1), h);
        sycl::local_accessor<int, 1> entries(sycl::range<1>(NE), h), inputs(sycl::range<1>(NE), h);
        h.parallel_for(sycl::nd_range<1>(size_t(groups) * tiles * threads, threads),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(STRATA_SUB_GROUP(SG))]] {
            const size_t tid = it.get_local_linear_id(), tile = it.get_group_linear_id() % tiles;
            const int grp = int(it.get_group_linear_id() / tiles);
            const uint8_t* blob;
            int first, end;
            if (!group(a, grp, blob, first, end)) return;
            if constexpr (Ty<T>::TAB > 0)
                for (size_t j = tid; j < size_t(Ty<T>::TAB); j += threads) tab[j] = table<T>()[j];
            const auto* grid = tab.template get_multi_ptr<sycl::access::decorated::no>().get();
            const int lane = int(tid % SG), row = int(tile * warps + tid / SG);
            const int ib = lane % BPS, first_block = lane / BPS;
            const int subblocks = a.n_embd / 32, blocks = subblocks / BPS;
            const uint8_t* g = blob + a.tr[0] + size_t(row) * rb;
            const uint8_t* u = blob + a.tr[1] + size_t(row) * rb;
            const auto sg = it.get_sub_group();
            for (int member = first; member < end; member += NE) {
                const int count = sycl::min(NE, end - member);
                it.barrier(sycl::access::fence_space::local_space);
                if (tid < size_t(count)) {
                    const int entry = a.grp_ptr ? a.grp_ent[member + int(tid)] : member + int(tid);
                    entries[tid] = entry;
                    inputs[tid] = a.grp_ptr ? entry >> 3 : 0;
                }
                it.barrier(sycl::access::fence_space::local_space);
                float ag[NE], au[NE];
#pragma unroll
                for (int i = 0; i < NE; ++i) ag[i] = au[i] = 0.0f;
                for (int block = first_block; block < blocks; block += STEP) {
                    const int kb = block * BPS + ib;
                    Sub gate{}, up{};
                    decode_b<T>(g + size_t(block) * Ty<T>::B, ib, grid, gate);
                    decode_b<T>(u + size_t(block) * Ty<T>::B, ib, grid, up);
#pragma unroll
                    for (int i = 0; i < NE; ++i) if (i < count) {
                        const int8_t* xp = a.xq + size_t(inputs[i]) * a.n_embd + size_t(kb) * 32;
                        const auto xa = load_codes(xp), xb = load_codes(xp + 16);
                        const float dx = a.xd[size_t(inputs[i]) * subblocks + size_t(kb)];
                        ag[i] = sycl::fma(dx, sub_dot<T>(gate, xa, xb), ag[i]);
                        au[i] = sycl::fma(dx, sub_dot<T>(up, xa, xb), au[i]);
                    }
                }
#pragma unroll
                for (int i = 0; i < NE; ++i) if (i < count) {
                    const float gv = lanes_sum<SG, 1>(sg, ag[i]), uv = lanes_sum<SG, 1>(sg, au[i]);
                    if (lane == 0) a.h[size_t(entries[i]) * a.n_ff + size_t(row)] = swiglu(gv, uv, a.limit);
                }
            }
        });
    });
}

// Upstream 870aedd9 ds4_iq.cu down_kernel<T,NE,2>: two independent half-subgroup rows.
template <int T, int NE, int SG>
void down_kernel(const Args& a, int groups, size_t warps, sycl::queue& q) {
    constexpr int RW = 2, LPR = SG / RW, BPS = kQK<T> / 32, STEP = LPR / BPS;
    static_assert(LPR % BPS == 0);
    const size_t threads = warps * SG, tiles = size_t(a.n_embd) / (warps * RW);
    const size_t rb = row_bytes_t<T>(a.n_ff);
    sycl_detail::submit(q, [&](sycl::handler& h) {
        sycl::local_accessor<uint32_t, 1> tab(sycl::range<1>(Ty<T>::TAB > 0 ? Ty<T>::TAB : 1), h);
        sycl::local_accessor<int, 1> entries(sycl::range<1>(NE), h);
        h.parallel_for(sycl::nd_range<1>(size_t(groups) * tiles * threads, threads),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(STRATA_SUB_GROUP(SG))]] {
            const size_t tid = it.get_local_linear_id(), tile = it.get_group_linear_id() % tiles;
            const int grp = int(it.get_group_linear_id() / tiles);
            const uint8_t* blob;
            int first, end;
            if (!group(a, grp, blob, first, end)) return;
            if constexpr (Ty<T>::TAB > 0)
                for (size_t j = tid; j < size_t(Ty<T>::TAB); j += threads) tab[j] = table<T>()[j];
            const auto* grid = tab.template get_multi_ptr<sycl::access::decorated::no>().get();
            const int lane_sg = int(tid % SG), lane = lane_sg % LPR;
            const int row = int((tile * warps + tid / SG) * RW) + lane_sg / LPR;
            const int ib = lane % BPS, first_block = lane / BPS;
            const int subblocks = a.n_ff / 32, blocks = subblocks / BPS;
            const uint8_t* d = blob + a.tr[2] + size_t(row) * rb;
            const auto sg = it.get_sub_group();
            for (int member = first; member < end; member += NE) {
                const int count = sycl::min(NE, end - member);
                it.barrier(sycl::access::fence_space::local_space);
                if (tid < size_t(count))
                    entries[tid] = a.grp_ptr ? a.grp_ent[member + int(tid)] : member + int(tid);
                it.barrier(sycl::access::fence_space::local_space);
                float accum[NE];
#pragma unroll
                for (int i = 0; i < NE; ++i) accum[i] = 0.0f;
                for (int block = first_block; block < blocks; block += STEP) {
                    const int kb = block * BPS + ib;
                    Sub down{};
                    decode_b<T>(d + size_t(block) * Ty<T>::B, ib, grid, down);
#pragma unroll
                    for (int i = 0; i < NE; ++i) if (i < count) {
                        const int8_t* hp = a.hq + size_t(entries[i]) * a.n_ff + size_t(kb) * 32;
                        const auto ha = load_codes(hp), hb = load_codes(hp + 16);
                        const float scale = a.hd[size_t(entries[i]) * subblocks + size_t(kb)];
                        accum[i] = sycl::fma(scale, sub_dot<T>(down, ha, hb), accum[i]);
                    }
                }
#pragma unroll
                for (int i = 0; i < NE; ++i) if (i < count) {
                    const float value = lanes_sum<SG, RW>(sg, accum[i]);
                    if (lane == 0) a.part[size_t(entries[i]) * a.n_embd + size_t(row)] = value;
                }
            }
        });
    });
}

template <int T, bool GU, int SG>
bool launch_width(const Args& a, int groups, int ne_max, sycl::queue& q, const Caps& device) {
    constexpr size_t RW = GU ? 1 : 2;
    // Eight subgroups is upstream's block geometry ceiling. Smaller devices keep
    // the lane/sub-block mapping and use fewer output rows per work-group.
    const size_t warps = std::min(size_t(8), device.group / SG);
    const int n = GU ? a.n_ff : a.n_embd;
    const int entries = ne_max <= 1 ? 1 : ne_max <= 4 ? 4 : ne_max <= 8 ? 8 : 16;
    const size_t local = size_t(Ty<T>::TAB > 0 ? Ty<T>::TAB : 1) * sizeof(uint32_t) +
                         size_t(entries) * sizeof(int) * (GU ? 2 : 1);
    if (!warps || size_t(n) % (warps * RW) || local > device.local) return false;
    const auto run = [&]<int NE>() {
        if constexpr (GU) gate_up_kernel<T, NE, SG>(a, groups, warps, q);
        else down_kernel<T, NE, SG>(a, groups, warps, q);
    };
    if (entries == 1) run.template operator()<1>();
    else if (entries == 4) run.template operator()<4>();
    else if (entries == 8) run.template operator()<8>();
    else run.template operator()<16>();
    return true;
}

template <int T, bool GU>
bool launch(const Args& a, int groups, int ne_max, sycl::queue& q) {
    if (groups <= 0 || ne_max <= 0) return true;
    const int n = GU ? a.n_ff : a.n_embd, k = GU ? a.n_embd : a.n_ff;
    const auto* codes = GU ? a.xq : a.hq;
    // Preserve the upstream complete-block and int4-load eligibility. The public
    // entry retains its portable fallback for other dimensions or byte alignment.
    if (n <= 0 || k <= 0 || k % 256 || (reinterpret_cast<uintptr_t>(codes) & 15u)) return false;
    sycl_detail::require_in_order(q);
    const auto& device = caps(q);
    const auto has = [&](size_t sg) {
        return std::find(device.subgroups.begin(), device.subgroups.end(), sg) != device.subgroups.end();
    };
    if (has(32)) return launch_width<T, GU, 32>(a, groups, ne_max, q, device);
    if (has(16)) return launch_width<T, GU, 16>(a, groups, ne_max, q, device);
    return false;
}

}  // namespace strata::ds4::gguf::iq::token_detail
