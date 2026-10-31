// SPDX-FileCopyrightText: 2026 Peasant Smith and the Project Maya contributors
// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/main/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
#pragma once

#include "ds4_gguf.cuh"
#include "ds4_iq_token.hpp"
#include "ds4_sycl.hpp"
#include "../kernels/xe/cuda_intrinsics.hpp"

namespace strata::ds4::gguf::mv_detail {

// Upstream uses __launch_bounds__(8 * 32). Give the compiler the corresponding
// exact work-group only when the device-reported WG/local-memory limits selected
// eight subgroups. Smaller selections retain the dynamic body and subgroup size.
// This is a launch-shape annotation; its performance effect is measured separately.
template <int SG, class Body>
struct BoundBody {
    Body body;
    [[sycl::reqd_work_group_size(8 * SG), sycl::reqd_sub_group_size(STRATA_SUB_GROUP(SG))]]
    void operator()(sycl::nd_item<1> item) const { body(item); }
};
template <int SG, class Body>
struct DynamicBody {
    Body body;
    [[sycl::reqd_sub_group_size(STRATA_SUB_GROUP(SG))]]
    void operator()(sycl::nd_item<1> item) const { body(item); }
};
template <int SG, class Submitter, class Body>
void submit_body(Submitter& submitter, sycl::nd_range<1> range, size_t warps, Body body) {
    if (warps == 8) submitter.parallel_for(range, BoundBody<SG, Body>{body});
    else submitter.parallel_for(range, DynamicBody<SG, Body>{body});
}

// Upstream 870aedd9 ds4_gguf.cu BfJob/BfJobs and QuantItems. Job metadata is copied
// into each kernel; the caller's temporary jobs array is never device storage.
struct Job {
    const void* w = nullptr;
    size_t ld = 0;
    const float* x = nullptr;
    size_t xcs = 0;
    float* y = nullptr;
    size_t ycs = 0;
    const float* bias = nullptr;
    float alpha = 0.f;
    int k = 0, n = 0, ncol = 0;
    const int8_t* xq = nullptr;
    const float* xd = nullptr;
};
struct Jobs { Job j[x3::kMaxMvJobs]; };
struct QuantItems {
    const float* x[x3::kMaxMvJobs] = {};
    size_t ldx[x3::kMaxMvJobs] = {};
    int k[x3::kMaxMvJobs] = {}, rows[x3::kMaxMvJobs] = {};
    int8_t* q[x3::kMaxMvJobs] = {};
    float* d[x3::kMaxMvJobs] = {};
    int n = 0;
};

template <int SG>
inline float warp_sum(sycl::sub_group group, float value) {
#pragma unroll
    for (int offset = SG / 2; offset > 0; offset >>= 1)
        value += sycl::permute_group_by_xor(group, value, offset);
    return value;
}
template <class T, size_t Alignment>
inline T load(const void* p) {
    T value;
    __builtin_memcpy(&value, __builtin_assume_aligned(p, Alignment), sizeof(value));
    return value;
}

// Upstream q8_quant_multi_kernel: grid (max blocks / warps, max rows, items).
// SG16 owns the two halves of one storage block, retaining the same max and rounding.
template <int SG>
void quantize(const QuantItems& items, int max_k, int max_rows, size_t warps, sycl::queue& q) {
    const size_t tiles = (size_t(max_k) / 32 + warps - 1) / warps, threads = warps * SG;
    submit_body<SG>(q, sycl::nd_range<1>(tiles * max_rows * items.n * threads, threads), warps,
                   [=](sycl::nd_item<1> it) {
        const size_t group = it.get_group_linear_id(), tile = group % tiles;
        const int row = int(group / tiles % size_t(max_rows)), item = int(group / (tiles * max_rows));
        const size_t block = tile * warps + it.get_local_linear_id() / SG;
        const int k = items.k[item];
        if (row >= items.rows[item] || block >= size_t(k) / 32) return;
        const auto sg = it.get_sub_group();
        const size_t lane = sg.get_local_linear_id();
        const float v0 = items.x[item][size_t(row) * items.ldx[item] + block * 32 + lane];
        float v1 = 0.f, amax = sycl::fabs(v0);
        if constexpr (SG == 16) {
            v1 = items.x[item][size_t(row) * items.ldx[item] + block * 32 + lane + 16];
            amax = sycl::fmax(amax, sycl::fabs(v1));
        }
#pragma unroll
        for (int offset = SG / 2; offset > 0; offset >>= 1)
            amax = sycl::fmax(amax, sycl::permute_group_by_xor(sg, amax, offset));
        const float scale = amax / 127.f;
        items.q[item][size_t(row) * k + block * 32 + lane] = int8_t(amax > 0.f ? sycl::rint(v0 / scale) : 0.f);
        if constexpr (SG == 16)
            items.q[item][size_t(row) * k + block * 32 + lane + 16] = int8_t(amax > 0.f ? sycl::rint(v1 / scale) : 0.f);
        if (lane == 0) items.d[item][size_t(row) * (k / 32) + block] = scale;
    });
}

// Upstream bf16_mv_kernel<NC,VEC>: one subgroup/output row, eight BF16 values and
// two float4 loads per lane/step when aligned; scalar lane-strided FMA otherwise.
template <int NC, bool VEC, int SG>
void bf16_kernel(const Jobs& jobs, int count, int max_n, size_t warps, sycl::queue& q) {
    const size_t tiles = (size_t(max_n) + warps - 1) / warps, threads = warps * SG;
    submit_body<SG>(q, sycl::nd_range<1>(tiles * count * threads, threads), warps,
                   [=](sycl::nd_item<1> it) {
        const Job& b = jobs.j[it.get_group_linear_id() / tiles];
        const size_t row = it.get_group_linear_id() % tiles * warps + it.get_local_linear_id() / SG;
        const int lane = int(it.get_local_linear_id() % SG);
        if (row >= size_t(b.n)) return;
        const auto* w = static_cast<const uint16_t*>(b.w) + row * b.ld;
        float acc[NC] = {};
        if constexpr (VEC) {
            for (int i = lane * 8; i < b.k; i += SG * 8) {
                const auto words = load<sycl::uint4, 16>(w + i);
                const float wf[8] = {sycl::bit_cast<float>(words.x() << 16), sycl::bit_cast<float>(words.x() & 0xffff0000u),
                    sycl::bit_cast<float>(words.y() << 16), sycl::bit_cast<float>(words.y() & 0xffff0000u),
                    sycl::bit_cast<float>(words.z() << 16), sycl::bit_cast<float>(words.z() & 0xffff0000u),
                    sycl::bit_cast<float>(words.w() << 16), sycl::bit_cast<float>(words.w() & 0xffff0000u)};
#pragma unroll
                for (int c = 0; c < NC; ++c) if (c < b.ncol) {
                    const auto x0 = load<sycl::float4, 16>(b.x + size_t(c) * b.xcs + i);
                    const auto x1 = load<sycl::float4, 16>(b.x + size_t(c) * b.xcs + i + 4);
                    float a = acc[c];
                    a = sycl::fma(wf[0], x0.x(), a); a = sycl::fma(wf[1], x0.y(), a);
                    a = sycl::fma(wf[2], x0.z(), a); a = sycl::fma(wf[3], x0.w(), a);
                    a = sycl::fma(wf[4], x1.x(), a); a = sycl::fma(wf[5], x1.y(), a);
                    a = sycl::fma(wf[6], x1.z(), a); a = sycl::fma(wf[7], x1.w(), a);
                    acc[c] = a;
                }
            }
        } else {
            for (int i = lane; i < b.k; i += SG) {
                const float wf = sycl_detail::bf16_value(w[i]);
#pragma unroll
                for (int c = 0; c < NC; ++c) if (c < b.ncol)
                    acc[c] = sycl::fma(wf, b.x[size_t(c) * b.xcs + i], acc[c]);
            }
        }
#pragma unroll
        for (int c = 0; c < NC; ++c) if (c < b.ncol) {
            const float value = warp_sum<SG>(it.get_sub_group(), acc[c]);
            if (lane == 0) b.y[size_t(c) * b.ycs + row] = b.alpha * value + (b.bias ? b.bias[row] : 0.f);
        }
    });
}

inline uint32_t funnel(uint32_t lo, uint32_t hi, int shift) {
    return shift ? (lo >> shift) | (hi << (32 - shift)) : lo;
}
inline float q8_words_global(const uint8_t* p, int* values) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const uint32_t bits = uint32_t(p[2 + 4 * j]) | (uint32_t(p[3 + 4 * j]) << 8) |
                              (uint32_t(p[4 + 4 * j]) << 16) | (uint32_t(p[5 + 4 * j]) << 24);
        values[j] = sycl::bit_cast<int>(bits);
    }
    return sycl_detail::f16_value(uint16_t(p[0] | (uint16_t(p[1]) << 8)));
}
inline float q8_words_local(const uint32_t* stage, int lane, int* values) {
    const int offset = kQ8Bytes * lane, word = offset / 4, shift = (offset % 4) * 8;
    uint32_t a[9], sw[9];
#pragma unroll
    for (int j = 0; j < 9; ++j) a[j] = stage[word + j];
#pragma unroll
    for (int j = 0; j < 8; ++j) sw[j] = funnel(a[j], a[j + 1], shift);
    sw[8] = a[8] >> shift;
#pragma unroll
    for (int j = 0; j < 8; ++j) values[j] = sycl::bit_cast<int>(funnel(sw[j], sw[j + 1], 16));
    return sycl_detail::f16_value(uint16_t(sw[0] & 0xffffu));
}

// Upstream Fetch<kQ8Bytes>: prefetch the next subgroup-sized block chunk in
// registers, then cooperatively publish its 4-byte words to the subgroup's stage.
template <int SG>
struct Fetch {
    static constexpr int Words = (SG * kQ8Bytes / 4 + SG - 1) / SG;
    uint32_t r[Words] = {};
    int n = 0;
    void read(const uint8_t* row, int first, int blocks, int lane) {
        n = blocks * kQ8Bytes / 4;
#pragma unroll
        for (int j = 0; j < Words; ++j) {
            const int i = lane + SG * j;
            r[j] = i < n ? load<uint32_t, 4>(row + size_t(first) * kQ8Bytes + size_t(i) * 4) : 0u;
        }
    }
    void store(uint32_t* stage, int lane, sycl::sub_group sg) const {
        sycl::group_barrier(sg, sycl::memory_scope::sub_group);
#pragma unroll
        for (int j = 0; j < Words; ++j) {
            const int i = lane + SG * j;
            if (i < n) stage[i] = r[j];
        }
        sycl::group_barrier(sg, sycl::memory_scope::sub_group);
    }
};

// Upstream q8_mv_kernel<NC,ST>: one subgroup/output row, one complete Q8_0
// block/lane/step, exact integer dot then F32 scale and FMA. ST retains Fetch's
// register prefetch/local stage; unaligned rows use the upstream byte-load path.
template <int NC, bool ST, int SG>
void q8_kernel(const Jobs& jobs, int count, int max_n, size_t warps, sycl::queue& q) {
    const size_t tiles = (size_t(max_n) + warps - 1) / warps, threads = warps * SG;
    sycl_detail::submit(q, [&](sycl::handler& h) {
        sycl::local_accessor<uint32_t, 1> storage(ST ? warps * SG * kQ8Bytes / 4 : 1, h);
        submit_body<SG>(h, sycl::nd_range<1>(tiles * count * threads, threads), warps,
                       [=](sycl::nd_item<1> it) {
            const Job& b = jobs.j[it.get_group_linear_id() / tiles];
            const size_t warp = it.get_local_linear_id() / SG;
            const size_t row = it.get_group_linear_id() % tiles * warps + warp;
            const int lane = int(it.get_local_linear_id() % SG);
            if (row >= size_t(b.n)) return;
            const auto* w = static_cast<const uint8_t*>(b.w) + row * b.ld;
            const int blocks = b.k / 32;
            auto* stage = storage.template get_multi_ptr<sycl::access::decorated::no>().get();
            if constexpr (ST) stage += warp * SG * kQ8Bytes / 4;
            const auto sg = it.get_sub_group();
            float acc[NC] = {};
            Fetch<SG> fetch;
            if constexpr (ST) fetch.read(w, 0, sycl::min(SG, blocks), lane);
            for (int first = 0; first < blocks; first += SG) {
                const int count_blocks = sycl::min(SG, blocks - first), block = first + lane;
                if constexpr (ST) {
                    fetch.store(stage, lane, sg);
                    if (first + SG < blocks) fetch.read(w, first + SG, sycl::min(SG, blocks - first - SG), lane);
                }
                if (lane >= count_blocks) continue;
                int v[8];
                const float dw = ST ? q8_words_local(stage, lane, v) : q8_words_global(w + size_t(block) * kQ8Bytes, v);
#pragma unroll
                for (int c = 0; c < NC; ++c) if (c < b.ncol) {
                    const int8_t* xp = b.xq + size_t(c) * b.k + size_t(block) * 32;
                    const auto xa = load<sycl::int4, 16>(xp), xb = load<sycl::int4, 16>(xp + 16);
                    using strata::kernels::xe::dp4a;
                    int sum = dp4a(v[0], xa.x(), 0);
                    sum = dp4a(v[1], xa.y(), sum); sum = dp4a(v[2], xa.z(), sum); sum = dp4a(v[3], xa.w(), sum);
                    sum = dp4a(v[4], xb.x(), sum); sum = dp4a(v[5], xb.y(), sum);
                    sum = dp4a(v[6], xb.z(), sum); sum = dp4a(v[7], xb.w(), sum);
                    acc[c] = sycl::fma(dw * b.xd[size_t(c) * blocks + block], float(sum), acc[c]);
                }
            }
#pragma unroll
            for (int c = 0; c < NC; ++c) if (c < b.ncol) {
                const float value = warp_sum<SG>(sg, acc[c]);
                if (lane == 0) b.y[size_t(c) * b.ycs + row] = b.alpha * value + (b.bias ? b.bias[row] : 0.f);
            }
        });
    });
}

// Upstream ds4_iq.cu mv_kernel<T,NC>: one subgroup/output row, the lanes over the
// row's 32-value sub-blocks, each sub-block's exact integer dot with the int8
// activations, then its F32 scale and FMA, and the subgroup's sum.
template <int T, int NC, int SG>
void iq_kernel(const Jobs& jobs, int count, int max_n, size_t warps, sycl::queue& q) {
    const size_t tiles = (size_t(max_n) + warps - 1) / warps, threads = warps * SG;
    submit_body<SG>(q, sycl::nd_range<1>(tiles * count * threads, threads), warps,
                   [=](sycl::nd_item<1> it) {
        const Job& b = jobs.j[it.get_group_linear_id() / tiles];
        const size_t row = it.get_group_linear_id() % tiles * warps + it.get_local_linear_id() / SG;
        const int lane = int(it.get_local_linear_id() % SG);
        if (row >= size_t(b.n)) return;
        const auto* w = static_cast<const uint8_t*>(b.w) + row * b.ld;
        const int blocks = b.k / 32;
        float acc[NC] = {};
        for (int kb = lane; kb < blocks; kb += SG) {
            iq::sycl_decode::Sub sub{};
            iq::sycl_decode::decode<T>(w, kb, nullptr, sub);
#pragma unroll
            for (int c = 0; c < NC; ++c) if (c < b.ncol) {
                const int8_t* xp = b.xq + size_t(c) * b.k + size_t(kb) * 32;
                const auto xa = load<sycl::int4, 16>(xp), xb = load<sycl::int4, 16>(xp + 16);
                acc[c] = sycl::fma(b.xd[size_t(c) * blocks + kb], iq::token_detail::sub_dot<T>(sub, xa, xb), acc[c]);
            }
        }
#pragma unroll
        for (int c = 0; c < NC; ++c) if (c < b.ncol) {
            const float value = warp_sum<SG>(it.get_sub_group(), acc[c]);
            if (lane == 0) b.y[size_t(c) * b.ycs + row] = b.alpha * value + (b.bias ? b.bias[row] : 0.f);
        }
    });
}
template <int T, int SG>
void launch_iq(const Jobs& jobs, int count, int max_n, int max_cols, size_t warps, sycl::queue& q) {
    if (count == 0) return;
    // Upstream mv_nc's activation-count specializations.
    if (max_cols <= 1) iq_kernel<T, 1, SG>(jobs, count, max_n, warps, q);
    else if (max_cols <= 4) iq_kernel<T, 4, SG>(jobs, count, max_n, warps, q);
    else iq_kernel<T, 16, SG>(jobs, count, max_n, warps, q);
}

template <int NC, int SG>
void launch(bool quantized, const Jobs& jobs, int count, int max_n, bool aligned, size_t warps, sycl::queue& q) {
    if (quantized) {
        if (aligned) q8_kernel<NC, true, SG>(jobs, count, max_n, warps, q);
        else q8_kernel<NC, false, SG>(jobs, count, max_n, warps, q);
    } else {
        if (aligned) bf16_kernel<NC, true, SG>(jobs, count, max_n, warps, q);
        else bf16_kernel<NC, false, SG>(jobs, count, max_n, warps, q);
    }
}
template <int SG>
void launch(bool quantized, const Jobs& jobs, int count, int max_n, int max_cols,
            bool aligned, size_t warps, sycl::queue& q) {
    if (count == 0) return;
    // Upstream mv_set's activation-count specializations.
    if (max_cols <= 1) launch<1, SG>(quantized, jobs, count, max_n, aligned, warps, q);
    else if (max_cols <= 2) launch<2, SG>(quantized, jobs, count, max_n, aligned, warps, q);
    else if (max_cols <= 4) launch<4, SG>(quantized, jobs, count, max_n, aligned, warps, q);
    else if (max_cols <= 8) launch<8, SG>(quantized, jobs, count, max_n, aligned, warps, q);
    else launch<16, SG>(quantized, jobs, count, max_n, aligned, warps, q);
}

}  // namespace strata::ds4::gguf::mv_detail
