// SPDX-FileCopyrightText: 2026 Peasant Smith and the Project Maya contributors
// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/main/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
#include "ds4_gguf.cuh"
#include "ds4_kernels.cuh"
#include "ds4_iq.cuh"
#include "ds4_iq_decode.hpp"
#include "ds4_tq_decode.hpp"
#include "ds4_numerics.cuh"
#include "strata/core/per_device.hpp"
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace strata::ds4::gguf {
namespace {
using sycl_detail::parallel;
using sycl_detail::f16_value;
using sycl_detail::f16_bits;
using sycl_detail::bf16_value;
std::atomic<int> errors{0};

bool fail(const char* what) {
    std::fprintf(stderr, "ds4 gguf: %s\n", what);
    errors.fetch_add(1, std::memory_order_relaxed);
    return false;
}
size_t row_stride(const x3::Mat& m) {
    if (m.ld > 0) return size_t(m.ld);
    return m.fmt == kFmtMXFP4 ? mx_row_bytes(m.k) : m.fmt == kFmtQ8_0 ? q8_row_bytes(m.k) :
           m.fmt == kFmtGG ? type_row_bytes(m.K, m.k) : size_t(m.k);
}
float weight(int type, const uint8_t* row, int i) {
    return iq::is_tq(type) ? iq::tq::sycl_decode::value(type, row, i) : iq::sycl_decode::value(type, row, i);
}
float q8_weight(const uint8_t* row, int i) {
    const uint8_t* b = row + size_t(i / 32) * kQ8Bytes;
    return f16_value(uint16_t(b[0] | uint16_t(b[1]) << 8)) * float(reinterpret_cast<const int8_t*>(b + 2)[i % 32]);
}
float silu_product(float gate, float up, float limit) {
    if (limit > 0.f) { gate = sycl::fmin(gate, limit); up = sycl::clamp(up, -limit, limit); }
    return gate / (1.f + sycl::exp(-gate)) * up;
}
float mx_quant_dot(const uint8_t* row, const float* x, int k) {
    float sum = 0.f;
    for (int block = 0; block < k / 32; ++block) {
        const int first = block * 32;
        float amax = 0.f;
        for (int j = 0; j < 32; ++j) amax = sycl::fmax(amax, sycl::fabs(x[first + j]));
        const float dx = amax / 127.f;
        iq::sycl_decode::Sub decoded{};
        iq::sycl_decode::decode<iq::sycl_decode::kMXFP4>(row, block, nullptr, decoded);
        float dot = 0.f;
        for (int j = 0; j < 32; ++j) {
            const float code = amax > 0.f ? sycl::rint(x[first + j] / dx) : 0.f;
            const uint32_t byte = (uint32_t(decoded.q[j / 4]) >> (8 * (j % 4))) & 255u;
            const int quant = int(byte ^ 128u) - 128;
            const float value = decoded.d * float(decoded.s0 * quant) - decoded.m0;
            dot += code * value;
        }
        // Only the repeated weight decode is removed: both accumulation orders and activation quantization stay.
        sum = sycl::fma(dx, dot, sum);
    }
    return sum;
}

// Match upstream's per-32 activation quantizer without per-stream, device-sized
// scratch. Quantization blocks are a storage format, not a subgroup assumption.
float quant_dot(int type, const uint8_t* row, const float* x, int k, bool q8 = false) {
    if (!q8 && type == kTypeMXFP4) return mx_quant_dot(row, x, k);
    float sum = 0.f;
    for (int b = 0; b < k; b += 32) {
        float amax = 0.f;
        for (int j = 0; j < 32; ++j) amax = sycl::fmax(amax, sycl::fabs(x[b + j]));
        const float dx = amax / 127.f;
        float dot = 0.f;
        for (int j = 0; j < 32; ++j) {
            const float code = amax > 0.f ? sycl::rint(x[b + j] / dx) : 0.f;
            dot += code * (q8 ? q8_weight(row, b + j) : weight(type, row, b + j));
        }
        sum = sycl::fma(dx, dot, sum);
    }
    return sum;
}
bool layout_ok(const x3::ExpertLayout& lay, int n_embd, int n_ff) {
    for (int r = 0; r < 3; ++r) {
        const int type = expert_type(lay, r), k = r == 2 ? n_ff : n_embd;
        if (!type_on_device(type) || k <= 0 || type_row_bytes(type, k) == 0) return false;
    }
    return true;
}
iq::SetArgs light_args(const x3::SetArgs& a) {
    iq::SetArgs q{};
    q.wbase = a.wbase; q.stride = a.stride;
    for (int r = 0; r < 3; ++r) q.tr[r] = a.lay.trellis[r];
    q.bounds = a.bounds; q.row0 = a.row0; q.max_rows = a.max_rows; q.n_exp = a.n_exp;
    q.n_embd = a.n_embd; q.n_ff = a.n_ff; q.limit = a.limit; q.skip_ge = a.skip_ge;
    q.tg = expert_type(a.lay, 0); q.td = expert_type(a.lay, 2);
    return q;
}
bool iq_tiled_set_type(int type) {
    return type == iq::kIQ1_S || type == iq::kIQ1_M || type == iq::kIQ2_XXS ||
           type == iq::kIQ2_XS || type == iq::kIQ2_S || type == iq::kIQ3_XXS || type == kTypeMXFP4;
}

struct RowCaps {
    size_t max_group = 0, local_bytes = 0;
    std::vector<size_t> sub_groups;
};
float q5_value(const iq::sycl_decode::Sub& decoded, int j) {
    const int code = int((uint32_t(decoded.q[j / 4]) >> (8 * (j % 4))) & 255u);
    return decoded.d * float(decoded.s0 * code) - decoded.m0;
}
float q5_quant_dot(const uint8_t* row, const float* x, int k) {
    float sum = 0.f;
    for (int block = 0; block < k / 32; ++block) {
        const int first = block * 32;
        float amax = 0.f;
        for (int j = 0; j < 32; ++j) amax = sycl::fmax(amax, sycl::fabs(x[first + j]));
        const float dx = amax / 127.f;
        iq::sycl_decode::Sub decoded{};
        iq::sycl_decode::decode<iq::kQ5_K>(row, block, nullptr, decoded);
        float dot = 0.f;
        for (int j = 0; j < 32; ++j) {
            const float code = amax > 0.f ? sycl::rint(x[first + j] / dx) : 0.f;
            dot += code * q5_value(decoded, j);
        }
        sum = sycl::fma(dx, dot, sum);
    }
    return sum;
}
bool q5_mv(const x3::MvJob& job, sycl::queue& s) {
    const x3::Mat m = *job.m;
    if (m.fmt != kFmtGG || m.K != iq::kQ5_K) return false;
    sycl_detail::require_in_order(s);
    const x3::MvJob j = job;
    const size_t ld = row_stride(m), xcs = j.xf_cs ? j.xf_cs : size_t(m.k), ycs = j.y_cs ? j.y_cs : size_t(m.n);
    const auto* raw = static_cast<const uint8_t*>(m.raw);
    const auto device = s.get_device();
    static core::PerDevice<RowCaps> cache;
    const auto& caps = cache.get(device, [&] {
        return RowCaps{std::min(device.get_info<sycl::info::device::max_work_group_size>(),
                                device.get_info<sycl::info::device::max_work_item_sizes<1>>()[0]),
                       size_t(device.get_info<sycl::info::device::local_mem_size>()),
                       device.get_info<sycl::info::device::sub_group_sizes>()};
    });
    const size_t k = size_t(m.k), blocks = k / 32, shared_values = k + blocks;
    const size_t limit = std::min(size_t(m.n), caps.max_group);
    if (!limit || shared_values > caps.local_bytes / sizeof(float)) {
        // Small local memory still gets one weight decode per sub-block; input quantization stays per row.
        parallel(s, size_t(j.ncol) * m.n, [=](size_t z) {
            const size_t c = z / m.n, r = z % m.n;
            const float sum = q5_quant_dot(raw + r * ld, j.xf + c * xcs, m.k);
            j.y[c * ycs + r] = j.alpha * sum + (j.bias ? j.bias[r] : 0.f);
        });
        return true;
    }
    size_t local = 1;
    for (const size_t width : caps.sub_groups) if (width <= limit) local = std::max(local, width);
    const size_t groups_per_col = (size_t(m.n) + local - 1) / local;
    const size_t groups = size_t(j.ncol) * groups_per_col;
    s.submit([&](sycl::handler& h) {
        // Float codes preserve the previous behavior for nonfinite inputs; no integer conversion is introduced.
        sycl::local_accessor<float, 1> activation(shared_values, h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(groups * local), sycl::range<1>(local)),
                       [=](sycl::nd_item<1> item) {
            const size_t lane = item.get_local_linear_id(), group = item.get_group_linear_id();
            const size_t c = group / groups_per_col, row = (group % groups_per_col) * local + lane;
            const float* x = j.xf + c * xcs;
            for (size_t block = lane; block < blocks; block += local) {
                const size_t first = block * 32;
                float amax = 0.f;
                for (size_t i = 0; i < 32; ++i) amax = sycl::fmax(amax, sycl::fabs(x[first + i]));
                const float dx = amax / 127.f;
                activation[k + block] = dx;
                for (size_t i = 0; i < 32; ++i)
                    activation[first + i] = amax > 0.f ? sycl::rint(x[first + i] / dx) : 0.f;
            }
            sycl::group_barrier(item.get_group());
            if (row >= size_t(m.n)) return;
            float sum = 0.f;
            for (size_t block = 0; block < blocks; ++block) {
                iq::sycl_decode::Sub decoded{};
                iq::sycl_decode::decode<iq::kQ5_K>(raw + row * ld, int(block), nullptr, decoded);
                float dot = 0.f;
                // Keep the old coefficient order inside each block and the block order in the outer FMA chain.
                for (int i = 0; i < 32; ++i) dot += activation[block * 32 + size_t(i)] * q5_value(decoded, i);
                sum = sycl::fma(activation[k + block], dot, sum);
            }
            j.y[c * ycs + row] = j.alpha * sum + (j.bias ? j.bias[row] : 0.f);
        });
    });
    return true;
}

bool q5_rows(const x3::Mat& m, size_t ld, uint16_t* dst, bool bf16, sycl::queue& s) {
    if (m.fmt != kFmtGG || m.K != iq::kQ5_K) return false;
    sycl_detail::require_in_order(s);
    const auto device = s.get_device();
    static core::PerDevice<RowCaps> cache;
    const auto& caps = cache.get(device, [&] {
        return RowCaps{std::min(device.get_info<sycl::info::device::max_work_group_size>(),
                                device.get_info<sycl::info::device::max_work_item_sizes<1>>()[0]),
                       size_t(device.get_info<sycl::info::device::local_mem_size>()),
                       device.get_info<sycl::info::device::sub_group_sizes>()};
    });
    // Each work-item decodes one 32-value quantization sub-block. A padding element separates the local rows
    // before the group writes them in contiguous order; 32 is the storage format, not a required subgroup width.
    constexpr size_t values_per_block = 32, local_pitch = values_per_block + 1;
    const size_t blocks_per_row = size_t(m.k) / values_per_block, blocks = size_t(m.n) * blocks_per_row;
    const size_t limit = std::min({blocks, caps.max_group, caps.local_bytes / (local_pitch * sizeof(uint16_t))});
    if (!limit) return false;
    size_t subgroup = 1;
    for (const size_t width : caps.sub_groups) if (width <= limit) subgroup = std::max(subgroup, width);
    const size_t target = std::min(limit, std::max(blocks_per_row, subgroup));
    const size_t local = std::max<size_t>(1, target / subgroup * subgroup);
    const size_t groups = (blocks + local - 1) / local;
    const auto* raw = static_cast<const uint8_t*>(m.raw);
    s.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint16_t, 1> staging(local * local_pitch, h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(groups * local), sycl::range<1>(local)),
                       [=](sycl::nd_item<1> item) {
            const size_t lane = item.get_local_linear_id(), base = item.get_group_linear_id() * local;
            const size_t block = base + lane;
            if (block < blocks) {
                const size_t row = block / blocks_per_row, sub_block = block % blocks_per_row;
                iq::sycl_decode::Sub decoded{};
                iq::sycl_decode::decode<iq::kQ5_K>(raw + row * ld, int(sub_block), nullptr, decoded);
                for (size_t j = 0; j < values_per_block; ++j) {
                    // Q5_K has unsigned 5-bit codes and one scale/minimum for this whole 32-value sub-block.
                    // Keep Decoder::value's integer scale product and FP32 multiply/subtract order exactly.
                    const int code = int((uint32_t(decoded.q[j / 4]) >> (8 * (j % 4))) & 255u);
                    const float value = decoded.d * float(decoded.s0 * code) - decoded.m0;
                    staging[lane * local_pitch + j] = bf16 ? bf16_rn(value) : f16_bits(value);
                }
            }
            sycl::group_barrier(item.get_group());
            const size_t count = std::min(local, blocks - base) * values_per_block;
            for (size_t j = lane; j < count; j += local)
                dst[base * values_per_block + j] = staging[(j / values_per_block) * local_pitch + j % values_per_block];
        });
    });
    return true;
}
}

size_t type_row_bytes(int type, int64_t k) {
    if (k <= 0) return 0;
    if (type == kTypeMXFP4) return k % kMxBlock == 0 ? mx_row_bytes(k) : 0;
    const int block = iq::known(type) ? iq::block_values(type) : 0;
    return block > 0 && k % block == 0 ? iq::row_bytes(type, k) : 0;
}
int type_block(int type) { return type == kTypeMXFP4 ? kMxBlock : iq::known(type) ? iq::block_values(type) : 0; }
const char* type_name(int type) { return type == kTypeMXFP4 ? "MXFP4" : iq::name(type); }
bool type_on_device(int type) { return type == kTypeMXFP4 || iq::supported(type); }
bool type_on_chunk(int type) { return type_on_device(type); }
bool prepare(sycl::queue& s) { return s.has_property<sycl::property::queue::in_order>(); } // No persistent scratch.
int launch_errors() { return errors.load(std::memory_order_relaxed); }

x3::Mat view(const x3::Mat& m, int c0, int nc) {
    x3::Mat result = m;
    const size_t ld = row_stride(m);
    result.ld = int(ld);
    result.raw = m.fmt == kFmtBF16 ? static_cast<const void*>(static_cast<const uint16_t*>(m.raw) + size_t(c0) * ld) :
                 static_cast<const void*>(static_cast<const uint8_t*>(m.raw) + size_t(c0) * ld);
    result.n = nc;
    return result;
}
bool mv(const x3::MvJob* jobs, int n, sycl::queue& s) {
    if (n <= 0 || n > x3::kMaxMvJobs) return fail("invalid matvec job count");
    for (int i = 0; i < n; ++i) {
        const auto& j = jobs[i];
        if (!j.m || !j.xf || !j.y || j.ncol <= 0) return fail("invalid matvec job");
        const auto& m = *j.m;
        if (!m.raw || m.k <= 0 || m.n <= 0 || m.fmt < kFmtBF16 || m.fmt > kFmtGG ||
            (m.fmt != kFmtBF16 && m.k % 32) ||
            (m.fmt == kFmtGG && (!type_on_device(m.K) || !type_row_bytes(m.K, m.k))))
            return fail("unsupported matvec matrix");
    }
    for (int i = 0; i < n; ++i) {
        const x3::MvJob j = jobs[i];
        if (q5_mv(j, s)) continue;
        const x3::Mat m = *j.m;
        const size_t ld = row_stride(m), xcs = j.xf_cs ? j.xf_cs : size_t(m.k), ycs = j.y_cs ? j.y_cs : size_t(m.n);
        parallel(s, size_t(j.ncol) * m.n, [=](size_t z) {
            const size_t c = z / m.n, r = z % m.n;
            const float* x = j.xf + c * xcs;
            float sum = 0.f;
            if (m.fmt == kFmtBF16) {
                const uint16_t* w = static_cast<const uint16_t*>(m.raw) + r * ld;
                for (int k = 0; k < m.k; ++k) sum += bf16_value(w[k]) * x[k];
            } else {
                const uint8_t* w = static_cast<const uint8_t*>(m.raw) + r * ld;
                if (m.fmt == kFmtQ8_0) sum = quant_dot(0, w, x, m.k, true);
                else if (m.fmt == kFmtGG) sum = quant_dot(m.K, w, x, m.k);
                else for (int k = 0; k < m.k; ++k) sum = sycl::fma(weight(kTypeMXFP4, w, k), x[k], sum);
            }
            j.y[c * ycs + r] = j.alpha * sum + (j.bias ? j.bias[r] : 0.f);
        });
    }
    return true;
}
void rows(const x3::Mat& m, uint16_t* dst, bool bf16, sycl::queue& s) {
    if (!m.raw || !dst || m.k <= 0 || m.n <= 0 || m.fmt < kFmtBF16 || m.fmt > kFmtGG ||
        (m.fmt != kFmtBF16 && m.k % 32) ||
        (m.fmt == kFmtGG && (!type_on_device(m.K) || !type_row_bytes(m.K, m.k)))) {
        fail("unsupported row reconstruction"); return;
    }
    const size_t ld = row_stride(m);
    if (q5_rows(m, ld, dst, bf16, s)) return;
    parallel(s, size_t(m.n) * m.k, [=](size_t z) {
        const size_t r = z / m.k;
        const int k = int(z % m.k);
        float value;
        if (m.fmt == kFmtBF16) value = bf16_value(static_cast<const uint16_t*>(m.raw)[r * ld + k]);
        else {
            const uint8_t* w = static_cast<const uint8_t*>(m.raw) + r * ld;
            value = m.fmt == kFmtQ8_0 ? q8_weight(w, k) : weight(m.fmt == kFmtMXFP4 ? kTypeMXFP4 : m.K, w, k);
        }
        dst[z] = bf16 ? bf16_rn(value) : f16_bits(value);
    });
}
void moe_gate_up(const x3::MoeArgs& a, const float* x, float* h, sycl::queue& s) {
    if (a.k < 0 || a.k > kPicks || !layout_ok(a.lay, a.n_embd, a.n_ff)) { fail("invalid expert layout"); return; }
    const int tg = expert_type(a.lay, 0), tu = expert_type(a.lay, 1);
    const size_t rg = type_row_bytes(tg, a.n_embd), ru = type_row_bytes(tu, a.n_embd);
    parallel(s, size_t(a.k) * a.n_ff, [=](size_t z) {
        const size_t e = z / a.n_ff, c = z % a.n_ff;
        const auto* blob = reinterpret_cast<const uint8_t*>(a.plan_ptr[e]);
        if (!blob) { h[z] = 0.f; return; }
        const float g = quant_dot(tg, blob + a.lay.trellis[0] + c * rg, x, a.n_embd);
        const float u = quant_dot(tu, blob + a.lay.trellis[1] + c * ru, x, a.n_embd);
        h[z] = silu_product(g, u, a.limit);
    });
}
void moe_down(const x3::MoeArgs& a, const float* h, const float* sh_out, float* out, sycl::queue& s) {
    if (a.k < 0 || a.k > kPicks || !layout_ok(a.lay, a.n_embd, a.n_ff)) { fail("invalid expert layout"); return; }
    const int td = expert_type(a.lay, 2);
    const size_t rd = type_row_bytes(td, a.n_ff);
    parallel(s, size_t(a.n_embd), [=](size_t c) {
        float result = 0.f;
        for (int e = 0; e < a.k; ++e) {
            const auto* blob = reinterpret_cast<const uint8_t*>(a.plan_ptr[e]);
            if (blob) result += a.plan_w[e] * quant_dot(td, blob + a.lay.trellis[2] + c * rd, h + size_t(e) * a.n_ff, a.n_ff);
        }
        if (a.cpu_part && a.cpu_flag && *a.cpu_flag) result += a.cpu_part[c];
        if (sh_out) result += sh_out[c];
        out[c] = result;
    });
}
bool win_experts(const x3::WinArgs& a, sycl::queue& s) {
    if (a.T <= 0 || a.T > kWin || a.k < 0 || a.k > kPicks || !a.plan_ptr || !a.x || !a.h || !a.part || !a.out ||
        !layout_ok(a.lay, a.n_embd, a.n_ff)) return fail("invalid expert window");
    const int tg = expert_type(a.lay, 0), tu = expert_type(a.lay, 1), td = expert_type(a.lay, 2);
    const size_t rg = type_row_bytes(tg, a.n_embd), ru = type_row_bytes(tu, a.n_embd), rd = type_row_bytes(td, a.n_ff);
    parallel(s, size_t(a.T) * kPicks * a.n_ff, [=](size_t z) {
        const size_t entry = z / a.n_ff, t = entry / kPicks, pick = entry % kPicks, c = z % a.n_ff;
        const auto* blob = pick < size_t(a.k) ? reinterpret_cast<const uint8_t*>(a.plan_ptr[entry]) : nullptr;
        if (!blob) { a.h[z] = 0.f; return; }
        const float* x = a.x + t * a.x_stride;
        const float g = quant_dot(tg, blob + a.lay.trellis[0] + c * rg, x, a.n_embd);
        const float u = quant_dot(tu, blob + a.lay.trellis[1] + c * ru, x, a.n_embd);
        a.h[z] = silu_product(g, u, a.limit);
    });
    parallel(s, size_t(a.T) * kPicks * a.n_embd, [=](size_t z) {
        const size_t entry = z / a.n_embd, pick = entry % kPicks, c = z % a.n_embd;
        const auto* blob = pick < size_t(a.k) ? reinterpret_cast<const uint8_t*>(a.plan_ptr[entry]) : nullptr;
        a.part[z] = blob ? quant_dot(td, blob + a.lay.trellis[2] + c * rd, a.h + entry * a.n_ff, a.n_ff) : 0.f;
    });
    if (a.wait_ptr) wait_seq(a.wait_ptr, a.wait_val, s);
    const size_t cpu_stride = a.cpu_stride ? a.cpu_stride : size_t(a.n_embd);
    parallel(s, size_t(a.T) * a.n_embd, [=](size_t z) {
        const size_t t = z / a.n_embd, c = z % a.n_embd;
        float result = 0.f;
        for (int i = 0; i < a.k; ++i) if (a.plan_ptr[t * kPicks + i])
            result += a.plan_w[t * kPicks + i] * a.part[(t * kPicks + i) * a.n_embd + c];
        if (a.cpu_part && a.cpu_flag && a.cpu_flag[t]) result += a.cpu_part[t * cpu_stride + c];
        if (a.sh_out) result += a.sh_out[z];
        a.out[t * a.out_stride + c] = result;
    });
    return true;
}
void set_gate_up(const x3::SetArgs& a, const float* x, int ldx, const int* row_idx, uint16_t* h16, int ldh, sycl::queue& s) {
    if (a.n_exp <= 0 || a.max_rows <= 0) return;
    if (!layout_ok(a.lay, a.n_embd, a.n_ff)) { fail("invalid expert set layout"); return; }
    const int tg = expert_type(a.lay, 0), tu = expert_type(a.lay, 1);
    const size_t rg = type_row_bytes(tg, a.n_embd), ru = type_row_bytes(tu, a.n_embd);
    if (a.max_rows > iq::kLightRows && tg == tu && row_idx && iq_tiled_set_type(tg)) {
        auto q = light_args(a);
        q.xf = x; q.xf16 = tg == kTypeMXFP4 ? nullptr : a.x16;
        q.idx = row_idx; q.ldx = ldx; q.h = h16; q.ldo = ldh;
        if (!iq::set_gate_up(q, s)) fail("unsupported IQ expert set");
        return;
    }
    int skip_le = 0;
    if (tg == tu) {
        auto q = light_args(a);
        q.xf = x; q.idx = row_idx; q.ldx = ldx; q.h = h16; q.ldo = ldh;
        if (iq::light_gate_up(q, s)) {
            if (a.max_rows <= iq::kLightRows) return;
            skip_le = iq::kLightRows;
        }
    }
    const bool fp32_input = tg != kTypeMXFP4 && tu != kTypeMXFP4 && !iq::is_tq(tg) && !iq::is_tq(tu) && x;
    parallel(s, size_t(a.n_exp) * a.max_rows * a.n_ff, [=](size_t z) {
        const size_t c = z % a.n_ff, r = (z / a.n_ff) % a.max_rows, e = z / (size_t(a.n_ff) * a.max_rows);
        const int count = a.bounds[e + 1] - a.bounds[e];
        if (count <= skip_le || r >= size_t(count) || (a.skip_ge > 0 && count >= a.skip_ge)) return;
        const int local = a.bounds[e] - a.row0 + int(r), src = row_idx ? row_idx[local] : local;
        const uint8_t* blob = a.wbase + e * a.stride;
        float g = 0.f, u = 0.f;
        for (int k = 0; k < a.n_embd; ++k) {
            const float v = fp32_input || !a.x16 ? x[size_t(src) * ldx + k] : f16_value(a.x16[size_t(src) * ldx + k]);
            g = sycl::fma(weight(tg, blob + a.lay.trellis[0] + c * rg, k), v, g);
            u = sycl::fma(weight(tu, blob + a.lay.trellis[1] + c * ru, k), v, u);
        }
        h16[size_t(local) * ldh + c] = f16_bits(silu_product(g, u, a.limit));
    });
}
void set_down(const x3::SetArgs& a, const uint16_t* h16, int ldh, float* y, int ldy, sycl::queue& s) {
    if (a.n_exp <= 0 || a.max_rows <= 0) return;
    if (!layout_ok(a.lay, a.n_embd, a.n_ff)) { fail("invalid expert set layout"); return; }
    const int td = expert_type(a.lay, 2);
    const size_t rd = type_row_bytes(td, a.n_ff);
    int skip_le = 0;
    auto q = light_args(a);
    q.xh = h16; q.ldx = ldh; q.y = y; q.ldo = ldy;
    if (a.max_rows > iq::kLightRows && iq_tiled_set_type(td)) {
        if (!iq::set_down(q, s)) fail("unsupported IQ expert set");
        return;
    }
    if (iq::light_down(q, s)) {
        if (a.max_rows <= iq::kLightRows) return;
        skip_le = iq::kLightRows;
    }
    parallel(s, size_t(a.n_exp) * a.max_rows * a.n_embd, [=](size_t z) {
        const size_t c = z % a.n_embd, r = (z / a.n_embd) % a.max_rows, e = z / (size_t(a.n_embd) * a.max_rows);
        const int count = a.bounds[e + 1] - a.bounds[e];
        if (count <= skip_le || r >= size_t(count) || (a.skip_ge > 0 && count >= a.skip_ge)) return;
        const int local = a.bounds[e] - a.row0 + int(r);
        const uint8_t* row = a.wbase + e * a.stride + a.lay.trellis[2] + c * rd;
        float sum = 0.f;
        for (int k = 0; k < a.n_ff; ++k) sum = sycl::fma(weight(td, row, k), f16_value(h16[size_t(local) * ldh + k]), sum);
        y[size_t(local) * ldy + c] = sum;
    });
}

}  // namespace strata::ds4::gguf
