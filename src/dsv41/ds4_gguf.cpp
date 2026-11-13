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
#include "ds4_gguf_mv.hpp"
#include "ds4_tq_decode.hpp"
#include "ds4_numerics.cuh"
#include "strata/core/device.hpp"
#include "strata/core/per_device.hpp"
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <exception>
#include <limits>
#include <memory>
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
    q.bounds = a.bounds; q.host_bounds = a.host_bounds; q.row0 = a.row0; q.max_rows = a.max_rows; q.n_exp = a.n_exp;
    q.n_embd = a.n_embd; q.n_ff = a.n_ff; q.limit = a.limit; q.skip_ge = a.skip_ge;
    q.tg = expert_type(a.lay, 0); q.td = expert_type(a.lay, 2);
    q.native_f16_output = a.native_f16_output;
    return q;
}
bool iq_tiled_set_type(int type) {
    return type == iq::kIQ1_S || type == iq::kIQ1_M || type == iq::kIQ2_XXS ||
           type == iq::kIQ2_XS || type == iq::kIQ2_S || type == iq::kIQ3_XXS || type == kTypeMXFP4;
}

struct TokenCaps {
    size_t subgroup = 0, quant_group = 0, combine_group = 0;
    size_t max_alloc = 0, memory = 0, local_bytes = 0;
};
const TokenCaps& token_caps(sycl::queue& s) {
    static core::PerDevice<TokenCaps> cache;
    const auto device = s.get_device();
    return cache.get(device, [&] {
        TokenCaps caps;
        const size_t max_group = std::min(device.get_info<sycl::info::device::max_work_group_size>(),
                                         device.get_info<sycl::info::device::max_work_item_sizes<1>>()[0]);
        const auto subgroups = device.get_info<sycl::info::device::sub_group_sizes>();
        for (const size_t width : {size_t(32), size_t(16)})
            if (width <= max_group && std::find(subgroups.begin(), subgroups.end(), width) != subgroups.end()) {
                caps.subgroup = width;
                break;
            }
        // upstream q8_quant_kernel: up to eight independent 32-value quantization blocks per work-group.
        if (caps.subgroup) caps.quant_group = std::min<size_t>(8, max_group / caps.subgroup) * caps.subgroup;
        // upstream mx_combine_kernel: one output per work-item, at most 128 per work-group.
        caps.combine_group = std::min<size_t>(128, max_group);
        const auto max_size = uint64_t(std::numeric_limits<size_t>::max());
        caps.max_alloc = size_t(std::min<uint64_t>(device.get_info<sycl::info::device::max_mem_alloc_size>(), max_size));
        caps.memory = size_t(std::min<uint64_t>(device.get_info<sycl::info::device::global_mem_size>(), max_size));
        caps.local_bytes = size_t(std::min<uint64_t>(device.get_info<sycl::info::device::local_mem_size>(), max_size));
        if (const auto limit = core::max_alloc_limit_bytes()) caps.max_alloc = size_t(std::min<uint64_t>(caps.max_alloc, limit));
        if (const auto limit = core::vram_limit_bytes()) caps.memory = size_t(std::min<uint64_t>(caps.memory, limit));
        return caps;
    });
}

// Quantized-input counterpart of upstream Scratch. Keep the queue/context alive with the allocations. Each host thread
// has its own entries: concurrent callers cannot interleave quantization and consumption of the same scratch.
// Equal devices or contexts alone are insufficient: reuse requires the exact in-order queue.
struct TokenScratch {
    sycl::queue queue;
    int8_t* codes = nullptr;
    float *scales = nullptr, *part = nullptr;
    size_t code_capacity = 0, part_capacity = 0;
    explicit TokenScratch(const sycl::queue& s) : queue(s) {}
    TokenScratch(const TokenScratch&) = delete;
    TokenScratch& operator=(const TokenScratch&) = delete;
    ~TokenScratch() {
        if (!codes && !scales && !part) return;
        try {
            queue.wait(); // Destruction only; every submitted reader must finish before its storage is released.
            if (codes) sycl::free(codes, queue);
            if (scales) sycl::free(scales, queue);
            if (part) sycl::free(part, queue);
        } catch (...) { std::terminate(); }
    }
    bool reserve(size_t nc, size_t np, const TokenCaps& caps) {
        if (nc <= code_capacity && np <= part_capacity) return true;
        nc = std::max(nc, code_capacity);
        np = std::max(np, part_capacity);
        if (nc > caps.max_alloc || nc / 32 > caps.max_alloc / sizeof(float) ||
            np > caps.max_alloc / sizeof(float)) return false;
        const size_t scale_bytes = (nc / 32) * sizeof(float), part_bytes = np * sizeof(float);
        if (nc > caps.memory || scale_bytes > caps.memory - nc || part_bytes > caps.memory - nc - scale_bytes)
            return false;
        int8_t* next_codes = nullptr;
        float *next_scales = nullptr, *next_part = nullptr;
        const auto release_next = [&] {
            if (next_codes) { auto* old = next_codes; next_codes = nullptr; sycl::free(old, queue); }
            if (next_scales) { auto* old = next_scales; next_scales = nullptr; sycl::free(old, queue); }
            if (next_part) { auto* old = next_part; next_part = nullptr; sycl::free(old, queue); }
        };
        try {
            next_codes = sycl::malloc_device<int8_t>(nc, queue);
            next_scales = sycl::malloc_device<float>(nc / 32, queue);
            if (np) next_part = sycl::malloc_device<float>(np, queue);
            if (!next_codes || !next_scales || (np && !next_part)) { release_next(); return false; }
            // Growth may retire outstanding work. The steady-state path above has neither a wait nor allocation.
            if (codes || scales || part) queue.wait_and_throw();
            code_capacity = 0; part_capacity = 0;
            if (codes) { auto* old = codes; codes = nullptr; sycl::free(old, queue); }
            if (scales) { auto* old = scales; scales = nullptr; sycl::free(old, queue); }
            if (part) { auto* old = part; part = nullptr; sycl::free(old, queue); }
        } catch (...) { release_next(); throw; }
        codes = next_codes; scales = next_scales; part = next_part;
        code_capacity = nc; part_capacity = np;
        return true;
    }
};

TokenScratch* quant_scratch(size_t nc, size_t np, const TokenCaps& caps, sycl::queue& s) {
    thread_local std::vector<std::unique_ptr<TokenScratch>> entries;
    for (auto& entry : entries)
        if (entry->queue == s) return entry->reserve(nc, np, caps) ? entry.get() : nullptr;
    auto entry = std::make_unique<TokenScratch>(s);
    if (!entry->reserve(nc, np, caps)) return nullptr;
    auto* result = entry.get();
    entries.push_back(std::move(entry));
    return result;
}
TokenScratch* token_scratch(const x3::MoeArgs& a, const TokenCaps& caps, sycl::queue& s) {
    const size_t picks = size_t(a.k), embd = size_t(a.n_embd), ff = size_t(a.n_ff);
    if (!picks || ff > std::numeric_limits<size_t>::max() / picks ||
        embd > std::numeric_limits<size_t>::max() / picks) return nullptr;
    return quant_scratch(std::max(embd, picks * ff), picks * embd, caps, s);
}

// Upstream mv's BF16/Q8_0 job partition and quant_add/quant_flush. Keep unsupported
// formats or column counts on the existing path; all metadata below is copied by value.
// A set of jobs is independent, as in upstream's batching by format rather than job order.
// The K types that decode without a table (Q2_K to Q6_K: the dense attention and shared expert) take upstream's
// low-bit GEMV on the same int8 activations as Q8_0, the jobs of one type in one launch.
template <int T> constexpr bool kMvNoTable = iq::sycl_decode::Ty<T>::TAB == 0 && T != iq::sycl_decode::kMXFP4;
bool mv_k(const x3::Mat& m) {
    bool ok = false;
    if (m.fmt == kFmtGG && m.k % 256 == 0) iq::sycl_decode::dispatch(m.K, [&](auto t) { ok = kMvNoTable<t.value>; });
    return ok;
}
bool mv_iq(const x3::Mat& m) { return m.fmt == kFmtGG && m.K != kTypeMXFP4 && iq_tiled_set_type(m.K) && m.k % 256 == 0; }
// The other ggml block types ds4_iq decodes (IQ4_NL, IQ4_XS, IQ3_S and the like), and a K type mv_batch left, take the
// same int8 GEMV a job at a time, as upstream's mv sends every such matrix to iq::mv; Q5_K keeps q5_mv.
bool mv_low(const x3::Mat& m) {
    return m.fmt == kFmtGG && m.K != kTypeMXFP4 && m.K != iq::kQ5_K && iq::supported(m.K) && m.k % 256 == 0;
}
unsigned mv_batch(const x3::MvJob* jobs, int n, sycl::queue& s) {
    bool any = false;
    for (int i = 0; i < n; ++i)
        any = any || ((jobs[i].m->fmt == kFmtBF16 || jobs[i].m->fmt == kFmtQ8_0 || mv_k(*jobs[i].m) || (mv_iq(*jobs[i].m) && jobs[i].ncol > 1)) &&
                      jobs[i].ncol <= x3::kMaxMvCols);
    if (!any) return 0;
    const auto& caps = token_caps(s);
    if (!caps.subgroup) return 0;
    mv_detail::Jobs batch[4];
    mv_detail::QuantItems items;
    int counts[4] = {}, max_n[4] = {}, max_cols[4] = {};
    bool aligned[3] = {true, true, true};
    size_t offsets[x3::kMaxMvJobs] = {}, code_count = 0;
    int iq_types[x3::kMaxMvJobs] = {}, k_types[x3::kMaxMvJobs] = {};
    unsigned accepted = 0;
    for (int i = 0; i < n; ++i) {
        const auto& job = jobs[i];
        const auto& m = *job.m;
        if ((m.fmt != kFmtBF16 && m.fmt != kFmtQ8_0 && !mv_k(m) && !(mv_iq(m) && job.ncol > 1)) || job.ncol > x3::kMaxMvCols) continue;
        const int kind = m.fmt == kFmtQ8_0 ? 1 : m.fmt == kFmtGG ? (mv_iq(m) ? 3 : 2) : 0;
        if (kind == 3) iq_types[counts[kind]] = m.K;
        if (kind == 2) k_types[counts[kind]] = m.K;
        auto& b = batch[kind].j[counts[kind]++];
        b.w = m.raw; b.ld = row_stride(m); b.x = job.xf; b.xcs = job.xf_cs ? job.xf_cs : size_t(m.k);
        b.y = job.y; b.ycs = job.y_cs ? job.y_cs : size_t(m.n); b.bias = job.bias; b.alpha = job.alpha;
        b.k = m.k; b.n = m.n; b.ncol = job.ncol;
        max_n[kind] = std::max(max_n[kind], m.n);
        max_cols[kind] = std::max(max_cols[kind], job.ncol);
        if (kind == 0) {
            const bool xa = (reinterpret_cast<uintptr_t>(b.x) & 15u) == 0 && (b.ncol == 1 || b.xcs % 4 == 0);
            aligned[0] = aligned[0] && xa && m.k % 8 == 0 && b.ld % 8 == 0 &&
                         (reinterpret_cast<uintptr_t>(b.w) & 15u) == 0;
        } else {
            const size_t ldx = b.ncol == 1 ? size_t(b.k) : b.xcs;
            int item = 0;
            for (; item < items.n; ++item)
                if (items.x[item] == b.x && items.ldx[item] == ldx && items.k[item] == b.k && items.rows[item] == b.ncol)
                    break;
            if (item == items.n) {
                if (size_t(b.k) > (std::numeric_limits<size_t>::max() - code_count) / size_t(b.ncol)) return 0;
                offsets[item] = code_count;
                code_count += size_t(b.k) * b.ncol;
                items.x[item] = b.x; items.ldx[item] = ldx; items.k[item] = b.k; items.rows[item] = b.ncol;
                ++items.n;
            }
            if (kind == 1)
                aligned[1] = aligned[1] && b.ld % 4 == 0 && (reinterpret_cast<uintptr_t>(b.w) & 3u) == 0 &&
                             (b.k / 32) % 2 == 0;
        }
        accepted |= 1u << i;
    }
    if (!accepted) return 0;
    sycl_detail::require_in_order(s);
    size_t warps[3] = {caps.quant_group / caps.subgroup, caps.quant_group / caps.subgroup,
                       caps.quant_group / caps.subgroup};
    if (aligned[1]) {
        const size_t fits = caps.local_bytes / (caps.subgroup * kQ8Bytes);
        if (fits) warps[1] = std::min(warps[1], fits);
        else aligned[1] = false; // Upstream ST=false global-byte path, without a local stage.
    }
    if (items.n) {
        auto* scratch = quant_scratch(code_count, 0, caps, s);
        if (!scratch) return 0;
        int max_k = 0, max_rows = 0;
        for (int j = 0; j < items.n; ++j) {
            items.q[j] = scratch->codes + offsets[j];
            items.d[j] = scratch->scales + offsets[j] / 32;
            max_k = std::max(max_k, items.k[j]); max_rows = std::max(max_rows, items.rows[j]);
        }
        for (int kind = 1; kind < 4; ++kind)
            for (int j = 0; j < counts[kind]; ++j) {
                auto& b = batch[kind].j[j];
                const size_t ldx = b.ncol == 1 ? size_t(b.k) : b.xcs;
                for (int item = 0; item < items.n; ++item)
                    if (items.x[item] == b.x && items.ldx[item] == ldx && items.k[item] == b.k && items.rows[item] == b.ncol) {
                        b.xq = items.q[item]; b.xd = items.d[item]; break;
                    }
            }
        if (caps.subgroup == 32) mv_detail::quantize<32>(items, max_k, max_rows, warps[0], s);
        else mv_detail::quantize<16>(items, max_k, max_rows, warps[0], s);
    }
    for (int kind = 0; kind < 2; ++kind) {
        if (caps.subgroup == 32)
            mv_detail::launch<32>(kind != 0, batch[kind], counts[kind], max_n[kind], max_cols[kind], aligned[kind], warps[kind], s);
        else mv_detail::launch<16>(kind != 0, batch[kind], counts[kind], max_n[kind], max_cols[kind], aligned[kind], warps[kind], s);
    }
    for (unsigned left = (1u << counts[2]) - 1; left;) {   // a launch for each K type among the jobs
        int type = 0, count = 0, rows = 0, cols = 0;
        mv_detail::Jobs group;
        for (int j = 0; j < counts[2]; ++j) {
            if (!(left >> j & 1u) || (count && k_types[j] != type)) continue;
            type = k_types[j];
            group.j[count++] = batch[2].j[j];
            rows = std::max(rows, batch[2].j[j].n); cols = std::max(cols, batch[2].j[j].ncol);
            left &= ~(1u << j);
        }
        iq::sycl_decode::dispatch(type, [&](auto t) {
            if constexpr (kMvNoTable<t.value>) {
                if (caps.subgroup == 32) mv_detail::launch_iq<t.value, 32>(group, count, rows, cols, warps[2], s);
                else mv_detail::launch_iq<t.value, 16>(group, count, rows, cols, warps[2], s);
            }
        });
    }
    if (counts[3]) {
        iq::MvJob jobs_iq[x3::kMaxMvJobs] = {};
        for (int i = 0; i < counts[3]; ++i) {
            const auto& b = batch[3].j[i]; auto& j = jobs_iq[i];
            j.w = static_cast<const uint8_t*>(b.w); j.ld = b.ld; j.type = iq_types[i];
            j.k = b.k; j.n = b.n; j.ncol = b.ncol; j.xq = b.xq; j.xd = b.xd;
            j.y = b.y; j.ycs = b.ycs; j.bias = b.bias; j.alpha = b.alpha;
        }
        iq::mv(jobs_iq, counts[3], s);
    }
    return accepted;
}

template<int SG>
void q8_quant_token(const float* x, size_t ldx, int rows, int k, int8_t* q, float* d,
                    size_t local, sycl::queue& s) {
    const size_t blocks = size_t(k) / 32, warps = local / SG;
    const size_t groups_per_row = (blocks + warps - 1) / warps;
    // Upstream q8_quant_kernel: one subgroup per storage block. SG16 holds the two halves of that same block;
    // its first max is exactly the SG32 xor-16 step, followed by the remaining xor reductions.
    sycl_detail::launch(s, sycl::nd_range<1>(sycl::range<1>(size_t(rows) * groups_per_row * local), sycl::range<1>(local)),
                   [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(STRATA_SUB_GROUP(SG))]] {
        const size_t row = item.get_group_linear_id() / groups_per_row;
        const size_t block = (item.get_group_linear_id() % groups_per_row) * warps + item.get_local_linear_id() / SG;
        if (block >= blocks) return; // Uniform within each subgroup; no work-group barrier follows.
        const auto subgroup = item.get_sub_group();
        const size_t lane = subgroup.get_local_linear_id();
        const float v0 = x[row * ldx + block * 32 + lane];
        float v1 = 0.f, amax = sycl::fabs(v0);
        if constexpr (SG == 16) {
            v1 = x[row * ldx + block * 32 + lane + 16];
            amax = sycl::fmax(amax, sycl::fabs(v1));
        }
        for (int offset = SG / 2; offset > 0; offset /= 2)
            amax = sycl::fmax(amax, sycl::permute_group_by_xor(subgroup, amax, offset));
        const float scale = amax / 127.f;
        q[row * size_t(k) + block * 32 + lane] = int8_t(amax > 0.f ? sycl::rint(v0 / scale) : 0.f);
        if constexpr (SG == 16)
            q[row * size_t(k) + block * 32 + lane + 16] = int8_t(amax > 0.f ? sycl::rint(v1 / scale) : 0.f);
        if (lane == 0) d[row * blocks + block] = scale;
    });
}
void q8_quant_token(const float* x, int rows, int k, TokenScratch& scratch, const TokenCaps& caps, sycl::queue& s) {
    if (caps.subgroup == 32)
        q8_quant_token<32>(x, size_t(k), rows, k, scratch.codes, scratch.scales, caps.quant_group, s);
    else q8_quant_token<16>(x, size_t(k), rows, k, scratch.codes, scratch.scales, caps.quant_group, s);
}

bool token_iq_args(const x3::MoeArgs& a, iq::Args& q) {
    const int tg = expert_type(a.lay, 0), tu = expert_type(a.lay, 1), td = expert_type(a.lay, 2);
    if (a.k <= 0 || tg != tu || iq::is_tq(tg) || iq::is_tq(td) ||
        (!iq::supported(tg) && tg != kTypeMXFP4) || (!iq::supported(td) && td != kTypeMXFP4)) return false;
    q.plan_ptr = a.plan_ptr;
    for (int r = 0; r < 3; ++r) q.tr[r] = a.lay.trellis[r];
    q.n_embd = a.n_embd; q.n_ff = a.n_ff; q.limit = a.limit; q.tg = tg; q.td = td;
    return true;
}

void token_combine(const x3::MoeArgs& a, const float* part, const float* shared, float* out,
                   size_t local, sycl::queue& s) {
    const size_t groups = (size_t(a.n_embd) + local - 1) / local;
    // Upstream mx_combine_kernel (T=1): plan-order FMA, then the CPU lane and shared expert as separate additions.
    sycl_detail::launch(s, sycl::nd_range<1>(sycl::range<1>(groups * local), sycl::range<1>(local)),
                   [=](sycl::nd_item<1> item) {
        const size_t row = item.get_global_linear_id();
        if (row >= size_t(a.n_embd)) return;
        float sum = 0.f;
        for (int e = 0; e < a.k; ++e) {
            const float value = a.plan_ptr[e] ? part[size_t(e) * a.n_embd + row] : 0.f;
            sum = sycl::fma(a.plan_w[e], value, sum);
        }
        if (a.cpu_part && a.cpu_flag && *a.cpu_flag) sum += a.cpu_part[row];
        out[row] = sum + (shared ? shared[row] : 0.f);
    });
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
    sycl_detail::submit(s, [&](sycl::handler& h) {
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
    sycl_detail::submit(s, [&](sycl::handler& h) {
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
bool prepare(sycl::queue& s) { return s.has_property<sycl::property::queue::in_order>(); } // Scratch grows from actual token shapes.
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
    const unsigned batched = mv_batch(jobs, n, s);
    for (int i = 0; i < n; ++i) {
        if (batched & (1u << i)) continue;
        const x3::MvJob j = jobs[i];
        // Upstream's low-bit jobs quantize each activation row once, then share
        // it across the output rows. Keep the scalar path for unavailable scratch.
        if (mv_low(*j.m)) {
            const auto& caps = token_caps(s);
            const size_t xcs = j.xf_cs ? j.xf_cs : size_t(j.m->k);
            if (auto* scratch = caps.subgroup ? quant_scratch(size_t(j.ncol) * j.m->k, 0, caps, s) : nullptr) {
                if (caps.subgroup == 32)
                    q8_quant_token<32>(j.xf, xcs, j.ncol, j.m->k, scratch->codes, scratch->scales, caps.quant_group, s);
                else q8_quant_token<16>(j.xf, xcs, j.ncol, j.m->k, scratch->codes, scratch->scales, caps.quant_group, s);
                iq::MvJob q{};
                q.w = static_cast<const uint8_t*>(j.m->raw); q.ld = row_stride(*j.m); q.type = j.m->K;
                q.k = j.m->k; q.n = j.m->n; q.ncol = j.ncol; q.xq = scratch->codes; q.xd = scratch->scales;
                q.y = j.y; q.ycs = j.y_cs ? j.y_cs : size_t(j.m->n); q.bias = j.bias; q.alpha = j.alpha;
                if (!iq::mv(&q, 1, s)) return fail("unsupported IQ matvec");
                continue;
            }
        }
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
    sycl_detail::require_in_order(s);
    iq::Args q;
    if (token_iq_args(a, q)) {
        const auto& caps = token_caps(s);
        if (caps.subgroup) if (auto* scratch = token_scratch(a, caps, s)) {
            q.h = h; q.xq = scratch->codes; q.xd = scratch->scales;
            q8_quant_token(x, 1, a.n_embd, *scratch, caps, s);
            if (!iq::gate_up(q, a.k, 1, s)) fail("unsupported IQ token gate/up");
            return;
        }
    }
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
    sycl_detail::require_in_order(s);
    iq::Args q;
    if (token_iq_args(a, q)) {
        const auto& caps = token_caps(s);
        if (caps.subgroup) if (auto* scratch = token_scratch(a, caps, s)) {
            q.part = scratch->part; q.hq = scratch->codes; q.hd = scratch->scales;
            q8_quant_token(h, a.k, a.n_ff, *scratch, caps, s);
            if (!iq::down(q, a.k, 1, s)) { fail("unsupported IQ token down"); return; }
            token_combine(a, scratch->part, sh_out, out, caps.combine_group, s);
            return;
        }
    }
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
    const size_t cpu_stride = a.cpu_stride ? a.cpu_stride : size_t(a.n_embd);
    const auto combine = [&] {
        if (a.wait_ptr) wait_seq(a.wait_ptr, a.wait_val, s);
        parallel(s, size_t(a.T) * a.n_embd, [=](size_t z) {
            const size_t t = z / a.n_embd, c = z % a.n_embd;
            float result = 0.f;
            for (int i = 0; i < a.k; ++i) if (a.plan_ptr[t * kPicks + i])
                result += a.plan_w[t * kPicks + i] * a.part[(t * kPicks + i) * a.n_embd + c];
            if (a.cpu_part && a.cpu_flag && a.cpu_flag[t]) result += a.cpu_part[t * cpu_stride + c];
            if (a.sh_out) result += a.sh_out[z];
            a.out[t * a.out_stride + c] = result;
        });
    };
    // Upstream win_experts for low-bit experts: the window's distinct experts as moe_win_group's groups, each read once
    // for its tokens by ds4_iq's kernels on int8 rows (x, then h, through the same scratch on the in-order queue).
    x3::MoeArgs ma;
    ma.plan_ptr = a.plan_ptr; ma.lay = a.lay; ma.k = a.k; ma.n_embd = a.n_embd; ma.n_ff = a.n_ff; ma.limit = a.limit;
    iq::Args q;
    if (a.grp_ptr && a.grp_start && a.grp_ent && a.n_groups && token_iq_args(ma, q)) {
        const auto& caps = token_caps(s);
        const size_t entries = size_t(a.T) * kPicks;
        TokenScratch* scratch = caps.subgroup ?
            quant_scratch(std::max(size_t(a.T) * a.n_embd, entries * a.n_ff), 0, caps, s) : nullptr;
        if (scratch) {
            sycl_detail::require_in_order(s);
            q.grp_ptr = a.grp_ptr; q.grp_start = a.grp_start; q.grp_ent = a.grp_ent; q.n_groups = a.n_groups;
            q.h = a.h; q.part = a.part;
            q.xq = scratch->codes; q.xd = scratch->scales;
            q.hq = scratch->codes; q.hd = scratch->scales;
            const int groups = a.T * a.k;   // at most (the groups past the real count return at once)
            if (caps.subgroup == 32)
                q8_quant_token<32>(a.x, a.x_stride, a.T, a.n_embd, scratch->codes, scratch->scales, caps.quant_group, s);
            else q8_quant_token<16>(a.x, a.x_stride, a.T, a.n_embd, scratch->codes, scratch->scales, caps.quant_group, s);
            if (!iq::gate_up(q, groups, a.T, s)) return fail("unsupported IQ window gate/up");
            q8_quant_token(a.h, int(entries), a.n_ff, *scratch, caps, s);
            if (!iq::down(q, groups, a.T, s)) return fail("unsupported IQ window down");
            combine();
            return true;
        }
    }
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
    combine();
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
