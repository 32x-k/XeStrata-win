// SPDX-FileCopyrightText: 2026 Peasant Smith and the Project Maya contributors
// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/main/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// Keep matrix kernels in their own device image: portable light/FMA kernels must
// remain loadable on devices without the matrix or fp16 aspects.
#include "ds4_iq_matrix.hpp"
#include "ds4_iq_decode.hpp"
#include "ds4_quant_sycl.hpp"
#include "kernels/xe/mma_gemm.hpp"
#include "kernels/xe/amd_wmma.hpp"
#include "kernels/xe/device_target.hpp"
#include "strata/core/per_device.hpp"
#define STRATA_DEVICE_ARCH_TAG_TU dsv41_iq_matrix
#include "strata/kernels/device_arch_tag.hpp"
#include <algorithm>
#include <cstddef>
#include <vector>
#include <memory>
#include <limits>
#include "strata/core/runtime.hpp"

#if (defined(__SYCL_DEVICE_ONLY__) && defined(__AMDGCN__) && !STRATA_AMD_WMMA) || \
    (STRATA_NV_ARCH > 0 && STRATA_NV_ARCH < 700)
#define STRATA_IQ_MATRIX_SCALAR 1
#else
#define STRATA_IQ_MATRIX_SCALAR 0
#endif

namespace strata::ds4::gguf::iq::matrix_detail {
template <int T, bool GU, int STAGE> class LargeKernel;
namespace {
using namespace sycl_decode;
namespace xe = strata::kernels::xe;
namespace mx = sycl::ext::oneapi::experimental::matrix;

// The exact combination is checked through the existing matrix dispatcher.
// K advances by one quantization sub-block, in two matrix instructions.
constexpr size_t MM = 16, MN = 16, MK = 16, SG = 32, QK = 32;
struct Caps { size_t group = 0, local = 0; std::vector<size_t> subgroups; };
const Caps& caps(const sycl::queue& s) {
    const auto device = s.get_device();
    static core::PerDevice<Caps> cache;
    return cache.get(device, [&] {
        return Caps{std::min(device.get_info<sycl::info::device::max_work_group_size>(),
                             device.get_info<sycl::info::device::max_work_item_sizes<1>>()[0]),
                    size_t(device.get_info<sycl::info::device::local_mem_size>()),
                    device.get_info<sycl::info::device::sub_group_sizes>()};
    });
}

#include "ds4_iq_matrix_grid.inc"

// Matrix dispatch already requires fp16. Native RTE is used only where the
// result is a finite normal half; the bit-defined path keeps subnormals and
// special values independent of the device's half denormal modes.
uint16_t matrix_half_bits(float value) {
    const uint32_t bits = sycl::bit_cast<uint32_t>(value), magnitude = bits & 0x7fffffffu;
    if (magnitude == 0) return uint16_t(bits >> 16);
    if (magnitude >= 0x38800000u && magnitude <= 0x477fe000u) [[likely]] {
        const auto rounded = sycl::vec<float, 1>(value).convert<sycl::half, sycl::rounding_mode::rte>();
        return sycl::bit_cast<uint16_t>(sycl::half(rounded[0]));
    }
    return sycl_detail::f16_bits(value);
}
float matrix_half_value(uint16_t bits) {
    const uint16_t magnitude = bits & 0x7fffu, exponent = bits & 0x7c00u;
    if (magnitude == 0) return sycl::bit_cast<float>(uint32_t(bits & 0x8000u) << 16);
    if (exponent != 0 && exponent != 0x7c00u) [[likely]]
        return float(sycl::bit_cast<sycl::half>(bits));
    return sycl_detail::f16_value(bits);
}

template <int T>
uint16_t matrix_weight(const Sub& sub, int j) {
    const uint32_t byte = (uint32_t(sub.q[j / 4]) >> (8 * (j % 4))) & 255u;
    const int code = int(byte ^ 128u) - 128;
    // MXFP4 rounds the decoded product once. Rounding its power-of-two scale
    // first would lose small representable products or turn zero into NaN.
    if constexpr (T == kMXFP4) return matrix_half_bits(sub.d * float(code));
    // Upstream hipk first rounds the sub-block scale, then its product with the
    // integer code. IQ1's +/-1/8 offset is already in its signed integer codes.
    const float scale = matrix_half_value(matrix_half_bits(sub.d * float(j < 16 ? sub.s0 : sub.s1)));
    return matrix_half_bits(scale * float(code));
}

template <int T, bool GU>
bool launch_base(const SetArgs& a, sycl::queue& s) {
    static_assert(Ty<T>::MIN == 0);
    const int n = GU ? a.n_ff : a.n_embd, k = GU ? a.n_embd : a.n_ff;
    if (a.n_exp <= 0 || a.max_rows <= kLightRows || n <= 0 || k <= 0 ||
        (a.skip_ge > 0 && a.skip_ge <= kLightRows + 1)) return false;
    if constexpr (T == kMXFP4) {
        // The GGUF MXFP4 matrix path gathers scalars, including odd strides.
        if (k % 64 || n % 64 || (GU ? !a.xf : !a.xh)) return false;
    } else {
        // This is upstream's default runr input-precision boundary, not our tile
        // shape: its 128-column eligibility leaves narrower outputs on FP32 FMA.
        if (k % 256 || n % 128 || a.ldx % 8 || a.ldo % 2) return false;
        const void* input = GU ? static_cast<const void*>(a.xf) : static_cast<const void*>(a.xh);
        if constexpr (GU) { if (!a.xf && !a.xf16) return false; }
        else if (!a.xh) return false;
        if ((reinterpret_cast<uintptr_t>(input) & 15u) || (GU && a.xf16 && (reinterpret_cast<uintptr_t>(a.xf16) & 15u)))
            return false;
    }
    if (!s.get_device().has(sycl::aspect::fp16) || !xe::mma_shape_is(s, int(MM), int(MN), int(MK), int(SG)))
        return false;
    const auto& c = caps(s);
    if (std::find(c.subgroups.begin(), c.subgroups.end(), SG) == c.subgroups.end()) return false;
    constexpr size_t roles = GU ? 2 : 1;
    constexpr size_t table_words = Ty<T>::TAB > 0 ? Ty<T>::TAB : 1;
    // Balance subgroup rows and columns from the reported group limit. Each
    // subgroup keeps the same matrix shape and ascending-K accumulation, while
    // a work-group's expanded weights serve all of its row subgroups.
    size_t row_groups = 1;
    const size_t needed = (size_t(a.max_rows) + MM - 1) / MM;
    while (row_groups * 2 <= needed && (row_groups * 2) * (row_groups * 2) <= c.group / SG)
        row_groups *= 2;
    size_t column_groups = 0, rows = 0;
    for (;;) {
        rows = row_groups * MM;
        const size_t fixed = table_words * sizeof(uint32_t) + rows * (QK * sizeof(uint16_t) + sizeof(int));
        const size_t per_column = roles * MN * (QK * sizeof(uint16_t) + rows * sizeof(float));
        if (c.local > fixed)
            column_groups = std::min({c.group / (SG * row_groups), (c.local - fixed) / per_column,
                                      (size_t(n) + MN - 1) / MN});
        if (column_groups) break;
        if (row_groups == 1) return false;
        row_groups /= 2;
    }
    sycl_detail::require_in_order(s);
    const size_t bn = column_groups * MN, threads = row_groups * column_groups * SG;
    const size_t nc = (size_t(n) + bn - 1) / bn, nr = (size_t(a.max_rows) + rows - 1) / rows;
    const size_t rb = row_bytes_t<T>(k);
    const int skip = std::max(a.skip_le, kLightRows);
    const MatrixGrid grid = matrix_grid(a, s, rows, nc, skip);
    const size_t groups = grid.groups;
    if (!groups) return true;
    sycl_detail::submit(s, [&](sycl::handler& h) {
        sycl::local_accessor<uint32_t, 1> table_data(sycl::range<1>(table_words), h);
        sycl::local_accessor<int, 1> source(sycl::range<1>(rows), h);
        sycl::local_accessor<sycl::half, 1> xs(sycl::range<1>(rows * QK), h), ws(sycl::range<1>(roles * bn * QK), h);
        sycl::local_accessor<float, 1> result(sycl::range<1>(roles * rows * bn), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(groups * threads), sycl::range<1>(threads)),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const size_t id = it.get_local_linear_id(), group = it.get_group_linear_id();
            const size_t e = grid.jobs ? size_t(grid.jobs[group / nc][0]) : group / (nc * nr);
            const size_t r0 = grid.jobs ? size_t(grid.jobs[group / nc][1]) : ((group / nc) % nr) * rows;
            const size_t c0 = (group % nc) * bn;
            const int first = a.bounds[e] - a.row0, count = a.bounds[e + 1] - a.bounds[e];
            if (count <= skip || (a.skip_ge > 0 && count >= a.skip_ge) || r0 >= size_t(count)) return;
            if constexpr (Ty<T>::TAB > 0)
                for (size_t j = id; j < size_t(Ty<T>::TAB); j += threads) table_data[j] = table<T>()[j];
            if (id < rows) {
                const size_t r = r0 + id, row = size_t(first) + r;
                source[id] = r < size_t(count) && r < size_t(a.max_rows) ?
                             (GU && a.idx ? a.idx[row] : int(row)) : -1;
            }
            it.barrier(sycl::access::fence_space::local_space);
            const auto* grids = table_data.template get_multi_ptr<sycl::access::decorated::no>().get();
            const auto xp = xs.template get_multi_ptr<sycl::access::decorated::no>();
            const auto wp = ws.template get_multi_ptr<sycl::access::decorated::no>();
            const auto op = result.template get_multi_ptr<sycl::access::decorated::no>();
            const auto sg = it.get_sub_group();
            const size_t row_block = (sg.get_group_linear_id() / column_groups) * MM;
            const size_t column = (sg.get_group_linear_id() % column_groups) * MN;
#if STRATA_AMD_WMMA
            xe::wmma::C accum[roles];
            for (size_t role = 0; role < roles; ++role) xe::wmma::fill(sg, accum[role], 0.0f);
#elif STRATA_IQ_MATRIX_SCALAR
            // The same device-pass exclusions as mma_gemm. Keep a real scalar
            // body where matrix instructions cannot be compiled.
            float accum[roles][MM * MN / SG] = {};
#else
            mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, MM, MN> accum[roles];
            for (size_t role = 0; role < roles; ++role) mx::joint_matrix_fill(sg, accum[role], 0.0f);
#endif
            const uint8_t* blob = a.wbase + e * a.stride;
            for (int kb = 0; kb < k / int(QK); ++kb) {
                for (size_t j = id; j < roles * bn; j += threads) {
                    const size_t role = j / bn, col = j % bn;
                    Sub sub{};
                    if (c0 + col < size_t(n))
                        decode<T>(blob + (GU ? (role ? a.tr[1] : a.tr[0]) : a.tr[2]) + (c0 + col) * rb, kb, grids, sub);
                    // Fixed QK indices retain the adopted register expansion.
#pragma unroll
                    for (size_t q = 0; q < QK; ++q)
                        ws[(role * bn + col) * QK + q] = sycl::bit_cast<sycl::half>(matrix_weight<T>(sub, int(q)));
                }
                for (size_t j = id; j < rows * QK; j += threads) {
                    const int src = source[j / QK];
                    uint16_t value = 0;
                    if (src >= 0) {
                        const size_t at = size_t(src) * a.ldx + size_t(kb) * QK + j % QK;
                        if constexpr (GU && T == kMXFP4) value = sycl_detail::f16_bits(a.xf[at]);
                        else value = GU ? (a.xf16 ? a.xf16[at] : sycl_detail::f16_bits(a.xf[at])) : a.xh[at];
                    }
                    xs[j] = sycl::bit_cast<sycl::half>(value);
                }
                it.barrier(sycl::access::fence_space::local_space);
                for (size_t kk = 0; kk < QK; kk += MK) {
#if STRATA_AMD_WMMA
                    xe::wmma::A<sycl::half> av;
                    xe::wmma::load(sg, av, xp + std::ptrdiff_t(row_block * QK + kk), QK);
                    for (size_t role = 0; role < roles; ++role) {
                        xe::wmma::Bc<sycl::half> bv;
                        xe::wmma::load(sg, bv, wp + std::ptrdiff_t((role * bn + column) * QK + kk), QK);
                        xe::wmma::mad(sg, accum[role], av, bv, accum[role]);
                    }
#elif STRATA_IQ_MATRIX_SCALAR
                    const size_t lane = sg.get_local_linear_id();
                    for (size_t j = 0; j < MM * MN / SG; ++j) {
                        const size_t out = j * SG + lane, r = out / MN, col = column + out % MN;
                        for (size_t q = kk; q < kk + MK; ++q) {
                            const float x = sycl_detail::f16_value(sycl::bit_cast<uint16_t>(xs[(row_block + r) * QK + q]));
                            for (size_t role = 0; role < roles; ++role)
                                accum[role][j] = sycl::fma(x, sycl_detail::f16_value(sycl::bit_cast<uint16_t>(
                                    ws[(role * bn + col) * QK + q])), accum[role][j]);
                        }
                    }
#else
                    mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::a, MM, MK, mx::layout::row_major> av;
                    mx::joint_matrix_load(sg, av, xp + std::ptrdiff_t(row_block * QK + kk), QK);
                    for (size_t role = 0; role < roles; ++role) {
                        mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::b, MK, MN, mx::layout::col_major> bv;
                        mx::joint_matrix_load(sg, bv, wp + std::ptrdiff_t((role * bn + column) * QK + kk), QK);
                        mx::joint_matrix_mad(sg, accum[role], av, bv, accum[role]);
                    }
#endif
                }
                it.barrier(sycl::access::fence_space::local_space);
            }
            for (size_t role = 0; role < roles; ++role) {
#if STRATA_AMD_WMMA
                xe::wmma::store(sg, accum[role], op + std::ptrdiff_t(role * rows * bn + row_block * bn + column), bn);
#elif STRATA_IQ_MATRIX_SCALAR
                const size_t lane = sg.get_local_linear_id();
                for (size_t j = 0; j < MM * MN / SG; ++j) {
                    const size_t out = j * SG + lane;
                    result[role * rows * bn + (row_block + out / MN) * bn + column + out % MN] = accum[role][j];
                }
#else
                mx::joint_matrix_store(sg, accum[role], op + std::ptrdiff_t(role * rows * bn + row_block * bn + column), bn, mx::layout::row_major);
#endif
            }
            it.barrier(sycl::access::fence_space::local_space);
            for (size_t j = id; j < rows * bn; j += threads) {
                const size_t row = j / bn, col = j % bn;
                if (r0 + row >= size_t(count) || r0 + row >= size_t(a.max_rows) || c0 + col >= size_t(n)) continue;
                const size_t out = (size_t(first) + r0 + row) * a.ldo + c0 + col;
                if constexpr (GU) a.h[out] = sycl_detail::f16_bits(portable::swiglu(result[j], result[rows * bn + j], a.limit));
                else a.y[out] = result[j];
            }
        });
    });
    return true;
}

// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Lane-owned groups of eight IQ1 values. The existing matrix's two FP16
// rounding boundaries are retained, without materializing a 32-value Sub.
template <int T>
sycl::vec<uint16_t, 8> register_eight(const uint8_t* row, int kb, int group, const uint32_t* tab) {
    const uint8_t* b = row + size_t(kb >> 3) * Ty<T>::B;
    const int ib = kb & 7;
    uint32_t index; bool negative; int sc; float d;
    if constexpr (T == kIQ1_S) {
        const uint32_t high = uint32_t(b[34 + 2 * ib]) | uint32_t(b[35 + 2 * ib]) << 8;
        index = b[2 + 4 * ib + group] | (((high >> (3 * group)) & 7u) << 8);
        negative = (high & 0x8000u) != 0;
        sc = 2 * int((high >> 12) & 7u) + 1; d = h2f(b) * 0.125f;
    } else {
        const uint32_t high = b[32 + 2 * ib + group / 2];
        index = b[4 * ib + group] | (((high >> (4 * (group & 1))) & 7u) << 8);
        negative = (high & (8u << (4 * (group & 1)))) != 0;
        const uint32_t scales = uint32_t(b[48 + 2 * (ib >> 1)]) | uint32_t(b[49 + 2 * (ib >> 1)]) << 8;
        sc = 2 * int((scales >> (6 * (ib & 1) + 3 * (group / 2))) & 7u) + 1;
        d = sycl_detail::f16_value(iq1m_d(b)) * 0.125f;
    }
    const uint32_t grid = tab[index];
    const float scale = matrix_half_value(matrix_half_bits(d * float(sc)));
    sycl::vec<uint16_t, 8> result;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int code = 8 * int((grid >> ((j & 3) * 8 + (j >> 2) * 4)) & 15u) - 7 - (negative ? 2 : 0);
        result[j] = matrix_half_bits(scale * float(code));
    }
    return result;
}

template <int T, bool GU, int RT>
bool register_launch(const SetArgs& a, sycl::queue& s) {
    if constexpr (T != kIQ1_S && T != kIQ1_M) return false;
    else {
        const int n = GU ? a.n_ff : a.n_embd, k = GU ? a.n_embd : a.n_ff;
        if (a.n_exp <= 0 || a.max_rows <= kLightRows || n <= 0 || k <= 0 || k % 256 || n % 128 || a.ldx % 8 || a.ldo % 2 ||
            (a.skip_ge > 0 && a.skip_ge <= kLightRows + 1)) return false;
        if constexpr (GU) { if (!a.xf && !a.xf16) return false; } else if (!a.xh) return false;
        const void* input = GU ? static_cast<const void*>(a.xf) : static_cast<const void*>(a.xh);
        if ((reinterpret_cast<uintptr_t>(input) & 15u) || (GU && a.xf16 && (reinterpret_cast<uintptr_t>(a.xf16) & 15u))) return false;
        if (!s.get_device().has(sycl::aspect::fp16) || !xe::mma_shape_is(s, int(MM), int(MN), int(MK), int(SG)) || !xe::amd_wmma(s.get_device())) return false;
        const auto& c = caps(s);
        if (std::find(c.subgroups.begin(), c.subgroups.end(), SG) == c.subgroups.end()) return false;
        [[maybe_unused]] constexpr size_t rows = RT * MM, roles = GU ? 2 : 1, table_words = Ty<T>::TAB;
        if (c.local < table_words * sizeof(uint32_t) + rows * (QK * sizeof(uint16_t) + sizeof(int))) return false;
        size_t columns = 1;
        while ((columns * 2) * (columns * 2) <= c.group / SG) columns *= 2;
        columns = std::min(columns, (size_t(n) + MN - 1) / MN);
        if (!columns || c.group < columns * SG) return false;
        const size_t bn = columns * MN, threads = columns * SG;
        const size_t nc = (size_t(n) + bn - 1) / bn, nr = (size_t(a.max_rows) + rows - 1) / rows;
        const size_t rb = row_bytes_t<T>(k);
        const int skip = std::max(a.skip_le, kLightRows);
        sycl_detail::require_in_order(s);
        const MatrixGrid grid = matrix_grid(a, s, rows, nc, skip);
        const size_t groups = grid.groups;
        if (!groups) return true;
        sycl_detail::submit(s, [&](sycl::handler& h) {
            sycl::local_accessor<uint32_t, 1> table_data(sycl::range<1>(table_words), h);
            sycl::local_accessor<int, 1> sources(sycl::range<1>(rows), h);
            sycl::local_accessor<sycl::half, 1> xs(sycl::range<1>(rows * QK), h);
            h.parallel_for(sycl::nd_range<1>(groups * threads, threads), [a, nc, nr, bn, skip, threads, grid, table_data, sources, xs, rb, k, n](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                [[maybe_unused]] constexpr size_t roles = GU ? 2 : 1, rows = RT * MM, table_words = Ty<T>::TAB;
                // The host and device closures have the same explicit layout.
                (void)it; (void)a; (void)nc; (void)nr; (void)bn; (void)skip; (void)threads;
                (void)table_data; (void)sources; (void)xs; (void)rb; (void)k; (void)n; (void)grid;
#if STRATA_AMD_WMMA
                const size_t id = it.get_local_linear_id(), g = it.get_group_linear_id();
                const size_t e = grid.jobs ? size_t(grid.jobs[g / nc][0]) : g / (nc * nr);
                const size_t r0 = grid.jobs ? size_t(grid.jobs[g / nc][1]) : ((g / nc) % nr) * rows;
                const size_t c0 = (g % nc) * bn;
                const int count = a.bounds[e + 1] - a.bounds[e], first = a.bounds[e] - a.row0;
                if (count <= skip || (a.skip_ge > 0 && count >= a.skip_ge) || r0 >= size_t(count)) return;
                for (size_t j = id; j < table_words; j += threads) table_data[j] = table<T>()[j];
                if (id < rows) sources[id] = r0 + id < size_t(count) ? (GU && a.idx ? a.idx[size_t(first) + r0 + id] : first + int(r0 + id)) : -1;
                it.barrier(sycl::access::fence_space::local_space);
                const auto sg = it.get_sub_group();
                const size_t column = sg.get_group_linear_id() * MN + xe::wmma::lane16(sg);
                const uint8_t* blob = a.wbase + e * a.stride;
                const auto* grids = table_data.template get_multi_ptr<sycl::access::decorated::no>().get();
                const auto xp = xs.template get_multi_ptr<sycl::access::decorated::no>();
                xe::wmma::C accum[RT][roles];
#pragma unroll
                for (int r = 0; r < RT; ++r)
#pragma unroll
                    for (size_t role = 0; role < roles; ++role) xe::wmma::fill(sg, accum[r][role], 0.f);
                for (int kb = 0; kb < k / int(QK); ++kb) {
                    for (size_t j = id; j < rows * QK; j += threads) {
                        const int src = sources[j / QK]; const size_t at = size_t(src) * a.ldx + size_t(kb) * QK + j % QK;
                        xs[j] = sycl::bit_cast<sycl::half>(src < 0 ? uint16_t(0) : GU ? (a.xf16 ? a.xf16[at] : sycl_detail::f16_bits(a.xf[at])) : a.xh[at]);
                    }
                    it.barrier(sycl::access::fence_space::local_space);
#pragma unroll
                    for (int kk = 0; kk < int(QK); kk += int(MK)) {
                        xe::wmma::Bc<sycl::half> bv[roles];
#pragma unroll
                        for (size_t role = 0; role < roles; ++role) {
                            const uint8_t* w = blob + (GU ? (role ? a.tr[1] : a.tr[0]) : a.tr[2]) + (c0 + column) * rb;
#pragma unroll
                            for (int part = 0; part < xe::wmma::KN / 8; ++part) {
                                sycl::vec<uint16_t, 8> bits(uint16_t(0));
                                if (c0 + column < size_t(n)) bits = register_eight<T>(w, kb, (kk + xe::wmma::half_of(sg) * xe::wmma::KOFF) / 8 + part, grids);
#pragma unroll
                                for (int j = 0; j < 8; ++j) bv[role].v[part * 8 + j] = c0 + column < size_t(n) ? sycl::bit_cast<_Float16>(bits[j]) : _Float16(0);
                            }
                        }
#pragma unroll
                        for (int r = 0; r < RT; ++r) {
                            xe::wmma::A<sycl::half> av;
                            xe::wmma::load(sg, av, xp + std::ptrdiff_t(size_t(r) * MM * QK + kk), QK);
#pragma unroll
                            for (size_t role = 0; role < roles; ++role) xe::wmma::mad(sg, accum[r][role], av, bv[role], accum[r][role]);
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }
#pragma unroll
                for (int r = 0; r < RT; ++r)
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const size_t row = r0 + size_t(r) * MM + xe::wmma::row_of(xe::wmma::half_of(sg), j);
                        if (row >= size_t(count) || row >= size_t(a.max_rows) || c0 + column >= size_t(n)) continue;
                        const size_t out = (size_t(first) + row) * a.ldo + c0 + column;
                        if constexpr (GU) a.h[out] = sycl_detail::f16_bits(portable::swiglu(accum[r][0].v[j], accum[r][1].v[j], a.limit));
                        else a.y[out] = accum[r][0].v[j];
                    }
#endif
            });
        });
        return true;
    }
}

// Share an expanded weight tile across row subgroups while each lane keeps
// only one output fragment per role. The bounded four/eight-subgroup family
// is selected from actual rows and the device's WG/SG/local-memory limits.
template <int T, bool GU, int RG>
bool shared_weight_launch(const SetArgs& a, sycl::queue& s) {
    if constexpr (T != kIQ1_S && T != kIQ1_M) return false;
    else {
        const int n = GU ? a.n_ff : a.n_embd, k = GU ? a.n_embd : a.n_ff;
        constexpr int cutoff = int(2 * MM);
        if (a.n_exp <= 0 || a.max_rows <= cutoff ||
            (a.skip_ge > 0 && a.skip_ge <= cutoff + 1) || n <= 0 || k <= 0 ||
            k % 256 || n % 128 || a.ldx % 8 || a.ldo % 2) return false;
        if constexpr (GU) { if (!a.xf && !a.xf16) return false; }
        else if (!a.xh) return false;
        const void* input = GU ? static_cast<const void*>(a.xf) : static_cast<const void*>(a.xh);
        if ((reinterpret_cast<uintptr_t>(input) & 15u) ||
            (GU && a.xf16 && (reinterpret_cast<uintptr_t>(a.xf16) & 15u))) return false;
        if (!s.get_device().has(sycl::aspect::fp16) ||
            !xe::mma_shape_is(s, int(MM), int(MN), int(MK), int(SG)) ||
            !xe::amd_wmma(s.get_device())) return false;
        const auto& c = caps(s);
        if (std::find(c.subgroups.begin(), c.subgroups.end(), SG) == c.subgroups.end()) return false;
        if ((size_t(a.max_rows) + MM - 1) / MM < size_t(RG)) return false;
        [[maybe_unused]] constexpr size_t rows = RG * MM, roles = GU ? 2 : 1;
        [[maybe_unused]] constexpr size_t table_words = Ty<T>::TAB, pitch = QK + MK / 2;
        size_t columns = 1;
        while ((columns * 2) * (columns * 2) <= c.group / SG) columns *= 2;
        columns = std::min(columns, (size_t(n) + MN - 1) / MN);
        const size_t bn = columns * MN, threads = RG * columns * SG;
        const size_t local = table_words * sizeof(uint32_t) + rows * sizeof(int) +
                             (rows + roles * bn) * pitch * sizeof(uint16_t);
        if (!columns || threads > c.group || local > c.local) return false;
        const size_t nc = (size_t(n) + bn - 1) / bn, nr = (size_t(a.max_rows) + rows - 1) / rows;
        const size_t rb = row_bytes_t<T>(k);
        const int skip = std::max(a.skip_le, cutoff);
        sycl_detail::require_in_order(s);
        // Small experts retain p2 with its actual row upper bound. This also
        // avoids dispatching empty high row tiles for those experts.
        if (a.skip_le < cutoff) {
            SetArgs small = a;
            small.max_rows = cutoff;
            small.skip_ge = cutoff + 1;
            if (!register_launch<T, GU, 2>(small, s)) return false;
        }
        const MatrixGrid grid = matrix_grid(a, s, rows, nc, skip);
        const size_t groups = grid.groups;
        if (!groups) return true;
        sycl_detail::submit(s, [&](sycl::handler& h) {
            sycl::local_accessor<uint32_t, 1> table_data(sycl::range<1>(table_words), h);
            sycl::local_accessor<int, 1> sources(sycl::range<1>(rows), h);
            sycl::local_accessor<sycl::half, 1> xs(sycl::range<1>(rows * pitch), h);
            sycl::local_accessor<sycl::half, 1> ws(sycl::range<1>(roles * bn * pitch), h);
            h.parallel_for(sycl::nd_range<1>(groups * threads, threads),
                [a, nc, nr, bn, skip, threads, table_data, sources, xs, ws, rb, k, n, columns, grid]
                (sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                [[maybe_unused]] constexpr size_t rows = RG * MM, roles = GU ? 2 : 1;
                [[maybe_unused]] constexpr size_t table_words = Ty<T>::TAB, pitch = QK + MK / 2;
                (void)it; (void)a; (void)nc; (void)nr; (void)bn; (void)skip; (void)threads;
                (void)table_data; (void)sources; (void)xs; (void)ws; (void)rb; (void)k; (void)n; (void)columns; (void)grid;
#if STRATA_AMD_WMMA
                const size_t id = it.get_local_linear_id(), group = it.get_group_linear_id();
                const size_t e = grid.jobs ? size_t(grid.jobs[group / nc][0]) : group / (nc * nr);
                const size_t r0 = grid.jobs ? size_t(grid.jobs[group / nc][1]) : ((group / nc) % nr) * rows;
                const size_t c0 = (group % nc) * bn;
                const int first = a.bounds[e] - a.row0, count = a.bounds[e + 1] - a.bounds[e];
                if (count <= skip || (a.skip_ge > 0 && count >= a.skip_ge) || r0 >= size_t(count)) return;
                for (size_t j = id; j < table_words; j += threads) table_data[j] = table<T>()[j];
                if (id < rows) sources[id] = r0 + id < size_t(count) ?
                    (GU && a.idx ? a.idx[size_t(first) + r0 + id] : first + int(r0 + id)) : -1;
                it.barrier(sycl::access::fence_space::local_space);
                const auto sg = it.get_sub_group();
                const size_t row_block = (sg.get_group_linear_id() / columns) * MM;
                const size_t column = (sg.get_group_linear_id() % columns) * MN;
                const size_t lane_column = column + xe::wmma::lane16(sg);
                const uint8_t* blob = a.wbase + e * a.stride;
                const auto* grids = table_data.template get_multi_ptr<sycl::access::decorated::no>().get();
                const auto xp = xs.template get_multi_ptr<sycl::access::decorated::no>();
                const auto wp = ws.template get_multi_ptr<sycl::access::decorated::no>();
                xe::wmma::C accum[roles];
#pragma unroll
                for (size_t role = 0; role < roles; ++role) xe::wmma::fill(sg, accum[role], 0.f);
                for (int kb = 0; kb < k / int(QK); ++kb) {
                    for (size_t j = id; j < roles * bn * (QK / 8); j += threads) {
                        const size_t role = j / (bn * (QK / 8)), col = (j / (QK / 8)) % bn, piece = j % (QK / 8);
                        sycl::vec<uint16_t, 8> bits(uint16_t(0));
                        if (c0 + col < size_t(n)) {
                            const uint8_t* w = blob + (GU ? (role ? a.tr[1] : a.tr[0]) : a.tr[2]) + (c0 + col) * rb;
                            bits = register_eight<T>(w, kb, int(piece), grids);
                        }
#pragma unroll
                        for (int q = 0; q < 8; ++q)
                            ws[(role * bn + col) * pitch + piece * 8 + size_t(q)] = sycl::bit_cast<sycl::half>(bits[q]);
                    }
                    for (size_t j = id; j < rows * QK; j += threads) {
                        const int src = sources[j / QK];
                        uint16_t value = 0;
                        if (src >= 0) {
                            const size_t at = size_t(src) * a.ldx + size_t(kb) * QK + j % QK;
                            value = GU ? (a.xf16 ? a.xf16[at] : sycl_detail::f16_bits(a.xf[at])) : a.xh[at];
                        }
                        xs[(j / QK) * pitch + j % QK] = sycl::bit_cast<sycl::half>(value);
                    }
                    it.barrier(sycl::access::fence_space::local_space);
#pragma unroll
                    for (size_t kk = 0; kk < QK; kk += MK) {
                        xe::wmma::A<sycl::half> av;
                        xe::wmma::load(sg, av, xp + std::ptrdiff_t(row_block * pitch + kk), pitch);
#pragma unroll
                        for (size_t role = 0; role < roles; ++role) {
                            xe::wmma::Bc<sycl::half> bv;
                            xe::wmma::load(sg, bv, wp + std::ptrdiff_t((role * bn + column) * pitch + kk), pitch);
                            xe::wmma::mad(sg, accum[role], av, bv, accum[role]);
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const size_t row = r0 + row_block + xe::wmma::row_of(xe::wmma::half_of(sg), j);
                    if (row >= size_t(count) || row >= size_t(a.max_rows) || c0 + lane_column >= size_t(n)) continue;
                    const size_t out = (size_t(first) + row) * a.ldo + c0 + lane_column;
                    if constexpr (GU) a.h[out] = sycl_detail::f16_bits(portable::swiglu(accum[0].v[j], accum[1].v[j], a.limit));
                    else a.y[out] = accum[0].v[j];
                }
#endif
            });
        });
        return true;
    }
}


template <int T, bool GU>
bool established_launch(const SetArgs& a, sycl::queue& s) {
    if (shared_weight_launch<T, GU, 8>(a, s) || shared_weight_launch<T, GU, 4>(a, s)) return true;
    const size_t needed = (size_t(std::max(0, a.max_rows)) + MM - 1) / MM;
    const auto& c = caps(s); size_t r = 1;
    // Keep the existing light/FP16 precision boundary: the device must report
    // the matrix shape above, and model dimensions must meet its alignment.
    // Nonzero synthetic 64/128-token and 197-group 4096-token fixtures measured
    // this path with ascending K16 and unchanged rounding (2026-11-06).
    // Choose the largest compiled R1/R2/R4 fitting ceil(max_rows/MM) and the
    // reported WG/SG limit; cap the family to bound register accumulators.
    // Local storage follows rows and the quantization table. Unsupported
    // layouts or insufficient device resources use the existing matrix path.
    // Compile a bounded family, selecting from device limits and actual rows.
    while (r < 4 && r * 2 <= needed && (r * 2) * (r * 2) <= c.group / SG) r *= 2;
    if (r == 4 && register_launch<T, GU, 4>(a, s)) return true;
    if (r == 2 && register_launch<T, GU, 2>(a, s)) return true;
    if (r == 1 && register_launch<T, GU, 1>(a, s)) return true;
    return launch_base<T, GU>(a, s);
}

#include "ds4_iq_matrix_large.inc"

template <bool GU>
bool dispatch_matrix(const SetArgs& a, sycl::queue& s) {
    bool launched = false;
    dispatch(GU ? a.tg : a.td, [&](auto type) {
        if constexpr (type.value == kIQ1_S || type.value == kIQ1_M || type.value == kIQ2_XXS ||
                      type.value == kIQ2_XS || type.value == kIQ2_S || type.value == kIQ3_XXS || type.value == kMXFP4)
            launched = launch<type.value, GU>(a, s);
    });
    return launched;
}
}  // namespace
bool gate_up(const SetArgs& a, sycl::queue& s) { return dispatch_matrix<true>(a, s); }
bool down(const SetArgs& a, sycl::queue& s) { return dispatch_matrix<false>(a, s); }
}  // namespace strata::ds4::gguf::iq::matrix_detail

#undef STRATA_IQ_MATRIX_SCALAR
