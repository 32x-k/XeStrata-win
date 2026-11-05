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

#if (defined(__SYCL_DEVICE_ONLY__) && defined(__AMDGCN__) && !STRATA_AMD_WMMA) || \
    (STRATA_NV_ARCH > 0 && STRATA_NV_ARCH < 700)
#define STRATA_IQ_MATRIX_SCALAR 1
#else
#define STRATA_IQ_MATRIX_SCALAR 0
#endif

namespace strata::ds4::gguf::iq::matrix_detail {
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
bool launch(const SetArgs& a, sycl::queue& s) {
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
    const size_t rb = row_bytes_t<T>(k), groups = size_t(a.n_exp) * nc * nr;
    const int skip = std::max(a.skip_le, kLightRows);
    sycl_detail::submit(s, [&](sycl::handler& h) {
        sycl::local_accessor<uint32_t, 1> table_data(sycl::range<1>(table_words), h);
        sycl::local_accessor<int, 1> source(sycl::range<1>(rows), h);
        sycl::local_accessor<sycl::half, 1> xs(sycl::range<1>(rows * QK), h), ws(sycl::range<1>(roles * bn * QK), h);
        sycl::local_accessor<float, 1> result(sycl::range<1>(roles * rows * bn), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(groups * threads), sycl::range<1>(threads)),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const size_t id = it.get_local_linear_id(), group = it.get_group_linear_id();
            const size_t e = group / (nc * nr), r0 = ((group / nc) % nr) * rows, c0 = (group % nc) * bn;
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
