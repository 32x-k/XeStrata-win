// SPDX-FileCopyrightText: 2026 Peasant Smith and the Project Maya contributors
// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// SPDX-FileCopyrightText: 2023-2026 The ggml authors

// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// src/ds4/ds4_iq.cu - a GGUF's low-bit routed experts on the device (see ds4_iq.cuh).  The block layouts and grids are
// ggml's (ggml-common.h, MIT, third_party/ggml/LICENSE); the sign and grid tricks are llama.cpp's vecdotq.cuh.
#include "ds4_quant_sycl.hpp"
#include "ds4_numerics.cuh"
#include "ds4_iq_decode.hpp"
#include "ds4_iq_matrix.hpp"
#include "ds4_iq_token.hpp"
#include "strata/core/per_device.hpp"
#include <algorithm>
#include <cstdlib>
#include <vector>

namespace strata::ds4::gguf::iq {
using namespace sycl_decode;

const token_detail::Caps& token_detail::caps(const sycl::queue& q) {
    static core::PerDevice<Caps> cache;
    const auto device = q.get_device();
    return cache.get(device, [&] {
        return Caps{std::min(device.get_info<sycl::info::device::max_work_group_size>(),
                             device.get_info<sycl::info::device::max_work_item_sizes<1>>()[0]),
                    size_t(device.get_info<sycl::info::device::local_mem_size>()),
                    device.get_info<sycl::info::device::sub_group_sizes>()};
    });
}

namespace {
bool light_on() {
    static const bool on = [] {
        // Test-only parity switch: bypass the per-32 Q8 light-set contract so
        // small experts exercise the decoded-weight FMA fallback instead.
        const char* v = std::getenv("STRATA_DS4_NO_LIGHT");
        return !(v && v[0] == '1');
    }();
    return on;
}

struct SetCaps {
    size_t max_group = 0, local_bytes = 0;
    std::vector<size_t> sub_groups;
};
struct SetTile { size_t cols = 0, rows = 0; };

const SetCaps& set_caps(const sycl::queue& s) {
    const auto device = s.get_device();
    static core::PerDevice<SetCaps> cache;
    return cache.get(device, [&] {
        return SetCaps{std::min(device.get_info<sycl::info::device::max_work_group_size>(),
                                device.get_info<sycl::info::device::max_work_item_sizes<1>>()[0]),
                        size_t(device.get_info<sycl::info::device::local_mem_size>()),
                        device.get_info<sycl::info::device::sub_group_sizes>()};
    });
}

template <int T, bool GU>
SetTile set_tile(const SetArgs& a, const sycl::queue& s) {
    const auto& caps = set_caps(s);
    const size_t n = size_t(GU ? a.n_ff : a.n_embd);
    SetTile tile;
    for (const size_t cols : caps.sub_groups) {
        if (!cols || cols > n || cols > caps.max_group || cols <= tile.cols) continue;
        // 32 is the quantization sub-block. Padding separates adjacent local-memory rows.
        const size_t fixed = size_t(Ty<T>::TAB > 0 ? Ty<T>::TAB : 1) * sizeof(uint32_t) +
                             32 * (cols + 1) * sizeof(float) * (GU ? 2 : 1);
        const size_t row_bytes = 33 * sizeof(float) + sizeof(int);
        if (caps.local_bytes <= fixed) continue;
        const size_t rows = std::min({size_t(a.max_rows), caps.max_group / cols,
                                      (caps.local_bytes - fixed) / row_bytes});
        if (rows > 1) tile = {cols, rows};
    }
    return tile;
}

template <int T, bool GU>
SetTile light_tile(const SetArgs& a, const sycl::queue& s) {
    const auto& caps = set_caps(s);
    const size_t n = size_t(GU ? a.n_ff : a.n_embd);
    SetTile tile;
    for (const size_t cols : caps.sub_groups) {
        if (!cols || cols > n || cols > caps.max_group || cols <= tile.cols) continue;
        const size_t fixed = size_t(Ty<T>::TAB > 0 ? Ty<T>::TAB : 1) * sizeof(uint32_t) +
                             cols * sizeof(Sub) * (GU ? 2 : 1);
        const size_t row_bytes = 32 * sizeof(int8_t) + sizeof(float) + sizeof(int);
        if (caps.local_bytes <= fixed) continue;
        const size_t rows = std::min({size_t(std::min(a.max_rows, kLightRows)), caps.max_group / cols,
                                      (caps.local_bytes - fixed) / row_bytes});
        if (rows) tile = {cols, rows};
    }
    return tile;
}

template <int T, bool GU>
bool light_set_tiled(const SetArgs& a, size_t rb, sycl::queue& s) {
    if (a.n_exp <= 0 || a.max_rows <= 0) return false;
    const int n = GU ? a.n_ff : a.n_embd, k = GU ? a.n_embd : a.n_ff;
    if (n <= 0 || k <= 0) return false;
    const auto tile = light_tile<T, GU>(a, s);
    if (!tile.cols) return false;
    sycl_detail::require_in_order(s);
    const size_t bn = tile.cols, rm = tile.rows, threads = bn * rm;
    const size_t rows = size_t(std::min(a.max_rows, kLightRows));
    const size_t nc = (size_t(n) + bn - 1) / bn, nr = (rows + rm - 1) / rm;
    const size_t groups = size_t(a.n_exp) * nc * nr;
    // Keep the Q8 sub-block arithmetic and ascending-K accumulation. Only the
    // repeated activation quantizers and weight decoders are shared by the tile.
    s.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint32_t, 1> tab(sycl::range<1>(Ty<T>::TAB > 0 ? Ty<T>::TAB : 1), h);
        sycl::local_accessor<Sub, 1> weights(sycl::range<1>(bn * (GU ? 2 : 1)), h);
        sycl::local_accessor<int8_t, 1> codes(sycl::range<1>(rm * 32), h);
        sycl::local_accessor<float, 1> scales(sycl::range<1>(rm), h);
        sycl::local_accessor<int, 1> sources(sycl::range<1>(rm), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(groups * threads), sycl::range<1>(threads)),
                       [=](sycl::nd_item<1> it) {
            const size_t id = it.get_local_linear_id(), group = it.get_group_linear_id();
            const size_t e = group / (nc * nr), r0 = ((group / nc) % nr) * rm, c0 = (group % nc) * bn;
            const int count = a.bounds[e + 1] - a.bounds[e], first = a.bounds[e] - a.row0;
            if (count <= 0 || count > kLightRows || r0 >= size_t(count)) return;
            if constexpr (Ty<T>::TAB > 0)
                for (size_t j = id; j < size_t(Ty<T>::TAB); j += threads) tab[j] = table<T>()[j];
            if (id < rm) {
                const size_t row = r0 + id, local = size_t(first) + row;
                sources[id] = row < size_t(count) && row < rows ?
                              (GU && a.idx ? a.idx[local] : int(local)) : -1;
            }
            it.barrier(sycl::access::fence_space::local_space);
            const auto* grids = tab.template get_multi_ptr<sycl::access::decorated::no>().get();
            const uint8_t* blob = a.wbase + e * a.stride;
            const size_t row = id / bn, col = id % bn;
            float sum = 0.0f, usum = 0.0f;
            for (int kb = 0; kb < k / 32; ++kb) {
                if (id < bn) {
                    Sub g{}, u{};
                    if (c0 + id < size_t(n)) {
                        decode<T>(blob + a.tr[GU ? 0 : 2] + (c0 + id) * rb, kb, grids, g);
                        if constexpr (GU) decode<T>(blob + a.tr[1] + (c0 + id) * rb, kb, grids, u);
                    }
                    weights[id] = g;
                    if constexpr (GU) weights[bn + id] = u;
                }
                if (id < rm) {
                    const int src = sources[id];
                    float values[32], amax = 0.0f;
                    for (int j = 0; j < 32; ++j) {
                        float x = 0.0f;
                        if (src >= 0) {
                            const size_t at = size_t(src) * a.ldx + size_t(kb) * 32 + size_t(j);
                            x = GU ? a.xf[at] : sycl_detail::f16_value(a.xh[at]);
                        }
                        values[j] = x;
                        amax = sycl::fmax(amax, sycl::fabs(x));
                    }
                    const float scale = amax / 127.0f;
                    scales[id] = scale;
                    for (int j = 0; j < 32; ++j)
                        codes[id * 32 + size_t(j)] = int8_t(amax > 0.0f ? sycl::rint(values[j] / scale) : 0.0f);
                }
                it.barrier(sycl::access::fence_space::local_space);
                const Sub g = weights[col];
                Sub u{};
                if constexpr (GU) u = weights[bn + col];
                int lo = 0, hi = 0, ulo = 0, uhi = 0, sl = 0, sh = 0;
                for (int j = 0; j < 32; ++j) {
                    const int v = int(uint8_t(codes[row * 32 + size_t(j)]) ^ 128u) - 128;
                    const uint32_t byte = (uint32_t(g.q[j / 4]) >> (8 * (j % 4))) & 255u;
                    const int code = int(byte ^ 128u) - 128;
                    if (j < 16) { lo += code * v; sl += v; }
                    else { hi += code * v; sh += v; }
                    if constexpr (GU) {
                        const uint32_t ub = (uint32_t(u.q[j / 4]) >> (8 * (j % 4))) & 255u;
                        const int uc = int(ub ^ 128u) - 128;
                        if (j < 16) ulo += uc * v;
                        else uhi += uc * v;
                    }
                }
                const float dot = g.d * float(g.s0 * lo + g.s1 * hi) - (g.m0 * float(sl) + g.m1 * float(sh));
                sum = sycl::fma(scales[row], dot, sum);
                if constexpr (GU) {
                    const float udot = u.d * float(u.s0 * ulo + u.s1 * uhi) - (u.m0 * float(sl) + u.m1 * float(sh));
                    usum = sycl::fma(scales[row], udot, usum);
                }
                it.barrier(sycl::access::fence_space::local_space);
            }
            if (r0 + row >= size_t(count) || r0 + row >= rows || c0 + col >= size_t(n)) return;
            const size_t out = (size_t(first) + r0 + row) * a.ldo + c0 + col;
            if constexpr (GU) a.h[out] = sycl_detail::f16_bits(portable::swiglu(sum, usum, a.limit));
            else a.y[out] = sum;
        });
    });
    return true;
}

float sub_value(const Sub& d, int j) {
    const uint32_t byte = (uint32_t(d.q[j / 4]) >> (8 * (j % 4))) & 255u;
    const int code = int(byte ^ 128u) - 128;
    return d.d * float((j < 16 ? d.s0 : d.s1) * code) - (j < 16 ? d.m0 : d.m1);
}

template <int T, bool GU>
bool busy_set(const SetArgs& a, size_t rb, sycl::queue& s) {
    if (a.n_exp <= 0 || a.max_rows <= kLightRows ||
        (a.skip_ge > 0 && a.skip_ge <= kLightRows + 1)) return false;
    const int n = GU ? a.n_ff : a.n_embd, k = GU ? a.n_embd : a.n_ff;
    if (n <= 0 || k <= 0 || k % 32) return false;
    const auto tile = set_tile<T, GU>(a, s);
    if (!tile.cols) return false;
    sycl_detail::require_in_order(s);
    const size_t bn = tile.cols, rm = tile.rows, threads = bn * rm;
    const size_t nc = (size_t(n) + bn - 1) / bn, nr = (size_t(a.max_rows) + rm - 1) / rm;
    const size_t groups = size_t(a.n_exp) * nc * nr;
    const int skip = std::max(a.skip_le, kLightRows);
    // Upstream's set_kernel shares decoded K sub-blocks and gathered inputs across a tile.
    // The tile dimensions follow device limits; each output retains the scalar path's ascending-K FP32 FMAs.
    s.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint32_t, 1> tab(sycl::range<1>(Ty<T>::TAB > 0 ? Ty<T>::TAB : 1), h);
        sycl::local_accessor<float, 1> weights(sycl::range<1>(32 * (bn + 1) * (GU ? 2 : 1)), h);
        sycl::local_accessor<float, 1> inputs(sycl::range<1>(rm * 33), h);
        sycl::local_accessor<int, 1> sources(sycl::range<1>(rm), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(groups * threads), sycl::range<1>(threads)),
                       [=](sycl::nd_item<1> it) {
            const size_t id = it.get_local_linear_id(), group = it.get_group_linear_id();
            const size_t e = group / (nc * nr), r0 = ((group / nc) % nr) * rm, c0 = (group % nc) * bn;
            const int lo = a.bounds[e] - a.row0, count = a.bounds[e + 1] - a.bounds[e];
            // Every work item takes the same return before the first barrier.
            if (count <= skip || (a.skip_ge > 0 && count >= a.skip_ge) || r0 >= size_t(count)) return;
            if constexpr (Ty<T>::TAB > 0)
                for (size_t j = id; j < size_t(Ty<T>::TAB); j += threads) tab[j] = table<T>()[j];
            if (id < rm) {
                const size_t row = r0 + id, local = size_t(lo) + row;
                sources[id] = row < size_t(count) && row < size_t(a.max_rows) ?
                              (GU && a.idx ? a.idx[local] : int(local)) : -1;
            }
            it.barrier(sycl::access::fence_space::local_space);
            const auto* grids = tab.template get_multi_ptr<sycl::access::decorated::no>().get();
            const uint8_t* blob = a.wbase + e * a.stride;
            const size_t row = id / bn, col = id % bn, pitch = bn + 1, up_offset = 32 * pitch;
            float sum = 0.0f, usum = 0.0f;
            for (int kb = 0; kb < k / 32; ++kb) {
                if (id < bn) {
                    Sub g{}, u{};
                    if (c0 + id < size_t(n)) {
                        decode<T>(blob + a.tr[GU ? 0 : 2] + (c0 + id) * rb, kb, grids, g);
                        if constexpr (GU) decode<T>(blob + a.tr[1] + (c0 + id) * rb, kb, grids, u);
                    }
                    for (int j = 0; j < 32; ++j) {
                        weights[size_t(j) * pitch + id] = sub_value(g, j);
                        if constexpr (GU) weights[up_offset + size_t(j) * pitch + id] = sub_value(u, j);
                    }
                }
                for (size_t j = id; j < rm * 32; j += threads) {
                    const size_t r = j / 32, c = j % 32;
                    const int src = sources[r];
                    float x = 0.0f;
                    if (src >= 0) {
                        const size_t at = size_t(src) * a.ldx + size_t(kb) * 32 + c;
                        x = GU ? (a.xf16 ? sycl_detail::f16_value(a.xf16[at]) : a.xf[at]) :
                                 sycl_detail::f16_value(a.xh[at]);
                    }
                    inputs[r * 33 + c] = x;
                }
                it.barrier(sycl::access::fence_space::local_space);
                for (int j = 0; j < 32; ++j) {
                    const float x = inputs[row * 33 + size_t(j)];
                    sum = sycl::fma(x, weights[size_t(j) * pitch + col], sum);
                    if constexpr (GU) usum = sycl::fma(x, weights[up_offset + size_t(j) * pitch + col], usum);
                }
                it.barrier(sycl::access::fence_space::local_space);
            }
            if (r0 + row >= size_t(count) || r0 + row >= size_t(a.max_rows) || c0 + col >= size_t(n)) return;
            const size_t out = (size_t(lo) + r0 + row) * a.ldo + c0 + col;
            if constexpr (GU) a.h[out] = sycl_detail::f16_bits(portable::swiglu(sum, usum, a.limit));
            else a.y[out] = sum;
        });
    });
    return true;
}

template <int T, bool GU>
void launch_set(const SetArgs& args, sycl::queue& s) {
    SetArgs a = args;
    // The FMA fallbacks use FP32 operands, like upstream's set_kernel. Its FP32 input is authoritative;
    // xf16 is an optional copy for half-operand matrix tiles, or a fallback when xf is absent.
    if constexpr (GU) if (a.xf) a.xf16 = nullptr;
    const size_t rb = row_bytes_t<T>(GU ? a.n_embd : a.n_ff);
    if constexpr (T == kIQ1_S || T == kIQ1_M || T == kIQ2_XXS || T == kIQ2_XS || T == kIQ2_S || T == kIQ3_XXS ||
                  T == kMXFP4) {
        const bool matrix = GU ? matrix_detail::gate_up(args, s) : matrix_detail::down(args, s);
        if (matrix || busy_set<T, GU>(a, rb, s)) {
            // If the Q8 light path was unavailable, keep its small sets on the original FP32 fallback.
            if (a.skip_le < kLightRows) {
                SetArgs small = a;
                small.max_rows = kLightRows;
                small.skip_ge = a.skip_ge > 0 ? std::min(a.skip_ge, kLightRows + 1) : kLightRows + 1;
                portable::set<GU>(small, rb, Decoder<T>{}, s);
            }
            return;
        }
    }
    portable::set<GU>(a, rb, Decoder<T>{}, s);
}

template <bool GU, typename Decoder>
void light_set(const SetArgs& a, size_t rb, Decoder decode, sycl::queue& s) {
    if (a.n_exp <= 0 || a.max_rows <= 0) return;
    const int n = GU ? a.n_ff : a.n_embd, k = GU ? a.n_embd : a.n_ff;
    const int rows = std::min(a.max_rows, kLightRows);
    sycl_detail::parallel(s, (size_t) a.n_exp * rows * n, [=](size_t id) {
        const int col = (int) (id % n), row = (int) ((id / n) % rows);
        const int e = (int) (id / ((size_t) n * rows));
        const int count = a.bounds[e + 1] - a.bounds[e];
        if (row >= count || count > kLightRows) return;
        const int first = a.bounds[e] - a.row0;
        const size_t local = (size_t) first + (size_t) row;
        const int input = GU && a.idx ? a.idx[local] : (int) local;
        const uint8_t* blob = a.wbase + (size_t) e * a.stride;
        const uint8_t* w = blob + a.tr[GU ? 0 : 2] + (size_t) col * rb;
        const uint8_t* up = blob + a.tr[1] + (size_t) col * rb;
        float sum = 0.0f, usum = 0.0f;
        for (int b = 0; b < k / 32; ++b) {
            const size_t base = (size_t) input * a.ldx + (size_t) b * 32;
            float values[32], amax = 0.0f;
            for (int j = 0; j < 32; ++j) {
                values[j] = GU ? a.xf[base + j] : sycl_detail::f16_value(a.xh[base + j]);
                amax = sycl::fmax(amax, sycl::fabs(values[j]));
            }
            const float scale = amax / 127.0f;
            int8_t codes[32];
            for (int j = 0; j < 32; ++j) codes[j] = (int8_t) (amax > 0.0f ? sycl::rint(values[j] / scale) : 0.0f);
            sum = sycl::fma(scale, decode.dot_q8_block(w, b, codes), sum);
            if constexpr (GU) usum = sycl::fma(scale, decode.dot_q8_block(up, b, codes), usum);
        }
        if constexpr (GU) a.h[local * a.ldo + col] = sycl_detail::f16_bits(portable::swiglu(sum, usum, a.limit));
        else a.y[local * a.ldo + col] = sum;
    });
}

template <bool GU>
bool launch_light(const SetArgs& a, sycl::queue& s) {
    const int type = GU ? a.tg : a.td;
    if (is_tq(type) || type == kQ5_K || type == kQ6_K || !light_on() ||
        a.n_embd % 256 != 0 || a.n_ff % 256 != 0 || (GU && !a.xf)) return false;
    return dispatch(type, [&](auto t) {
        // Gate/up shares its quantized activation tile across both projections.
        // Down retains the output-owned reduction.
        if constexpr (GU && (t.value == kIQ1_S || t.value == kIQ1_M || t.value == kIQ2_XXS ||
                             t.value == kIQ2_XS || t.value == kIQ2_S || t.value == kIQ3_XXS || t.value == kMXFP4)) {
            if (light_set_tiled<t.value, GU>(a, row_bytes_t<t.value>(GU ? a.n_embd : a.n_ff), s)) return;
        }
        light_set<GU>(a, row_bytes_t<t.value>(GU ? a.n_embd : a.n_ff), Decoder<t.value>{}, s);
    });
}
}  // namespace

bool gate_up(const Args& a, int groups, int ne_max, sycl::queue& s) {
    if (is_tq(a.tg)) return tq::gate_up(a, groups, ne_max, s);
    return dispatch(a.tg, [&](auto t) {
        if constexpr (t.value == kIQ1_S || t.value == kIQ1_M || t.value == kIQ2_XXS ||
                      t.value == kIQ2_XS || t.value == kIQ2_S || t.value == kIQ3_XXS) {
            if (token_detail::launch<t.value, true>(a, groups, ne_max, s)) return;
        }
        portable::token<true>(a, groups, ne_max, row_bytes_t<t.value>(a.n_embd), Decoder<t.value>{}, s);
    });
}
bool down(const Args& a, int groups, int ne_max, sycl::queue& s) {
    if (is_tq(a.td)) return tq::down(a, groups, ne_max, s);
    return dispatch(a.td, [&](auto t) {
        if constexpr (t.value == kIQ1_S || t.value == kIQ1_M || t.value == kIQ2_XXS ||
                      t.value == kIQ2_XS || t.value == kIQ2_S || t.value == kIQ3_XXS) {
            if (token_detail::launch<t.value, false>(a, groups, ne_max, s)) return;
        }
        portable::token<false>(a, groups, ne_max, row_bytes_t<t.value>(a.n_ff), Decoder<t.value>{}, s);
    });
}
bool set_gate_up(const SetArgs& a, sycl::queue& s) {
    if (is_tq(a.tg)) return tq::set_gate_up(a, s);
    SetArgs rest = a;
    if (light_gate_up(a, s)) {
        if (a.max_rows <= kLightRows) return true;
        rest.skip_le = std::max(a.skip_le, kLightRows);
    }
    return dispatch(a.tg, [&](auto t) { launch_set<t.value, true>(rest, s); });
}
bool set_down(const SetArgs& a, sycl::queue& s) {
    if (is_tq(a.td)) return tq::set_down(a, s);
    SetArgs rest = a;
    if (light_down(a, s)) {
        if (a.max_rows <= kLightRows) return true;
        rest.skip_le = std::max(a.skip_le, kLightRows);
    }
    return dispatch(a.td, [&](auto t) { launch_set<t.value, false>(rest, s); });
}
bool light_gate_up(const SetArgs& a, sycl::queue& s) { return launch_light<true>(a, s); }
bool light_down(const SetArgs& a, sycl::queue& s) { return launch_light<false>(a, s); }
bool hipk_force(int) { return false; }
long long hipk_launches() { return -1; }

bool mv(const MvJob* jobs, int n, sycl::queue& s) {
    for (int i = 0; i < n; ++i) if (!supported(jobs[i].type)) return false;
    for (int i = 0; i < n; ++i) {
        const MvJob j = jobs[i];
        if (is_tq(j.type)) { if (!tq::mv(&j, 1, s)) return false; }
        else if (!dispatch(j.type, [&](auto t) { portable::mv(j, Decoder<t.value>{}, s); })) return false;
    }
    return true;
}
bool dequant_rows(int type, const void* raw, size_t ld, int k, int n, uint16_t* dst, bool bf16, sycl::queue& s) {
    return dispatch(type, [&](auto t) {
        sycl_detail::parallel(s, (size_t) n * k, [=](size_t i) {
            const float v = Decoder<t.value>{}.value((const uint8_t*) raw + (i / k) * ld, (int) (i % k));
            dst[i] = bf16 ? bf16_rn(v) : sycl_detail::f16_bits(v);
        });
    });
}
bool f16_rows_ok(int type) { return supported(type) && !is_tq(type) && type != kQ5_K && type != kQ6_K; }
bool f16_rows(int type, const uint8_t* w, int k, int n, uint16_t* dst, int ldd, sycl::queue& s) {
    if (!f16_rows_ok(type)) return false;
    return dispatch(type, [&](auto t) {
        const size_t rb = row_bytes_t<t.value>(k);
        sycl_detail::parallel(s, (size_t) n * k, [=](size_t i) {
            dst[(i / k) * ldd + i % k] = sycl_detail::f16_bits(Decoder<t.value>{}.value(w + (i / k) * rb, (int) (i % k)));
        });
    });
}
void gather_f16(const uint16_t* x, int ldx, const int* idx, int n, int k, uint16_t* dst, sycl::queue& s) {
    sycl_detail::parallel(s, (size_t) n * k, [=](size_t i) { dst[i] = x[(size_t) idx[i / k] * ldx + i % k]; });
}
void swiglu_f16(const float* gu, int ld, int n, int ff, float limit, uint16_t* h, int ldh, sycl::queue& s) {
    sycl_detail::parallel(s, (size_t) n * ff, [=](size_t i) {
        const size_t row = i / ff, col = i % ff;
        h[row * ldh + col] = sycl_detail::f16_bits(portable::swiglu(gu[row * ld + col], gu[row * ld + ff + col], limit));
    });
}
}  // namespace strata::ds4::gguf::iq
