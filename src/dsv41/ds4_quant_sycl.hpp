// SPDX-FileCopyrightText: 2026 Peasant Smith and the Project Maya contributors
// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/main/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// The portable expert paths share the upstream entry ordering and weight layouts. Each work item owns one output;
// the runtime chooses its work-group size. Quantization block sizes here describe the file, not the device.
#pragma once
#include "ds4_iq.cuh"
#include "ds4_sycl.hpp"

namespace strata::ds4::gguf::iq::portable {

inline float swiglu(float g, float u, float limit) {
    if (limit > 0.0f) {
        g = sycl::fmin(g, limit);
        u = sycl::fmin(sycl::fmax(u, -limit), limit);
    }
    return (g / (1.0f + sycl::exp(-g))) * u;
}

template <bool GU, typename Decoder>
void token(const Args& a, int groups, int ne_max, size_t rb, Decoder decode, sycl::queue& s) {
    if (groups <= 0 || ne_max <= 0) return;
    const int n = GU ? a.n_ff : a.n_embd, k = GU ? a.n_embd : a.n_ff;
    sycl_detail::parallel(s, (size_t) groups * ne_max * n, [=](size_t id) {
        const int row = (int) (id % n);
        const int member = (int) ((id / n) % ne_max), grp = (int) (id / ((size_t) n * ne_max));
        const uint8_t* blob;
        int entry, xr;
        if (a.grp_ptr) {
            if (grp >= *a.n_groups) return;
            const int at = a.grp_start[grp] + member;
            if (at >= a.grp_start[grp + 1]) return;
            blob = reinterpret_cast<const uint8_t*>(a.grp_ptr[grp]);
            entry = a.grp_ent[at];
            xr = entry >> 3;
        } else {
            if (member || !a.plan_ptr[grp]) return;
            blob = reinterpret_cast<const uint8_t*>(a.plan_ptr[grp]);
            entry = grp;
            xr = 0;
        }
        if constexpr (GU) {
            const int8_t* x = a.xq + (size_t) xr * k;
            const float* xd = a.xd + (size_t) xr * (k / 32);
            const float g = decode.dot_q8(blob + a.tr[0] + (size_t) row * rb, x, xd, k);
            const float u = decode.dot_q8(blob + a.tr[1] + (size_t) row * rb, x, xd, k);
            a.h[(size_t) entry * n + row] = swiglu(g, u, a.limit);
        } else {
            a.part[(size_t) entry * n + row] = decode.dot_q8(blob + a.tr[2] + (size_t) row * rb,
                a.hq + (size_t) entry * k, a.hd + (size_t) entry * (k / 32), k);
        }
    });
}

template <bool GU, typename Decoder>
void set(const SetArgs& a, size_t rb, Decoder decode, sycl::queue& s) {
    if (a.n_exp <= 0 || a.max_rows <= 0) return;
    const int n = GU ? a.n_ff : a.n_embd, k = GU ? a.n_embd : a.n_ff;
    sycl_detail::parallel(s, (size_t) a.n_exp * a.max_rows * n, [=](size_t id) {
        const int col = (int) (id % n), row = (int) ((id / n) % a.max_rows);
        const int e = (int) (id / ((size_t) n * a.max_rows));
        const int lo = a.bounds[e] - a.row0, hi = a.bounds[e + 1] - a.row0, count = hi - lo;
        if (row >= count || count <= a.skip_le || (a.skip_ge > 0 && count >= a.skip_ge)) return;
        const size_t local = (size_t) lo + (size_t) row;
        const int input = GU ? a.idx[local] : (int) local;
        const uint8_t* blob = a.wbase + (size_t) e * a.stride;
        const uint8_t* w = blob + a.tr[GU ? 0 : 2] + (size_t) col * rb;
        const uint8_t* up = blob + a.tr[1] + (size_t) col * rb;
        float sum = 0.0f, usum = 0.0f;
        for (int j = 0; j < k; ++j) {
            const size_t at = (size_t) input * a.ldx + j;
            const float x = GU ? (a.xf16 ? sycl_detail::f16_value(a.xf16[at]) : a.xf[at])
                               : sycl_detail::f16_value(a.xh[at]);
            sum = sycl::fma(x, decode.value(w, j), sum);
            if constexpr (GU) usum = sycl::fma(x, decode.value(up, j), usum);
        }
        if constexpr (GU) a.h[local * a.ldo + col] = sycl_detail::f16_bits(swiglu(sum, usum, a.limit));
        else a.y[local * a.ldo + col] = sum;
    });
}

template <typename Decoder>
void mv(const MvJob& j, Decoder decode, sycl::queue& s) {
    sycl_detail::parallel(s, (size_t) j.n * j.ncol, [=](size_t id) {
        const int row = (int) (id % j.n), col = (int) (id / j.n);
        const float sum = decode.dot_q8(j.w + (size_t) row * j.ld, j.xq + (size_t) col * j.k,
                                        j.xd + (size_t) col * (j.k / 32), j.k);
        j.y[(size_t) col * j.ycs + row] = j.alpha * sum + (j.bias ? j.bias[row] : 0.0f);
    });
}

}  // namespace strata::ds4::gguf::iq::portable
