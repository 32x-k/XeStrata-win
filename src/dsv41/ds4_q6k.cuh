// SPDX-FileCopyrightText: 2026 Peasant Smith and the Project Maya contributors
// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/main/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
#pragma once

#include "ds4_sycl.hpp"

namespace strata::ds4 {

constexpr int kQ6kBytes = 210;

inline float q6k_value(const uint8_t* b, int i) {
    const uint8_t* ql = b;
    const uint8_t* qh = b + 128;
    const int8_t* sc = reinterpret_cast<const int8_t*>(b + 192);
    const float d = sycl_detail::f16_value(uint16_t(b[208] | (uint16_t(b[209]) << 8)));
    const int h = i >> 7, w = i & 127, quarter = w >> 5, l = w & 31;
    ql += size_t(h) * 64;
    qh += size_t(h) * 32;
    sc += size_t(h) * 8;
    const int lo = quarter & 1 ? ql[l + 32] : ql[l];
    const int q = ((quarter & 2 ? lo >> 4 : lo & 15) | (((qh[l] >> (2 * quarter)) & 3) << 4)) - 32;
    return (d * float(sc[(l >> 4) + 2 * quarter])) * float(q);
}

}  // namespace strata::ds4
