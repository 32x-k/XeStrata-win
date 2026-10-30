// SPDX-FileCopyrightText: 2026 Peasant Smith and the Project Maya contributors
// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/main/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
#pragma once

#include "ds4_sycl.hpp"

namespace strata::ds4 {

inline uint16_t bf16_rn(float f) {
    const uint32_t u = sycl::bit_cast<uint32_t>(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return uint16_t((u >> 16) | 0x40u);
    return uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

inline float e4m3_value(uint8_t q) {
    const uint32_t e = (q >> 3) & 15u, m = q & 7u;
    float v;
    if (e == 15u && m == 7u) v = sycl::bit_cast<float>(0x7fc00000u);
    else if (e == 0u) v = float(m) * 0.001953125f;
    else v = sycl::bit_cast<float>(((e + 120u) << 23) | (m << 20));
    return (q & 0x80u) ? -v : v;
}

inline float e4m3_round(float v) {
    const float a = sycl::fabs(v);
    if (sycl::isnan(a)) return v;
    float r;
    if (a >= 448.0f) r = 448.0f;
    else {
        const int e = int((sycl::bit_cast<uint32_t>(a) >> 23) & 0xffu) - 127;
        const float step = e < -6 ? 0.001953125f : sycl::bit_cast<float>(uint32_t(e - 3 + 127) << 23);
        r = a == 0.0f ? 0.0f : sycl::rint(a / step) * step;
        r = sycl::fmin(r, 448.0f);
    }
    return sycl::copysign(r, v);
}

}  // namespace strata::ds4
