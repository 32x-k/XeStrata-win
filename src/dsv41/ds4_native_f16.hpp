// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include "ds4_sycl.hpp"

namespace strata::ds4::native_f16 {
inline uint16_t bits(float value) {
    const uint32_t raw = sycl::bit_cast<uint32_t>(value);
    if ((raw & 0x7fffffffu) > 0x7f800000u) return uint16_t((raw >> 16 & 0x8000u) | 0x7e00u);
    return sycl::bit_cast<uint16_t>(sycl::half(value));
}
// Called during model initialization, before recording or timing work.
bool qualify(sycl::queue& queue);
void rows(const float* x, int ldx, const int* idx, int count, int cols,
          uint16_t* out, sycl::queue& queue, bool qualified);
}  // namespace strata::ds4::native_f16
