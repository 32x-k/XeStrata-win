// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <sycl/sycl.hpp>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace strata::ds4::sycl_detail {

inline void require_in_order(const sycl::queue& q) {
    if (!q.has_property<sycl::property::queue::in_order>())
        throw std::invalid_argument("ds4 kernels require an in-order SYCL queue");
}

// The runtime chooses the work-group shape; these portable kernels need neither
// a particular sub-group size nor a matrix extension.
template <class F>
inline sycl::event parallel(sycl::queue& q, size_t n, F f) {
    require_in_order(q);
    if (!n) return {};
    return q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { f(i[0]); });
}

inline float bf16_value(uint16_t v) {
    return sycl::bit_cast<float>(uint32_t(v) << 16);
}

// Integer conversion also works on devices without the optional fp16 aspect.
inline float f16_value(uint16_t v) {
    const uint32_t sign = uint32_t(v & 0x8000u) << 16;
    uint32_t e = (v >> 10) & 31u, m = v & 1023u;
    if (e == 31) return sycl::bit_cast<float>(sign | 0x7f800000u | (m << 13));
    if (e == 0) {
        if (!m) return sycl::bit_cast<float>(sign);
        int exp = -14;
        while (!(m & 1024u)) { m <<= 1; --exp; }
        return sycl::bit_cast<float>(sign | (uint32_t(exp + 127) << 23) | ((m & 1023u) << 13));
    }
    return sycl::bit_cast<float>(sign | ((e + 112u) << 23) | (m << 13));
}

inline uint16_t f16_bits(float v) {
    const uint32_t u = sycl::bit_cast<uint32_t>(v), sign = (u >> 16) & 0x8000u;
    const uint32_t a = u & 0x7fffffffu;
    if (a >= 0x7f800000u) return uint16_t(sign | 0x7c00u | (a > 0x7f800000u ? 0x0200u : 0u));
    const int e = int(a >> 23) - 127;
    if (e > 15) return uint16_t(sign | 0x7c00u);
    if (e < -25) return uint16_t(sign);
    if (e < -14) {
        const uint32_t m = (a & 0x7fffffu) | 0x800000u;
        const int shift = -e - 1;
        const uint32_t r = m + ((1u << (shift - 1)) - 1u) + ((m >> shift) & 1u);
        return uint16_t(sign | (r >> shift));
    }
    const uint32_t r = a + 0xfffu + ((a >> 13) & 1u);
    return uint16_t(sign | ((r - 0x38000000u) >> 13));
}

}  // namespace strata::ds4::sycl_detail
