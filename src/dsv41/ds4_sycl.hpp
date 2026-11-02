// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <sycl/sycl.hpp>
#if defined(SYCL_EXT_ONEAPI_ENQUEUE_FUNCTIONS)
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#endif
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace strata::ds4::sycl_detail {

inline void require_in_order(const sycl::queue& q) {
    if (!q.has_property<sycl::property::queue::in_order>())
        throw std::invalid_argument("ds4 kernels require an in-order SYCL queue");
}

// Kernels are enqueued without an event (sycl_ext_oneapi_enqueue_functions) where the compiler has the extension:
// the in-order queue keeps their order, and no event is made a kernel.  Measured on the R9700 (HIP backend,
// Windows), 50 small kernels took 1.7 ms enqueued with events and 0.19 ms without; without the extension, the
// queue's own calls.
#if defined(SYCL_EXT_ONEAPI_ENQUEUE_FUNCTIONS)
namespace sx = sycl::ext::oneapi::experimental;
template <class CGF>
inline void submit(sycl::queue& q, CGF&& cgf) { sx::submit(q, std::forward<CGF>(cgf)); }
template <int D, class K>
inline void launch(sycl::queue& q, sycl::nd_range<D> range, const K& k) { sx::nd_launch(q, range, k); }
template <int D, class K>
inline void launch(sycl::queue& q, sycl::range<D> range, const K& k) { sx::parallel_for(q, range, k); }
template <class K>
inline void single(sycl::queue& q, const K& k) { sx::single_task(q, k); }
#else
template <class CGF>
inline void submit(sycl::queue& q, CGF&& cgf) { q.submit(std::forward<CGF>(cgf)); }
template <int D, class K>
inline void launch(sycl::queue& q, sycl::nd_range<D> range, const K& k) { q.parallel_for(range, k); }
template <int D, class K>
inline void launch(sycl::queue& q, sycl::range<D> range, const K& k) { q.parallel_for(range, k); }
template <class K>
inline void single(sycl::queue& q, const K& k) { q.single_task(k); }
#endif

// The runtime chooses the work-group shape; these portable kernels need neither
// a particular sub-group size nor a matrix extension.
template <class F>
inline void parallel(sycl::queue& q, size_t n, F f) {
    require_in_order(q);
    if (!n) return;
    launch(q, sycl::range<1>(n), [=](sycl::id<1> i) { f(i[0]); });
}

inline float bf16_value(uint16_t v) {
    return sycl::bit_cast<float>(uint32_t(v) << 16);
}

// Integer conversion also works on devices without the optional fp16 aspect. Zeros and subnormals come out
// of one float subtraction, exact (both operands and the result are normal), instead of a normalizing loop.
inline float f16_value(uint16_t v) {
    constexpr uint32_t exponent = 0x7c00u << 13;
    uint32_t bits = uint32_t(v & 0x7fffu) << 13;
    const uint32_t e = bits & exponent;
    bits += (127u - 15u) << 23;
    float value = sycl::bit_cast<float>(bits);
    if (e == exponent) value = sycl::bit_cast<float>(bits + ((128u - 16u) << 23));
    else if (e == 0) value = sycl::bit_cast<float>(bits + (1u << 23)) - sycl::bit_cast<float>(113u << 23);
    return sycl::bit_cast<float>(sycl::bit_cast<uint32_t>(value) | (uint32_t(v & 0x8000u) << 16));
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
