// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/kernels/canon_math.hpp - the arithmetic an expert's rows take on the CPU and on every GPU alike.
//
// A verify window's expert runs on the GPU when it is in VRAM (or read over PCIe) and on the CPU when not, and which
// it is changes with the window (how many experts miss, the PCIe share): the two must give the same bits, or the
// greedy text depends on the drafting (#152).  Additions, multiplications and fused multiply-adds are exact IEEE
// operations on every CPU and GPU; a division, an exp and the hardware's reciprocal are not (each maker's precision
// differs).  So the expert path's SwiGLU and its activations' int8 scale are built from the first kind only, with
// the fused multiply-adds written out (a compiler that contracted `a * b + c` on one side and not on the other would
// round differently), and no value leaves the normal range (devices differ in how they flush subnormals).
//
// Usable in SYCL device code and in plain host code (the CPU kernels are not SYCL translation units).
#pragma once

#include <cstdint>
#include <cstring>

#if defined(SYCL_LANGUAGE_VERSION)
#include <sycl/sycl.hpp>
#else
#include <cmath>
#endif

// a * b + c is not fused into one rounding behind these functions' backs (clang contracts across statements for CUDA
// and HIP device code by default)
#if defined(__clang__)
#define STRATA_CANON_NO_CONTRACT _Pragma("clang fp contract(off)")
#else
#define STRATA_CANON_NO_CONTRACT
#endif

namespace strata::kernels::canon {

inline float fma(float a, float b, float c) {
#if defined(SYCL_LANGUAGE_VERSION)
    return sycl::fma(a, b, c);
#else
    return std::fma(a, b, c);
#endif
}

inline uint32_t bits(float v) {
    uint32_t u;
    std::memcpy(&u, &v, sizeof u);
    return u;
}
inline float from_bits(uint32_t u) {
    float v;
    std::memcpy(&v, &u, sizeof v);
    return v;
}

/// 1 / v for a positive normal v whose reciprocal is normal too: a seed from the exponent bits, then three Newton
/// steps (relative error well below one ulp of the result; the same bits everywhere, not always the rounded quotient).
inline float rcp(float v) {
    STRATA_CANON_NO_CONTRACT
    float r = from_bits(0x7EF311C3u - bits(v));
    for (int i = 0; i < 3; ++i) {
        const float e = fma(-v, r, 1.0f);
        r = fma(r, e, r);
    }
    return r;
}

/// e^x for x in [-80, 80] (clamped): x = k ln2 + r, |r| <= ln2 / 2, a degree-6 polynomial in r, and k added to the
/// exponent bits.  Results stay normal (e^-80 = 1.8e-35).
inline float exp(float x) {
    STRATA_CANON_NO_CONTRACT
    x = x < -80.0f ? -80.0f : x > 80.0f ? 80.0f : x;
    const float kf = (x * 1.44269504f + 12582912.0f) - 12582912.0f;   // round(x / ln2), to nearest even
    float r = fma(-kf, 0.693145752f, x);                                // ln2's high bits: kf * hi is exact
    r = fma(-kf, 1.42860677e-06f, r);                                   // and its low bits
    float p = 1.0f / 720.0f;
    p = fma(p, r, 1.0f / 120.0f);
    p = fma(p, r, 1.0f / 24.0f);
    p = fma(p, r, 1.0f / 6.0f);
    p = fma(p, r, 0.5f);
    p = fma(p, r, 1.0f);
    p = fma(p, r, 1.0f);
    return from_bits(bits(p) + (uint32_t) ((int32_t) kf << 23));
}

/// The expert's SwiGLU: g / (1 + e^-g) * u.
inline float silu_mul(float g, float u) {
    STRATA_CANON_NO_CONTRACT
    return g * rcp(1.0f + exp(-g)) * u;
}

constexpr float kHalfMax = 65504.0f;

/// A q8_1 block's scale for the largest |x| `amax` of its 32 values, kept finite as an fp16 (#606).
inline float q8_scale(float amax) {
    STRATA_CANON_NO_CONTRACT
    const float d = amax * (1.0f / 127.0f);
    return d > kHalfMax ? kHalfMax : d;
}
/// The inverse of that scale; 0 (every value 0) for an all-zero block and below 1e-30, where rcp's seed and steps
/// would meet subnormals.
inline float q8_inv(float d) { return d >= 1e-30f ? rcp(d) : 0.0f; }
/// One value's int8: rounded half away from zero, held to [-127, 127].
inline int8_t q8_value(float x, float id) {
    STRATA_CANON_NO_CONTRACT
    const float t = x * id;
    const float r = t + (t < 0.0f ? -0.5f : 0.5f);
    const float c = r > 127.0f ? 127.0f : r < -127.0f ? -127.0f : r;
    return (int8_t) (int32_t) c;
}

}  // namespace strata::kernels::canon
