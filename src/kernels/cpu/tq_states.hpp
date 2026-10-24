// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/cpu/tq_states.hpp - the trellis states of one TQ2_T / TQK6 / TQK7 block (32) or TQP5 / TQP6 / TQP7
// block (64) in plain C++ (all three kernels: scalar code beat byte-shuffle versions on AVX2 and AVX-512).  Internal
// to src/kernels/cpu/tq_*.cpp.
//
// TQ2_T (K = 8): state t = qs[(t + 31) % 32] << 8 | qs[t], masked to 15 bits.
// TQK (K = 6, 7; NS = 32 steps) and TQP (K = 5, 6, 7; NS = 64): the 16 stream bits from bit (NS - 1 - t)*K, LSB
// first, circular modulo NS*K bits, masked to 15 bits.  Every window is one unaligned 32-bit little-endian read
// inside the block's NS*K/8 bytes, except the windows that run past the end, which read a 64-bit view of the last
// four bytes followed by the first four.
#pragma once

#include <cstdint>
#include <cstring>

namespace strata::kernels::cpu::tq_detail {

inline constexpr uint32_t kTqStateMask = 0x7fffu;

inline uint32_t tq_ld32(const uint8_t* p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return v;   // x86 (and every target Strata builds for) is little-endian
}

/// A trellis type's step bits KB and steps per block NS (TQ2_T, TQK6, TQK7, TQP5, TQP6, TQP7: ggml types 144-149).
template <int TY>
struct TqFmt {
    static_assert(TY >= 144 && TY <= 149, "not a trellis type");
    static constexpr int KB = TY == 144 ? 8 : TY == 145 ? 6 : TY == 146 ? 7 : TY - 142;
    static constexpr int NS = TY <= 146 ? 32 : 64;   // weights per step: 128 / NS (4 or 2)
    static constexpr int NB = NS * KB / 8;           // qs bytes
};

/// State t alone (t a constant after unrolling); `wrap` = tq_wrap<TY>(qs) for the windows past the end.
template <int TY>
inline uint64_t tq_wrap(const uint8_t* qs) {
    if constexpr (TY == 144) return 0;
    else return (uint64_t) tq_ld32(qs + TqFmt<TY>::NB - 4) | ((uint64_t) tq_ld32(qs) << 32);
}
template <int TY>
inline uint32_t tq_state(const uint8_t* qs, uint64_t wrap, int t) {
    if constexpr (TY == 144) {
        return (((uint32_t) qs[(t + 31) & 31] << 8) | qs[t]) & kTqStateMask;
    } else {
        constexpr int KB = TqFmt<TY>::KB, NS = TqFmt<TY>::NS, NB = TqFmt<TY>::NB;
        const int off = (NS - 1 - t) * KB, b = off >> 3;
        const uint32_t v = b + 4 <= NB ? tq_ld32(qs + b) >> (off & 7) : (uint32_t) (wrap >> (off - 8 * (NB - 4)));
        return v & kTqStateMask;
    }
}

template <int TY>
inline void tq_states(const uint8_t* qs, uint32_t* st) {
    if constexpr (TY == 144) {
        st[0] = (((uint32_t) qs[31] << 8) | qs[0]) & kTqStateMask;
#if defined(__GNUC__)
#pragma GCC unroll 31
#endif
        for (int t = 1; t < 32; ++t) st[t] = (((uint32_t) qs[t - 1] << 8) | qs[t]) & kTqStateMask;
    } else {
        const uint64_t wrap = tq_wrap<TY>(qs);
#if defined(__GNUC__)
#pragma GCC unroll 64
#endif
        for (int t = 0; t < TqFmt<TY>::NS; ++t) st[t] = tq_state<TY>(qs, wrap, t);
    }
}

/// fp16 bits -> float, exact (the block scales and the Q8_0 activation scales).
inline float tq_half_to_float(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu, man = h & 0x3ffu, bits;
    if (exp == 0x1fu) {
        bits = sign | 0x7f800000u | (man << 13);
    } else if (exp != 0) {
        bits = sign | ((exp + 112u) << 23) | (man << 13);
    } else if (man == 0) {
        bits = sign;
    } else {   // subnormal: man * 2^-24, exact in float
        const float m = (float) man * (1.0f / 16777216.0f);
        return sign ? -m : m;
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

inline uint16_t tq_ld16(const uint8_t* p) {
    uint16_t v;
    std::memcpy(&v, p, 2);
    return v;
}

}  // namespace strata::kernels::cpu::tq_detail
