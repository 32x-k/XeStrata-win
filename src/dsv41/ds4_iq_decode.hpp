// SPDX-FileCopyrightText: 2026 Peasant Smith and the Project Maya contributors
// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// SPDX-FileCopyrightText: 2023-2026 The ggml authors

// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
#pragma once
#include "ds4_iq_types.hpp"
#include "ds4_sycl.hpp"
#include "../kernels/xe/cuda_intrinsics.hpp"
#define GGML_COMMON_DECL_SYCL
#define GGML_COMMON_IMPL_SYCL
#include "ggml-common.h"
#include <type_traits>
namespace strata::ds4::gguf::iq::sycl_decode {

struct I2 { int x, y; };
struct U2 { uint32_t x, y; };
// ---------------------------------------------------------------- the types
template <int T>
struct Ty;
template <> struct Ty<kIQ2_XXS> { static constexpr int B = 66, TAB = 512, MIN = 0; };
template <> struct Ty<kIQ2_XS> { static constexpr int B = 74, TAB = 1024, MIN = 0; };
template <> struct Ty<kIQ2_S> { static constexpr int B = 82, TAB = 2048, MIN = 0; };
template <> struct Ty<kIQ3_XXS> { static constexpr int B = 98, TAB = 256, MIN = 0; };
template <> struct Ty<kIQ3_S> { static constexpr int B = 110, TAB = 512, MIN = 0; };
template <> struct Ty<kIQ4_XS> { static constexpr int B = 136, TAB = 4, MIN = 0; };
template <> struct Ty<kQ4_K> { static constexpr int B = 144, TAB = 0, MIN = 1; };
// The 1.5- and 1.75-bit types (IQ1_S, IQ1_M: an 11-bit index into iq1s_grid_gpu, whose 8 values are 4-bit codes) and
// IQ4_NL (32 values a block: its codebook's 16 int8 values are the first 4 words of its table)
template <> struct Ty<kIQ1_S> { static constexpr int B = 50, TAB = 2048, MIN = 0; };
template <> struct Ty<kIQ1_M> { static constexpr int B = 56, TAB = 2048, MIN = 0; };
template <> struct Ty<kIQ4_NL> { static constexpr int B = 18, TAB = 16, MIN = 0; };
template <> struct Ty<kQ2_K> { static constexpr int B = 84, TAB = 0, MIN = 1; };
template <> struct Ty<kQ3_K> { static constexpr int B = 110, TAB = 0, MIN = 0; };
// the dense matrices' 5- and 6-bit K-quants (GEMV only: ds4_iq's mv; not an expert type)
template <> struct Ty<kQ5_K> { static constexpr int B = 176, TAB = 0, MIN = 1; };
template <> struct Ty<kQ6_K> { static constexpr int B = 210, TAB = 0, MIN = 0; };
// MXFP4 (ds4_gguf.cu's own kernels everywhere else): a block of 32 values, an E8M0 scale then 16 code bytes; the light
// sets take it too
constexpr int kMXFP4 = 39;
template <> struct Ty<kMXFP4> { static constexpr int B = 17, TAB = 0, MIN = 0; };
template <int T> constexpr int kQK = T == kMXFP4 || T == kIQ4_NL ? 32 : 256;   // values a block
static_assert(sizeof(block_iq2_xxs) == 66 && sizeof(block_iq2_xs) == 74 && sizeof(block_iq2_s) == 82 &&
                  sizeof(block_iq3_xxs) == 98 &&
                  sizeof(block_iq3_s) == 110 && sizeof(block_iq4_xs) == 136 && sizeof(block_q4_K) == 144 &&
                  sizeof(block_q2_K) == 84 && sizeof(block_q3_K) == 110 && sizeof(block_iq1_s) == 50 &&
                  sizeof(block_iq1_m) == 56 && sizeof(block_iq4_nl) == 18,
              "ggml's block sizes");

template <int T>
inline size_t row_bytes_t(int k) { return (size_t) (k / kQK<T>) * Ty<T>::B; }

inline uint32_t ld2(const uint8_t* p) {   // 4 bytes from a 2-byte aligned address
    return (uint32_t) p[0] | (uint32_t) p[1] << 8 | (uint32_t) p[2] << 16 | (uint32_t) p[3] << 24;
}
inline uint32_t ld4(const uint8_t* p) { return ld2(p); }
inline float h2f(const uint8_t* p) { return sycl_detail::f16_value((uint16_t) ((uint32_t) p[0] | (uint32_t) p[1] << 8)); }
// 7 sign bits (bits 0..6 of v) -> the 8 signs, bit 7 their parity (ggml's ksigns_iq2xs)
inline uint32_t ksigns8(uint32_t v) {
    v &= 127u;
    return v | ((sycl::popcount(v) & 1u) << 7);
}
// 4 grid bytes with the signs of bits 0..3 of b applied (bit i: byte i negative) - llama.cpp's HIP form, cheap where the
// SIMD byte instructions (__vcmpne4 / strata::kernels::xe::vsub4) are emulated (Volta, RDNA): every grid byte is nonzero and <= 62, so
// (g ^ 0xff) + 1 = -g in each byte with no carry into the next
inline int sgn4(uint32_t g, uint32_t b) {
    const uint32_t lo = (((b & 15u) * 0x10204080u) & 0x80808080u) >> 7;
    return (int) ((g ^ (lo * 255u)) + lo);
}
inline I2 tab16(uint32_t q4, const uint32_t* t) {   // vecdotq's get_int_from_table_16
    uint32_t tmp[2];
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(t);
    const uint32_t t0 = ld4(bytes), t1 = ld4(bytes + 4), t2 = ld4(bytes + 8), t3 = ld4(bytes + 12);
    const uint32_t sel = 0x32103210u | ((q4 & 0x88888888u) >> 1);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const uint32_t sh = 16 * i;
        const uint32_t lo = strata::kernels::xe::byte_perm(t0, t1, q4 >> sh), hi = strata::kernels::xe::byte_perm(t2, t3, q4 >> sh);
        tmp[i] = strata::kernels::xe::byte_perm(lo, hi, sel >> sh);
    }
    return I2{(int) strata::kernels::xe::byte_perm(tmp[0], tmp[1], 0x6420), (int) strata::kernels::xe::byte_perm(tmp[0], tmp[1], 0x7531)};
}
// iq1s_grid_gpu's 8 values in 4 bytes, 4 bits each (low nibbles: values 0..3, high: 4..7; 0 is -1, 2 is +1) -> the 4
// values as int8 codes 8 (n - 1) + 1, or - 1 when the group's delta is negative: ggml's weight dl ((n - 1) +- 0.125), whose
// dl (2 sc + 1) / 8 is this kernel's per-sub-block scale, is then d * (s0 lo + s1 hi) exactly
inline uint32_t iq1_codes(uint32_t n, bool neg) {
    const uint32_t c = (uint32_t) strata::kernels::xe::vsub4((int) (n * 8u), 0x07070707);   // (n * 8 - 7) in each byte: no carry (n <= 15)
    return neg ? (uint32_t) strata::kernels::xe::vsub4((int) c, 0x02020202) : c;
}
// IQ1_M has no d field: its F16 scale is the top nibbles of its four u16 scales (ggml's dequantize_row_iq1_m)
inline uint16_t iq1m_d(const uint8_t* b) {
    const uint32_t s0 = (uint32_t) b[48] | (uint32_t) b[49] << 8, s1 = (uint32_t) b[50] | (uint32_t) b[51] << 8;
    const uint32_t s2 = (uint32_t) b[52] | (uint32_t) b[53] << 8, s3 = (uint32_t) b[54] | (uint32_t) b[55] << 8;
    return (uint16_t) ((s0 >> 12) | ((s1 >> 8) & 0x00f0u) | ((s2 >> 4) & 0x0f00u) | (s3 & 0xf000u));
}
inline int q3k_scale(const uint8_t* sc, int is) {
    const int lo = is < 8 ? sc[is] & 15 : sc[is - 8] >> 4;
    const int hi = (sc[8 + (is & 3)] >> (2 * (is >> 2))) & 3;
    return (lo | (hi << 4)) - 32;
}
inline void q4k_scale(const uint8_t* q, int j, int& d, int& m) {   // ggml's get_scale_min_k4
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// a 32-value sub-block (kb: its index in the row) as 32 int8 codes and its scales: weight i = d s q_i - m, s and m of
// its half (i < 16: s0, m0; else s1, m1)
struct Sub {
    int q[8];
    float d, m0, m1;
    int s0, s1;
};
// (b: the block's address, ib: the sub-block in it (0 for the 32-value types); the token kernels keep ib fixed a lane)
template <int T>
inline void decode_b(const uint8_t* __restrict__ b, int ib, const uint32_t* tab, Sub& s) {
    if constexpr (T == kMXFP4) {   // codes: value j in byte j's low nibble, j + 16 in its high one (any alignment)
        const uint32_t t[4] = {0x03020100u, 0x0C080604u, 0xFDFEFF00u, 0xF4F8FAFCu};   // ggml's kvalues_mxfp4
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint8_t* q = b + 1 + 4 * i;
            const I2 v = tab16((uint32_t) q[0] | (uint32_t) q[1] << 8 | (uint32_t) q[2] << 16 | (uint32_t) q[3] << 24, t);
            s.q[i] = v.x;
            s.q[4 + i] = v.y;
        }
        const uint32_t e = b[0];   // 2^(e - 128): ggml's GGML_E8M0_TO_FP32_HALF (the codes are the E2M1 values doubled)
        s.d = sycl::bit_cast<float>(e < 2 ? 0x00200000u << e : (e - 1) << 23);
        s.s0 = s.s1 = 1;
        (void) ib;
        (void) tab;
    } else if constexpr (T == kIQ2_XXS) {
        const uint8_t* q = b + 2 + 8 * ib;
        const uint32_t a0 = ld2(q), a1 = ld2(q + 4);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const U2 g = ((const U2*) tab)[(a0 >> (8 * l)) & 255u];
            const uint32_t sg = ksigns8(a1 >> (7 * l));
            s.q[2 * l] = sgn4(g.x, sg);
            s.q[2 * l + 1] = sgn4(g.y, sg >> 4);
        }
        s.s0 = s.s1 = 2 * (int) (a1 >> 28) + 1;
        s.d = h2f(b) * 0.125f;
    } else if constexpr (T == kIQ2_XS) {   // d, qs[32] u16 (9-bit grid index, 7-bit sign index), scales[8] (two 4-bit)
        const uint8_t* q = b + 2 + 8 * ib;
        const uint32_t a0 = ld2(q), a1 = ld2(q + 4);
        const uint32_t v[4] = {a0 & 0xffffu, a0 >> 16, a1 & 0xffffu, a1 >> 16};
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const U2 g = ((const U2*) tab)[v[l] & 511u];
            const uint32_t sg = ksigns8(v[l] >> 9);
            s.q[2 * l] = sgn4(g.x, sg);
            s.q[2 * l + 1] = sgn4(g.y, sg >> 4);
        }
        const uint8_t sc = b[66 + ib];
        s.s0 = 2 * (int) (sc & 15) + 1;
        s.s1 = 2 * (int) (sc >> 4) + 1;
        s.d = h2f(b) * 0.125f;
    } else if constexpr (T == kIQ2_S) {
        const uint8_t* qs = b + 2;
        const uint32_t i4 = ld2(qs + 4 * ib), sg4 = ld2(qs + 32 + 4 * ib), h = b[66 + ib], sc = b[74 + ib];
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const U2 g = ((const U2*) tab)[((i4 >> (8 * l)) & 255u) | ((h << (8 - 2 * l)) & 0x300u)];
            const uint32_t sg = sg4 >> (8 * l);
            s.q[2 * l] = sgn4(g.x, sg);
            s.q[2 * l + 1] = sgn4(g.y, sg >> 4);
        }
        s.s0 = 2 * (int) (sc & 15) + 1;
        s.s1 = 2 * (int) (sc >> 4) + 1;
        s.d = h2f(b) * 0.125f;
    } else if constexpr (T == kIQ3_XXS) {
        const uint8_t* qs = b + 2;
        const uint32_t w0 = ld2(qs + 8 * ib), w1 = ld2(qs + 8 * ib + 4), aux = ld2(qs + 64 + 4 * ib);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t w = l < 2 ? w0 : w1, sh = 16 * (l & 1);
            const uint32_t sg = ksigns8(aux >> (7 * l));
            s.q[2 * l] = sgn4(tab[(w >> sh) & 255u], sg);
            s.q[2 * l + 1] = sgn4(tab[(w >> (sh + 8)) & 255u], sg >> 4);
        }
        s.s0 = s.s1 = 2 * (int) (aux >> 28) + 1;
        s.d = h2f(b) * 0.25f;
    } else if constexpr (T == kIQ3_S) {
        const uint8_t* qs = b + 2;
        const uint32_t w0 = ld2(qs + 8 * ib), w1 = ld2(qs + 8 * ib + 4), h = b[66 + ib], s4 = ld2(b + 74 + 4 * ib);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t w = l < 2 ? w0 : w1, sh = 16 * (l & 1);
            const uint32_t i1 = ((w >> sh) & 255u) | ((h << (8 - 2 * l)) & 256u);
            const uint32_t i2 = ((w >> (sh + 8)) & 255u) | ((h << (7 - 2 * l)) & 256u);
            const uint32_t sg = s4 >> (8 * l);
            s.q[2 * l] = sgn4(tab[i1], sg);
            s.q[2 * l + 1] = sgn4(tab[i2], sg >> 4);
        }
        s.s0 = s.s1 = 1 + 2 * (int) ((b[106 + (ib >> 1)] >> (4 * (ib & 1))) & 15);
        s.d = h2f(b);
    } else if constexpr (T == kIQ4_XS) {
        const uint32_t sh = *(const uint16_t*) (b + 2);
        const int ls = ((b[4 + (ib >> 1)] >> (4 * (ib & 1))) & 15) | (((sh >> (2 * ib)) & 3) << 4);
        const uint8_t* qs = b + 8 + 16 * ib;
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const I2 v = tab16(ld4(qs + 4 * t), tab);
            s.q[t] = v.x;
            s.q[4 + t] = v.y;
        }
        s.s0 = s.s1 = ls - 32;
        s.d = h2f(b);
    } else if constexpr (T == kIQ1_S) {   // d, qs[32] (4 a 32-value sub-block), qh[8] u16: 11-bit grid, 3-bit scale, delta
        const uint8_t* qs = b + 2;
        const uint16_t qh = ((const uint16_t*) (b + 34))[ib];
        const bool neg = (qh & 0x8000u) != 0;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t g = tab[qs[4 * ib + l] | (((qh >> (3 * l)) & 7u) << 8)];
            s.q[2 * l] = (int) iq1_codes(g & 0x0F0F0F0Fu, neg);
            s.q[2 * l + 1] = (int) iq1_codes((g >> 4) & 0x0F0F0F0Fu, neg);
        }
        s.s0 = s.s1 = 2 * (int) ((qh >> 12) & 7u) + 1;
        s.d = h2f(b) * 0.125f;
    } else if constexpr (T == kIQ1_M) {   // qs[32], qh[16], scales[8]: no d field; 11-bit grid, two 3-bit scales a 32
        const uint8_t* qs = b + 4 * ib;
        const uint8_t qh0 = b[32 + 2 * ib], qh1 = b[33 + 2 * ib];
        const uint16_t sc = ((const uint16_t*) (b + 48))[ib >> 1];
        const int sh = 6 * (ib & 1);
        const uint32_t ii[4] = {qs[0] | ((uint32_t) (qh0 << 8) & 0x700u), qs[1] | ((uint32_t) (qh0 << 4) & 0x700u),
                                qs[2] | ((uint32_t) (qh1 << 8) & 0x700u), qs[3] | ((uint32_t) (qh1 << 4) & 0x700u)};
        const bool neg[4] = {(qh0 & 0x08u) != 0, (qh0 & 0x80u) != 0, (qh1 & 0x08u) != 0, (qh1 & 0x80u) != 0};
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t g = tab[ii[l]];
            s.q[2 * l] = (int) iq1_codes(g & 0x0F0F0F0Fu, neg[l]);
            s.q[2 * l + 1] = (int) iq1_codes((g >> 4) & 0x0F0F0F0Fu, neg[l]);
        }
        s.s0 = 2 * (int) ((sc >> sh) & 7u) + 1;
        s.s1 = 2 * (int) ((sc >> (sh + 3)) & 7u) + 1;
        s.d = sycl_detail::f16_value(iq1m_d(b)) * 0.125f;
    } else if constexpr (T == kIQ4_NL) {   // d, qs[16]: a block of 32 values, its codebook's int8 a nibble
        const uint8_t* qs = b + 2;   // (18-byte blocks: even, so ld2's 2-byte alignment - not ld4's 4)
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const I2 v = tab16(ld2(qs + 4 * t), tab);
            s.q[t] = v.x;
            s.q[4 + t] = v.y;
        }
        s.s0 = s.s1 = 1;
        s.d = h2f(b);
    } else if constexpr (T == kQ4_K) {
        int sc, m;
        q4k_scale(b + 4, ib, sc, m);
        const uint8_t* qs = b + 16 + 32 * (ib >> 1);
        const int sh = 4 * (ib & 1);
#pragma unroll
        for (int t = 0; t < 8; ++t) s.q[t] = (int) ((ld4(qs + 4 * t) >> sh) & 0x0f0f0f0fu);
        s.s0 = s.s1 = sc;
        s.d = h2f(b);
        s.m0 = s.m1 = h2f(b + 2) * (float) m;
    } else if constexpr (T == kQ5_K) {   // d, dmin, scales[12], qh[32], qs[128]: weight = d sc (nibble + 16 hbit) - dmin m
        int sc, m;
        q4k_scale(b + 4, ib, sc, m);
        const uint8_t* qs = b + 48 + 32 * (ib >> 1);
        const uint8_t* qh = b + 16;
        const int sh = 4 * (ib & 1);
#pragma unroll
        for (int t = 0; t < 8; ++t)
            s.q[t] = (int) (((ld4(qs + 4 * t) >> sh) & 0x0f0f0f0fu) | (((ld4(qh + 4 * t) >> ib) & 0x01010101u) << 4));
        s.s0 = s.s1 = sc;
        s.d = h2f(b);
        s.m0 = s.m1 = h2f(b + 2) * (float) m;
    } else if constexpr (T == kQ6_K) {   // ql[128], qh[64], scales int8[16], d: weight = d sc (q - 32), q 6 bits
        const int h = ib >> 2, quarter = ib & 3;
        const uint8_t* ql = b + 64 * h + (quarter & 1 ? 32 : 0);
        const uint8_t* qh = b + 128 + 32 * h;
        const int sh = 4 * (quarter >> 1);
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            const uint32_t x = ((ld2(ql + 4 * t) >> sh) & 0x0f0f0f0fu) | (((ld2(qh + 4 * t) >> (2 * quarter)) & 0x03030303u) << 4);
            s.q[t] = (int) strata::kernels::xe::vsub4(x, 0x20202020u);
        }
        const int8_t* scl = (const int8_t*) (b + 192) + 8 * h + 2 * quarter;
        s.s0 = scl[0];
        s.s1 = scl[1];
        s.d = h2f(b + 208);
    } else if constexpr (T == kQ2_K) {
        const int n = ib >> 2, j = ib & 3;
        const uint8_t* qs = b + 16 + 32 * n;
#pragma unroll
        for (int t = 0; t < 8; ++t) s.q[t] = (int) ((ld4(qs + 4 * t) >> (2 * j)) & 0x03030303u);
        const int c0 = b[8 * n + 2 * j], c1 = b[8 * n + 2 * j + 1];
        const float dmin = h2f(b + 82);
        s.s0 = c0 & 15;
        s.s1 = c1 & 15;
        s.d = h2f(b + 80);
        s.m0 = dmin * (float) (c0 >> 4);
        s.m1 = dmin * (float) (c1 >> 4);
    } else {   // Q3_K
        const int n = ib >> 2, j = ib & 3;
        const uint8_t* qs = b + 32 + 32 * n;
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            const uint32_t ql = (ld2(qs + 4 * t) >> (2 * j)) & 0x03030303u;
            const uint32_t neg = ((~ld2(b + 4 * t)) >> (4 * n + j)) & 0x01010101u;
            s.q[t] = (int) (ql | (neg * 0xFCu));   // (q - 4 in a byte: q + 0xFC, no carry for q <= 3)
        }
        s.s0 = q3k_scale(b + 96, 8 * n + 2 * j);
        s.s1 = q3k_scale(b + 96, 8 * n + 2 * j + 1);
        s.d = h2f(b + 108);
    }
}
template <int T>
inline void decode(const uint8_t* __restrict__ row, int kb, const uint32_t* tab, Sub& s) {
    decode_b<T>(row + (size_t) (kQK<T> == 32 ? kb : kb >> 3) * Ty<T>::B, kb & 7, tab, s);
}

template <int T> inline const uint32_t* table() {
    if constexpr (T == kIQ2_XXS) return (const uint32_t*) iq2xxs_grid;
    if constexpr (T == kIQ2_XS) return (const uint32_t*) iq2xs_grid;
    if constexpr (T == kIQ2_S) return (const uint32_t*) iq2s_grid;
    if constexpr (T == kIQ3_XXS) return iq3xxs_grid;
    if constexpr (T == kIQ3_S) return iq3s_grid;
    if constexpr (T == kIQ1_S || T == kIQ1_M) return iq1s_grid_gpu;
    if constexpr (T == kIQ4_XS || T == kIQ4_NL) return (const uint32_t*) kvalues_iq4nl;
    return nullptr;
}
template <int T> struct Decoder {
    float value(const uint8_t* row, int i) const {
        Sub d{};
        decode<T>(row, i / 32, table<T>(), d);
        const int j = i % 32;
        const uint32_t byte = ((uint32_t) d.q[j / 4] >> (8 * (j % 4))) & 255u;
        const int code = (int) (byte ^ 128u) - 128;
        return d.d * (float) ((j < 16 ? d.s0 : d.s1) * code) - (j < 16 ? d.m0 : d.m1);
    }
    float dot_q8(const uint8_t* row, const int8_t* x, const float* xd, int k) const {
        float sum = 0.0f;
        for (int b = 0; b < k / 32; ++b) {
            Sub d{};
            decode<T>(row, b, table<T>(), d);
            int lo = 0, hi = 0, sl = 0, sh = 0;
            for (int j = 0; j < 32; ++j) {
                const uint32_t byte = ((uint32_t) d.q[j / 4] >> (8 * (j % 4))) & 255u;
                const int code = (int) (byte ^ 128u) - 128;
                const int v = (int) ((uint8_t) x[b * 32 + j] ^ 128u) - 128;
                if (j < 16) { lo += code * v; sl += v; }
                else { hi += code * v; sh += v; }
            }
            const float dot = d.d * (float) (d.s0 * lo + d.s1 * hi) - (d.m0 * (float) sl + d.m1 * (float) sh);
            sum = sycl::fma(xd[b], dot, sum);
        }
        return sum;
    }
};
template <typename F> bool dispatch(int type, F fn) {
#define DS4_TYPE(T) case T: fn(std::integral_constant<int, T>{}); return true
    switch (type) {
        DS4_TYPE(kIQ2_XXS); DS4_TYPE(kIQ2_XS); DS4_TYPE(kIQ2_S); DS4_TYPE(kIQ3_XXS); DS4_TYPE(kIQ3_S);
        DS4_TYPE(kIQ4_XS); DS4_TYPE(kIQ1_S); DS4_TYPE(kIQ1_M); DS4_TYPE(kIQ4_NL);
        DS4_TYPE(kQ2_K); DS4_TYPE(kQ3_K); DS4_TYPE(kQ4_K); DS4_TYPE(kQ5_K); DS4_TYPE(kQ6_K);
        DS4_TYPE(kMXFP4);
        default: return false;
    }
#undef DS4_TYPE
}

inline float value(int type, const uint8_t* row, int index) {
#define DS4_VALUE(T) case T: return Decoder<T>{}.value(row, index)
    switch (type) {
        DS4_VALUE(kIQ2_XXS); DS4_VALUE(kIQ2_XS); DS4_VALUE(kIQ2_S); DS4_VALUE(kIQ3_XXS); DS4_VALUE(kIQ3_S);
        DS4_VALUE(kIQ4_XS); DS4_VALUE(kIQ1_S); DS4_VALUE(kIQ1_M); DS4_VALUE(kIQ4_NL);
        DS4_VALUE(kQ2_K); DS4_VALUE(kQ3_K); DS4_VALUE(kQ4_K); DS4_VALUE(kQ5_K); DS4_VALUE(kQ6_K); DS4_VALUE(kMXFP4);
        default: return 0.0f;
    }
#undef DS4_VALUE
}
}  // namespace strata::ds4::gguf::iq::sycl_decode
