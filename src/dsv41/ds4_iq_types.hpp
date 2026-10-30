// SPDX-FileCopyrightText: 2026 Peasant Smith and the Project Maya contributors
// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT

// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
#pragma once
#include <cstddef>
#include <cstdint>
namespace strata::ds4::gguf::iq {
// ggml type ids
constexpr int kQ5_K = 13, kQ6_K = 14;   // (the dense matrices' K-quants, GEMV only)
constexpr int kQ2_K = 10, kQ3_K = 11, kQ4_K = 12, kIQ2_XXS = 16, kIQ2_XS = 17, kIQ3_XXS = 18, kIQ1_S = 19,
              kIQ4_NL = 20, kIQ3_S = 21, kIQ2_S = 22, kIQ4_XS = 23, kIQ1_M = 29;

/// The trellis types (Strata's, ds4_tq.cu): ggml's TQK6 / TQK7 (145 / 146, the table codebook) and the experimental
/// TQK5X / TQK4X (224 / 225, computed codebook) and TQK5T / TQK4T (226 / 227, table): a block of 128 values is
/// {fp16 d; 32 steps of K bits}, 2 + 4 K bytes (tools/trellis/tqenc.hpp).  They go through the same entry points below.
constexpr int kTQK6 = 145, kTQK7 = 146, kTQK5X = 224, kTQK4X = 225, kTQK5T = 226, kTQK4T = 227;
constexpr int tq_kbits(int type) {
    return type == kTQK6 ? 6 : type == kTQK7 ? 7 : (type == kTQK5X || type == kTQK5T) ? 5 : (type == kTQK4X || type == kTQK4T) ? 4 : 0;
}
constexpr bool is_tq(int type) { return tq_kbits(type) > 0; }

inline int block_bytes(int t);
inline bool supported(int t) {

    // the types the kernels below are instantiated for (DS4_IQ_SWITCH + the trellis ones + the dense K-quants)
    switch (t) {
        case kIQ2_XXS: case kIQ2_XS: case kIQ2_S: case kIQ3_XXS: case kIQ3_S: case kIQ4_XS:
        case kIQ1_S: case kIQ1_M: case kIQ4_NL:
        case kQ4_K: case kQ5_K: case kQ6_K: case kQ2_K: case kQ3_K:
            return true;
        default:
            return is_tq(t);
    }
}

inline bool known(int t) { return block_bytes(t) > 0; }

inline int block_values(int t) {
    if (t == kIQ4_NL) return 32;   // (ggml's QK4_NL: the only type here whose block is not 256 values)
    return is_tq(t) ? 128 : block_bytes(t) > 0 ? 256 : 0;
}

inline int block_bytes(int t) {
    if (is_tq(t)) return 2 + 4 * tq_kbits(t);
    switch (t) {
        case kIQ2_XXS: return 66;
        case kIQ2_XS: return 74;
        case kIQ2_S: return 82;
        case kIQ3_XXS: return 98;
        case kIQ3_S: return 110;
        case kIQ4_XS: return 136;
        case kQ4_K: return 144;
        case kQ5_K: return 176;
        case kQ6_K: return 210;
        case kQ2_K: return 84;
        case kQ3_K: return 110;
        case kIQ1_S: return 50;
        case kIQ1_M: return 56;
        case kIQ4_NL: return 18;
        default: return 0;
    }
}

inline const char* name(int t) {
    switch (t) {
        case kTQK6: return "TQK6";
        case kTQK7: return "TQK7";
        case kTQK5X: return "TQK5X";
        case kTQK4X: return "TQK4X";
        case kTQK5T: return "TQK5T";
        case kTQK4T: return "TQK4T";
        case kIQ2_XXS: return "IQ2_XXS";
        case kIQ2_XS: return "IQ2_XS";
        case kIQ2_S: return "IQ2_S";
        case kIQ3_XXS: return "IQ3_XXS";
        case kIQ3_S: return "IQ3_S";
        case kIQ4_XS: return "IQ4_XS";
        case kQ4_K: return "Q4_K";
        case kQ5_K: return "Q5_K";
        case kQ6_K: return "Q6_K";
        case kQ2_K: return "Q2_K";
        case kQ3_K: return "Q3_K";
        case kIQ1_S: return "IQ1_S";
        case kIQ1_M: return "IQ1_M";
        case kIQ4_NL: return "IQ4_NL";
        default: return "?";
    }
}


inline size_t row_bytes(int type, int64_t k) { const int b = block_values(type); return b ? (size_t) (k / b) * block_bytes(type) : 0; }
}  // namespace strata::ds4::gguf::iq
