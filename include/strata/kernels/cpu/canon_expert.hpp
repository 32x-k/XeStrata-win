// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/kernels/cpu/canon_expert.hpp - an expert's rows on the CPU in the same arithmetic as the GPU's grouped
// experts (canon_math.hpp), so a verify window's expert gives the same bits on either side (#152): the activations
// as q8_1 blocks with the canonical scale, each row's 32-value calls summed in the GPU's lane order.  native_quant_act,
// native_quant_h, native_gu_rows and native_down_rows take these for the layers canon_layer accepts.
#pragma once

#include "strata/kernels/cpu/native_expert.hpp"

#include <cstdint>

namespace strata::kernels::cpu {

/// Whether this layer's experts take the canonical rows.
bool canon_layer(const NativeFmt& f);
/// The order of an activation's int8 values in memory (the values and their scales are the GPU's either way).
enum class CanonOrder { plain, q2_planes, iq4nl_calls };

/// n floats -> the canonical int8 values and their scales, one for each `group` values (32 or 256; n a multiple of
/// it): the GPU's q8_1 blocks' values, laid out for the CPU (the scales as floats, each group's integer sum, then the
/// values from a 64-byte boundary).  `dst` 64-byte aligned, n + 64 + n / group x 8 bytes at most.  `order`: a down
/// projection's values in the order its rows decode their weights (Q2_0's 2-bit planes, IQ4_NL's calls).
void canon_quant(const float* x, int64_t n, int group, void* dst, CanonOrder order = CanonOrder::plain);
/// ff[t][r] = the SwiGLU of gate row r and up row r against act[t] (q8_1), rows [r0, r1).
void canon_gu_rows(const NativeFmt& f, const uint8_t* blob, const void* const* act, int nt, float* const* ff, int r0,
                   int r1);
/// out[t][r] = down row r against hq[t] (q8_1), rows [r0, r1).
void canon_down_rows(const NativeFmt& f, const uint8_t* blob, const void* const* hq, int nt, float* const* out, int r0,
                     int r1);

}  // namespace strata::kernels::cpu
