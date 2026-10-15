// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/qi_kernels.hpp - the image engine's element kernels for Qwen-Image-2.1's DiT.
//
// The DiT runs on one sequence of T tokens: the n_text text tokens, then the image tokens.  Each block scales its
// two LayerNorms and gates its two residual additions by the time modulation, one row for the image tokens (the
// step's time) and one for the text tokens (time 0): `mod` is [2][C], row 0 the image's, row 1 the text's.
// Activations are FP32 [T, C] row-major; the products' inputs FP16 bits.
#pragma once

#include <cstdint>

namespace strata::image {

/// y16[t] = LayerNorm(x[t]) * (1 + mod[t < n_text][...]) (no affine, eps 1e-6), FP16.
void qi_ln_mod16(const float* x, const float* mod, int64_t n_text, int64_t T, int64_t C, uint16_t* y16, void* stream);

/// x[t] += h[t] * tanh(mod[t < n_text][...]).
void qi_gate_add(float* x, const float* h, const float* mod, int64_t n_text, int64_t T, int64_t C, void* stream);

/// For each token and head: q and k RMS-normed over the head (weights wq, wk; eps 1e-6), then rotated pairwise
/// (2j, 2j + 1) by cos[t][j], sin[t][j] (rope tables [T][head_dim / 2]); written as FP16 q16, k16, and v as v16, all
/// [T][heads * head_dim].
void qi_qk_norm_rope16(const float* q, const float* k, const float* v, const float* wq, const float* wk,
                       const float* cos, const float* sin, int64_t T, int64_t heads, int64_t head_dim,
                       uint16_t* q16, uint16_t* k16, uint16_t* v16, void* stream);

/// out[t, h, :] = softmax(q . k^T / sqrt(head_dim)) . v over the rows token t sees: a text token (t < n_text) the
/// text tokens up to itself, an image token every token.  FP32 accumulation; the plain form, every GPU (the check).
void qi_attention_plain(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t T, int64_t heads,
                        int64_t head_dim, int64_t n_text, float* out, void* stream);

/// The rows q16, k16 and v16 must hold for qi_attention: T rounded up to a multiple of 128, the rows past T zero.
constexpr int64_t qi_attn_rows(int64_t T) { return (T + 127) / 128 * 128; }

/// The same attention on the GPU's matrix engines where it has them (src/image/qi_attn.cpp), FP16 out16
/// [T][heads * head_dim]; else qi_attention_plain through `scratch` (FP32 [T][heads * head_dim]).
void qi_attention(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t T, int64_t heads,
                  int64_t head_dim, int64_t n_text, uint16_t* out16, float* scratch, void* stream);

/// The path qi_attention takes on the current GPU, for the report.
const char* qi_attention_path();

/// h16[t, i] = gu[t, inter + i] * silu(gu[t, i]) (the fused gate_up product: gate first, then up).
void qi_swiglu16(const float* gu, int64_t T, int64_t inter, uint16_t* h16, void* stream);

/// y16 = FP16(x), n elements.
void qi_to_f16(const float* x, int64_t n, uint16_t* y16, void* stream);

/// y16[r] = FP16(x[r] / rms(x[r]) * w) (eps 1e-6), rows of C.
void qi_rms_norm16(const float* x, const float* w, int64_t rows, int64_t C, uint16_t* y16, void* stream);

/// y16 = FP16(GELU(x)), GELU's tanh form, n elements.
void qi_gelu16(const float* x, int64_t n, uint16_t* y16, void* stream);

/// x = SiLU(x) in place, and y16 = FP16 of it (y16 may be null), n elements.
void qi_silu(float* x, int64_t n, uint16_t* y16, void* stream);

/// tokens16[p][c] = FP16(latent[c][p]), p < HW, c < C: a latent [C][H][W] as the image's tokens.
void qi_latent_tokens16(const float* latent, int64_t C, int64_t HW, uint16_t* tokens16, void* stream);

/// latent[c][p] = tokens[p][c]: the tokens' outputs [HW][C] back as a latent [C][H][W].
void qi_tokens_latent(const float* tokens, int64_t C, int64_t HW, float* latent, void* stream);

/// x += a * v, n elements.
void qi_axpy(float* x, const float* v, float a, int64_t n, void* stream);

}  // namespace strata::image
