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

/// The text encoder's q and k ([T][q_heads * head_dim], [T][kv_heads * head_dim]): each head RMS-normed (wq, wk; eps
/// 1e-6), then rotated as NEOX does (pair (j, j + head_dim / 2) by position t times theta^(-2j / head_dim)); FP16 out,
/// v as v16.
void qi_text_qk_rope16(const float* q, const float* k, const float* v, const float* wq, const float* wk, int64_t T,
                       int64_t q_heads, int64_t kv_heads, int64_t head_dim, float theta, uint16_t* q16, uint16_t* k16,
                       uint16_t* v16, void* stream);

/// Causal attention with grouped K/V heads (query head h reads K/V head h / (q_heads / kv_heads)): out [T][q_heads *
/// head_dim] FP32, the plain form (the text encoder's few tokens).
void qi_attention_causal(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t T, int64_t q_heads,
                         int64_t kv_heads, int64_t head_dim, float* out, void* stream);

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

// --- the vision encoder (Qwen3-VL's ViT, src/image/qi_vision.cpp)

/// y[t, :] += b, T rows of N.
void qi_bias_add(float* y, const float* b, int64_t T, int64_t N, void* stream);

/// y16 = LayerNorm(x) * w + b per row of C (eps), FP16 out.
void qi_layer_norm16(const float* x, const float* w, const float* b, int64_t T, int64_t C, float eps, uint16_t* y16,
                     void* stream);

/// y16 = GELU(x), the exact (erf) form, n elements.
void qi_gelu_erf16(const float* x, int64_t n, uint16_t* y16, void* stream);

/// qkv [T][3 * heads * hd] (biased) -> q16, k16, v16 [rows][heads][128]: each head's hd channels, the rest zero (the
/// attention takes 128); q and k rotated in 2D (channel j paired with j + hd / 2; the first hd / 4 pairs by the
/// token's row yx[2t], the next by its column yx[2t + 1], each at theta^(-4j / hd)); q times q_scale.
void qi_vit_qkv_rope16(const float* qkv, const int32_t* yx, int64_t T, int64_t heads, int64_t hd, float theta,
                       float q_scale, uint16_t* q16, uint16_t* k16, uint16_t* v16, void* stream);

/// out16 [T][heads * hd] = the first hd channels of each head of in16 [T][heads][128].
void qi_unpad_heads16(const uint16_t* in16, int64_t T, int64_t heads, int64_t hd, uint16_t* out16, void* stream);

}  // namespace strata::image
