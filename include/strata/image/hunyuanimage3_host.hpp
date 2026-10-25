// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/hunyuanimage3_host.hpp - one HunyuanImage 3.0 decoder layer on the host, FP32: the reference the GPU
// engine's layer is checked against (records/2026-10-23-hunyuanimage3-plan/hunyuanimage3_ref.py is the same layer in NumPy).
//
// Not the engine's path: it reads the quantized file's weights, dequantizes them a matrix at a time and multiplies on
// the CPU, so that the layer's arithmetic - the per-KV-head qkv interleave, the qk-norm, the 2D RoPE, the causal
// mask, the router and the SwiGLU order - can be settled before any of it becomes a GPU kernel.
#pragma once

#include "strata/image/hunyuanimage3_file.hpp"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace strata::image {

/// One layer's intermediates, in the shapes `HunyuanImage3Scratch` holds them, so `probe stages` can hold the device's walk
/// to the host's FP32 stage by stage (records/.../HANDOFF.md: the layer's own error was not in the products, the
/// attention, the router's precision, the rotation's rounding nor the quantized form - so it is somewhere in the
/// chain, and the chain's first differing stage is the answer).  Every array is FP32; the device's FP16 stages are
/// read back as the values they hold, so a stage's rel has FP16's 5e-4 floor and is told from a real difference.
/// A null pointer costs the host layer nothing.
struct HunyuanImage3HostStages {
    std::vector<float> xn32,     // [t][hidden] the input norm
        xr16,                    // [t][hidden] it rotated (the product's operand)
        qkv32,                   // [t][qkv_n] the qkv product, the file's [kv head][q0..q(groups-1),k,v] order
        q16,                     // [t][heads * head_dim] after the rope and the qk-norm
        k16,                     // [t][heads * head_dim] its KV head repeated over the query heads it serves
        v16,                     // [t][heads * head_dim] the same, unrotated
        a16,                     // [t][hidden] the attention's output, before the rotation
        o32,                     // [t][hidden] the o product (the residual's addend)
        xn232,                   // [t][hidden] the post-attention norm
        xr216,                   // [t][hidden] it rotated
        act16,                   // [t][inter] the shared expert's SwiGLU, rotated
        shared32,                // [t][hidden] the shared expert's output
        logits;                  // [t][experts] the router's raw logits
};

/// Layer `layer` of the model in `f`, `t` tokens of `x` ([t][4096], FP32) -> `out` ([t][4096]).  Text positions
/// only (y = x = the token's index) and a causal mask.  False with `err` on a missing or malformed tensor.
bool hunyuanimage3_layer_host(const HunyuanImage3File& f, int64_t layer, const float* x, int64_t t, float* out,
                              std::string& err);

/// The same layer on a sequence that ends with an image block: `t` tokens whose 2D rope positions are `pos_y`,
/// `pos_x`, the last `n_img` of them the block.  The mask is the reference's - causal for every row, the block
/// filled True - so a block row sees all the text and all of the block, and a text row its prefix.  Passing
/// `pos_y = pos_x = nullptr` gives y = x = the index.  (records/.../hy3_torch_ref.py runs Tencent's own classes
/// and `hy3_conform.py` is the NumPy form of these formulas; both hold this to 1e-6 on random weights.)
bool hunyuanimage3_layer_host(const HunyuanImage3File& f, int64_t layer, const float* x, int64_t t, float* out,
                              const int32_t* pos_y, const int32_t* pos_x, int64_t n_img, std::string& err,
                              HunyuanImage3HostStages* stages = nullptr);

/// The model's own special token ids (its config's `im_start_id`/`im_end_id`/`image_token_id` and its tokenizer's
/// `added_tokens` for the rest): the markers a sequence is built with, and the three whose rows a step replaces.
/// `<img_size_256>` and `<img_ratio_0>` are the first of their runs (the sizes 128034 .. 128043, the ratio buckets
/// 0 .. 32 from 128044 and 33 .. 36 - the four sizes the image processor adds - from 130103, added to the vocabulary
/// later): which ones a prompt carries is the caller's choice, not the engine's.
constexpr int32_t kHunyuanImage3Boi = 128000, kHunyuanImage3Eoi = 128001, kHunyuanImage3Image = 128006;
constexpr int32_t kHunyuanImage3Timestep = 128017, kHunyuanImage3Guidance = 130082;
constexpr int32_t kHunyuanImage3TimestepR = 130107;
constexpr int32_t kHunyuanImage3ImgSize256 = 128034, kHunyuanImage3ImgRatio0 = 128044, kHunyuanImage3ImgRatio33 = 130103;
constexpr int32_t kHunyuanImage3RatioBuckets = 37;
constexpr int32_t kHunyuanImage3JointImgSep = 128077;   // `<joint_img_sep>`: between a reference's VAE and ViT tokens
/// `<img_size_N>`: the run's ids are 128034 .. 128043 for N = 256, 512, 768, 1024, ... (the model's sizes grow by
/// 256).
constexpr int64_t hunyuanimage3_img_size_of(int32_t id) { return 256 * (int64_t) (id - kHunyuanImage3ImgSize256 + 1); }
/// i of `<img_ratio_i>` (-1 for an id that is none), and the id of bucket i: the two runs above.
constexpr int32_t hunyuanimage3_ratio_of(int32_t id) {
    return id >= kHunyuanImage3ImgRatio0 && id < kHunyuanImage3ImgRatio0 + 33 ? id - kHunyuanImage3ImgRatio0
         : id >= kHunyuanImage3ImgRatio33 && id < kHunyuanImage3ImgRatio33 + 4 ? 33 + id - kHunyuanImage3ImgRatio33
         : -1;
}
constexpr int32_t hunyuanimage3_ratio_id(int32_t i) {
    return i < 33 ? kHunyuanImage3ImgRatio0 + i : kHunyuanImage3ImgRatio33 + i - 33;
}

/// One image block of a sequence as its markers lay it out.  A reference image (an edit's) is `<boi> <img_size_N>
/// <img_ratio_i> <timestep> <img> * vae_n <joint_img_sep> <img> * vit_n <eoi>`: its VAE latent's tokens, then its
/// SigLIP2 features' (the processor's fixed 1024 slots, the real patches first); the generated image's block is
/// `<boi> <img_size_N> <img_ratio_i> <timestep> <guidance> <timestep_r> <img> * n` and ends the sequence.  The
/// grids are not in the ids: the VAE grid comes from the bucket (N, i) and the ViT grid from the image's naflex
/// size, which the caller sets (hunyuanimage3_positions_of takes them).
struct HunyuanImage3Block {
    bool reference = false;     // a reference image's block (else the generated image's)
    int64_t boi = 0;            // the `<boi>`
    int64_t timestep = -1;      // the `<timestep>` (a reference's row is timestep_emb(0))
    int64_t base_size = 0;      // N of `<img_size_N>`
    int32_t ratio_index = 0;    // i of `<img_ratio_i>`
    int64_t vae_start = 0, vae_n = 0;   // the VAE tokens (the generated image's block: its tokens)
    int64_t vit_start = 0, vit_n = 0;   // a reference's ViT tokens (0 for the generated image)
    int64_t end = 0;            // one past the block's last token (`<eoi>` included for a reference)
    int64_t vae_h = 0, vae_w = 0, vit_h = 0, vit_w = 0;   // the grids, filled by the caller
};

/// The image blocks of `ids` ([t], already trimmed at the last `<img>`): the references in order, then the
/// generated image's.  False with `err` on a layout that is not one of the two above.
bool hunyuanimage3_blocks(const int32_t* ids, int64_t t, std::vector<HunyuanImage3Block>& blocks, std::string& err);

/// The 2D rope positions and the attention's key limits of a sequence with `blocks` (their grids filled): text
/// keeps y = x = its index; every image grid (a reference's VAE and ViT grids, the generated block's) gets the
/// reference's `build_2d_rope` positions from its start L - beta_y = L + (w h - h)/2, beta_x = L + (w h - w)/2,
/// (r, c) at (beta_y + r, beta_x + c) truncated - and a ViT slot past its real patches keeps its index.  `key_lim`
/// ([t]) is the first key a row does not see: a row inside a reference's joint span (its VAE tokens, the separator
/// and its ViT tokens) sees up to the span's end, every other row before the generated block its own prefix (the
/// generated block's rows see everything: qi_attention's n_text is its start).
bool hunyuanimage3_positions_of(int64_t t, const std::vector<HunyuanImage3Block>& blocks, std::vector<int32_t>& y,
                                std::vector<int32_t>& x, std::vector<int32_t>& key_lim, std::string& err);

/// The engine's sequence: `ids` cut at the last `<img>` token.  The reference's own sequence ends with `<eoi>`
/// and `</answer>` after the block; none of that can reach the image (its rows are causal and come after the
/// block), and the reference drops it from step 1 on.  Returns the length to keep, -1 with `err` if the ids hold
/// no image block, no `<boi>`, more than one block, or something other than `<eoi>` after the block.
int64_t hunyuanimage3_trim(const int32_t* ids, int64_t t, std::string& err);

/// The 2D rope positions of `ids` ([t]), whose tail is `token_h * token_w` `<img>` tokens: text keeps y = x = its
/// index, the block gets beta_y = L + (w h - h)/2, beta_x = L + (w h - w)/2 with the grid (r, c) at
/// (beta_y + r, beta_x + c), truncated.  `img_start` comes back as L.  False with `err` when the tail is not the
/// block (the engine's sequence is text, then the image, because post-image text cannot reach the image).
bool hunyuanimage3_positions(const int32_t* ids, int64_t t, int64_t token_h, int64_t token_w, std::vector<int32_t>& y,
                             std::vector<int32_t>& x, int64_t& img_start, std::string& err);

/// The model's input for one denoise step, host FP32: `hidden` ([t][4096]) with the text rows from `wte`, the
/// three meta rows replaced by their embedders (`<timestep>` at `timestep`, `<guidance>` at `guidance`,
/// `<timestep_r>` at `timestep_r` - the reference's `instantiate_*_tokens`) and the image block from
/// `patch_embed` at `timestep` (its own `time_embed`, on the same `timestep`).  `hunyuanimage3_step_host`
/// assembles its input this way; the GPU walk keeps these pieces on the host until they have kernels of their
/// own, so a step that runs only the layers on the card enters through here.
bool hunyuanimage3_hidden_in_host(const HunyuanImage3File& f, const int32_t* ids, int64_t t, int64_t token_h,
                                  int64_t token_w, const float* latent, float timestep, float timestep_r,
                                  float guidance, float* hidden, std::string& err);

/// The three meta rows a step's embedders write over the `wte` rows: `<timestep>` at `timestep`, `<guidance>` at
/// `guidance` and `<timestep_r>` at `timestep_r`, into the rows of the `t` text ids in `hidden` ([t][4096]).  Those
/// are all that changes in the text between a request's steps, so a walk that keeps the `wte` rows refreshes only
/// these (and the block); `hunyuanimage3_hidden_in_host` is the whole input, `wte` and `patch_embed` included.
bool hunyuanimage3_meta_rows_host(const HunyuanImage3File& f, const int32_t* ids, int64_t t, float timestep,
                                  float timestep_r, float guidance, float* hidden, std::string& err);

/// One denoise step on the host, FP32: the model's `layers` decoder layers over the sequence `ids` ([t], its tail
/// the `token_h * token_w` `<img>` tokens), the latent [32][token_h][token_w] at `timestep`, and the velocity in
/// [32][token_h][token_w].  `timestep_r` feeds `<timestep_r>` and `guidance` (the model's own number, the
/// reference's `1000 * guidance_scale`) feeds `<guidance>`; those three ids' embeddings are the embedders' outputs,
/// the rest go through `wte`.  `layer_done(l)` after each layer (without it the layers print with their times).
bool hunyuanimage3_step_host(const HunyuanImage3File& f, const int32_t* ids, int64_t t, int64_t token_h,
                             int64_t token_w,
                             const float* latent, float timestep, float timestep_r, float guidance, int64_t layers,
                             float* velocity, std::string& err,
                             const std::function<void(int64_t)>& layer_done = {});

/// The whole sampler on the host: Euler from the seed's Gaussian noise through `sigmas` (steps + 1, falling to 0),
/// the latent out [32][token_h][token_w].
bool hunyuanimage3_sample_host(const HunyuanImage3File& f, const int32_t* ids, int64_t t, int64_t token_h,
                               int64_t token_w,
                               const std::vector<float>& sigmas, float guidance_scale, int64_t layers, uint32_t seed,
                               float* latent_out, std::string& err);

/// The Instruct-Distil sampler's schedule: `steps` + 1 sigmas from 1 down to 0, shifted by `shift` the way the
/// reference's FlowMatchDiscreteScheduler does it (sd3's shift: s*t / (1 + (shift - 1)*t) on a linspace(1, 0)),
/// and the timesteps the model sees (`sigmas * 1000`).  `timestep_r` of step i is the *next* entry of the full
/// schedule (meanflow's r).  The Euler step is fp32 and not cast back (the reference has its cast commented out).
void hunyuanimage3_sigmas(int64_t steps, float shift, float* sigmas /* steps + 1 */);
void hunyuanimage3_euler_step(float* x, const float* v, int64_t n, float sigma, float sigma_next);

}  // namespace strata::image

namespace strata::image {

/// The pieces around the decoder layers, host FP32, from the quantized file's BF16 tensors:
///
///   wte:   `t` token ids -> [t][4096] (model.wte.weight).
///   time:  one of the model's TimestepEmbedders - "time_embed" (patch_embed's), "time_embed_2" (final_layer's),
///          "timestep_emb", "guidance_emb", "timestep_r_emb" (the meta tokens') - at scalar `t`: the sinusoidal 256
///          values through Linear(4096) -> GELU -> Linear(4096), [4096].
///   patch: the VAE latent [32][h][w] at `t` -> [h*w][4096]: conv 3x3 (32->1024), then the ResBlock (1024->4096)
///          whose embedding is time_embed(t).
///   final: hidden states [h*w][4096] at `t` -> the velocity [32][h][w]: the ResBlock (4096->1024) with
///          time_embed_2(t), then GroupNorm(32) -> SiLU -> conv 3x3 (1024->32).
bool hunyuanimage3_wte_host(const HunyuanImage3File& f, const int32_t* ids, int64_t t, float* out, std::string& err);
bool hunyuanimage3_time_host(const HunyuanImage3File& f, const std::string& which, float t, float* out,
                             std::string& err);
bool hunyuanimage3_patch_embed_host(const HunyuanImage3File& f, const float* latent, int64_t h, int64_t w, float t,
                                    float* out, std::string& err);
bool hunyuanimage3_final_layer_host(const HunyuanImage3File& f, const float* x, int64_t h, int64_t w, float t,
                                    float* out, std::string& err);

}  // namespace strata::image
