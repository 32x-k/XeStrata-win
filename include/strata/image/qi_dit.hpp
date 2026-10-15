// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/qi_dit.hpp - Qwen-Image-2.1's DiT on the GPU: the velocity of a latent at a time, and the
// flow sampler over Turbo's schedule.
//
// The sequence is the n_text text tokens (the text encoder's states, projected once by set_context) and the image's
// H x W tokens (the latent's 64 channels each).  A text token sees the text before it and its time is 0, so its
// hidden states, and its K and V in every block, are the same at every step: the first step keeps the blocks' text K
// and V (the prefix cache), and the later ones compute the image's rows alone.
#pragma once

#include "strata/image/qi_stream.hpp"
#include "strata/prefill/gemm.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata {
class GgufFile;
}

namespace strata::image {

/// A request stopped on its caller's word (the step callback said so).
struct QiStopped : std::runtime_error {
    QiStopped() : std::runtime_error("image engine: stopped") {}
};

/// Philox 4x32-10 normal values as PyTorch's CUDA randn gives them for `seed` (the generator stable-diffusion.cpp
/// uses): value i from counter (offset, 0, i, 0) and key (seed low, seed high), Box-Muller's sine half.
std::vector<float> philox_randn(uint64_t seed, uint64_t n, uint32_t offset = 0);

/// Qwen-Image-2.1-Turbo's own 8-step schedule (its model_index.json's sample_sigmas) and the terminal 0.
const std::vector<float>& turbo_sigmas();

class QiDit {
public:
    QiDit();
    ~QiDit();
    QiDit(const QiDit&) = delete;
    QiDit& operator=(const QiDit&) = delete;

    /// Reads the DiT's GGUF (mapped) and puts its weights on the current GPU.  False with `err` on failure.
    bool open(const std::string& path, void* stream, std::string& err);

    /// The text encoder's states [n_text][4096] (host FP32): projected into the sequence's text rows; the prefix
    /// cache starts again.
    void set_context(const float* ctx, int64_t n_text);

    /// out = the velocity [64][H][W] (device FP32) of the latent x (device FP32, the same shape) at time t (sigma x
    /// 1000).  `prefix`: the text rows' K and V from the cache (a step after the first of one context).
    void forward(const float* x, float t, int64_t H, int64_t W, bool prefix, float* out);

    /// The flow sampler: x (device FP32 [64][H][W], the starting noise) through `sigmas` by Euler's steps.
    /// `step` (if set) is called after each step with (steps done, steps); it returns false to stop (the sampler then
    /// throws QiStopped).
    void sample(float* x, int64_t H, int64_t W, const std::vector<float>& sigmas,
                const std::function<bool(int, int)>& step = {});

    int64_t channels() const { return in_ch_; }
    int64_t hidden() const { return C_; }
    /// The bytes the weights outside the blocks take on the GPU (always there), and the blocks' (in the file).
    uint64_t weight_bytes() const { return weight_bytes_; }
    uint64_t block_bytes() const { return blocks_stream_.total(); }
    /// The two slots a streamed block walk takes at least.
    uint64_t block_min() const { return 2 * (blocks_stream_.count() ? blocks_stream_.bytes(0) : 0); }
    /// The VRAM the blocks may take: those that fit stay, the rest are streamed (LayerStream).
    void set_block_budget(uint64_t bytes) { blocks_stream_.configure(bytes, s_); }
    int resident_blocks() const { return blocks_stream_.resident(); }
    /// Frees the blocks' VRAM (they come back from the file on use).
    void drop_blocks() { blocks_stream_.drop(); }
    /// Frees the activations (grown again on use).
    void drop_activations();
    /// The GPU memory the activations of a T-token sequence take, the MLP computed `mlp_rows` rows at a time (0: all).
    uint64_t activation_bytes(int64_t T, int64_t mlp_rows = 0) const;
    /// The rows the MLP takes at a time (0: all of them): its 2 x 12288-wide product is the largest activation.
    void set_mlp_rows(int64_t rows);

private:
    struct W {
        int type = 0;
        int64_t n = 0, k = 0;   // n rows of k
        const void* d = nullptr;
        uint64_t row_bytes = 0;
    };
    struct Block {
        W q, k, v, o, gu, down;   // d: the offset within the block's region (LayerStream)
        uint64_t nq = 0, nk = 0;  // the FP32 norms' offsets
        uint16_t *pk = nullptr, *pv = nullptr;   // the text rows' K and V (the prefix cache)
    };
    W upload(const std::string& name);
    const float* upload_f32(const std::string& name, float add = 0.0f);
    void product(const uint16_t* x16, const W& w, float* y, int64_t T, int64_t ldy = 0, int64_t row0 = 0,
                 int64_t rows = -1);
    void step_modulation(float t);
    void ensure(int64_t T, int64_t HW);

    std::unique_ptr<GgufFile> g_;
    void* s_ = nullptr;
    prefill::Gemm gemm_;
    std::vector<void*> allocs_;
    uint64_t weight_bytes_ = 0;
    int64_t C_ = 4096, in_ch_ = 64, heads_ = 32, inter_ = 12288, n_text_ = 0;
    W t1_, t2_, mod_, nout_, pout_, tin_, tout_, iin_;
    const float* tnorm_ = nullptr;
    std::vector<Block> blocks_;
    LayerStream blocks_stream_;
    // activations, grown on demand
    int64_t cap_T_ = 0, mlp_cap_ = 0, mlp_rows_ = 0;
    float *x_ = nullptr, *q_ = nullptr, *k_ = nullptr, *v_ = nullptr, *h_ = nullptr, *gu_ = nullptr;
    uint16_t *h16_ = nullptr, *q16_ = nullptr, *k16_ = nullptr, *v16_ = nullptr, *o16_ = nullptr;
    float *cos_ = nullptr, *sin_ = nullptr, *attn_scratch_ = nullptr;
    int64_t rope_H_ = -1, rope_W_ = -1, rope_text_ = -1;
    float *text_ = nullptr;           // the projected text rows [n_text][C]
    float *mods_ = nullptr;           // [4][2][C]: the step's modulation, row 0 the image's, row 1 the text's
    float *scale_ = nullptr;          // [2][C]: the final norm's scale (row 0)
    float *temb_ = nullptr;           // [2][C] the time embeddings (t, 0), after their SiLU
    uint16_t *small16_ = nullptr;     // [2][C] FP16 operands of the time path
};

}  // namespace strata::image
