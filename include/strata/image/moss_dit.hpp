// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/moss_dit.hpp - MOSS-Image's DiT on the GPU: one stream of 42 blocks (3072 wide, 24 heads of
// 128) over a context's rows, the image's latent tokens [h w][128] and then the text's, the velocity of a
// flow-matching step.
//
// The text encoder's states go through the DiT's text embedder once a context: an input projection and two blocks of
// attention among the text rows alone (no time in them), so the rows they give are the same at every step.  Each of
// the 42 blocks then takes all of a context's rows: RMSNorm scaled and shifted by the time (one modulation the blocks
// share: shift, scale and gate of the attention and of the MLP), q and k RMS-normed a head and rotated by a RoPE of
// four axes, one attention over every row (no mask), and an MLP whose activation is SiTU (4 tanh(g / 4) sigmoid(g)
// times 25 tanh(h / 25)), each added back through its gate.  The RoPE gives a token four coordinates, 16 frequencies
// each (10000^(-k / 16)), pairs (j, j + 64) rotated: an image row (0, the text's length, its row and its column, both
// centred), a text row (0, i, 0, 0).  The last norm and projection take the image's rows alone.
//
// The weights are the diffusers folder's bf16 safetensors (its shards, as they are).  With q8 each block is quantized
// on the host as it is read, as MfDit's are (dit_rows.hpp): W8A8 where the GPU's library takes int8 products
// (prefill::Gemm::i8_usable), else Q8_0; without q8 the bf16 bytes go to the GPU and are rewritten there to FP16 in
// place.  The products take FP16 activations, each row scaled into FP16's range.  The blocks go to the GPU through a
// LayerStream: those that fit stay, the rest are streamed.  The parts outside the blocks (the embedders, the time's
// layers, the shared modulation, the head) are FP16 on the GPU.
//
// Guidance: up to two contexts (the prompt and the negative prompt) run through each product together, a context's
// rows after the other's; a context may skip blocks (the reference pipeline's negative branch skips block 22).
// Skip-layer guidance runs the prompt a third time with its blocks skipped (sample).
//
// The text embedder's weights (0.4 GiB) are needed only by set_context: drop_text gives their VRAM back, and the next
// set_context reads them again from the file.
#pragma once

#include "strata/image/mf_dit.hpp"
#include "strata/image/qi_stream.hpp"
#include "strata/prefill/gemm.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace strata::image {

class MossDit {
public:
    MossDit();
    ~MossDit();
    MossDit(const MossDit&) = delete;
    MossDit& operator=(const MossDit&) = delete;

    /// `path`: the transformer's folder (its *.safetensors) or one file.  `q8`: the blocks in 8 bits (the default),
    /// else FP16 from the file's bf16.  `dry`: read for its sizes only.
    bool open(const std::string& path, void* stream, std::string& err, bool dry = false, bool q8 = true);

    /// Context i (0: the prompt, 1: the negative prompt) from the text encoder's states [rows][4096] (host FP32, every
    /// row of the prompt): through the text embedder, kept on the GPU.
    void set_context(int i, const float* states, int64_t rows);
    int64_t context_rows(int i) const { return ctx_[i].rows; }
    /// The blocks context i skips (none by default).
    void set_skip(int i, std::vector<int> blocks) { ctx_[i].skip = std::move(blocks); }

    /// The sampler's guidance (the reference pipeline's arguments).  cfg: with context 1, u + cfg (c - u);
    /// uncond_skip: the blocks context 1 skips; slg (with slg_scale above 0): a third pass of context 0 with those
    /// blocks skipped, slg_scale (c - k) added.  Both skips act in the steps whose share of the schedule is in
    /// [start, stop).
    struct Guide {
        float cfg = 1.f;
        std::vector<int> uncond_skip, slg;
        float slg_scale = 0.f, start = 0.f, stop = 1.f;
    };
    /// Euler steps over sigmas (n steps and the terminal 0) from x [h w][128] (device FP32, in place), each at the
    /// timestep sigma times 1000.  first, total: the schedule's steps before sigmas' first and in all (the shares the
    /// guide's window is in).  step: the progress, false to stop (QiStopped); after: at each step's next sigma (a mask's
    /// blend).
    void sample(float* x, int64_t h, int64_t w, const std::vector<float>& sigmas, const Guide& g, int first, int total,
                const std::function<bool(int, int)>& step = {}, const std::function<void(float*, float)>& after = {});

    /// The velocity [n][h w][128] (device FP32) of the latent tokens x [h w][128] (device FP32) at `timestep` (sigma
    /// times 1000, as the reference pipeline gives it), for the contexts 0 .. n - 1 (n 1 or 2).
    void forward(const float* x, float timestep, int64_t h, int64_t w, int n, float* out);

    int64_t hidden() const { return C_; }
    int64_t latent_channels() const { return in_ch_; }
    int64_t text_dim() const { return txt_dim_; }
    int blocks() const { return (int) blocks_.size(); }
    bool q8() const { return q8_; }
    bool w8a8() const { return w8_; }
    /// As MfDit's: the time of `rows` rows through the MLP's first product in FP16 and in W8A8 (0: not taken).
    std::pair<double, double> time_forms(int64_t rows) const;
    uint64_t block_bytes() const { return blocks_stream_.total(); }
    uint64_t block_min() const { return blocks_stream_.min_budget(); }
    void set_block_budget(uint64_t bytes) { blocks_stream_.configure(bytes, s_); }
    int resident_blocks() const { return blocks_stream_.resident(); }
    uint64_t blocks_vram() const { return blocks_stream_.vram(); }
    void drop_blocks() { blocks_stream_.drop(); }
    /// Frees the streamed blocks' copies in RAM (LayerStream's homes).
    void drop_block_homes() { blocks_stream_.drop_homes(); }
    uint64_t weight_bytes() const { return weight_bytes_; }   // what stays on the GPU besides the blocks and the text's
    uint64_t text_bytes() const { return text_bytes_; }       // the text embedder's weights
    bool text_loaded() const { return text_loaded_; }
    void drop_text();
    uint64_t scratch_bytes() const { return (uint64_t) scratch_ * 2; }   // the products' dequantization scratch
    /// The working buffers n contexts of hw image rows take (text rows up to max_text a context), with the MLP a band
    /// of mlp_rows rows at a time (0: all).
    uint64_t activation_bytes(int64_t hw, int n, int64_t max_text, int64_t mlp_rows = 0) const;
    void set_mlp_rows(int64_t rows) { mlp_cap_ = rows; }
    /// What the working buffers take now
    uint64_t activations_held() const {
        return cap_hw_ > 0 ? activation_bytes(cap_hw_, (int) cap_n_, cap_text_, mlp_rows_) : 0;
    }
    void drop_activations();

    /// The rows of context 0 after block 0 and the text embedder's rows of context 0 (device FP32 [h w + rows][C],
    /// [rows][C]), for the probe (kept when keep_block0(true)).
    void keep_block0(bool on) { keep0_ = on; }
    const float* block0() const { return b0_; }
    const float* text_rows(int i) const { return ctx_[i].txt0; }

private:
    struct W {
        int type = 0;            // 8 Q8_0, 1 F16, -8 W8A8's int8 rows and their scales
        int64_t n = 0, k = 0;    // n rows of k
        const void* d = nullptr; // a block's: its offset in the block's region; else a device pointer
        uint64_t row_bytes = 0;
    };
    struct Block {
        W q, k, v, o, up_gate, down;
        uint64_t n1 = 0, n2 = 0, nq = 0, nk = 0;   // the FP32 norms' offsets
    };
    struct TextBlock {
        W q, k, v, o, up_gate, down;               // up_gate: ffn_up's rows, then ffn_gate's
        const float *bq = nullptr, *bk = nullptr, *bv = nullptr, *bo = nullptr, *bug = nullptr, *bdown = nullptr;
        const float *n1 = nullptr, *n2 = nullptr, *nq = nullptr, *nk = nullptr;
    };
    struct Ctx {
        int64_t rows = 0;
        float* txt0 = nullptr;   // [rows][C]: the text embedder's output
        std::vector<int> skip;
        // the RoPE's tables of the context's rows, [h w + rows][64] (for the size they were made for)
        float *cos = nullptr, *sin = nullptr;
        int64_t rope_h = -1, rope_w = -1, rope_rows = -1;
    };
    W dense(const std::vector<std::string>& names, bool biased, const float** bias, std::vector<void*>& owned,
            uint64_t& bytes);
    const float* vec(const std::string& name, std::vector<void*>& owned, uint64_t& bytes);
    void load_text();
    MfOut product(const uint16_t* x16, const float* rs, const W& w, const uint8_t* base, float* y, int64_t T,
                  const float* bias, bool same_input = false, bool defer = false);
    void ensure(int64_t hw, int n, int64_t max_text);
    void rope_tables(Ctx& c, int64_t h, int64_t w);
    void time_embedding(float timestep);
    // a block over x's rows, its segments' (a context each, its attention among its own rows) one after the other:
    // w q, k, v, o, up_gate, down; bias (null: none) theirs; mod null (the text embedder's blocks): no time
    void run_block(const W* const* w, const float* const* bias, const float* n1, const float* n2, const float* nq,
                   const float* nk, const uint8_t* base, const float* mod, float* x, const std::vector<int64_t>& seg,
                   const std::vector<const float*>& cos, const std::vector<const float*>& sin);

    struct Shards;
    std::unique_ptr<Shards> files_;
    std::vector<std::vector<uint8_t>> host_;   // the blocks' quantized weights and FP32 vectors, when not in place
    void* s_ = nullptr;
    prefill::Gemm gemm_;
    std::vector<void*> allocs_, text_allocs_;
    uint64_t weight_bytes_ = 0, text_bytes_ = 0;
    int64_t scratch_ = 0;
    bool dry_ = false, q8_ = true, w8_ = false, keep0_ = false, text_loaded_ = false;
    int64_t C_ = 3072, F_ = 8192, heads_ = 24, hd_ = 128, in_ch_ = 128, txt_dim_ = 4096, freq_ = 256;
    double rope_base_ = 10000.0;
    // the parts outside the blocks, FP16 on the GPU
    W vin_, tproj_, t1_, t2_, smod_, hmod_, fout_;
    const float *b_vin_ = nullptr, *b_tproj_ = nullptr, *b_t1_ = nullptr, *b_t2_ = nullptr, *b_smod_ = nullptr,
                *b_hmod_ = nullptr, *fnorm_ = nullptr;
    std::vector<TextBlock> tblocks_;
    std::vector<Block> blocks_;
    LayerStream blocks_stream_;
    Ctx ctx_[2];
    // activations, grown on demand
    int64_t cap_hw_ = 0, cap_n_ = 0, cap_text_ = 0, mlp_cap_ = 0, mlp_rows_ = 0;
    int64_t dirty_[2] = {0, 0};   // the rows of each attention slot (jq16_ ...) that may not be zero
    float *x_ = nullptr, *q_ = nullptr, *k_ = nullptr, *v_ = nullptr, *a_ = nullptr, *gu_ = nullptr,
          *temb_ = nullptr, *mod_ = nullptr, *head_ = nullptr;
    // the FP16 inputs' row scales (rs_ the current product's, rs2_ the MLP's second's while rs_ is still read, rs_o_
    // the attention's output's), the v heads' scales (hs_, a context's heads) and their rows' maxima
    float *rs_ = nullptr, *rs2_ = nullptr, *rs_o_ = nullptr, *hs_ = nullptr, *vpart_ = nullptr;
    uint16_t *h16_ = nullptr, *jq16_ = nullptr, *jk16_ = nullptr, *jv16_ = nullptr, *jo16_ = nullptr,
             *o16_ = nullptr, *t16_ = nullptr;
    float* attn_scratch_ = nullptr;
    int64_t jrows_ = 0;   // the rows of jq16_ .. jv16_ a context (qi_attn_rows of its rows at most)
    // W8A8's input (w8_): xq8_rows_ rows of up to xq8_cols_ int8 values, then two arrays of their scales, taken in turn
    int8_t* xq8_ = nullptr;
    int64_t xq8_rows_ = 0, xq8_cols_ = 0;
    int xs_next_ = 0;
    float* b0_ = nullptr;
};

}  // namespace strata::image
