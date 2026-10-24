// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/mf_dit.hpp - Mage-Flow's DiT on the GPU: a two-stream MMDiT (12 blocks of 3072, 24 heads of
// 128) over the image's latent tokens [h w][128] and the text encoder's states, the velocity of a flow-matching step.
//
// Each block modulates the image and the text stream by the time (its own six shift / scale / gate rows for each),
// projects them to q, k and v with each stream's own weights, runs one attention over [text, image] of a context (no
// mask: the text rows see the image and the other way round; the image rows rotated by the 2D multi-scale RoPE, the
// text rows not), and adds each stream's projection and MLP (GELU, tanh form) back through its gates.  The text stream
// changes with the time, so nothing of it is kept between steps beyond its input projection.
//
// The weights come from the bf16 safetensors (Comfy-Org/Mage-Flow's files as they are).  A file with a shared low-rank
// modulation (MageTrail's, bluvoll's mageflow-lowrank-modulation-v1) projects SiLU(temb) once a step to a few hundred
// values (modulation_down) and each block's modulations from those: its FP32 factors stay FP32, as the format asks.
// With q8 each block is quantized to 8 bits on the host as it is read, once, and the file's pages are dropped from the
// page cache: half the VRAM and the bytes a streamed block moves.  Where the GPU's library takes int8 products (prefill::Gemm::i8_usable) the
// weights are rotated in groups and quantized with a scale a row, and so are the activations a token at each product
// (W8A8, qi_rows_i8: ComfyUI's int8 form, the rotation spreading the outlier channels below); elsewhere as Q8_0, the
// products dequantizing it to FP16 each time.  Without q8 the bf16 bytes
// go from the mapped file to the GPU as they are and are rewritten there to FP16 in place (the same two bytes, the few
// values past FP16's range saturated), which the products take as they are: no dequantization, the matrix units fed
// straight (MfPipeline takes this when the blocks fit in VRAM).  The products take FP16 activations (rows scaled into
// FP16's range, mf_dit.cpp): not q8_mma's int8 activations, which the image
// stream's outlier channels (1e5 among values of 1) cost too much (RTX 4070, Turbo's 4 steps from the same noise: the
// final latent's cosine to the reference 0.955 with them, 0.992 with FP16 on the B70, 0.996 in bf16).  The blocks go to
// the GPU through a LayerStream: those that fit stay, the rest are streamed.
//
// Guidance: up to two contexts (the prompt and the negative prompt) run through each product together, the image's
// rows twice, the weights read once a step; the attention one context at a time.
//
// Edits (the edit models): the references' latent tokens (clean, the same every step) follow the target's rows in
// each context, [target, reference 1, ..., reference N], and take part in every block; each image's rows are rotated
// by its own size's centred positions and by its frame (0 the target, j reference j), and only the target's rows
// give the velocity.
#pragma once

#include "strata/image/qi_stream.hpp"
#include "strata/prefill/gemm.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace strata::image {

/// A product's output [T][C] as its readers take it: FP32 sums (f) times the rows' scales (rs, null: 1), or a W8A8
/// product's int32 sums (i) times the activations' scales (sx, a row) and the weights' (sw, a column); plus the bias
/// (b, null: none).
struct MfOut {
    const float* f = nullptr;
    const int32_t* i = nullptr;
    const float *rs = nullptr, *sx = nullptr, *sw = nullptr, *b = nullptr;
    int64_t C = 0;
    float at(int64_t t, int64_t c) const {
        const float v = i != nullptr ? (float) i[t * C + c] * sx[t] * sw[c]
                                     : (rs != nullptr ? f[t * C + c] * rs[t] : f[t * C + c]);
        return b != nullptr ? v + b[c] : v;
    }
    /// the rows from r on
    MfOut from(int64_t r) const {
        MfOut o = *this;
        if (f != nullptr) o.f += r * C;
        if (i != nullptr) o.i += r * C;
        if (rs != nullptr) o.rs += r;
        if (sx != nullptr) o.sx += r;
        return o;
    }
};

class MappedFile;

class MfDit {
public:
    MfDit();
    ~MfDit();
    MfDit(const MfDit&) = delete;
    MfDit& operator=(const MfDit&) = delete;

    /// `q8`: the blocks as Q8_0 (the default), else FP16 on the GPU from the file's bf16.  `dry`: read for its sizes
    /// only.
    bool open(const std::string& path, void* stream, std::string& err, bool dry = false, bool q8 = true);

    /// Context i (0: the prompt, 1: the negative prompt) from the text encoder's states [rows][2560] (host FP32, the
    /// rows after the system turn): their norm and input projection, kept on the GPU.
    void set_context(int i, const float* states, int64_t rows);
    int64_t context_rows(int i) const { return ctx_[i].rows; }

    /// An edit's references: their latent tokens z (device FP32, [sum of h w][128], reference by reference) and their
    /// sizes in latent positions (h, w); none (the default) for a generation.
    void set_refs(const float* z, const std::vector<std::pair<int64_t, int64_t>>& sizes);
    /// The rows the references add to each context's image rows.
    int64_t ref_rows() const { return ref_rows_; }

    /// The velocity [n][h w][128] (device FP32) of the latent tokens x [h w][128] (device FP32) at noise level sigma,
    /// for the contexts 0 .. n - 1 (n 1 or 2).
    void forward(const float* x, float sigma, int64_t h, int64_t w, int n, float* out);

    /// Sampling: x [h w][128] (device FP32, the initial noise) stepped through sigmas (descending, the last 0) by
    /// Euler; with cfg > 1 and a negative context, the velocity neg + cfg (pos - neg).  step(i, n) after each step:
    /// false stops (QiStopped).
    void sample(float* x, int64_t h, int64_t w, const std::vector<float>& sigmas, float cfg,
                const std::function<bool(int, int)>& step = {});

    int64_t hidden() const { return C_; }
    int64_t latent_channels() const { return in_ch_; }
    bool q8() const { return q8_; }
    int64_t text_dim() const { return txt_dim_; }
    uint64_t block_bytes() const { return blocks_stream_.total(); }
    uint64_t block_min() const { return blocks_stream_.min_budget(); }
    void set_block_budget(uint64_t bytes) { blocks_stream_.configure(bytes, s_); }
    int resident_blocks() const { return blocks_stream_.resident(); }
    uint64_t blocks_vram() const { return blocks_stream_.vram(); }
    void drop_blocks() { blocks_stream_.drop(); }
    uint64_t weight_bytes() const { return weight_bytes_; }   // what stays on the GPU besides the blocks
    /// The working buffers n contexts of hw image rows (an edit's references' included) take (text rows up to
    /// max_text a context), with the MLP a band of mlp_rows image rows at a time (0: all).
    uint64_t activation_bytes(int64_t hw, int n, int64_t max_text, int64_t mlp_rows = 0) const;
    void set_mlp_rows(int64_t rows) { mlp_cap_ = rows; }
    uint64_t activations_held() const {
        return cap_hw_ > 0 ? activation_bytes(cap_hw_, (int) cap_n_, cap_text_, mlp_rows_) : 0;
    }
    void drop_activations();

    /// The image and text streams after block 0 of the last forward's context 0 (device FP32 [h w][C], [rows][C]), for
    /// the probe (kept when keep_block0(true)).
    void keep_block0(bool on) { keep0_ = on; }
    const float* block0_img() const { return b0_img_; }
    const float* block0_txt() const { return b0_txt_; }

private:
    struct W {
        int type = 0;            // 8 Q8_0, 1 F16, 32 F32 (the low-rank modulations)
        int64_t n = 0, k = 0;    // n rows of k
        const void* d = nullptr; // a block's: its offset in the block's region; else a device pointer
        uint64_t row_bytes = 0;
    };
    struct Block {
        W imod, tmod, q, k, v, tq, tk, tv, o, to, imlp1, imlp2, tmlp1, tmlp2;
        uint64_t b_imod = 0, b_tmod = 0, b_q = 0, b_k = 0, b_v = 0, b_tq = 0, b_tk = 0, b_tv = 0, b_o = 0, b_to = 0,
                 b_imlp1 = 0, b_imlp2 = 0, b_tmlp1 = 0, b_tmlp2 = 0;   // the FP32 biases' offsets
        uint64_t nq = 0, nk = 0, ntq = 0, ntk = 0;                     // the FP32 q / k norms' offsets
    };
    struct Ctx {
        int64_t rows = 0;
        float* txt0 = nullptr;   // [rows][C]: txt_in(txt_norm(states))
    };
    // y [T][w.n] = (x16 [T][w.k], rows scaled down by rs) . w^T, the rows times rs again, plus bias (rs, bias null:
    // none).  same_input: x16 is the previous product's, which was W8A8 too (its int8 input is taken again).  defer:
    // y is left the product's sums, for the reader to scale and add the bias to as the view says.
    MfOut product(const uint16_t* x16, const float* rs, const W& w, const uint8_t* base, float* y, int64_t T,
                  const float* bias, bool same_input = false, bool defer = false);
    // a block's modulation y [w.n] = SiLU(temb) . w^T + bias (tsil: its FP16 copy, for a 16-bit w); an F32 w (the
    // low-rank form) reads the shared projection mlow_ instead
    void modulation(const uint16_t* tsil, const W& w, const uint8_t* base, float* y, const float* bias);
    void ensure(int64_t hw, int n, int64_t max_text);
    void rope_tables(int64_t h, int64_t w);
    void drop_refs();
    void time_embedding(float sigma);

    std::unique_ptr<MappedFile> file_;
    std::vector<std::vector<uint8_t>> host_;   // the blocks' Q8_0 (and every block's FP32 vectors), when not in place
    void* s_ = nullptr;
    prefill::Gemm gemm_;
    std::vector<void*> allocs_;
    uint64_t weight_bytes_ = 0;
    bool dry_ = false, q8_ = true, w8_ = false, keep0_ = false;
    int64_t C_ = 3072, F_ = 12288, heads_ = 24, hd_ = 128, in_ch_ = 128, txt_dim_ = 2560;
    std::vector<int64_t> axes_{16, 56, 56};
    // the parts outside the blocks, FP16 on the GPU
    W img_in_, txt_in_, t1_, t2_, nout_, pout_;
    W mdown_;   // the shared low-rank modulation's projection (F32, device; n 0 without one)
    float *b_mdown_ = nullptr, *mlow_ = nullptr;
    const float *b_img_in_ = nullptr, *b_txt_in_ = nullptr, *b_t1_ = nullptr, *b_t2_ = nullptr, *b_nout_ = nullptr,
                *b_pout_ = nullptr, *txt_norm_ = nullptr;
    std::vector<Block> blocks_;
    LayerStream blocks_stream_;
    Ctx ctx_[2];
    // activations, grown on demand
    int64_t cap_hw_ = 0, cap_n_ = 0, cap_text_ = 0, mlp_cap_ = 0, mlp_rows_ = 0, dirty_ = 0;
    float *img_ = nullptr, *txt_ = nullptr, *q_ = nullptr, *k_ = nullptr, *v_ = nullptr, *a_ = nullptr,
          *gu_ = nullptr, *mod_i_ = nullptr, *mod_t_ = nullptr, *temb_ = nullptr, *nmod_ = nullptr;
    float *tq_ = nullptr, *tk_ = nullptr, *tv_ = nullptr;   // the text rows' q, k, v
    // the FP16 inputs' row scales (rs_ the current product's, rs2_ the text rows' q, k, v's and the MLP's second's
    // while rs_ is still read, rs_io_ / rs_to_ the attention's outputs by stream), the
    // attention's v heads' scales (hs_, a context's heads) and their rows' maxima
    float *rs_ = nullptr, *rs2_ = nullptr, *rs_io_ = nullptr, *rs_to_ = nullptr, *hs_ = nullptr, *vpart_ = nullptr;
    uint16_t *h16_ = nullptr, *jq16_ = nullptr, *jk16_ = nullptr, *jv16_ = nullptr, *jo16_ = nullptr,
             *io16_ = nullptr, *to16_ = nullptr, *t16_ = nullptr;
    float *cos_ = nullptr, *sin_ = nullptr, *cos_id_ = nullptr, *sin_id_ = nullptr, *attn_scratch_ = nullptr;
    // W8A8's input (w8_): xq8_rows_ rows of up to xq8_cols_ int8 values, then two arrays of their scales, taken in
    // turn (xs_next_), so that the deferred sums of one input keep theirs while the next input is quantized
    int8_t* xq8_ = nullptr;
    int64_t xq8_rows_ = 0, xq8_cols_ = 0;
    int xs_next_ = 0;
    float *b0_img_ = nullptr, *b0_txt_ = nullptr;
    int64_t rope_h_ = -1, rope_w_ = -1, rope_refs_ = -1;
    // an edit's references
    float* refs_ = nullptr;   // [ref_rows_][in_ch_]
    int64_t ref_rows_ = 0, refs_gen_ = 0;
    std::vector<std::pair<int64_t, int64_t>> ref_sizes_;
    float* xin_ = nullptr;    // the image rows' input [hw + ref_rows_][in_ch_]
};

}  // namespace strata::image
