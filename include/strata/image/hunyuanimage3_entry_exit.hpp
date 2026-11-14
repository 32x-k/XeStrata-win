// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/hunyuanimage3_entry_exit.hpp - HunyuanImage 3.0's pieces around the decoder layers, on the card:
// `patch_embed` (the latent into the image block's rows) and `final_layer` (those rows into the velocity).
//
// The two are convolutions at the token grid: a 3x3 one from the latent's 32 channels to 1024, a ResBlock
// (GroupNorm(32) + swish + 3x3 convolution, the timestep's embedding as a per-channel scale and shift, a 1x1 skip)
// from 1024 to 4096, and at the exit the same block back to 1024, a GroupNorm, a swish and a 3x3 convolution to the
// 32 channels of the velocity.  The products are `prefill::Gemm`'s FP16 ones over an unfolded 3x3 neighbourhood
// (the bands the VAE's convolutions unfold), each operand as two FP16 halves (the value and the rounding it left),
// the norms and the residual stream FP32.  `probe around` holds them to
// the host FP32 reference, and the shipped path to the golden's picture (`probe pipeline`).
//
// What is NOT here is what costs a step little: `wte` gathers a few hundred rows out of a BF16 table (the
// pipeline's, once a request) and the five TimestepEmbedders are 4096-wide products over one row each, which the
// host does in the same FP32 the golden was built with.  The convolutions are the step's seconds and they are
// this file's.
//
// The shapes are read from the file and only their relationships are assumed (the exit convolves what the entry
// made, 32 channels at both ends); the memory a grid takes comes from its token count.  A part opened `dry` reads
// the header for the sizes a budget is planned with and puts nothing on the GPU.
#pragma once

#include "strata/image/hunyuanimage3_file.hpp"
#include "strata/prefill/gemm.hpp"

#include <vector>

namespace strata::image {

class HunyuanImage3EntryExit {
public:
    HunyuanImage3EntryExit();
    ~HunyuanImage3EntryExit();
    HunyuanImage3EntryExit(const HunyuanImage3EntryExit&) = delete;
    HunyuanImage3EntryExit& operator=(const HunyuanImage3EntryExit&) = delete;

    /// The file's `patch_embed` and `final_layer` as the two paths' weights: the convolutions, the skips and the
    /// embedding projections of the file's BF16 as FP16 on the device, the norms and the biases as FP32.  `dry`:
    /// their sizes only, nothing on the GPU (a part opened so cannot run).
    bool open(const std::string& path, void* stream, std::string& err, bool dry = false);

    /// The model's hidden width: what `patch_embed` writes and `final_layer` reads (4096 in the files).
    int64_t hidden() const { return hidden_; }
    /// The latent's channels: the entry's first convolution's input and the exit's last output (32).
    int64_t latent_channels() const { return latent_; }
    /// The VRAM the weights take: what a budget has beside the decoder's and the VAE's.
    uint64_t weight_bytes() const { return weight_bytes_; }
    /// What the part holds on the GPU now: the weights and, with a grid set, its buffers.
    uint64_t held() const;

    /// What a `th` x `tw` token grid's buffers take (device): the FP16 operands, the unfold's bands, the FP32
    /// products and the norms' statistics.  Zero for a grid with no tokens.
    uint64_t scratch_bytes(int64_t th, int64_t tw) const;
    /// The grid the two paths run at: allocates its buffers (freeing the last grid's).  The pipeline sets it for
    /// the image's token grid and drops it before the VAE decodes, whose buffers it would otherwise take space from.
    bool set_grid(int64_t th, int64_t tw, std::string& err);
    void drop_grid();
    int64_t grid_tokens() const { return n_; }

    /// `latent`: the sampler's host FP32 [32][th*tw] (channels first, as the reference's tensor is), `t`: the
    /// step's timestep, `rows`: the host buffer of `grid_tokens() * hidden()` floats that `final_layer` and the
    /// decoder layers take.  False with `err`.
    bool patch_embed(const float* latent, int64_t th, int64_t tw, float t, float* rows, std::string& err);
    /// `rows`: the block's host FP32 [th*tw][hidden] as the layers left it, `vel`: the host buffer of
    /// `32 * grid_tokens()` floats, [32][th*tw] as the sampler's latent is.  False with `err`.
    bool final_layer(const float* rows, int64_t th, int64_t tw, float t, float* vel, std::string& err);

private:
    /// A convolution as its product's weight: W[out][taps * in] FP16 (the 3x3's taps in (dy, dx) order, matching
    /// hunyuanimage3_im2col16; a 1x1 has one tap) and its bias FP32.
    struct Conv {
        const uint16_t* w = nullptr;
        const float* b = nullptr;
        int64_t in = 0, out = 0, taps = 1;
    };
    /// A ResBlock: three convolutions, the embedding's hidden -> 2 out projection and the two norms.
    struct Res {
        Conv in, out, skip;
        const uint16_t* emb = nullptr;
        const float* emb_b = nullptr;
        int64_t emb_n = 0, emb_k = 0;
        const float* g1 = nullptr;        const float* b1 = nullptr;
        const float* g2 = nullptr;
        const float* b2 = nullptr;
    };
    struct Gn {
        const float* w = nullptr;
        const float* b = nullptr;
        int64_t c = 0;
    };

    /// A tensor of the file's BF16 as FP16 rows on the device, `kind` saying how it lies: 9 a [out][in][3][3]
    /// convolution packed as [out][tap * in + c] (the unfold's order), 1 a [out][in][1][1] one as [out][in], 0 a
    /// [out][in] matrix as it lies.  Its bias (for a convolution) and its shape come out here too.
    void load_rows(const std::string& name, int64_t kind, Conv& c, std::string& err);
    /// A BF16 vector (a norm's or a convolution's weight or bias) as FP32 on the device.
    void load_vec(const std::string& name, int64_t n, const float*& d, std::string& err);
    /// Y[P][out] = X[P][9 in] unfolded . W: the bands in row chunks the col buffer holds, then the bias.
    void conv3x3(const Conv& c, const uint16_t* in16, int64_t th, int64_t tw, float* out, const float* bias);
    void conv1x1(const Conv& c, const uint16_t* in16, int64_t P, float* out, const float* bias);
    /// The GroupNorm(32) + swish (+ a ResBlock's embedding scale and shift) as the product's FP16 operand.
    void norm(const float* x, const Gn& g, const float* scale, const float* shift, int64_t P, uint16_t* y16,
              const char* tag);
    /// A ResBlock: `x` FP32 [P][in] in (with `x16` its FP16 image, the skip's operand), `mid` FP32 [P][out] for the
    /// inner product, `out` FP32 [P][out] for the outer one, `skip32` its skip's product; `a_in` and `a_out` FP16
    /// operands at the block's input's and output's widths.
    void res_block(const Res& r, const float* x, const uint16_t* x16, uint16_t* a_in, uint16_t* a_out, int64_t P,
                   int64_t th, int64_t tw, float* mid, float* out, float* skip32);
    /// The timestep's embedding, host FP32 (the golden's), as the products' one FP16 row.
    void embed16(const std::string& which, float t, std::string& err);
    /// One stage of a path, FP32 [n] as it lies on the device, into `dump_dir_` when the environment names one
    /// (`HUNYUANIMAGE3_ENTRY_DUMP`, the VAE's `HUNYUANIMAGE3_VAE_DUMP`): the record's stage walk reads these
    /// against the NumPy reference.
    void dump(const std::string& name, const float* x, int64_t n);
    /// The same for an FP16 buffer (a product's operand), widened on the host.
    void dump16(const std::string& name, const uint16_t* x, int64_t n);
    void release(std::vector<void*>& owned);
    void release_grid();

    HunyuanImage3File f_;
    void* s_ = nullptr;
    bool dry_ = false;
    std::string dump_dir_, stage_;
    prefill::Gemm gemm_;
    int64_t hidden_ = 0, latent_ = 0;
    uint64_t weight_bytes_ = 0;
    // the entry: conv3x3(32 -> 1024), then the ResBlock 1024 -> 4096
    Conv patch_in_;
    Res patch_res_;
    // the exit: the ResBlock 4096 -> 1024, then the norm and conv3x3(1024 -> 32)
    Res final_res_;
    Gn final_gn_;
    Conv final_out_;
    // the grid's buffers
    int64_t n_ = 0, th_ = 0, tw_ = 0, col_elems_ = 0;
    uint16_t *x16_ = nullptr, *a16s_ = nullptr, *a16c_ = nullptr, *skip16_ = nullptr, *rows16_ = nullptr,
             *col16_ = nullptr, *emb16_ = nullptr;
    float *f32s_[3] = {nullptr, nullptr, nullptr}, *f32c_[3] = {nullptr, nullptr, nullptr};
    float *lat32_ = nullptr, *vel32_ = nullptr, *es32_ = nullptr, *gn_ws_ = nullptr, *lo32_ = nullptr;
    std::vector<void*> weights_;
    std::vector<void*> grid_;
};

}  // namespace strata::image