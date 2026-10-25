// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/hunyuanimage3_vae.hpp - HunyuanImage 3.0's VAE decoder on the GPU: the DiT's latent
// [32][h][w] to an RGB image [3][16 h][16 w] (the reference's decode of one image: T = 1 in, the LAST of the
// four output frames out, about -1 .. 1).
//
// The decoder is a DCAE 3D VAE, not an AutoencoderKL one (Tencent's AutoencoderKLConv3D): no quant_conv or
// post_quant_conv, no nin_shortcut, and no nearest-interpolation upsampling.  The graph
// (records/2026-10-23-hunyuanimage3-plan/VAE.md, checked against the file's 280 tensors and stage by stage
// against the record's hy3_vae_ref.py at rel <= 3.2e-6):
//
//   z / scaling_factor -> conv_in (3 x 3 x 3) + repeat_interleave(32) shortcut -> mid block_1, attn_1, block_2
//   -> five up levels of three ResnetBlocks each, with an UpsampleDCAE after the first four -> norm_out ->
//   swish -> conv_out.  A ResnetBlock is GroupNorm(32, eps 1e-6) -> swish -> conv -> GroupNorm -> swish -> conv
//   -> residual, the channels equal in and out (the DCAE form puts every channel change in the up samples).
//   An UpsampleDCAE is a conv to factor * C channels, then einops "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2)
//   (w r3)" (r1 = 2 at the two temporal levels up.0 and up.1, r1 = 1 at the two spatial ones), plus a shortcut
//   with no weights of its own: the input repeat_interleave'd and split the same way.  Time: 1 -> 2 -> 4 -> 4;
//   space: x2 four times, /16 in all.  norm_out and conv_out take 128 channels.
//
// The arithmetic follows the reference's autocast(fp16) as VAE.md measured it on this machine: the
// convolutions read FP16 and their products accumulate FP32 (prefill::Gemm's f16 path, the one the DiT
// uses); the residual stream, the group norms and the attention's softmax are FP32.  The 3 x 3 x 3
// convolutions unfold with the time axis explicit (zero padding on all three sides); a tensor with T = 1 reads
// only its kernels' CENTER time tap - the other taps multiply its zero padding exactly (VAE.md: zeroing them
// moves the output by 0.0), so open() keeps 1 / 3 of the weights of the twelve T = 1 convolutions, 340 M
// numbers of them no image ever reads.  The activations are channels-last ([pixel][channel]) like the other
// VAEs', the pixels the linear (t, y, x) index.
//
// The ENCODER (an edit's reference image into its latent, the reference's `vae_encode`) is the same graph the
// other way (VAE.md's section 4, held to the reference at rel 3.1e-6 on the moments): the image replicated into
// four frames -> conv_in (3 -> 128) -> five down levels of two ResnetBlocks each with a DownsampleDCAE after the
// first four (space /2 each; time /2 at the third and fourth, 4 -> 1) -> mid block_1, attn_1, block_2 -> a
// shortcut that averages the 1024 channels in 16s to 64 -> norm_out, swish, conv_out (1024 -> 64) + the shortcut
// = the moments, whose first 32 channels are the posterior's mean.  A DownsampleDCAE is a conv to out / factor
// channels, then "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", plus the input folded the same way and
// averaged in groups of factor * in / out channels.  encode() samples the posterior with the caller's noise, as the
// reference does with its generator, or gives its mean (the golden's deterministic choice) without; both times
// scaling_factor.
#pragma once

#include "strata/prefill/gemm.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::image {

class HunyuanImage3Vae {
public:
    HunyuanImage3Vae();
    ~HunyuanImage3Vae();
    HunyuanImage3Vae(const HunyuanImage3Vae&) = delete;
    HunyuanImage3Vae& operator=(const HunyuanImage3Vae&) = delete;

    /// Reads the VAE's safetensors (the data folder's hy3/vae/hunyuan_image_3_vae_fp16.safetensors: 280 tensors, all F16, the
    /// reference's decoder.* names) and puts the decoder's weights on the current GPU.  `dry`: sizes only,
    /// nothing uploaded; such a part cannot run.
    bool open(const std::string& path, void* stream, std::string& err, bool dry = false, bool encoder = false);
    /// Whether open() read the encoder's weights too (encode() needs them).
    bool has_encoder() const { return enc_open_; }

    /// rgb [3][16 h][16 w] (device FP32, about -1 .. 1 - the reference's decoder's own output, the last frame)
    /// from the DiT's latent z [32][h][w] (device FP32, model space: the decode divides by scaling_factor
    /// first, where the reference's pipeline does).  `dump` (empty, or a directory): the probe's stage-by-stage
    /// look - every stage boundary is also written to <dump>/NN_name.bin as FP32 [pixels][channels].
    void decode(const float* z, int64_t h, int64_t w, float* rgb, const std::string& dump = std::string());

    /// z [32][H/16][W/16] (device FP32, model space: a sample of the posterior times scaling_factor, as the DiT's
    /// latent is) from an image rgb [3][H][W] (device FP32, -1 .. 1; H and W multiples of 16): the reference's
    /// `vae_encode`.  `eps`: the sample's standard normal noise, device FP32 [32][H/16][W/16]; null takes the mean.
    /// Needs open(..., encoder = true).
    void encode(const float* rgb, int64_t H, int64_t W, float* z, const float* eps = nullptr,
                const std::string& dump = std::string());
    /// The GPU memory an encode of an H x W image works in (the buffers, besides the weights).
    uint64_t encode_scratch_bytes(int64_t H, int64_t W) const;

    /// The model's vae scaling_factor: the data folder's hy3/config.json `vae.scaling_factor`, what the decode divides by.
    static double scaling_factor() { return kScalingFactor; }

    /// The GPU memory a decode of an h x w latent works in (the buffers, besides the weights).
    uint64_t scratch_bytes(int64_t h, int64_t w) const;
    /// Frees the working buffers (grown again on use).
    void drop_scratch();
    uint64_t scratch_held() const { return held_bytes_; }
    uint64_t weight_bytes() const { return weight_bytes_; }

private:
    struct Conv {
        int64_t out = 0, in = 0, kt = 0;   // kt: 0 a 1x1x1 (no unfold), 1 the center time tap, 3 all three
        const uint16_t* w = nullptr;        // [out][kt * 9][in] (kt 0: [out][in]) FP16
        const float* b = nullptr;           // FP32
    };
    struct Res {                            // no shortcut: the channels are equal (the file holds none)
        const float *g1 = nullptr, *b1 = nullptr, *g2 = nullptr, *b2 = nullptr;
        Conv c1, c2;
    };
    struct Up {                             // an up level: three blocks, then a DCAE upsample (up.4: none)
        Res r[3];
        Conv up;                            // up.out 0: no upsample
        int temporal = 0;                   // 1: r1 = 2 (the time doubles too), 0: r1 = 1
    };
    struct Down {                           // a down level: two blocks, then a DCAE downsample (down.4: none)
        Res r[2];
        Conv down;                          // down.out 0: no downsample; else its conv's out = level out / factor
        int64_t out = 0;                    // the level's output channels after the downsample
        int temporal = 0;                   // 1: r1 = 2 (the time halves too), 0: r1 = 1
    };

    Conv conv_w(class SafeTensors& st, const std::string& name, int64_t kt, int64_t pad_in = 0);
    Conv point_w(class SafeTensors& st, const std::string& name);
    const float* vec(class SafeTensors& st, const std::string& name);
    Res res_w(class SafeTensors& st, const std::string& p, int64_t kt);
    void grow(int64_t elems32, int64_t elems16, int64_t col16, int64_t scores);
    void dump_stage(int index, const std::string& name, const float* p, int64_t n);
    void dump_stage16(const std::string& name, const uint16_t* p, int64_t n);
    void conv(const Conv& c, const uint16_t* in16, int64_t T, int64_t H, int64_t W, float* out);
    void groupnorm(const float* x, const float* g, const float* b, int64_t P, int64_t C, bool silu,
                   uint16_t* y16, const std::string& tag = std::string());
    void resblock(const Res& r, const float* x, int64_t T, int64_t H, int64_t W, int64_t C, float* y,
                  const std::string& tag = std::string());
    void attention(float* x, int64_t P, int64_t C, const std::string& tag = std::string());
    void upsample(const Up& u, const float* x, int64_t T, int64_t H, int64_t W, int64_t C, float* y);
    void downsample(const Down& d, const float* x, int64_t T, int64_t H, int64_t W, int64_t C, float* y);
    void attention_w(const float* g, const float* b, const Conv& q, const Conv& k, const Conv& v, const Conv& proj,
                     float* x, int64_t P, int64_t C, const std::string& tag);
    void dump_stage(const std::string& dir, int index, const std::string& name, const float* p, int64_t n);

    void* s_ = nullptr;
    std::string dump_;   // decode()'s stage-dump directory (empty: no dumps)
    prefill::Gemm gemm_;
    std::vector<void*> allocs_;
    uint64_t weight_bytes_ = 0, held_bytes_ = 0;
    bool dry_ = false;
    Conv cin_, cout_;
    Res mid0_, mid2_;
    const float *ag_ = nullptr, *ab_ = nullptr;
    Conv aq_, ak_, av_, aproj_;
    Up up_[5];
    const float *gout_ = nullptr, *bout_ = nullptr;
    // the encoder (open(..., encoder = true))
    bool enc_open_ = false;
    Conv ecin_, ecout_;
    Down down_[5];
    Res emid0_, emid2_;
    const float *eag_ = nullptr, *eab_ = nullptr;
    Conv eaq_, eak_, eav_, eaproj_;
    const float *egout_ = nullptr, *ebout_ = nullptr;
    // scratch, grown on demand
    int64_t cap32_ = 0, cap16_ = 0, col_cap_ = 0, score_cap_ = 0;
    float *t0_ = nullptr, *t1_ = nullptr, *tmp32_ = nullptr, *scores_ = nullptr, *gn_ws_ = nullptr;
    uint16_t *a16_ = nullptr, *col16_ = nullptr;

    static constexpr double kScalingFactor = 0.562679178327931;   // the data folder's hy3/config.json vae.scaling_factor
};

}  // namespace strata::image
