// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/hunyuanimage3_refs.hpp - an edit's reference images as HunyuanImage 3.0 reads them, on the
// host: the model's resolution buckets, the two copies its image processor makes of a reference (the VAE's, resized
// and center-cropped to the bucket nearest the image's shape; SigLIP2's, resized to at most 1024 patches of 16 and
// cut into them), and the resampling both go through (PIL's, as the reference's processors do it).
//
// Nothing here is a kernel: a reference is a few megapixels once a request, and the values must be the reference's
// own (its PIL resize and crop, its naflex sizing), which the host does exactly.  The GPU parts that take these -
// hunyuanimage3_vae's encode and hunyuanimage3_vision - read what this file writes.
#pragma once

#include "strata/image/pipeline.hpp"

#include <cstdint>
#include <vector>

namespace strata::image {

/// One of the model's resolution buckets: its `<img_ratio_i>` index, the size an image is made (or a reference
/// read) at, and its height over width.
struct HunyuanImage3Bucket {
    int32_t ratio_index = 0;
    int64_t height = 0, width = 0;
    double ratio = 1.0;
};

/// The model's resolution buckets at `base_size` (its image processor's ResolutionGroup: the base square, then
/// (height + step, width - step) to the tallest and narrowest, then (height - step, width + step) to the shortest
/// and widest, every side a multiple of `align`, sorted by height / width; then the four extra ratios the processor
/// adds when they are not among them).  37 at the model's base of 1024; `step` 0 is the processor's base / 16.
/// tools/image/qimage.py's hunyuanimage3_buckets is the same table, and serve/images.py snaps a request's size with
/// it, so the ids a request carries and the grids the engine reads agree.
std::vector<HunyuanImage3Bucket> hunyuanimage3_buckets(int64_t base_size, int64_t align = 16, int64_t step = 0);

/// The bucket nearest a `width` x `height` image's shape: the least |ratio - height / width|, the first of equals
/// (the processor's argmin).
const HunyuanImage3Bucket& hunyuanimage3_nearest_bucket(const std::vector<HunyuanImage3Bucket>& table,
                                                        int64_t width, int64_t height);

/// SigLIP2's naflex sizing (transformers' get_image_size_for_max_num_patches): the size, multiples of `patch`, a
/// `width` x `height` image is resized to so that it holds at most `max_patches` patches - the largest scale the
/// bisection finds, each side at least one patch.
void hunyuanimage3_naflex_size(int64_t width, int64_t height, int64_t patch, int64_t max_patches, int64_t& out_w,
                               int64_t& out_h);

/// rgb [3][H][W] (0 .. 255) resized to [3][H2][W2] the way PIL resizes an RGB image: a horizontal pass then a
/// vertical one, each tap's weight from the filter over a support widened by the scale when shrinking, and each
/// pass rounding to 8 bits as PIL's 8-bit images do.  `lanczos`: LANCZOS (the reference's VAE copy), else BILINEAR
/// (its SigLIP2 processor's).  `round8` false keeps the float values (the positional embedding's resize).
std::vector<float> hunyuanimage3_resize(const std::vector<float>& rgb, int64_t channels, int64_t W, int64_t H,
                                        int64_t W2, int64_t H2, bool lanczos, bool round8 = true);

/// A reference's RGB [3][H][W] (0 .. 255), its alpha dropped as PIL's convert("RGB") drops it.
std::vector<float> hunyuanimage3_ref_rgb(const QiRef& f);

/// The VAE's copy of a reference: resized keeping its aspect so that it covers `tw` x `th` (PIL LANCZOS, the sides
/// Python-rounded), center-cropped to it (the reference's resize_and_crop), and normalized to -1 .. 1: [3][th][tw].
std::vector<float> hunyuanimage3_vae_copy(const QiRef& f, int64_t tw, int64_t th);

/// SigLIP2's copy of a reference: resized to its naflex size (BILINEAR), normalized to -1 .. 1 ((x / 255 - 0.5) /
/// 0.5), cut into `patch` x `patch` patches in row order with (dy, dx, channel) inside each ([ph * pw][patch *
/// patch * 3]), and zero-padded to `max_patches` rows.  `ph`, `pw`: the real grid.
std::vector<float> hunyuanimage3_vit_patches(const QiRef& f, int64_t patch, int64_t max_patches, int64_t& ph,
                                             int64_t& pw);

}  // namespace strata::image
