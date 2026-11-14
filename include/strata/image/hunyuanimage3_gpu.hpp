// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/hunyuanimage3_gpu.hpp - one HunyuanImage 3.0 decoder layer on the GPU, FP16 products read
// straight from the quantized file: the number `hunyuanimage3_layer_host` is compared with, and the body
// HunyuanImage3Dit streams.
#pragma once

#include <cstdint>
#include <string>

namespace strata::image {

class HunyuanImage3File;

/// Layer `layer` of the model in `f`: device FP32 `x` ([T][4096]) -> device FP32 `out` ([T][4096]).  The 2D rope
/// positions are `pos_y`/`pos_x` and the image block is the LAST `n_img` tokens (its rows see all the text and the
/// whole block; a text row its prefix), so `n_img` 0 is a plain causal text run.  `stage` is a caller-owned FP32
/// buffer of at least 6144*4096 values, reused for reading each matrix out of the file; `scratch16` one FP16
/// buffer of the same size for its FP16 image.
///
/// This is the DIRECT path: every expert of the layer's bank is dequantized and uploaded as it is used, which is
/// 1.36 GB over PCIe a layer (w4a8) - right, and slow.  The streaming resident prefix is HunyuanImage3Dit's; the
/// arithmetic below is what it will run.  False with `err` on a missing tensor or a failed allocation.
bool hunyuanimage3_layer_gpu(HunyuanImage3File& f, int64_t layer, const float* x, float* out, int64_t T,
                   const int32_t* pos_y, const int32_t* pos_x, int64_t n_img, float* stage, uint16_t* scratch16,
                   void* stream, std::string& err);

}  // namespace strata::image
