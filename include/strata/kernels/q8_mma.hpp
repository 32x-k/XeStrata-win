// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/kernels/q8_mma.hpp - a dense product with Q8_0 weights on NVIDIA's int8 tensor cores (sm_80 on):
// the weights' values and scales as Q8_0 has them, rearranged in place, the activations as int8 with a scale for each
// 32 values (iq_mmq_quantize's layout), each block's integer sum scaled by both into FP32.  FP16 products with FP32
// sums run at half the rate of FP16 sums on GeForce cards, and int8 at twice that again; the per-block scales cost
// two FP32 instructions an element.
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>

namespace strata::kernels {

/// Whether q8_mma_gemm runs on `q`'s device: an NVIDIA GPU whose device code here was compiled for sm_80 or later
/// (asked of the device once), with the local memory and work-group size the kernel takes.  STRATA_NO_XMX=1 says no.
bool q8_mma_usable(sycl::queue& q);

/// N rows of Q8_0 blocks at w (K / 32 * 34 bytes a row) rewritten in place as q8_mma_gemm reads them: each row's K
/// int8 values, then its K / 32 FP16 scales.
void q8_mma_repack(sycl::queue& q, void* w, int64_t N, int64_t K);

/// Y[T, N] (row stride ldy) = X[T, K] . W[N, K]^T.  xq: T rows of K int8 values, then T x K / 32 FP32 scales
/// (iq_mmq_quantize without the sums); w: N rows as q8_mma_repack leaves them, 16-byte aligned.  N a multiple of 128,
/// K of 256 (so that every row is 16-byte aligned).
sycl::event q8_mma_gemm(sycl::queue& q, const void* xq, const void* w, float* y, int64_t T, int64_t N, int64_t K,
                        int64_t ldy);

}  // namespace strata::kernels
