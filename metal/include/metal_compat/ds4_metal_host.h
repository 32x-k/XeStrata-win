// metal/include/metal_compat/ds4_metal_host.h - the dsv41 kernels that have a Metal dispatch, as the engine calls them.
//
// A launch site in the shared dsv41 sources becomes, under STRATA_METAL, a call to one of these instead of the generic
// strata_ds4_launch() stub.  Each implementation wraps the engine's own device buffers into the Metal runtime and dispatches
// the MSL kernel of the same name; metal/src/engine/ds4_host.cpp holds them.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::gpu {
struct StreamImpl;
using Stream = StreamImpl*;
}   // namespace strata::gpu

namespace strata::dsv41::metal_host {

// ds4_kernels.cu:1477 init_pre_kernel: pre[i] = (i % 8) == 0 ? 1 : 0, i < T*8
bool init_pre(float* pre, int T, void* stream);
// ds4_kernels.cu:130 embed_kernel: h[(t*hc + s)*dim + e] = bf16(rows[t*dim + e])
// ds4_model.cu:2737 cublasSgemm(OP_T, OP_N, ...): C[t][m] = sum_k A[m][k] * B[t][k], F32, alpha 1, beta 0
bool sgemm_tn(const float* A, const float* B, float* C, int m, int nt, int k, int lda, int ldb, int ldc);

bool embed(const uint16_t* rows, int T, float* h, int dim, int hc, void* stream);

}   // namespace strata::dsv41::metal_host
