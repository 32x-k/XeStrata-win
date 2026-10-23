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
// ds4_kernels.cu:1894 finish_kernel<false>, the engine's Finish as ds4_kernels.cuh declares it
struct FinishIn {
    const float* w;
    float eps;
    int rd, pos, inverse, vpp, pos_step;
    const float* freqs;
    int quant;
    const uint16_t* dst;
    long long dst_stride;
    uint8_t* pdst;
    long long pdst_stride;
};
// ds4_kernels.cu:1908 hc_split (the sinkhorn normalisation)
bool hc_split(const float* mixes, const float* scale, const float* base, int hc, int iters, float hc_eps, int T,
              float* pre, float* post, float* comb);

// ds4_kernels.cu:1901 hc_mix
bool hc_mix(const float* h, const float* fn, int hcdim, int n_mix, int T, float eps, float* mixes);

// ds4_kernels.cu:1692 hc_mix_scale
bool hc_mix_scale(const float* h, int hcdim, int n_mix, int T, float eps, float* mixes);

// ds4_kernels.cu:1677 hc_post_inplace
bool hc_post(const float* y, float* h, const float* post, const float* comb, int dim, int hc, int T);

bool finish(float* x, int n_vec, int len, long long stride, const FinishIn& f);

bool sgemm_tn(const float* A, const float* B, float* C, int m, int nt, int k, int lda, int ldb, int ldc);

bool embed(const uint16_t* rows, int T, float* h, int dim, int hc, void* stream);

}   // namespace strata::dsv41::metal_host
