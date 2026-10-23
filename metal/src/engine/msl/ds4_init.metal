// metal/src/engine/msl/ds4_init.metal - the first kernels the engine asks for, ported from ds4_kernels.cu.
//
// ds4_kernels.cu:1477  __global__ void init_pre_kernel(float* pre, int T) {
//     const int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < T * 8) pre[i] = (i % 8) == 0 ? 1.0f : 0.0f;
// }
#include <metal_stdlib>

using namespace metal;

// the grid is threadgroups of 128; the flat id is what CUDA's blockIdx*blockDim+threadIdx was
kernel void ds4_init_pre(device float* pre [[buffer(0)]], constant uint& T [[buffer(1)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid < T * 8u) pre[gid] = (gid % 8u) == 0u ? 1.0f : 0.0f;
}

// ds4_kernels.cu:130  embed_kernel: v = bf16(rows[t*dim + e]); h[(t*hc + s)*dim + e] = v for each of hc copies
kernel void ds4_embed(device const ushort* rows [[buffer(0)]], device float* h [[buffer(1)]],
                      constant uint& dim [[buffer(2)]], constant uint& hc [[buffer(3)]],
                      uint2 tgp [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_threadgroup]]) {
    const uint e = tgp.x * 256u + uint(lane);
    if (e >= dim) return;
    const uint t = tgp.y;
    const uint bits = uint(rows[(size_t) t * dim + e]) << 16;   // BF16 -> F32 is the top half
    const float v = as_type<float>(bits);
    for (uint s = 0; s < hc; ++s) h[((size_t) t * hc + s) * dim + e] = v;
}
