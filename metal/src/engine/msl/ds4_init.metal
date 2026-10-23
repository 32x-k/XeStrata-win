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

// ds4_model.cu:2737  cublasSgemm(OP_T, OP_N, n_mix, T, hcd, ...) - the mHC mixes: mixes[t][m] = h[t] . fn[m]
// one thread an output element; both operands F32, alpha = 1, beta = 0 (every call site seen)
kernel void ds4_sgemm_tn(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],
                         device float* C [[buffer(2)]], constant uint& m [[buffer(3)]], constant uint& nt [[buffer(4)]],
                         constant uint& k [[buffer(5)]], constant uint& lda [[buffer(6)]], constant uint& ldb [[buffer(7)]],
                         constant uint& ldc [[buffer(8)]], uint gid [[thread_position_in_grid]]) {
    const uint total = m * nt;
    if (gid >= total) return;
    const uint t = gid / m, mm = gid % m;
    const device float* a = A + (size_t) mm * lda;    // row mm of A
    const device float* b = B + (size_t) t * ldb;     // row t of B
    float acc = 0.0f;
    for (uint i = 0; i < k; ++i) acc += a[i] * b[i];
    C[(size_t) t * ldc + mm] = acc;
}

// ds4_kernels.cu:1894  finish_kernel<false> - RMS norm (f.w), mrope (f.rd), rounding (f.quant), then store to x or dst.
// The mrope and rounding branches are not written yet; the host wrapper will decline until they are.
struct Ds4Finish {
    device float* w;
    float eps;
    int rd, pos, inverse, vpp, pos_step;
    device float* freqs;
    int quant;
    device ushort* dst;
    long dst_stride;
    device uchar* pdst;
    long pdst_stride;
};

static inline float ds4_warp_sum(float v) {
    v += simd_shuffle_xor(v, 16);
    v += simd_shuffle_xor(v, 8);
    v += simd_shuffle_xor(v, 4);
    v += simd_shuffle_xor(v, 2);
    v += simd_shuffle_xor(v, 1);
    return v;
}
static inline float ds4_block_sum(float v, threadgroup float* red, ushort lane, ushort w) {
    v = ds4_warp_sum(v);
    if (lane == 0) red[w] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (w == 0) {
        float t = lane < 8 ? red[lane] : 0.0f;
        t = ds4_warp_sum(t);
        if (lane == 0) red[0] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return red[0];
}

kernel void ds4_finish(device float* x [[buffer(0)]], constant long& stride [[buffer(1)]],
                       constant uint& len [[buffer(2)]], constant Ds4Finish& f [[buffer(3)]],
                       uint tid [[thread_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
                       ushort w [[simdgroup_index_in_threadgroup]], uint b [[threadgroup_position_in_grid]]) {
    threadgroup float red[8];
    device float* v = x + (long) b * stride;
    const uint nj = (len + 255u) / 256u;
    float r[32];
    float ss = 0.0f;
    for (uint j = 0; j < 32u; ++j) {
        if (j >= nj) break;
        const uint e = tid + j * 256u;
        r[j] = e < len ? v[e] : 0.0f;
        ss += r[j] * r[j];
    }
    if (f.w != nullptr) {
        const float tot = ds4_block_sum(ss, red, lane, w);
        const float inv = rsqrt(tot / float(len) + f.eps);
        for (uint j = 0; j < 32u; ++j) {
            if (j >= nj) break;
            const uint e = tid + j * 256u;
            if (e < len) r[j] = r[j] * inv * f.w[e];
        }
    }
    for (uint j = 0; j < 32u; ++j) {
        if (j >= nj) break;
        const uint e = tid + j * 256u;
        if (e >= len) continue;
        if (f.dst != nullptr) {
            const uint bits = as_type<uint>(r[j]);
            f.dst[e] = ushort((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);   // BF16 round to nearest even
        } else {
            v[e] = r[j];
        }
    }
}

// ds4_kernels.cu:1677  hc_post_inplace_kernel: h[t][k][e] = post[t][k]*y[t][e] + sum_j comb[t][j][k]*h_old[t][j][e]
kernel void ds4_hc_post(device const float* y [[buffer(0)]], device float* h [[buffer(1)]],
                        device const float* post [[buffer(2)]], device const float* comb [[buffer(3)]],
                        constant uint& dim [[buffer(4)]], constant uint& hc [[buffer(5)]],
                        uint2 tgp [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_threadgroup]]) {
    const uint e = tgp.x * 256u + uint(lane);
    if (e >= dim) return;
    const uint t = tgp.y;
    device float* r = h + (size_t) t * hc * dim;
    float rv[8];
    for (uint j = 0; j < 8u; ++j) if (j < hc) rv[j] = r[(size_t) j * dim + e];
    const float yv = y[(size_t) t * dim + e];
    for (uint k = 0; k < 8u; ++k) {
        if (k >= hc) break;
        float a = post[(size_t) t * 8u + k] * yv;
        for (uint j = 0; j < 8u; ++j) if (j < hc) a += comb[(size_t) t * 64u + j * hc + k] * rv[j];
        r[(size_t) k * dim + e] = a;
    }
}

// ds4_kernels.cu:1692  hc_mix_scale_kernel: mixes[t][i] *= rsqrt(sum(h[t])^2 / hcdim + eps), one threadgroup a token
kernel void ds4_hc_mix_scale(device const float* h [[buffer(0)]], device float* mixes [[buffer(1)]],
                             constant uint& hcdim [[buffer(2)]], constant uint& n_mix [[buffer(3)]],
                             constant float& eps [[buffer(4)]], uint b [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
                             ushort w [[simdgroup_index_in_threadgroup]]) {
    threadgroup float red[8];
    const device float* x = h + (size_t) b * hcdim;
    float ss = 0.0f;
    for (uint k = tid * 4u; k < hcdim; k += 256u * 4u) {
        const float4 hv = *(const device float4*) (x + k);
        ss += hv.x * hv.x + hv.y * hv.y + hv.z * hv.z + hv.w * hv.w;
    }
    ss = ds4_block_sum(ss, red, lane, w);
    const float inv = rsqrt(ss / float(hcdim) + eps);
    if (tid < n_mix) mixes[(size_t) b * 32u + tid] *= inv;
}

// ds4_kernels.cu:1017  hc_mix_kernel: mixes[t][m] = (h[t] . fn[m]) * rsqrt(mean(h[t]^2) + eps), one threadgroup a (m, t)
kernel void ds4_hc_mix(device const float* h [[buffer(0)]], device const float* fn [[buffer(1)]],
                       device float* mixes [[buffer(2)]], constant uint& hcdim [[buffer(3)]],
                       constant float& eps [[buffer(4)]], uint2 tgp [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
                       ushort w [[simdgroup_index_in_threadgroup]]) {
    threadgroup float red[8];
    const uint m = tgp.x, t = tgp.y;
    const device float* x = h + (size_t) t * hcdim;
    const device float* f = fn + (size_t) m * hcdim;
    float acc = 0.0f, ss = 0.0f;
    for (uint k = tid * 4u; k < hcdim; k += 256u * 4u) {
        const float4 hv = *(const device float4*) (x + k);
        const float4 fv = *(const device float4*) (f + k);
        acc += hv.x * fv.x + hv.y * fv.y + hv.z * fv.z + hv.w * fv.w;
        ss += hv.x * hv.x + hv.y * hv.y + hv.z * hv.z + hv.w * hv.w;
    }
    acc = ds4_block_sum(acc, red, lane, w);
    ss = ds4_block_sum(ss, red, lane, w);
    if (tid == 0) mixes[(size_t) t * 32u + m] = acc * rsqrt(ss / float(hcdim) + eps);
}
