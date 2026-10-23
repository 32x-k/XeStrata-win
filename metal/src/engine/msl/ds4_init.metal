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

// ds4_kernels.cu:268  hc_split_kernel - the sinkhorn normalisation of the hyper-connection mix.
// This is the SCALAR path of the CUDA kernel (lane 0 of the threadgroup does the token, a local 8x8 matrix), applied for every
// hc including 4: it computes the same mathematics as the warp-lane version, only without the shuffles.  Correctness first -
// the lane-parallel hc == 4 version can be added behind a profile.
kernel void ds4_hc_split(device const float* mixes [[buffer(0)]], device const float* scale [[buffer(1)]],
                         device const float* base [[buffer(2)]], device float* pre [[buffer(3)]],
                         device float* post [[buffer(4)]], device float* comb [[buffer(5)]],
                         constant uint& hc [[buffer(6)]], constant uint& iters [[buffer(7)]],
                         constant float& eps [[buffer(8)]], uint b [[threadgroup_position_in_grid]],
                         uint lane [[thread_index_in_threadgroup]]) {
    if (lane != 0u) return;
    const device float* mx = mixes + (size_t) b * 32u;
    device float* pr = pre + (size_t) b * 8u;
    device float* po = post + (size_t) b * 8u;
    device float* cb = comb + (size_t) b * 64u;
    for (uint i = 0; i < 8u; ++i) {
        if (i >= hc) break;
        pr[i] = 1.0f / (1.0f + exp(-(mx[i] * scale[0] + base[i]))) + eps;
        po[i] = 2.0f / (1.0f + exp(-(mx[hc + i] * scale[1] + base[hc + i])));
    }
    float c[8][8];
    for (uint j = 0; j < 8u; ++j) {
        if (j >= hc) break;
        float m = -3.402823466e+38f;
        for (uint k = 0; k < 8u; ++k) {
            if (k >= hc) break;
            c[j][k] = mx[2u * hc + j * hc + k] * scale[2] + base[2u * hc + j * hc + k];
            m = fmax(m, c[j][k]);
        }
        float s = 0.0f;
        for (uint k = 0; k < 8u; ++k) {
            if (k >= hc) break;
            c[j][k] = exp(c[j][k] - m);
            s += c[j][k];
        }
        for (uint k = 0; k < 8u; ++k) {
            if (k >= hc) break;
            c[j][k] = c[j][k] / s + eps;
        }
    }
    for (uint k = 0; k < 8u; ++k) {
        if (k >= hc) break;
        float s = 0.0f;
        for (uint j = 0; j < 8u; ++j) if (j < hc) s += c[j][k];
        for (uint j = 0; j < 8u; ++j) if (j < hc) c[j][k] /= s + eps;
    }
    for (uint it = 1; it < iters; ++it) {
        for (uint j = 0; j < 8u; ++j) {
            if (j >= hc) break;
            float s = 0.0f;
            for (uint k = 0; k < 8u; ++k) if (k < hc) s += c[j][k];
            for (uint k = 0; k < 8u; ++k) if (k < hc) c[j][k] /= s + eps;
        }
        for (uint k = 0; k < 8u; ++k) {
            if (k >= hc) break;
            float s = 0.0f;
            for (uint j = 0; j < 8u; ++j) if (j < hc) s += c[j][k];
            for (uint j = 0; j < 8u; ++j) if (j < hc) c[j][k] /= s + eps;
        }
    }
    for (uint j = 0; j < 8u; ++j) {
        if (j >= hc) break;
        for (uint k = 0; k < 8u; ++k) { if (k >= hc) break; cb[j * hc + k] = c[j][k]; }
    }
}

// ds4_kernels.cu:1938  moe_sort - count each virtual slot's picks, prefix-sum the counts, then fill in token order.
// kPicks = 8 (ds4_kernels.cuh:19): idx is [T][8], so q -> (t = q/8, i = q%8).
kernel void ds4_bounds_scan(device int* bounds [[buffer(0)]], constant uint& n_vs [[buffer(1)]],
                            uint tid [[thread_index_in_threadgroup]]) {
    if (tid != 0u) return;
    bounds[0] = 0;
    for (uint s = 0; s < n_vs; ++s) bounds[s + 1] += bounds[s];
}

// simd_ballot returns a simd_vote, which does not cast to an integer in this toolchain; the same counts come from the
// prefix-sum and sum primitives, which give a lane its offset inside its simdgroup and the group its total.
static inline int ds4_warp_prefix(bool hit) { return simd_prefix_exclusive_sum(hit ? 1 : 0); }
static inline int ds4_warp_total(bool hit) { return simd_sum(hit ? 1 : 0); }

kernel void ds4_moe_sort_count(device const int* idx [[buffer(0)]], constant uint& T [[buffer(1)]],
                               constant uint& kk [[buffer(2)]], device const int* vs [[buffer(3)]],
                               device int* bounds [[buffer(4)]], uint slot [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]]) {
    threadgroup int warp_n[8];
    threadgroup int base_s;
    const uint wp = tid / 32u, lane = tid % 32u;
    if (tid == 0u) base_s = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint n = T * kk;
    for (uint q = tid; q < n; q += 256u) {
        const uint t = q / kk, i = q - t * kk;
        const bool hit = vs[idx[t * 8u + i]] == int(slot);
        // one vote a warp: the running offset needs each warp's count, accumulated in slot order
        const int wtot = ds4_warp_total(hit);
        if ((tid % 32u) == 0u) warp_n[wp] = wtot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        int total = 0;
        for (uint w = 0; w < 8u; ++w) total += warp_n[w];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) base_s += total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) bounds[slot + 1] = base_s;
}

kernel void ds4_moe_sort_fill(device const int* idx [[buffer(0)]], constant uint& T [[buffer(1)]],
                              constant uint& kk [[buffer(2)]], device const int* vs [[buffer(3)]],
                              device int* bounds [[buffer(4)]], device int* row_tok [[buffer(5)]],
                              device int* pos [[buffer(6)]], uint slot [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]]) {
    threadgroup int warp_n[8];
    threadgroup int base_s;
    const uint wp = tid / 32u, lane = tid % 32u;
    if (tid == 0u) base_s = bounds[slot];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint n = T * kk;
    for (uint q = tid; q < n; q += 256u) {
        const uint t = q / kk, i = q - t * kk;
        const bool hit = vs[idx[t * 8u + i]] == int(slot);
        const int wpre = ds4_warp_prefix(hit), wtot = ds4_warp_total(hit);
        if ((tid % 32u) == 0u) warp_n[wp] = wtot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        int before = 0, total = 0;
        for (uint w = 0; w < 8u; ++w) { if (w < wp) before += warp_n[w]; total += warp_n[w]; }
        if (hit) {
            const int r = base_s + before + wpre;
            row_tok[r] = int(t);
            pos[t * 8u + i] = r;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) base_s += total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
