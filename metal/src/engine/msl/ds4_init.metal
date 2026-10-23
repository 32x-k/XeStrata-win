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

// ds4_kernels.cu:65 pow2_ceil / :88 fp8_rt / :90 e2m1_rt - the KV quantiser's scale and roundings, shared by finish_pack
// (the packed rows) and ds4_finish's rounding branch
static inline float ds4_pow2_ceil(float x) {
    const uint b = as_type<uint>(x);
    const int e = int((b >> 23) & 0xffu) - 127 + ((b & 0x7fffffu) != 0u ? 1 : 0);
    return as_type<float>(uint(e + 127) << 23);
}

static inline float ds4_e4m3_round(float v) {   // E4M3, nearest even (the input is clamped)
    const float a = fabs(v);
    if (!(a == a)) return v;
    float r;
    if (a >= 448.0f) r = 448.0f;
    else {
        const int e = int((as_type<uint>(a) >> 23) & 0xffu) - 127;
        const float step = e < -6 ? 0.001953125f : as_type<float>(uint(e - 3 + 127) << 23);
        r = a == 0.0f ? 0.0f : rint(a / step) * step;
        r = fmin(r, 448.0f);
    }
    return copysign(r, v);
}

static inline float ds4_e2m1_rt(float v) {   // E2M1 magnitudes 0 .5 1 1.5 2 3 4 6, nearest, ties to even
    const float a = fabs(v);
    float r;
    if (a <= 0.25f) r = 0.0f;
    else if (a < 0.75f) r = 0.5f;
    else if (a <= 1.25f) r = 1.0f;
    else if (a < 1.75f) r = 1.5f;
    else if (a <= 2.5f) r = 2.0f;
    else if (a < 3.5f) r = 3.0f;
    else if (a <= 5.0f) r = 4.0f;
    else r = 6.0f;
    return copysign(r, v);
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
    if (f.rd > 0) {
        // mrope: the last rd values move in (re, im) pairs a lane apart (ds4_kernels.cu:161): the partner comes from a
        // lane xor 1, the angle from the token's position and the frequency of the pair
        const uint r0 = len - uint(f.rd);
        const uint pos = uint(f.pos) + (b / uint(f.vpp)) * uint(f.pos_step);
        for (uint j = 0; j < 32u; ++j) {
            if (j >= nj) break;
            const uint e = tid + j * 256u;
            const float other = simd_shuffle_xor(r[j], 1);
            if (e >= r0 && e < len) {
                const uint i = (e - r0) >> 1;
                const float ang = float(pos) * f.freqs[i];
                const float c = cos(ang);
                const float sn = f.inverse ? -sin(ang) : sin(ang);
                r[j] = ((e - r0) & 1u) == 0u ? r[j] * c - other * sn : other * sn + r[j] * c;
            }
        }
    }
    if (f.quant != 0) {   // the KV rounding without the pack: the group's scale, then the value rounded to the format
        const uint g = f.quant == 2 ? 16u : 32u;
        for (uint j = 0; j < 32u; ++j) {
            if (j >= nj) break;
            const uint e = tid + j * 256u;
            float a = e < len ? fabs(r[j]) : 0.0f;
            for (uint o = g / 2u; o > 0u; o >>= 1) a = fmax(a, simd_shuffle_xor(a, ushort(o)));
            if (e >= len) continue;
            if (f.quant == 1) {
                const float s = ds4_pow2_ceil(fmax(a, 1e-4f) * (1.0f / 448.0f));
                r[j] = ds4_e4m3_round(fmin(fmax(r[j] / s, -448.0f), 448.0f)) * s;
            } else if (f.quant == 2) {
                const float s = ds4_e4m3_round(fmax(a, 6.0f * 0.001953125f) / 6.0f);
                r[j] = ds4_e2m1_rt(fmin(fmax(r[j] / s, -6.0f), 6.0f)) * s;
            } else {
                const float s = ds4_pow2_ceil(fmax(a, 6.0f * 1.1754943508222875e-38f) * (1.0f / 6.0f));
                r[j] = ds4_e2m1_rt(fmin(fmax(r[j] / s, -6.0f), 6.0f)) * s;
            }
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

// ds4_kernels.cu:1690  engram_dequant_tok_kernel - one hash-table row a threadgroup: IQ4_NL blocks to F32.
// fmt 2 (this model's Engram): an 18-byte block is fp16 d then 16 packed nibbles; the low nibble of q[j] is value j and the
// high nibble value j + 16, each an index into ggml-common.h's kvalues_iq4nl (ds4_kernels.cu:1280).
constant int ds4_kIq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

kernel void ds4_engram_dequant(device const uchar* src [[buffer(0)]], constant ulong& per [[buffer(1)]],
                               constant uint& cols [[buffer(2)]], constant uint& hd [[buffer(3)]],
                               device float* out [[buffer(4)]], constant int& fmt [[buffer(5)]],
                               uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint t = r / cols, c = r - t * cols;
    const device uchar* rows = src + (size_t) t * per + (size_t) c * hd;
    if (fmt == 2) {
        for (uint e = tid; e < hd; e += 256u) {
            const device uchar* b = rows + (size_t) (e / 32u) * 18u;
            const uint q = b[2u + (e & 15u)];
            const int v = (e & 16u) ? int(q >> 4) : int(q & 15u);
            const ushort dh = ushort(b[0]) | (ushort(b[1]) << 8);
            out[(size_t) r * hd + e] = float(as_type<half>(dh)) * float(ds4_kIq4nl[v]);
        }
        return;
    }
    // fmt 1 (Q6_K) and fmt 0 (E4M3 x scale) are not ported: write NaN, so a wrong path shows as NaN in the output rather than
    // as a silent zero.
    for (uint e = tid; e < hd; e += 256u) out[(size_t) r * hd + e] = NAN;
}

// ds4_kernels.cu:341  hc_pre_norm_kernel - per token: a = sum_s mix[s]*h[token][s][e], one RMS over dim, then x = a*inv
// (times w when given).  One threadgroup a token, 1024 threads; a thread holds at most 8 values (dim <= 8*1024 = 8192).
kernel void ds4_hc_pre_norm(device const float* h [[buffer(0)]], device const float* mix [[buffer(1)]],
                            device const float* w [[buffer(2)]], constant uint& dim [[buffer(3)]],
                            constant uint& hc [[buffer(4)]], constant float& eps [[buffer(5)]],
                            device float* x [[buffer(6)]], constant uint& has_w [[buffer(7)]],
                            uint b [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                            ushort lane [[thread_index_in_simdgroup]], ushort wg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float red[32];
    const device float* hh = h + (size_t) b * hc * dim;
    const device float* mm = mix + (size_t) b * 8u;
    device float* xx = x + (size_t) b * dim;
    float r[8];
    float ss = 0.0f;
    uint n = 0;
    for (uint e = tid; e < dim; e += 1024u, ++n) {
        float a = 0.0f;
        for (uint s = 0; s < 8u; ++s) if (s < hc) a += mm[s] * hh[(size_t) s * dim + e];
        if (n < 8u) r[n] = a;
        ss += a * a;
    }
    // a 1024-thread reduction: 32 simdgroups, each into red[], then the first simdgroup sums those
    float v = ds4_warp_sum(ss);
    if (lane == 0u) red[wg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (wg == 0u) {
        float t = lane < 32u ? red[lane] : 0.0f;
        t = ds4_warp_sum(t);
        if (lane == 0u) red[0] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = rsqrt(red[0] / float(dim) + eps);
    n = 0;
    for (uint e = tid; e < dim; e += 1024u, ++n) {
        if (n >= 8u) break;
        xx[e] = has_w != 0u ? r[n] * inv * w[e] : r[n];
    }
}

// ds4_gguf.cu:186  q8_quant_kernel: a warp a 32-group - amax across the warp, sc = amax/127, q = rint(v/sc), d = sc.
// grid = (ceil((k/32) / 8 warps), rows), threadgroup = 256 (8 warps of 32).
kernel void ds4_q8_quant(device const float* x [[buffer(0)]], constant ulong& ldx [[buffer(1)]],
                         constant uint& k [[buffer(2)]], device char* q [[buffer(3)]],
                         device float* d [[buffer(4)]], uint2 tgp [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]]) {
    const uint warp = tid / 32u, lane = tid % 32u;
    const uint b = tgp.x * 8u + warp;         // the 32-group index
    const uint r = tgp.y;
    if ((size_t) b * 32u >= (size_t) k) return;
    const device float* xr = x + (size_t) r * ldx;
    const float v = xr[(size_t) b * 32u + lane];
    const float am = simd_max(fabs(v));
    const float sc = am / 127.0f;
    q[(size_t) r * k + (size_t) b * 32u + lane] = char(am > 0.0f ? rint(v / sc) : 0.0f);
    if (lane == 0u) d[(size_t) r * (k / 32u) + b] = sc;
}

// ds4_gguf.cu:371  mx_mv_kernel - the MXFP4 expert matrix-vector.
// grid = (ceil(n / 8 warps), jobs), threadgroup = 256.  A warp a row; a lane strides the 32-value blocks; NC columns of
// activations share each decoded block (the T=8 window), then a warp sum and y[c][row] = alpha*v + bias[row].
// Block layout (kMxBytes = 17 a 32 values): the E8M0 scale byte, then 16 bytes whose low nibble is value j and high nibble
// value j+16, each an index into the 16-entry codebook {0,1,2,3,4,6,8,12} and its negatives.
struct Ds4MxJob {
    device const void* w;
    ulong ld;           // bytes a row
    device const float* x;
    ulong xcs;
    device float* y;
    ulong ycs;
    device const float* bias;
    float alpha;
    int k, n, ncol;
    device const char* xq;
    device const float* xd;
};

static inline float ds4_e8m0(uint e) {   // 2^(e - 128), ggml's GGML_E8M0_TO_FP32_HALF
    return as_type<float>(e < 2u ? (0x00200000u << e) : ((e - 1u) << 23));
}

static inline void ds4_mx_block(device const uchar* p, threadgroup const float* lut, thread float* w) {
    const float d = ds4_e8m0(uint(p[0]));
    for (uint j = 0; j < 16u; ++j) {
        const uint q = uint(p[1u + j]);
        w[j] = lut[q & 15u] * d;
        w[j + 16u] = lut[q >> 4] * d;
    }
}

kernel void ds4_mx_mv(device const uchar* jobs [[buffer(0)]],
                      uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                      ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float lut[16];
    if (tid < 16u) {
        const float v[8] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f, 12.0f};
        lut[tid] = tid < 8u ? v[tid] : -v[tid - 8u];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const device Ds4MxJob& b = *(const device Ds4MxJob*) (jobs + (size_t) tgp.y * sizeof(Ds4MxJob));
    const uint row = tgp.x * 8u + (tid >> 5);
    if (int(row) >= b.n) return;
    const device uchar* w = (const device uchar*) b.w + (size_t) row * b.ld;
    thread float acc[8];
    for (uint c = 0; c < 8u; ++c) acc[c] = 0.0f;
    const uint nblk = uint(b.k) / 32u;
    for (uint kb = lane; kb < nblk; kb += 32u) {
        float wv[32];
        ds4_mx_block(w + (size_t) kb * 17u, lut, wv);
        for (uint c = 0; c < 8u; ++c) {
            if (int(c) >= b.ncol) break;
            const device float* x = b.x + c * b.xcs + (size_t) kb * 32u;
            float a = acc[c];
            for (uint j = 0; j < 32u; ++j) a = fma(wv[j], x[j], a);
            acc[c] = a;
        }
    }
    for (uint c = 0; c < 8u; ++c) {
        if (int(c) >= b.ncol) break;
        float v = acc[c];
        v += simd_shuffle_xor(v, 16);
        v += simd_shuffle_xor(v, 8);
        v += simd_shuffle_xor(v, 4);
        v += simd_shuffle_xor(v, 2);
        v += simd_shuffle_xor(v, 1);
        if (lane == 0u) b.y[(size_t) c * b.ycs + row] = b.alpha * v + (b.bias != nullptr ? b.bias[row] : 0.0f);
    }
}

// ds4_gguf.cu:320  q8_mv_kernel - the Q8_0 matvec against quantised activations.
// grid = (ceil(n / 8), jobs), threadgroup 256: a warp a row, a lane strides the 34-byte blocks (fp16 d + 32 int8), the dot is
// integer (CUDA's __dp4a in four-value groups; here the same sum written out), then acc += (dw * xd[c]) * sum.
// The staged-prefetch (ST) and vector variants of the CUDA kernel are prefetch strategies only - this reads the blocks directly.
kernel void ds4_q8_mv(device const uchar* jobs [[buffer(0)]], uint2 tgp [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const device Ds4MxJob& b = *(const device Ds4MxJob*) (jobs + (size_t) tgp.y * sizeof(Ds4MxJob));
    const uint row = tgp.x * 8u + (tid >> 5);
    if (int(row) >= b.n) return;
    const device uchar* w = (const device uchar*) b.w + (size_t) row * b.ld;
    const uint nb = uint(b.k) / 32u;
    float acc[8];
    for (uint c = 0; c < 8u; ++c) acc[c] = 0.0f;
    for (uint kb = lane; kb < nb; kb += 32u) {
        const device uchar* p = w + (size_t) kb * 34u;
        const ushort dh = ushort(p[0]) | (ushort(p[1]) << 8);
        const float dw = float(as_type<half>(dh));
        int wv[32];
        for (uint j = 0; j < 32u; ++j) wv[j] = int((char) p[2u + j]);
        for (uint c = 0; c < 8u; ++c) {
            if (int(c) >= b.ncol) break;
            const device char* xq = b.xq + (size_t) c * b.k + (size_t) kb * 32u;
            int sum = 0;
            for (uint j = 0; j < 32u; ++j) sum += wv[j] * int(xq[j]);
            acc[c] = fma(dw * b.xd[(size_t) c * nb + kb], float(sum), acc[c]);
        }
    }
    for (uint c = 0; c < 8u; ++c) {
        if (int(c) >= b.ncol) break;
        float v = acc[c];
        v += simd_shuffle_xor(v, 16);
        v += simd_shuffle_xor(v, 8);
        v += simd_shuffle_xor(v, 4);
        v += simd_shuffle_xor(v, 2);
        v += simd_shuffle_xor(v, 1);
        if (lane == 0u) b.y[(size_t) c * b.ycs + row] = b.alpha * v + (b.bias != nullptr ? b.bias[row] : 0.0f);
    }
}

// ds4_kernels.cu:1092  ring_to_ext_kernel - one ring row of the KV cache to BF16: each value an E4M3 code times its group's
// E8M0 power of two (row[e] and row[d + e/32]); rows before the window start write zero.
static inline float ds4_e4m3(uint b) {   // E4M3 (bias 7, no infinities, max 448): sign, 4-bit exp, 3-bit mantissa
    const float s = (b & 0x80u) ? -1.0f : 1.0f;
    const uint ex = (b >> 3) & 0xfu, m = b & 7u;
    if (ex == 0u) return s * float(m) * (1.0f / 8.0f) * (1.0f / 64.0f);
    if (ex == 15u && m == 7u) return NAN;
    return s * exp2(float(ex) - 7.0f) * (1.0f + float(m) * 0.125f);
}
static inline float ds4_ue8m0(uint b) { return as_type<float>(b << 23); }   // 2^(b - 127)
static inline ushort ds4_f2bf16(float f) {
    const uint u = as_type<uint>(f);
    return ushort((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}


// ds4_kernels.cu:1104  ext_to_ring_kernel - the window's BF16 rows packed back into the E4M3 ring, each 32-value group at the
// smallest power-of-two scale it fits under (k), so v / 2^k is an E4M3 value times a power of two.
static inline uint ds4_e4m3_code(float v, int k) {
    const uint sign = (v < 0.0f) ? 0x80u : 0u;
    float a = fabs(v) * exp2(-float(k));
    if (a == 0.0f) return sign;
    int e = int(floor(log2(a)));
    int mant = int(round((a / exp2(float(e)) - 1.0f) * 8.0f));
    int ex = e + 7;
    if (mant >= 8) { mant = 0; ++ex; }
    if (ex <= 0) { mant = 0; ex = 0; }
    if (ex > 15) { mant = 7; ex = 15; }
    return sign | (uint(ex) << 3) | uint(mant & 7);
}

kernel void ds4_ext_to_ring(device const ushort* ext [[buffer(0)]], device uchar* ring [[buffer(1)]],
                            device uchar* backup [[buffer(2)]], constant long& p0 [[buffer(3)]],
                            constant uint& n [[buffer(4)]], constant uint& T [[buffer(5)]],
                            constant uint& d [[buffer(6)]], constant ulong& rb [[buffer(7)]],
                            uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    if (T - t > n) return;                       // overwritten by a later row of the window anyway
    device uchar* row = ring + (size_t) ((size_t(p0) + t) % n) * rb;
    if (backup != nullptr) {
        for (size_t i = tid; i < rb; i += 256u) backup[(size_t) t * rb + i] = row[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const device ushort* src = ext + (size_t) (n + t) * d;
    for (uint e0 = 0; e0 < d; e0 += 256u) {
        const uint e = e0 + tid;
        const float v = e < d ? float(as_type<half>(src[e])) : 0.0f;
        uint a = as_type<uint>(v) & 0x7fffffffu;             // magnitudes ordered as bits
        for (uint o = 16u; o > 0u; o >>= 1u) a = max(a, simd_shuffle_xor(a, o));
        const int k = a == 0u ? 0 : int(a >> 23) - 135 + (((a & 0x7fffffu) > 0x600000u) ? 1 : 0);
        if (e >= d) continue;
        row[e] = uchar(ds4_e4m3_code(v, k));
        if ((e & 31u) == 0u) row[d + e / 32u] = uchar(k + 127);
    }
}


// ds4_kernels.cu:916  unpack32 - 16 threads expand one MXFP4 KV row (288 B at d=512) into 32 BF16 values a thread:
// 16 code bytes at part*16, two E4M3 scale bytes at 256 + part*2, and the 4-bit code is itself an FP4 bit pattern
// (e2m1_val14: sign and exponent/mantissa bits spliced into a half).
static inline float ds4_e2m1_val14(uint c) {
    const ushort h = (ushort) (((c & 8u) << 12) | ((c & 7u) << 9));
    return float(as_type<half>(h));
}

// ds4_kernels.cu:937  unpack_sel_kernel - the selected rows of the compressed cache into the attention scratch, one row a
// group of 16 threads.  n_sel comes either from the generated schedule (gen: min(kcap, (pos+1)/R)) or from nsel[t].
kernel void ds4_unpack_sel(device const uchar* ckv [[buffer(0)]], device const int* sel [[buffer(1)]],
                           constant uint& sel_stride [[buffer(2)]], constant int& gen [[buffer(3)]],
                           constant long& p0 [[buffer(4)]], constant int& t0 [[buffer(5)]],
                           constant int& R [[buffer(6)]], constant int& kcap [[buffer(7)]],
                           device const int* nsel [[buffer(8)]], constant uint& cap [[buffer(9)]],
                           device ushort* out [[buffer(10)]], constant ulong& rb [[buffer(11)]],
                           uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint t = tgp.y;
    int n_sel;
    if (gen != 0) {
        const long pa = p0 + long(t0) + long(t);
        n_sel = R > 0 ? int(min(long(kcap), (pa + 1) / long(R))) : 0;
    } else {
        n_sel = nsel[t];
    }
    const uint i = tgp.x * 16u + (tid >> 4u), part = tid & 15u;
    if (int(i) >= n_sel) return;
    const size_t r = (size_t) t * cap + i;
    const device uchar* rr = ckv + (size_t) sel[(size_t) t * sel_stride + i] * rb;
    const uint4 cw = *(const device uint4*) (rr + part * 16u);
    const float k0 = ds4_e4m3(uint(rr[256u + part * 2u])) * 16384.0f;
    const float k1 = ds4_e4m3(uint(rr[256u + part * 2u + 1u])) * 16384.0f;
    const uint wv[4] = {cw.x, cw.y, cw.z, cw.w};
    uint o[16];
    for (uint k = 0; k < 16u; ++k) {
        const uint b = wv[k >> 2] >> (8u * (k & 3u));
        const float ks = k < 8u ? k0 : k1;
        const float lo = ds4_e2m1_val14(b) * ks, hi = ds4_e2m1_val14(b >> 4) * ks;
        o[k] = (as_type<uint>(lo) >> 16) | (as_type<uint>(hi) & 0xffff0000u);
    }
    device uint* dst = (device uint*) (out + r * 512u + part * 32u);
    for (uint q = 0; q < 4u; ++q) {
        dst[q * 4u + 0u] = o[4u * q + 0u];
        dst[q * 4u + 1u] = o[4u * q + 1u];
        dst[q * 4u + 2u] = o[4u * q + 2u];
        dst[q * 4u + 3u] = o[4u * q + 3u];
    }
}


// ds4_kernels.cu:1772  moe_combine_kernel - a token's expert outputs back into its row: for each pick whose routed row is in
// [r0, r1), acc += w[t][i] * y[r - r0][e]; untouched when no pick falls in range.
kernel void ds4_moe_combine(device float* acc [[buffer(0)]], device const float* y [[buffer(1)]],
                            device const int* pos [[buffer(2)]], device const float* w [[buffer(3)]],
                            constant int& kk [[buffer(4)]], constant int& r0 [[buffer(5)]],
                            constant int& r1 [[buffer(6)]], constant uint& dim [[buffer(7)]],
                            uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint e = tgp.x * 256u + tid;
    if (e >= dim) return;
    const uint t = tgp.y;
    float a = acc[(size_t) t * dim + e];
    bool any = false;
    for (int i = 0; i < 8; ++i) {
        if (i >= kk) break;
        const int r = pos[(size_t) t * 8u + i];
        if (r < r0 || r >= r1) continue;
        a = fma(w[(size_t) t * 8u + i], y[(size_t) (r - r0) * dim + e], a);
        any = true;
    }
    if (any) acc[(size_t) t * dim + e] = a;
}


// ds4_kernels.cu:362  hc_post_kernel - like hc_post_inplace, but reads a separate residual and writes a separate output:
// out[t][k][e] = post[t][k]*y[t][e] + sum_j comb[t][j][k]*res[t][j][e]
kernel void ds4_hc_post_v(device const float* y [[buffer(0)]], device const float* res [[buffer(1)]],
                          device const float* post [[buffer(2)]], device const float* comb [[buffer(3)]],
                          device float* out [[buffer(4)]], constant uint& dim [[buffer(5)]],
                          constant uint& hc [[buffer(6)]], uint2 tgp [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]]) {
    const uint i = tgp.x * 256u + tid;
    if (i >= dim * hc) return;
    const uint t = tgp.y, k = i / dim, e = i - k * dim;
    const device float* r = res + (size_t) t * hc * dim;
    float a = post[(size_t) t * 8u + k] * y[(size_t) t * dim + e];
    for (uint j = 0; j < 8u; ++j) {
        if (j >= hc) break;
        a += comb[(size_t) t * 64u + j * hc + k] * r[(size_t) j * dim + e];
    }
    out[(size_t) t * hc * dim + i] = a;
}


// ds4_kernels.cu:441  gemv_f32_kernel - a warp a row of F32 weights: float4 loads of W[row] and x[t], four FMAs a step,
// lane-strided over k/4, then the warp sum into y[t][row].
kernel void ds4_gemv_f32(device const float* W [[buffer(0)]], device const float* x [[buffer(1)]],
                         constant uint& n [[buffer(2)]], constant uint& kk [[buffer(3)]],
                         device float* y [[buffer(4)]], constant uint& ldy [[buffer(5)]],
                         uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                         ushort lane [[thread_index_in_simdgroup]]) {
    const uint row = tgp.x * 8u + (tid >> 5);
    if (row >= n) return;
    const uint t = tgp.y;
    const device float4* w = (const device float4*) (W + (size_t) row * kk);
    const device float4* xv = (const device float4*) (x + (size_t) t * kk);
    float acc = 0.0f;
    for (uint c = lane; c < kk / 4u; c += 32u) {
        const float4 a = w[c], b = xv[c];
        acc = fma(a.x, b.x, acc);
        acc = fma(a.y, b.y, acc);
        acc = fma(a.z, b.z, acc);
        acc = fma(a.w, b.w, acc);
    }
    acc += simd_shuffle_xor(acc, 16);
    acc += simd_shuffle_xor(acc, 8);
    acc += simd_shuffle_xor(acc, 4);
    acc += simd_shuffle_xor(acc, 2);
    acc += simd_shuffle_xor(acc, 1);
    if (lane == 0u) y[(size_t) t * ldy + row] = acc;
}


// ds4_kernels.cu:1353  route_publish_kernel - the MoE router: sc = sqrt(softplus(logit)), bs = sc + bias, then k rounds of a
// block-wide argmax (a strided max, a warp reduction, a 16-warp pass), the picks' sc summed into the weights, and each pick's
// device address published to plan_ptr.  The prefetch requests (req) are NOT published here yet: that is the schedule, not the
// answer, and it can be turned on separately.
kernel void ds4_route_publish(device const float* logits [[buffer(0)]], device const float* bias [[buffer(1)]],
                              constant uint& n [[buffer(2)]], constant uint& kk [[buffer(3)]],
                              constant float& route_scale [[buffer(4)]], device const ulong* vram [[buffer(5)]],
                              device int* idx [[buffer(6)]], device float* w [[buffer(7)]],
                              device ulong* plan_ptr [[buffer(8)]], device const float* bias_vl [[buffer(9)]],
                              constant uint& vl [[buffer(10)]], constant uint& has_vl [[buffer(11)]],
                              uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sc[1024];
    threadgroup float bs[1024];
    threadgroup float bv[16];
    threadgroup int bi[16];
    threadgroup int pick[8];
    const device float* lg = logits + (size_t) t * 1024u;
    const device float* b = (has_vl != 0u && ((vl >> t) & 1u) != 0u) ? bias_vl : bias;
    for (uint i = tid; i < n; i += 512u) {
        const float l = lg[i];
        const float sp = l > 20.0f ? l : log(1.0f + exp(l));
        sc[i] = sqrt(sp);
        bs[i] = sc[i] + b[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint lane = tid & 31u, wp = tid >> 5u;
    for (uint r = 0; r < 8u; ++r) {
        if (r >= kk) break;
        float best = -3.402823466e+38f;
        int bix = 0x7fffffff;
        for (uint i = tid; i < n; i += 512u) {
            if (bs[i] > best || (bs[i] == best && int(i) < bix)) { best = bs[i]; bix = int(i); }
        }
        for (uint o = 16u; o > 0u; o >>= 1u) {
            const float ov = simd_shuffle_xor(best, (ushort) o);
            const int oi = simd_shuffle_xor(bix, (ushort) o);
            if (ov > best || (ov == best && oi < bix)) { best = ov; bix = oi; }
        }
        if (lane == 0u && wp < 16u) { bv[wp] = best; bi[wp] = bix; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float bb = -3.402823466e+38f;
            int ix = 0x7fffffff;
            for (uint q = 0; q < 16u; ++q) {
                if (bv[q] > bb || (bv[q] == bb && bi[q] < ix)) { bb = bv[q]; ix = bi[q]; }
            }
            pick[r] = ix;
            bs[ix] = -3.402823466e+38f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        float sum = 0.0f;
        for (uint r = 0; r < 8u; ++r) { if (r >= kk) break; sum += sc[pick[r]]; }
        for (uint r = 0; r < 8u; ++r) {
            if (r >= kk) break;
            const uint p_ = uint(pick[r]);
            const float wv = sc[p_] / (sum + 1e-20f) * route_scale;
            idx[(size_t) t * 8u + r] = int(p_);
            w[(size_t) t * 8u + r] = wv;
            plan_ptr[(size_t) t * 8u + r] = vram[p_];
        }
    }
}


// ds4_kernels.cu  swiglu_kernel - the expert MLP's activation: h = silu(g) * u, with the clamp when lim > 0
kernel void ds4_swiglu(device const float* g [[buffer(0)]], device const float* u [[buffer(1)]],
                       constant uint& n [[buffer(2)]], constant float& lim [[buffer(3)]],
                       device float* h [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    float gv = g[gid], uv = u[gid];
    if (lim > 0.0f) {                                   // both inputs clamped, then the gate itself
        gv = fmin(gv, lim);
        uv = fmin(fmax(uv, -lim), lim);
    }
    h[gid] = (gv / (1.0f + exp(-gv))) * uv;             // silu(g) * u
}

kernel void ds4_ring_to_ext(device const uchar* ring [[buffer(0)]], device ushort* ext [[buffer(1)]],
                            constant long& p0 [[buffer(2)]], constant uint& n [[buffer(3)]],
                            constant uint& d [[buffer(4)]], constant ulong& rb [[buffer(5)]],
                            uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const long p = p0 - long(n) + long(r);
    const device uchar* row = ring + (size_t) (p >= 0 ? (p % long(n)) : 0) * rb;
    for (uint e = tid; e < d; e += 256u) {
        ext[(size_t) r * d + e] = p >= 0 ? ds4_f2bf16(ds4_e4m3(uint(row[e])) * ds4_ue8m0(uint(row[d + e / 32u]))) : ushort(0);
    }
}

// ---------------------------------------------------------------- ds4_gguf.cu:577 mx_combine_kernel
//
// A token's expert outputs summed in plan order (an expert off the device adds 0), the CPU lane's part added when its
// flag is set, then the shared expert's - ds4_gguf.cu:169's combine().  Grid (n_embd / 128, T), 128 threads.
struct Ds4Comb {
    device const ulong* plan_ptr;   // [T][8]: nonzero when the device holds that expert
    device const float* plan_w;     // [T][8]
    device const float* part;       // [entries][n_embd]
    device const float* cpu_part;   // [T][cpu_stride], or null
    device const int* cpu_flag;     // [T], or null
    device const float* sh_out;     // [T][n_embd], or null
    device float* out;
    long cpu_stride, out_stride;
    int k, n_embd;
};

kernel void ds4_mx_combine(constant Ds4Comb& a [[buffer(0)]], uint2 tgp [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]]) {
    const uint r = tgp.x * 128u + tid;
    if (int(r) >= a.n_embd) return;
    const int t = int(tgp.y);
    float m = 0.0f;
    for (int i = 0; i < a.k; ++i) {   // kPicks <= 8
        const int e = t * 8 + i;
        const float pi = a.plan_ptr[e] != 0ul ? a.part[(size_t) e * (size_t) a.n_embd + r] : 0.0f;
        m = fma(a.plan_w[t * 8 + i], pi, m);
    }
    if (a.cpu_flag != nullptr && a.cpu_flag[t] != 0) m += a.cpu_part[(size_t) t * (size_t) a.cpu_stride + r];
    m += a.sh_out != nullptr ? a.sh_out[(size_t) t * (size_t) a.n_embd + r] : 0.0f;
    a.out[(size_t) t * (size_t) a.out_stride + r] = m;
}

// ---------------------------------------------------------------- ds4_kernels.cu:1286 engram_dequant_kernel / :1311 engram_gate_kernel
//
// The Engram tables' row decoder and gate (this model's tables are fmt 2: IQ4_NL, an 18-byte block of 32 values whose
// nibbles index ds4_kIq4nl above).
kernel void ds4_engram_dequant_rows(device const uchar* rows [[buffer(0)]], constant uint& hd [[buffer(1)]],
                                    device float* out [[buffer(2)]], constant int& fmt [[buffer(3)]],
                                    uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    if (fmt == 2) {
        const device uchar* row = rows + (size_t) r * (size_t) hd;
        for (uint e = tid; e < hd; e += 256u) {
            const device uchar* b = row + (size_t) (e / 32u) * 18u;
            const uint q = b[2u + (e & 15u)];
            const int v = (e & 16u) ? int(q >> 4) : int(q & 15u);
            const ushort dh = ushort(b[0]) | (ushort(b[1]) << 8);
            out[(size_t) r * (size_t) hd + e] = float(as_type<half>(dh)) * float(ds4_kIq4nl[v]);
        }
        return;
    }
    // fmt 1 (Q6_K) and fmt 0 (E4M3 x scale) are not ported: NaN rather than a silent zero (as ds4_engram_dequant does)
    for (uint e = tid; e < hd; e += 256u) out[(size_t) r * (size_t) hd + e] = NAN;
}

// ds4_kernels.cu's block_sum for a 1024-thread threadgroup: a warp sum a warp, then every warp sums the 32 partials (the
// same result in all of them, so no warp diverges at a barrier)
static inline float ds4_block_sum32(float v, threadgroup float* red, ushort lane, ushort w) {
    v = ds4_warp_sum(v);
    if (lane == 0) red[w] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t = lane < 32u ? red[lane] : 0.0f;
    t = ds4_warp_sum(t);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return t;
}

kernel void ds4_engram_gate(device float* h [[buffer(0)]], device const float* kv [[buffer(1)]],
                            device const float* w [[buffer(2)]], constant uint& hc [[buffer(3)]],
                            constant uint& dim [[buffer(4)]], constant float& eps [[buffer(5)]],
                            uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                            ushort lane [[thread_index_in_simdgroup]], ushort wg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float red[32];
    const uint k = tgp.x, t = tgp.y;
    device float* hk = h + ((size_t) t * hc + (size_t) k) * (size_t) dim;
    const device float* kvt = kv + (size_t) t * (size_t) (hc + 1u) * (size_t) dim;
    const device float* key = kvt + (size_t) k * (size_t) dim;
    const device float* val = kvt + (size_t) hc * (size_t) dim;
    const device float* wk = w + (size_t) k * (size_t) dim;
    float sh = 0.0f, sk = 0.0f, sd = 0.0f;
    for (uint e = tid; e < dim; e += 1024u) {
        const float hv = hk[e], kve = key[e];
        sh += hv * hv;
        sk += kve * kve;
        sd += hv * wk[e] * kve;
    }
    sh = ds4_block_sum32(sh, red, lane, wg);
    sk = ds4_block_sum32(sk, red, lane, wg);
    sd = ds4_block_sum32(sd, red, lane, wg);
    const float rstd = rsqrt(sh / float(dim) + eps) * rsqrt(sk / float(dim) + eps);
    const float dot = sd * rstd * rsqrt(float(dim));
    const float sq = copysign(sqrt(fmax(fabs(dot), 1e-6f)), dot);
    const float gate = 1.0f / (1.0f + exp(-sq));
    for (uint e = tid; e < dim; e += 1024u) hk[e] = hk[e] + gate * val[e];
}

// ds4_gguf.cu:72  bf16_mv_kernel<NC, true> - a BF16 matrix's matvec against F32 activations (the engram WKV here, and the
// BF16 dense matrices the jobs' f == 0 names).  Same frame as ds4_mx_mv above: a warp a row, a lane strides the row; here a
// lane 8 BF16 weights (16 bytes = one uint4) and 8 activations, so the row is walked in 256-value steps.
// A pair's F32 is bit-exact - the low half zeroed - which is bf16_mv_kernel's __uint_as_float(w << 16).
kernel void ds4_bf16_mv(device const uchar* jobs [[buffer(0)]], uint2 tgp [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const device Ds4MxJob& b = *(const device Ds4MxJob*) (jobs + (size_t) tgp.y * sizeof(Ds4MxJob));
    const uint row = tgp.x * 8u + (tid >> 5);
    if (int(row) >= b.n) return;
    const device ushort* w = (const device ushort*) b.w + (size_t) row * b.ld;
    thread float acc[16];   // (the CUDA template's largest NC; ncol cuts the rest)
    for (uint c = 0; c < 16u; ++c) acc[c] = 0.0f;
    for (uint i = uint(lane) * 8u; i < uint(b.k); i += 256u) {
        const uint4 q = *(const device uint4*) (w + i);
        const float wf[8] = {as_type<float>(q.x << 16), as_type<float>(q.x & 0xffff0000u),
                             as_type<float>(q.y << 16), as_type<float>(q.y & 0xffff0000u),
                             as_type<float>(q.z << 16), as_type<float>(q.z & 0xffff0000u),
                             as_type<float>(q.w << 16), as_type<float>(q.w & 0xffff0000u)};
        for (uint c = 0; c < 16u; ++c) {
            if (int(c) >= b.ncol) break;
            const device float4* xp = (const device float4*) (b.x + c * b.xcs + (size_t) i);
            const float4 x0 = xp[0], x1 = xp[1];
            float a = acc[c];
            a = fma(wf[0], x0.x, a);
            a = fma(wf[1], x0.y, a);
            a = fma(wf[2], x0.z, a);
            a = fma(wf[3], x0.w, a);
            a = fma(wf[4], x1.x, a);
            a = fma(wf[5], x1.y, a);
            a = fma(wf[6], x1.z, a);
            a = fma(wf[7], x1.w, a);
            acc[c] = a;
        }
    }
    for (uint c = 0; c < 16u; ++c) {
        if (int(c) >= b.ncol) break;
        float v = acc[c];
        v += simd_shuffle_xor(v, 16);
        v += simd_shuffle_xor(v, 8);
        v += simd_shuffle_xor(v, 4);
        v += simd_shuffle_xor(v, 2);
        v += simd_shuffle_xor(v, 1);
        if (lane == 0u) b.y[(size_t) c * b.ycs + row] = b.alpha * v + (b.bias != nullptr ? b.bias[row] : 0.0f);
    }
}

// ---------------------------------------------------------------- ds4_kernels.cu:524 compress_kernel / :582 index_score_kernel
//
// The compressor (ratio > 1 layers): each token's kv / score row into its slot, and every ratio'th token pooled by a
// per-value softmax over the slots, scaled by a row RMS and the norm weights.  Grid (1), 512 threads (d <= 1024 values).
static inline float ds4_block_sum_n(float v, threadgroup float* red, ushort lane, ushort w, uint nw) {
    v = ds4_warp_sum(v);
    if (lane == 0) red[w] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t = lane < nw ? red[lane] : 0.0f;
    t = ds4_warp_sum(t);   // (every warp sums the same partials: no divergence at a barrier)
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return t;
}

kernel void ds4_compress(device const float* kv [[buffer(0)]], device const float* score [[buffer(1)]],
                         device float* kv_state [[buffer(2)]], device float* score_state [[buffer(3)]],
                         constant int& ratio [[buffer(4)]], constant int& p0 [[buffer(5)]], constant int& T [[buffer(6)]],
                         constant uint& d [[buffer(7)]], device const float* wn [[buffer(8)]],
                         constant float& eps [[buffer(9)]], device float* latent [[buffer(10)]],
                         uint tid [[thread_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
                         ushort wg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float red[32];
    const float kfmax = -3.4028234663852886e+38f;   // -FLT_MAX, as the CUDA writes it
    int g = 0;
    for (int t = 0; t < T; ++t) {
        const int slot = (p0 + t) % ratio;
        const bool pool = (p0 + t + 1) % ratio == 0;
        for (uint e = tid; e < d; e += 512u) {
            kv_state[(size_t) slot * d + e] = kv[(size_t) t * d + e];
            score_state[(size_t) slot * d + e] = score[(size_t) t * d + e];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!pool) continue;
        float val[2] = {0.0f, 0.0f};   // (d <= 2 * 512)
        float ss = 0.0f;
        uint n = 0;
        for (uint e = tid; e < d; e += 512u, ++n) {
            float m = kfmax;
            for (int s = 0; s < ratio; ++s) m = fmax(m, score_state[(size_t) s * d + e]);
            float den = 0.0f;
            for (int s = 0; s < ratio; ++s) den += exp(score_state[(size_t) s * d + e] - m);
            float acc = 0.0f;
            for (int s = 0; s < ratio; ++s)
                acc += kv_state[(size_t) s * d + e] * (exp(score_state[(size_t) s * d + e] - m) / den);
            val[n] = acc;
            ss += acc * acc;
        }
        const float inv = rsqrt(ds4_block_sum_n(ss, red, lane, wg, 16u) / float(d) + eps);
        n = 0;
        for (uint e = tid; e < d; e += 512u, ++n) latent[(size_t) g * d + e] = val[n] * inv * wn[e];
        ++g;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ---------------------------------------------------------------- ds4_kernels.cu:582 index_score_kernel<128, KF>
//
// A thread a position: its key row (kfmt 3: 64 bytes of E2M1 codes (value 2i in the low nibble) then 4 UE8M0 scales; kfmt
// 0: BF16) against every head's query (a threadgroup tile), relu, then the heads' weighted sum.  Grid (ceil(n / 256), T).
struct Ds4Rows {   // ds4_kernels.cuh:115 Rows (kWin = 16)
    int n[16];
    int k[16];
    int gen, R, kcap, ccap, blk;
    long p0;
};

static inline void ds4_rows_at(constant Ds4Rows& r, int t, thread int& n, thread int& k) {
    if (r.gen == 0) {
        n = r.n[t];
        k = r.k[t];
        return;
    }
    const int nn = int((r.p0 + t + 1) / r.R);
    const int cb = min(r.ccap, (nn + r.blk - 1) / r.blk);
    switch (r.gen) {
        case 1: n = nn; k = min(r.kcap, nn); break;
        case 2: n = nn; k = cb; break;
        case 3: n = cb * r.blk; k = min(r.kcap, nn); break;
        default: n = (nn + r.blk - 1) / r.blk; k = r.kcap; break;
    }
}

kernel void ds4_index_score(device const float* q [[buffer(0)]], device const float* wq [[buffer(1)]],
                            device const uchar* kc [[buffer(2)]], device const int* cand [[buffer(3)]],
                            device float* out [[buffer(4)]], constant Ds4Rows& lim [[buffer(5)]],
                            constant int& heads [[buffer(6)]], constant int& n [[buffer(7)]],
                            constant int& cand_stride [[buffer(8)]], constant int& blk [[buffer(9)]],
                            constant long& out_stride [[buffer(10)]], constant int& kfmt [[buffer(11)]],
                            constant int& has_cand [[buffer(12)]], uint2 tgp [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]]) {
    constexpr int HD = 128;
    threadgroup float qs[32 * HD + 32];
    const int t = int(tgp.y);
    for (uint i = tid; i < uint(heads) * uint(HD); i += 256u) qs[i] = q[(size_t) t * (size_t) heads * HD + i];
    for (uint i = tid; i < uint(heads); i += 256u) qs[uint(heads) * HD + i] = wq[(size_t) t * 64 + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int i = int(tgp.x) * 256 + int(tid);
    if (i >= n) return;
    device float* o = out + (long) t * out_stride;
    int ln, lk;
    ds4_rows_at(lim, t, ln, lk);
    int p = i;
    if (has_cand != 0) {
        if (i >= lk * blk) {
            o[i] = -INFINITY;
            return;
        }
        p = cand[(size_t) t * (size_t) cand_stride + (size_t) (i / blk)] * blk + i % blk;
    }
    if (p >= ln) {
        o[i] = -INFINITY;
        return;
    }
    const size_t kb = kfmt == 3 ? (size_t) ((HD / 2 + HD / 32 + 15) / 16 * 16) : (size_t) HD * 2;
    const device uchar* kr = kc + (size_t) p * kb;
    float acc[32];
    for (uint h = 0; h < 32u; ++h) acc[h] = 0.0f;
    for (int c = 0; c < HD; c += 8) {
        float kvv[8];
        if (kfmt == 3) {   // 8 codes in 4 bytes, the group's power-of-two scale (2^(b - 127))
            const uint cw = uint(kr[c / 2]) | (uint(kr[c / 2 + 1]) << 8) | (uint(kr[c / 2 + 2]) << 16) |
                            (uint(kr[c / 2 + 3]) << 24);
            const float ks = as_type<float>(uint(kr[HD / 2 + c / 32]) << 23);
            for (uint x = 0; x < 8u; ++x) kvv[x] = ds4_e2m1_val14(cw >> (4 * x)) * 16384.0f * ks;
        } else {
            const device ushort* b = (const device ushort*) (kr + (size_t) c * 2);
            for (uint x = 0; x < 8u; ++x) kvv[x] = as_type<float>(uint(b[x]) << 16);
        }
        for (uint h = 0; h < 32u; ++h) {
            if (int(h) >= heads) break;
            threadgroup const float* qh = qs + h * HD + c;
            for (uint x = 0; x < 8u; ++x) acc[h] += qh[x] * kvv[x];
        }
    }
    float s = 0.0f;
    for (uint h = 0; h < 32u; ++h) {
        if (int(h) >= heads) break;
        s += fmax(acc[h], 0.0f) * qs[uint(heads) * HD + h];
    }
    o[i] = s;
}

// ---------------------------------------------------------------- ds4_kernels.cu:142 finish_kernel<true> (the KV pack)
//
// The same load, RMS and mrope as ds4_finish above, then the row written as the cache's packed codes (kv_row_bytes): the
// quantiser's scale byte a group (32 values, or 16 for the 4-bit MXFP4-style rows), the codes themselves from bit 0 up -
// for the 4-bit formats two values a byte, the second from the next lane.  Written from the CUDA of the same name.



static inline uint ds4_e2m1_code(float q) {   // (bit 3 the sign, -0 kept; the magnitudes as 0 .. 7)
    const float a = fabs(q);
    const uint m = a <= 1.5f ? uint(a * 2.0f) : a <= 2.0f ? 4u : a <= 3.0f ? 5u : a <= 4.0f ? 6u : 7u;
    return m | ((as_type<uint>(q) >> 31) << 3);
}

kernel void ds4_finish_pack(device float* x [[buffer(0)]], constant long& stride [[buffer(1)]],
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
    if (f.rd > 0) {
        const uint r0 = len - uint(f.rd);
        const uint pos = uint(f.pos) + (b / uint(f.vpp)) * uint(f.pos_step);
        for (uint j = 0; j < 32u; ++j) {
            if (j >= nj) break;
            const uint e = tid + j * 256u;
            const float other = simd_shuffle_xor(r[j], 1);
            if (e >= r0 && e < len) {
                const uint i = (e - r0) >> 1;
                const float ang = float(pos) * f.freqs[i];
                const float c = cos(ang);
                const float sn = f.inverse ? -sin(ang) : sin(ang);
                r[j] = ((e - r0) & 1u) == 0u ? r[j] * c - other * sn : other * sn + r[j] * c;
            }
        }
    }
    // the packed row: a group's scale and the group's codes as the format's (quant 1: d E4M3 codes at pr[e] and d/32
    // scale bytes at pr[len + e/32]; 2: E2M1 pairs at pr[e/2] with E4M3 scales at pr[len/2 + e/16]; 3: E2M1 pairs with
    // UE8M0 scales at pr[len/2 + e/32])
    const uint g = f.quant == 2 ? 16u : 32u;
    device uchar* pr = f.pdst + (long) b * f.pdst_stride;
    for (uint j = 0; j < 32u; ++j) {
        if (j >= nj) break;
        const uint e = tid + j * 256u;
        float a = e < len ? fabs(r[j]) : 0.0f;
        for (uint o = g / 2u; o > 0u; o >>= 1) a = fmax(a, simd_shuffle_xor(a, ushort(o)));
        float q = 0.0f, s = 1.0f;
        if (e < len) {
            if (f.quant == 1) {
                s = ds4_pow2_ceil(fmax(a, 1e-4f) * (1.0f / 448.0f));
                q = ds4_e4m3_round(fmin(fmax(r[j] / s, -448.0f), 448.0f));
            } else if (f.quant == 2) {
                s = ds4_e4m3_round(fmax(a, 6.0f * 0.001953125f) / 6.0f);
                q = ds4_e2m1_rt(fmin(fmax(r[j] / s, -6.0f), 6.0f));
            } else {
                s = ds4_pow2_ceil(fmax(a, 6.0f * 1.1754943508222875e-38f) * (1.0f / 6.0f));
                q = ds4_e2m1_rt(fmin(fmax(r[j] / s, -6.0f), 6.0f));
            }
            r[j] = q * s;
        }
        if (f.quant == 1) {
            if (e < len) pr[e] = uchar(ds4_e4m3_code(q, 0));
        } else {
            const uint c = ds4_e2m1_code(q);
            const uint hi = simd_shuffle_down(c, 1);   // (value e + 1: the next lane's)
            if (e < len && (e & 1u) == 0u) pr[e >> 1] = uchar(c | (hi << 4));
        }
        if (e < len && e % g == 0u)
            pr[(f.quant == 1 ? len : len / 2u) + e / g] =
                uchar(f.quant == 2 ? ds4_e4m3_code(s, 0) : (as_type<uint>(s) >> 23));
    }
    device ushort* d = f.dst != nullptr ? f.dst + (long) b * f.dst_stride : nullptr;
    for (uint j = 0; j < 32u; ++j) {
        if (j >= nj) break;
        const uint e = tid + j * 256u;
        if (e >= len) continue;
        if (d != nullptr) {
            const uint bits = as_type<uint>(r[j]);
            d[e] = ushort((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);   // BF16 round to nearest even
        }
    }
}

// ds4_kernels.cu:401  gemv_f16_rows_kernel - F16 weights against F32 activations (the indexer's wproj here): a warp a row
// of the matrix, a lane 8 halves (16 bytes, the uint4 above) and 8 activations a step, kRowsT = 2 tokens a block.  Grid
// (ceil(n / 8), ceil(T / 2)), threadgroup 256.
kernel void ds4_gemv_f16_rows(device const ushort* W [[buffer(0)]], device const float* x [[buffer(1)]],
                              constant ulong& ldx [[buffer(2)]], constant int& n [[buffer(3)]],
                              constant int& k [[buffer(4)]], constant int& T [[buffer(5)]],
                              constant float& alpha [[buffer(6)]], device float* y [[buffer(7)]],
                              constant ulong& ldy [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int kRowsT = 2;
    const int row = int(tgp.x) * 8 + int(tid >> 5);
    const int t0 = int(tgp.y) * kRowsT, nt = min(kRowsT, T - t0);
    if (row >= n) return;
    const device ushort* w = W + (size_t) row * (size_t) k;
    float acc[kRowsT];
    for (int t = 0; t < kRowsT; ++t) acc[t] = 0.0f;
    for (int c = int(lane) * 8; c < k; c += 32 * 8) {
        const uint4 q = *(const device uint4*) (w + c);
        const float2 f0 = float2(as_type<half2>(q.x)), f1 = float2(as_type<half2>(q.y));
        const float2 f2 = float2(as_type<half2>(q.z)), f3 = float2(as_type<half2>(q.w));
        const float2 fs[4] = {f0, f1, f2, f3};
        float xv[kRowsT][8];
        for (int t = 0; t < kRowsT; ++t)
            if (t < nt) {
                const device float4* xp = (const device float4*) (x + (size_t) (t0 + t) * ldx + (size_t) c);
                const float4 a4 = xp[0], b4 = xp[1];
                xv[t][0] = a4.x;
                xv[t][1] = a4.y;
                xv[t][2] = a4.z;
                xv[t][3] = a4.w;
                xv[t][4] = b4.x;
                xv[t][5] = b4.y;
                xv[t][6] = b4.z;
                xv[t][7] = b4.w;
            }
        for (int i = 0; i < 4; ++i) {
            const float2 f = fs[i];
            for (int t = 0; t < kRowsT; ++t)
                if (t < nt) acc[t] += f.x * xv[t][2 * i] + f.y * xv[t][2 * i + 1];
        }
    }
    for (int t = 0; t < kRowsT; ++t)
        if (t < nt) {
            float a = acc[t];
            a += simd_shuffle_xor(a, 16);
            a += simd_shuffle_xor(a, 8);
            a += simd_shuffle_xor(a, 4);
            a += simd_shuffle_xor(a, 2);
            a += simd_shuffle_xor(a, 1);
            if (lane == 0u) y[(size_t) (t0 + t) * ldy + row] = alpha * a;
        }
}

// ---------------------------------------------------------------- ds4_kernels.cu:650 topk_kernel
//
// A token's k largest scores: four 8-bit radix passes over the order keys of the row (a threadgroup of 1024, shared
// histograms), then the picks written in index order by a scan a pass, so the attention sums them in the same order every
// run and the ties kept are the lowest positions.
static inline uint ds4_ord_key(float f) {
    const uint b = as_type<uint>(f);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

kernel void ds4_topk(device const float* vals [[buffer(0)]], constant long& stride [[buffer(1)]],
                     constant Ds4Rows& r [[buffer(2)]], device const int* cand [[buffer(3)]],
                     constant int& cand_stride [[buffer(4)]], constant int& blk [[buffer(5)]],
                     device int* out [[buffer(6)]], constant int& out_stride [[buffer(7)]],
                     constant int& has_cand [[buffer(8)]], uint tgp [[threadgroup_position_in_grid]],
                     uint tid [[thread_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
                     ushort wid [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint hist[256];
    threadgroup uint prefix_s, mask_s;
    threadgroup int remaining_s, cnt_gt, cnt_eq;
    threadgroup int wgt[32], weq[32];
    const int t = int(tgp);
    int n, k;
    ds4_rows_at(r, t, n, k);
    device const float* v = vals + (long) t * stride;
    device int* o = out + (size_t) t * (size_t) out_stride;
    device const int* cd = has_cand != 0 ? cand + (size_t) t * (size_t) cand_stride : nullptr;
    if (n <= 0) return;
    if (k >= n) {
        for (uint i = tid; i < uint(n); i += 1024u)
            o[i] = cd != nullptr ? cd[i / uint(blk)] * blk + int(i % uint(blk)) : int(i);
        return;
    }
    if (tid == 0u) {
        prefix_s = 0u;
        mask_s = 0u;
        remaining_s = k;
        cnt_gt = 0;
        cnt_eq = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (uint i = tid; i < 256u; i += 1024u) hist[i] = 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint prefix = prefix_s, mask = mask_s;
        for (int i = int(tid); i < n; i += 1024) {
            const uint u = ds4_ord_key(v[i]);
            if ((u & mask) == prefix) {
                threadgroup atomic_uint* hp = (threadgroup atomic_uint*) &hist[(u >> shift) & 0xffu];
                atomic_fetch_add_explicit(hp, 1u, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            const int rem = remaining_s;
            int cum = 0;
            for (int d = 255; d >= 0; --d) {
                const int h = int(hist[d]);
                if (cum + h >= rem) {
                    prefix_s = prefix | (uint(d) << shift);
                    mask_s = mask | (0xffu << shift);
                    remaining_s = rem - cum;
                    break;
                }
                cum += h;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint thr = prefix_s;
    const int remaining = remaining_s;
    const int n_gt = k - remaining;
    for (int i0 = 0; i0 < n; i0 += 1024) {
        const int i = i0 + int(tid);
        const uint u = i < n ? ds4_ord_key(v[i]) : 0u;
        const bool gt = i < n && u > thr, eq = i < n && u == thr;
        const auto bgv = simd_ballot(gt);
        const auto bev = simd_ballot(eq);
        const uint bg = uint((simd_vote::vote_t)bgv), be = uint((simd_vote::vote_t)bev);
        if (lane == 0u) {
            wgt[wid] = int(popcount(bg));
            weq[wid] = int(popcount(be));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (wid == 0u) {
            const int g0 = wgt[lane], e0 = weq[lane];
            int gs = g0, es = e0;
            for (uint o2 = 1u; o2 < 32u; o2 <<= 1) {
                const int a = simd_shuffle_up(gs, ushort(o2)), b = simd_shuffle_up(es, ushort(o2));
                if (uint(lane) >= o2) {
                    gs += a;
                    es += b;
                }
            }
            wgt[lane] = cnt_gt + gs - g0;
            weq[lane] = cnt_eq + es - e0;
            simdgroup_barrier(mem_flags::mem_none);
            if (lane == 31u) {
                cnt_gt += gs;
                cnt_eq += es;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        int slot = -1;
        const uint below = (1u << lane) - 1u;
        if (gt) slot = wgt[wid] + int(popcount(bg & below));
        else if (eq) {
            const int e = weq[wid] + int(popcount(be & below));
            if (e < remaining) slot = n_gt + e;
        }
        if (slot >= 0) o[slot] = cd != nullptr ? cd[i / blk] * blk + i % blk : i;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ds4_kernels.cu:751 block_max_kernel - a block of `blk` positions' largest score (the indexer's candidate blocks), the
// newest position's block pinned in at +inf.  Grid (ceil(nb_max / 256), T), 256 threads.
kernel void ds4_block_max(device const float* score [[buffer(0)]], constant long& stride [[buffer(1)]],
                          constant Ds4Rows& r [[buffer(2)]], constant int& blk [[buffer(3)]],
                          device float* bm [[buffer(4)]], constant long& bm_stride [[buffer(5)]],
                          uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const int b = int(tgp.x) * 256 + int(tid), t = int(tgp.y);
    int n, k_;
    ds4_rows_at(r, t, n, k_);
    const int nb = (n + blk - 1) / blk;
    if (b >= nb) return;
    float m = -INFINITY;
    for (int j = 0; j < blk; ++j) {
        const int p = b * blk + j;
        if (p < n) m = fmax(m, score[(long) t * stride + p]);
    }
    if (b == (n - 1) / blk) m = INFINITY;   // the newest position's block is pinned in
    bm[(long) t * bm_stride + b] = m;
}

// ---------------------------------------------------------------- ds4_kernels.cu:1141 hc_mean_kernel / :1150 iota_rows_kernel
//
// The two small shapes the decode loop asks for: a token's mean over its hc streams, and a row's indices.
kernel void ds4_hc_mean(device const float* h [[buffer(0)]], constant int& hc [[buffer(1)]],
                        constant int& dim [[buffer(2)]], device float* out [[buffer(3)]],
                        constant long& out_stride [[buffer(4)]], uint2 tgp [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]]) {
    const int e = int(tgp.x) * 256 + int(tid), t = int(tgp.y);
    if (e >= dim) return;
    float a = 0.0f;
    for (int s = 0; s < hc; ++s) a += h[((size_t) t * (size_t) hc + (size_t) s) * (size_t) dim + (size_t) e];
    out[(long) t * out_stride + e] = a / float(hc);
}

kernel void ds4_iota_rows(device int* out [[buffer(0)]], constant int& n [[buffer(1)]], constant int& stride [[buffer(2)]],
                          uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint t = tg;
    for (uint i = tid; i < uint(n); i += 64u) out[(size_t) t * (size_t) stride + i] = int(i);
}

// ---------------------------------------------------------------- vendor/sampler.cu:93 sampler_greedy_kernel
//
// The plain argmax over the vocabulary, ONE BLOCK (1024) a token: a thread's strided candidates with the repetition
// penalties applied (a threadgroup bitmap of the history window says which tokens can be penalised at all), a warp
// reduction, then warp 0's - a tie keeps the larger value and the SMALLER index, so this is the serial scan's answer.
kernel void ds4_sampler_greedy(device const float* logits [[buffer(0)]], constant int& n_vocab [[buffer(1)]],
                               device const int* history [[buffer(2)]], constant int& history_len [[buffer(3)]],
                               constant int& plen [[buffer(4)]], constant float& pen_repeat [[buffer(5)]],
                               constant float& pen_freq [[buffer(6)]], constant float& pen_present [[buffer(7)]],
                               device const float* logit_bias [[buffer(8)]], constant int& has_bias [[buffer(9)]],
                               device int* out [[buffer(10)]], uint tg [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
                               ushort wid [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint penal_bits[4096];   // (n_vocab + 31) / 32 words (the host checks it fits)
    threadgroup float sv[32];
    threadgroup int si[32];
    const int t = int(tg);
    device const float* l = logits + (size_t) t * (size_t) n_vocab;
    device const int* hrow = history != nullptr ? history + (size_t) t * (size_t) history_len : nullptr;
    int hlen = 0;
    if (hrow != nullptr) {
        hlen = min(plen, history_len);
        if (hlen < 0) hlen = 0;
        hrow += history_len - hlen;   // the window is the tail
    }
    const int bits_words = (n_vocab + 31) / 32;
    const bool use_bits = hrow != nullptr && hlen > 0 && bits_words > 0;
    if (use_bits) {
        for (int w = int(tid); w < bits_words; w += 1024) penal_bits[w] = 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = int(tid); i < hlen; i += 1024)
            if (hrow[i] >= 0 && hrow[i] < n_vocab) {
                threadgroup atomic_uint* pb = (threadgroup atomic_uint*) &penal_bits[hrow[i] >> 5];
                atomic_fetch_or_explicit(pb, 1u << (hrow[i] & 31), memory_order_relaxed);
            }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float bv = -INFINITY;
    int best = n_vocab;   // (the "no candidate" index: it loses every comparison)
    for (int v = int(tid); v < n_vocab; v += 1024) {
        float s = has_bias != 0 ? l[v] + logit_bias[v] : l[v];
        int cnt = 0;
        if (use_bits && (penal_bits[v >> 5] & (1u << (v & 31))) != 0u)
            for (int i = 0; i < hlen; ++i)
                if (hrow[i] == v) ++cnt;
        if (cnt > 0) {
            if (s <= 0.0f) s *= pen_repeat;
            else s /= pen_repeat;
            s -= float(cnt) * pen_freq + 1.0f * pen_present;
        }
        if (s > bv) {
            bv = s;
            best = v;
        }
    }
    for (uint o = 16u; o > 0u; o >>= 1) {
        const float ov = simd_shuffle_down(bv, ushort(o));
        const int oi = simd_shuffle_down(best, ushort(o));
        if (ov > bv || (ov == bv && oi < best)) {
            bv = ov;
            best = oi;
        }
    }
    if (lane == 0u) {
        sv[wid] = bv;
        si[wid] = best;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (wid == 0u) {
        float wv = sv[lane];
        int wi = si[lane];
        for (uint o = 16u; o > 0u; o >>= 1) {
            const float ov = simd_shuffle_down(wv, ushort(o));
            const int oi = simd_shuffle_down(wi, ushort(o));
            if (ov > wv || (ov == wv && oi < wi)) {
                wv = ov;
                wi = oi;
            }
        }
        if (lane == 0u) out[t] = (wi < n_vocab) ? wi : 0;   // (a tie between two -inf candidates: 0)
    }
}
