// src/kernels/cuda/dflash2.cu - the small kernels of the DFlash2 drafter.  See include/strata/kernels/dflash2.hpp.
#include "strata/kernels/dflash2.hpp"

#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

constexpr int HD = kDf2Hd;

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "dflash2 kernel %s: %s\n", what, cudaGetErrorString(e));
        throw std::runtime_error(std::string("dflash2 kernel ") + what + ": " + cudaGetErrorString(e));
    }
}

__device__ __forceinline__ float bf16f(uint16_t h) { return __uint_as_float((uint32_t) h << 16); }

// ---- the grouped dynamic 2-tap convolution over the block's rows
__global__ void conv_kernel(const float* __restrict__ x, const float* __restrict__ dyn, int dyn_stride, const float* __restrict__ base,
                            const float* __restrict__ resid, float* __restrict__ out, int rows, int H, int group) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * H) return;
    const int b = i / H, c = i % H;
    const int groups = H / group;
    float acc = 0.0f;
#pragma unroll
    for (int o = 0; o < 2; ++o) {
        if (b - o < 0) continue;
        const float k = base[o * H + c] + dyn[(size_t) b * dyn_stride + o * groups + c / group];
        acc = fmaf(k, x[(size_t) (b - o) * H + c], acc);
    }
    out[i] = resid ? resid[i] + acc : acc;
}

__global__ void rope_table_kernel(const int32_t* __restrict__ pos, int rows, float theta, float* __restrict__ table) {
    const int r = blockIdx.x, i = threadIdx.x;   // 64 threads
    if (r >= rows) return;
    const double inv = pow((double) theta, -(double) (2 * i) / (double) HD);
    double ang = (double) pos[r] * inv;
    const double two_pi = 6.283185307179586476925286766559;
    ang -= two_pi * rint(ang / two_pi);   // [-pi, pi]
    float s, c;
    sincosf((float) ang, &s, &c);
    table[(r * 2 + 0) * 64 + i] = c;
    table[(r * 2 + 1) * 64 + i] = s;
}

// one block per (row, head), 128 threads: out = rope(rmsnorm(in) * gamma)
__global__ void norm_rope_kernel(const float* __restrict__ in, int in_stride, const float* __restrict__ gamma, const int32_t* __restrict__ pos,
                                 const float* __restrict__ table, float* __restrict__ out, int slot_mod, int heads, float eps) {
    const int r = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    __shared__ float red[4];
    __shared__ float y[HD];
    const float x = in[(size_t) r * in_stride + h * HD + d];
    float ss = x * x;
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if ((d & 31) == 0) red[d >> 5] = ss;
    __syncthreads();
    const float tot = red[0] + red[1] + red[2] + red[3];
    const float v = x * rsqrtf(tot / (float) HD + eps) * gamma[d];
    float res = v;
    if (table != nullptr) {
        y[d] = v;
        __syncthreads();
        const int i = d & 63;
        const float c = table[(r * 2 + 0) * 64 + i], s = table[(r * 2 + 1) * 64 + i];
        res = d < 64 ? v * c - y[d + 64] * s : v * c + y[d - 64] * s;
    }
    const int slot = slot_mod > 0 ? (int) (pos[r] % slot_mod) : r;
    out[((size_t) slot * heads + h) * HD + d] = res;
}

__global__ void copy_rows_kernel(const float* __restrict__ in, float* __restrict__ out, const int32_t* __restrict__ pos, int slot_mod, int rows, int cols) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * cols) return;
    const int r = i / cols, c = i % cols;
    const int slot = slot_mod > 0 ? (int) (pos[r] % slot_mod) : r;
    out[(size_t) slot * cols + c] = in[i];
}

// ---- attention.  grid (8 K/V heads, kDf2MaxChunks), 256 threads.  32 queries (4 heads x 8 rows) per block share each key tile.
constexpr int KS = 129;   // smem row stride of the K / V / Q tiles (conflict-free)
__global__ void __launch_bounds__(256) attn_part_kernel(const float* __restrict__ q, const float* __restrict__ kring, const float* __restrict__ vring,
                                                       const float* __restrict__ kblk, const float* __restrict__ vblk,
                                                       const int32_t* __restrict__ anchor_pos, int ring_slots, int window, float* __restrict__ part) {
    __shared__ float Qs[32 * KS];
    __shared__ float Ks[32 * KS];   // the K tile, then the V tile
    __shared__ float Ss[32 * 33];
    const int kvh = blockIdx.x, chunk = blockIdx.y, tid = threadIdx.x;
    const int s = *anchor_pos;
    const int lo = s - (window - 1) > 0 ? s - (window - 1) : 0;
    const int nctx = s - lo;
    const int nkeys = nctx + kDf2Rows;
    const int qi = tid >> 3, tg = tid & 7;
    const int qrow = qi >> 2, qg = qi & 3;
    float* const pout = part + (((size_t) chunk * kDf2Kv + kvh) * 32 + qi) * 130;
    const int k_begin = chunk * kDf2Chunk;
    if (k_begin >= nkeys) {
        if (tg == 0) { pout[0] = -INFINITY; pout[1] = 0.0f; }
        for (int e = 0; e < 16; ++e) pout[2 + e * 8 + tg] = 0.0f;
        return;
    }
    const float scale = rsqrtf((float) HD);
    for (int i = tid; i < 32 * HD; i += 256) {
        const int a = i / HD, d = i % HD;
        const int row = a >> 2, g = a & 3;
        Qs[a * KS + d] = q[((size_t) row * 32 + kvh * kDf2Gqa + g) * HD + d] * scale;
    }
    float m = -INFINITY, l = 0.0f;
    float O[16];
#pragma unroll
    for (int e = 0; e < 16; ++e) O[e] = 0.0f;
    const int k_end = min(k_begin + kDf2Chunk, nkeys);
    for (int k0 = k_begin; k0 < k_end; k0 += 32) {
        __syncthreads();   // Ks free (previous tile's V reads done), Qs written (first tile)
        for (int i = tid; i < 32 * HD; i += 256) {
            const int t = i / HD, d = i % HD, k = k0 + t;
            float v = 0.0f;
            if (k < nctx) v = kring[((size_t) ((lo + k) % ring_slots) * kDf2Kv + kvh) * HD + d];
            else if (k < nkeys) v = kblk[((size_t) (k - nctx) * kDf2Kv + kvh) * HD + d];
            Ks[t * KS + d] = v;
        }
        __syncthreads();
        float sc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 8
        for (int d = 0; d < HD; ++d) {
            const float qv = Qs[qi * KS + d];
#pragma unroll
            for (int u = 0; u < 4; ++u) sc[u] = fmaf(qv, Ks[(tg * 4 + u) * KS + d], sc[u]);
        }
        float tmax = -INFINITY;
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int k = k0 + tg * 4 + u;
            bool ok = k < nkeys;
            if (ok && k < nctx) ok = (lo + k) > (s + qrow - window);   // s + row - j < window
            sc[u] = ok ? sc[u] : -INFINITY;
            tmax = fmaxf(tmax, sc[u]);
        }
        tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, 1));
        tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, 2));
        tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, 4));
        const float mn = fmaxf(m, tmax);
        const float corr = (m == -INFINITY) ? 0.0f : __expf(m - mn);
        float psum = 0.0f;
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const float p = sc[u] == -INFINITY ? 0.0f : __expf(sc[u] - mn);
            Ss[qi * 33 + tg * 4 + u] = p;
            psum += p;
        }
        psum += __shfl_xor_sync(0xffffffffu, psum, 1);
        psum += __shfl_xor_sync(0xffffffffu, psum, 2);
        psum += __shfl_xor_sync(0xffffffffu, psum, 4);
        l = l * corr + psum;
        m = mn;
#pragma unroll
        for (int e = 0; e < 16; ++e) O[e] *= corr;
        __syncthreads();   // Ss complete; every read of Ks done
        for (int i = tid; i < 32 * HD; i += 256) {
            const int t = i / HD, d = i % HD, k = k0 + t;
            float v = 0.0f;
            if (k < nctx) v = vring[((size_t) ((lo + k) % ring_slots) * kDf2Kv + kvh) * HD + d];
            else if (k < nkeys) v = vblk[((size_t) (k - nctx) * kDf2Kv + kvh) * HD + d];
            Ks[t * KS + d] = v;
        }
        __syncthreads();
        for (int t = 0; t < 32; ++t) {
            const float p = Ss[qi * 33 + t];
#pragma unroll
            for (int e = 0; e < 16; ++e) O[e] = fmaf(p, Ks[t * KS + e * 8 + tg], O[e]);
        }
    }
    if (tg == 0) { pout[0] = m; pout[1] = l; }
#pragma unroll
    for (int e = 0; e < 16; ++e) pout[2 + e * 8 + tg] = O[e];
    (void) qg;
}

__global__ void attn_merge_kernel(const float* __restrict__ part, float* __restrict__ out) {
    const int a = blockIdx.x;                      // kvh * 32 + qi
    const int kvh = a >> 5, qi = a & 31, d = threadIdx.x;
    float M = -INFINITY;
    for (int c = 0; c < kDf2MaxChunks; ++c) M = fmaxf(M, part[(((size_t) c * kDf2Kv + kvh) * 32 + qi) * 130]);
    float L = 0.0f, o = 0.0f;
    for (int c = 0; c < kDf2MaxChunks; ++c) {
        const float* p = part + (((size_t) c * kDf2Kv + kvh) * 32 + qi) * 130;
        const float mc = p[0];
        const float w = (mc == -INFINITY) ? 0.0f : __expf(mc - M);
        L += p[1] * w;
        o = fmaf(p[2 + d], w, o);
    }
    const int row = qi >> 2, g = qi & 3;
    out[((size_t) row * 32 + kvh * kDf2Gqa + g) * HD + d] = L > 0.0f ? o / L : 0.0f;
}

// ---- top-16
constexpr int TK_CHUNK_MAX = 4096;
__device__ __forceinline__ bool better(float va, int ia, float vb, int ib) { return va > vb || (va == vb && ia < ib); }

__global__ void topk_part_kernel(const float* __restrict__ logits, int n, float* __restrict__ cval, int32_t* __restrict__ cidx) {
    __shared__ float vals[TK_CHUNK_MAX];
    __shared__ float wv[8];
    __shared__ int wi[8];
    __shared__ int win;
    const int row = blockIdx.x, blk = blockIdx.y, tid = threadIdx.x;
    const int cs = (n + kDf2TopBlocks - 1) / kDf2TopBlocks;
    const int c0 = blk * cs, c1 = min(c0 + cs, n);
    for (int i = tid; i < cs; i += 256) vals[i] = (c0 + i < c1) ? logits[(size_t) row * n + c0 + i] : -INFINITY;
    __syncthreads();
    for (int k = 0; k < kDf2TopK; ++k) {
        float bv = -INFINITY;
        int bi = 0x7fffffff;
        for (int i = tid; i < cs; i += 256)
            if (better(vals[i], i, bv, bi)) { bv = vals[i]; bi = i; }
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (better(ov, oi, bv, bi)) { bv = ov; bi = oi; }
        }
        if ((tid & 31) == 0) { wv[tid >> 5] = bv; wi[tid >> 5] = bi; }
        __syncthreads();
        if (tid == 0) {
            float v = wv[0];
            int ix = wi[0];
            for (int w = 1; w < 8; ++w)
                if (better(wv[w], wi[w], v, ix)) { v = wv[w]; ix = wi[w]; }
            const size_t o = ((size_t) row * kDf2TopBlocks + blk) * kDf2TopK + k;
            cval[o] = v;
            cidx[o] = v == -INFINITY ? -1 : c0 + ix;
            win = ix;
            if (v != -INFINITY) vals[ix] = -INFINITY;
        }
        __syncthreads();
    }
}

__global__ void topk_merge_kernel(const float* __restrict__ cval, const int32_t* __restrict__ cidx, const int32_t* __restrict__ idx_map,
                                  float* __restrict__ val, int32_t* __restrict__ tok) {
    constexpr int NC = kDf2TopBlocks * kDf2TopK;
    __shared__ float sv[NC];
    __shared__ int si[NC];
    __shared__ float wv[8];
    __shared__ int wi[8], wp[8];
    const int row = blockIdx.x, tid = threadIdx.x;
    for (int i = tid; i < NC; i += 256) { sv[i] = cval[(size_t) row * NC + i]; si[i] = cidx[(size_t) row * NC + i]; }
    __syncthreads();
    for (int k = 0; k < kDf2TopK; ++k) {
        float bv = -INFINITY;
        int bi = 0x7fffffff, bp = -1;
        for (int i = tid; i < NC; i += 256)
            if (si[i] >= 0 && better(sv[i], si[i], bv, bi)) { bv = sv[i]; bi = si[i]; bp = i; }
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            const int op = __shfl_xor_sync(0xffffffffu, bp, o);
            if (better(ov, oi, bv, bi)) { bv = ov; bi = oi; bp = op; }
        }
        if ((tid & 31) == 0) { wv[tid >> 5] = bv; wi[tid >> 5] = bi; wp[tid >> 5] = bp; }
        __syncthreads();
        if (tid == 0) {
            float v = wv[0];
            int ix = wi[0], p = wp[0];
            for (int w = 1; w < 8; ++w)
                if (better(wv[w], wi[w], v, ix)) { v = wv[w]; ix = wi[w]; p = wp[w]; }
            if (p >= 0) { si[p] = -1; }
            val[row * kDf2TopK + k] = v;
            tok[row * kDf2TopK + k] = p >= 0 ? (idx_map ? idx_map[ix] : ix) : 0;
        }
        __syncthreads();
    }
}

// ---- the selector's walk: one block of 256 threads
__global__ void select_kernel(const float* __restrict__ hp, const float* __restrict__ unary, const int32_t* __restrict__ tok,
                              const uint16_t* __restrict__ pred_cb, const uint16_t* __restrict__ succ_cb, int rank, int rows,
                              const int32_t* __restrict__ anchor_tok, int32_t* __restrict__ out_tok, float* __restrict__ out_prob) {
    __shared__ float vp[256];
    __shared__ float sc[kDf2TopK];
    __shared__ int prev;
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
    if (tid == 0) prev = *anchor_tok;
    __syncthreads();
    for (int i = 0; i < rows; ++i) {
        vp[tid] = bf16f(pred_cb[(size_t) prev * rank + tid]) * hp[(size_t) i * rank + tid];
        __syncthreads();
        for (int j = w * 2; j < w * 2 + 2; ++j) {
            const int t = tok[i * kDf2TopK + j];
            const uint4 raw = *reinterpret_cast<const uint4*>(succ_cb + (size_t) t * rank + lane * 8);
            const uint32_t u[4] = {raw.x, raw.y, raw.z, raw.w};
            float acc = 0.0f;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                acc = fmaf(__uint_as_float(u[e] << 16), vp[lane * 8 + e * 2], acc);
                acc = fmaf(__uint_as_float(u[e] & 0xffff0000u), vp[lane * 8 + e * 2 + 1], acc);
            }
            for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
            if (lane == 0) sc[j] = unary[i * kDf2TopK + j] + acc;
        }
        __syncthreads();
        if (tid == 0) {
            int bj = 0;
            float bv = sc[0];
            for (int j = 1; j < kDf2TopK; ++j)
                if (sc[j] > bv) { bv = sc[j]; bj = j; }
            float z = 0.0f;
            for (int j = 0; j < kDf2TopK; ++j) z += __expf(sc[j] - bv);
            out_tok[i] = tok[i * kDf2TopK + bj];
            out_prob[i] = 1.0f / z;
            prev = out_tok[i];
        }
        __syncthreads();
    }
}

__global__ void tap_rows_kernel(const float* __restrict__ R, float* __restrict__ out, int64_t out_stride, int64_t n, int hc) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y;
    if (i >= n) return;
    float s = 0.0f;
    for (int c = 0; c < hc; ++c) s += R[((size_t) r * hc + c) * n + i];
    out[(size_t) r * out_stride + i] = s / (float) hc;
}

__global__ void tap_gr_kernel(const float* __restrict__ R, int64_t R_stride, const float* __restrict__ bo, int64_t bo_stride,
                              const float* __restrict__ inject, int64_t inj_stride, float* __restrict__ out, int64_t out_stride, int64_t n, int hc) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y;
    if (i >= n) return;
    float wsum = 0.0f, rs = 0.0f;
    for (int c = 0; c < hc; ++c) {
        const float x = inject[(size_t) r * inj_stride + c] / (float) hc;
        wsum += 2.0f / (1.0f + expf(-x));
        rs += R[(size_t) r * R_stride + (size_t) c * n + i];
    }
    out[(size_t) r * out_stride + i] = (rs + bo[(size_t) r * bo_stride + i] * wsum) / (float) hc;
}

}  // namespace

void df2_tap_gr(const float* R, int64_t R_stride, const float* bo, int64_t bo_stride, const float* inject, int64_t inj_stride, float* out, int64_t out_stride,
                int rows, int64_t n_embd, int hc, void* stream) {
    tap_gr_kernel<<<dim3((unsigned) ((n_embd + 255) / 256), rows), 256, 0, (cudaStream_t) stream>>>(R, R_stride, bo, bo_stride, inject, inj_stride, out,
                                                                                                  out_stride, n_embd, hc);
    check("tap_gr");
}

void df2_conv(const float* x, const float* dyn, int dyn_stride, const float* base, const float* resid, float* out, int rows, int H, int group,
              void* stream) {
    const int n = rows * H;
    conv_kernel<<<(n + 255) / 256, 256, 0, (cudaStream_t) stream>>>(x, dyn, dyn_stride, base, resid, out, rows, H, group);
    check("conv");
}

void df2_rope_table(const int32_t* pos, int rows, float theta, float* table, void* stream) {
    rope_table_kernel<<<rows, 64, 0, (cudaStream_t) stream>>>(pos, rows, theta, table);
    check("rope_table");
}

void df2_norm_rope(const float* in, int in_stride, const float* gamma, const int32_t* pos, const float* rope_table, float* out, int slot_mod,
                   int rows, int heads, float eps, void* stream) {
    norm_rope_kernel<<<dim3(rows, heads), HD, 0, (cudaStream_t) stream>>>(in, in_stride, gamma, pos, rope_table, out, slot_mod, heads, eps);
    check("norm_rope");
}

void df2_copy_rows(const float* in, float* out, const int32_t* pos, int slot_mod, int rows, int cols, void* stream) {
    const int n = rows * cols;
    copy_rows_kernel<<<(n + 255) / 256, 256, 0, (cudaStream_t) stream>>>(in, out, pos, slot_mod, rows, cols);
    check("copy_rows");
}

void df2_attention(const float* q, const float* kring, const float* vring, const float* kblk, const float* vblk, const int32_t* anchor_pos,
                   int ring_slots, int window, float* part, float* out, void* stream) {
    attn_part_kernel<<<dim3(kDf2Kv, kDf2MaxChunks), 256, 0, (cudaStream_t) stream>>>(q, kring, vring, kblk, vblk, anchor_pos, ring_slots, window, part);
    check("attn_part");
    attn_merge_kernel<<<kDf2Kv * 32, HD, 0, (cudaStream_t) stream>>>(part, out);
    check("attn_merge");
}

void df2_topk(const float* logits, int n, const int32_t* idx_map, int rows, uint32_t* scratch, float* val, int32_t* tok, void* stream) {
    if ((n + kDf2TopBlocks - 1) / kDf2TopBlocks > TK_CHUNK_MAX) throw std::invalid_argument("df2_topk: too many logits per row");
    float* cval = reinterpret_cast<float*>(scratch);
    int32_t* cidx = reinterpret_cast<int32_t*>(scratch + (size_t) rows * kDf2TopBlocks * kDf2TopK);
    topk_part_kernel<<<dim3(rows, kDf2TopBlocks), 256, 0, (cudaStream_t) stream>>>(logits, n, cval, cidx);
    check("topk_part");
    topk_merge_kernel<<<rows, 256, 0, (cudaStream_t) stream>>>(cval, cidx, idx_map, val, tok);
    check("topk_merge");
}

void df2_select(const float* hp, const float* unary, const int32_t* tok, const uint16_t* pred_cb, const uint16_t* succ_cb, int rank, int rows,
                const int32_t* anchor_tok, int32_t* out_tok, float* out_prob, void* stream) {
    if (rank != 256) throw std::invalid_argument("df2_select: the selector rank must be 256");
    select_kernel<<<1, 256, 0, (cudaStream_t) stream>>>(hp, unary, tok, pred_cb, succ_cb, rank, rows, anchor_tok, out_tok, out_prob);
    check("select");
}

void df2_tap_rows(const float* R, float* out, int64_t out_stride, int64_t n_embd, int hc, int rows, void* stream) {
    tap_rows_kernel<<<dim3((unsigned) ((n_embd + 255) / 256), rows), 256, 0, (cudaStream_t) stream>>>(R, out, out_stride, n_embd, hc);
    check("tap_rows");
}

}  // namespace strata::kernels
