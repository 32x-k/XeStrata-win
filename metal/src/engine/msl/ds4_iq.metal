#include <metal_stdlib>
using namespace metal;
// ds4_iq.cu:809  mv_kernel - the I-quant expert matvec.  Same frame as the MXFP4/Q8_0 ones (a warp a row, a lane strides the
// 256-value blocks, sixteen accumulator columns for the token window), with the block decoder chosen per job by b.type:
// 16 IQ2_XXS (66 B), 17 IQ2_XS (74 B), 22 IQ2_S (82 B).  The activations are the quantiser's int8 (xq) with their scales (xd),
// so each sub-group's dot is integer and the two scales fold in at the end - the same arithmetic as the float probes, at a
// quarter of the traffic.
struct Ds4MvJob {
    device const uchar* w;
    ulong ld;
    int type, k, n, ncol;
    device const char* xq;
    device const float* xd;
    device float* y;
    ulong ycs;
    device const float* bias;
    float alpha;
};

kernel void ds4_iq_mv(device const uchar* jobs [[buffer(0)]], uint2 tgp [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const device Ds4MvJob& b = *(const device Ds4MvJob*) (jobs + (size_t) tgp.y * sizeof(Ds4MvJob));
    const uint row = tgp.x * 8u + (tid >> 5);
    if (int(row) >= b.n) return;
    const uint nsb = uint(b.k) / 32u;
    const uint nblk = uint(b.k) / 256u;
    const uint bs = b.type == 16 ? 66u : (b.type == 17 ? 74u : (b.type == 18 ? 98u : 82u));
    const device uchar* w = b.w + (size_t) row * b.ld;
    float acc[16];
    for (uint c = 0; c < 16u; ++c) acc[c] = 0.0f;
    for (uint blk = lane; blk < nblk; blk += 32u) {
        const device uchar* p = w + (size_t) blk * bs;
        const float d = float(as_type<half>(*(const device ushort*) p));
        for (uint ib = 0; ib < 8u; ++ib) {
            if (b.type == 18) {   // IQ3_XXS: d, qs[64] (two grid bytes a 8 values), scales_and_signs[8] words (top nibble the scale)
                const device uchar* qs = p + 2;
                const uint w0 = uint(qs[8 * ib]) | (uint(qs[8 * ib + 1]) << 8) | (uint(qs[8 * ib + 2]) << 16) | (uint(qs[8 * ib + 3]) << 24);
                const uint w1 = uint(qs[8 * ib + 4]) | (uint(qs[8 * ib + 5]) << 8) | (uint(qs[8 * ib + 6]) << 16) | (uint(qs[8 * ib + 7]) << 24);
                const uint aux = uint(qs[64 + 4 * ib]) | (uint(qs[65 + 4 * ib]) << 8) | (uint(qs[66 + 4 * ib]) << 16) | (uint(qs[67 + 4 * ib]) << 24);
                const float db3 = d * (0.5f + float(aux >> 28)) * 0.5f;
                for (uint l = 0; l < 4u; ++l) {
                    const uint wv = l < 2u ? w0 : w1, sh = 16u * (l & 1u);
                    const uint sgn = uint(ksigns_iq2xs[(aux >> (7u * l)) & 127u]);
                    const uint g0 = iq3xxs_grid[(wv >> sh) & 255u], g1 = iq3xxs_grid[(wv >> (sh + 8u)) & 255u];
                    const uint off = (size_t) blk * 256u + ib * 32u + l * 8u;
                    for (uint c = 0; c < 16u; ++c) {
                        if (int(c) >= b.ncol) break;
                        const device char* xq = b.xq + (size_t) c * b.k + off;
                        int sum = 0;
                        for (uint j = 0; j < 4u; ++j) {
                            int v0 = int((g0 >> (8u * j)) & 0xffu), v1 = int((g1 >> (8u * j)) & 0xffu);
                            if ((sgn & (1u << j)) != 0u) v0 = -v0;
                            if ((sgn & (1u << (j + 4u))) != 0u) v1 = -v1;
                            sum += v0 * int(xq[j]) + v1 * int(xq[j + 4u]);
                        }
                        acc[c] = fma(b.xd[(size_t) c * nsb + (off / 32u)] * db3, float(sum), acc[c]);
                    }
                }
                continue;
            }
            float db0, db1;
            uint idx[4];
            uint sg[4];
            if (b.type == 16) {
                const uint aux0 = uint(p[2 + 8 * ib]) | (uint(p[3 + 8 * ib]) << 8) | (uint(p[4 + 8 * ib]) << 16) |
                                  (uint(p[5 + 8 * ib]) << 24);   // (8 bytes a sub-block: grid bytes, then signs and scale)
                const uint aux1 = uint(p[6 + 8 * ib]) | (uint(p[7 + 8 * ib]) << 8) | (uint(p[8 + 8 * ib]) << 16) |
                                  (uint(p[9 + 8 * ib]) << 24);
                const float db = d * (0.5f + float(aux1 >> 28)) * 0.25f;
                db0 = db1 = db;
                for (uint l = 0; l < 4u; ++l) {
                    idx[l] = (aux0 >> (8u * l)) & 0xffu;
                    sg[l] = uint(ksigns_iq2xs[(aux1 >> (7u * l)) & 127u]);
                }
            } else if (b.type == 17) {
                const device ushort* qs = (const device ushort*) (p + 2);
                const device uchar* sc = p + 66;
                db0 = d * (0.5f + float(sc[ib] & 0xfu)) * 0.25f;
                db1 = d * (0.5f + float(sc[ib] >> 4)) * 0.25f;
                for (uint l = 0; l < 4u; ++l) {
                    const uint q = uint(qs[4u * ib + l]);
                    idx[l] = q & 511u;
                    sg[l] = uint(ksigns_iq2xs[(q >> 9) & 127u]);   // the sign byte of the 7 stored bits (bit j set: value j negated)
                }
            } else {
                const device uchar* qs = p + 2;
                const device uchar* sgb = qs + 32;
                const device uchar* qh = p + 66;
                const device uchar* sc = p + 74;
                db0 = d * (0.5f + float(sc[ib] & 0xfu)) * 0.25f;
                db1 = d * (0.5f + float(sc[ib] >> 4)) * 0.25f;
                for (uint l = 0; l < 4u; ++l) {
                    idx[l] = uint(qs[4u * ib + l]) | ((uint(qh[ib]) << (8u - 2u * l)) & 0x300u);
                    sg[l] = uint(sgb[4u * ib + l]);
                }
            }
            for (uint l = 0; l < 4u; ++l) {
                ulong g;
                if (b.type == 16) g = iq2xxs_grid[idx[l]];
                else if (b.type == 17) g = iq2xs_grid[idx[l]];
                else g = iq2s_grid[idx[l] & 1023u];
                const float db = (l < 2u) ? db0 : db1;
                const uint off = (size_t) blk * 256u + ib * 32u + l * 8u;
                for (uint c = 0; c < 16u; ++c) {
                    if (int(c) >= b.ncol) break;
                    const device char* xq = b.xq + (size_t) c * b.k + off;
                    int sum = 0;
                    for (uint j = 0; j < 8u; ++j) {
                        const int code = int((g >> (8u * j)) & 0xffu);
                        const bool neg = b.type == 16 ? ((sg[l] & uint(kmask_iq2xs[j])) != 0u)
                                                      : ((sg[l] & (1u << j)) != 0u);
                        sum += (neg ? -code : code) * int(xq[j]);
                    }
                    acc[c] = fma(b.xd[(size_t) c * nsb + (off / 32u)] * db, float(sum), acc[c]);
                }
            }
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

// Q5_K / Q6_K (the dense matrices' K-quants; ggml blocks of 176 / 210 bytes): the same warp a row frame, a lane a
// 256-value block, each value's code times its activation's int8.  Q5_K's weight is d*sc*q - dmin*m (both parts of the
// 32-value group's scales), Q6_K's d*sc*(q - 32).
static inline void ds4_k_scale(const device uchar* q, int j, thread int& d, thread int& m) {
    if (j < 4) {
        d = int(q[j]) & 63;
        m = int(q[j + 4]) & 63;
    } else {
        d = (int(q[j + 4]) & 0xf) | ((int(q[j - 4]) >> 6) << 4);
        m = (int(q[j + 4]) >> 4) | ((int(q[j]) >> 6) << 4);
    }
}

kernel void ds4_iq_mv_k(device const uchar* jobs [[buffer(0)]], uint2 tgp [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const device Ds4MvJob& b = *(const device Ds4MvJob*) (jobs + (size_t) tgp.y * sizeof(Ds4MvJob));
    const uint row = tgp.x * 8u + (tid >> 5);
    if (int(row) >= b.n) return;
    const uint nsb = uint(b.k) / 32u;
    const uint nblk = uint(b.k) / 256u;
    const uint bs = b.type == 13 ? 176u : 210u;
    const device uchar* w = b.w + (size_t) row * b.ld;
    float acc[16];
    for (uint c = 0; c < 16u; ++c) acc[c] = 0.0f;
    for (uint blk = lane; blk < nblk; blk += 32u) {
        const device uchar* p = w + (size_t) blk * bs;
        if (b.type == 13) {
            const float d = float(as_type<half>(*(const device ushort*) p));
            const float dmin = float(as_type<half>(*(const device ushort*) (p + 2)));
            const device uchar* scales = p + 4;
            const device uchar* qh = p + 16;
            const device uchar* qs = p + 48;
            for (uint ib = 0; ib < 8u; ++ib) {
                int sc, mm;
                ds4_k_scale(scales, int(ib), sc, mm);
                const device uchar* qsp = qs + 32u * (ib >> 1);
                const uint sh = 4u * (ib & 1u);
                for (uint t = 0; t < 8u; ++t) {
                    const uint q4 = (*(const device uint*) (qsp + 4u * t) >> sh) & 0x0f0f0f0fu;
                    const uint h4 = (*(const device uint*) (qh + 4u * t) >> ib) & 0x01010101u;
                    const uint q = q4 | (h4 << 4);
                    const uint off = blk * 256u + ib * 32u + t * 4u;
                    for (uint c = 0; c < 16u; ++c) {
                        if (int(c) >= b.ncol) break;
                        const device char* xq = b.xq + (size_t) c * b.k + off;
                        int sumq = 0, sumx = 0;
                        for (uint j = 0; j < 4u; ++j) {
                            const int code = int((q >> (8u * j)) & 0xffu);
                            const int xv = int(xq[j]);
                            sumq += code * xv;
                            sumx += xv;
                        }
                        const float dot = d * float(sc) * float(sumq) - dmin * float(mm) * float(sumx);
                        acc[c] = fma(b.xd[(size_t) c * nsb + (off / 32u)], dot, acc[c]);
                    }
                }
            }
        } else {
            const float d = float(as_type<half>(*(const device ushort*) (p + 208)));
            const device uchar* ql = p;
            const device uchar* qh = p + 128;
            const device uchar* scl = p + 192;
            for (uint ib = 0; ib < 8u; ++ib) {
                const uint h = ib >> 2, quarter = ib & 3u;
                const device uchar* qlp = ql + 64u * h + ((quarter & 1u) ? 32u : 0u);
                const device uchar* qhp = qh + 32u * h;
                const uint sh = 4u * (quarter >> 1);
                const int s0 = int(((const device char*) scl)[8u * h + 2u * quarter]);
                const int s1 = int(((const device char*) scl)[8u * h + 2u * quarter + 1u]);
                for (uint t = 0; t < 8u; ++t) {
                    const uint x = ((*(const device uint*) (qlp + 4u * t) >> sh) & 0x0f0f0f0fu) |
                                   (((*(const device uint*) (qhp + 4u * t) >> (2u * quarter)) & 0x03030303u) << 4);
                    const uint off = blk * 256u + ib * 32u + t * 4u;
                    const float s = float(t < 4u ? s0 : s1);
                    for (uint c = 0; c < 16u; ++c) {
                        if (int(c) >= b.ncol) break;
                        const device char* xq = b.xq + (size_t) c * b.k + off;
                        int sumq = 0;
                        for (uint j = 0; j < 4u; ++j) {
                            const int code = int((x >> (8u * j)) & 0xffu) - 32;
                            sumq += code * int(xq[j]);
                        }
                        acc[c] = fma(b.xd[(size_t) c * nsb + (off / 32u)] * d * s, float(sumq), acc[c]);
                    }
                }
            }
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

// ---------------------------------------------------------------- ds4_iq.cu:536 gate_up_kernel / :588 down_kernel
//
// The prompt / window path's expert GEMM (ds4_iq.cu's launch_gu / launch_dn): a threadgroup of 256 = 8 warps, a warp
// RW rows of the gate and of the up (down: RW rows of the down), a lane 32 / RW sub-blocks of the row, and the group's
// entries NE at a pass so that a sub-block the weights decode once serves them all.  Grid (rows / (8 RW), groups).
//
// The group's blob and its entries are the CUDA kernel's `group` (ds4_iq.cu:512): a plan group (grp_ptr null) is one
// expert whose entry is the group's own index and whose activation row is 0, a window group's entries come from
// grp_ent[grp_start[g] .. grp_start[g + 1]) and an entry e's activation row is e >> 3.
struct Ds4IqJob {
    device const ulong* plan_ptr;   // plan_ptr[g]: the expert's blob (0: the group is skipped)
    device const ulong* grp_ptr;    // grp_ptr[g]: a window set's blob
    device const int* grp_start;    // a set's first entry (n_groups + 1 of them)
    device const int* grp_ent;      // the entries (token * 8 + pick)
    device const int* n_groups;
    device const char* xq;          // the int8 activations of a row (n_embd an entry)
    device const float* xd;         // their scales (n_embd / 32 a row)
    device const char* hq;          // down's int8 input (n_ff an entry)
    device const float* hd;
    device float* h;                // [entries][n_ff]: the clamped swiglu
    device float* part;             // [entries][n_embd]: each entry's down output
    long tr0, tr1, tr2;             // the roles' offsets in a blob
    int n_embd, n_ff;
    int type;                       // 16 IQ2_XXS, 17 IQ2_XS, 22 IQ2_S (ds4_iq.cuh's kIQ2_*)
    float limit;                    // swiglu_limit (0: none)
};

// a 32-value sub-block as 32 int8 codes (8 packed ints, values 4 w .. 4 w + 3 in word w) and the two halves' scales
// (ggml's d (2 s + 1) / 8, the .cu's Sub.d * Sub.s0 in one F32: the codes of the second half are scaled by s1)
struct Ds4IqSub {
    int q[8];
    float s0, s1;
};

// the block bytes of the types these kernels take: 16 IQ2_XXS, 17 IQ2_XS, 18 IQ3_XXS, 22 IQ2_S (ds4_iq.cu's Ty<T>::B)
static inline uint ds4_iq_bs(int type) {
    return type == 16 ? 66u : type == 17 ? 74u : type == 18 ? 98u : 82u;
}

// ds4_iq.cu:139 decode<T>: the block's 8 sub-blocks, kb's is the (kb & 7)'th; the grids are ggml-common.h's
static inline Ds4IqSub ds4_iq_sub(device const uchar* row, uint bs, int kb, int type) {
    Ds4IqSub s;
    const int ib = kb & 7;
    const device uchar* p = row + (size_t) (kb >> 3) * bs;
    const float d = float(as_type<half>(*(const device ushort*) p));
    uint idx[4], sm[4];
    float db0, db1;
    if (type == 18) {   // IQ3_XXS: d, qs[64] grid bytes (4 a sub-block), scales[8] (the top nibble the group's s)
        const device uchar* qs = p + 2;
        const uint w0 = uint(qs[8 * ib]) | (uint(qs[8 * ib + 1]) << 8) | (uint(qs[8 * ib + 2]) << 16) |
                        (uint(qs[8 * ib + 3]) << 24);
        const uint w1 = uint(qs[8 * ib + 4]) | (uint(qs[8 * ib + 5]) << 8) | (uint(qs[8 * ib + 6]) << 16) |
                        (uint(qs[8 * ib + 7]) << 24);
        const uint aux = uint(qs[64 + 4 * ib]) | (uint(qs[65 + 4 * ib]) << 8) | (uint(qs[66 + 4 * ib]) << 16) |
                         (uint(qs[67 + 4 * ib]) << 24);
        for (uint l = 0; l < 4u; ++l) {
            const uint w = l < 2u ? w0 : w1, sh = 16u * (l & 1u);
            const uint sg = uint(ksigns_iq2xs[(aux >> (7u * l)) & 127u]);
            const uint g0 = iq3xxs_grid[(w >> sh) & 255u], g1 = iq3xxs_grid[(w >> (sh + 8u)) & 255u];
            int a0 = 0, a1 = 0;
            for (uint j = 0; j < 4u; ++j) {
                int v0 = int((g0 >> (8u * j)) & 0xffu), v1 = int((g1 >> (8u * j)) & 0xffu);
                if ((sg & (1u << j)) != 0u) v0 = -v0;
                if ((sg & (1u << (j + 4u))) != 0u) v1 = -v1;
                a0 |= (v0 & 0xff) << int(8u * j);
                a1 |= (v1 & 0xff) << int(8u * j);
            }
            s.q[2u * l] = a0;
            s.q[2u * l + 1u] = a1;
        }
        const float ds = d * (0.5f + float(aux >> 28)) * 0.5f;   // ggml's db: d (0.5 + s) 0.5
        s.s0 = ds;
        s.s1 = ds;
        return s;
    }
    if (type == 16) {   // d, qs[8] i16 (4 grid bytes, then 4 sign/scale bytes), scales in the top nibbles
        const uint aux0 = uint(p[2 + 8 * ib]) | (uint(p[3 + 8 * ib]) << 8) | (uint(p[4 + 8 * ib]) << 16) |
                          (uint(p[5 + 8 * ib]) << 24);
        const uint aux1 = uint(p[6 + 8 * ib]) | (uint(p[7 + 8 * ib]) << 8) | (uint(p[8 + 8 * ib]) << 16) |
                          (uint(p[9 + 8 * ib]) << 24);
        const float db = d * (0.5f + float(aux1 >> 28)) * 0.25f;
        db0 = db1 = db;
        for (uint l = 0; l < 4u; ++l) {
            idx[l] = (aux0 >> (8u * l)) & 0xffu;
            sm[l] = uint(ksigns_iq2xs[(aux1 >> (7u * l)) & 127u]);
        }
    } else if (type == 17) {   // d, qs[16] i16 (9-bit grid, 7 sign bits), scales[8] (two 4-bit)
        const device ushort* qs = (const device ushort*) (p + 2);
        const device uchar* sc = p + 66;
        db0 = d * (0.5f + float(sc[ib] & 0x0fu)) * 0.25f;
        db1 = d * (0.5f + float(sc[ib] >> 4)) * 0.25f;
        for (uint l = 0; l < 4u; ++l) {
            const uint q = uint(qs[4u * uint(ib) + l]);
            idx[l] = q & 511u;
            sm[l] = uint(ksigns_iq2xs[(q >> 9) & 127u]);
        }
    } else {   // IQ2_S: d, qs[32] (8-bit grid), signs[32], qh[8] (2 bits more), scales[8]
        const device uchar* qs = p + 2;
        const device uchar* sgb = qs + 32;
        const device uchar* qh = p + 66;
        const device uchar* sc = p + 74;
        db0 = d * (0.5f + float(sc[ib] & 0x0fu)) * 0.25f;
        db1 = d * (0.5f + float(sc[ib] >> 4)) * 0.25f;
        for (uint l = 0; l < 4u; ++l) {
            idx[l] = uint(qs[4u * uint(ib) + l]) | ((uint(qh[ib]) << (8u - 2u * l)) & 0x300u);
            sm[l] = uint(sgb[4u * uint(ib) + l]);
        }
    }
    for (uint l = 0; l < 4u; ++l) {
        const ulong g = type == 16 ? iq2xxs_grid[idx[l]] : (type == 17 ? iq2xs_grid[idx[l]] : iq2s_grid[idx[l] & 1023u]);
        int w0 = 0, w1 = 0;
        for (uint j = 0; j < 4u; ++j) {
            int v0 = int((g >> (8u * j)) & 0xffu), v1 = int((g >> (8u * (j + 4u))) & 0xffu);
            if ((sm[l] & (1u << j)) != 0u) v0 = -v0;
            if ((sm[l] & (1u << (j + 4u))) != 0u) v1 = -v1;
            w0 |= (v0 & 0xff) << int(8u * j);
            w1 |= (v1 & 0xff) << int(8u * j);
        }
        s.q[2u * l] = w0;
        s.q[2u * l + 1u] = w1;
    }
    s.s0 = db0;
    s.s1 = db1;
    return s;
}

// 4 int8 codes against the 4 int8 activations packed in x (ds4_iq.cu:350's __dp4a: Metal has no such instruction)
static inline int ds4_iq_dot4(int q, int x) {
    const char4 a = as_type<char4>(q), b = as_type<char4>(x);
    return int(a.x) * int(b.x) + int(a.y) * int(b.y) + int(a.z) * int(b.z) + int(a.w) * int(b.w);
}

// ds4_iq.cu:349 sub_dot: the sub-block against its 32 int8 activations (8 packed ints), the scales in F32
static inline float ds4_iq_dot(Ds4IqSub s, device const int* xp) {
    const int lo = ds4_iq_dot4(s.q[0], xp[0]) + ds4_iq_dot4(s.q[1], xp[1]) + ds4_iq_dot4(s.q[2], xp[2]) +
                   ds4_iq_dot4(s.q[3], xp[3]);
    const int hi = ds4_iq_dot4(s.q[4], xp[4]) + ds4_iq_dot4(s.q[5], xp[5]) + ds4_iq_dot4(s.q[6], xp[6]) +
                   ds4_iq_dot4(s.q[7], xp[7]);
    return s.s0 * float(lo) + s.s1 * float(hi);
}

// ds4_iq.cu:502 lanes_sum: the row's sum over the 32 / RW lanes that share it
template <int RW>
static inline float ds4_iq_lsum(float v) {
    for (int o = 16 / RW; o > 0; o >>= 1) v += simd_shuffle_xor(v, ushort(o));
    return v;
}

// ds4_iq.cu:497 swiglu: the limit's clamps, then g / (1 + e^-g) * u
static inline float ds4_iq_swiglu(float gv, float uv, float limit) {
    if (limit > 0.0f) {
        gv = min(gv, limit);
        uv = min(max(uv, -limit), limit);
    }
    return (gv / (1.0f + exp(-gv))) * uv;
}

// the group's blob and its entry range (ds4_iq.cu:512 group); false: the threadgroup has nothing to do
static inline bool ds4_iq_group(constant Ds4IqJob& j, uint g, thread ulong& blob, thread int& m0, thread int& m1) {
    if (j.grp_ptr == nullptr) {
        const ulong p = j.plan_ptr[g];
        if (p == 0ul) return false;
        blob = p;
        m0 = int(g);
        m1 = m0 + 1;
        return true;
    }
    if (int(g) >= *j.n_groups) return false;
    blob = j.grp_ptr[g];
    m0 = j.grp_start[g];
    m1 = j.grp_start[g + 1];
    return true;
}

// gate_up_kernel<T, NE> (kGuRW = 1: a warp a row, a lane a sub-block): h = the clamped swiglu of the gate and the up
kernel void ds4_iq_gu(constant Ds4IqJob& j [[buffer(0)]], uint2 tgp [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    ulong blob;
    int m0, m1;
    if (!ds4_iq_group(j, tgp.y, blob, m0, m1)) return;
    const int row = int(tgp.x) * 8 + int(tid >> 5);
    if (row >= j.n_ff) return;
    const uint bs = ds4_iq_bs(j.type);
    const uint nsb = uint(j.n_embd) / 32u;
    const size_t rb = (size_t) (j.n_embd / 256) * (size_t) bs;
    const device uchar* g = (const device uchar*) blob + (size_t) j.tr0 + (size_t) row * rb;
    const device uchar* u = (const device uchar*) blob + (size_t) j.tr1 + (size_t) row * rb;
    for (int mm = m0; mm < m1; mm += 4) {
        const int ne = min(4, m1 - mm);
        int es[4], xr[4];
        for (int i = 0; i < 4; ++i) {
            if (i < ne) {
                const int e = j.grp_ptr == nullptr ? (mm + i) : j.grp_ent[mm + i];
                es[i] = e;
                xr[i] = j.grp_ptr == nullptr ? 0 : (e >> 3);
            }
        }
        float ag[4] = {0.0f, 0.0f, 0.0f, 0.0f}, au[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint kb = uint(lane); kb < nsb; kb += 32u) {
            const Ds4IqSub sg = ds4_iq_sub(g, bs, int(kb), j.type);
            const Ds4IqSub su = ds4_iq_sub(u, bs, int(kb), j.type);
            for (int i = 0; i < 4; ++i) {
                if (i < ne) {
                    const device int* xp = (const device int*) (j.xq + (size_t) xr[i] * (size_t) j.n_embd +
                                                               (size_t) kb * 32u);
                    const float dx = j.xd[(size_t) xr[i] * nsb + kb];
                    ag[i] += dx * ds4_iq_dot(sg, xp);
                    au[i] += dx * ds4_iq_dot(su, xp);
                }
            }
        }
        for (int i = 0; i < 4; ++i) {
            if (i < ne) {
                const float gv = ds4_iq_lsum<1>(ag[i]), uv = ds4_iq_lsum<1>(au[i]);
                if (lane == 0u) j.h[(size_t) es[i] * (size_t) j.n_ff + (size_t) row] = ds4_iq_swiglu(gv, uv, j.limit);
            }
        }
    }
}

// down_kernel<T, NE> (kDnRW = 2: a warp two rows, 16 lanes a row): part = the down against the int8 h
kernel void ds4_iq_dn(constant Ds4IqJob& j [[buffer(0)]], uint2 tgp [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    ulong blob;
    int m0, m1;
    if (!ds4_iq_group(j, tgp.y, blob, m0, m1)) return;
    const int row = (int(tgp.x) * 8 + int(tid >> 5)) * 2 + int(lane) / 16;
    if (row >= j.n_embd) return;
    const uint kl = uint(lane) % 16u;
    const uint bs = ds4_iq_bs(j.type);
    const uint nsb = uint(j.n_ff) / 32u;
    const size_t rb = (size_t) (j.n_ff / 256) * (size_t) bs;
    const device uchar* w = (const device uchar*) blob + (size_t) j.tr2 + (size_t) row * rb;
    for (int mm = m0; mm < m1; mm += 4) {
        const int ne = min(4, m1 - mm);
        int es[4];
        for (int i = 0; i < 4; ++i)
            if (i < ne) es[i] = j.grp_ptr == nullptr ? (mm + i) : j.grp_ent[mm + i];
        float ad[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint kb = kl; kb < nsb; kb += 16u) {
            const Ds4IqSub sd = ds4_iq_sub(w, bs, int(kb), j.type);
            for (int i = 0; i < 4; ++i) {
                if (i < ne) {
                    const device int* hp = (const device int*) (j.hq + (size_t) es[i] * (size_t) j.n_ff +
                                                               (size_t) kb * 32u);
                    ad[i] += j.hd[(size_t) es[i] * nsb + kb] * ds4_iq_dot(sd, hp);
                }
            }
        }
        for (int i = 0; i < 4; ++i) {
            if (i < ne) {
                const float v = ds4_iq_lsum<2>(ad[i]);
                if (kl == 0u) j.part[(size_t) es[i] * (size_t) j.n_embd + (size_t) row] = v;
            }
        }
    }
}

// ---------------------------------------------------------------- ds4_gguf.cu:454 mx_gate_up_kernel / :519 mx_down_kernel
//
// The MXFP4 expert GEMM (ds4_gguf.cu's moe_gate_up / moe_down / win_experts): the same frame as ds4_iq_gu / ds4_iq_dn
// above - a threadgroup of 256 is 8 warps, a warp a row of the gate and of the up (down: a row of the down), a lane
// strides the 32-value blocks of the row, and the group's entries go sixteen at a pass so that a block the weights decode
// once serves them all.  A block is 17 bytes (kMxBytes): the E8M0 scale byte, then 16 bytes whose low nibble is value j and
// whose high nibble is value j + 16, each an index into the codebook {0, 1, 2, 3, 4, 6, 8, 12} and its negatives - the same
// decode as ds4_init.metal's ds4_mx_block / ds4_e8m0 (ds4_gguf.cu:143 / :154), kept as int8 codes here because the
// activations are the quantiser's int8, so the dot is integer (ds4_gguf.cu:225 mx_dot) and the two scales fold in at the
// end.
//
// The group's blob and its entries are read as ds4_iq_group above reads them (ds4_gguf.cu:429 mx_group): a plan group
// (grp_ptr null) is one expert whose entry is the group's own index and whose activation row is 0, a window group's
// entries come from grp_ent[grp_start[g] .. grp_start[g + 1]) and an entry e's activation row is e >> 3.
struct Ds4MxJob {
    device const ulong* plan_ptr;   // plan_ptr[g]: the expert's blob (0: the group is skipped)
    device const ulong* grp_ptr;    // grp_ptr[g]: a window set's blob
    device const int* grp_start;    // a set's first entry (n_groups + 1 of them)
    device const int* grp_ent;      // the entries (token * 8 + pick)
    device const int* n_groups;
    device const char* xq;          // gate / up's int8 activations (n_embd an entry)
    device const float* xd;         // their scales (n_embd / 32 a row)
    device const char* hq;          // down's int8 input (n_ff an entry)
    device const float* hd;         // its scales
    device float* h;                // [entries][n_ff]: the clamped swiglu
    device float* part;             // [entries][n_embd]: each entry's down output
    long tr0, tr1, tr2;             // the roles' offsets in a blob
    int n_embd, n_ff;
    float limit;                    // swiglu_limit (0: none)
};

// the codebook of a nibble, in index order (ds4_init.metal's lut_init: {0,1,2,3,4,6,8,12}, then its negatives)
constant char ds4_mx_code[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

// 2^(e - 128) (ds4_init.metal:489 ds4_e8m0, ggml's GGML_E8M0_TO_FP32_HALF)
static inline float ds4_mx_e8m0(uint e) {
    return as_type<float>(e < 2u ? (0x00200000u << e) : ((e - 1u) << 23));
}

// a block's 16 code bytes as 8 int8 words: word j the low nibbles of bytes 4 j .. (values 4 j ..), word 4 + j their high
// nibbles (values 16 + 4 j ..) - the pairs ds4_gguf.cu:225 mx_dot reads xp[j] and xp[j + 4] with
static inline void ds4_mx_words(device const uchar* p, thread int* q) {
    for (uint j = 0; j < 4u; ++j) {
        uint lo = 0u, hi = 0u;
        for (uint b = 0; b < 4u; ++b) {
            const uint v = uint(p[1u + 4u * j + b]);
            lo |= uint(uchar(ds4_mx_code[v & 15u])) << (8u * b);
            hi |= uint(uchar(ds4_mx_code[v >> 4u])) << (8u * b);
        }
        q[j] = int(lo);
        q[4u + j] = int(hi);
    }
}

// a block's 32 int8 codes against its 32 int8 activations (8 packed words, 4 values a word)
static inline int ds4_mx_dot(thread const int* q, device const int* xp) {
    int s = 0;
    for (uint t = 0; t < 8u; ++t) s += ds4_iq_dot4(q[t], xp[t]);
    return s;
}

// the group's blob and its entry range (ds4_iq_group's MX form, ds4_gguf.cu:429 mx_group); false: nothing to do
static inline bool ds4_mx_group(constant Ds4MxJob& j, uint g, thread ulong& blob, thread int& m0, thread int& m1) {
    if (j.grp_ptr == nullptr) {
        const ulong p = j.plan_ptr[g];
        if (p == 0ul) return false;
        blob = p;
        m0 = int(g);
        m1 = m0 + 1;
        return true;
    }
    if (int(g) >= *j.n_groups) return false;
    blob = j.grp_ptr[g];
    m0 = j.grp_start[g];
    m1 = j.grp_start[g + 1];
    return true;
}

// mx_gate_up_kernel<NE> (a warp a row of the gate and of the up, a lane a block a step): h = the clamped swiglu
kernel void ds4_mx_gu(constant Ds4MxJob& j [[buffer(0)]], uint2 tgp [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    ulong blob;
    int m0, m1;
    if (!ds4_mx_group(j, tgp.y, blob, m0, m1)) return;
    const int row = int(tgp.x) * 8 + int(tid >> 5);
    if (row >= j.n_ff) return;
    const uint nblk = uint(j.n_embd) / 32u;
    const size_t rb = (size_t) nblk * 17u;
    const device uchar* g = (const device uchar*) blob + (size_t) j.tr0 + (size_t) row * rb;
    const device uchar* u = (const device uchar*) blob + (size_t) j.tr1 + (size_t) row * rb;
    for (int mm = m0; mm < m1; mm += 16) {
        const int ne = min(16, m1 - mm);
        int es[16], xr[16];
        for (int i = 0; i < 16; ++i) {
            if (i < ne) {
                const int e = j.grp_ptr == nullptr ? (mm + i) : j.grp_ent[mm + i];
                es[i] = e;
                xr[i] = j.grp_ptr == nullptr ? 0 : (e >> 3);
            }
        }
        float ag[16], au[16];
        for (int i = 0; i < 16; ++i) {
            ag[i] = 0.0f;
            au[i] = 0.0f;
        }
        for (uint b = uint(lane); b < nblk; b += 32u) {
            const device uchar* pg = g + (size_t) b * 17u;
            const device uchar* pu = u + (size_t) b * 17u;
            int wg[8], wu[8];
            ds4_mx_words(pg, wg);
            ds4_mx_words(pu, wu);
            const float dg = ds4_mx_e8m0(uint(pg[0])), du = ds4_mx_e8m0(uint(pu[0]));
            for (int i = 0; i < 16; ++i) {
                if (i < ne) {
                    const device int* xp =
                        (const device int*) (j.xq + (size_t) xr[i] * (size_t) j.n_embd + (size_t) b * 32u);
                    const float dx = j.xd[(size_t) xr[i] * nblk + b];
                    ag[i] = fma(dg * dx, float(ds4_mx_dot(wg, xp)), ag[i]);
                    au[i] = fma(du * dx, float(ds4_mx_dot(wu, xp)), au[i]);
                }
            }
        }
        for (int i = 0; i < 16; ++i) {
            if (i < ne) {
                const float gv = ds4_iq_lsum<1>(ag[i]), uv = ds4_iq_lsum<1>(au[i]);
                if (lane == 0u) j.h[(size_t) es[i] * (size_t) j.n_ff + (size_t) row] = ds4_iq_swiglu(gv, uv, j.limit);
            }
        }
    }
}

// mx_down_kernel<NE> (a warp a row of the down, a lane a block a step): part = the down against the int8 h
kernel void ds4_mx_dn(constant Ds4MxJob& j [[buffer(0)]], uint2 tgp [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    ulong blob;
    int m0, m1;
    if (!ds4_mx_group(j, tgp.y, blob, m0, m1)) return;
    const int row = int(tgp.x) * 8 + int(tid >> 5);
    if (row >= j.n_embd) return;
    const uint nblk = uint(j.n_ff) / 32u;
    const size_t rb = (size_t) nblk * 17u;
    const device uchar* w = (const device uchar*) blob + (size_t) j.tr2 + (size_t) row * rb;
    for (int mm = m0; mm < m1; mm += 16) {
        const int ne = min(16, m1 - mm);
        int es[16];
        for (int i = 0; i < 16; ++i)
            if (i < ne) es[i] = j.grp_ptr == nullptr ? (mm + i) : j.grp_ent[mm + i];
        float ad[16];
        for (int i = 0; i < 16; ++i) ad[i] = 0.0f;
        for (uint b = uint(lane); b < nblk; b += 32u) {
            const device uchar* pw = w + (size_t) b * 17u;
            int ww[8];
            ds4_mx_words(pw, ww);
            const float dd = ds4_mx_e8m0(uint(pw[0]));
            for (int i = 0; i < 16; ++i) {
                if (i < ne) {
                    const device int* hp =
                        (const device int*) (j.hq + (size_t) es[i] * (size_t) j.n_ff + (size_t) b * 32u);
                    ad[i] = fma(dd * j.hd[(size_t) es[i] * nblk + b], float(ds4_mx_dot(ww, hp)), ad[i]);
                }
            }
        }
        for (int i = 0; i < 16; ++i) {
            if (i < ne) {
                const float v = ds4_iq_lsum<1>(ad[i]);
                if (lane == 0u) j.part[(size_t) es[i] * (size_t) j.n_embd + (size_t) row] = v;
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// One-activation matvecs (the decode step): every lane of the row's simdgroup works, on a 32-value sub-block (IQ2_XXS) or
// a 64-value pair (Q5_K), where the 16-column kernels above give a whole 256-value block to a lane (a k = 1280 row used 5
// lanes of 32) and read it a byte at a time.  Same results up to the order of the float sums.

kernel void ds4_q5k_mv1(device const uchar* jobs [[buffer(0)]], uint2 tgp [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const device Ds4MvJob& b = *(const device Ds4MvJob*) (jobs + (size_t) tgp.y * sizeof(Ds4MvJob));
    const uint row = tgp.x * 8u + (tid >> 5);
    if (int(row) >= b.n) return;
    const uint nblk = uint(b.k) / 256u;
    const device uchar* w = b.w + (size_t) row * b.ld;
    const uint j = lane & 3u;          // this lane's pair of sub-blocks: 2j and 2j+1 (one 32-byte run of qs, both nibbles)
    const uint bsel = lane >> 2;       // eight blocks a pass
    float acc = 0.0f;
    for (uint blk = bsel; blk < nblk; blk += 8u) {
        const device uchar* p = w + (size_t) blk * 176u;
        const float d = float(as_type<half>(*(const device ushort*) p));
        const float dmin = float(as_type<half>(*(const device ushort*) (p + 2)));
        int sc0, mm0, sc1, mm1;
        ds4_k_scale(p + 4, int(2u * j), sc0, mm0);
        ds4_k_scale(p + 4, int(2u * j + 1u), sc1, mm1);
        const device uint4* qs4 = (const device uint4*) (p + 48u + 32u * j);
        const device uint4* qh4 = (const device uint4*) (p + 16);
        const device uint4* xq4 = (const device uint4*) (b.xq + (size_t) blk * 256u + 64u * j);
        const uint4 qa = qs4[0], qb = qs4[1], ha = qh4[0], hb = qh4[1];
        const uint4 xa = xq4[0], xb = xq4[1], xc = xq4[2], xd4 = xq4[3];   // sub-block 2j: xa xb; 2j+1: xc xd4
        const uint qsw[8] = {qa.x, qa.y, qa.z, qa.w, qb.x, qb.y, qb.z, qb.w};
        const uint qhw[8] = {ha.x, ha.y, ha.z, ha.w, hb.x, hb.y, hb.z, hb.w};
        const uint xlo[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
        const uint xhi[8] = {xc.x, xc.y, xc.z, xc.w, xd4.x, xd4.y, xd4.z, xd4.w};
        int sq0 = 0, sx0 = 0, sq1 = 0, sx1 = 0;
        for (uint t = 0; t < 8u; ++t) {
            const uint q0 = (qsw[t] & 0x0f0f0f0fu) | (((qhw[t] >> (2u * j)) & 0x01010101u) << 4);
            const uint q1 = ((qsw[t] >> 4) & 0x0f0f0f0fu) | (((qhw[t] >> (2u * j + 1u)) & 0x01010101u) << 4);
            for (uint i = 0; i < 4u; ++i) {
                const int x0 = int(char(xlo[t] >> (8u * i))), x1 = int(char(xhi[t] >> (8u * i)));
                sq0 += int((q0 >> (8u * i)) & 0xffu) * x0;
                sq1 += int((q1 >> (8u * i)) & 0xffu) * x1;
                sx0 += x0;
                sx1 += x1;
            }
        }
        acc = fma(b.xd[blk * 8u + 2u * j], d * float(sc0) * float(sq0) - dmin * float(mm0) * float(sx0), acc);
        acc = fma(b.xd[blk * 8u + 2u * j + 1u], d * float(sc1) * float(sq1) - dmin * float(mm1) * float(sx1), acc);
    }
    const float v = simd_sum(acc);
    if (lane == 0u) b.y[row] = b.alpha * v + (b.bias != nullptr ? b.bias[row] : 0.0f);
}

kernel void ds4_iq2xxs_mv1(device const uchar* jobs [[buffer(0)]], uint2 tgp [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const device Ds4MvJob& b = *(const device Ds4MvJob*) (jobs + (size_t) tgp.y * sizeof(Ds4MvJob));
    const uint row = tgp.x * 8u + (tid >> 5);
    if (int(row) >= b.n) return;
    const uint nblk = uint(b.k) / 256u;
    const device uchar* w = b.w + (size_t) row * b.ld;
    const uint ib = lane & 7u;         // this lane's 32-value sub-block
    const uint bsel = lane >> 3;       // four blocks a pass
    float acc = 0.0f;
    for (uint blk = bsel; blk < nblk; blk += 4u) {
        const device uchar* p = w + (size_t) blk * 66u;
        const float d = float(as_type<half>(*(const device ushort*) p));
        const device ushort* us = (const device ushort*) (p + 2u + 8u * ib);   // (2-byte aligned: four 16-bit loads)
        const uint aux0 = uint(us[0]) | (uint(us[1]) << 16);
        const uint aux1 = uint(us[2]) | (uint(us[3]) << 16);
        const float db = d * (0.5f + float(aux1 >> 28)) * 0.25f;
        const device uint4* xq4 = (const device uint4*) (b.xq + (size_t) blk * 256u + 32u * ib);
        const uint4 xa = xq4[0], xb = xq4[1];
        const uint xw[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
        int sum = 0;
        for (uint l = 0; l < 4u; ++l) {
            const ulong g = iq2xxs_grid[(aux0 >> (8u * l)) & 0xffu];
            const uint sg = uint(ksigns_iq2xs[(aux1 >> (7u * l)) & 127u]);
            for (uint h = 0; h < 2u; ++h) {
                const uint gw = uint(g >> (32u * h));
                const uint x = xw[2u * l + h];
                for (uint i = 0; i < 4u; ++i) {
                    const int code = int((gw >> (8u * i)) & 0xffu);
                    const int xv = int(char(x >> (8u * i)));
                    sum += ((sg >> (4u * h + i)) & 1u) != 0u ? -code * xv : code * xv;
                }
            }
        }
        acc = fma(b.xd[blk * 8u + ib] * db, float(sum), acc);
    }
    const float v = simd_sum(acc);
    if (lane == 0u) b.y[row] = b.alpha * v + (b.bias != nullptr ? b.bias[row] : 0.0f);
}

// The other low-bit expert types, one activation (see ds4_iq2xxs_mv1): a lane a 32-value sub-block, four blocks a pass.

kernel void ds4_iq2xs_mv1(device const uchar* jobs [[buffer(0)]], uint2 tgp [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const device Ds4MvJob& b = *(const device Ds4MvJob*) (jobs + (size_t) tgp.y * sizeof(Ds4MvJob));
    const uint row = tgp.x * 8u + (tid >> 5);
    if (int(row) >= b.n) return;
    const uint nblk = uint(b.k) / 256u;
    const device uchar* w = b.w + (size_t) row * b.ld;
    const uint ib = lane & 7u;
    const uint bsel = lane >> 3;
    float acc = 0.0f;
    for (uint blk = bsel; blk < nblk; blk += 4u) {
        const device uchar* p = w + (size_t) blk * 74u;
        const float d = float(as_type<half>(*(const device ushort*) p));
        const device ushort* qs = (const device ushort*) (p + 2u + 8u * ib);
        const uint sc = uint(p[66u + ib]);
        const float db0 = d * (0.5f + float(sc & 0xfu)) * 0.25f, db1 = d * (0.5f + float(sc >> 4)) * 0.25f;
        const device uint4* xq4 = (const device uint4*) (b.xq + (size_t) blk * 256u + 32u * ib);
        const uint4 xa = xq4[0], xb = xq4[1];
        const uint xw[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
        int s0 = 0, s1 = 0;
        for (uint l = 0; l < 4u; ++l) {
            const uint q = uint(qs[l]);
            const ulong g = iq2xs_grid[q & 511u];
            const uint sg = uint(ksigns_iq2xs[(q >> 9) & 127u]);
            int sum = 0;
            for (uint h = 0; h < 2u; ++h) {
                const uint gw = uint(g >> (32u * h));
                const uint x = xw[2u * l + h];
                for (uint i = 0; i < 4u; ++i) {
                    const int code = int((gw >> (8u * i)) & 0xffu);
                    const int xv = int(char(x >> (8u * i)));
                    sum += ((sg >> (4u * h + i)) & 1u) != 0u ? -code * xv : code * xv;
                }
            }
            if (l < 2u) s0 += sum; else s1 += sum;
        }
        acc = fma(b.xd[blk * 8u + ib], db0 * float(s0) + db1 * float(s1), acc);
    }
    const float v = simd_sum(acc);
    if (lane == 0u) b.y[row] = b.alpha * v + (b.bias != nullptr ? b.bias[row] : 0.0f);
}

kernel void ds4_iq2s_mv1(device const uchar* jobs [[buffer(0)]], uint2 tgp [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const device Ds4MvJob& b = *(const device Ds4MvJob*) (jobs + (size_t) tgp.y * sizeof(Ds4MvJob));
    const uint row = tgp.x * 8u + (tid >> 5);
    if (int(row) >= b.n) return;
    const uint nblk = uint(b.k) / 256u;
    const device uchar* w = b.w + (size_t) row * b.ld;
    const uint ib = lane & 7u;
    const uint bsel = lane >> 3;
    float acc = 0.0f;
    for (uint blk = bsel; blk < nblk; blk += 4u) {
        const device uchar* p = w + (size_t) blk * 82u;
        const float d = float(as_type<half>(*(const device ushort*) p));
        const device uchar* qs = p + 2u + 4u * ib;
        const device uchar* sgb = p + 34u + 4u * ib;
        const uint qh = uint(p[66u + ib]);
        const uint sc = uint(p[74u + ib]);
        const float db0 = d * (0.5f + float(sc & 0xfu)) * 0.25f, db1 = d * (0.5f + float(sc >> 4)) * 0.25f;
        const device uint4* xq4 = (const device uint4*) (b.xq + (size_t) blk * 256u + 32u * ib);
        const uint4 xa = xq4[0], xb = xq4[1];
        const uint xw[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
        int s0 = 0, s1 = 0;
        for (uint l = 0; l < 4u; ++l) {
            const uint idx = uint(qs[l]) | ((qh << (8u - 2u * l)) & 0x300u);
            const ulong g = iq2s_grid[idx];
            const uint sg = uint(sgb[l]);
            int sum = 0;
            for (uint h = 0; h < 2u; ++h) {
                const uint gw = uint(g >> (32u * h));
                const uint x = xw[2u * l + h];
                for (uint i = 0; i < 4u; ++i) {
                    const int code = int((gw >> (8u * i)) & 0xffu);
                    const int xv = int(char(x >> (8u * i)));
                    sum += ((sg >> (4u * h + i)) & 1u) != 0u ? -code * xv : code * xv;
                }
            }
            if (l < 2u) s0 += sum; else s1 += sum;
        }
        acc = fma(b.xd[blk * 8u + ib], db0 * float(s0) + db1 * float(s1), acc);
    }
    const float v = simd_sum(acc);
    if (lane == 0u) b.y[row] = b.alpha * v + (b.bias != nullptr ? b.bias[row] : 0.0f);
}

kernel void ds4_iq3xxs_mv1(device const uchar* jobs [[buffer(0)]], uint2 tgp [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const device Ds4MvJob& b = *(const device Ds4MvJob*) (jobs + (size_t) tgp.y * sizeof(Ds4MvJob));
    const uint row = tgp.x * 8u + (tid >> 5);
    if (int(row) >= b.n) return;
    const uint nblk = uint(b.k) / 256u;
    const device uchar* w = b.w + (size_t) row * b.ld;
    const uint ib = lane & 7u;
    const uint bsel = lane >> 3;
    float acc = 0.0f;
    for (uint blk = bsel; blk < nblk; blk += 4u) {
        const device uchar* p = w + (size_t) blk * 98u;
        const float d = float(as_type<half>(*(const device ushort*) p));
        const device uchar* qs = p + 2u;
        const uint w0 = uint(qs[8u * ib]) | (uint(qs[8u * ib + 1u]) << 8) | (uint(qs[8u * ib + 2u]) << 16) | (uint(qs[8u * ib + 3u]) << 24);
        const uint w1 = uint(qs[8u * ib + 4u]) | (uint(qs[8u * ib + 5u]) << 8) | (uint(qs[8u * ib + 6u]) << 16) | (uint(qs[8u * ib + 7u]) << 24);
        const uint aux = uint(qs[64u + 4u * ib]) | (uint(qs[65u + 4u * ib]) << 8) | (uint(qs[66u + 4u * ib]) << 16) | (uint(qs[67u + 4u * ib]) << 24);
        const float db = d * (0.5f + float(aux >> 28)) * 0.5f;
        const device uint4* xq4 = (const device uint4*) (b.xq + (size_t) blk * 256u + 32u * ib);
        const uint4 xa = xq4[0], xb = xq4[1];
        const uint xw[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
        int sum = 0;
        for (uint l = 0; l < 4u; ++l) {
            const uint wv = l < 2u ? w0 : w1, sh = 16u * (l & 1u);
            const uint sgn = uint(ksigns_iq2xs[(aux >> (7u * l)) & 127u]);
            const uint g0 = iq3xxs_grid[(wv >> sh) & 255u], g1 = iq3xxs_grid[(wv >> (sh + 8u)) & 255u];
            const uint xl = xw[2u * l], xh = xw[2u * l + 1u];
            for (uint i = 0; i < 4u; ++i) {
                int v0 = int((g0 >> (8u * i)) & 0xffu), v1 = int((g1 >> (8u * i)) & 0xffu);
                if ((sgn & (1u << i)) != 0u) v0 = -v0;
                if ((sgn & (1u << (i + 4u))) != 0u) v1 = -v1;
                sum += v0 * int(char(xl >> (8u * i))) + v1 * int(char(xh >> (8u * i)));
            }
        }
        acc = fma(b.xd[blk * 8u + ib] * db, float(sum), acc);
    }
    const float v = simd_sum(acc);
    if (lane == 0u) b.y[row] = b.alpha * v + (b.bias != nullptr ? b.bias[row] : 0.0f);
}
