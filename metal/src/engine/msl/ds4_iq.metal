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
    const uint bs = b.type == 16 ? 66u : (b.type == 17 ? 74u : 82u);
    const device uchar* w = b.w + (size_t) row * b.ld;
    float acc[16];
    for (uint c = 0; c < 16u; ++c) acc[c] = 0.0f;
    for (uint blk = lane; blk < nblk; blk += 32u) {
        const device uchar* p = w + (size_t) blk * bs;
        const float d = float(as_type<half>(*(const device ushort*) p));
        for (uint ib = 0; ib < 8u; ++ib) {
            float db0, db1;
            uint idx[4];
            uint sg[4];
            if (b.type == 16) {
                const uint aux0 = uint(p[2 + 4 * ib]) | (uint(p[3 + 4 * ib]) << 8) | (uint(p[4 + 4 * ib]) << 16) |
                                  (uint(p[5 + 4 * ib]) << 24);
                const uint aux1 = uint(p[6 + 4 * ib]) | (uint(p[7 + 4 * ib]) << 8) | (uint(p[8 + 4 * ib]) << 16) |
                                  (uint(p[9 + 4 * ib]) << 24);
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
                    sg[l] = 0xffffffffu ^ ((q >> 9) * 0x00204081u);   // the 7 sign bits, expanded (ggml's sign trick)
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
