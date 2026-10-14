// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/xe/s2_expert_grouped.cpp - the Xe port of Strata's src/kernels/cuda/s2_expert_grouped.cu: the S2 (Strata Q2
// blob) experts computed on the GPU - one hit at a time, per-hit on device counts, grouped by expert for a window,
// and in the CPU pool's summation order - plus the hit selection, grouping and combination.
//
// Work shapes and orders are the CUDA kernels'.  __dp4a is SPIR-V's integer dot product (cuda_intrinsics.hpp); a warp
// is a sub-group of 32 whose xor butterflies reproduce lane 0 of CUDA's shuffle-down trees, including the CPU-order
// reduction over eight lanes (offsets 4, 1, 2).  The ballot/popc compactions are integer group scans, which place
// every hit where CUDA placed it.  Work-items past the last row keep computing (on row 0) and do not write, because
// a sub-group permute needs every lane.
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/core/runtime.hpp"
#include "cuda_intrinsics.hpp"
#define STRATA_DEVICE_ARCH_TAG_TU xe_s2_expert_grouped
#include "strata/kernels/device_arch_tag.hpp"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

namespace strata::kernels {
namespace {

using xe::dp4a;

constexpr int H = 2560;
constexpr int FF = 640;
static_assert(FF % 32 == 0, "swiglu_q8_launch works in whole 32-value blocks");
constexpr int QK = 64;                       // Q2_0's group: one fp16 scale per 64 weights
constexpr int ROW_GU = H / 4;                // 640 B of codes per gate/up row (2 bits per element)
constexpr int ROW_D = FF / 4;                // 160 B per down row
constexpr int SC_GU = H / QK;                // 40 fp16 scales per gate/up row
constexpr int SC_D = FF / QK;                // 10 per down row
constexpr size_t O_D_CODES = (size_t) 2 * FF * ROW_GU;
constexpr size_t O_GU_SCALES = O_D_CODES + (size_t) H * ROW_D;
constexpr size_t O_D_SCALES = O_GU_SCALES + (size_t) 2 * FF * SC_GU * 2;
constexpr int THREADS = 256;
constexpr int WARP = 32;

sycl::queue& Q(void* stream) { return core::Runtime::get().stream(stream); }
[[noreturn]] void fail(const std::string& what) { throw core::DeviceError(what); }

inline float f16_at(const uint8_t* p) { return f32_from_f16((uint16_t) (p[0] | (p[1] << 8))); }

inline int code_word(unsigned cbyte) {
    return (int) ((cbyte & 3u) | (((cbyte >> 2) & 3u) << 8) | (((cbyte >> 4) & 3u) << 16) | (((cbyte >> 6) & 3u) << 24));
}

inline float row_dot_s2_q8(const uint8_t* codes, const uint8_t* scales, const uint8_t* x_q8_0, int n_chunks, int lane,
                           const float* x_scales) {
    float acc = 0.0f;
    for (int c = lane; c < n_chunks; c += 32) {
        const uint8_t* cb = codes + (size_t) c * 8;             // 8 code bytes = 32 elements
        const uint8_t* xb = x_q8_0 + (size_t) c * 34;           // one block_q8_0
        const float dx = x_scales ? x_scales[c] : f16_at(xb);
        const int8_t* xq = (const int8_t*) (xb + 2);
        int s = 0;      // sum of code * x
        int hx = 0;     // sum of x, the weight-independent term, as ones * x
        const int ones = 0x01010101;
        for (int j = 0; j < 8; ++j) {
            const int cw = code_word(cb[j]);
            int xw;
            std::memcpy(&xw, xq + 4 * j, 4);   // a block_q8_0's data is never 4-byte aligned
            s = dp4a(cw, xw, s);
            hx = dp4a(ones, xw, hx);
        }
        const float dw = f16_at(scales + (size_t) (c >> 1) * 2);   // one weight scale per two 32-element chunks
        acc += dw * dx * (float) (s - hx);
    }
    return acc;
}

inline float warp_sum(const sycl::sub_group& sg, float v) {
    for (int off = 16; off > 0; off >>= 1) v += sycl::permute_group_by_xor(sg, v, off);
    return v;
}

// gate/up rows, one sub-group per row; the output is gate-major (every hit's gate rows first, then every hit's up
// rows), so the swiglu and the quantizer read contiguous ranges
sycl::event gu_launch(sycl::queue& q, const uint8_t* blob_base, const int32_t* slot_index, long long blob_bytes,
                      const uint8_t* x_q8_0, const float* x_scales, float* gate_up, int n_hits,
                      const int32_t* d_count = nullptr, const int32_t* dst_index = nullptr, int tok_div = 0) {
    const long long total = (long long) n_hits * 2LL * FF;
    const int warps = THREADS / WARP;
    const size_t groups = (size_t) ((total + warps - 1) / warps);
    return q.parallel_for(sycl::nd_range<1>(groups * THREADS, THREADS), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const long long slot = (long long) it.get_group(0) * warps + (long long) sg.get_group_linear_id();
        if (slot >= total) return;   // uniform over the sub-group
        const long long rows_per_hit = 2LL * FF;
        const int h = (int) (slot / rows_per_hit);
        if (d_count != nullptr && h >= *d_count) return;   // token graph: capacity layout, device count
        const int i = (int) (slot % rows_per_hit);
        const int lane = (int) sg.get_local_linear_id();
        const uint8_t* blob = blob_base + (size_t) slot_index[h] * (size_t) blob_bytes;
        const uint8_t* xq = x_q8_0;
        const float* xs = x_scales;
        if (tok_div > 0) {   // a verify window: each hit reads its own token's activation
            const int tok = dst_index[h] / tok_div;
            xq += (size_t) tok * (size_t) (H / 32) * 34;
            if (xs != nullptr) xs += (size_t) tok * (size_t) (H / 32);
        }
        const float acc = row_dot_s2_q8(blob + (size_t) i * ROW_GU, blob + O_GU_SCALES + (size_t) i * SC_GU * 2, xq,
                                        H / 32, lane, xs);
        const float s = warp_sum(sg, acc);
        if (lane != 0) return;
        const int r = i >> 1;
        const size_t base = (i & 1) ? ((size_t) n_hits * FF + (size_t) h * FF) : ((size_t) h * FF);
        gate_up[base + (size_t) r] = s;
    });
}

sycl::event swiglu_launch(sycl::queue& q, float* gate_up, long long n_pairs) {
    return q.parallel_for(sycl::range<1>((size_t) n_pairs), [=](sycl::id<1> id) {
        const long long i = (long long) id[0];
        const float g = gate_up[i];
        const float u = gate_up[n_pairs + i];
        gate_up[i] = (g / (1.0f + sycl::exp(-g))) * u;
    });
}

// swiglu_launch and quantize_q8_0_scaled in one kernel (upstream 3281ac32): one sub-group a 32-value block, each lane
// its value's SwiGLU (written back to gate_up) and the block's scaled Q8_0, the same arithmetic as the two launches
sycl::event swiglu_q8_launch(sycl::queue& q, float* gate_up, long long n_pairs, uint8_t* blocks, float* scales) {
    const long long n_blocks = n_pairs / 32;
    const size_t groups = (size_t) ((n_blocks + 7) / 8);
    return q.parallel_for(sycl::nd_range<1>(groups * 256, 256), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const long long b = (long long) it.get_group(0) * 8 + (long long) sg.get_group_linear_id();
        if (b >= n_blocks) return;   // uniform over the sub-group
        const int lane = (int) sg.get_local_linear_id();
        const long long i = b * 32 + lane;
        const float g = gate_up[i];
        const float u = gate_up[n_pairs + i];
        const float xv = (g / (1.0f + sycl::exp(-g))) * u;
        gate_up[i] = xv;
        uint8_t* out = blocks + b * 34;
        float amax = sycl::fabs(xv);
        for (int o = 16; o > 0; o >>= 1) amax = sycl::fmax(amax, sycl::permute_group_by_xor(sg, amax, o));
        const float sc = amax > 0.f ? amax / 127.f : 0.f;
        const float inv = sc > 0.f ? 1.f / sc : 0.f;
        if (lane == 0) {
            scales[b] = sc;
            const uint16_t d16bits = f16_from_f32(sc);
            out[0] = (uint8_t) (d16bits & 0xFF);
            out[1] = (uint8_t) (d16bits >> 8);
        }
        const float t = xv * inv;
        const float r = t + (t >= 0.f ? 0.5f : -0.5f);
        int v = (int) r;
        v = v < -127 ? -127 : (v > 127 ? 127 : v);
        out[2 + lane] = (uint8_t) (int8_t) v;
    });
}

sycl::event down_launch(sycl::queue& q, const uint8_t* blob_base, const int32_t* slot_index, const int32_t* dst_index,
                        long long blob_bytes, const uint8_t* h_q8_0, const float* h_scales, float* out, int n_hits,
                        const int32_t* d_count = nullptr) {
    const long long total = (long long) n_hits * H;
    const int warps = THREADS / WARP;
    const size_t groups = (size_t) ((total + warps - 1) / warps);
    return q.parallel_for(sycl::nd_range<1>(groups * THREADS, THREADS), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const long long row = (long long) it.get_group(0) * warps + (long long) sg.get_group_linear_id();
        if (row >= total) return;
        const int h = (int) (row / H);
        if (d_count != nullptr && h >= *d_count) return;
        const int r = (int) (row % H);
        const int lane = (int) sg.get_local_linear_id();
        const uint8_t* blob = blob_base + (size_t) slot_index[h] * (size_t) blob_bytes;
        const uint8_t* xb = h_q8_0 + (size_t) h * (size_t) (FF / 32) * 34;
        const float acc = row_dot_s2_q8(blob + O_D_CODES + (size_t) r * ROW_D, blob + O_D_SCALES + (size_t) r * SC_D * 2,
                                        xb, FF / 32, lane, h_scales ? h_scales + (size_t) h * (size_t) (FF / 32) : nullptr);
        const float s = warp_sum(sg, acc);
        if (lane == 0) out[(size_t) dst_index[h] * H + r] = s;
    });
}

// ---- the CPU pool's summation order (bench/micro/hit_cpu_order_parity in the CUDA tree)
inline int dot4(const uint8_t* codes, const int8_t* q) {
    int xw;
    std::memcpy(&xw, q, sizeof xw);
    return dp4a(code_word(*codes), xw, 0);
}

// eight lanes per row; only lane 0 of each eight is consumed.  _mm_add_ps(low128, high128), then two _mm_hadd_ps:
// the pair order is 4, 1, 2
inline float row_dot_cpu_order(const sycl::sub_group& sg, const uint8_t* codes, const uint8_t* scales,
                               const uint8_t* xq, const float* xs, const float* hx, int blocks, int lane) {
    float acc = 0.0f;
    float corr = 0.0f;
    for (int b = 0; b < blocks; ++b) {
        const float d = f16_at(scales + 2 * b);
        const int lo = dot4(codes + b * 16 + lane, (const int8_t*) (xq + (size_t) (2 * b) * 34 + 2) + lane * 4);
        const int hi = dot4(codes + b * 16 + 8 + lane, (const int8_t*) (xq + (size_t) (2 * b + 1) * 34 + 2) + lane * 4);
        acc = sycl::fma(d * xs[2 * b], (float) lo, acc);
        acc = sycl::fma(d * xs[2 * b + 1], (float) hi, acc);
        if (lane == 0) corr = corr + d * (hx[2 * b] + hx[2 * b + 1]);
    }
    acc = acc + sycl::permute_group_by_xor(sg, acc, 4);
    acc = acc + sycl::permute_group_by_xor(sg, acc, 1);
    acc = acc + sycl::permute_group_by_xor(sg, acc, 2);
    return acc - corr;
}

template <bool DOWN>
sycl::event cpu_order_projection(sycl::queue& q, const uint8_t* blob_base, const int32_t* slots,
                                 const int32_t* destinations, long long blob_bytes, const uint8_t* xq, const float* xs,
                                 const float* hx, float* out, int n_hits) {
    constexpr int rows_per_hit = DOWN ? H : 2 * FF;
    const int rows = n_hits * rows_per_hit;
    const int rows_per_group = THREADS / 8;
    const size_t groups = (size_t) ((rows + rows_per_group - 1) / rows_per_group);
    return q.parallel_for(sycl::nd_range<1>(groups * THREADS, THREADS), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const int t = (int) it.get_local_id(0);
        const int row_any = (int) it.get_group(0) * rows_per_group + t / 8;
        const bool valid = row_any < rows;
        const int row = valid ? row_any : 0;
        const int h = row / rows_per_hit;
        const int r = row % rows_per_hit;
        const int lane = t & 7;
        const uint8_t* blob = blob_base + (size_t) slots[h] * (size_t) blob_bytes;
        const int chunks_offset = DOWN ? h * (FF / 32) : 0;
        const uint8_t* codes = DOWN ? blob + O_D_CODES + (size_t) r * ROW_D : blob + (size_t) r * ROW_GU;
        const uint8_t* scales = DOWN ? blob + O_D_SCALES + (size_t) r * SC_D * 2 : blob + O_GU_SCALES + (size_t) r * SC_GU * 2;
        const float value = row_dot_cpu_order(sg, codes, scales, xq + (size_t) chunks_offset * 34, xs + chunks_offset,
                                              hx + chunks_offset, DOWN ? SC_D : SC_GU, lane);
        if (!valid || lane != 0) return;
        if (DOWN) out[(size_t) destinations[h] * H + r] = value;
        else out[((r & 1) ? (size_t) n_hits * FF : 0) + (size_t) h * FF + (r >> 1)] = value;
    });
}

// ---- grouped by expert (a verify window)
constexpr int GU_CHUNKS = (H / 32 + 31) / 32;   // 3: chunks of a gate/up row per lane (80 chunks / 32 lanes)
constexpr int GMAX = 8;                          // entries per group (tokens routed to one expert in a window)
// a group holds one entry per token of the window routed to its expert, and the kernels below keep at most GMAX of
// them: a longer window would drop entries without a word (upstream 64cfe3c)
static_assert(GMAX >= kVerifyMaxT, "a verify window's group can exceed GMAX entries");
constexpr int GU_ROWS = 32;                      // gate/up rows per work-group: 4 per sub-group
constexpr int D_ROWS = 64;                       // down rows per work-group: 8 per sub-group

inline float chunk_dot(const uint8_t* cbytes, const int* xw, float dw, float dx) {
    int s = 0, hx = 0;
    const int ones = 0x01010101;
    for (int j = 0; j < 8; ++j) {
        const int cw = code_word(cbytes[j]);
        s = dp4a(cw, xw[j], s);
        hx = dp4a(ones, xw[j], hx);
    }
    return dw * dx * (float) (s - hx);
}

// ---- THE SAME INTEGERS FROM FEWER INSTRUCTIONS (upstream e155b07b).  `STRATA_OLD_GROUPED=1` keeps the kernels above.
//
// **INSIDE A CHUNK, WHICH CODE MEETS WHICH ACTIVATION IN A `dp4a` WORD IS FREE.**  `s = sum_e code_e * x_e` and
// `hx = sum_e x_e` are exact integer sums (|s| <= 32 * 3 * 128), so any grouping of the 32 products into words
// gives the same two integers.  `(w >> 2f) & 0x03030303` holds field `f` of each of a code word's four bytes -
// elements `4b + f`, plus 16 for the second word - so it is the ACTIVATIONS that are regrouped, once per chunk, into
//
//     X[4h + f] = { x[16h + f], x[16h + 4 + f], x[16h + 8 + f], x[16h + 12 + f] }
//
// a 4x4 byte transpose of each half of the chunk.  The kernels above spend seven integer operations per code BYTE and
// a second `dp4a` chain for `hx` on every row; here a code word costs seven for 16 elements and `hx` is summed once
// per chunk.  `s`, `hx` and the float expression `dw * dx * (float) (s - hx)` are unchanged, so every output is
// bitwise the previous kernels'.

// fp16 at an even address, one 16-bit load (`f16_at` reads it as two bytes); same value.
inline float f16_ld(const uint8_t* p) { return f32_from_f16(*(const uint16_t*) p); }

// A chunk's two code words as the eight `dp4a` operands that pair with `X[0..7]`: `m[4h + f]` byte `b` is the code
// of element `16h + 4b + f`.
inline void expand_codes(uint32_t lo, uint32_t hi, int m[8]) {
    const uint32_t M = 0x03030303u;
    m[0] = (int) (lo & M);
    m[1] = (int) ((lo >> 2) & M);
    m[2] = (int) ((lo >> 4) & M);
    m[3] = (int) ((lo >> 6) & M);
    m[4] = (int) (hi & M);
    m[5] = (int) ((hi >> 2) & M);
    m[6] = (int) ((hi >> 4) & M);
    m[7] = (int) ((hi >> 6) & M);
}
// the 8-byte aligned code words of chunk `c` (two 32-bit loads; CUDA's uint2)
inline void load_codes(const uint8_t* codes, int c, int m[8]) {
    const uint32_t* w = (const uint32_t*) (codes + (size_t) c * 8);
    expand_codes(w[0], w[1], m);
}

// The 32 int8 of the `block_q8_0` at `xb` regrouped into `X[0..7]` (see above), and their sum `hx`.
//
// **NINE ALIGNED WORD LOADS INSTEAD OF 32 BYTE LOADS.**  The caller guarantees a 4-byte aligned activation row, so
// chunk `c`'s int8 start at `34c + 2`, i.e. 0 bytes (odd `c`) or 2 bytes (even `c`) into an aligned word; the words
// are read aligned and shifted into place.  The ninth word is read only in the 2-byte case, and then it lies inside
// block `c + 1` of the same row: both row lengths (80 and 20 chunks) are even, so an even chunk is never a row's last.
inline int load_x_chunk(const uint8_t* xb, int X[8]) {
    const uint8_t* q = xb + 2;
    const int off = (int) ((uintptr_t) q & 3);
    const uint32_t* p = (const uint32_t*) (q - off);
    const uint32_t sh = (uint32_t) off * 8;
    uint32_t v[9];
    for (int j = 0; j < 8; ++j) v[j] = p[j];
    v[8] = off != 0 ? p[8] : 0u;
    uint32_t n[8];
    int hx = 0;
    for (int j = 0; j < 8; ++j) {
        n[j] = (uint32_t) ((((uint64_t) v[j + 1] << 32) | v[j]) >> sh);   // natural word j: x[4j .. 4j+3]
        hx = dp4a(0x01010101, (int) n[j], hx);
    }
    // CUDA's __byte_perm transposes: t0 = a0 b0 a1 b1, t1 = a2 b2 a3 b3 (and t2, t3 of c, d), then
    // X = a0 b0 c0 d0 / a1 b1 c1 d1 / a2 b2 c2 d2 / a3 b3 c3 d3
    auto lo_pairs = [](uint32_t a, uint32_t b) {
        return (a & 0xFFu) | ((b & 0xFFu) << 8) | (((a >> 8) & 0xFFu) << 16) | (((b >> 8) & 0xFFu) << 24);
    };
    auto hi_pairs = [](uint32_t a, uint32_t b) {
        return ((a >> 16) & 0xFFu) | (((b >> 16) & 0xFFu) << 8) | (((a >> 24) & 0xFFu) << 16) | ((b >> 24) << 24);
    };
    for (int b = 0; b < 8; b += 4) {   // the two halves: words b .. b + 3
        const uint32_t t0 = lo_pairs(n[b], n[b + 1]);
        const uint32_t t1 = hi_pairs(n[b], n[b + 1]);
        const uint32_t t2 = lo_pairs(n[b + 2], n[b + 3]);
        const uint32_t t3 = hi_pairs(n[b + 2], n[b + 3]);
        X[b + 0] = (int) ((t0 & 0xFFFFu) | (t2 << 16));
        X[b + 1] = (int) ((t0 >> 16) | (t2 & 0xFFFF0000u));
        X[b + 2] = (int) ((t1 & 0xFFFFu) | (t3 << 16));
        X[b + 3] = (int) ((t1 >> 16) | (t3 & 0xFFFF0000u));
    }
    return hx;
}

// `s` for one chunk: the eight expanded code words against the eight regrouped activation words.
inline int chunk_s(const int m[8], const int X[8]) {
    int s = 0;
    for (int j = 0; j < 8; ++j) s = dp4a(m[j], X[j], s);
    return s;
}

// `gu_launch`, new: ONE SUB-GROUP PER (gate, up) PAIR - row-slots `2r` and `2r + 1`, adjacent in the blob - so each
// activation chunk is loaded and regrouped once for both rows.  Per row, the lane's chunks, their order, the float
// expression and the butterfly are `row_dot_s2_q8`'s; the outputs land where `gu_launch` puts them.
sycl::event gu_pair_launch(sycl::queue& q, const uint8_t* blob_base, const int32_t* slot_index, long long blob_bytes,
                           const uint8_t* x_q8_0, const float* x_scales, float* gate_up, int n_hits,
                           const int32_t* d_count, const int32_t* dst_index, int tok_div) {
    const long long total = (long long) n_hits * FF;
    const int warps = THREADS / WARP;
    const size_t groups = (size_t) ((total + warps - 1) / warps);
    return q.parallel_for(sycl::nd_range<1>(groups * THREADS, THREADS), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const long long pair = (long long) it.get_group(0) * warps + (long long) sg.get_group_linear_id();
        if (pair >= total) return;   // uniform over the sub-group
        const int h = (int) (pair / FF);
        if (d_count != nullptr && h >= *d_count) return;
        const int r = (int) (pair % FF);
        const int lane = (int) sg.get_local_linear_id();
        const uint8_t* blob = blob_base + (size_t) slot_index[h] * (size_t) blob_bytes;
        const uint8_t* xq = x_q8_0;
        const float* xs = x_scales;
        if (tok_div > 0) {
            const int tok = dst_index[h] / tok_div;
            xq += (size_t) tok * (size_t) (H / 32) * 34;
            if (xs != nullptr) xs += (size_t) tok * (size_t) (H / 32);
        }
        const uint8_t* codes = blob + (size_t) (2 * r) * ROW_GU;   // gate row; the up row follows it
        const uint8_t* scales = blob + O_GU_SCALES + (size_t) (2 * r) * SC_GU * 2;
        float acc_g = 0.0f, acc_u = 0.0f;
        for (int c = lane; c < H / 32; c += 32) {
            const uint8_t* xb = xq + (size_t) c * 34;
            const float dx = xs ? xs[c] : f16_ld(xb);
            int X[8], m[8];
            const int hx = load_x_chunk(xb, X);
            load_codes(codes, c, m);
            const float dw_g = f16_ld(scales + (size_t) (c >> 1) * 2);
            acc_g += dw_g * dx * (float) (chunk_s(m, X) - hx);
            load_codes(codes + ROW_GU, c, m);
            const float dw_u = f16_ld(scales + (size_t) SC_GU * 2 + (size_t) (c >> 1) * 2);
            acc_u += dw_u * dx * (float) (chunk_s(m, X) - hx);
        }
        const float sgate = warp_sum(sg, acc_g);
        const float sup = warp_sum(sg, acc_u);
        if (lane != 0) return;
        gate_up[(size_t) h * FF + (size_t) r] = sgate;   // gate-major, as in `gu_launch`
        gate_up[(size_t) n_hits * FF + (size_t) h * FF + (size_t) r] = sup;
    });
}

// `down_launch`, new: ONE SUB-GROUP PER PAIR OF ROWS `r, r + 1` of one hit, the intermediate's chunk loaded once for both.
sycl::event down_pair_launch(sycl::queue& q, const uint8_t* blob_base, const int32_t* slot_index,
                             const int32_t* dst_index, long long blob_bytes, const uint8_t* h_q8_0,
                             const float* h_scales, float* out, int n_hits, const int32_t* d_count) {
    const long long total = (long long) n_hits * (H / 2);
    const int warps = THREADS / WARP;
    const size_t groups = (size_t) ((total + warps - 1) / warps);
    return q.parallel_for(sycl::nd_range<1>(groups * THREADS, THREADS), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const long long pair = (long long) it.get_group(0) * warps + (long long) sg.get_group_linear_id();
        if (pair >= total) return;
        const int h = (int) (pair / (H / 2));
        if (d_count != nullptr && h >= *d_count) return;
        const int r = 2 * (int) (pair % (H / 2));
        const int lane = (int) sg.get_local_linear_id();
        const uint8_t* blob = blob_base + (size_t) slot_index[h] * (size_t) blob_bytes;
        const uint8_t* xrow = h_q8_0 + (size_t) h * (size_t) (FF / 32) * 34;
        const float* xs = h_scales ? h_scales + (size_t) h * (size_t) (FF / 32) : nullptr;
        const uint8_t* codes = blob + O_D_CODES + (size_t) r * ROW_D;
        const uint8_t* scales = blob + O_D_SCALES + (size_t) r * SC_D * 2;
        float acc0 = 0.0f, acc1 = 0.0f;
        for (int c = lane; c < FF / 32; c += 32) {
            const uint8_t* xb = xrow + (size_t) c * 34;
            const float dx = xs ? xs[c] : f16_ld(xb);
            int X[8], m[8];
            const int hx = load_x_chunk(xb, X);
            load_codes(codes, c, m);
            const float dw0 = f16_ld(scales + (size_t) (c >> 1) * 2);
            acc0 += dw0 * dx * (float) (chunk_s(m, X) - hx);
            load_codes(codes + ROW_D, c, m);
            const float dw1 = f16_ld(scales + (size_t) SC_D * 2 + (size_t) (c >> 1) * 2);
            acc1 += dw1 * dx * (float) (chunk_s(m, X) - hx);
        }
        const float s0 = warp_sum(sg, acc0);
        const float s1 = warp_sum(sg, acc1);
        if (lane != 0) return;
        const size_t o = (size_t) dst_index[h] * H + (size_t) r;
        out[o] = s0;
        out[o + 1] = s1;
    });
}

// The previous kernels stay selectable for A/B - `STRATA_OLD_GROUPED=1` in the environment, or
// `moe_grouped_select_old`.  The environment is read once, on first use; the choice is made at each launch, so a
// recorded graph keeps the kernels it was recorded with.
std::atomic<int> g_select_old{-1};
// Which kernels the last launch of an entry point used: 1 = new, 0 = previous, -1 = none since the last query.
std::atomic<int> g_last_path{-1};

bool old_kernels() {
    const int s = g_select_old.load(std::memory_order_relaxed);
    if (s >= 0) return s != 0;
    static const bool env = [] {
        const char* e = std::getenv("STRATA_OLD_GROUPED");
        return e != nullptr && e[0] == '1';
    }();
    return env;
}

// `STRATA_GROUPED_PAIR_MIN_HITS=N`: the per-hit path keeps the previous one-row-a-sub-group kernels below N hits of
// capacity (the new ones launch half the sub-groups, two rows each).  Bitwise the same either way; default 0 (always
// the new kernels).  Ignored while `moe_grouped_select_old` forces a choice.
long long pair_min_hits() {
    if (g_select_old.load(std::memory_order_relaxed) >= 0) return 0;
    static const long long n = [] {
        const char* e = std::getenv("STRATA_GROUPED_PAIR_MIN_HITS");
        return e != nullptr ? std::strtoll(e, nullptr, 10) : 0LL;
    }();
    return n;
}

// The new kernels read the activations as aligned words (`load_x_chunk`), so they need 4-byte aligned activation
// rows - the input's and the intermediate's in `scratch`; the per-hit ones also read a blob's codes as two aligned
// words, which needs an 8-byte aligned arena and slot size (the grouped kernels' groups are VRAM slots, already so).
// Anything else keeps the previous kernels rather than issuing a misaligned load.
bool new_grouped(const void* x_q8_0, const void* scratch) {
    const bool fast = !old_kernels() && ((uintptr_t) x_q8_0 & 3) == 0 && ((uintptr_t) scratch & 3) == 0;
    g_last_path.store(fast ? 1 : 0, std::memory_order_relaxed);
    return fast;
}

bool new_hit(const void* blob_base, long long blob_bytes, const void* x_q8_0, const void* scratch, long long cap) {
    const bool fast = new_grouped(x_q8_0, scratch) && ((uintptr_t) blob_base & 7) == 0 && (blob_bytes & 7) == 0 &&
                      cap >= pair_min_hits();
    g_last_path.store(fast ? 1 : 0, std::memory_order_relaxed);
    return fast;
}

// ---- The grouped kernels with the staged activations laid out for the reads (upstream e155b07b).
//
// **THE LAYOUT ABOVE IS AN 8-WAY BANK CONFLICT.**  `xs_q[k * (H / 4) + c * 8 + j]` puts lane `L`'s word `j` at
// `8L + j`, so lanes `L, L + 4, L + 8, ...` share a bank on every one of the chunk's eight reads; and `chunk_dot`
// recomputes `hx`, which depends on the entry alone, for every row.  Here the staging writes, once per (entry, chunk):
//
//     xs_w[j][k * NC + c]   word j of the regrouped chunk (`load_x_chunk`), so lanes read consecutive words
//     xs_dh[k * NC + c]     (dx, hx) - one 64-bit read per chunk instead of a float and eight `dp4a`
//
// Word-major, the staging writes are consecutive as well (the flat index `i` IS `k * NC + c`).  Each sub-group takes
// its rows TWO at a time (gate `r` and up `r`, or down rows `r, r + 1`), so every local read serves both.  Per
// (row, entry) the lane's chunks, their order, the float expression and the butterfly are the kernels' above.
sycl::event gu_grouped_t_launch(sycl::queue& q, size_t cap_groups, const unsigned long long* grp_ptr,
                                const int32_t* grp_start, const int32_t* n_groups, const int32_t* ent_tok,
                                const uint8_t* x_q8_0, const float* x_scales, float* gate_up, int cap_e) {
    constexpr int NC = H / 32;
    return q.submit([&](sycl::handler& hd) {
        sycl::local_accessor<int, 1> xs_w(sycl::range<1>((size_t) 8 * GMAX * NC), hd);    // word j of entry k's chunk c at [j][k * NC + c]
        sycl::local_accessor<sycl::int2, 1> xs_dh(sycl::range<1>((size_t) GMAX * NC), hd);  // (dx as bits, hx)
        hd.parallel_for(sycl::nd_range<2>({cap_groups, (size_t) (2 * FF / GU_ROWS) * 256}, {1, 256}),
                        [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int g = (int) it.get_group(0);
            if (g >= *n_groups) return;   // uniform
            const int e0 = grp_start[g], ne = sycl::min(grp_start[g + 1] - e0, GMAX);
            const int t = (int) it.get_local_id(1), lane = t & 31, warp = t >> 5;
            for (int i = t; i < ne * NC; i += 256) {
                const int k = i / NC, c = i - k * NC;
                const int tok = ent_tok[e0 + k];
                const uint8_t* xb = x_q8_0 + (size_t) tok * (size_t) NC * 34 + (size_t) c * 34;
                const float dx = x_scales ? x_scales[(size_t) tok * NC + c] : f16_ld(xb);
                int X[8];
                const int hx = load_x_chunk(xb, X);
                for (int j = 0; j < 8; ++j) xs_w[j * (GMAX * NC) + i] = X[j];
                xs_dh[i] = sycl::int2(sycl::bit_cast<int>(dx), hx);
            }
            sycl::group_barrier(it.get_group());
            const uint8_t* blob = (const uint8_t*) grp_ptr[g];
            const int row0 = (int) it.get_group(1) * GU_ROWS;   // even: a pair is always (gate r, up r)
            for (int pp = warp; pp < GU_ROWS / 2; pp += 8) {
                const int i = row0 + 2 * pp;
                const uint8_t* codes = blob + (size_t) i * ROW_GU;
                const uint8_t* scales = blob + O_GU_SCALES + (size_t) i * SC_GU * 2;
                int m0[GU_CHUNKS][8], m1[GU_CHUNKS][8];
                float dw0[GU_CHUNKS] = {}, dw1[GU_CHUNKS] = {};
                for (int qc = 0; qc < GU_CHUNKS; ++qc) {
                    const int c = lane + 32 * qc;
                    if (c < NC) {
                        load_codes(codes, c, m0[qc]);
                        load_codes(codes + ROW_GU, c, m1[qc]);
                        dw0[qc] = f16_ld(scales + (size_t) (c >> 1) * 2);
                        dw1[qc] = f16_ld(scales + (size_t) SC_GU * 2 + (size_t) (c >> 1) * 2);
                    }
                }
                for (int k = 0; k < ne; ++k) {
                    float acc0 = 0.0f, acc1 = 0.0f;
                    for (int qc = 0; qc < GU_CHUNKS; ++qc) {
                        const int c = lane + 32 * qc;
                        if (c >= NC) break;
                        const int at = k * NC + c;
                        int X[8];
                        for (int j = 0; j < 8; ++j) X[j] = xs_w[j * (GMAX * NC) + at];
                        const sycl::int2 dh = xs_dh[at];
                        const float dx = sycl::bit_cast<float>(dh.x());
                        acc0 += dw0[qc] * dx * (float) (chunk_s(m0[qc], X) - dh.y());
                        acc1 += dw1[qc] * dx * (float) (chunk_s(m1[qc], X) - dh.y());
                    }
                    const float s0 = warp_sum(sg, acc0);
                    const float s1 = warp_sum(sg, acc1);
                    if (lane == 0) {
                        const int e = e0 + k, r = i >> 1;
                        gate_up[(size_t) e * FF + (size_t) r] = s0;
                        gate_up[(size_t) cap_e * FF + (size_t) e * FF + (size_t) r] = s1;
                    }
                }
            }
        });
    });
}

sycl::event down_grouped_t_launch(sycl::queue& q, size_t cap_groups, const unsigned long long* grp_ptr,
                                  const int32_t* grp_start, const int32_t* n_groups, const int32_t* ent_dst,
                                  const uint8_t* h_q8_0, const float* hs, float* out) {
    constexpr int NC = FF / 32;
    return q.submit([&](sycl::handler& hd) {
        sycl::local_accessor<int, 1> hs_w(sycl::range<1>((size_t) 8 * GMAX * NC), hd);
        sycl::local_accessor<sycl::int2, 1> hs_dh(sycl::range<1>((size_t) GMAX * NC), hd);
        hd.parallel_for(sycl::nd_range<2>({cap_groups, (size_t) (H / D_ROWS) * 256}, {1, 256}),
                        [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int g = (int) it.get_group(0);
            if (g >= *n_groups) return;
            const int e0 = grp_start[g], ne = sycl::min(grp_start[g + 1] - e0, GMAX);
            const int t = (int) it.get_local_id(1), lane = t & 31, warp = t >> 5;
            for (int i = t; i < ne * NC; i += 256) {
                const int k = i / NC, c = i - k * NC;
                const uint8_t* xb = h_q8_0 + (size_t) (e0 + k) * (size_t) NC * 34 + (size_t) c * 34;
                const float dx = hs ? hs[(size_t) (e0 + k) * NC + c] : f16_ld(xb);
                int X[8];
                const int hx = load_x_chunk(xb, X);
                for (int j = 0; j < 8; ++j) hs_w[j * (GMAX * NC) + i] = X[j];
                hs_dh[i] = sycl::int2(sycl::bit_cast<int>(dx), hx);
            }
            sycl::group_barrier(it.get_group());
            const uint8_t* blob = (const uint8_t*) grp_ptr[g];
            const int row0 = (int) it.get_group(1) * D_ROWS;
            for (int pp = warp; pp < D_ROWS / 2; pp += 8) {
                const int r = row0 + 2 * pp;
                const uint8_t* codes = blob + O_D_CODES + (size_t) r * ROW_D;
                const uint8_t* scales = blob + O_D_SCALES + (size_t) r * SC_D * 2;
                const int c = lane;   // lanes 0..19 hold one chunk each, as above
                int m0[8] = {}, m1[8] = {};
                float dw0 = 0.0f, dw1 = 0.0f;
                if (c < NC) {
                    load_codes(codes, c, m0);
                    load_codes(codes + ROW_D, c, m1);
                    dw0 = f16_ld(scales + (size_t) (c >> 1) * 2);
                    dw1 = f16_ld(scales + (size_t) SC_D * 2 + (size_t) (c >> 1) * 2);
                }
                for (int k = 0; k < ne; ++k) {
                    float acc0 = 0.0f, acc1 = 0.0f;
                    if (c < NC) {
                        const int at = k * NC + c;
                        int X[8];
                        for (int j = 0; j < 8; ++j) X[j] = hs_w[j * (GMAX * NC) + at];
                        const sycl::int2 dh = hs_dh[at];
                        const float dx = sycl::bit_cast<float>(dh.x());
                        acc0 += dw0 * dx * (float) (chunk_s(m0, X) - dh.y());
                        acc1 += dw1 * dx * (float) (chunk_s(m1, X) - dh.y());
                    }
                    const float s0 = warp_sum(sg, acc0);
                    const float s1 = warp_sum(sg, acc1);
                    if (lane == 0) {
                        const size_t o = (size_t) ent_dst[e0 + k] * H + (size_t) r;
                        out[o] = s0;
                        out[o + 1] = s1;
                    }
                }
            }
        });
    });
}

}  // namespace

void moe_grouped_select_old(int old) { g_select_old.store(old < 0 ? -1 : (old != 0 ? 1 : 0)); }

int moe_grouped_last_path() { return g_last_path.exchange(-1, std::memory_order_relaxed); }

uint64_t moe_hit_grouped_scratch_bytes(int64_t n_hits, int64_t n_embd, int64_t n_ff) {
    if (n_hits <= 0) return 0;
    const uint64_t gu = (uint64_t) n_hits * (uint64_t) (2 * n_ff) * 4;
    const uint64_t q8 = (uint64_t) n_hits * (uint64_t) (n_ff / 32) * 34;
    const uint64_t hs = (uint64_t) n_hits * (uint64_t) (n_ff / 32) * 4;
    const uint64_t xh = (uint64_t) (n_embd / 32) * 4;
    return ((gu + 15) & ~15ull) + ((q8 + 15) & ~15ull) + 2 * ((hs + 15) & ~15ull) + ((xh + 15) & ~15ull);
}

namespace {
void hit_pipeline(const uint8_t* blob_base, const int32_t* slot_index, const int32_t* dst_index,
                  const int32_t* d_count, int64_t n, int64_t blob_bytes, const uint8_t* x_q8_0, void* scratch,
                  float* out, void* stream, const float* x_scales, int tok_div) {
    auto& q = Q(stream);
    const uint64_t gu_bytes = ((uint64_t) n * (uint64_t) (2 * FF) * 4 + 15) & ~15ull;
    const uint64_t q8_bytes = ((uint64_t) n * (uint64_t) (FF / 32) * 34 + 15) & ~15ull;
    float* gate_up = (float*) scratch;
    uint8_t* h_q8_0 = (uint8_t*) scratch + gu_bytes;
    float* h_scales = (float*) ((uint8_t*) scratch + gu_bytes + q8_bytes);
    const bool fast = new_hit(blob_base, (long long) blob_bytes, x_q8_0, scratch, (long long) n);
    if (fast)
        gu_pair_launch(q, blob_base, slot_index, blob_bytes, x_q8_0, x_scales, gate_up, (int) n, d_count, dst_index,
                       tok_div);
    else
        gu_launch(q, blob_base, slot_index, blob_bytes, x_q8_0, x_scales, gate_up, (int) n, d_count, dst_index, tok_div);
    void* s = &q;
    if (x_scales != nullptr) {
        swiglu_q8_launch(q, gate_up, n * (long long) FF, h_q8_0, h_scales);
    } else {
        swiglu_launch(q, gate_up, n * (long long) FF);
        quantize_q8_0(gate_up, h_q8_0, n * (int64_t) FF, s);
    }
    const auto e = fast ? down_pair_launch(q, blob_base, slot_index, dst_index, blob_bytes, h_q8_0,
                                           x_scales != nullptr ? h_scales : nullptr, out, (int) n, d_count)
                        : down_launch(q, blob_base, slot_index, dst_index, blob_bytes, h_q8_0,
                                      x_scales != nullptr ? h_scales : nullptr, out, (int) n, d_count);
    if (!stream) core::Runtime::get().wait(e, "moe_hit_grouped_s2");
}
}  // namespace

void moe_hit_grouped_s2(const uint8_t* blob_base, const int32_t* slot_index, const int32_t* dst_index,
                        int64_t n_hits, int64_t blob_bytes, const uint8_t* x_q8_0, void* scratch, float* out,
                        void* stream, const float* x_scales) {
    if (n_hits <= 0) return;
    hit_pipeline(blob_base, slot_index, dst_index, nullptr, n_hits, blob_bytes, x_q8_0, scratch, out, stream, x_scales, 0);
}

void moe_hit_select(const int32_t* ids, const int32_t* res_row, int k, int n_expert, int32_t* slot, int32_t* dst,
                    int32_t* count, void* stream) {
    if (k < 1 || k > 32) fail("moe_hit_select: k must be 1..32");
    const auto e = Q(stream).parallel_for(sycl::nd_range<1>(32, 32), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const int lane = (int) sg.get_local_linear_id();
        int s = -1;
        if (lane < k) {
            const int ex = ids[lane];
            if (ex >= 0 && ex < n_expert) s = res_row[ex];
        }
        const int flag = s >= 0 ? 1 : 0;
        const int at = sycl::exclusive_scan_over_group(sg, flag, sycl::plus<int>());
        const int total = sycl::reduce_over_group(sg, flag, sycl::plus<int>());
        if (s >= 0) { slot[at] = s; dst[at] = lane; }
        if (lane == 0) *count = total;
    });
    if (!stream) core::Runtime::get().wait(e, "moe_hit_select");
}

void moe_hit_grouped_s2_dev(const uint8_t* blob_base, const int32_t* slot_index, const int32_t* dst_index,
                            const int32_t* d_count, int64_t cap, int64_t blob_bytes, const uint8_t* x_q8_0,
                            void* scratch, float* out, void* stream, const float* x_scales) {
    if (cap <= 0) return;
    hit_pipeline(blob_base, slot_index, dst_index, d_count, cap, blob_bytes, x_q8_0, scratch, out, stream, x_scales, 0);
}

void moe_hit_select_multi(const int32_t* ids, const int32_t* res_row, int n, int n_expert, int32_t* slot, int32_t* dst,
                          int32_t* count, void* stream) {
    if (n < 1 || n > 128) fail("moe_hit_select_multi: n must be 1..128");
    const auto e = Q(stream).parallel_for(sycl::nd_range<1>(128, 128), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
        const int i = (int) it.get_local_id(0);
        int s = -1;
        if (i < n) {
            const int ex = ids[i];
            if (ex >= 0 && ex < n_expert) s = res_row[ex];
        }
        const int flag = s >= 0 ? 1 : 0;
        const int at = sycl::exclusive_scan_over_group(it.get_group(), flag, sycl::plus<int>());
        const int total = sycl::reduce_over_group(it.get_group(), flag, sycl::plus<int>());
        if (s >= 0) { slot[at] = s; dst[at] = i; }
        if (i == 0) *count = total;
    });
    if (!stream) core::Runtime::get().wait(e, "moe_hit_select_multi");
}

void moe_hit_grouped_s2_multi(const uint8_t* blob_base, const int32_t* slot_index, const int32_t* dst_index,
                              const int32_t* d_count, int64_t cap, int64_t blob_bytes, const uint8_t* x_q8_0,
                              const float* x_scales, int k_per_token, void* scratch, float* out, void* stream) {
    if (cap <= 0) return;
    hit_pipeline(blob_base, slot_index, dst_index, d_count, cap, blob_bytes, x_q8_0, scratch, out, stream, x_scales,
                 k_per_token);
}

void moe_group_resident(const int32_t* ids, int n, int k_per_tok, const uint8_t* base, int64_t blob,
                        unsigned long long* grp_ptr, int32_t* grp_start, int32_t* counts, int32_t* ent_dst,
                        int32_t* ent_tok, void* stream) {
    if (n < 1 || n > 128) fail("moe_group_resident: n must be 1..128");
    const long long blob_ll = (long long) blob;
    const auto e = Q(stream).submit([&](sycl::handler& h) {
        sycl::local_accessor<int, 1> e_s(128, h), first_s(128, h), size_s(128, h), gidx_s(128, h), gstart_s(129, h);
        h.parallel_for(sycl::nd_range<1>(128, 128), [=](sycl::nd_item<1> it) {
            const int i = (int) it.get_local_id(0);
            const int ex = i < n ? ids[i] : -1;
            e_s[i] = ex;
            sycl::group_barrier(it.get_group());
            int first = i, rank = 0, size = 0;
            if (i < n) {
                for (int j = 0; j < i; ++j)
                    if (e_s[j] == ex) { if (first == i) first = j; ++rank; }
                if (first == i)
                    for (int j = i; j < n; ++j) size += e_s[j] == ex;
            }
            first_s[i] = first;
            size_s[i] = (i < n && first == i) ? size : 0;
            sycl::group_barrier(it.get_group());
            if (i == 0) {
                int gi = 0, acc = 0;
                for (int j = 0; j < n; ++j)
                    if (first_s[j] == j) {
                        gidx_s[j] = gi;
                        gstart_s[gi] = acc;
                        grp_ptr[gi] = (unsigned long long) (base + (size_t) e_s[j] * (size_t) blob_ll);
                        grp_start[gi] = acc;
                        acc += size_s[j];
                        ++gi;
                    }
                grp_start[gi] = acc;
                counts[0] = gi;
                counts[1] = acc;
            }
            sycl::group_barrier(it.get_group());
            if (i < n) {
                const int at = gstart_s[gidx_s[first]] + rank;
                ent_dst[at] = i;
                ent_tok[at] = i / k_per_tok;
            }
        });
    });
    if (!stream) core::Runtime::get().wait(e, "moe_group_resident");
}

void moe_grouped_s2(const unsigned long long* grp_ptr, const int32_t* grp_start, const int32_t* n_groups,
                    const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups, int64_t cap_entries,
                    const uint8_t* x_q8_0, const float* x_scales, void* scratch, float* out, void* stream) {
    if (cap_groups <= 0 || cap_entries <= 0) return;
    auto& q = Q(stream);
    const uint64_t gu_bytes = ((uint64_t) cap_entries * (uint64_t) (2 * FF) * 4 + 15) & ~15ull;
    const uint64_t q8_bytes = ((uint64_t) cap_entries * (uint64_t) (FF / 32) * 34 + 15) & ~15ull;
    float* gate_up = (float*) scratch;
    uint8_t* h_q8_0 = (uint8_t*) scratch + gu_bytes;
    float* h_scales = (float*) ((uint8_t*) scratch + gu_bytes + q8_bytes);
    const int cap_e = (int) cap_entries;
    const bool fast = new_grouped(x_q8_0, scratch);
    if (fast)
        gu_grouped_t_launch(q, (size_t) cap_groups, grp_ptr, grp_start, n_groups, ent_tok, x_q8_0, x_scales, gate_up,
                            cap_e);
    else
    q.submit([&](sycl::handler& hd) {
        sycl::local_accessor<int, 1> xs_q(sycl::range<1>(GMAX * (H / 4)), hd);   // the entries' int8 activations as words
        sycl::local_accessor<float, 1> xs_d(sycl::range<1>(GMAX * (H / 32)), hd);
        hd.parallel_for(sycl::nd_range<2>({(size_t) cap_groups, (size_t) (2 * FF / GU_ROWS) * 256}, {1, 256}),
                        [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int g = (int) it.get_group(0);
            if (g >= *n_groups) return;   // uniform
            const int e0 = grp_start[g], ne = sycl::min(grp_start[g + 1] - e0, GMAX);
            const int t = (int) it.get_local_id(1), lane = t & 31, warp = t >> 5;
            for (int i = t; i < ne * (H / 32); i += 256) {
                const int k = i / (H / 32), c = i - k * (H / 32);
                const uint8_t* xb = x_q8_0 + (size_t) ent_tok[e0 + k] * (size_t) (H / 32) * 34 + (size_t) c * 34;
                xs_d[k * (H / 32) + c] = x_scales ? x_scales[(size_t) ent_tok[e0 + k] * (H / 32) + c] : f16_at(xb);
                const int8_t* qq = (const int8_t*) (xb + 2);
                for (int w = 0; w < 8; ++w) {
                    int v;
                    std::memcpy(&v, qq + 4 * w, 4);
                    xs_q[k * (H / 4) + c * 8 + w] = v;
                }
            }
            sycl::group_barrier(it.get_group());
            const uint8_t* blob = (const uint8_t*) grp_ptr[g];
            const int row0 = (int) it.get_group(1) * GU_ROWS;
            for (int rr = warp; rr < GU_ROWS; rr += 8) {
                const int i = row0 + rr;
                const uint8_t* codes = blob + (size_t) i * ROW_GU;
                const uint8_t* scales = blob + O_GU_SCALES + (size_t) i * SC_GU * 2;
                uint8_t cb[GU_CHUNKS][8];
                float dw[GU_CHUNKS];
                for (int qc = 0; qc < GU_CHUNKS; ++qc) {
                    const int c = lane + 32 * qc;
                    if (c < H / 32) {
                        std::memcpy(cb[qc], codes + (size_t) c * 8, 8);
                        dw[qc] = f16_at(scales + (size_t) (c >> 1) * 2);
                    }
                }
                for (int k = 0; k < ne; ++k) {
                    float acc = 0.0f;
                    for (int qc = 0; qc < GU_CHUNKS; ++qc) {
                        const int c = lane + 32 * qc;
                        if (c >= H / 32) break;
                        int xw[8];
                        for (int j = 0; j < 8; ++j) xw[j] = xs_q[k * (H / 4) + c * 8 + j];
                        acc += chunk_dot(cb[qc], xw, dw[qc], xs_d[k * (H / 32) + c]);
                    }
                    const float sum = warp_sum(sg, acc);
                    if (lane == 0) {
                        const int en = e0 + k, r = i >> 1;
                        const size_t b = (i & 1) ? ((size_t) cap_e * FF + (size_t) en * FF) : ((size_t) en * FF);
                        gate_up[b + (size_t) r] = sum;
                    }
                }
            }
        });
    });
    void* s = &q;
    if (x_scales != nullptr) {
        swiglu_q8_launch(q, gate_up, cap_entries * (long long) FF, h_q8_0, h_scales);
    } else {
        swiglu_launch(q, gate_up, cap_entries * (long long) FF);
        quantize_q8_0(gate_up, h_q8_0, cap_entries * (int64_t) FF, s);
    }
    const float* hs = x_scales != nullptr ? h_scales : nullptr;
    const auto e = fast ? down_grouped_t_launch(q, (size_t) cap_groups, grp_ptr, grp_start, n_groups, ent_dst, h_q8_0,
                                                hs, out)
                        : q.submit([&](sycl::handler& hd) {
        sycl::local_accessor<int, 1> hs_q(sycl::range<1>(GMAX * (FF / 4)), hd);
        sycl::local_accessor<float, 1> hs_d(sycl::range<1>(GMAX * (FF / 32)), hd);
        hd.parallel_for(sycl::nd_range<2>({(size_t) cap_groups, (size_t) (H / D_ROWS) * 256}, {1, 256}),
                        [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int g = (int) it.get_group(0);
            if (g >= *n_groups) return;
            const int e0 = grp_start[g], ne = sycl::min(grp_start[g + 1] - e0, GMAX);
            const int t = (int) it.get_local_id(1), lane = t & 31, warp = t >> 5;
            for (int i = t; i < ne * (FF / 32); i += 256) {
                const int k = i / (FF / 32), c = i - k * (FF / 32);
                const uint8_t* xb = h_q8_0 + (size_t) (e0 + k) * (size_t) (FF / 32) * 34 + (size_t) c * 34;
                hs_d[k * (FF / 32) + c] = hs ? hs[(size_t) (e0 + k) * (FF / 32) + c] : f16_at(xb);
                const int8_t* qq = (const int8_t*) (xb + 2);
                for (int w = 0; w < 8; ++w) {
                    int v;
                    std::memcpy(&v, qq + 4 * w, 4);
                    hs_q[k * (FF / 4) + c * 8 + w] = v;
                }
            }
            sycl::group_barrier(it.get_group());
            const uint8_t* blob = (const uint8_t*) grp_ptr[g];
            const int row0 = (int) it.get_group(1) * D_ROWS;
            for (int rr = warp; rr < D_ROWS; rr += 8) {
                const int r = row0 + rr;
                const uint8_t* codes = blob + O_D_CODES + (size_t) r * ROW_D;
                const uint8_t* scales = blob + O_D_SCALES + (size_t) r * SC_D * 2;
                const int c = lane;
                uint8_t cb[8] = {};
                float dw = 0.0f;
                if (c < FF / 32) {
                    std::memcpy(cb, codes + (size_t) c * 8, 8);
                    dw = f16_at(scales + (size_t) (c >> 1) * 2);
                }
                for (int k = 0; k < ne; ++k) {
                    float acc = 0.0f;
                    if (c < FF / 32) {
                        int xw[8];
                        for (int j = 0; j < 8; ++j) xw[j] = hs_q[k * (FF / 4) + c * 8 + j];
                        acc += chunk_dot(cb, xw, dw, hs_d[k * (FF / 32) + c]);
                    }
                    const float sum = warp_sum(sg, acc);
                    if (lane == 0) out[(size_t) ent_dst[e0 + k] * H + r] = sum;
                }
            }
        });
    });
    if (!stream) core::Runtime::get().wait(e, "moe_grouped_s2");
}

void moe_hit_add(float* parts, const float* hit_out, const int32_t* dst, const int32_t* count, int64_t cap,
                 int64_t n_embd, void* stream) {
    if (cap <= 0) return;
    const int ne = (int) n_embd;
    const size_t gx = (size_t) ((n_embd + 255) / 256 < 8 ? (n_embd + 255) / 256 : 8);
    const auto e = Q(stream).parallel_for(sycl::nd_range<2>({(size_t) cap, gx * 256}, {1, 256}), [=](sycl::nd_item<2> it) {
        const int h = (int) it.get_group(0);
        if (h >= *count) return;
        const size_t row = (size_t) dst[h] * (size_t) ne;
        for (int i = (int) it.get_global_id(1); i < ne; i += (int) it.get_global_range(1)) parts[row + i] += hit_out[row + i];
    });
    if (!stream) core::Runtime::get().wait(e, "moe_hit_add");
}

void moe_hit_grouped_s2_cpu_order(const uint8_t* blob_base, const int32_t* slot_index,
                                 const int32_t* dst_index, int64_t n_hits, int64_t blob_bytes,
                                 const uint8_t* x_q8_0, void* scratch, float* out, void* stream,
                                 const float* x_scales, float* gate_up_trace) {
    if (n_hits <= 0) return;
    if (x_scales == nullptr) fail("moe_hit_grouped_s2_cpu_order requires fp32 activation scales");
    auto& q = Q(stream);
    const uint64_t gu_bytes = ((uint64_t) n_hits * 2 * FF * 4 + 15) & ~15ull;
    const uint64_t q8_bytes = ((uint64_t) n_hits * (FF / 32) * 34 + 15) & ~15ull;
    const uint64_t scale_bytes = ((uint64_t) n_hits * (FF / 32) * 4 + 15) & ~15ull;
    float* gu = (float*) scratch;
    uint8_t* hq = (uint8_t*) scratch + gu_bytes;
    float* hs = (float*) (hq + q8_bytes);
    float* hh = (float*) ((uint8_t*) hs + scale_bytes);
    float* xh = (float*) ((uint8_t*) hh + scale_bytes);
    // the input's correction term: scale * sum of its int8 values, per chunk
    q.parallel_for(sycl::range<1>(H / 32), [=](sycl::id<1> id) {
        const int c = (int) id[0];
        const int8_t* qv = (const int8_t*) (x_q8_0 + (size_t) c * 34 + 2);
        int sum = 0;
        for (int j = 0; j < 32; ++j) sum += qv[j];
        xh[c] = x_scales[c] * (float) sum;
    });
    cpu_order_projection<false>(q, blob_base, slot_index, dst_index, blob_bytes, x_q8_0, x_scales, xh, gu, (int) n_hits);
    if (gate_up_trace != nullptr) q.memcpy(gate_up_trace, gu, (size_t) n_hits * 2 * FF * sizeof(float));
    const int pairs = (int) n_hits * FF;
    q.parallel_for(sycl::range<1>((size_t) pairs), [=](sycl::id<1> id) {
        const int i = (int) id[0];
        const float g = gu[i];
        const float eg = sycl::exp(-g);
        gu[i] = (g / (1.0f + eg)) * gu[pairs + i];
    });
    q.parallel_for(sycl::range<1>((size_t) (n_hits * (FF / 32))), [=](sycl::id<1> id) {
        const int c = (int) id[0];
        const float* xb = gu + c * 32;
        uint8_t* o = hq + (size_t) c * 34;
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) amax = sycl::fmax(amax, sycl::fabs(xb[j]));
        const float s = amax > 0.0f ? amax / 127.0f : 0.0f;
        const float inv = s > 0.0f ? 1.0f / s : 0.0f;
        hs[c] = s;
        const uint16_t bits = f16_from_f32(s);
        o[0] = (uint8_t) bits;
        o[1] = (uint8_t) (bits >> 8);
        int sum = 0;
        for (int j = 0; j < 32; ++j) {
            const float t = xb[j] * inv;
            int v = (int) (t + (t >= 0.0f ? 0.5f : -0.5f));
            v = v < -127 ? -127 : (v > 127 ? 127 : v);
            o[2 + j] = (uint8_t) (int8_t) v;
            sum += v;
        }
        hh[c] = s * (float) sum;
    });
    const auto e = cpu_order_projection<true>(q, blob_base, slot_index, dst_index, blob_bytes, hq, hs, hh, out, (int) n_hits);
    if (!stream) core::Runtime::get().wait(e, "moe_hit_grouped_s2_cpu_order");
}

}  // namespace strata::kernels
