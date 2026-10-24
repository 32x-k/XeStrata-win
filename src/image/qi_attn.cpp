// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/qi_attn.cpp - the DiT's attention on the GPU's matrix engines (qi_kernels.hpp: qi_attention).
//
// A sub-group takes TM query rows of one head, all 128 dims, and walks the keys in chunks of CH on its own (the
// sub-groups of a work-group take neighbouring rows of one head, so they read the same keys' tiles out of the caches):
//   * S = Q . K^T on the matrix engines (Q's tiles in registers, K's from global memory), stored to local memory
//     row-major: the accumulators' layout is the vendor's, local memory's is ours;
//   * the online softmax of each row there, L lanes a row, writing the chunk's probabilities P as FP16; a row's
//     reference (its largest score) moves only when a score passes it by more than TAU, as FlashAttention-3 does, so
//     most chunks leave the output alone, and when one moves the output goes through local memory to be scaled;
//   * O += P . V on the matrix engines, the output kept in the accumulators.
// Q, K and V hold their rows rounded up to a multiple of 128 (qi_attn_rows), the rows past T zero.
// The matrix calls are joint_matrix's where the device reports the shape (Intel's XMX: 8 x 16 x 16 on 16 lanes, or
// 8 x 8 x 16 on 8; NVIDIA's tensor cores: 16 x 16 x 16 on 32), or RDNA3 and RDNA4's WMMA instructions (amd_wmma.hpp,
// 16 x 16 x 16 on 32).  A GPU without matrix engines, or STRATA_NO_XMX=1, takes qi_attention_plain.

#include "strata/image/qi_kernels.hpp"

#include "strata/core/per_device.hpp"
#include "strata/core/runtime.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/matrix_report.hpp"
#include "../kernels/xe/amd_wmma.hpp"
#include "../kernels/xe/device_target.hpp"

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace strata::image {

namespace {

namespace mx = sycl::ext::oneapi::experimental::matrix;
namespace syclex = sycl::ext::oneapi::experimental;
using strata::kernels::f16_from_f32;

constexpr int D = 128;      // head_dim
constexpr int CH = 64;      // keys a chunk

template<typename A>
inline auto at(const A& a, size_t off) {
    return a.template get_multi_ptr<sycl::access::decorated::no>() + off;
}

// joint_matrix's tiles of TM x TN x TK on SG lanes
template<int TM_, int TN_, int TK_, int SG_>
struct Jm {
    static constexpr int TM = TM_, TN = TN_, TK = TK_, SG = SG_;
    using A = mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::a, TM, TK, mx::layout::row_major>;
    using Bc = mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::b, TK, TN, mx::layout::col_major>;
    using Br = mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::b, TK, TN, mx::layout::row_major>;
    using C = mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, TN>;
    template<typename M, typename P> static void load(sycl::sub_group sg, M& m, P p, size_t ld) {
        if constexpr (std::is_pointer_v<P>)   // FP16 bits in global memory
            mx::joint_matrix_load(sg, m,
                                  sycl::address_space_cast<sycl::access::address_space::global_space,
                                                           sycl::access::decorated::no>(
                                      const_cast<sycl::half*>(reinterpret_cast<const sycl::half*>(p))),
                                  ld);
        else
            mx::joint_matrix_load(sg, m, p, ld);
    }
    static void zero(sycl::sub_group sg, C& c) { mx::joint_matrix_fill(sg, c, 0.0f); }
    template<typename B> static void mad(sycl::sub_group sg, C& c, A& a, B& b) { mx::joint_matrix_mad(sg, c, a, b, c); }
    template<typename P> static void store(sycl::sub_group sg, C& c, P p, size_t ld) {
        mx::joint_matrix_store(sg, c, p, ld, mx::layout::row_major);
    }
};

#if STRATA_AMD_WMMA
namespace w = strata::kernels::xe::wmma;
struct Wm {
    static constexpr int TM = 16, TN = 16, TK = 16, SG = 32;
    using A = w::A<sycl::half>;
    using Bc = w::Bc<sycl::half>;
    using Br = w::Br<sycl::half>;
    using C = w::C;
    template<typename M, typename P> static void load(sycl::sub_group sg, M& m, P p, size_t ld) {
        w::load<sycl::half>(sg, m, p, ld);
    }
    static void zero(sycl::sub_group sg, C& c) { w::fill(sg, c, 0.0f); }
    template<typename B> static void mad(sycl::sub_group sg, C& c, A& a, B& b) { w::mad(sg, c, a, b, c); }
    template<typename P> static void store(sycl::sub_group sg, C& c, P p, size_t ld) { w::store(sg, c, p, ld); }
};
#else
using Wm = Jm<16, 16, 16, 32>;   // the host's and the other devices' view: the shape the kernel is launched with
#endif

// the instances a device compile carries: SPIR-V Intel's shapes, NVPTX NVIDIA's, AMDGCN RDNA's WMMA
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
// (the FP16 tensor cores from sm_70 on: Pascal's code has no instance, and its GPUs report no shape: the plain kernel)
[[maybe_unused]] constexpr bool built(int p) { return p == 2 && (STRATA_NV_ARCH == 0 || STRATA_NV_ARCH >= 700); }
#elif defined(__SYCL_DEVICE_ONLY__) && defined(__AMDGCN__)
[[maybe_unused]] constexpr bool built(int p) { return p == 3 && STRATA_AMD_WMMA; }
#else
[[maybe_unused]] constexpr bool built(int p) { return p == 0 || p == 1; }
#endif

template<int K> struct Path;   // which tiles a kernel instance takes: 0 Xe2 (8x16x16/16), 1 Alchemist, 2 NVIDIA, 3 RDNA
template<> struct Path<0> { using Ops = Jm<8, 16, 16, 16>; };
template<> struct Path<1> { using Ops = Jm<8, 8, 16, 8>; };
template<> struct Path<2> { using Ops = Jm<16, 16, 16, 32>; };
template<> struct Path<3> { using Ops = Wm; };

template<int P>
struct AttnKernel {
    using Ops = typename Path<P>::Ops;
    static constexpr int TM = Ops::TM, TN = Ops::TN, TK = Ops::TK, SG = Ops::SG;
    static constexpr int L = SG / TM;              // lanes a row in the softmax
    static constexpr int CW = CH / L;              // a lane's scores: contiguous columns of its row
    static constexpr int NT = D / TN;              // the output's column tiles
    static constexpr int SLD = CH + 4, PLD = CH + 8;   // S's and P's row strides in local memory: rows apart in banks
    static constexpr int SREG = std::max(TM * SLD, TM * D);   // S, or the output staged [TM][D], FP32
    static_assert(SG % TM == 0 && CW % 8 == 0, "the softmax's rows split evenly over the lanes, 8 columns at a time");
    // local memory a sub-group takes: S (or the staged output), P, a row's m and l
    static constexpr size_t SG_BYTES = (size_t) SREG * 4 + (size_t) TM * PLD * 2 + (size_t) 2 * TM * 4;

    const uint16_t *q16, *k16, *v16;
    uint16_t* out16;
    int64_t T, heads, n_text;
    const int32_t* key_lim;   // the prefix rows' key limits (null: a text row its own prefix)
    float scale;
    int64_t row0;             // the first query row (a multiple of TM): out16's row 0 is query row `first`
    int64_t first;
    int64_t row_end;          // the rows past it are not asked (the vendor's kernel takes them)
    sycl::local_accessor<sycl::half, 1> ps;      // per sub-group P [TM][PLD]
    sycl::local_accessor<float, 1> st;           // per sub-group S [TM][SLD], or the output [TM][D]
    sycl::local_accessor<float, 1> rows;         // per sub-group m and l [2][TM]

    static void load_acc(sycl::sub_group sg, typename Ops::C& c, const sycl::local_accessor<float, 1>& st,
                         size_t off) {
#if STRATA_AMD_WMMA
        if constexpr (P == 3) {
            const int lo = (int) (sg.get_local_linear_id() & 15), hi = (int) (sg.get_local_linear_id() >> 4);
            for (int e = 0; e < 8; ++e) c.v[e] = st[off + (size_t) w::row_of(hi, e) * D + lo];
            return;
        }
#endif
        if constexpr (P != 3) mx::joint_matrix_load(sg, c, at(st, off), D, mx::layout::row_major);
    }

    // Xe2's instance takes the large register file: the output's 8 accumulator tiles and Q's 8 operand tiles
    auto get(syclex::properties_tag) const {
        if constexpr (P == 0)
            return syclex::properties{sycl::ext::intel::experimental::grf_size<256>,
                                      syclex::sub_group_size<STRATA_SUB_GROUP(SG)>};
        else
            return syclex::properties{syclex::sub_group_size<STRATA_SUB_GROUP(SG)>};
    }

    void operator()(sycl::nd_item<2> it) const {
#if defined(__SYCL_DEVICE_ONLY__)
        if constexpr (!built(P)) {
            return;   // another GPU's instance: never launched on this target
        } else {
        constexpr float TAU = 8.0f;   // a row's reference moves only when a score passes it by more: p <= e^8
        sycl::sub_group sg = it.get_sub_group();
        const int g = (int) sg.get_group_linear_id(), lane = (int) sg.get_local_linear_id();
        const int nsg = (int) (it.get_local_range(1) / SG);
        const int64_t h = (int64_t) it.get_group(0), my0 = row0 + ((int64_t) it.get_group(1) * nsg + g) * TM;
        if (my0 >= row_end) return;
        const int64_t row = heads * D;
        const size_t soff = (size_t) g * SREG, poff = (size_t) g * TM * PLD, roff = (size_t) g * 2 * TM;
        for (int e = lane; e < TM; e += SG) {
            rows[roff + e] = -INFINITY;
            rows[roff + TM + e] = 0.0f;
        }
        typename Ops::A qa[D / TK];
        for (int kk = 0; kk < D / TK; ++kk) Ops::load(sg, qa[kk], q16 + my0 * row + h * D + (int64_t) kk * TK, row);
        typename Ops::C o[NT];
        for (int nt = 0; nt < NT; ++nt) Ops::zero(sg, o[nt]);
        const int r = lane / L, part = lane % L;               // the softmax's row and its lanes' share
        const int64_t t = my0 + r;
        const int64_t lim = t >= T ? 0 : (t < n_text ? (key_lim ? key_lim[t] : t + 1) : T);
        const int64_t last = std::min(T, my0 + TM) - 1;
        const int64_t keys = last < n_text ? (key_lim ? key_lim[last] : last + 1) : T;   // the limits never fall
        sycl::group_barrier(sg);
        for (int64_t j0 = 0; j0 < keys; j0 += CH) {
            const uint16_t* kc0 = k16 + j0 * row + h * D;
            if constexpr (P == 0 || P == 1) {
                // Intel's: the next chunk's keys and values into L1 while this one is worked, in blocks of 32 rows x
                // 32 columns shared out over the work-group's sub-groups (they read the same chunks)
                const int64_t jn = j0 + CH;
                if (jn + CH <= keys) {
                    constexpr int PR = 32, PC = 32, NB = (CH / PR) * (D / PC);
                    for (int b = g; b < 2 * NB; b += nsg) {
                        const uint16_t* base = (b < NB ? k16 : v16) + (jn + (int64_t) ((b % NB) / (D / PC)) * PR) * row +
                                               h * D + (int64_t) ((b % NB) % (D / PC)) * PC;
                        mx::joint_matrix_prefetch<PR, PC>(sg, const_cast<sycl::half*>(
                                                                  reinterpret_cast<const sycl::half*>(base)),
                                                          row, mx::layout::row_major,
                                                          syclex::properties{syclex::prefetch_hint_L1});
                    }
                }
            }
            for (int nt = 0; nt < CH / TN; ++nt) {   // S = Q . K^T, K's tiles from global memory (the caches)
                typename Ops::C c;
                Ops::zero(sg, c);
                for (int kk = 0; kk < D / TK; ++kk) {
                    typename Ops::Bc b;
                    Ops::load(sg, b, kc0 + (int64_t) nt * TN * row + (int64_t) kk * TK, row);
                    Ops::mad(sg, c, qa[kk], b);
                }
                Ops::store(sg, c, at(st, soff + (size_t) nt * TN), SLD);
            }
            sycl::group_barrier(sg);
            // the softmax: L lanes a row, CW contiguous scores each
            float s[CW];
            float mx_ = -INFINITY;
            const int c0 = part * CW;
            for (int i = 0; i < CW; i += 4) {
                const auto v = *reinterpret_cast<const sycl::vec<float, 4>*>(&st[soff + (size_t) r * SLD + c0 + i]);
                for (int u = 0; u < 4; ++u) {
                    s[i + u] = j0 + c0 + i + u < lim ? v[u] * scale : -INFINITY;
                    mx_ = sycl::fmax(mx_, s[i + u]);
                }
            }
            for (int off = 1; off < L; off <<= 1) mx_ = sycl::fmax(mx_, sycl::permute_group_by_xor(sg, mx_, off));
            const float m_old = rows[roff + r];
            const bool move = mx_ > m_old + TAU || (m_old == -INFINITY && mx_ > -INFINITY);
            const float m_new = move ? mx_ : m_old;
            if (sycl::any_of_group(sg, move)) {
                // the rows whose reference moved scale their output and sum by exp(m_old - m_new), through local
                // memory: the accumulators' layout is the vendor's
                const float corr = move && m_old > -INFINITY ? sycl::exp(m_old - m_new) : 1.0f;
                sycl::group_barrier(sg);
                if (part == 0) rows[roff + TM + r] *= corr;
                float f[TM];
                for (int i = 0; i < TM; ++i) f[i] = sycl::select_from_group(sg, corr, (size_t) i * L);
                for (int nt = 0; nt < NT; ++nt) Ops::store(sg, o[nt], at(st, soff + (size_t) nt * TN), D);
                sycl::group_barrier(sg);
                for (int e = lane; e < TM * D; e += SG) st[soff + e] *= f[e / D];
                sycl::group_barrier(sg);
                for (int nt = 0; nt < NT; ++nt) load_acc(sg, o[nt], st, soff + (size_t) nt * TN);
            }
            float sum = 0.0f;
            for (int i = 0; i < CW; i += 8) {
                sycl::vec<sycl::half, 8> pv;
                for (int u = 0; u < 8; ++u) {
                    // native::exp: P is FP16 anyway (the block's error unchanged; B70 11.4 -> 10.3 ms)
                    const float p = m_new == -INFINITY ? 0.0f : sycl::native::exp(s[i + u] - m_new);
                    sum += p;
                    pv[u] = sycl::half(p);
                }
                *reinterpret_cast<sycl::vec<sycl::half, 8>*>(&ps[poff + (size_t) r * PLD + c0 + i]) = pv;
            }
            for (int off = 1; off < L; off <<= 1) sum += sycl::permute_group_by_xor(sg, sum, off);
            if (part == 0) {
                rows[roff + r] = m_new;
                rows[roff + TM + r] += sum;
            }
            sycl::group_barrier(sg);
            const uint16_t* vc0 = v16 + j0 * row + h * D;
            // O += P . V, V's tiles from global memory.  Intel's shapes read a tile of P once for the output's NT tiles
            // (B70, a 1024 x 1024 block: 12.5 -> 11.9 ms); the RTX 4070 was 1.4% slower that way and keeps the order
            // of reading it again for each
            if constexpr (P == 0 || P == 1) {
                for (int kc = 0; kc < CH / TK; ++kc) {
                    typename Ops::A pa;
                    Ops::load(sg, pa, at(ps, poff + (size_t) kc * TK), PLD);
                    for (int nt = 0; nt < NT; ++nt) {
                        typename Ops::Br b;
                        Ops::load(sg, b, vc0 + (int64_t) kc * TK * row + (int64_t) nt * TN, row);
                        Ops::mad(sg, o[nt], pa, b);
                    }
                }
            } else {
                for (int nt = 0; nt < NT; ++nt) {
                    for (int kc = 0; kc < CH / TK; ++kc) {
                        typename Ops::A pa;
                        typename Ops::Br b;
                        Ops::load(sg, pa, at(ps, poff + (size_t) kc * TK), PLD);
                        Ops::load(sg, b, vc0 + (int64_t) kc * TK * row + (int64_t) nt * TN, row);
                        Ops::mad(sg, o[nt], pa, b);
                    }
                }
            }
            sycl::group_barrier(sg);
        }
        for (int nt = 0; nt < NT; ++nt) Ops::store(sg, o[nt], at(st, soff + (size_t) nt * TN), D);
        sycl::group_barrier(sg);
        for (int e = lane; e < TM * D; e += SG) {
            const int rr = e / D;
            const int64_t tt = my0 + rr;
            if (tt < row_end && tt >= first)
                out16[(tt - first) * row + h * D + e % D] = f16_from_f32(st[soff + e] / rows[roff + TM + rr]);
        }
        }
#else
        (void) it;
#endif
    }
};

bool env_on(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && std::strtol(v, nullptr, 10) != 0;
}

// the path for the device: 0-3 as Path<>, -1 none (the plain kernel)
int pick(const sycl::device& d) {
    if (env_on("STRATA_NO_XMX")) return -1;
    if (strata::kernels::xe::amd_wmma(d)) return 3;
    const auto sgs = d.get_info<sycl::info::device::sub_group_sizes>();
    struct S { int m, n, k, sg; };
    const S shapes[] = {{8, 16, 16, 16}, {8, 8, 16, 8}, {16, 16, 16, 32}};
    for (int i = 0; i < 3; ++i) {
        const S& s = shapes[i];
        if (std::find(sgs.begin(), sgs.end(), (size_t) s.sg) == sgs.end()) continue;
        for (const auto& c : strata::kernels::matrix_combinations(d)) {
            const bool m = c.msize == (size_t) s.m || (c.msize == 0 && c.max_msize >= (size_t) s.m);
            const bool n = c.nsize == (size_t) s.n || (c.nsize == 0 && c.max_nsize >= (size_t) s.n);
            const bool k = c.ksize == (size_t) s.k || (c.ksize == 0 && c.max_ksize >= (size_t) s.k);
            if (c.atype == mx::matrix_type::fp16 && c.btype == mx::matrix_type::fp16 &&
                c.ctype == mx::matrix_type::fp32 && c.dtype == mx::matrix_type::fp32 && m && n && k)
                return i;
        }
    }
    return -1;
}

// The most sub-groups a work-group the device's local memory and work-group size take, up to 8.
template<int P>
int nsg_most(const sycl::device& d) {
    using K = AttnKernel<P>;
    const size_t local = d.get_info<sycl::info::device::local_mem_size>();
    const size_t wg = d.get_info<sycl::info::device::max_work_group_size>();
    int nsg = 8;
    while (nsg > 1 && ((size_t) nsg * K::SG_BYTES > local || (size_t) nsg * K::SG > wg)) nsg /= 2;
    return nsg;
}

template<int P>
sycl::event launch(sycl::queue& q, const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t T,
                   int64_t heads, int64_t n_text, const int32_t* key_lim, uint16_t* out16, int nsg, int64_t first,
                   int64_t row_end) {
    using K = AttnKernel<P>;
    const int64_t bq = (int64_t) nsg * K::TM, row0 = first / K::TM * K::TM;   // its tiles stay inside the rows
    const size_t groups = (size_t) ((row_end - row0 + bq - 1) / bq);
    return q.submit([&](sycl::handler& h) {
        K k{q16, k16, v16, out16, T, heads, n_text, key_lim, 1.0f / std::sqrt((float) D), row0, first, row_end,
            sycl::local_accessor<sycl::half, 1>(sycl::range<1>((size_t) nsg * K::TM * K::PLD), h),
            sycl::local_accessor<float, 1>(sycl::range<1>((size_t) nsg * K::SREG), h),
            sycl::local_accessor<float, 1>(sycl::range<1>((size_t) nsg * 2 * K::TM), h)};
        h.parallel_for(sycl::nd_range<2>({(size_t) heads, groups * nsg * K::SG}, {1, (size_t) nsg * K::SG}), k);
    });
}

}  // namespace

const char* qi_attention_path() {
    static const char* names[] = {"matrix engines 8x16x16", "matrix engines 8x8x16", "matrix engines 16x16x16",
                                  "WMMA 16x16x16"};
    const int p = pick(core::Runtime::get().device());
    return p < 0 ? "plain (FP32)" : names[p];
}

void qi_attention(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t T, int64_t heads,
                  int64_t head_dim, int64_t n_text, uint16_t* out16, float* scratch, void* stream,
                  const int32_t* key_lim, int64_t first) {
    if (T <= first) return;
    if (head_dim != D) throw std::invalid_argument("qi_attention: head_dim 128");
    // the image rows (they see every key) through the GPU maker's kernel where there is one; the prefix's own rows, a
    // few dozen, by the kernel below
    int64_t row_end = T;
    const int64_t r0 = std::max(first, n_text);
    if (T - r0 >= 256 && qi_sdpa(q16, k16, v16, r0, T, T, heads, head_dim, out16 + (r0 - first) * heads * head_dim,
                                 stream)) {
        if (first >= n_text) {
            if (!stream) core::Runtime::get().stream(stream).wait();
            return;
        }
        row_end = n_text;
    }
    sycl::queue& q = core::Runtime::get().stream(stream);
    static core::PerDevice<int> per_device;
    const int p = per_device.get(q.get_device(), [&q] { return pick(q.get_device()); });
    // The sub-groups a work-group: more share the keys' tiles in the caches, fewer leave the GPU freer to overlap
    // work-groups.  The fastest of the most that fit, a half and a quarter of it, timed once a GPU on the first
    // attention of 2048 rows or more (the images' own inputs; a small first image would not show it): 1024 x 1024
    // block, the B70 4 (11.88 -> 11.44 ms), the RTX 4070 2 (11.55 -> 10.53 ms).  Until then, the most.
    static core::PerDevice<std::shared_ptr<std::atomic<int>>> tuned_of;
    std::atomic<int>& tuned = *tuned_of.get(q.get_device(), [] { return std::make_shared<std::atomic<int>>(0); });
    auto run = [&](int nsg) -> sycl::event {
        switch (p) {
        case 0: return launch<0>(q, q16, k16, v16, T, heads, n_text, key_lim, out16, nsg, first, row_end);
        case 1: return launch<1>(q, q16, k16, v16, T, heads, n_text, key_lim, out16, nsg, first, row_end);
        case 2: return launch<2>(q, q16, k16, v16, T, heads, n_text, key_lim, out16, nsg, first, row_end);
        default: return launch<3>(q, q16, k16, v16, T, heads, n_text, key_lim, out16, nsg, first, row_end);
        }
    };
    const int most = p == 0 ? nsg_most<0>(q.get_device()) : p == 1 ? nsg_most<1>(q.get_device())
                   : p == 2 ? nsg_most<2>(q.get_device()) : p == 3 ? nsg_most<3>(q.get_device()) : 1;
    int nsg = tuned.load();
    if (p >= 0 && nsg == 0 && T >= 2048) {
        q.wait();
        double best = 1e30;
        std::string said;
        for (int n = most; n >= std::max(1, most / 4); n /= 2) {
            run(n).wait();   // its kernel built, its caches warm
            const auto t0 = std::chrono::steady_clock::now();
            run(n).wait();
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            said += (said.empty() ? "" : ", ") + std::to_string(n) + ": " + std::to_string(ms).substr(0, 5) + " ms";
            if (ms < best) { best = ms; nsg = n; }
        }
        tuned.store(nsg);
        std::fprintf(stderr, "strata-image: the attention takes %d sub-groups a work-group (timed at %lld rows: %s)\n",
                     nsg, (long long) T, said.c_str());
    }
    if (nsg == 0) nsg = most;
    sycl::event e;
    switch (p) {
    case 0: case 1: case 2: case 3: e = run(nsg); break;
    default:
        qi_attention_plain(q16, k16, v16, T, heads, head_dim, n_text, scratch, stream, key_lim);
        qi_to_f16(scratch + first * heads * head_dim, (row_end - first) * heads * head_dim, out16, stream);
        return;
    }
    if (!stream) core::Runtime::get().wait(e, "qi_attention");
}

}  // namespace strata::image
