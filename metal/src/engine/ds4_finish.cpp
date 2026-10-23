// metal/src/engine/ds4_finish.cpp - the host side of ds4_finish (ds4_kernels.cu:1894, finish_kernel<false>).
//
// The kernel takes the engine's own device pointer for x as a bound buffer plus an offset, and the Ds4Finish struct as a constant
// buffer whose pointer fields are the engine's gpuAddresses (Metal lets a kernel dereference them - ss the repo's note that
// "kernels may take raw gpuAddress pointers").  Strides and counts travel as plain values.
//
// The mrope (rd > 0) and rounding (quant != 0) branches are not written yet, and neither is the KV pack (pdst): this function
// returns false for those, so the engine reports the launch line instead of a step being skipped.
#include "mtl_rt.h"

#include <strata/gpu/runtime.hpp>

#include <cstdio>
#include <cstdint>
#include <cstring>

namespace strata::dsv41::metal_host {

// mirrors metal/src/engine/msl/ds4_init.metal's Ds4Finish (Metal has no 'long long': 64-bit is 'long')
struct GpuFinish {
    uint64_t w;          // device float*  (the engine's gpuAddress, 0 = none)
    float eps;
    int rd, pos, inverse, vpp, pos_step;
    uint64_t freqs;      // device float*
    int quant;
    uint64_t dst;        // device ushort*
    int64_t dst_stride;
    uint64_t pdst;       // device uchar*
    int64_t pdst_stride;
};

namespace {
mtl::Context* g_cx = nullptr;
int g_lib = -1;
int g_lib_iq = -1;

bool ensure() {
    if (g_cx) return g_lib >= 0;
    g_cx = new mtl::Context();
    g_lib = g_cx->compile({"../src/engine/msl/ds4_init.metal"}, "", false);
    if (g_lib < 0) std::fprintf(stderr, "dsv41 metal: compiling ds4_init.metal failed\n");
    g_lib_iq = g_cx->compile({"../src/engine/msl/ds4_iq.metal"}, "", false);
    if (g_lib_iq < 0) std::fprintf(stderr, "dsv41 metal: compiling ds4_iq.metal failed\n");
    return g_lib >= 0;
}
}  // namespace

// the engine's Finish, as ds4_kernels.cuh declares it
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

bool finish(float* x, int n_vec, int len, long long stride, const FinishIn& f) {
    if (n_vec <= 0 || len <= 0) return true;
    // finish_kernel<true> (the KV pack: the row quantised to codes + group scales) or <false>; the <false> rounding
    // branch (quant without pdst) is the one still unwritten
    const bool pack = f.pdst != nullptr;
    if (!ensure()) return false;
    void* hx = strata::gpu::host_view(x);
    if (!hx) { std::fprintf(stderr, "dsv41 metal: finish has no host view\n"); return false; }
    const size_t page = 16384;
    const size_t bytes = (size_t) (n_vec - 1) * (size_t) stride * sizeof(float) + (size_t) len * sizeof(float);
    const size_t base = ((uintptr_t) hx) & ~(uintptr_t) (page - 1);
    const size_t off = ((uintptr_t) hx) - base;
    mtl::Buffer bx = g_cx->wrap((void*) base, ((bytes + off + page - 1) / page) * page);
    if (!bx.handle) { std::fprintf(stderr, "dsv41 metal: finish wrap failed\n"); return false; }

    GpuFinish gf{};
    gf.w = (uint64_t) (uintptr_t) f.w;
    gf.eps = f.eps;
    gf.rd = f.rd; gf.pos = f.pos; gf.inverse = f.inverse; gf.vpp = f.vpp; gf.pos_step = f.pos_step;
    gf.freqs = (uint64_t) (uintptr_t) f.freqs;
    gf.quant = f.quant;
    gf.dst = (uint64_t) (uintptr_t) f.dst;
    gf.dst_stride = f.dst_stride;
    gf.pdst = (uint64_t) (uintptr_t) f.pdst;
    gf.pdst_stride = f.pdst_stride;

    mtl::Pipeline pl;
    try {
        pl = g_cx->pipeline(g_lib, pack ? "ds4_finish_pack" : "ds4_finish");
    } catch (...) {
        return false;
    }
    // one dispatch, one threadgroup a vector (the kernel's blockIdx.x is threadgroup_position_in_grid)
    mtl::Dispatch d;
    d.p = &pl;
    d.args = {mtl::Arg::B(bx, off), mtl::Arg::V((long) stride), mtl::Arg::V((unsigned) len), mtl::Arg::V(gf)};
    d.grid[0] = (uint32_t) n_vec;
    d.grid[1] = d.grid[2] = 1;
    d.tg[0] = 256;
    d.tg[1] = d.tg[2] = 1;
    g_cx->run({d});
    return true;
}


// ds4_kernels.cu:1677 - hc_post_inplace: h[t][k][e] = post[t][k]*y[t][e] + sum_j comb[t][j][k]*h_old[t][j][e]
bool hc_post(const float* y, float* h, const float* post, const float* comb, int dim, int hc, int T) {
    if (!ensure()) return false;
    void* hy = strata::gpu::host_view((void*) y);
    void* hh = strata::gpu::host_view(h);
    if (!hy || !hh) { std::fprintf(stderr, "dsv41 metal: hc_post has no host view\n"); return false; }
    const size_t page = 16384;
    auto wrap = [&](void* host, size_t bytes) {   // the pages the buffer itself spans (see q8_quant_rows' note)
        const size_t base = (size_t) ((uintptr_t) host) & ~(size_t) (page - 1);
        const size_t end = (size_t) ((uintptr_t) host) + bytes;
        return g_cx->wrap((void*) base, ((end - base + page - 1) / page) * page);
    };
    auto off = [&](void* host) { return (size_t) ((uintptr_t) host - (((uintptr_t) host) & ~(uintptr_t) (page - 1))); };
    mtl::Buffer by = wrap(hy, (size_t) T * dim * 4 );
    mtl::Buffer bh = wrap(hh, (size_t) T * hc * dim * 4 );
    if (!by.handle || !bh.handle) return false;
    // post is [T][8], comb is [T][64], straight from the caller's host/shared memory
    mtl::Buffer bp = g_cx->alloc((size_t) T * 8 * 4);
    mtl::Buffer bc = g_cx->alloc((size_t) T * 64 * 4);
    if (!bp.ptr || !bc.ptr) return false;
    std::memcpy(bp.ptr, post, (size_t) T * 8 * 4);
    std::memcpy(bc.ptr, comb, (size_t) T * 64 * 4);
    mtl::Pipeline pl;
    try { pl = g_cx->pipeline(g_lib, "ds4_hc_post"); } catch (...) { return false; }
    mtl::Dispatch d;
    d.p = &pl;
    d.args = {mtl::Arg::B(by, off(hy)), mtl::Arg::B(bh, off(hh)), mtl::Arg::B(bp), mtl::Arg::B(bc),
              mtl::Arg::V((unsigned) dim), mtl::Arg::V((unsigned) hc)};
    d.grid[0] = (uint32_t) ((dim + 255) / 256);
    d.grid[1] = (uint32_t) T;
    d.grid[2] = 1;
    d.tg[0] = 256; d.tg[1] = d.tg[2] = 1;
    g_cx->run({d});
    g_cx->free(bp);
    g_cx->free(bc);
    return true;
}


// ds4_kernels.cu:1692 - hc_mix_scale: mixes[t][i] *= rsqrt(sum(h[t])^2 / hcdim + eps)
bool hc_mix_scale(const float* h, int hcdim, int n_mix, int T, float eps, float* mixes) {
    if (!ensure()) return false;
    void* hh = strata::gpu::host_view((void*) h);
    void* hm = strata::gpu::host_view((void*) mixes);
    if (!hh || !hm) { std::fprintf(stderr, "dsv41 metal: hc_mix_scale has no host view\n"); return false; }
    const size_t page = 16384;
    auto wrap = [&](void* host, size_t bytes) {   // the pages the buffer itself spans (see q8_quant_rows' note)
        const size_t base = (size_t) ((uintptr_t) host) & ~(size_t) (page - 1);
        const size_t end = (size_t) ((uintptr_t) host) + bytes;
        return g_cx->wrap((void*) base, ((end - base + page - 1) / page) * page);
    };
    auto off = [&](void* host) { return (size_t) ((uintptr_t) host - (((uintptr_t) host) & ~(uintptr_t) (page - 1))); };
    mtl::Buffer bh = wrap(hh, (size_t) T * hcdim * 4 );
    mtl::Buffer bm = wrap(hm, (size_t) T * 32 * 4 );
    if (!bh.handle || !bm.handle) return false;
    mtl::Pipeline pl;
    try { pl = g_cx->pipeline(g_lib, "ds4_hc_mix_scale"); } catch (...) { return false; }
    mtl::Dispatch d;
    d.p = &pl;
    d.args = {mtl::Arg::B(bh, off(hh)), mtl::Arg::B(bm, off(hm)), mtl::Arg::V((unsigned) hcdim),
              mtl::Arg::V((unsigned) n_mix), mtl::Arg::V(eps)};
    d.grid[0] = (uint32_t) T;
    d.grid[1] = d.grid[2] = 1;
    d.tg[0] = 256; d.tg[1] = d.tg[2] = 1;
    g_cx->run({d});
    return true;
}


// ds4_kernels.cu:1901 - hc_mix: mixes[t][m] = (h[t] . fn[m]) * rsqrt(mean(h[t]^2) + eps)
bool hc_mix(const float* h, const float* fn, int hcdim, int n_mix, int T, float eps, float* mixes) {
    if (!ensure()) return false;
    void* hh = strata::gpu::host_view((void*) h);
    void* hf = strata::gpu::host_view((void*) fn);
    void* hm = strata::gpu::host_view((void*) mixes);
    if (!hh || !hf || !hm) { std::fprintf(stderr, "dsv41 metal: hc_mix has no host view\n"); return false; }
    const size_t page = 16384;
    auto wrap = [&](void* host, size_t bytes) {   // the pages the buffer itself spans (see q8_quant_rows' note)
        const size_t base = (size_t) ((uintptr_t) host) & ~(size_t) (page - 1);
        const size_t end = (size_t) ((uintptr_t) host) + bytes;
        return g_cx->wrap((void*) base, ((end - base + page - 1) / page) * page);
    };
    auto off = [&](void* host) { return (size_t) ((uintptr_t) host - (((uintptr_t) host) & ~(uintptr_t) (page - 1))); };
    mtl::Buffer bh = wrap(hh, (size_t) T * hcdim * 4 );
    mtl::Buffer bf = wrap(hf, (size_t) n_mix * hcdim * 4 );
    mtl::Buffer bm = wrap(hm, (size_t) T * 32 * 4 );
    if (!bh.handle || !bf.handle || !bm.handle) return false;
    mtl::Pipeline pl;
    try { pl = g_cx->pipeline(g_lib, "ds4_hc_mix"); } catch (...) { return false; }
    mtl::Dispatch d;
    d.p = &pl;
    d.args = {mtl::Arg::B(bh, off(hh)), mtl::Arg::B(bf, off(hf)), mtl::Arg::B(bm, off(hm)),
              mtl::Arg::V((unsigned) hcdim), mtl::Arg::V(eps)};
    d.grid[0] = (uint32_t) n_mix;
    d.grid[1] = (uint32_t) T;
    d.grid[2] = 1;
    d.tg[0] = 256; d.tg[1] = d.tg[2] = 1;
    g_cx->run({d});
    return true;
}


// ds4_kernels.cu:1908 - hc_split: the sinkhorn normalisation (scalar path; see the MSL comment)
bool hc_split(const float* mixes, const float* scale, const float* base, int hc, int iters, float hc_eps, int T,
              float* pre, float* post, float* comb) {
    if (!ensure()) return false;
    void* hm = strata::gpu::host_view((void*) mixes);
    void* hp = strata::gpu::host_view((void*) pre);
    void* ho = strata::gpu::host_view((void*) post);
    void* hc_ = strata::gpu::host_view((void*) comb);
    if (!hm || !hp || !ho || !hc_) { std::fprintf(stderr, "dsv41 metal: hc_split has no host view\n"); return false; }
    const size_t page = 16384;
    auto wrap = [&](void* host, size_t bytes) {   // the pages the buffer itself spans (see q8_quant_rows' note)
        const size_t base = (size_t) ((uintptr_t) host) & ~(size_t) (page - 1);
        const size_t end = (size_t) ((uintptr_t) host) + bytes;
        return g_cx->wrap((void*) base, ((end - base + page - 1) / page) * page);
    };
    auto off = [&](void* host) { return (size_t) ((uintptr_t) host - (((uintptr_t) host) & ~(uintptr_t) (page - 1))); };
    mtl::Buffer bm = wrap(hm, (size_t) T * 32 * 4 );
    mtl::Buffer bp = wrap(hp, (size_t) T * 8 * 4 );
    mtl::Buffer bo = wrap(ho, (size_t) T * 8 * 4 );
    mtl::Buffer bc = wrap(hc_, (size_t) T * 64 * 4 );
    if (!bm.handle || !bp.handle || !bo.handle || !bc.handle) return false;
    // scale and base come from the engine like every other operand: reach them through host_view, not as raw host pointers
    void* hs = strata::gpu::host_view((void*) scale);
    void* hb = strata::gpu::host_view((void*) base);
    if (!hs || !hb) { std::fprintf(stderr, "dsv41 metal: hc_split scale/base have no host view\n"); return false; }
    const size_t nbase = (size_t) (2 * hc + hc * hc);
    mtl::Buffer bs = wrap(hs, 256);
    mtl::Buffer bb = wrap(hb, (nbase * 4 + 65535) / 65536 * 65536 );
    if (!bs.handle || !bb.handle) return false;
    mtl::Pipeline pl;
    try { pl = g_cx->pipeline(g_lib, "ds4_hc_split"); } catch (...) { return false; }
    mtl::Dispatch d;
    d.p = &pl;
    d.args = {mtl::Arg::B(bm, off(hm)), mtl::Arg::B(bs, off(hs)), mtl::Arg::B(bb, off(hb)), mtl::Arg::B(bp, off(hp)),
              mtl::Arg::B(bo, off(ho)), mtl::Arg::B(bc, off(hc_)), mtl::Arg::V((unsigned) hc),
              mtl::Arg::V((unsigned) iters), mtl::Arg::V((float) hc_eps)};
    d.grid[0] = (uint32_t) T;
    d.grid[1] = d.grid[2] = 1;
    d.tg[0] = 32; d.tg[1] = d.tg[2] = 1;
    g_cx->run({d});
    return true;
}


// ds4_kernels.cu:1938 - moe_sort: count each virtual slot's picks, prefix-sum the counts, fill in token order.
// Three dispatches on one command buffer, so their order is the order of the calls (serial dispatch).
bool moe_sort(const int* idx, int T, int k, const int* vs, int n_vs, int* bounds, int* row_tok, int* pos) {
    if (!ensure()) return false;
    void* hi = strata::gpu::host_view((void*) idx);
    void* hv = strata::gpu::host_view((void*) vs);
    void* hb = strata::gpu::host_view((void*) bounds);
    void* hr = strata::gpu::host_view((void*) row_tok);
    void* hp = strata::gpu::host_view((void*) pos);
    if (!hi || !hv || !hb || !hr || !hp) { std::fprintf(stderr, "dsv41 metal: moe_sort has no host view\n"); return false; }
    const size_t page = 16384;
    auto wrap = [&](void* host, size_t bytes) {   // the pages the buffer itself spans (see q8_quant_rows' note)
        const size_t base = (size_t) ((uintptr_t) host) & ~(size_t) (page - 1);
        const size_t end = (size_t) ((uintptr_t) host) + bytes;
        return g_cx->wrap((void*) base, ((end - base + page - 1) / page) * page);
    };
    auto off = [&](void* host) { return (size_t) ((uintptr_t) host - (((uintptr_t) host) & ~(uintptr_t) (page - 1))); };
    const size_t t_bytes = (size_t) T * k * 4 + 65536;
    mtl::Buffer bi = wrap(hi, t_bytes);
    mtl::Buffer bv = wrap(hv, (size_t) n_vs * 4 );
    mtl::Buffer bb = wrap(hb, (size_t) (n_vs + 2) * 4 );
    mtl::Buffer br = wrap(hr, t_bytes);
    mtl::Buffer bp = wrap(hp, t_bytes);
    if (!bi.handle || !bv.handle || !bb.handle || !br.handle || !bp.handle) return false;
    mtl::Pipeline pc, pf, ps;
    try {
        pc = g_cx->pipeline(g_lib, "ds4_moe_sort_count");
        pf = g_cx->pipeline(g_lib, "ds4_moe_sort_fill");
        ps = g_cx->pipeline(g_lib, "ds4_bounds_scan");
    } catch (...) { return false; }
    // 1. counts
    mtl::Dispatch d1;
    d1.p = &pc;
    d1.args = {mtl::Arg::B(bi, off(hi)), mtl::Arg::V((unsigned) T), mtl::Arg::V((unsigned) k),
               mtl::Arg::B(bv, off(hv)), mtl::Arg::B(bb, off(hb))};
    d1.grid[0] = (uint32_t) n_vs;
    d1.grid[1] = d1.grid[2] = 1;
    d1.tg[0] = 256; d1.tg[1] = d1.tg[2] = 1;
    g_cx->run({d1});
    // 2. the exclusive prefix over the slots
    mtl::Dispatch d2;
    d2.p = &ps;
    d2.args = {mtl::Arg::B(bb, off(hb)), mtl::Arg::V((unsigned) n_vs)};
    d2.grid[0] = 1;
    d2.grid[1] = d2.grid[2] = 1;
    d2.tg[0] = 32; d2.tg[1] = d2.tg[2] = 1;
    g_cx->run({d2});
    // 3. the fill
    mtl::Dispatch d3;
    d3.p = &pf;
    d3.args = {mtl::Arg::B(bi, off(hi)), mtl::Arg::V((unsigned) T), mtl::Arg::V((unsigned) k),
               mtl::Arg::B(bv, off(hv)), mtl::Arg::B(bb, off(hb)), mtl::Arg::B(br, off(hr)), mtl::Arg::B(bp, off(hp))};
    d3.grid[0] = (uint32_t) n_vs;
    d3.grid[1] = d3.grid[2] = 1;
    d3.tg[0] = 256; d3.tg[1] = d3.tg[2] = 1;
    g_cx->run({d3});
    return true;
}


// ds4_kernels.cu:1690 - the Engram row decoder (fmt 2 = IQ4_NL, this model's tables)
bool engram_dequant_tok(const uint8_t* src, size_t per, int cols, int T, int hd, float* out, int fmt) {
    if (T <= 0) return true;
    if (!ensure()) return false;
    void* hs = strata::gpu::host_view((void*) src);
    void* ho = strata::gpu::host_view((void*) out);
    if (!hs || !ho) { std::fprintf(stderr, "dsv41 metal: engram_dequant_tok has no host view\n"); return false; }
    const size_t page = 16384;
    auto wrap = [&](void* host, size_t bytes) {   // the pages the buffer itself spans (see q8_quant_rows' note)
        const size_t base = (size_t) ((uintptr_t) host) & ~(size_t) (page - 1);
        const size_t end = (size_t) ((uintptr_t) host) + bytes;
        return g_cx->wrap((void*) base, ((end - base + page - 1) / page) * page);
    };
    auto off = [&](void* host) { return (size_t) ((uintptr_t) host - (((uintptr_t) host) & ~(uintptr_t) (page - 1))); };
    const size_t rows = (size_t) T * (size_t) cols;
    mtl::Buffer bs = wrap(hs, (size_t) T * per );
    mtl::Buffer bo = wrap(ho, rows * (size_t) hd * 4 );
    if (!bs.handle || !bo.handle) return false;
    mtl::Pipeline pl;
    try { pl = g_cx->pipeline(g_lib, "ds4_engram_dequant"); } catch (...) { return false; }
    mtl::Dispatch d;
    d.p = &pl;
    d.args = {mtl::Arg::B(bs, off(hs)), mtl::Arg::V((unsigned long) per), mtl::Arg::V((unsigned) cols),
              mtl::Arg::V((unsigned) hd), mtl::Arg::B(bo, off(ho)), mtl::Arg::V((int) fmt)};
    d.grid[0] = (uint32_t) rows;
    d.grid[1] = d.grid[2] = 1;
    d.tg[0] = 256; d.tg[1] = d.tg[2] = 1;
    g_cx->run({d});
    return true;
}


// ds4_kernels.cu:341 - hc_pre_norm: a = sum_s mix[s]*h[s][e]; x = a * rsqrt(mean(a^2) + eps) (x w when given)
bool hc_pre_norm(const float* h, const float* mix, const float* w, int dim, int hc, int T, float eps, float* x) {
    if (!ensure()) return false;
    void* hh = strata::gpu::host_view((void*) h);
    void* hm = strata::gpu::host_view((void*) mix);
    void* hx = strata::gpu::host_view((void*) x);
    if (!hh || !hm || !hx) { std::fprintf(stderr, "dsv41 metal: hc_pre_norm has no host view\n"); return false; }
    const size_t page = 16384;
    auto wrap = [&](void* host, size_t bytes) {   // the pages the buffer itself spans (see q8_quant_rows' note)
        const size_t base = (size_t) ((uintptr_t) host) & ~(size_t) (page - 1);
        const size_t end = (size_t) ((uintptr_t) host) + bytes;
        return g_cx->wrap((void*) base, ((end - base + page - 1) / page) * page);
    };
    auto off = [&](void* host) { return (size_t) ((uintptr_t) host - (((uintptr_t) host) & ~(uintptr_t) (page - 1))); };
    mtl::Buffer bh = wrap(hh, (size_t) T * hc * dim * 4 );
    mtl::Buffer bm = wrap(hm, (size_t) T * 8 * 4 );
    mtl::Buffer bx = wrap(hx, (size_t) T * dim * 4 );
    if (!bh.handle || !bm.handle || !bx.handle) return false;
    mtl::Buffer bw;
    size_t w_off = 0;
    if (w) {
        void* hw = strata::gpu::host_view((void*) w);
        if (!hw) { std::fprintf(stderr, "dsv41 metal: hc_pre_norm weight has no host view\n"); return false; }
        bw = wrap(hw, (size_t) dim * 4 );
        w_off = off(hw);
        if (!bw.handle) return false;
    } else {
        bw = bx;   // never read when has_w == 0
    }
    mtl::Pipeline pl;
    try { pl = g_cx->pipeline(g_lib, "ds4_hc_pre_norm"); } catch (...) { return false; }
    mtl::Dispatch d;
    d.p = &pl;
    d.args = {mtl::Arg::B(bh, off(hh)), mtl::Arg::B(bm, off(hm)), mtl::Arg::B(bw, w_off),
              mtl::Arg::V((unsigned) dim), mtl::Arg::V((unsigned) hc), mtl::Arg::V(eps), mtl::Arg::B(bx, off(hx)),
              mtl::Arg::V((unsigned) (w ? 1 : 0))};
    d.grid[0] = (uint32_t) T;
    d.grid[1] = d.grid[2] = 1;
    d.tg[0] = 1024; d.tg[1] = d.tg[2] = 1;
    g_cx->run({d});
    return true;
}


// ds4_gguf.cu:186 - the activation quantiser the expert matmuls consume (Q8_0: 32 values, one scale)
bool q8_quant_rows(const float* x, size_t ldx, int rows, int k, int8_t* q, float* d) {
    if (rows <= 0 || k <= 0) return true;
    if (!ensure()) return false;
    void* hx = strata::gpu::host_view((void*) x);
    void* hq = strata::gpu::host_view((void*) q);
    void* hd = strata::gpu::host_view((void*) d);
    if (!hx || !hq || !hd) { std::fprintf(stderr, "dsv41 metal: q8_quant has no host view\n"); return false; }
    const size_t page = 16384;
    // The pages the buffer itself spans: [base, base + off + bytes).  A fixed slack (the +65536 the call sites used to
    // carry) walks off the end of the mapping when the buffer is small and sits at an arena chunk's edge, and Metal then
    // refuses the no-copy wrap ("mtl: wrap 81920 failed" - the hd scales here, a few pages).
    auto wrap = [&](void* host, size_t bytes) {
        const size_t base = (size_t) ((uintptr_t) host) & ~(size_t) (page - 1);
        const size_t end = (size_t) ((uintptr_t) host) + bytes;
        return g_cx->wrap((void*) base, ((end - base + page - 1) / page) * page);
    };
    auto off = [&](void* host) { return (size_t) ((uintptr_t) host - (((uintptr_t) host) & ~(uintptr_t) (page - 1))); };
    mtl::Buffer bx = wrap(hx, (size_t) rows * ldx * 4);
    mtl::Buffer bq = wrap(hq, (size_t) rows * k);
    mtl::Buffer bd = wrap(hd, (size_t) rows * (k / 32) * 4);
    if (!bx.handle || !bq.handle || !bd.handle) return false;
    mtl::Pipeline pl;
    try { pl = g_cx->pipeline(g_lib, "ds4_q8_quant"); } catch (...) { return false; }
    mtl::Dispatch dsp;
    dsp.p = &pl;
    dsp.args = {mtl::Arg::B(bx, off(hx)), mtl::Arg::V((unsigned long) ldx), mtl::Arg::V((unsigned) k),
                mtl::Arg::B(bq, off(hq)), mtl::Arg::B(bd, off(hd))};
    const uint groups = (uint32_t) ((k / 32 + 7) / 8);
    dsp.grid[0] = groups;
    dsp.grid[1] = (uint32_t) rows;
    dsp.grid[2] = 1;
    dsp.tg[0] = 256; dsp.tg[1] = dsp.tg[2] = 1;
    g_cx->run({dsp});
    return true;
}


// ds4_gguf.cu:371 - the MXFP4 expert matvec.  The engine builds its BfJobs array on the HOST (with device pointers inside), so
// the jobs go to the kernel as an opaque constant copy: same bytes, same layout, indexed by threadgroup_position_in_grid.y.
bool mx_mv(const void* jobs, int nj, int max_n, size_t job_bytes, int fmt) {
    if (nj <= 0 || max_n <= 0) return true;
    if (!ensure()) return false;
    mtl::Buffer bj = g_cx->alloc((size_t) nj * job_bytes);
    if (!bj.ptr) { std::fprintf(stderr, "dsv41 metal: mx_mv job buffer failed\n"); return false; }
    std::memcpy(bj.ptr, jobs, (size_t) nj * job_bytes);
    mtl::Pipeline pl;
    // fmt 1 = MXFP4, 2 = Q8_0, 0 = BF16 (a dense matrix: the engram's wkv, ds4_gguf.cu:1064)
    const char* fn = fmt == 2 ? "ds4_q8_mv" : (fmt == 1 ? "ds4_mx_mv" : "ds4_bf16_mv");
    try { pl = g_cx->pipeline(g_lib, fn); } catch (...) { return false; }
    mtl::Dispatch d;
    d.p = &pl;
    d.args = {mtl::Arg::B(bj)};
    d.grid[0] = (uint32_t) ((max_n + 7) / 8);
    d.grid[1] = (uint32_t) nj;
    d.grid[2] = 1;
    d.tg[0] = 256; d.tg[1] = d.tg[2] = 1;
    g_cx->run({d});
    g_cx->free(bj);
    return true;
}


// ds4_iq.cu:809 - the I-quant expert matvec.  The jobs are the engine's own MvJob array (80 bytes each, type included), so they
// travel as an opaque constant copy and the kernel picks the decoder per job; one launch covers every type in it.
bool iq_mv(const void* jobs, int nj) {
    if (nj <= 0) return true;
    if (!ensure()) return false;
    // max rows across the jobs: the grid's x extent
    const unsigned char* jb = (const unsigned char*) jobs;
    int max_n = 0;
    for (int i = 0; i < nj; ++i) {
        int n = 0;
        std::memcpy(&n, jb + (size_t) i * 80 + 24, sizeof(int));   // MvJob::n
        if (n > max_n) max_n = n;
    }
    if (max_n <= 0) return true;
    mtl::Buffer bj = g_cx->alloc((size_t) nj * 80);
    if (!bj.ptr) { std::fprintf(stderr, "dsv41 metal: iq_mv job buffer failed\n"); return false; }
    std::memcpy(bj.ptr, jobs, (size_t) nj * 80);
    mtl::Pipeline pl;
    try { pl = g_cx->pipeline(g_lib_iq, "ds4_iq_mv"); } catch (...) { return false; }
    mtl::Dispatch d;
    d.p = &pl;
    d.args = {mtl::Arg::B(bj)};
    d.grid[0] = (uint32_t) ((max_n + 7) / 8);
    d.grid[1] = (uint32_t) nj;
    d.grid[2] = 1;
    d.tg[0] = 256; d.tg[1] = d.tg[2] = 1;
    g_cx->run({d});
    g_cx->free(bj);
    return true;
}


// ds4_iq.cu:536 gate_up_kernel<T, NE> / :588 down_kernel<T, NE> - the prompt / window path's expert GEMM (the launches at
// :958 launch_gu / :960 launch_dn).  A threadgroup of 256 is 8 warps: a warp RW rows (gate / up 1, down 2), a lane the
// row's sub-blocks, the group's entries four at a pass.  The group's blob and its entries are read the way the CUDA kernel
// reads them, so the plan / window pointer arrays travel as they are: every pointer here is a CPU address (unified memory)
// and a kernel may take raw gpuAddresses (as ds4_finish's f.w does).
namespace {
// (not an entry point: the generator takes only the ^bool lines) mirrors metal/src/engine/msl/ds4_iq.metal's Ds4IqJob (Metal has no 'long long': 64-bit is 'long')
struct IqJob {
    const void* plan_ptr;
    const void* grp_ptr;
    const void* grp_start;
    const void* grp_ent;
    const void* n_groups;
    const void* xq;
    const void* xd;
    const void* hq;
    const void* hd;
    void* h;
    void* part;
    int64_t tr0, tr1, tr2;
    int n_embd, n_ff, type;
    float limit;
};

static bool iq_gemm(const IqJob& j, const char* fn, int grid_x, int groups) {
    if (!ensure()) return false;
    mtl::Buffer bj = g_cx->alloc(sizeof(j));
    if (!bj.ptr) {
        std::fprintf(stderr, "dsv41 metal: %s job buffer failed\n", fn);
        return false;
    }
    std::memcpy(bj.ptr, &j, sizeof(j));
    mtl::Pipeline pl;
    try { pl = g_cx->pipeline(g_lib_iq, fn); } catch (...) { g_cx->free(bj); return false; }
    mtl::Dispatch d;
    d.p = &pl;
    d.args = {mtl::Arg::B(bj)};
    d.grid[0] = (uint32_t) grid_x;
    d.grid[1] = (uint32_t) groups;
    d.grid[2] = 1;
    d.tg[0] = 256; d.tg[1] = d.tg[2] = 1;
    g_cx->run({d});
    g_cx->free(bj);
    return true;
}
}  // namespace

// gate / up of the group's entries: h = the clamped swiglu (ds4_iq.cuh's Args: tr[0] gate, tr[1] up, tr[2] down)
bool iq_gu(const unsigned long long* plan_ptr, const unsigned long long* grp_ptr, const int* grp_start,
           const int* grp_ent, const int* n_groups, const size_t* tr, int n_embd, int n_ff, float limit,
           const int8_t* xq, const float* xd, float* h, int type, int groups) {
    if (groups <= 0) return true;
    if (n_embd <= 0 || n_ff <= 0 || n_embd % 256 != 0 || n_ff % 8 != 0) return false;
    if (xq == nullptr || xd == nullptr || h == nullptr) return false;
    if (plan_ptr == nullptr && grp_ptr == nullptr) return false;
    IqJob j{};
    j.plan_ptr = plan_ptr;
    j.grp_ptr = grp_ptr;
    j.grp_start = grp_start;
    j.grp_ent = grp_ent;
    j.n_groups = n_groups;
    j.xq = xq;
    j.xd = xd;
    j.h = h;
    j.tr0 = (int64_t) tr[0];
    j.tr1 = (int64_t) tr[1];
    j.tr2 = (int64_t) tr[2];
    j.n_embd = n_embd;
    j.n_ff = n_ff;
    j.type = type;
    j.limit = limit;
    return iq_gemm(j, "ds4_iq_gu", n_ff / 8, groups);
}

// down of the group's entries against their int8 h: part (a row of n_embd an entry)
bool iq_dn(const unsigned long long* plan_ptr, const unsigned long long* grp_ptr, const int* grp_start,
           const int* grp_ent, const int* n_groups, const size_t* tr, int n_embd, int n_ff, const int8_t* hq,
           const float* hd, float* part, int type, int groups) {
    if (groups <= 0) return true;
    if (n_embd <= 0 || n_ff <= 0 || n_ff % 256 != 0 || n_embd % 16 != 0) return false;
    if (hq == nullptr || hd == nullptr || part == nullptr) return false;
    if (plan_ptr == nullptr && grp_ptr == nullptr) return false;
    IqJob j{};
    j.plan_ptr = plan_ptr;
    j.grp_ptr = grp_ptr;
    j.grp_start = grp_start;
    j.grp_ent = grp_ent;
    j.n_groups = n_groups;
    j.hq = hq;
    j.hd = hd;
    j.part = part;
    j.tr0 = (int64_t) tr[0];
    j.tr1 = (int64_t) tr[1];
    j.tr2 = (int64_t) tr[2];
    j.n_embd = n_embd;
    j.n_ff = n_ff;
    j.type = type;
    j.limit = 0.0f;
    return iq_gemm(j, "ds4_iq_dn", n_embd / 16, groups);
}


// ds4_gguf.cu:577 mx_combine_kernel - a token's expert outputs summed in plan order (the experts off the device add 0),
// the CPU lane's part when its flag is set, then the shared expert's: ds4_gguf.cu:169's combine(), 128 threads a block
namespace {
// mirrors metal/src/engine/msl/ds4_init.metal's Ds4Comb (a 'long' there is int64_t here)
struct MxCombJob {
    const void* plan_ptr;
    const void* plan_w;
    const void* part;
    const void* cpu_part;
    const int* cpu_flag;
    const void* sh_out;
    void* out;
    int64_t cpu_stride, out_stride;
    int k, n_embd;
};
}  // namespace

bool mx_combine(const void* plan_ptr, const void* plan_w, const void* part, const void* cpu_part, size_t cpu_stride,
                const int* cpu_flag, const void* sh_out, void* out, size_t out_stride, int T, int k, int n_embd) {
    if (T <= 0 || n_embd <= 0) return true;
    if (k <= 0 || plan_ptr == nullptr || plan_w == nullptr || part == nullptr || out == nullptr) return false;
    if (!ensure()) return false;
    MxCombJob j{};
    j.plan_ptr = plan_ptr;
    j.plan_w = plan_w;
    j.part = part;
    j.cpu_part = cpu_part;
    j.cpu_flag = cpu_flag;
    j.sh_out = sh_out;
    j.out = out;
    j.cpu_stride = (int64_t) cpu_stride;
    j.out_stride = (int64_t) out_stride;
    j.k = k;
    j.n_embd = n_embd;
    mtl::Buffer bj = g_cx->alloc(sizeof(j));
    if (!bj.ptr) {
        std::fprintf(stderr, "dsv41 metal: mx_combine job buffer failed\n");
        return false;
    }
    std::memcpy(bj.ptr, &j, sizeof(j));
    mtl::Pipeline pl;
    try { pl = g_cx->pipeline(g_lib, "ds4_mx_combine"); } catch (...) { g_cx->free(bj); return false; }
    mtl::Dispatch d;
    d.p = &pl;
    d.args = {mtl::Arg::B(bj)};
    d.grid[0] = (uint32_t) ((n_embd + 127) / 128);
    d.grid[1] = (uint32_t) T;
    d.grid[2] = 1;
    d.tg[0] = 128; d.tg[1] = d.tg[2] = 1;
    g_cx->run({d});
    g_cx->free(bj);
    return true;
}

// ds4_kernels.cu:1092 - the KV ring row to BF16 (E4M3 code x E8M0 group scale)
bool ring_to_ext(const void* ring, void* ext, long long p0, int n, int d, size_t row_bytes) {
    if (n <= 0 || d <= 0) return true;
    if (!ensure()) return false;
    void* hr = strata::gpu::host_view((void*) ring);
    void* he = strata::gpu::host_view(ext);
    if (!hr || !he) { std::fprintf(stderr, "dsv41 metal: ring_to_ext has no host view\n"); return false; }
    const size_t page = 16384;
    auto wrap = [&](void* host, size_t bytes) {   // the pages the buffer itself spans (see q8_quant_rows' note)
        const size_t base = (size_t) ((uintptr_t) host) & ~(size_t) (page - 1);
        const size_t end = (size_t) ((uintptr_t) host) + bytes;
        return g_cx->wrap((void*) base, ((end - base + page - 1) / page) * page);
    };
    auto off = [&](void* host) { return (size_t) ((uintptr_t) host - (((uintptr_t) host) & ~(uintptr_t) (page - 1))); };
    mtl::Buffer br = wrap(hr, (size_t) n * row_bytes );
    mtl::Buffer be = wrap(he, (size_t) n * d * 2 );
    if (!br.handle || !be.handle) return false;
    mtl::Pipeline pl;
    try { pl = g_cx->pipeline(g_lib, "ds4_ring_to_ext"); } catch (...) { return false; }
    mtl::Dispatch dsp;
    dsp.p = &pl;
    dsp.args = {mtl::Arg::B(br, off(hr)), mtl::Arg::B(be, off(he)), mtl::Arg::V((long) p0),
                mtl::Arg::V((unsigned) n), mtl::Arg::V((unsigned) d), mtl::Arg::V((unsigned long) row_bytes)};
    dsp.grid[0] = (uint32_t) n;
    dsp.grid[1] = dsp.grid[2] = 1;
    dsp.tg[0] = 256; dsp.tg[1] = dsp.tg[2] = 1;
    g_cx->run({dsp});
    return true;
}


// ds4_kernels.cu:1104 - the window back into the ring (E4M3 codes + E8M0 group scales, optional backup)
bool ext_to_ring(const void* ext, void* ring, long long p0, int n, int T, int d, size_t row_bytes, void* backup) {
    if (T <= 0 || n <= 0 || d <= 0) return true;
    if (!ensure()) return false;
    void* he = strata::gpu::host_view((void*) ext);
    void* hr = strata::gpu::host_view(ring);
    if (!he || !hr) { std::fprintf(stderr, "dsv41 metal: ext_to_ring has no host view\n"); return false; }
    void* hb = backup ? strata::gpu::host_view(backup) : nullptr;
    const size_t page = 16384;
    auto wrap = [&](void* host, size_t bytes) {   // the pages the buffer itself spans (see q8_quant_rows' note)
        const size_t base = (size_t) ((uintptr_t) host) & ~(size_t) (page - 1);
        const size_t end = (size_t) ((uintptr_t) host) + bytes;
        return g_cx->wrap((void*) base, ((end - base + page - 1) / page) * page);
    };
    auto off = [&](void* host) { return (size_t) ((uintptr_t) host - (((uintptr_t) host) & ~(uintptr_t) (page - 1))); };
    mtl::Buffer be = wrap(he, (size_t) (n + T) * d * 2 );
    mtl::Buffer br = wrap(hr, (size_t) n * row_bytes );
    if (!be.handle || !br.handle) return false;
    mtl::Buffer bb = br;   // never written when backup is null
    size_t b_off = 0;
    if (hb) { bb = wrap(hb, (size_t) T * row_bytes ); b_off = off(hb); if (!bb.handle) return false; }
    mtl::Pipeline pl;
    try { pl = g_cx->pipeline(g_lib, "ds4_ext_to_ring"); } catch (...) { return false; }
    mtl::Dispatch dsp;
    dsp.p = &pl;
    dsp.args = {mtl::Arg::B(be, off(he)), mtl::Arg::B(br, off(hr)), mtl::Arg::B(bb, b_off),
                mtl::Arg::V((long) p0), mtl::Arg::V((unsigned) n), mtl::Arg::V((unsigned) T),
                mtl::Arg::V((unsigned) d), mtl::Arg::V((unsigned long) row_bytes)};
    dsp.grid[0] = (uint32_t) T;
    dsp.grid[1] = dsp.grid[2] = 1;
    dsp.tg[0] = 256; dsp.tg[1] = dsp.tg[2] = 1;
    g_cx->run({dsp});
    return true;
}


// ds4_kernels.cu:937 - the selected compressed rows into the scratch (MXFP4 -> BF16, 16 threads a row)

// On Apple silicon every device pointer is a CPU address (unified memory), so a pointer the gpu allocator does not know about
// (the engine builds some of its small arrays itself) still wraps correctly: host_view when it has one, the pointer otherwise.
static inline void* cpu_addr(void* p) {
    void* h = strata::gpu::host_view(p);
    return h ? h : p;
}

// ds4_kernels.cu:1286 engram_dequant_kernel - the Engram rows (fmt 2 = IQ4_NL, the tables this model's pack carries; the
// other formats are left to the CUDA launch, which reports the line)
bool engram_dequant(const uint8_t* rows, const uint8_t* scales, int n, int hd, float* out, int fmt) {
    (void) scales;   // (fmt 2 keeps every block's scale in its own 18 bytes)
    if (n <= 0 || hd <= 0) return true;
    if (fmt != 2 || rows == nullptr || out == nullptr) return false;
    if (!ensure()) return false;
    void* hr = cpu_addr((void*) rows);
    void* ho = cpu_addr((void*) out);
    if (!hr || !ho) { std::fprintf(stderr, "dsv41 metal: engram_dequant has no host view\n"); return false; }
    const size_t page = 16384;
    auto wrap = [&](void* host, size_t bytes) {   // the pages the buffer spans (see q8_quant_rows' note)
        const size_t base = (size_t) ((uintptr_t) host) & ~(size_t) (page - 1);
        const size_t end = (size_t) ((uintptr_t) host) + bytes;
        return g_cx->wrap((void*) base, ((end - base + page - 1) / page) * page);
    };
    auto off = [&](void* host) { return (size_t) ((uintptr_t) host - (((uintptr_t) host) & ~(uintptr_t) (page - 1))); };
    mtl::Buffer br = wrap(hr, (size_t) n * (size_t) hd);
    mtl::Buffer bo = wrap(ho, (size_t) n * (size_t) hd * 4);
    if (!br.handle || !bo.handle) return false;
    mtl::Pipeline pl;
    try { pl = g_cx->pipeline(g_lib, "ds4_engram_dequant_rows"); } catch (...) { return false; }
    mtl::Dispatch d;
    d.p = &pl;
    d.args = {mtl::Arg::B(br, off(hr)), mtl::Arg::V((unsigned) hd), mtl::Arg::B(bo, off(ho)), mtl::Arg::V(fmt)};
    d.grid[0] = (uint32_t) n;
    d.grid[1] = d.grid[2] = 1;
    d.tg[0] = 256; d.tg[1] = d.tg[2] = 1;
    g_cx->run({d});
    return true;
}

// ds4_kernels.cu:1311 engram_gate_kernel - the Engram gate over a (token, copy) row: h += sigmoid(normalised dot) * val
bool engram_gate(float* h, const float* kv, const float* w, int hc, int dim, int T, float eps) {
    if (T <= 0 || hc <= 0 || dim <= 0) return true;
    if (h == nullptr || kv == nullptr || w == nullptr) return false;
    if (!ensure()) return false;
    void* hh = cpu_addr((void*) h);
    void* hk = cpu_addr((void*) kv);
    void* hw = cpu_addr((void*) w);
    if (!hh || !hk || !hw) { std::fprintf(stderr, "dsv41 metal: engram_gate has no host view\n"); return false; }
    const size_t page = 16384;
    auto wrap = [&](void* host, size_t bytes) {
        const size_t base = (size_t) ((uintptr_t) host) & ~(size_t) (page - 1);
        const size_t end = (size_t) ((uintptr_t) host) + bytes;
        return g_cx->wrap((void*) base, ((end - base + page - 1) / page) * page);
    };
    auto off = [&](void* host) { return (size_t) ((uintptr_t) host - (((uintptr_t) host) & ~(uintptr_t) (page - 1))); };
    mtl::Buffer bh = wrap(hh, (size_t) T * (size_t) hc * (size_t) dim * 4);
    mtl::Buffer bk = wrap(hk, (size_t) T * (size_t) (hc + 1) * (size_t) dim * 4);
    mtl::Buffer bw = wrap(hw, (size_t) hc * (size_t) dim * 4);
    if (!bh.handle || !bk.handle || !bw.handle) {
        std::fprintf(stderr, "dsv41 metal: engram_gate wrap failed (h %p kv %p w %p; %p %p %p)\n", hh, hk, hw,
                     bh.handle, bk.handle, bw.handle);
        return false;
    }
    mtl::Pipeline pl;
    try { pl = g_cx->pipeline(g_lib, "ds4_engram_gate"); } catch (...) { return false; }
    mtl::Dispatch d;
    d.p = &pl;
    d.args = {mtl::Arg::B(bh, off(hh)), mtl::Arg::B(bk, off(hk)), mtl::Arg::B(bw, off(hw)),
              mtl::Arg::V((unsigned) hc), mtl::Arg::V((unsigned) dim), mtl::Arg::V(eps)};
    d.grid[0] = (uint32_t) hc;
    d.grid[1] = (uint32_t) T;
    d.grid[2] = 1;
    d.tg[0] = 1024; d.tg[1] = d.tg[2] = 1;
    g_cx->run({d});
    return true;
}

namespace {
// mirrors metal/src/engine/msl/ds4_init.metal's Ds4Rows (ds4_kernels.cuh:115 Rows, kWin = 16)
struct Ds4RowsJob {
    int n[16];
    int k[16];
    int gen, R, kcap, ccap, blk;
    int64_t p0;
};
}  // namespace

// ds4_kernels.cu:524 compress_kernel - the window compressor (ratio > 1): each token's kv / score row into its slot, and
// every ratio'th token pooled by a per-value softmax over the slots, scaled by the row RMS and the norm weights
bool compress_win(const float* kv, const float* score, float* kv_state, float* score_state, int ratio, int p0, int T,
                  int d, const float* w, float eps, float* latent) {
    if (T <= 0 || d <= 0 || ratio <= 0) return true;
    if (d > 2 * 512) return false;   // (the kernel holds d <= 2 * blockDim)
    if (kv == nullptr || score == nullptr || kv_state == nullptr || score_state == nullptr || w == nullptr ||
        latent == nullptr)
        return false;
    if (!ensure()) return false;
    void* hkv = cpu_addr((void*) kv);
    void* hsc = cpu_addr((void*) score);
    void* hks = cpu_addr((void*) kv_state);
    void* hss = cpu_addr((void*) score_state);
    void* hw = cpu_addr((void*) w);
    void* hla = cpu_addr((void*) latent);
    if (!hkv || !hsc || !hks || !hss || !hw || !hla) {
        std::fprintf(stderr, "dsv41 metal: compress_win has no host view\n");
        return false;
    }
    const size_t page = 16384;
    auto wrap = [&](void* host, size_t bytes) {   // the pages the buffer spans (see q8_quant_rows' note)
        const size_t base = (size_t) ((uintptr_t) host) & ~(size_t) (page - 1);
        const size_t end = (size_t) ((uintptr_t) host) + bytes;
        return g_cx->wrap((void*) base, ((end - base + page - 1) / page) * page);
    };
    auto off = [&](void* host) { return (size_t) ((uintptr_t) host - (((uintptr_t) host) & ~(uintptr_t) (page - 1))); };
    mtl::Buffer bkv = wrap(hkv, (size_t) T * (size_t) d * 4);
    mtl::Buffer bsc = wrap(hsc, (size_t) T * (size_t) d * 4);
    mtl::Buffer bks = wrap(hks, (size_t) ratio * (size_t) d * 4);
    mtl::Buffer bss = wrap(hss, (size_t) ratio * (size_t) d * 4);
    mtl::Buffer bw = wrap(hw, (size_t) d * 4);
    mtl::Buffer bla = wrap(hla, (size_t) T * (size_t) d * 4);
    if (!bkv.handle || !bsc.handle || !bks.handle || !bss.handle || !bw.handle || !bla.handle) return false;
    mtl::Pipeline pl;
    try { pl = g_cx->pipeline(g_lib, "ds4_compress"); } catch (...) { return false; }
    mtl::Dispatch dsp;
    dsp.p = &pl;
    dsp.args = {mtl::Arg::B(bkv, off(hkv)), mtl::Arg::B(bsc, off(hsc)), mtl::Arg::B(bks, off(hks)),
                mtl::Arg::B(bss, off(hss)), mtl::Arg::V(ratio), mtl::Arg::V(p0), mtl::Arg::V(T),
                mtl::Arg::V((unsigned) d), mtl::Arg::B(bw, off(hw)), mtl::Arg::V(eps), mtl::Arg::B(bla, off(hla))};
    dsp.grid[0] = 1;
    dsp.grid[1] = dsp.grid[2] = 1;
    dsp.tg[0] = 512; dsp.tg[1] = dsp.tg[2] = 1;
    g_cx->run({dsp});
    return true;
}

// ds4_kernels.cu:582 index_score_kernel<128, kfmt> - the indexer's scores: a position's key row against every head's
// query, relu, the heads' weighted sum (rows = ds4_kernels.cuh's Rows, by value there and copied here)
bool index_score(const float* q, const float* wq, const void* kc, int heads, int hd, int T, int n, const int* cand,
                 int cand_stride, int blk, const void* rows, float* out, long long out_stride, int kfmt) {
    if (n <= 0 || T <= 0) return true;
    if (hd != 128 || heads <= 0 || heads > 32 || (kfmt != 3 && kfmt != 0)) return false;
    if (q == nullptr || wq == nullptr || kc == nullptr || out == nullptr || rows == nullptr) return false;
    if (!ensure()) return false;
    void* hq = cpu_addr((void*) q);
    void* hwq = cpu_addr((void*) wq);
    void* hkc = cpu_addr((void*) kc);
    void* hout = cpu_addr((void*) out);
    if (!hq || !hwq || !hkc || !hout) {
        std::fprintf(stderr, "dsv41 metal: index_score has no host view\n");
        return false;
    }
    const size_t page = 16384;
    auto wrap = [&](void* host, size_t bytes) {
        const size_t base = (size_t) ((uintptr_t) host) & ~(size_t) (page - 1);
        const size_t end = (size_t) ((uintptr_t) host) + bytes;
        return g_cx->wrap((void*) base, ((end - base + page - 1) / page) * page);
    };
    auto off = [&](void* host) { return (size_t) ((uintptr_t) host - (((uintptr_t) host) & ~(uintptr_t) (page - 1))); };
    const size_t kb = kfmt == 3 ? (size_t) ((hd / 2 + hd / 32 + 15) / 16 * 16) : (size_t) hd * 2;
    mtl::Buffer bq = wrap(hq, (size_t) T * (size_t) heads * (size_t) hd * 4);
    mtl::Buffer bwq = wrap(hwq, (size_t) T * 64 * 4);
    mtl::Buffer bkc = wrap(hkc, (size_t) n * kb);
    mtl::Buffer bo = wrap(hout, ((size_t) (T - 1) * (size_t) out_stride + (size_t) n) * 4);
    if (!bq.handle || !bwq.handle || !bkc.handle || !bo.handle) return false;
    const size_t cbytes = (size_t) T * (size_t) (cand_stride > 0 ? cand_stride : 1) * 4;
    mtl::Buffer bc = cand != nullptr ? wrap(cpu_addr((void*) cand), cbytes) : g_cx->alloc(4);
    if (!bc.handle) return false;
    // Rows is a host struct: it travels as a copy (no gpu address)
    mtl::Buffer br = g_cx->alloc(sizeof(Ds4RowsJob));
    if (!br.ptr) { std::fprintf(stderr, "dsv41 metal: index_score rows buffer failed\n"); return false; }
    std::memcpy(br.ptr, rows, sizeof(Ds4RowsJob));
    mtl::Pipeline pl;
    try { pl = g_cx->pipeline(g_lib, "ds4_index_score"); } catch (...) { return false; }
    mtl::Dispatch dsp;
    dsp.p = &pl;
    dsp.args = {mtl::Arg::B(bq, off(hq)), mtl::Arg::B(bwq, off(hwq)), mtl::Arg::B(bkc, off(hkc)),
                mtl::Arg::B(bc, cand != nullptr ? off(cpu_addr((void*) cand)) : 0), mtl::Arg::B(bo, off(hout)),
                mtl::Arg::B(br), mtl::Arg::V(heads), mtl::Arg::V(n), mtl::Arg::V(cand_stride), mtl::Arg::V(blk),
                mtl::Arg::V((long) out_stride), mtl::Arg::V(kfmt), mtl::Arg::V(cand != nullptr ? 1 : 0)};
    dsp.grid[0] = (uint32_t) ((n + 255) / 256);
    dsp.grid[1] = (uint32_t) T;
    dsp.grid[2] = 1;
    dsp.tg[0] = 256; dsp.tg[1] = dsp.tg[2] = 1;
    g_cx->run({dsp});
    g_cx->free(br);
    if (cand == nullptr) g_cx->free(bc);
    return true;
}

// ds4_kernels.cu:401 gemv_f16_rows_kernel - F16 weights against F32 activations, a warp a row of the matrix and 2 tokens
// a block (the indexer's wproj: ds4_model.cu:7743)
bool gemv_f16_rows(const uint16_t* W, const float* x, size_t ldx, int n, int k, int T, float alpha, float* y, size_t ldy) {
    if (T <= 0 || n <= 0 || k <= 0) return true;
    if (k % 8 != 0 || W == nullptr || x == nullptr || y == nullptr) return false;
    if (!ensure()) return false;
    void* hw = cpu_addr((void*) W);
    void* hx = cpu_addr((void*) x);
    void* hy = cpu_addr((void*) y);
    if (!hw || !hx || !hy) {
        std::fprintf(stderr, "dsv41 metal: gemv_f16_rows has no host view\n");
        return false;
    }
    const size_t page = 16384;
    auto wrap = [&](void* host, size_t bytes) {   // the pages the buffer spans (see q8_quant_rows' note)
        const size_t base = (size_t) ((uintptr_t) host) & ~(size_t) (page - 1);
        const size_t end = (size_t) ((uintptr_t) host) + bytes;
        return g_cx->wrap((void*) base, ((end - base + page - 1) / page) * page);
    };
    auto off = [&](void* host) { return (size_t) ((uintptr_t) host - (((uintptr_t) host) & ~(uintptr_t) (page - 1))); };
    mtl::Buffer bw = wrap(hw, (size_t) n * (size_t) k * 2);
    mtl::Buffer bx = wrap(hx, ((size_t) (T - 1) * ldx + (size_t) k) * 4);
    mtl::Buffer by = wrap(hy, ((size_t) (T - 1) * ldy + (size_t) n) * 4);
    if (!bw.handle || !bx.handle || !by.handle) return false;
    mtl::Pipeline pl;
    try { pl = g_cx->pipeline(g_lib, "ds4_gemv_f16_rows"); } catch (...) { return false; }
    mtl::Dispatch dsp;
    dsp.p = &pl;
    dsp.args = {mtl::Arg::B(bw, off(hw)), mtl::Arg::B(bx, off(hx)), mtl::Arg::V((unsigned long) ldx), mtl::Arg::V(n),
                mtl::Arg::V(k), mtl::Arg::V(T), mtl::Arg::V(alpha), mtl::Arg::B(by, off(hy)),
                mtl::Arg::V((unsigned long) ldy)};
    dsp.grid[0] = (uint32_t) ((n + 7) / 8);
    dsp.grid[1] = (uint32_t) ((T + 1) / 2);
    dsp.grid[2] = 1;
    dsp.tg[0] = 256; dsp.tg[1] = dsp.tg[2] = 1;
    g_cx->run({dsp});
    return true;
}

// ds4_kernels.cu:650 topk_kernel - a token's k largest indexer scores (four 8-bit radix passes, then the picks in index
// order); Rows by value there and copied here
bool topk(const float* vals, long long stride, int T, const void* rows, const int* cand, int cand_stride, int blk, int* out,
          int out_stride) {
    if (T <= 0) return true;
    if (vals == nullptr || out == nullptr || rows == nullptr) return false;
    if (!ensure()) return false;
    void* hv = cpu_addr((void*) vals);
    void* ho = cpu_addr((void*) out);
    if (!hv || !ho) {
        std::fprintf(stderr, "dsv41 metal: topk has no host view\n");
        return false;
    }
    const size_t page = 16384;
    auto wrap = [&](void* host, size_t bytes) {   // the pages the buffer spans (see q8_quant_rows' note)
        const size_t base = (size_t) ((uintptr_t) host) & ~(size_t) (page - 1);
        const size_t end = (size_t) ((uintptr_t) host) + bytes;
        return g_cx->wrap((void*) base, ((end - base + page - 1) / page) * page);
    };
    auto off = [&](void* host) { return (size_t) ((uintptr_t) host - (((uintptr_t) host) & ~(uintptr_t) (page - 1))); };
    mtl::Buffer bv = wrap(hv, ((size_t) (T - 1) * (size_t) stride + 1) * 4);
    mtl::Buffer bo = wrap(ho, ((size_t) (T - 1) * (size_t) out_stride + 1) * 4);
    if (!bv.handle || !bo.handle) return false;
    const size_t cbytes = (size_t) T * (size_t) (cand_stride > 0 ? cand_stride : 1) * 4;
    mtl::Buffer bc = cand != nullptr ? wrap(cpu_addr((void*) cand), cbytes) : g_cx->alloc(4);
    if (!bc.handle) return false;
    mtl::Buffer br = g_cx->alloc(sizeof(Ds4RowsJob));
    if (!br.ptr) { std::fprintf(stderr, "dsv41 metal: topk rows buffer failed\n"); return false; }
    std::memcpy(br.ptr, rows, sizeof(Ds4RowsJob));
    mtl::Pipeline pl;
    try { pl = g_cx->pipeline(g_lib, "ds4_topk"); } catch (...) { return false; }
    mtl::Dispatch dsp;
    dsp.p = &pl;
    dsp.args = {mtl::Arg::B(bv, off(hv)), mtl::Arg::V((long) stride), mtl::Arg::B(br),
                mtl::Arg::B(bc, cand != nullptr ? off(cpu_addr((void*) cand)) : 0), mtl::Arg::V(cand_stride),
                mtl::Arg::V(blk), mtl::Arg::B(bo, off(ho)), mtl::Arg::V(out_stride),
                mtl::Arg::V(cand != nullptr ? 1 : 0)};
    dsp.grid[0] = (uint32_t) T;
    dsp.grid[1] = dsp.grid[2] = 1;
    dsp.tg[0] = 1024; dsp.tg[1] = dsp.tg[2] = 1;
    g_cx->run({dsp});
    g_cx->free(br);
    if (cand == nullptr) g_cx->free(bc);
    return true;
}

// ds4_kernels.cu:751 block_max_kernel - a block of the indexer's positions and its largest score (the newest pinned at +inf)
bool block_max(const float* score, long long stride, int T, const void* rows, int blk, float* bm, long long bm_stride,
               int nb_max) {
    if (T <= 0 || nb_max <= 0 || blk <= 0) return true;
    if (score == nullptr || bm == nullptr || rows == nullptr) return false;
    if (!ensure()) return false;
    void* hs = cpu_addr((void*) score);
    void* hb = cpu_addr((void*) bm);
    if (!hs || !hb) {
        std::fprintf(stderr, "dsv41 metal: block_max has no host view\n");
        return false;
    }
    const size_t page = 16384;
    auto wrap = [&](void* host, size_t bytes) {   // the pages the buffer spans (see q8_quant_rows' note)
        const size_t base = (size_t) ((uintptr_t) host) & ~(size_t) (page - 1);
        const size_t end = (size_t) ((uintptr_t) host) + bytes;
        return g_cx->wrap((void*) base, ((end - base + page - 1) / page) * page);
    };
    auto off = [&](void* host) { return (size_t) ((uintptr_t) host - (((uintptr_t) host) & ~(uintptr_t) (page - 1))); };
    mtl::Buffer bs = wrap(hs, ((size_t) (T - 1) * (size_t) stride + (size_t) nb_max * (size_t) blk) * 4);
    mtl::Buffer bb = wrap(hb, ((size_t) (T - 1) * (size_t) bm_stride + (size_t) nb_max) * 4);
    if (!bs.handle || !bb.handle) return false;
    mtl::Buffer br = g_cx->alloc(sizeof(Ds4RowsJob));
    if (!br.ptr) { std::fprintf(stderr, "dsv41 metal: block_max rows buffer failed\n"); return false; }
    std::memcpy(br.ptr, rows, sizeof(Ds4RowsJob));
    mtl::Pipeline pl;
    try { pl = g_cx->pipeline(g_lib, "ds4_block_max"); } catch (...) { return false; }
    mtl::Dispatch dsp;
    dsp.p = &pl;
    dsp.args = {mtl::Arg::B(bs, off(hs)), mtl::Arg::V((long) stride), mtl::Arg::B(br), mtl::Arg::V(blk),
                mtl::Arg::B(bb, off(hb)), mtl::Arg::V((long) bm_stride)};
    dsp.grid[0] = (uint32_t) ((nb_max + 255) / 256);
    dsp.grid[1] = (uint32_t) T;
    dsp.grid[2] = 1;
    dsp.tg[0] = 256; dsp.tg[1] = dsp.tg[2] = 1;
    g_cx->run({dsp});
    g_cx->free(br);
    return true;
}

// ds4_kernels.cu:1141 hc_mean_kernel - a token's mean over its hc streams (grid (ceil(dim / 256), T), 256 threads)
bool hc_mean(const float* h, int T, int hc, int dim, float* out, long long out_stride) {
    if (T <= 0 || hc <= 0 || dim <= 0) return true;
    if (h == nullptr || out == nullptr) return false;
    if (!ensure()) return false;
    void* hh = cpu_addr((void*) h);
    void* ho = cpu_addr((void*) out);
    if (!hh || !ho) {
        std::fprintf(stderr, "dsv41 metal: hc_mean has no host view\n");
        return false;
    }
    const size_t page = 16384;
    auto wrap = [&](void* host, size_t bytes) {   // the pages the buffer spans (see q8_quant_rows' note)
        const size_t base = (size_t) ((uintptr_t) host) & ~(size_t) (page - 1);
        const size_t end = (size_t) ((uintptr_t) host) + bytes;
        return g_cx->wrap((void*) base, ((end - base + page - 1) / page) * page);
    };
    auto off = [&](void* host) { return (size_t) ((uintptr_t) host - (((uintptr_t) host) & ~(uintptr_t) (page - 1))); };
    mtl::Buffer bh = wrap(hh, (size_t) T * (size_t) hc * (size_t) dim * 4);
    mtl::Buffer bo = wrap(ho, ((size_t) (T - 1) * (size_t) out_stride + (size_t) dim) * 4);
    if (!bh.handle || !bo.handle) return false;
    mtl::Pipeline pl;
    try { pl = g_cx->pipeline(g_lib, "ds4_hc_mean"); } catch (...) { return false; }
    mtl::Dispatch dsp;
    dsp.p = &pl;
    dsp.args = {mtl::Arg::B(bh, off(hh)), mtl::Arg::V(hc), mtl::Arg::V(dim), mtl::Arg::B(bo, off(ho)),
                mtl::Arg::V((long) out_stride)};
    dsp.grid[0] = (uint32_t) ((dim + 255) / 256);
    dsp.grid[1] = (uint32_t) T;
    dsp.grid[2] = 1;
    dsp.tg[0] = 256; dsp.tg[1] = dsp.tg[2] = 1;
    g_cx->run({dsp});
    return true;
}

// ds4_kernels.cu:1150 iota_rows_kernel - out[t][i] = i
bool iota_rows(int* out, int T, int n, int stride) {
    if (T <= 0 || n <= 0) return true;
    if (out == nullptr) return false;
    if (!ensure()) return false;
    void* ho = cpu_addr((void*) out);
    if (!ho) {
        std::fprintf(stderr, "dsv41 metal: iota_rows has no host view\n");
        return false;
    }
    const size_t page = 16384;
    auto wrap = [&](void* host, size_t bytes) {
        const size_t base = (size_t) ((uintptr_t) host) & ~(size_t) (page - 1);
        const size_t end = (size_t) ((uintptr_t) host) + bytes;
        return g_cx->wrap((void*) base, ((end - base + page - 1) / page) * page);
    };
    auto off = [&](void* host) { return (size_t) ((uintptr_t) host - (((uintptr_t) host) & ~(uintptr_t) (page - 1))); };
    mtl::Buffer bo = wrap(ho, ((size_t) (T - 1) * (size_t) stride + (size_t) n) * 4);
    if (!bo.handle) return false;
    mtl::Pipeline pl;
    try { pl = g_cx->pipeline(g_lib, "ds4_iota_rows"); } catch (...) { return false; }
    mtl::Dispatch dsp;
    dsp.p = &pl;
    dsp.args = {mtl::Arg::B(bo, off(ho)), mtl::Arg::V(n), mtl::Arg::V(stride)};
    dsp.grid[0] = (uint32_t) T;
    dsp.grid[1] = dsp.grid[2] = 1;
    dsp.tg[0] = 64; dsp.tg[1] = dsp.tg[2] = 1;
    g_cx->run({dsp});
    return true;
}

// vendor/sampler.cu:93 sampler_greedy_kernel - the plain argmax over the vocabulary, one block (1024) a token, the
// repetition penalties from the history window's bitmap (the caller takes this path when greedy or temperature <= 0)
bool sample_greedy(const float* logits, int n_tokens, int n_vocab, const int* history, int history_len, int penalty_last_n,
                   float penalty_repeat, float penalty_freq, float penalty_present, const float* logit_bias, int* out) {
    if (n_tokens <= 0 || n_vocab <= 0) return true;
    if (logits == nullptr || out == nullptr) return false;
    if ((n_vocab + 31) / 32 > 4096) return false;   // (the kernel's threadgroup bitmap)
    if (!ensure()) return false;
    void* hl = cpu_addr((void*) logits);
    void* ho = cpu_addr((void*) out);
    if (!hl || !ho) {
        std::fprintf(stderr, "dsv41 metal: sample_greedy has no host view\n");
        return false;
    }
    const size_t page = 16384;
    auto wrap = [&](void* host, size_t bytes) {   // the pages the buffer spans (see q8_quant_rows' note)
        const size_t base = (size_t) ((uintptr_t) host) & ~(size_t) (page - 1);
        const size_t end = (size_t) ((uintptr_t) host) + bytes;
        return g_cx->wrap((void*) base, ((end - base + page - 1) / page) * page);
    };
    auto off = [&](void* host) { return (size_t) ((uintptr_t) host - (((uintptr_t) host) & ~(uintptr_t) (page - 1))); };
    mtl::Buffer bl = wrap(hl, (size_t) n_tokens * (size_t) n_vocab * 4);
    mtl::Buffer bo = wrap(ho, (size_t) n_tokens * 4);
    if (!bl.handle || !bo.handle) return false;
    const size_t hbytes = (size_t) n_tokens * (size_t) (history_len > 0 ? history_len : 1) * 4;
    mtl::Buffer bh = history != nullptr ? wrap(cpu_addr((void*) history), hbytes) : g_cx->alloc(4);
    mtl::Buffer bb = logit_bias != nullptr ? wrap(cpu_addr((void*) logit_bias), (size_t) n_vocab * 4) : g_cx->alloc(4);
    if (!bh.handle || !bb.handle) return false;
    mtl::Pipeline pl;
    try { pl = g_cx->pipeline(g_lib, "ds4_sampler_greedy"); } catch (...) { return false; }
    mtl::Dispatch dsp;
    dsp.p = &pl;
    dsp.args = {mtl::Arg::B(bl, off(hl)), mtl::Arg::V(n_vocab),
                mtl::Arg::B(bh, history != nullptr ? off(cpu_addr((void*) history)) : 0),
                mtl::Arg::V(history_len), mtl::Arg::V(penalty_last_n), mtl::Arg::V(penalty_repeat),
                mtl::Arg::V(penalty_freq), mtl::Arg::V(penalty_present),
                mtl::Arg::B(bb, logit_bias != nullptr ? off(cpu_addr((void*) logit_bias)) : 0),
                mtl::Arg::V(logit_bias != nullptr ? 1 : 0), mtl::Arg::B(bo, off(ho))};
    dsp.grid[0] = (uint32_t) n_tokens;
    dsp.grid[1] = dsp.grid[2] = 1;
    dsp.tg[0] = 1024; dsp.tg[1] = dsp.tg[2] = 1;
    g_cx->run({dsp});
    if (history == nullptr) g_cx->free(bh);
    if (logit_bias == nullptr) g_cx->free(bb);
    return true;
}

bool unpack_sel(const void* ckv, const int* sel, int sel_stride, int gen, long long p0, int t0, int R, int kcap,
                const int* nsel, int cap, void* out, int T, int nsel_count, size_t row_bytes) {
    if (T <= 0 || sel_stride <= 0 || nsel_count <= 0) return true;
    // nothing selected yet (a prompt at position 0): the scratch stays untouched, which is what the CUDA launch would do with
    // no rows to write
    if (ckv == nullptr || sel == nullptr || out == nullptr) return true;
    if (!ensure()) return false;
    void* hc = cpu_addr((void*) ckv);
    void* hs = cpu_addr((void*) sel);
    void* ho = cpu_addr((void*) out);
    if (!hc || !hs || !ho) { std::fprintf(stderr, "dsv41 metal: unpack_sel has no host view\n"); return false; }
    const size_t page = 16384;
    auto wrap = [&](void* host, size_t bytes) {   // the pages the buffer itself spans (see q8_quant_rows' note)
        const size_t base = (size_t) ((uintptr_t) host) & ~(size_t) (page - 1);
        const size_t end = (size_t) ((uintptr_t) host) + bytes;
        return g_cx->wrap((void*) base, ((end - base + page - 1) / page) * page);
    };
    auto off = [&](void* host) { return (size_t) ((uintptr_t) host - (((uintptr_t) host) & ~(uintptr_t) (page - 1))); };
    mtl::Buffer bc = wrap(hc, (size_t) 4096 * row_bytes );
    mtl::Buffer bsl = wrap(hs, (size_t) T * sel_stride * 4 );
    mtl::Buffer bo = wrap(ho, (size_t) T * cap * 512 * 2 );
    if (!bc.handle || !bsl.handle || !bo.handle) return false;
    mtl::Buffer bn = g_cx->alloc((size_t) nsel_count * 4);
    if (!bn.ptr) return false;
    std::memcpy(bn.ptr, nsel, (size_t) nsel_count * 4);
    mtl::Pipeline pl;
    try { pl = g_cx->pipeline(g_lib, "ds4_unpack_sel"); } catch (...) { return false; }
    mtl::Dispatch d;
    d.p = &pl;
    d.args = {mtl::Arg::B(bc, off(hc)), mtl::Arg::B(bsl, off(hs)), mtl::Arg::V((unsigned) sel_stride),
              mtl::Arg::V((int) gen), mtl::Arg::V((long) p0), mtl::Arg::V((int) t0), mtl::Arg::V((int) R),
              mtl::Arg::V((int) kcap), mtl::Arg::B(bn), mtl::Arg::V((unsigned) cap), mtl::Arg::B(bo, off(ho)),
              mtl::Arg::V((unsigned long) row_bytes)};
    d.grid[0] = (uint32_t) ((sel_stride + 15) / 16);
    d.grid[1] = (uint32_t) T;
    d.grid[2] = 1;
    d.tg[0] = 256; d.tg[1] = d.tg[2] = 1;
    g_cx->run({d});
    g_cx->free(bn);
    return true;
}


// ds4_kernels.cu:1772 - the expert outputs back into the token rows
bool moe_combine(void* acc, const void* y, const int* pos, const float* w_, int T, int k, int r0, int r1, int dim) {
    if (T <= 0 || dim <= 0) return true;
    if (!ensure()) return false;
    void* ha = cpu_addr(acc);
    void* hy = cpu_addr((void*) y);
    void* hp = cpu_addr((void*) pos);
    void* hw = cpu_addr((void*) w_);
    if (!ha || !hy || !hp || !hw) { std::fprintf(stderr, "dsv41 metal: moe_combine has no host view\n"); return false; }
    const size_t page = 16384;
    auto wrap = [&](void* host, size_t bytes) {   // the pages the buffer itself spans (see q8_quant_rows' note)
        const size_t base = (size_t) ((uintptr_t) host) & ~(size_t) (page - 1);
        const size_t end = (size_t) ((uintptr_t) host) + bytes;
        return g_cx->wrap((void*) base, ((end - base + page - 1) / page) * page);
    };
    auto off = [&](void* host) { return (size_t) ((uintptr_t) host - (((uintptr_t) host) & ~(uintptr_t) (page - 1))); };
    mtl::Buffer ba = wrap(ha, (size_t) T * dim * 4 );
    mtl::Buffer by = wrap(hy, (size_t) (r1 > r0 ? r1 - r0 : 1) * dim * 4 );
    mtl::Buffer bp = wrap(hp, (size_t) T * 8 * 4 );
    mtl::Buffer bw = wrap(hw, (size_t) T * 8 * 4 );
    if (!ba.handle || !by.handle || !bp.handle || !bw.handle) return false;
    mtl::Pipeline pl;
    try { pl = g_cx->pipeline(g_lib, "ds4_moe_combine"); } catch (...) { return false; }
    mtl::Dispatch d;
    d.p = &pl;
    d.args = {mtl::Arg::B(ba, off(ha)), mtl::Arg::B(by, off(hy)), mtl::Arg::B(bp, off(hp)), mtl::Arg::B(bw, off(hw)),
              mtl::Arg::V((int) k), mtl::Arg::V((int) r0), mtl::Arg::V((int) r1), mtl::Arg::V((unsigned) dim)};
    d.grid[0] = (uint32_t) ((dim + 255) / 256);
    d.grid[1] = (uint32_t) T;
    d.grid[2] = 1;
    d.tg[0] = 256; d.tg[1] = d.tg[2] = 1;
    g_cx->run({d});
    return true;
}


// ds4_kernels.cu:362 - hc_post: out[t][k][e] = post[t][k]*y[t][e] + sum_j comb[t][j][k]*res[t][j][e]
bool hc_post_v(const float* y, const float* res, const float* post, const float* comb, int dim, int hc, int T, float* out) {
    if (T <= 0 || dim <= 0) return true;
    if (!ensure()) return false;
    void* hy = cpu_addr((void*) y);
    void* hr = cpu_addr((void*) res);
    void* hp = cpu_addr((void*) post);
    void* hco = cpu_addr((void*) comb);
    void* ho = cpu_addr((void*) out);
    if (!hy || !hr || !hp || !hco || !ho) { std::fprintf(stderr, "dsv41 metal: hc_post has no host view\n"); return false; }
    const size_t page = 16384;
    auto wrap = [&](void* host, size_t bytes) {   // the pages the buffer itself spans (see q8_quant_rows' note)
        const size_t base = (size_t) ((uintptr_t) host) & ~(size_t) (page - 1);
        const size_t end = (size_t) ((uintptr_t) host) + bytes;
        return g_cx->wrap((void*) base, ((end - base + page - 1) / page) * page);
    };
    auto off = [&](void* host) { return (size_t) ((uintptr_t) host - (((uintptr_t) host) & ~(uintptr_t) (page - 1))); };
    mtl::Buffer by = wrap(hy, (size_t) T * dim * 4 );
    mtl::Buffer br = wrap(hr, (size_t) T * hc * dim * 4 );
    mtl::Buffer bp = wrap(hp, (size_t) T * 8 * 4 );
    mtl::Buffer bc = wrap(hco, (size_t) T * 64 * 4 );
    mtl::Buffer bo = wrap(ho, (size_t) T * hc * dim * 4 );
    if (!by.handle || !br.handle || !bp.handle || !bc.handle || !bo.handle) return false;
    mtl::Pipeline pl;
    try { pl = g_cx->pipeline(g_lib, "ds4_hc_post_v"); } catch (...) { return false; }
    mtl::Dispatch d;
    d.p = &pl;
    d.args = {mtl::Arg::B(by, off(hy)), mtl::Arg::B(br, off(hr)), mtl::Arg::B(bp, off(hp)),
              mtl::Arg::B(bc, off(hco)), mtl::Arg::B(bo, off(ho)), mtl::Arg::V((unsigned) dim),
              mtl::Arg::V((unsigned) hc)};
    d.grid[0] = (uint32_t) ((dim * hc + 255) / 256);
    d.grid[1] = (uint32_t) T;
    d.grid[2] = 1;
    d.tg[0] = 256; d.tg[1] = d.tg[2] = 1;
    g_cx->run({d});
    return true;
}


// ds4_kernels.cu:441 - the F32 GEMV (residual / dense projections)
bool gemv_f32(const float* W, const float* x, int n, int k, int T, float* y, int ldy) {
    if (T <= 0 || n <= 0 || k <= 0) return true;
    if (!ensure()) return false;
    void* hw = cpu_addr((void*) W);
    void* hx = cpu_addr((void*) x);
    void* hy = cpu_addr((void*) y);
    if (!hw || !hx || !hy) { std::fprintf(stderr, "dsv41 metal: gemv_f32 has no host view\n"); return false; }
    const size_t page = 16384;
    auto wrap = [&](void* host, size_t bytes) {   // the pages the buffer itself spans (see q8_quant_rows' note)
        const size_t base = (size_t) ((uintptr_t) host) & ~(size_t) (page - 1);
        const size_t end = (size_t) ((uintptr_t) host) + bytes;
        return g_cx->wrap((void*) base, ((end - base + page - 1) / page) * page);
    };
    auto off = [&](void* host) { return (size_t) ((uintptr_t) host - (((uintptr_t) host) & ~(uintptr_t) (page - 1))); };
    mtl::Buffer bw = wrap(hw, (size_t) n * k * 4 );
    mtl::Buffer bx = wrap(hx, (size_t) T * k * 4 );
    mtl::Buffer by = wrap(hy, (size_t) T * (ldy > 0 ? ldy : n) * 4 );
    if (!bw.handle || !bx.handle || !by.handle) return false;
    mtl::Pipeline pl;
    try { pl = g_cx->pipeline(g_lib, "ds4_gemv_f32"); } catch (...) { return false; }
    mtl::Dispatch d;
    d.p = &pl;
    d.args = {mtl::Arg::B(bw, off(hw)), mtl::Arg::B(bx, off(hx)), mtl::Arg::V((unsigned) n),
              mtl::Arg::V((unsigned) k), mtl::Arg::B(by, off(hy)), mtl::Arg::V((unsigned) ldy)};
    d.grid[0] = (uint32_t) ((n + 7) / 8);
    d.grid[1] = (uint32_t) T;
    d.grid[2] = 1;
    d.tg[0] = 256; d.tg[1] = d.tg[2] = 1;
    g_cx->run({d});
    return true;
}


// ds4_kernels.cu:1353 - the router's selection (idx, w, plan_ptr).  The prefetch requests stay unpublished for now.
bool route_publish(const float* logits, const float* bias, int n, int k, int T, float route_scale, const void* vram,
                   int* idx, float* w, void* plan_ptr, const float* bias_vl, unsigned vl) {
    if (T <= 0 || n <= 0) return true;
    if (!ensure()) return false;
    void* hl = cpu_addr((void*) logits);
    void* hb = cpu_addr((void*) bias);
    void* hv = cpu_addr((void*) vram);
    void* hi = cpu_addr((void*) idx);
    void* hw = cpu_addr((void*) w);
    void* hp = cpu_addr((void*) plan_ptr);
    if (!hl || !hb || !hv || !hi || !hw || !hp) { std::fprintf(stderr, "dsv41 metal: route_publish has no host view\n"); return false; }
    const size_t page = 16384;
    auto wrap = [&](void* host, size_t bytes) {   // the pages the buffer itself spans (see q8_quant_rows' note)
        const size_t base = (size_t) ((uintptr_t) host) & ~(size_t) (page - 1);
        const size_t end = (size_t) ((uintptr_t) host) + bytes;
        return g_cx->wrap((void*) base, ((end - base + page - 1) / page) * page);
    };
    auto off = [&](void* host) { return (size_t) ((uintptr_t) host - (((uintptr_t) host) & ~(uintptr_t) (page - 1))); };
    mtl::Buffer bl = wrap(hl, (size_t) T * 1024 * 4 );
    mtl::Buffer bb = wrap(hb, (size_t) n * 4 );
    mtl::Buffer bv = wrap(hv, (size_t) n * 8 );
    mtl::Buffer bi = wrap(hi, (size_t) T * 8 * 4 );
    mtl::Buffer bw = wrap(hw, (size_t) T * 8 * 4 );
    mtl::Buffer bp = wrap(hp, (size_t) T * 8 * 8 );
    if (!bl.handle || !bb.handle || !bv.handle || !bi.handle || !bw.handle || !bp.handle) return false;
    mtl::Buffer bvl = bb;
    size_t vl_off = 0;
    if (bias_vl) {
        void* hvl = cpu_addr((void*) bias_vl);
        if (!hvl) return false;
        bvl = wrap(hvl, (size_t) n * 4 );
        vl_off = off(hvl);
    }
    mtl::Pipeline pl;
    try { pl = g_cx->pipeline(g_lib, "ds4_route_publish"); } catch (...) { return false; }
    mtl::Dispatch d;
    d.p = &pl;
    d.args = {mtl::Arg::B(bl, off(hl)), mtl::Arg::B(bb, off(hb)), mtl::Arg::V((unsigned) n), mtl::Arg::V((unsigned) k),
              mtl::Arg::V(route_scale), mtl::Arg::B(bv, off(hv)), mtl::Arg::B(bi, off(hi)), mtl::Arg::B(bw, off(hw)),
              mtl::Arg::B(bp, off(hp)), mtl::Arg::B(bvl, vl_off), mtl::Arg::V((unsigned) vl),
              mtl::Arg::V((unsigned) (bias_vl ? 1 : 0))};
    d.grid[0] = (uint32_t) T;
    d.grid[1] = d.grid[2] = 1;
    d.tg[0] = 512; d.tg[1] = d.tg[2] = 1;
    g_cx->run({d});
    return true;
}


// ds4_kernels.cu swiglu: h = silu(g) * u (clamped when lim > 0)
bool swiglu(const float* g, const float* u, int n, float lim, float* h) {
    if (n <= 0) return true;
    if (!ensure()) return false;
    void* hg = cpu_addr((void*) g);
    void* hu = cpu_addr((void*) u);
    void* hh = cpu_addr((void*) h);
    if (!hg || !hu || !hh) { std::fprintf(stderr, "dsv41 metal: swiglu has no host view\n"); return false; }
    const size_t page = 16384;
    auto wrap = [&](void* host) {
        const size_t bytes = (size_t) n * 4 + 65536;
        return g_cx->wrap((void*) (((uintptr_t) host) & ~(uintptr_t) (page - 1)), ((bytes + page - 1) / page) * page);
    };
    auto off = [&](void* host) { return (size_t) ((uintptr_t) host - (((uintptr_t) host) & ~(uintptr_t) (page - 1))); };
    mtl::Buffer bg = wrap(hg), bu = wrap(hu), bh = wrap(hh);
    if (!bg.handle || !bu.handle || !bh.handle) return false;
    mtl::Pipeline pl;
    try { pl = g_cx->pipeline(g_lib, "ds4_swiglu"); } catch (...) { return false; }
    mtl::Dispatch d;
    d.p = &pl;
    d.args = {mtl::Arg::B(bg, off(hg)), mtl::Arg::B(bu, off(hu)), mtl::Arg::V((unsigned) n), mtl::Arg::V(lim),
              mtl::Arg::B(bh, off(hh))};
    d.grid[0] = (uint32_t) ((n + 255) / 256);
    d.grid[1] = d.grid[2] = 1;
    d.tg[0] = 256; d.tg[1] = d.tg[2] = 1;
    g_cx->run({d});
    return true;
}


// vendor/moe_win.cu:49  moe_win_group - the expert -> group index for a window.  One thread in CUDA; on the Mac every device
// pointer is a CPU address, so the same 25-line loop runs on the host, which keeps the first-appearance order the kernel has and
// avoids mirroring the 17-field MoeWin struct through a constant buffer.
bool moe_win_group_cpu(const void* plan_ptr, void* grp_ptr, void* grp_start, void* grp_ent, void* n_groups, int T, int k) {
    if (T <= 0 || k <= 0) return true;
    if (T * 8 > 4096) { std::fprintf(stderr, "dsv41 metal: moe_win_group window of %d too wide\n", T); return false; }
    const unsigned long long* plan = (const unsigned long long*) cpu_addr((void*) plan_ptr);
    unsigned long long* gptr = (unsigned long long*) cpu_addr(grp_ptr);
    int* gstart = (int*) cpu_addr(grp_start);
    int* gent = (int*) cpu_addr(grp_ent);
    int* ng_out = (int*) cpu_addr(n_groups);
    if (!plan || !gptr || !gstart || !gent || !ng_out) return false;
    static unsigned long long gp[4096];
    static int cnt[4096], fill[4097], gof[4096];
    int ng = 0;
    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < 8; ++i) {
            const int e = t * 8 + i;
            gof[e] = -1;
            const unsigned long long p = i < k ? plan[e] : 0ull;
            if (p == 0ull) continue;
            int g = 0;
            while (g < ng && gp[g] != p) ++g;
            if (g == ng) {
                gp[ng] = p;
                cnt[ng++] = 0;
            }
            ++cnt[g];
            gof[e] = g;
        }
    }
    fill[0] = 0;
    for (int g = 0; g < ng; ++g) {
        gptr[g] = gp[g];
        gstart[g] = fill[g];
        fill[g + 1] = fill[g] + cnt[g];
    }
    gstart[ng] = fill[ng];
    for (int e = 0; e < T * 8; ++e) {
        if (gof[e] >= 0) gent[fill[gof[e]]++] = e;
    }
    *ng_out = ng;
    return true;
}

}  // namespace strata::dsv41::metal_host
