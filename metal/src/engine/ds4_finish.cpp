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

bool ensure() {
    if (g_cx) return g_lib >= 0;
    g_cx = new mtl::Context();
    g_lib = g_cx->compile({"../src/engine/msl/ds4_init.metal"}, "", false);
    if (g_lib < 0) std::fprintf(stderr, "dsv41 metal: compiling ds4_init.metal failed\n");
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
    if (f.rd > 0 || f.quant != 0 || f.pdst != nullptr) {
        std::fprintf(stderr,
                     "dsv41 metal: finish needs the unwritten branches (rd %d quant %d pdst %p) - not ported yet\n",
                     f.rd, f.quant, (const void*) f.pdst);
        return false;
    }
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
    gf.rd = 0; gf.pos = 0; gf.inverse = 0; gf.vpp = 1; gf.pos_step = 1;
    gf.freqs = (uint64_t) (uintptr_t) f.freqs;
    gf.quant = 0;
    gf.dst = (uint64_t) (uintptr_t) f.dst;
    gf.dst_stride = f.dst_stride;
    gf.pdst = 0;
    gf.pdst_stride = 0;

    mtl::Pipeline pl;
    try {
        pl = g_cx->pipeline(g_lib, "ds4_finish");
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
    auto wrap = [&](void* host, size_t bytes) {
        return g_cx->wrap((void*) (((uintptr_t) host) & ~(uintptr_t) (page - 1)), ((bytes + page - 1) / page) * page);
    };
    auto off = [&](void* host) { return (size_t) ((uintptr_t) host - (((uintptr_t) host) & ~(uintptr_t) (page - 1))); };
    mtl::Buffer by = wrap(hy, (size_t) T * dim * 4 + 65536);
    mtl::Buffer bh = wrap(hh, (size_t) T * hc * dim * 4 + 65536);
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
    auto wrap = [&](void* host, size_t bytes) {
        return g_cx->wrap((void*) (((uintptr_t) host) & ~(uintptr_t) (page - 1)), ((bytes + page - 1) / page) * page);
    };
    auto off = [&](void* host) { return (size_t) ((uintptr_t) host - (((uintptr_t) host) & ~(uintptr_t) (page - 1))); };
    mtl::Buffer bh = wrap(hh, (size_t) T * hcdim * 4 + 65536);
    mtl::Buffer bm = wrap(hm, (size_t) T * 32 * 4 + 65536);
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
    auto wrap = [&](void* host, size_t bytes) {
        return g_cx->wrap((void*) (((uintptr_t) host) & ~(uintptr_t) (page - 1)), ((bytes + page - 1) / page) * page);
    };
    auto off = [&](void* host) { return (size_t) ((uintptr_t) host - (((uintptr_t) host) & ~(uintptr_t) (page - 1))); };
    mtl::Buffer bh = wrap(hh, (size_t) T * hcdim * 4 + 65536);
    mtl::Buffer bf = wrap(hf, (size_t) n_mix * hcdim * 4 + 65536);
    mtl::Buffer bm = wrap(hm, (size_t) T * 32 * 4 + 65536);
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
    auto wrap = [&](void* host, size_t bytes) {
        return g_cx->wrap((void*) (((uintptr_t) host) & ~(uintptr_t) (page - 1)), ((bytes + page - 1) / page) * page);
    };
    auto off = [&](void* host) { return (size_t) ((uintptr_t) host - (((uintptr_t) host) & ~(uintptr_t) (page - 1))); };
    mtl::Buffer bm = wrap(hm, (size_t) T * 32 * 4 + 65536);
    mtl::Buffer bp = wrap(hp, (size_t) T * 8 * 4 + 65536);
    mtl::Buffer bo = wrap(ho, (size_t) T * 8 * 4 + 65536);
    mtl::Buffer bc = wrap(hc_, (size_t) T * 64 * 4 + 65536);
    if (!bm.handle || !bp.handle || !bo.handle || !bc.handle) return false;
    // scale and base come from the engine like every other operand: reach them through host_view, not as raw host pointers
    void* hs = strata::gpu::host_view((void*) scale);
    void* hb = strata::gpu::host_view((void*) base);
    if (!hs || !hb) { std::fprintf(stderr, "dsv41 metal: hc_split scale/base have no host view\n"); return false; }
    const size_t nbase = (size_t) (2 * hc + hc * hc);
    mtl::Buffer bs = wrap(hs, 256);
    mtl::Buffer bb = wrap(hb, (nbase * 4 + 65535) / 65536 * 65536 + 65536);
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

}  // namespace strata::dsv41::metal_host
