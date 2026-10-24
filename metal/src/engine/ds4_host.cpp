// metal/src/engine/ds4_host.cpp - host side of the ported dsv41 kernels: compile the MSL once, wrap the engine's own buffers,
// dispatch.  First entry point: the Q8 activation quantiser (ds4_gguf.cu:201).
//
// The two runtimes meet on pointers: strata::gpu owns the engine's device memory (a device pointer is its MTLBuffer's
// gpuAddress) and exposes host_view(device_pointer); mtl::Context::wrap() adopts that address without copying and a
// Dispatch names it through Arg::B().  See RIGS/m1002.md for the derivation.
#include "mtl_rt.h"

#include <strata/gpu/runtime.hpp>



#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <strata/gpu/runtime.hpp>
#include <vector>

namespace strata::dsv41::metal_host {

namespace {
mtl::Context* g_cx = nullptr;
int g_lib = -1;
bool g_tried = false;

bool ensure(mtl::Context& cx) {
    if (g_cx == &cx && g_lib >= 0) return true;
    if (g_tried && g_lib < 0) return false;
    g_tried = true;
    // the kernel lives in its own file and is self-contained (metal_stdlib + ggml_grids.h)
    g_lib = cx.compile({"../src/engine/msl/ds4_quant.metal"}, "", false);
    if (g_lib < 0) {
        std::fprintf(stderr, "dsv41 metal: compiling ds4_quant.metal failed\n");
        return false;
    }
    g_cx = &cx;
    return true;
}
}  // namespace

// q8(x, ldx, n) -> q (int8, n bytes) and d (half, n/32); ldx is the stride in values between blocks of 32
bool q8_quant(mtl::Context& cx, const float* x, int ldx, int n, void* q, void* d) {
    if (!ensure(cx)) return false;
    const size_t xb = (size_t) ldx * (size_t) (n / 32) * sizeof(float), qb = (size_t) n, db = (size_t) (n / 32) * 2;
    mtl::Buffer bx = cx.wrap(strata::gpu::host_view(x), xb);
    mtl::Buffer bq = cx.wrap(q, qb);
    mtl::Buffer bd = cx.wrap(d, db);
    mtl::Pipeline p;
    try {
        p = cx.pipeline(g_lib, "ds4_q8_quant");
    } catch (...) {
        return false;
    }
    mtl::Dispatch ds;
    ds.p = &p;
    ds.args = {mtl::Arg::B(bx), mtl::Arg::V((unsigned) ldx), mtl::Arg::B(bq), mtl::Arg::B(bd)};
    ds.grid[0] = (uint32_t) (n / 32);
    ds.grid[1] = ds.grid[2] = 1;
    ds.tg[0] = 32;
    ds.tg[1] = ds.tg[2] = 1;
    mtl::require_fit(p, 32, "ds4_q8_quant");
    cx.run({ds});
    return true;
}

}  // namespace strata::dsv41::metal_host

namespace strata::dsv41::metal_host {
namespace {
mtl::Context* g_init_cx = nullptr;
int g_lib_init = -1;

mtl::Context& shared_cx() {
    if (!g_init_cx) {
        g_init_cx = new mtl::Context();
        g_lib_init = g_init_cx->compile({"../src/engine/msl/ds4_init.metal"}, "", false);
        if (g_lib_init < 0) std::fprintf(stderr, "dsv41 metal: compiling ds4_init.metal failed\n");
    }
    return *g_init_cx;
}
}  // namespace

// ds4_kernels.cu:1477 - pre[i] = (i % 8) == 0 ? 1 : 0 for i < T*8
bool init_pre(float* pre, int T, void*) {
    strata::dsv41::metal_host::shared_cx();
    if (g_lib_init < 0) return false;
    void* host = strata::gpu::host_view(pre);
    if (!host) { std::fprintf(stderr, "dsv41 metal: init_pre has no host view\n"); return false; }
    const size_t bytes = (size_t) T * 8 * sizeof(float);
    const size_t page = 16384;
    mtl::Buffer bp = strata::dsv41::metal_host::shared_cx().wrap((void*) (((uintptr_t) host) & ~(uintptr_t) (page - 1)),
                                    ((bytes + page - 1) / page) * page);
    if (!bp.handle) { std::fprintf(stderr, "dsv41 metal: init_pre wrap failed\n"); return false; }
    const uintptr_t delta = (uintptr_t) host - (((uintptr_t) host) & ~(uintptr_t) (page - 1));
    mtl::Pipeline pl;
    try {
        pl = strata::dsv41::metal_host::shared_cx().pipeline(g_lib_init, "ds4_init_pre");
    } catch (...) {
        return false;
    }
    mtl::Dispatch d;
    d.p = &pl;
    d.args = {mtl::Arg::B(bp, delta), mtl::Arg::V((unsigned) T)};
    d.grid[0] = (uint32_t) ((T * 8 + 127) / 128);
    d.grid[1] = d.grid[2] = 1;
    d.tg[0] = 128;
    d.tg[1] = d.tg[2] = 1;
    strata::dsv41::metal_host::shared_cx().run({d});
    return true;
}


// ds4_kernels.cu:130 - the BF16 embedding gather, hc copies a token
bool embed(const uint16_t* rows, int T, float* h, int dim, int hc, void*) {
    strata::dsv41::metal_host::shared_cx();
    if (g_lib_init < 0) { std::fprintf(stderr, "dsv41 metal: embed: no library\n"); return false; }
    void* hrows = strata::gpu::host_view((void*) rows);
    void* hh = strata::gpu::host_view(h);
    if (!hrows || !hh) { std::fprintf(stderr, "dsv41 metal: embed has no host view\n"); return false; }
    const size_t page = 16384;
    auto wrap = [&](void* host, size_t bytes) {
        return strata::dsv41::metal_host::shared_cx().wrap((void*) (((uintptr_t) host) & ~(uintptr_t) (page - 1)), ((bytes + page - 1) / page) * page);
    };
    mtl::Buffer br = wrap(hrows, (size_t) T * dim * 2);
    mtl::Buffer bh = wrap(hh, (size_t) T * hc * dim * 4);
    if (!br.handle || !bh.handle) { std::fprintf(stderr, "dsv41 metal: embed wrap failed\n"); return false; }
    const size_t dro = (size_t) (((uintptr_t) hrows) - (((uintptr_t) hrows) & ~(uintptr_t) (page - 1)));
    const size_t dho = (size_t) (((uintptr_t) hh) - (((uintptr_t) hh) & ~(uintptr_t) (page - 1)));
    mtl::Pipeline pl;
    try { pl = strata::dsv41::metal_host::shared_cx().pipeline(g_lib_init, "ds4_embed"); } catch (...) { return false; }
    mtl::Dispatch d;
    d.p = &pl;
    d.args = {mtl::Arg::B(br, dro), mtl::Arg::B(bh, dho), mtl::Arg::V((unsigned) dim), mtl::Arg::V((unsigned) hc)};
    d.grid[0] = (uint32_t) ((dim + 255) / 256);
    d.grid[1] = (uint32_t) T;
    d.grid[2] = 1;
    d.tg[0] = 256; d.tg[1] = d.tg[2] = 1;
    strata::dsv41::metal_host::shared_cx().run({d});
    if (std::getenv("DS4_DBG_EMBWRITE")) {
        strata::dsv41::metal_host::shared_cx().flush_runs();
        const float* f0 = (const float*) hh;
        const uint16_t* r16 = (const uint16_t*) hrows;
        std::fprintf(stderr, "DBG embwrite: h=%p rows=%p h_off=%zu rows_off=%zu\n", hh, hrows, dho, dro);
        std::fprintf(stderr, "DBG embwrite: rows[0..3]=%04x %04x %04x %04x  h[0..3]=%g %g %g %g\n",
                     (unsigned) r16[0], (unsigned) r16[1], (unsigned) r16[2], (unsigned) r16[3],
                     f0[0], f0[1], f0[2], f0[3]);
    }
    return true;
}


// ds4_model.cu:2737 - C[t][m] = sum_k A[m][k] * B[t][k], F32 operands, alpha = 1, beta = 0
bool sgemm_tn(const float* A, const float* B, float* C, int m, int nt, int k, int lda, int ldb, int ldc) {
    strata::dsv41::metal_host::shared_cx();
    if (g_lib_init < 0) return false;
    void* ha = strata::gpu::host_view((void*) A);
    void* hb = strata::gpu::host_view((void*) B);
    void* hc = strata::gpu::host_view((void*) C);
    if (!ha || !hb || !hc) { std::fprintf(stderr, "dsv41 metal: sgemm_tn has no host view\n"); return false; }
    const size_t page = 16384;
    auto wrap = [&](void* host, size_t bytes) {
        return strata::dsv41::metal_host::shared_cx().wrap((void*) (((uintptr_t) host) & ~(uintptr_t) (page - 1)), ((bytes + page - 1) / page) * page);
    };
    auto off = [&](void* host) { return (size_t) ((uintptr_t) host - (((uintptr_t) host) & ~(uintptr_t) (page - 1))); };
    mtl::Buffer ba = wrap(ha, (size_t) m * lda * 4);
    mtl::Buffer bb = wrap(hb, (size_t) nt * ldb * 4);
    mtl::Buffer bc = wrap(hc, (size_t) nt * ldc * 4);
    if (!ba.handle || !bb.handle || !bc.handle) return false;
    mtl::Pipeline pl;
    try { pl = strata::dsv41::metal_host::shared_cx().pipeline(g_lib_init, "ds4_sgemm_tn"); } catch (...) { return false; }
    mtl::Dispatch d;
    d.p = &pl;
    d.args = {mtl::Arg::B(ba, off(ha)), mtl::Arg::B(bb, off(hb)), mtl::Arg::B(bc, off(hc)),
              mtl::Arg::V((unsigned) m), mtl::Arg::V((unsigned) nt), mtl::Arg::V((unsigned) k),
              mtl::Arg::V((unsigned) lda), mtl::Arg::V((unsigned) ldb), mtl::Arg::V((unsigned) ldc)};
    const uint64_t total = (uint64_t) m * (uint64_t) nt;
    d.grid[0] = (uint32_t) ((total + 127) / 128);
    d.grid[1] = d.grid[2] = 1;
    d.tg[0] = 128; d.tg[1] = d.tg[2] = 1;
    strata::dsv41::metal_host::shared_cx().run({d});
    return true;
}

}  // namespace strata::dsv41::metal_host
