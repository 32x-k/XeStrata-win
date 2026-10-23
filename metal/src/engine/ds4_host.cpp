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
    g_lib = cx.compile({"src/engine/msl/ds4_quant.metal"}, "", false);
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
