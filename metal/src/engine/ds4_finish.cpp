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

}  // namespace strata::dsv41::metal_host
