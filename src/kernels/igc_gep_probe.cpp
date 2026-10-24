// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/igc_gep_probe.cpp - the reads IGC 2.27 to 2.30 miscompile for Xe2 GPUs (AGENTS.md, "Intel Graphics
// Compiler"): x[0], x[N], x[2N] and x[3N] from one pointer, N a runtime value, which they turn into x[0], x[N], x[8N]
// and x[9N].  The buffer has room for the wrong reads, so an affected compiler gives wrong values instead of a fault.
// Through the Xe runtime, which turns IGC's GEP simplification off, it passes on every GPU; with
// IGC_EnableGEPSimplification=1 in the environment (the runtime keeps a value the user set) it shows the miscompile.
#include "parity_device.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    // N and T from the command line: runtime values, as the engine's (with constants the reads compile right)
    const int64_t N = argc > 1 ? std::strtoll(argv[1], nullptr, 10) : 4096;
    const int64_t T = argc > 2 ? std::strtoll(argv[2], nullptr, 10) : 3;
    constexpr int HC = 4;
    if (N <= 0 || T <= 0) {
        std::fprintf(stderr, "usage: igc_gep_probe [N [T]]\n");
        return 2;
    }
    const size_t rows = (size_t) (T * HC * N), pad = (size_t) (10 * N);   // the wrong reads reach x[(T-1)*4N + 9N]
    std::vector<float> hx(rows + pad);
    for (size_t i = 0; i < hx.size(); ++i) hx[i] = (float) (i % 97) * 0.01f;
    auto& runtime = strata::core::Runtime::get();
    std::printf("igc_gep_probe: %s, N %lld, T %lld\n", runtime.device().get_info<sycl::info::device::name>().c_str(),
                (long long) N, (long long) T);
    strata::parity::Device device;
    const float* x = device.upload(hx);
    float* y = device.alloc<float>((size_t) (T * N));
    runtime.compute().parallel_for(sycl::range<1>((size_t) (T * N)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0], t = i / N, n = i % N;
        const float* xt = x + t * HC * N + n;
        y[i] = (((xt[0] + xt[N]) + xt[2 * N]) + xt[3 * N]) * (1.0f / HC);
    });
    std::vector<float> hy((size_t) (T * N));
    strata::parity::copy_out(hy.data(), y, hy.size());
    double worst = 0, worst_wrong = 0;   // against the right reads, and against the miscompiled ones
    for (int64_t t = 0; t < T; ++t)
        for (int64_t n = 0; n < N; ++n) {
            const float* xt = hx.data() + t * HC * N + n;
            const double right = ((double) xt[0] + xt[N] + xt[2 * N] + xt[3 * N]) / HC;
            const double wrong = ((double) xt[0] + xt[N] + xt[8 * N] + xt[9 * N]) / HC;
            const double got = hy[(size_t) (t * N + n)];
            worst = std::max(worst, std::fabs(got - right));
            worst_wrong = std::max(worst_wrong, std::fabs(got - wrong));
        }
    const bool ok = worst <= 1e-6;
    std::printf("igc_gep_probe: worst %.2e against the right reads, %.2e against x[8N], x[9N]: %s\n", worst,
                worst_wrong, ok ? "PASS" : worst_wrong <= 1e-6 ? "FAIL (IGC's GEP simplification: x[8N], x[9N] read)"
                                                                  : "FAIL");
    return ok ? 0 : 1;
}
