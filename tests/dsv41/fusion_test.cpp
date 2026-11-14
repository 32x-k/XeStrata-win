// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// Upstream's DS4_HC_FUSE / DS4_ROUTE_FUSE load checks, as a device test for the SYCL port:
// hc_split_pre_norm's outputs are hc_split's and hc_pre_norm's bit for bit (and its int8 rounding the
// activations' own quantization's, the PreQ8 the GEMVs then read), gemv_f32_pair's rows gemv_f32's, and
// route_pair's picks, plan, lookahead and request route_publish's and route_ahead's.
#include "ds4_kernels.cuh"
#include "sycl_test.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {
namespace ds = strata::ds4;

int fails = 0;
void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++fails;
}
template <class T>
bool same(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}
std::vector<float> rnd(size_t n, unsigned& seed, float span = 4.f) {
    std::vector<float> v(n);
    for (auto& x : v) {
        seed = seed * 1664525u + 1013904223u;
        x = ((float) (seed >> 8) / (float) (1 << 24) - 0.5f) * span;
    }
    return v;
}

void test_hc(sycl::queue& q) {
    const int dim = 2048, hc = 4, iters = 2, T = 3;
    unsigned seed = 777u;
    auto h = rnd((size_t) T * hc * dim, seed), mixes = rnd((size_t) T * 32, seed, 2.f),
         scale = rnd(3, seed, 0.5f), base = rnd(64, seed), lag = rnd((size_t) T * 8, seed, 2.f),
         w = rnd(dim, seed, 1.f);
    float *dh, *dmix, *dsc, *dbs, *dlag, *dw, *dx, *dpre, *dpost, *dcomb;
    int8_t* dq;
    float* dqd;
    dsv41_test::device_alloc(dh, h.size() * 4);
    dsv41_test::device_alloc(dmix, mixes.size() * 4);
    dsv41_test::device_alloc(dsc, 3 * 4);
    dsv41_test::device_alloc(dbs, 64 * 4);
    dsv41_test::device_alloc(dlag, lag.size() * 4);
    dsv41_test::device_alloc(dw, w.size() * 4);
    dsv41_test::device_alloc(dx, (size_t) T * dim * 4);
    dsv41_test::device_alloc(dpre, (size_t) T * 8 * 4);
    dsv41_test::device_alloc(dpost, (size_t) T * 8 * 4);
    dsv41_test::device_alloc(dcomb, (size_t) T * 64 * 4);
    dsv41_test::device_alloc(dq, (size_t) T * dim);
    dsv41_test::device_alloc(dqd, (size_t) T * (dim / 32) * 4);
    q.memcpy(dh, h.data(), h.size() * 4);
    q.memcpy(dmix, mixes.data(), mixes.size() * 4);
    q.memcpy(dsc, scale.data(), 12);
    q.memcpy(dbs, base.data(), 256);
    q.memcpy(dlag, lag.data(), lag.size() * 4);
    q.memcpy(dw, w.data(), w.size() * 4);
    q.wait();

    // the two launches' outputs
    std::vector<float> preA(T * 8), postA(T * 8), combA(T * 64), xA(T * dim);
    ds::hc_split(dmix, dsc, dbs, hc, iters, 1e-5f, T, dpre, dpost, dcomb, q);
    ds::hc_pre_norm(dh, dlag, dw, dim, hc, T, 1e-5f, dx, q);
    q.memcpy(preA.data(), dpre, preA.size() * 4);
    q.memcpy(postA.data(), dpost, postA.size() * 4);
    q.memcpy(combA.data(), dcomb, combA.size() * 4);
    q.memcpy(xA.data(), dx, xA.size() * 4);
    q.wait();

    const ds::HcFuse a{dmix, dsc, dbs, hc, iters, 1e-5f, dpre, dpost, dcomb,
                       dh,   dlag, dw,  dim, 1e-5f, dx,   dq,    dqd};
    const int fused = ds::hc_split_pre_norm(a, q, T);
    check(fused > 0, "hc_split_pre_norm launches");
    if (fused == 0) return;
    std::vector<float> preB(T * 8), postB(T * 8), combB(T * 64), xB(T * dim);
    q.memcpy(preB.data(), dpre, preB.size() * 4);
    q.memcpy(postB.data(), dpost, postB.size() * 4);
    q.memcpy(combB.data(), dcomb, combB.size() * 4);
    q.memcpy(xB.data(), dx, xB.size() * 4);
    q.wait();
    check(same(preA, preB), "fused split: pre is hc_split's bits");
    check(same(postA, postB), "fused split: post is hc_split's bits");
    check(same(combA, combB), "fused split: comb is hc_split's bits");
    check(same(xA, xB), "fused pre_norm: x is hc_pre_norm's bits");
    if (fused == 1) {
        check(true, "fused pre_norm: no 32-lane sub-group - x unrounded (PreQ8 unused)");
        return;
    }
    // the int8 rounding, as the activations' quantization makes it: a 32-value block's amax / 127, rint
    std::vector<int8_t> qb((size_t) T * dim);
    std::vector<float> qdb((size_t) T * (dim / 32));
    q.memcpy(qb.data(), dq, qb.size());
    q.memcpy(qdb.data(), dqd, qdb.size() * 4);
    q.wait();
    bool qok = true;
    for (int t = 0; t < T && qok; ++t)
        for (int b = 0; b < dim / 32 && qok; ++b) {
            float am = 0.f;
            for (int e = 0; e < 32; ++e) am = std::max(am, std::fabs(xA[(size_t) t * dim + b * 32 + e]));
            const float sc = am / 127.f;
            qok = qok && qdb[(size_t) t * (dim / 32) + b] == sc;
            for (int e = 0; e < 32; ++e) {
                const int8_t want = int8_t(am > 0.f ? std::lrintf(xA[(size_t) t * dim + b * 32 + e] / sc) : 0);
                qok = qok && qb[(size_t) t * dim + b * 32 + e] == want;
            }
        }
    check(qok, "fused pre_norm: the int8 rounding is the activations' quantization's bits");
}

void test_gemv(sycl::queue& q) {
    const int n0 = 256, n1 = 384, k = 512, T = 3, ldy = 1024;
    unsigned seed = 4242u;
    auto W0 = rnd((size_t) n0 * k, seed), W1 = rnd((size_t) n1 * k, seed), x = rnd((size_t) T * k, seed);
    float *dW0, *dW1, *dxv, *dy0, *dy1;
    dsv41_test::device_alloc(dW0, W0.size() * 4);
    dsv41_test::device_alloc(dW1, W1.size() * 4);
    dsv41_test::device_alloc(dxv, x.size() * 4);
    dsv41_test::device_alloc(dy0, (size_t) T * ldy * 4);
    dsv41_test::device_alloc(dy1, (size_t) T * ldy * 4);
    q.memcpy(dW0, W0.data(), W0.size() * 4);
    q.memcpy(dW1, W1.data(), W1.size() * 4);
    q.memcpy(dxv, x.data(), x.size() * 4);
    q.wait();
    std::vector<float> y0a(T * ldy, -1.f), y1a(T * ldy, -1.f), y0b(T * ldy, -1.f), y1b(T * ldy, -1.f);
    ds::gemv_f32(dW0, dxv, n0, k, T, dy0, ldy, q);
    ds::gemv_f32(dW1, dxv, n1, k, T, dy1, ldy, q);
    q.memcpy(y0a.data(), dy0, y0a.size() * 4);
    q.memcpy(y1a.data(), dy1, y1a.size() * 4);
    q.wait();
    check(ds::gemv_f32_pair(dW0, n0, dy0, dW1, n1, dy1, dxv, k, q, T, ldy), "gemv_f32_pair launches");
    q.memcpy(y0b.data(), dy0, y0b.size() * 4);
    q.memcpy(y1b.data(), dy1, y1b.size() * 4);
    q.wait();
    check(same(y0a, y0b), "gemv_f32_pair: the first gate's rows are gemv_f32's bits");
    check(same(y1a, y1b), "gemv_f32_pair: the next layer's rows are gemv_f32's bits");
}

void test_route(sycl::queue& q) {
    const int n = 384, k = 6, an = 256, m = 20, T = 4, dim = 128;
    unsigned seed = 99u;
    auto logits = rnd((size_t) T * 1024, seed, 8.f), a_logits = rnd((size_t) T * 1024, seed, 8.f),
         bias = rnd(1024, seed, 0.5f), a_bias = rnd(1024, seed, 0.5f), x = rnd((size_t) T * dim, seed);
    std::vector<unsigned long long> vram(1024);
    for (int i = 0; i < 1024; ++i) vram[i] = 0x1000ull + (unsigned long long) i * 64;   // none on the CPU lane
    float *dlog, *dalog, *db, *dab, *dxv, *dw1, *dw2;
    int *dix1, *dix2;
    unsigned long long *dv, *dpl1, *dpl2;
    dsv41_test::device_alloc(dlog, logits.size() * 4);
    dsv41_test::device_alloc(dalog, a_logits.size() * 4);
    dsv41_test::device_alloc(db, 1024 * 4);
    dsv41_test::device_alloc(dab, 1024 * 4);
    dsv41_test::device_alloc(dxv, x.size() * 4);
    dsv41_test::device_alloc(dv, vram.size() * 8);
    dsv41_test::device_alloc(dix1, (size_t) T * ds::kPicks * 4);
    dsv41_test::device_alloc(dix2, (size_t) T * ds::kPicks * 4);
    dsv41_test::device_alloc(dw1, (size_t) T * ds::kPicks * 4);
    dsv41_test::device_alloc(dw2, (size_t) T * ds::kPicks * 4);
    dsv41_test::device_alloc(dpl1, (size_t) T * ds::kPicks * 8);
    dsv41_test::device_alloc(dpl2, (size_t) T * ds::kPicks * 8);
    ds::Req* reqA = sycl::malloc_host<ds::Req>(1, q);
    ds::Req* reqB = sycl::malloc_host<ds::Req>(1, q);
    if (!reqA || !reqB) { check(false, "host request allocation"); return; }
    std::memset(reqA, 0, sizeof *reqA);
    std::memset(reqB, 0, sizeof *reqB);
    q.memcpy(dlog, logits.data(), logits.size() * 4);
    q.memcpy(dalog, a_logits.data(), a_logits.size() * 4);
    q.memcpy(db, bias.data(), bias.size() * 4);
    q.memcpy(dab, a_bias.data(), a_bias.size() * 4);
    q.memcpy(dxv, x.data(), x.size() * 4);
    q.memcpy(dv, vram.data(), vram.size() * 8);
    q.wait();

    ds::route_ahead(dalog, dab, an, m, T, reqA, 0, q);
    ds::route_publish(dlog, db, n, k, T, 1.5f, dv, dxv, dim, dix1, dw1, dpl1, reqA, 77, q,
                      nullptr, 0, m, 0);
    q.wait();

    ds::RoutePair rp;
    rp.logits = dlog; rp.bias = db; rp.n = n; rp.k = k;
    rp.route_scale = 1.5f; rp.vram = dv; rp.x = dxv; rp.dim = dim;
    rp.idx = dix2; rp.w = dw2; rp.plan_ptr = dpl2; rp.req = reqB;
    rp.seq = 77; rp.na0 = m; rp.na1 = 0;
    rp.a_logits = dalog; rp.a_bias = dab; rp.a_n = an; rp.m = m; rp.off = 0;
    rp.T = T;
    check(ds::route_pair(rp, q), "route_pair launches");
    q.wait();

    std::vector<int> ix1(T * ds::kPicks), ix2(T * ds::kPicks);
    std::vector<float> w1(T * ds::kPicks), w2(T * ds::kPicks);
    std::vector<unsigned long long> pl1(T * ds::kPicks), pl2(T * ds::kPicks);
    q.memcpy(ix1.data(), dix1, ix1.size() * 4);
    q.memcpy(ix2.data(), dix2, ix2.size() * 4);
    q.memcpy(w1.data(), dw1, w1.size() * 4);
    q.memcpy(w2.data(), dw2, w2.size() * 4);
    q.memcpy(pl1.data(), dpl1, pl1.size() * 8);
    q.memcpy(pl2.data(), dpl2, pl2.size() * 8);
    q.wait();
    check(same(ix1, ix2), "route_pair: the picks are route_publish's bits");
    check(same(w1, w2), "route_pair: the weights are route_publish's bits");
    check(same(pl1, pl2), "route_pair: the plan is route_publish's");
    check(std::memcmp(reqA->ids, reqB->ids, sizeof reqA->ids) == 0 &&
          std::memcmp(reqA->w, reqB->w, sizeof reqA->w) == 0,
          "route_pair: the request's picks and weights are the separate calls'");
    check(std::memcmp(reqA->ahead, reqB->ahead, sizeof reqA->ahead) == 0,
          "route_pair: the lookahead is route_ahead's bits");
    check(reqA->na[0] == reqB->na[0] && reqA->na[1] == reqB->na[1] && reqA->T == reqB->T,
          "route_pair: the request's counts are the separate calls'");
    check(reqB->seq == 77, "route_pair: the request's sequence is published");
    q.wait();
    sycl::free(reqA, q);
    sycl::free(reqB, q);
}

}  // namespace

int main() try {
    if (sycl::device::get_devices(sycl::info::device_type::gpu).empty()) return 77;
    sycl::queue& q = dsv41_test::queue();
    test_hc(q);
    q.wait_and_throw();
    test_gemv(q);
    q.wait_and_throw();
    test_route(q);
    q.wait_and_throw();
    std::printf("%s\n", fails ? "fusion_test: FAILED" : "fusion_test: all ok");
    return fails ? 1 : 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "fusion_test: %s\n", error.what());
    return 1;
}
