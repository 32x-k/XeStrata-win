// SPDX-FileCopyrightText: 2026 Peasant Smith and the Project Maya contributors
// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// Strata. The trellis routed experts of the DeepSeek-V4.1 path (ds4_tq.cu on the device, ds4_tq_cpu.cpp on the CPU lane) against
// a double reference on host-decoded weights (tools/trellis/tqenc: decode_blocks), for the six types (TQK6 / TQK7 / TQK5X /
// TQK4X / TQK5T / TQK4T): the token path, the window path (grouped entries), the prompt sets (gate/up and down) and the CPU
// lane, plus the two lanes against each other.  Random legal streams: any bit pattern is one.
//   tq_expert_test [--host-only] [--type 145,146,...]
#include "../../src/dsv41/ds4_gguf_cpu.hpp"
#include "../../src/dsv41/ds4_iq.cuh"
#include "../../tools/trellis/tqenc.hpp"

#include "sycl_test.hpp"
using namespace dsv41_test;

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace iq = strata::ds4::gguf::iq;
namespace gg = strata::ds4::gguf;


template <typename T> static T* up(const std::vector<T>& v) {
    T* d; device_alloc(d, std::max<size_t>(1, v.size() * sizeof(T))); queue().memcpy(d, v.data(), v.size() * sizeof(T)).wait_and_throw(); return d;
}
template <typename T> static std::vector<T> down_(const T* d, size_t n) {
    std::vector<T> v(n); strata::core::readback(queue(), v.data(), d, n * sizeof(T)); return v;
}
static float h2f(uint16_t h) {
    const int s = h >> 15, e = (h >> 10) & 31, m = h & 1023;
    float f = e == 0 ? std::ldexp((float) m, -24) : std::ldexp(1.0f + m / 1024.0f, e - 15);
    return s ? -f : f;
}
static uint16_t f2h(float f) {   // round to nearest even, normal range
    uint32_t u; std::memcpy(&u, &f, 4);
    const uint32_t s = (u >> 16) & 0x8000u; int e = (int) ((u >> 23) & 0xff) - 127 + 15; uint32_t m = u & 0x7fffffu;
    uint32_t hm = m >> 13; const uint32_t rem = m & 0x1fffu;
    uint32_t v = ((uint32_t) e << 10) | hm;
    if (rem > 0x1000u || (rem == 0x1000u && (v & 1))) ++v;
    return (uint16_t) (s | v);
}

struct Q8 { std::vector<int8_t> q; std::vector<float> d; std::vector<float> xhat; };
// the engine's q8_quant: sc = max|x| / 127 per 32, q = round-half-even(x / sc)
static Q8 quant(const float* x, int k) {
    Q8 r; r.q.resize(k); r.d.resize(k / 32); r.xhat.resize(k);
    for (int b = 0; b < k / 32; ++b) {
        float am = 0; for (int j = 0; j < 32; ++j) am = std::max(am, std::fabs(x[b * 32 + j]));
        const float sc = am / 127.0f; r.d[b] = sc;
        for (int j = 0; j < 32; ++j) { const int q = am > 0 ? (int) std::nearbyint(x[b * 32 + j] / sc) : 0; r.q[b * 32 + j] = (int8_t) q; r.xhat[b * 32 + j] = (float) q * sc; }
    }
    return r;
}
static double swiglu(double g, double u, double lim) {
    if (lim > 0) { g = std::min(g, lim); u = std::min(std::max(u, -lim), lim); }
    return g / (1.0 + std::exp(-g)) * u;
}

struct Mat { std::vector<uint8_t> bytes; std::vector<float> w; int rows, k; };
static Mat make_mat(std::mt19937_64& rng, int type, int K, int rows, int k) {
    Mat m; m.rows = rows; m.k = k;
    const int bsz = 2 + 4 * K, nblk = k / 128;
    m.bytes.resize((size_t) rows * nblk * bsz);
    for (auto& b : m.bytes) b = (uint8_t) rng();
    for (int64_t b = 0; b < (int64_t) rows * nblk; ++b) { const uint16_t d = f2h(0.05f + 0.0005f * (float) (rng() % 600)); std::memcpy(&m.bytes[(size_t) b * bsz], &d, 2); }
    m.w.resize((size_t) rows * k);
    tqenc::decode_blocks(type, m.bytes.data(), (int64_t) rows * nblk, m.w.data());
    return m;
}
static double dotd(const float* w, const float* x, int k, double* sabs = nullptr) {
    double s = 0, sa = 0; for (int i = 0; i < k; ++i) { const double t = (double) w[i] * x[i]; s += t; sa += std::fabs(t); } if (sabs) *sabs = sa; return s;
}

// This path returns before device enumeration and never calls queue(). It also
// covers rows wider than the CPU vector kernels' fixed activation capacity.
static int host_test(const std::vector<int>& types) {
    if (!gg::tq_cpu_type(145)) {
        std::puts("SKIP: CPU trellis kernels are disabled");
        return 77;
    }
    std::mt19937_64 rng(11);
    std::normal_distribution<float> nd(0.f, 1.f);
    constexpr int FF = 256, T = 3;
    constexpr float limit = 7.f, guard = 123456.f;
    int fails = 0;
    auto relative = [](float value, double ref, double scale) {
        return std::isfinite(value) ? std::fabs(value - ref) / (scale + 1e-30) :
                                     std::numeric_limits<double>::infinity();
    };
    for (int type : types) for (int E : {512, 5120}) {
        gg::IqCpu fm;
        std::string err;
        if (!gg::iq_cpu_fmt(type, type, E, FF, fm, err)) {
            std::printf("type %d width %d CPU format refused: %s\n", type, E, err.c_str());
            ++fails; continue;
        }
        const int kb = tqenc::type_kbits(type);
        const Mat gate = make_mat(rng, type, kb, FF, E), up = make_mat(rng, type, kb, FF, E);
        const Mat down = make_mat(rng, type, kb, E, FF);
        std::vector<std::vector<float>> x(T, std::vector<float>(E)), h(T, std::vector<float>(FF + 2, guard)),
            y(T, std::vector<float>(E + 2, guard)), href(T, std::vector<float>(FF));
        std::vector<std::vector<uint8_t>> act(T, std::vector<uint8_t>(fm.act_bytes)),
            hact(T, std::vector<uint8_t>(fm.h_bytes));
        Q8 qx[T], qh[T];
        const void* ap[T]; const void* hp[T]; float* hout[T]; float* yout[T];
        bool exact_quant = true, guards = true;
        for (int t = 0; t < T; ++t) {
            // A zero row checks all-zero quantization; the others contain outliers.
            if (t) for (float& v : x[t]) v = nd(rng) * (rng() % 7 == 0 ? 4.f : 1.f);
            qx[t] = quant(x[t].data(), E);
            gg::iq_cpu_quant(fm.gu_act, x[t].data(), E, act[t].data());
            exact_quant &= std::memcmp(act[t].data(), qx[t].d.data(), size_t(E / 32) * 4) == 0 &&
                           std::memcmp(act[t].data() + size_t(E / 32) * 4, qx[t].q.data(), E) == 0;
            ap[t] = act[t].data(); hp[t] = hact[t].data();
            hout[t] = h[t].data() + 1; yout[t] = y[t].data() + 1;
            for (int r = 0; r < FF; ++r)
                href[t][r] = (float) swiglu(dotd(gate.w.data() + size_t(r) * E, qx[t].xhat.data(), E),
                                          dotd(up.w.data() + size_t(r) * E, qx[t].xhat.data(), E), limit);
            qh[t] = quant(href[t].data(), FF);
            gg::iq_cpu_quant(fm.d_act, href[t].data(), FF, hact[t].data());
            // The quantizer contract includes identical scale bits, including signed zero.
            // NOLINTNEXTLINE(bugprone-suspicious-memory-comparison)
            exact_quant &= std::memcmp(hact[t].data(), qh[t].d.data(), size_t(FF / 32) * 4) == 0 &&
                           std::memcmp(hact[t].data() + size_t(FF / 32) * 4, qh[t].q.data(), FF) == 0;
        }
        // Nonzero row bounds exercise the CPU lane's partitioned expert calls.
        gg::iq_cpu_gate_up(fm, gate.bytes.data(), up.bytes.data(), ap, T, hout, 0, 13, limit);
        for (int t = 0; t < T; ++t) for (int r = 13; r < FF; ++r) guards &= hout[t][r] == guard;
        gg::iq_cpu_gate_up(fm, gate.bytes.data(), up.bytes.data(), ap, T, hout, 13, FF, limit);
        gg::iq_cpu_down(fm, down.bytes.data(), hp, T, yout, 0, 17);
        for (int t = 0; t < T; ++t) for (int r = 17; r < E; ++r) guards &= yout[t][r] == guard;
        gg::iq_cpu_down(fm, down.bytes.data(), hp, T, yout, 17, E);
        double wg = 0, wd = 0;
        for (int t = 0; t < T; ++t) {
            guards &= h[t].front() == guard && h[t].back() == guard && y[t].front() == guard && y[t].back() == guard;
            for (int r = 0; r < FF; ++r) {
                double sg, su;
                const double g = dotd(gate.w.data() + size_t(r) * E, qx[t].xhat.data(), E, &sg),
                             u = dotd(up.w.data() + size_t(r) * E, qx[t].xhat.data(), E, &su);
                wg = std::max(wg, relative(hout[t][r], swiglu(g, u, limit), sg + su));
            }
            for (int r = 0; r < E; ++r) {
                double sa;
                const double ref = dotd(down.w.data() + size_t(r) * FF, qh[t].xhat.data(), FF, &sa);
                wd = std::max(wd, relative(yout[t][r], ref, sa));
            }
        }
        const bool ok = exact_quant && guards && wg <= 1e-5 && wd <= 1e-6;
        std::printf("type %d width %d host gate/up %.2e down %.2e quant %s guards %s: %s\n",
                    type, E, wg, wd, exact_quant ? "ok" : "FAIL", guards ? "ok" : "FAIL", ok ? "ok" : "FAIL");
        fails += !ok;
    }
    return fails ? 1 : 0;
}

// one decode token of the DeepSeek shape (n_embd 5120, n_ff 2304, top-6 experts, distinct blobs) through the token path: gate/up +
// down launches, elapsed time per layer; and a 5-entry verify window of one expert group
static void bench(int type) {
    const int K = iq::tq_kbits(type), E = 5120, FF = 2304, NE = 6;
    const size_t rb_g = (size_t) (E / 128) * (2 + 4 * K), rb_d = (size_t) (FF / 128) * (2 + 4 * K);
    auto al = [](size_t n) { return (n + 255) / 256 * 256; };
    const size_t off[3] = {0, al(FF * rb_g), al(FF * rb_g) * 2}, blob = off[2] + al((size_t) E * rb_d);
    std::mt19937_64 rng(5);
    std::vector<uint8_t> one(blob);
    for (auto& b : one) b = (uint8_t) rng();
    std::vector<unsigned long long> plan(NE);
    for (int e = 0; e < NE; ++e) { uint8_t* d; device_alloc(d, blob); queue().memcpy(d, one.data(), blob).wait_and_throw(); plan[e] = (unsigned long long) d; }
    std::vector<int8_t> xq(E, 3), hq((size_t) NE * FF, 3); std::vector<float> xd(E / 32, 0.01f), hd((size_t) NE * FF / 32, 0.01f);
    iq::Args a{};
    a.plan_ptr = up(plan); a.n_embd = E; a.n_ff = FF; a.limit = 10.0f; a.xq = up(xq); a.xd = up(xd); a.hq = up(hq); a.hd = up(hd); a.tg = a.td = type;
    for (int r = 0; r < 3; ++r) a.tr[r] = off[r];
    float *dh, *dp; device_alloc(dh, (size_t) NE * FF * 4); device_alloc(dp, (size_t) NE * E * 4); a.h = dh; a.part = dp;
    std::chrono::steady_clock::time_point e0, e1;
    for (int rep = 0; rep < 5; ++rep) { iq::gate_up(a, NE, 1, queue()); iq::down(a, NE, 1, queue()); }
    queue().wait_and_throw();
    float ms_gu = 0, ms_dn = 0;
    for (int part = 0; part < 2; ++part) {
        queue().wait_and_throw(); e0 = std::chrono::steady_clock::now();
        for (int rep = 0; rep < 100; ++rep) { if (part == 0) iq::gate_up(a, NE, 1, queue()); else iq::down(a, NE, 1, queue()); }
        queue().wait_and_throw(); e1 = std::chrono::steady_clock::now();
        float ms; ms = std::chrono::duration<float, std::milli>(e1 - e0).count();
        (part == 0 ? ms_gu : ms_dn) = ms / 100;
    }
    const double bytes = (double) NE * blob;
    std::printf("bench %-6s token path, 6 experts: gate/up %.0f us, down %.0f us, layer %.0f us (%.0f GB/s, %.2f Tw/s), x40 = %.1f ms per token\n", iq::name(type), ms_gu * 1e3,
                ms_dn * 1e3, (ms_gu + ms_dn) * 1e3, bytes / ((ms_gu + ms_dn) * 1e-3) / 1e9, (double) NE * 3 * FF * E / ((ms_gu + ms_dn) * 1e-3) / 1e12, 40.0 * (ms_gu + ms_dn));
}

int main(int argc, char** argv) try {
    std::vector<int> types = {145, 146, 224, 225, 226, 227};
    bool host_only = false, benchmark = false;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--host-only") host_only = true;
        else if (std::string(argv[i]) == "--bench") benchmark = true;
        else if (std::string(argv[i]) == "--type" && i + 1 < argc) { types.clear(); for (const char* p = argv[++i]; *p;) { types.push_back(std::atoi(p)); while (*p && *p != ',') ++p; if (*p) ++p; } }
    if (host_only && benchmark) { std::fprintf(stderr, "--host-only cannot run a GPU benchmark\n"); return 1; }
    for (int type : types) if (!iq::is_tq(type)) { std::fprintf(stderr, "unsupported trellis type %d\n", type); return 1; }
    if (host_only) return host_test(types);
    if (sycl::device::get_devices(sycl::info::device_type::gpu).empty()) {
        std::puts("SKIP: no SYCL GPU"); return 77;
    }
    if (benchmark) { for (int t : {225, 224, 145, 146}) bench(t); return 0; }
    std::mt19937_64 rng(11);
    std::normal_distribution<float> nd(0.f, 1.f);
    int fails = 0;
    auto report = [&](const char* what, int type, double worst, double tol) {
        const bool ok = worst <= tol;
        std::printf("%-6s %-22s worst %.2e (tol %.0e) %s\n", iq::name(type), what, worst, tol, ok ? "ok" : "FAIL");
        fails += !ok;
    };
    for (int type : types) {
        const int K = iq::tq_kbits(type);
        const int E = 512, FF = 256, NEXP = 3, T = 4;
        const float lim = 7.0f;
        const size_t rb_g = (size_t) (E / 128) * (2 + 4 * K), rb_d = (size_t) (FF / 128) * (2 + 4 * K);
        // experts: blob = gate | up | down, each 256-aligned
        auto al = [](size_t n) { return (n + 255) / 256 * 256; };
        const size_t off[3] = {0, al(FF * rb_g), al(FF * rb_g) * 2}, blob = off[2] + al((size_t) E * rb_d);
        std::vector<Mat> G(NEXP), U(NEXP), D(NEXP);
        // h of row r of expert e for the input xv (F32 values), and the size of the operands it was made of
        auto refh = [&](int e, int r, const float* xv, double& scale) {
            double sg, su;
            const double g = dotd(&G[e].w[(size_t) r * E], xv, E, &sg), u = dotd(&U[e].w[(size_t) r * E], xv, E, &su);
            scale = sg + su + 1e-30;
            return swiglu(g, u, lim);
        };
        std::vector<uint8_t> blobs((size_t) NEXP * blob, 0);
        for (int e = 0; e < NEXP; ++e) {
            G[e] = make_mat(rng, type, K, FF, E); U[e] = make_mat(rng, type, K, FF, E); D[e] = make_mat(rng, type, K, E, FF);
            std::memcpy(&blobs[e * blob + off[0]], G[e].bytes.data(), G[e].bytes.size());
            std::memcpy(&blobs[e * blob + off[1]], U[e].bytes.data(), U[e].bytes.size());
            std::memcpy(&blobs[e * blob + off[2]], D[e].bytes.data(), D[e].bytes.size());
        }
        uint8_t* dblob = up(blobs);
        // tokens
        std::vector<float> x((size_t) T * E);
        for (auto& v : x) v = nd(rng) * (1.0f + 3.0f * (rng() % 7 == 0));
        std::vector<Q8> qx(T);
        std::vector<int8_t> xq((size_t) T * E); std::vector<float> xd((size_t) T * E / 32);
        for (int t = 0; t < T; ++t) { qx[t] = quant(&x[(size_t) t * E], E); std::memcpy(&xq[(size_t) t * E], qx[t].q.data(), E); std::memcpy(&xd[(size_t) t * E / 32], qx[t].d.data(), E / 32 * 4); }
        const int8_t* dxq = up(xq); const float* dxd = up(xd);

        // ---- token path: plan = 3 experts, input row 0
        {
            std::vector<unsigned long long> plan(NEXP);
            for (int e = 0; e < NEXP; ++e) plan[e] = (unsigned long long) (dblob + e * blob);
            iq::Args a{};
            a.plan_ptr = up(plan); a.n_embd = E; a.n_ff = FF; a.limit = lim; a.xq = dxq; a.xd = dxd; a.tg = a.td = type;
            for (int r = 0; r < 3; ++r) a.tr[r] = off[r];
            float* dh; device_alloc(dh, (size_t) NEXP * FF * 4); a.h = dh;
            if (!iq::gate_up(a, NEXP, 1, queue())) { std::printf("gate_up refused\n"); return 1; }
            queue().wait_and_throw();
            const auto h = down_(dh, (size_t) NEXP * FF);
            double worst = 0;
            std::vector<float> href((size_t) NEXP * FF);
            for (int e = 0; e < NEXP; ++e)
                for (int r = 0; r < FF; ++r) {
                    double sc;
                    const double ref = refh(e, r, qx[0].xhat.data(), sc);
                    href[(size_t) e * FF + r] = (float) ref;
                    worst = std::max(worst, std::fabs(h[(size_t) e * FF + r] - ref) / sc);
                }
            report("token gate_up", type, worst, 1e-5);
            // down over the reference h, quantized
            std::vector<int8_t> hq((size_t) NEXP * FF); std::vector<float> hd((size_t) NEXP * FF / 32);
            std::vector<Q8> qh(NEXP);
            for (int e = 0; e < NEXP; ++e) { qh[e] = quant(&href[(size_t) e * FF], FF); std::memcpy(&hq[(size_t) e * FF], qh[e].q.data(), FF); std::memcpy(&hd[(size_t) e * FF / 32], qh[e].d.data(), FF / 32 * 4); }
            a.hq = up(hq); a.hd = up(hd);
            float* dp; device_alloc(dp, (size_t) NEXP * E * 4); a.part = dp;
            if (!iq::down(a, NEXP, 1, queue())) { std::printf("down refused\n"); return 1; }
            queue().wait_and_throw();
            const auto part = down_(dp, (size_t) NEXP * E);
            worst = 0;
            for (int e = 0; e < NEXP; ++e)
                for (int r = 0; r < E; ++r) {
                    double sa; const double ref = dotd(&D[e].w[(size_t) r * FF], qh[e].xhat.data(), FF, &sa);
                    worst = std::max(worst, std::fabs(part[(size_t) e * E + r] - ref) / (sa + 1e-30));
                }
            report("token down", type, worst, 1e-6);

            // ---- the CPU lane on the same experts and inputs; against the reference and the device
            gg::IqCpu fm; std::string err;
            if (!gg::iq_cpu_fmt(type, type, E, FF, fm, err)) { std::printf("cpu fmt: %s\n", err.c_str()); ++fails; }
            else {
                std::vector<uint8_t> act(fm.act_bytes), hact(fm.h_bytes);
                gg::iq_cpu_quant(fm.gu_act, x.data(), E, act.data());
                bool same = std::memcmp(act.data(), qx[0].d.data(), E / 32 * 4) == 0 && std::memcmp(act.data() + E / 32 * 4, qx[0].q.data(), E) == 0;
                std::printf("%-6s cpu activation == device quantizer: %s\n", iq::name(type), same ? "ok" : "FAIL");
                fails += !same;
                double wg = 0, wd = 0, wx = 0;
                for (int e = 0; e < NEXP; ++e) {
                    std::vector<float> hc(FF), yc(E);
                    float* hp[1] = {hc.data()}; float* yp[1] = {yc.data()};
                    const void* ap[1] = {act.data()};
                    gg::iq_cpu_gate_up(fm, &blobs[e * blob + off[0]], &blobs[e * blob + off[1]], ap, 1, hp, 0, FF, lim);
                    for (int r = 0; r < FF; ++r) { double sc; refh(e, r, qx[0].xhat.data(), sc); wg = std::max(wg, std::fabs(hc[r] - href[(size_t) e * FF + r]) / sc); wx = std::max(wx, std::fabs(hc[r] - h[(size_t) e * FF + r]) / sc); }
                    gg::iq_cpu_quant(fm.d_act, &href[(size_t) e * FF], FF, hact.data());
                    const void* hpp[1] = {hact.data()};
                    gg::iq_cpu_down(fm, &blobs[e * blob + off[2]], hpp, 1, yp, 0, E);
                    for (int r = 0; r < E; ++r) { double sa; const double ref = dotd(&D[e].w[(size_t) r * FF], qh[e].xhat.data(), FF, &sa); wd = std::max(wd, std::fabs(yc[r] - ref) / (sa + 1e-30)); }
                }
                report("cpu gate_up vs ref", type, wg, 1e-5);
                report("cpu down vs ref", type, wd, 1e-6);
                report("cpu vs device gate_up", type, wx, 1e-5);
            }
        }

        // ---- window path: groups of entries (entry e = t * 8 + i reads input row e >> 3)
        {
            std::vector<unsigned long long> grp = {(unsigned long long) (dblob + 0 * blob), (unsigned long long) (dblob + 2 * blob)};
            std::vector<int> start = {0, 3, 5}, ent = {1, 10, 19, 2, 8}, ng = {2};
            iq::Args a{};
            a.grp_ptr = up(grp); a.grp_start = up(start); a.grp_ent = up(ent); a.n_groups = up(ng);
            a.n_embd = E; a.n_ff = FF; a.limit = lim; a.xq = dxq; a.xd = dxd; a.tg = a.td = type;
            for (int r = 0; r < 3; ++r) a.tr[r] = off[r];
            const int NE = 24;
            float* dh; device_alloc(dh, (size_t) NE * FF * 4); queue().memset(dh, 0, (size_t) NE * FF * 4).wait_and_throw(); a.h = dh;
            iq::gate_up(a, 2, 3, queue());
            queue().wait_and_throw();
            const auto h = down_(dh, (size_t) NE * FF);
            double worst = 0;
            for (int gi = 0; gi < 2; ++gi)
                for (int j = start[gi]; j < start[gi + 1]; ++j) {
                    const int en = ent[j], t = en >> 3, e = gi == 0 ? 0 : 2;
                    for (int r = 0; r < FF; ++r) {
                        double sc;
                        const double ref = refh(e, r, qx[t].xhat.data(), sc);
                        worst = std::max(worst, std::fabs(h[(size_t) en * FF + r] - ref) / sc);
                    }
                }
            report("window gate_up", type, worst, 1e-5);
        }

        // ---- the prompt sets: expert 0 owns sorted rows [0, 37), expert 2 rows [37, 42); input rows idx[sorted]
        {
            const int NR = 42;
            std::vector<int> bounds = {0, 37, 37, 42}, idx(NR);
            for (int i = 0; i < NR; ++i) idx[i] = i % T;
            iq::SetArgs q{};
            q.wbase = dblob; q.stride = blob; for (int r = 0; r < 3; ++r) q.tr[r] = off[r];
            q.bounds = up(bounds); q.row0 = 0; q.max_rows = 37; q.n_exp = 3; q.n_embd = E; q.n_ff = FF; q.limit = lim;
            q.tg = q.td = type; q.xf = up(x); q.idx = up(idx); q.ldx = E;
            uint16_t* dh16; device_alloc(dh16, (size_t) NR * FF * 2); q.h = dh16; q.ldo = FF;
            iq::set_gate_up(q, queue());
            queue().wait_and_throw();
            const auto h16 = down_(dh16, (size_t) NR * FF);
            double worst = 0;
            for (int e = 0; e < 3; ++e)
                for (int lr = bounds[e]; lr < bounds[e + 1]; ++lr)
                    for (int r = 0; r < FF; ++r) {
                        double sc;
                        const double ref = refh(e, r, &x[(size_t) idx[lr] * E], sc);
                        worst = std::max(worst, std::fabs(h2f(h16[(size_t) lr * FF + r]) - ref) / sc);
                    }
            report("set gate_up (f16 out)", type, worst, 1e-3);
            // A missing index means the inputs already follow sorted rows.
            std::vector<float> sorted_x(size_t(NR) * size_t(E));
            for (int row = 0; row < NR; ++row)
                std::copy_n(x.data() + size_t(idx[row]) * size_t(E), E, sorted_x.data() + size_t(row) * size_t(E));
            const auto* original_x = q.xf; const auto* original_idx = q.idx;
            auto* sorted_device = up(sorted_x); q.xf = sorted_device; q.idx = nullptr;
            iq::set_gate_up(q, queue()); queue().wait_and_throw();
            const auto sorted_h = down_(dh16, size_t(NR) * size_t(FF));
            report("set gate_up null idx", type, std::memcmp(h16.data(), sorted_h.data(), h16.size() * sizeof(uint16_t)) != 0, 0);
            q.xf = original_x; q.idx = original_idx; sycl::free(sorted_device, queue());
            // down over those f16 h rows
            q.xh = dh16; q.ldx = FF;
            float* dy; device_alloc(dy, (size_t) NR * E * 4); q.y = dy; q.ldo = E;
            iq::set_down(q, queue());
            queue().wait_and_throw();
            const auto y = down_(dy, (size_t) NR * E);
            worst = 0;
            for (int e = 0; e < 3; ++e)
                for (int lr = bounds[e]; lr < bounds[e + 1]; ++lr) {
                    std::vector<float> hf(FF);
                    for (int c = 0; c < FF; ++c) hf[c] = h2f(h16[(size_t) lr * FF + c]);
                    for (int r = 0; r < E; ++r) {
                        double sa; const double ref = dotd(&D[e].w[(size_t) r * FF], hf.data(), FF, &sa);
                        worst = std::max(worst, std::fabs(y[(size_t) lr * E + r] - ref) / (sa + 1e-30));
                    }
                }
            report("set down", type, worst, 1e-6);
        }
    }
    std::printf("%s\n", fails ? "FAILED" : "ALL OK");
    return fails;
} catch (const std::exception& e) {
    std::fprintf(stderr, "tq_expert_test: %s\n", e.what());
    return 1;
} catch (...) {
    std::fputs("tq_expert_test: unknown failure\n", stderr);
    return 1;
}
