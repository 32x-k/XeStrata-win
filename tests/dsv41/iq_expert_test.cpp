// SPDX-FileCopyrightText: 2026 Peasant Smith and the Project Maya contributors
// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// Strata. The i-quant routed experts of the DeepSeek-V4.1 path (ds4_iq.cu on the device, ds4_iq_cpu.cpp on the CPU lane)
// against ggml's own decode of the same bytes plus a double reference: the token path (gate/up, down), the window path
// (grouped entries), the prompt sets (gate/up and down), and the CPU lane on the same experts.  Random legal blocks: any
// bit pattern with a sane F16 scale is one.  This is the H2 gate for a new type (IQ2_XS first): the kernel's integer sums
// must equal ggml's weights times the same activations.
//   iq_expert_test [--type 16,17,18,21,22,23,29] [--bench] [--iq256 0|1]
#include "../../src/dsv41/ds4_gguf_cpu.hpp"
#include "../../src/dsv41/ds4_gguf.cuh"
#include "../../src/dsv41/ds4_iq.cuh"
#include "../../src/dsv41/ds4_sycl.hpp"

#include "sycl_test.hpp"
using namespace dsv41_test;

#include "ggml.h"   // ggml_get_type_traits(type)->to_float: ggml's decode of the same blocks

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
static double dotd(const float* w, const float* x, int k, double* sabs = nullptr) {
    double s = 0, sa = 0; for (int i = 0; i < k; ++i) { const double t = (double) w[i] * x[i]; s += t; sa += std::fabs(t); } if (sabs) *sabs = sa; return s;
}
// The unquantized fallback accumulates one FP32 FMA per input. A double dot can differ substantially after
// cancellation, before SwiGLU and its FP16 output rounding. Weights still come from ggml's independent decoder.
static float dot_fma(const float* w, const float* x, int k) {
    float sum = 0.0f;
    for (int i = 0; i < k; ++i) sum = std::fma(w[i], x[i], sum);
    return sum;
}
static float swiglu_f32(float gate, float up_value, float limit) {
    if (limit > 0.0f) { gate = std::min(gate, limit); up_value = std::min(std::max(up_value, -limit), limit); }
    return (gate / (1.0f + std::exp(-gate))) * up_value;
}
// an activation buffer of a ggml type (Q8_K for the i-quant rows) decoded by ggml itself
static std::vector<float> act_values(int type, const void* act, int k) {
    std::vector<float> v((size_t) k);
    if (type == 15) {   // Q8_K: float d, int8 qs[256], int16 bsums[16] - no to_float (an intermediate type)
        const int nb = k / 256;
        const size_t bs = ggml_row_size((ggml_type) type, k) / (size_t) nb;
        for (int b = 0; b < nb; ++b) {
            const uint8_t* p = (const uint8_t*) act + (size_t) b * bs;
            float d; std::memcpy(&d, p, 4);
            const int8_t* qs = (const int8_t*) (p + 4);
            for (int j = 0; j < 256; ++j) v[(size_t) b * 256 + j] = d * (float) qs[j];
        }
        return v;
    }
    const ggml_type_traits* t = ggml_get_type_traits((ggml_type) type);
    if (t == nullptr || t->to_float == nullptr) { std::printf("no to_float for activation type %d\n", type); std::exit(2); }
    t->to_float(act, v.data(), k);
    return v;
}

struct Mat { std::vector<uint8_t> bytes; std::vector<float> w; int rows = 0, k = 0; };
// random legal blocks of `type` for a rows x k matrix, decoded by ggml itself
static Mat make_mat(std::mt19937_64& rng, int type, int rows, int k) {
    Mat m; m.rows = rows; m.k = k;
    const int bv = iq::block_values(type), bb = iq::block_bytes(type);
    if (bv <= 0 || bb <= 0 || rows <= 0 || k <= 0 || k % bv != 0)
        throw std::invalid_argument("invalid quantized test matrix shape or type");
    const size_t nblk = (size_t) rows * (k / bv);
    m.bytes.resize(nblk * (size_t) bb);
    for (auto& b : m.bytes) b = (uint8_t) rng();
    for (size_t b = 0; b < nblk; ++b) {   // a sane scale: F16 at offset 0 (every type here but IQ1_M and Q8_K)
        const uint16_t d = f2h(0.02f + 0.0002f * (float) (rng() % 600));
        uint8_t* p = &m.bytes[b * (size_t) bb];
        if (type == 15) {   // block_q8_K's d is F32 at offset 0
            const float df = 0.02f + 0.0002f * (float) (rng() % 600);
            std::memcpy(p, &df, 4);
        } else if (type == 29) {   // block_iq1_m has no d field: its F16 scale is the top nibbles of its four u16 scales
            p[49] = (uint8_t) ((p[49] & 0x0f) | ((d & 0x000f) << 4));
            p[51] = (uint8_t) ((p[51] & 0x0f) | (d & 0x00f0));
            p[53] = (uint8_t) ((p[53] & 0x0f) | ((d & 0x0f00) >> 4));
            p[55] = (uint8_t) ((p[55] & 0x0f) | ((d & 0xf000) >> 8));
        } else {
            std::memcpy(p, &d, 2);
        }
    }
    m.w.resize((size_t) rows * k);
    const size_t rb = (size_t) (k / bv) * (size_t) bb;
    if (type == 15) {   // Q8_K: an intermediate type, no to_float in ggml - {fp32 d; int8 qs[256]; u16 bsums[16]}
        for (int r = 0; r < rows; ++r)
            for (int b = 0; b < k / 256; ++b) {
                const uint8_t* p = &m.bytes[(size_t) r * rb + (size_t) b * 292];
                float d;
                std::memcpy(&d, p, 4);
                const int8_t* qs = (const int8_t*) (p + 4);
                for (int i = 0; i < 256; ++i) m.w[(size_t) r * k + 256 * b + i] = d * (float) qs[i];
            }
        return m;
    }
    if (type == 42) {   // Q2_0 (GSQ-RCO): no ggml decode - value = d (code - 1), u16 word j's low byte the
        for (int r = 0; r < rows; ++r)   // values 8j..8j+3, its high byte 8j+4..8j+7
            for (int b = 0; b < k / 64; ++b) {
                const uint8_t* p = &m.bytes[(size_t) r * rb + (size_t) b * 18];
                const float d = h2f((uint16_t) ((uint32_t) p[0] | (uint32_t) p[1] << 8));
                for (int j = 0; j < 8; ++j) {
                    const uint32_t q = (uint32_t) p[2 + 2 * j] | (uint32_t) p[3 + 2 * j] << 8;
                    for (int m4 = 0; m4 < 4; ++m4) {
                        m.w[(size_t) r * k + 64 * b + 8 * j + m4] = d * (float) ((int) ((q >> (2 * m4)) & 3u) - 1);
                        m.w[(size_t) r * k + 64 * b + 8 * j + 4 + m4] =
                            d * (float) ((int) ((q >> (8 + 2 * m4)) & 3u) - 1);
                    }
                }
            }
        return m;
    }
    const ggml_type_traits* tr = ggml_get_type_traits((ggml_type) type);
    if (tr == nullptr || tr->to_float == nullptr) { std::printf("ggml has no to_float for type %d\n", type); std::exit(2); }
    for (int r = 0; r < rows; ++r) tr->to_float(&m.bytes[(size_t) r * rb], &m.w[(size_t) r * k], k);
    return m;
}

// Exercise the public token APIs with a missing expert and independent ggml-decoded
// references. A byte-offset Q8 input is legal for the portable path but cannot use
// the upstream token kernel's 16-byte activation load.
template <class Report>
static void token_contract(int type, const std::vector<Mat>& g, const std::vector<Mat>& u,
                           const std::vector<Mat>& d, const std::vector<float>& x,
                           uint8_t* blobs, size_t stride, const size_t* offsets,
                           float limit, Report report) {
    if (type != iq::kIQ1_S && type != iq::kIQ1_M && type != iq::kIQ2_XXS &&
        type != iq::kIQ2_XS && type != iq::kIQ2_S && type != iq::kIQ3_XXS) return;
    const int k = g[0].k, ff = g[0].rows, entries = 3;
    constexpr size_t edge = 8;
    constexpr float sentinel = -123456.0f;
    const int expert[entries] = {1, -1, 0};
    const std::vector<unsigned long long> plan = {
        reinterpret_cast<unsigned long long>(blobs + stride), 0,
        reinterpret_cast<unsigned long long>(blobs)};
    const auto* device_plan = up(plan);
    const auto input = quant(x.data(), k);
    std::vector<float> hidden((size_t) entries * ff), hscale((size_t) entries * ff / 32);
    std::vector<int8_t> hcodes(hidden.size());
    std::vector<Q8> hquant(entries);
    for (int e = 0; e < entries; ++e) {
        std::copy_n(x.data() + (size_t) e * k, ff, hidden.data() + (size_t) e * ff);
        hquant[e] = quant(hidden.data() + (size_t) e * ff, ff);
        std::copy(hquant[e].q.begin(), hquant[e].q.end(), hcodes.data() + (size_t) e * ff);
        std::copy(hquant[e].d.begin(), hquant[e].d.end(), hscale.data() + (size_t) e * ff / 32);
    }
    std::vector<int8_t> unaligned_x(input.q.size() + 1), unaligned_h(hcodes.size() + 1);
    std::copy(input.q.begin(), input.q.end(), unaligned_x.begin() + 1);
    std::copy(hcodes.begin(), hcodes.end(), unaligned_h.begin() + 1);
    const auto* xcodes = up(input.q);
    const auto* hcodes_device = up(hcodes);
    const auto* xcodes_offset = up(unaligned_x) + 1;
    const auto* hcodes_offset = up(unaligned_h) + 1;
    const auto* xscale = up(input.d);
    const auto* hscale_device = up(hscale);
    const std::vector<float> initial_h((size_t) entries * ff + 2 * edge, sentinel);
    const std::vector<float> initial_p((size_t) entries * k + 2 * edge, sentinel);
    auto* hstorage = up(initial_h);
    auto* pstorage = up(initial_p);
    auto intact = [&](const std::vector<float>& values) {
        return std::all_of(values.begin(), values.begin() + edge, [&](float v) { return v == sentinel; }) &&
               std::all_of(values.end() - edge, values.end(), [&](float v) { return v == sentinel; });
    };
    auto ratio = [](double actual, double expected, double scale) {
        const double error = std::fabs(actual - expected) / (scale + 1e-30);
        return std::isfinite(actual) && std::isfinite(error) ? error : 1e30;
    };
    auto gate_error = [&](const std::vector<float>& values, bool check_skipped) {
        double worst = 0.0;
        for (int e = 0; e < entries; ++e) for (int r = 0; r < ff; ++r) {
            const float actual = values[edge + (size_t) e * ff + r];
            if (expert[e] < 0) {
                if (check_skipped) worst = std::max(worst, actual == sentinel ? 0.0 : 1e30);
                continue;
            }
            double ga, ua;
            const double gv = dotd(g[expert[e]].w.data() + (size_t) r * k, input.xhat.data(), k, &ga);
            const double uv = dotd(u[expert[e]].w.data() + (size_t) r * k, input.xhat.data(), k, &ua);
            worst = std::max(worst, ratio(actual, swiglu(gv, uv, limit), ga + ua));
        }
        return worst;
    };
    iq::Args a{};
    a.plan_ptr = device_plan; a.n_embd = k; a.n_ff = ff; a.limit = limit; a.tg = a.td = type;
    a.xd = xscale; a.hd = hscale_device; a.h = hstorage + edge; a.part = pstorage + edge;
    std::copy_n(offsets, 3, a.tr);
    for (bool offset : {false, true}) {
        a.xq = offset ? xcodes_offset : xcodes;
        a.hq = offset ? hcodes_offset : hcodes_device;
        queue().memcpy(hstorage, initial_h.data(), initial_h.size() * sizeof(float)).wait_and_throw();
        queue().memcpy(pstorage, initial_p.data(), initial_p.size() * sizeof(float)).wait_and_throw();
        if (!iq::gate_up(a, entries, 1, queue()) || !iq::down(a, entries, 1, queue()))
            throw std::runtime_error("token contract fixture refused");
        const auto actual_h = down_(hstorage, initial_h.size());
        const auto actual_p = down_(pstorage, initial_p.size());
        double worst = 0.0;
        for (int e = 0; e < entries; ++e) for (int r = 0; r < k; ++r) {
            const float actual = actual_p[edge + (size_t) e * k + r];
            if (expert[e] < 0) { worst = std::max(worst, actual == sentinel ? 0.0 : 1e30); continue; }
            double scale;
            const double expected = dotd(d[expert[e]].w.data() + (size_t) r * ff,
                                         hquant[e].xhat.data(), ff, &scale);
            worst = std::max(worst, ratio(actual, expected, scale));
        }
        report(offset ? "token offset gate/up" : "token aligned gate/up", type, gate_error(actual_h, true), 1e-5);
        report(offset ? "token offset down" : "token aligned down", type, worst, 1e-6);
        report(offset ? "token offset guards" : "token aligned guards", type,
               intact(actual_h) && intact(actual_p) ? 0.0 : 1.0, 0.0);
    }

    const std::vector<float> weights = {0.25f, 0.5f, 0.25f};
    std::vector<float> cpu(k), shared(k);
    for (int r = 0; r < k; ++r) { cpu[r] = x[r] * 0.125f; shared[r] = x[k + r] * 0.25f; }
    const auto* cpu_device = up(cpu);
    const auto* shared_device = up(shared);
    auto* flag = up(std::vector<int>{0});
    const auto* xf = up(x);
    const auto* hf = up(hidden);
    const std::vector<float> initial_y((size_t) k + 2 * edge, sentinel);
    auto* ystorage = up(initial_y);
    strata::dsv41k::exl3::MoeArgs ma{};
    ma.plan_ptr = device_plan; ma.plan_w = up(weights); ma.k = entries;
    ma.n_embd = k; ma.n_ff = ff; ma.limit = limit; ma.lay.fmt = gg::kFmtMXFP4;
    for (int role = 0; role < 3; ++role) { ma.lay.K[role] = type; ma.lay.trellis[role] = offsets[role]; }
    ma.cpu_part = cpu_device; ma.cpu_flag = flag;
    queue().memcpy(hstorage, initial_h.data(), initial_h.size() * sizeof(float)).wait_and_throw();
    gg::moe_gate_up(ma, xf, hstorage + edge, queue());
    const auto wrapper_h = down_(hstorage, initial_h.size());
    // The wrapper does not consume an absent expert's h; its contents are not an
    // output contract. The raw IQ API above must leave that skipped slot untouched.
    report("moe gate/up reference", type, gate_error(wrapper_h, false), 1e-5);
    report("moe gate/up guard", type, intact(wrapper_h) ? 0.0 : 1.0, 0.0);
    for (int active : {0, 1}) {
        queue().memcpy(flag, &active, sizeof(active)).wait_and_throw();
        queue().memcpy(ystorage, initial_y.data(), initial_y.size() * sizeof(float)).wait_and_throw();
        gg::moe_down(ma, hf, active ? shared_device : nullptr, ystorage + edge, queue());
        const auto actual_y = down_(ystorage, initial_y.size());
        double worst = 0.0;
        for (int r = 0; r < k; ++r) {
            double expected = 0.0, scale = 0.0;
            for (int e = 0; e < entries; ++e) if (expert[e] >= 0) {
                double sa;
                expected += weights[e] * dotd(d[expert[e]].w.data() + (size_t) r * ff,
                                               hquant[e].xhat.data(), ff, &sa);
                scale += weights[e] * sa;
            }
            if (active) { expected += (double) cpu[r] + shared[r]; scale += std::fabs(cpu[r]) + std::fabs(shared[r]); }
            worst = std::max(worst, ratio(actual_y[edge + r], expected, scale));
        }
        report(active ? "moe down CPU/shared" : "moe down CPU disabled", type, worst, 1e-6);
        report("moe down guard", type, intact(actual_y) ? 0.0 : 1.0, 0.0);
    }
}

// gg::win_experts on a window of T tokens (moe_win_group's groups: each distinct expert once, its entries t * 8 + i)
// against the same tokens one at a time through gg::moe_gate_up / moe_down: the same int8 rows and kernels.
template <class Report>
static void window_contract(int type, const std::vector<float>& x, int k, int ff, uint8_t* blobs, size_t stride,
                            const size_t* offsets, float limit, Report report) {
    if (type != iq::kIQ1_S && type != iq::kIQ1_M && type != iq::kIQ2_XXS &&
        type != iq::kIQ2_XS && type != iq::kIQ2_S && type != iq::kIQ3_XXS && type != 39) return;
    constexpr int T = 4, K = 3, P = 8;
    const int pick[T][K] = {{0, 2, -1}, {2, 1, 0}, {1, -1, 2}, {0, 1, 2}};
    std::vector<unsigned long long> plan((size_t) T * P, 0);
    std::vector<float> weights((size_t) T * P, 0.0f), shared((size_t) T * k);
    std::vector<unsigned long long> gp;
    std::vector<int> gs{0}, ge;
    for (int e = 0; e < 3; ++e) {
        for (int t = 0; t < T; ++t) for (int i = 0; i < K; ++i) if (pick[t][i] == e) ge.push_back(t * P + i);
        if ((int) ge.size() > gs.back()) { gp.push_back(reinterpret_cast<unsigned long long>(blobs + e * stride)); gs.push_back((int) ge.size()); }
    }
    for (int t = 0; t < T; ++t) for (int i = 0; i < K; ++i) {
        if (pick[t][i] >= 0) plan[(size_t) t * P + i] = reinterpret_cast<unsigned long long>(blobs + pick[t][i] * stride);
        weights[(size_t) t * P + i] = 0.125f * (float) (i + 1 + t);
    }
    for (size_t j = 0; j < shared.size(); ++j) shared[j] = x[j % x.size()] * 0.25f;
    const auto* dplan = up(plan);
    const auto* dw = up(weights);
    const auto* dshared = up(shared);
    const auto* dx = up(x);
    strata::dsv41k::exl3::WinArgs w{};
    w.grp_ptr = up(gp); w.grp_start = up(gs); w.grp_ent = up(ge); w.n_groups = up(std::vector<int>{(int) gp.size()});
    w.plan_ptr = dplan; w.plan_w = dw; w.x = dx; w.x_stride = (size_t) k; w.sh_out = dshared;
    float *h, *part, *out;
    device_alloc(h, (size_t) T * P * ff * 4); device_alloc(part, (size_t) T * P * k * 4); device_alloc(out, (size_t) T * k * 4);
    w.h = h; w.part = part; w.out = out; w.out_stride = (size_t) k;
    w.lay.fmt = gg::kFmtMXFP4;
    for (int role = 0; role < 3; ++role) { w.lay.K[role] = type; w.lay.trellis[role] = offsets[role]; }
    w.T = T; w.k = K; w.n_embd = k; w.n_ff = ff; w.limit = limit;
    if (!gg::win_experts(w, queue())) throw std::runtime_error("window contract fixture refused");
    const auto window = down_(out, (size_t) T * k);
    double worst = 0.0;
    bool same = true;
    float *th, *tout;
    device_alloc(th, (size_t) K * ff * 4); device_alloc(tout, (size_t) k * 4);
    for (int t = 0; t < T; ++t) {
        strata::dsv41k::exl3::MoeArgs ma{};
        ma.plan_ptr = dplan + (size_t) t * P; ma.plan_w = dw + (size_t) t * P; ma.k = K;
        ma.n_embd = k; ma.n_ff = ff; ma.limit = limit; ma.lay = w.lay;
        gg::moe_gate_up(ma, dx + (size_t) t * k, th, queue());
        gg::moe_down(ma, th, dshared + (size_t) t * k, tout, queue());
        const auto token = down_(tout, (size_t) k);
        for (int r = 0; r < k; ++r) {
            const float a = window[(size_t) t * k + r], b = token[r];
            uint32_t ab, bb;
            std::memcpy(&ab, &a, 4);
            std::memcpy(&bb, &b, 4);
            same = same && ab == bb;
            worst = std::max(worst, std::isfinite(a) ? std::fabs((double) a - b) / (std::fabs((double) b) + 1e-6) : 1e30);
        }
    }
    report("window against tokens", type, worst, 1e-5);
    std::printf("%-7s window against tokens: %s\n", iq::name(type), same ? "bit-equal" : "within tolerance only");
    if (type == 39) return;
    // The host-routed prompt's public matvec, against independently quantized
    // activation rows consumed by the established IQ entry point.
    std::vector<int8_t> codes(size_t(T) * k);
    std::vector<float> scales(size_t(T) * k / 32);
    for (int t = 0; t < T; ++t) for (int b = 0; b < k / 32; ++b) {
        float amax = 0.f;
        for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(x[size_t(t) * k + size_t(b) * 32 + j]));
        const float scale = amax / 127.f;
        scales[size_t(t) * (k / 32) + b] = scale;
        for (int j = 0; j < 32; ++j)
            codes[size_t(t) * k + size_t(b) * 32 + j] = int8_t(amax > 0 ? std::nearbyint(x[size_t(t) * k + size_t(b) * 32 + j] / scale) : 0);
    }
    strata::dsv41k::exl3::Mat matrix{};
    matrix.fmt = gg::kFmtGG; matrix.K = type; matrix.k = k; matrix.n = ff;
    matrix.raw = blobs + offsets[0];
    strata::dsv41k::exl3::MvJob job{};
    job.m = &matrix; job.xf = dx; job.y = h; job.ncol = T; job.xf_cs = k; job.y_cs = ff;
    if (!gg::mv(&job, 1, queue())) throw std::runtime_error("prompt matvec refused");
    const auto actual = down_(h, size_t(T) * ff);
    iq::MvJob reference{};
    reference.w = blobs + offsets[0]; reference.ld = gg::type_row_bytes(type, k);
    reference.type = type; reference.k = k; reference.n = ff; reference.ncol = T;
    reference.xq = up(codes); reference.xd = up(scales); reference.y = h; reference.ycs = ff;
    if (!iq::mv(&reference, 1, queue())) throw std::runtime_error("prompt reference refused");
    const auto expected = down_(h, size_t(T) * ff);
    double mv_error = 0;
    for (size_t j = 0; j < actual.size(); ++j)
        mv_error = std::max(mv_error, std::fabs(double(actual[j]) - expected[j]) / (std::fabs(double(expected[j])) + 1e-6));
    report("prompt matvec", type, mv_error, 1e-5);
}

template <class Report>
static void mx_contract(Report report) {
    constexpr int type = 39, k = 512, ff = 256, experts = 3;
    const size_t rb = gg::type_row_bytes(type, k), rd = gg::type_row_bytes(type, ff);
    const size_t offsets[3] = {0, size_t(ff) * rb, size_t(2 * ff) * rb};
    const size_t stride = offsets[2] + size_t(k) * rd;
    std::vector<uint8_t> blob(stride * experts);
    for (size_t b = 0; b < blob.size() / 17; ++b) {
        blob[b * 17] = 125;
        for (int j = 1; j < 17; ++j) blob[b * 17 + j] = uint8_t(b * 37 + size_t(j) * 13);
    }
    std::vector<float> x(size_t(4) * k);
    for (size_t j = 0; j < x.size(); ++j) x[j] = float(int(j % 31) - 15) * 0.001f;
    auto* device = up(blob);
    window_contract(type, x, k, ff, device, stride, offsets, 7.f, report);
    std::vector<unsigned long long> plan(experts);
    std::vector<float> weights{.25f, .5f, .75f}, shared(k, .125f);
    for (int e = 0; e < experts; ++e) plan[e] = reinterpret_cast<unsigned long long>(device + size_t(e) * stride);
    strata::dsv41k::exl3::MoeArgs a{};
    a.plan_ptr = up(plan); a.plan_w = up(weights); a.k = experts; a.n_embd = k; a.n_ff = ff; a.limit = 7.f;
    a.lay.fmt = gg::kFmtMXFP4;
    for (int r = 0; r < 3; ++r) a.lay.trellis[r] = offsets[r];
    float *h, *out; device_alloc(h, size_t(experts) * ff * 4); device_alloc(out, size_t(k) * 4);
    gg::moe_gate_up(a, up(x), h, queue());
    const auto actual_h = down_(h, size_t(experts) * ff);
    const auto dot = [](const float* w, const Q8& q) {
        float sum = 0;
        for (size_t b = 0; b < q.d.size(); ++b) {
            float block = 0;
            for (int j = 0; j < 32; ++j) block += float(q.q[b * 32 + j]) * w[b * 32 + j];
            sum = std::fma(q.d[b], block, sum);
        }
        return sum;
    };
    const Q8 input = quant(x.data(), k);
    std::vector<float> decoded(size_t(k) * ff), gate(ff), expected(k, .125f);
    double gu_error = 0;
    const auto* traits = ggml_get_type_traits((ggml_type)type);
    for (int e = 0; e < experts; ++e) {
        traits->to_float(blob.data() + size_t(e) * stride, decoded.data(), int64_t(k) * ff);
        for (int r = 0; r < ff; ++r) gate[r] = dot(decoded.data() + size_t(r) * k, input);
        traits->to_float(blob.data() + size_t(e) * stride + offsets[1], decoded.data(), int64_t(k) * ff);
        for (int r = 0; r < ff; ++r) {
            const float reference = swiglu_f32(gate[r], dot(decoded.data() + size_t(r) * k, input), 7.f);
            gu_error = std::max(gu_error, std::fabs(double(actual_h[size_t(e) * ff + r]) - reference) / (1 + std::fabs(double(reference))));
        }
        const Q8 hidden = quant(actual_h.data() + size_t(e) * ff, ff);
        traits->to_float(blob.data() + size_t(e) * stride + offsets[2], decoded.data(), int64_t(k) * ff);
        for (int r = 0; r < k; ++r) expected[r] += weights[e] * dot(decoded.data() + size_t(r) * ff, hidden);
    }
    iq::Args fallback{};
    fallback.plan_ptr = a.plan_ptr; fallback.n_embd = k; fallback.n_ff = ff; fallback.limit = a.limit;
    for (int r = 0; r < 3; ++r) fallback.tr[r] = offsets[r];
    // An unaligned code buffer selects the existing portable fallback on every
    // device, without changing its dimensions or quantized values.
    const auto unaligned_codes = [&](const std::vector<int8_t>& codes) {
        std::vector<int8_t> padded(codes.size() + 16, 0);
        auto* base = up(padded);
        const size_t offset = (reinterpret_cast<uintptr_t>(base) & 15u) == 0 ? 1 : 0;
        std::copy(codes.begin(), codes.end(), padded.begin() + static_cast<std::ptrdiff_t>(offset));
        strata::core::copy_async(queue(), base, padded.data(), padded.size()).wait_and_throw();
        return base + offset;
    };
    fallback.tg = fallback.td = type;
    fallback.xq = unaligned_codes(input.q); fallback.xd = up(input.d);
    float *fallback_h, *fallback_part;
    device_alloc(fallback_h, size_t(experts) * ff * 4); device_alloc(fallback_part, size_t(experts) * k * 4);
    fallback.h = fallback_h; fallback.part = fallback_part;
    if (!iq::gate_up(fallback, experts, 1, queue())) throw std::runtime_error("MX portable gate refused");
    const auto old_h = down_(fallback_h, size_t(experts) * ff);
    double fallback_gu = 0;
    for (size_t j = 0; j < old_h.size(); ++j)
        fallback_gu = std::max(fallback_gu, std::fabs(double(actual_h[j]) - old_h[j]) / (1 + std::fabs(double(old_h[j]))));
    std::vector<int8_t> hidden_codes;
    std::vector<float> hidden_scales;
    for (int e = 0; e < experts; ++e) {
        const Q8 hidden = quant(actual_h.data() + size_t(e) * ff, ff);
        hidden_codes.insert(hidden_codes.end(), hidden.q.begin(), hidden.q.end());
        hidden_scales.insert(hidden_scales.end(), hidden.d.begin(), hidden.d.end());
    }
    fallback.hq = unaligned_codes(hidden_codes); fallback.hd = up(hidden_scales);
    if (!iq::down(fallback, experts, 1, queue())) throw std::runtime_error("MX portable down refused");
    const auto old_part = down_(fallback_part, size_t(experts) * k);
    gg::moe_down(a, h, up(shared), out, queue());
    const auto actual = down_(out, k);
    double down_error = 0;
    for (int r = 0; r < k; ++r) down_error = std::max(down_error, std::fabs(double(actual[r]) - expected[r]) / (1 + std::fabs(double(expected[r]))));
    report("MX gate ggml reference", type, gu_error, 1e-5);
    report("MX down ggml reference", type, down_error, 1e-5);
    double fallback_down = 0;
    for (int r = 0; r < k; ++r) {
        float reference = 0;
        for (int e = 0; e < experts; ++e) reference = std::fma(weights[e], old_part[size_t(e) * k + r], reference);
        reference += shared[r];
        fallback_down = std::max(fallback_down, std::fabs(double(actual[r]) - reference) / (1 + std::fabs(double(reference))));
    }
    report("MX gate portable path", type, fallback_gu, 1e-5);
    report("MX down portable path", type, fallback_down, 1e-5);
}

// Default upstream runr declines N=64/192 although the vector-load alignment is
// valid. These 17-row sets must retain FP32 inputs/weights in the FMA fallback.
template <class Report>
static void matrix_precision_boundary(Report report) {
    constexpr int type = iq::kIQ2_XXS, k = 256, rows = 17;
    constexpr float limit = 7.0f, guard_y = -321.0f;
    constexpr uint16_t guard_h = 0x3555;
    // Fixed seed keeps the boundary fixture reproducible.
    std::mt19937_64 rng(8164);  // NOLINT(bugprone-random-generator-seed)
    std::uniform_real_distribution<float> values(-0.25f, 0.25f);
    for (const int n : {64, 192}) {
        const int ldo = n + 2;   // even, so the old matrix eligibility would accept it
        const Mat gate = make_mat(rng, type, n, k), up_mat = make_mat(rng, type, n, k);
        std::vector<uint8_t> blob = gate.bytes;
        blob.insert(blob.end(), up_mat.bytes.begin(), up_mat.bytes.end());
        std::vector<float> xf(size_t(rows) * k), half_values(xf.size());
        std::vector<uint16_t> xh(xf.size()), zeros(xf.size(), 0);
        for (size_t j = 0; j < xf.size(); ++j) {
            xf[j] = values(rng);
            xh[j] = strata::ds4::sycl_detail::f16_bits(xf[j]);
            half_values[j] = h2f(xh[j]);
        }
        std::vector<int> bounds{7, 7 + rows}, indices(rows);
        for (int r = 0; r < rows; ++r) indices[r] = rows - 1 - r;
        const std::vector<uint16_t> initial_h(size_t(rows + 1) * ldo, guard_h);
        const std::vector<float> initial_y(initial_h.size(), guard_y);
        iq::SetArgs a{};
        a.wbase = up(blob); a.stride = blob.size(); a.tr[1] = gate.bytes.size();
        a.bounds = up(bounds); a.row0 = 7; a.n_exp = 1; a.max_rows = rows;
        a.n_embd = k; a.n_ff = n; a.tg = a.td = type; a.limit = limit;
        a.idx = up(indices); a.xf = up(xf); a.xf16 = up(zeros); a.xh = up(xh);
        a.ldx = k; a.ldo = ldo; a.h = up(initial_h); a.y = up(initial_y);
        if (!iq::set_gate_up(a, queue())) throw std::runtime_error("precision-boundary gate/up refused");
        const auto actual_h = down_(a.h, initial_h.size());
        double gate_worst = 0;
        bool gate_guard = true;
        for (int r = 0; r <= rows; ++r) {
            for (int c = 0; c < ldo; ++c) {
                const size_t at = size_t(r) * ldo + c;
                if (r == rows || c >= n) { gate_guard = gate_guard && actual_h[at] == guard_h; continue; }
                const float* input = xf.data() + size_t(indices[r]) * k;
                const float g = dot_fma(gate.w.data() + size_t(c) * k, input, k);
                const float u = dot_fma(up_mat.w.data() + size_t(c) * k, input, k);
                const double expected = swiglu_f32(g, u, limit);
                const double error = std::abs(h2f(actual_h[at]) - expected) / (1e-4 + 1e-3 * std::abs(expected));
                gate_worst = std::max(gate_worst, std::isfinite(error) ? error : 1e30);
            }
        }
        char label[64];
        std::snprintf(label, sizeof(label), "set gate N%d FMA", n); report(label, type, gate_worst, 1.0);
        std::snprintf(label, sizeof(label), "set gate N%d guard", n); report(label, type, gate_guard ? 0.0 : 1.0, 0.0);
        // Only down's N-by-K weights are used. Reuse gate's decoded matrix and
        // avoid constructing unused opposite-projection rows with K=64/192.
        a.n_embd = n; a.n_ff = k;
        if (!iq::set_down(a, queue())) throw std::runtime_error("precision-boundary down refused");
        const auto actual_y = down_(a.y, initial_y.size());
        double down_worst = 0;
        bool down_guard = true;
        for (int r = 0; r <= rows; ++r) {
            for (int c = 0; c < ldo; ++c) {
                const size_t at = size_t(r) * ldo + c;
                if (r == rows || c >= n) { down_guard = down_guard && actual_y[at] == guard_y; continue; }
                const float* weight = gate.w.data() + size_t(c) * k;
                const float* input = half_values.data() + size_t(r) * k;
                double scale;
                dotd(weight, input, k, &scale);
                const double expected = dot_fma(weight, input, k);
                const double error = std::abs(actual_y[at] - expected) / (1e-7 + 1e-6 * scale);
                down_worst = std::max(down_worst, std::isfinite(error) ? error : 1e30);
            }
        }
        std::snprintf(label, sizeof(label), "set down N%d FMA", n); report(label, type, down_worst, 1.0);
        std::snprintf(label, sizeof(label), "set down N%d guard", n); report(label, type, down_guard ? 0.0 : 1.0, 0.0);
    }
}

// The 16-row boundary changes the activations' numerical representation, independently of the device's tiles.
// CPU references use ggml's decoded weights and the host quantizer above, never the device decoder.
template <class Report>
static void light_contract(int type, const std::vector<Mat>& g, const std::vector<Mat>& u,
                           const std::vector<Mat>& d, const std::vector<float>& x,
                           const uint8_t* weights, size_t stride, const size_t* offsets, float limit, Report report) {
    if (!iq::f16_rows_ok(type)) return;
    const char* disabled = std::getenv("STRATA_DS4_NO_LIGHT");
    if (disabled && disabled[0] == '1') return;
    const int k = g[0].k, ff = g[0].rows, nt = (int) (x.size() / k), nr = 33;
    if (nt <= 0) throw std::invalid_argument("empty light-set activation fixture");
    const int ldx = k + 3, ldh = ff + 3, ldy = k + 5;
    const uint16_t guard_h = 0x3555;
    const float guard_y = -321.0f;
    std::vector<int> bounds{7, 7, 23, 40}, idx(nr);
    std::vector<float> xp((size_t) nt * ldx, -19.0f), xhalf(xp.size());
    std::vector<uint16_t> x16(xp.size()), zeros(xp.size(), 0);
    for (int t = 0; t < nt; ++t)
        for (int j = 0; j < k; ++j) xp[(size_t) t * ldx + j] = x[(size_t) t * k + j];
    for (size_t i = 0; i < xp.size(); ++i) {
        x16[i] = strata::ds4::sycl_detail::f16_bits(xp[i]);
        xhalf[i] = h2f(x16[i]);
    }
    for (int r = 0; r < nr; ++r) idx[r] = (nr - 1 - r) % nt;
    std::vector<Q8> xquant(nt);
    for (int t = 0; t < nt; ++t) xquant[t] = quant(xp.data() + (size_t) t * ldx, k);
    iq::SetArgs a{};
    a.wbase = weights; a.stride = stride;
    for (int r = 0; r < 3; ++r) a.tr[r] = offsets[r];
    a.bounds = up(bounds); a.row0 = 7; a.n_exp = 3; a.max_rows = 17; a.n_embd = k; a.n_ff = ff;
    a.limit = limit; a.tg = a.td = type; a.idx = up(idx); a.ldx = ldx; a.ldo = ldh;
    const float* dxf = up(xp);
    const uint16_t* dx16 = up(x16), *dzero = up(zeros);
    std::vector<uint16_t> initial((size_t) nr * ldh, guard_h);
    a.h = up(initial);
    const char* labels[] = {"light gate FP32 priority", "set gate FP32 priority 16/17", "set gate FP32 only",
                            "set gate FP16 only", "set gate max_rows=16"};
    for (int phase = 0; phase < 5; ++phase) {
        a.n_exp = phase == 4 ? 2 : 3; a.max_rows = phase == 4 ? 16 : 17;
        a.xf = phase == 3 ? nullptr : dxf;
        a.xf16 = phase < 2 ? dzero : phase == 3 ? dx16 : nullptr;
        queue().memcpy(a.h, initial.data(), initial.size() * sizeof(uint16_t)).wait_and_throw();
        if (phase == 3 && iq::light_gate_up(a, queue()))
            throw std::runtime_error("light gate/up accepted a missing FP32 input");
        const bool accepted = phase == 0 ? iq::light_gate_up(a, queue()) : iq::set_gate_up(a, queue());
        if (!accepted) throw std::runtime_error("light/set gate/up refused the regression fixture");
        const auto actual = down_(a.h, initial.size());
        double worst = 0;
        int worst_row = -1, worst_col = -1;
        double worst_expected = 0;
        struct ReferenceDelta {
            double error = 0, gate = 0, gate_fma = 0, gate_abs = 0;
            double up_value = 0, up_fma = 0, up_abs = 0, h = 0, h_fma = 0;
            int row = 0, col = 0;
        } delta;
        const auto remember = [&](double error, int row, int col, double expected) {
            if (!std::isfinite(error)) error = 1e30;
            if (error > worst) { worst = error; worst_row = row; worst_col = col; worst_expected = expected; }
        };
        for (int r = 0; r < nr; ++r) {
            const int e = r < 16 ? 1 : 2;
            const bool untouched = (phase == 0 || phase == 4) && r >= 16;
            const float* input = phase == 3 ? xhalf.data() + (size_t) idx[r] * ldx :
                r < 16 ? xquant[idx[r]].xhat.data() : xp.data() + (size_t) idx[r] * ldx;
            for (int c = 0; c < ldh; ++c) {
                const size_t at = (size_t) r * ldh + c;
                if (c >= ff || untouched) {
                    if (actual[at] != guard_h) remember(1e30, r, c, h2f(guard_h));
                    continue;
                }
                const float* wg = g[e].w.data() + (size_t) c * k;
                const float* wu = u[e].w.data() + (size_t) c * k;
                double gate_abs, up_abs;
                const double gate = dotd(wg, input, k, &gate_abs), up_value = dotd(wu, input, k, &up_abs);
                double expected = swiglu(gate, up_value, limit);
                if (phase == 3 || r >= 16) {
                    const float gate_fma = dot_fma(wg, input, k), up_fma = dot_fma(wu, input, k);
                    const double h_fma = swiglu_f32(gate_fma, up_fma, limit);
                    const double difference = std::abs(h_fma - expected) / (1e-4 + 1e-3 * std::abs(expected));
                    if (difference > delta.error)
                        delta = {difference, gate, gate_fma, gate_abs, up_value, up_fma, up_abs,
                                    expected, h_fma, r, c};
                    expected = h_fma;
                }
                const double error = std::abs(h2f(actual[at]) - expected) / (1e-4 + 1e-3 * std::abs(expected));
                remember(error, r, c, expected);
            }
        }
        if (worst > 1.0) {
            const uint16_t bits = actual[(size_t) worst_row * ldh + worst_col];
            std::fprintf(stderr, "%s %s: row=%d col=%d expert=%d actual=%g bits=0x%04x expected=%g sentinel=%d\n",
                iq::name(type), labels[phase], worst_row, worst_col, worst_row < 16 ? 1 : 2,
                h2f(bits), (unsigned) bits, worst_expected, bits == guard_h);
        }
        if (delta.error > 1.0)
            std::printf("%s %s CPU reference: row=%d col=%d gate64=%.9g gate_fma=%.9g sum_abs=%.9g "
                        "up64=%.9g up_fma=%.9g sum_abs=%.9g h64=%.9g h_fma=%.9g\n",
                iq::name(type), labels[phase], delta.row, delta.col, delta.gate, delta.gate_fma, delta.gate_abs,
                delta.up_value, delta.up_fma, delta.up_abs, delta.h, delta.h_fma);
        report(labels[phase], type, worst, 1.0);
    }
    std::vector<uint16_t> hin((size_t) nr * ldh, guard_h);
    std::vector<std::vector<float>> hf(nr, std::vector<float>(ff));
    std::vector<Q8> hquant(nr);
    for (int r = 0; r < nr; ++r) {
        for (int c = 0; c < ff; ++c) {
            hin[(size_t) r * ldh + c] = strata::ds4::sycl_detail::f16_bits(x[((size_t) r * ff + c) % x.size()]);
            hf[r][c] = h2f(hin[(size_t) r * ldh + c]);
        }
        hquant[r] = quant(hf[r].data(), ff);
    }
    a.xh = up(hin); a.ldx = ldh; a.ldo = ldy;
    std::vector<float> initial_y((size_t) nr * ldy, guard_y);
    a.y = up(initial_y);
    for (int phase = 0; phase < 3; ++phase) {
        a.n_exp = phase == 2 ? 2 : 3; a.max_rows = phase == 2 ? 16 : 17;
        queue().memcpy(a.y, initial_y.data(), initial_y.size() * sizeof(float)).wait_and_throw();
        const bool accepted = phase == 0 ? iq::light_down(a, queue()) : iq::set_down(a, queue());
        if (!accepted) throw std::runtime_error("light/set down refused the regression fixture");
        const auto actual = down_(a.y, initial_y.size());
        double worst = 0;
        int worst_row = -1, worst_col = -1;
        double worst_expected = 0;
        const auto remember = [&](double error, int row, int col, double expected) {
            if (!std::isfinite(error)) error = 1e30;
            if (error > worst) { worst = error; worst_row = row; worst_col = col; worst_expected = expected; }
        };
        for (int r = 0; r < nr; ++r) {
            const int e = r < 16 ? 1 : 2;
            const float* input = r < 16 ? hquant[r].xhat.data() : hf[r].data();
            for (int c = 0; c < ldy; ++c) {
                const size_t at = (size_t) r * ldy + c;
                if (c >= k || ((phase == 0 || phase == 2) && r >= 16)) {
                    if (actual[at] != guard_y) remember(1e30, r, c, guard_y);
                    continue;
                }
                double scale;
                const double expected = dotd(d[e].w.data() + (size_t) c * ff, input, ff, &scale);
                const double error = std::abs(actual[at] - expected) / (scale + 1e-30);
                remember(error, r, c, expected);
            }
        }
        const char* label = phase == 0 ? "light down Q8 reference" : phase == 1 ? "set down 16/17" : "set down max_rows=16";
        if (worst > 1e-6) {
            const float value = actual[(size_t) worst_row * ldy + worst_col];
            std::fprintf(stderr, "%s %s: row=%d col=%d expert=%d actual=%g expected=%g sentinel=%d\n",
                iq::name(type), label, worst_row, worst_col, worst_row < 16 ? 1 : 2,
                value, worst_expected, value == guard_y);
        }
        report(label, type, worst, 1e-6);
    }
}

// one decode token of the DeepSeek shape (n_embd 5120, n_ff 2304, top-6 experts): gate/up + down launches, per-layer time
static void bench(int type) {
    const int E = 5120, FF = 2304, NE = 6;
    const size_t rb_g = iq::row_bytes(type, E), rb_d = iq::row_bytes(type, FF);
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
    std::printf("bench %-7s 6 experts x 40 layers: gate/up %.0f us, down %.0f us, layer %.0f us -> %.0f GB/s, %.2f Tw/s, 40 layers %.1f ms/token\n",
                iq::name(type), ms_gu * 1e3, ms_dn * 1e3, (ms_gu + ms_dn) * 1e3,
                bytes / ((ms_gu + ms_dn) * 1e-3) / 1e9,
                (double) NE * 3 * FF * E / ((ms_gu + ms_dn) * 1e-3) / 1e12, 40.0 * (ms_gu + ms_dn));
}


// --dump F / --cmp F: every device output of a run as raw bits, to a file / against one (a kernel change that keeps the sums
// exact and the float order must leave all of them identical)
static FILE* g_dump = nullptr; static FILE* g_cmp = nullptr; static long g_cmp_bad = 0, g_cmp_n = 0, g_cmp_rms_n = 0;
static void bits(const char* what, int type, const void* p, size_t bytes) {
    if (!g_dump && !g_cmp) return;
    if (g_dump) { std::fwrite(p, 1, bytes, g_dump); return; }
    std::vector<uint8_t> ref(bytes);
    if (std::fread(ref.data(), 1, bytes, g_cmp) != bytes) { std::printf("cmp: golden file too short at %s %s\n", iq::name(type), what); ++g_cmp_bad; return; }
    ++g_cmp_n;
    if (std::memcmp(ref.data(), p, bytes) != 0) {
        size_t bad = 0, first = 0; for (size_t i = 0; i < bytes; ++i) if (ref[i] != ((const uint8_t*) p)[i]) { if (!bad) first = i; ++bad; }
        std::printf("%-7s %-22s BITS DIFFER: %zu of %zu bytes (first at %zu)\n", iq::name(type), what, bad, bytes, first);
        ++g_cmp_bad;
    }
}
// the real shapes (n_embd 5120, n_ff 2304): the plan path (6 groups of one entry) and the grouped path (windows of 3 and 2)
// Compare the changed light outputs to the same independent FP64 reference.
// Other outputs retain the existing exact comparison. The golden file stores
// the previous kernel's values; its binary layout remains unchanged.
static double dot_q8_double(const float* w, const Q8& x, int k) {
    double sum = 0;
    for (int i = 0; i < k; ++i) sum += double(w[i]) * (double(x.d[size_t(i) / 32]) * double(x.q[i]));
    return sum;
}
template <typename T>
static void bits_rms(const char* what, int type, const std::vector<T>& actual,
                     const std::vector<double>& reference, const std::vector<size_t>& positions, int cols) {
    if (!g_cmp) { bits(what, type, actual.data(), actual.size() * sizeof(T)); return; }
    std::vector<T> previous(actual.size());
    if (std::fread(previous.data(), sizeof(T), previous.size(), g_cmp) != previous.size()) {
        std::printf("cmp: golden file too short at %s %s\n", iq::name(type), what); ++g_cmp_bad; return;
    }
    ++g_cmp_n; ++g_cmp_rms_n;
    if (reference.size() != positions.size() || reference.size() % size_t(cols)) throw std::runtime_error("RMS reference shape");
    const auto value = [](T v) { if constexpr (sizeof(T) == 2) return double(h2f(uint16_t(v))); else return double(v); };
    std::vector<bool> changed(actual.size(), false);
    std::vector<double> absolute[2], relative[2];
    bool finite = true;
    for (size_t row = 0; row < reference.size(); row += size_t(cols)) {
        double sum = 0;
        for (int col = 0; col < cols; ++col) sum += reference[row + size_t(col)] * reference[row + size_t(col)];
        const double rms = std::sqrt(sum / cols);
        for (int col = 0; col < cols; ++col) {
            const size_t at = row + size_t(col), pos = positions[at];
            if (pos >= actual.size()) throw std::runtime_error("RMS output position");
            changed[pos] = true;
            for (int arm = 0; arm < 2; ++arm) {
                const double v = arm ? value(actual[pos]) : value(previous[pos]);
                const double error = std::fabs(v - reference[at]);
                finite &= std::isfinite(v) && std::isfinite(error) && std::isfinite(rms);
                absolute[arm].push_back(error);
                if (rms > 0) relative[arm].push_back(error / rms);
            }
        }
    }
    bool exact = true;
    for (size_t i = 0; i < actual.size(); ++i)
        if (!changed[i]) exact &= std::memcmp(reinterpret_cast<const unsigned char*>(&actual[i]),
                                             reinterpret_cast<const unsigned char*>(&previous[i]), sizeof(T)) == 0;
    const auto median = [](std::vector<double>& values) {
        if (values.empty()) return 0.;
        const auto mid = static_cast<std::vector<double>::difference_type>(values.size() / 2);
        std::nth_element(values.begin(), values.begin() + mid, values.end());
        return values.size() % 2 ? values[mid] : (values[mid] + *std::max_element(values.begin(), values.begin() + mid)) * .5;
    };
    double stats[2][4]{};
    for (int arm = 0; arm < 2; ++arm) {
        stats[arm][0] = absolute[arm].empty() ? 0 : *std::max_element(absolute[arm].begin(), absolute[arm].end());
        stats[arm][2] = relative[arm].empty() ? 0 : *std::max_element(relative[arm].begin(), relative[arm].end());
        stats[arm][1] = median(absolute[arm]); stats[arm][3] = median(relative[arm]);
    }
    bool ok = finite && exact;
    for (int metric = 0; metric < 4; ++metric) ok &= stats[1][metric] <= 1.1 * stats[0][metric];
    std::printf("cmp %s %s FP64 abs max/median %.9g/%.9g -> %.9g/%.9g, row-RMS relative max/median %.9g/%.9g -> %.9g/%.9g: %s\n",
                iq::name(type), what, stats[0][0], stats[0][1], stats[1][0], stats[1][1],
                stats[0][2], stats[0][3], stats[1][2], stats[1][3], ok ? "ok" : "FAIL");
    if (!ok) ++g_cmp_bad;
}

static void real_shape_bits(int type) {
    std::mt19937_64 rng(1000 + type);   // (its own stream: the main loop's random experts are the same with or without this)
    const int E = 5120, FF = 2304, NG = 2, NEnt = 12;
    const size_t rb_g = iq::row_bytes(type, E), rb_d = iq::row_bytes(type, FF);
    auto al = [](size_t n) { return (n + 255) / 256 * 256; };
    const size_t off[3] = {0, al(FF * rb_g), al(FF * rb_g) * 2}, blob = off[2] + al((size_t) E * rb_d);
    std::vector<uint8_t> hb((size_t) NG * blob);
    for (auto& b : hb) b = (uint8_t) rng();
    for (size_t b = 0; b < hb.size() / (size_t) iq::block_bytes(type); ++b) {   // sane F16 scales (the same placement as make_mat)
        const uint16_t d = f2h(0.02f + 0.0002f * (float) (rng() % 600));
        uint8_t* p = &hb[b * (size_t) iq::block_bytes(type)];
        if (type == 29) { p[49] = (uint8_t) ((p[49] & 0x0f) | ((d & 0x000f) << 4)); p[51] = (uint8_t) ((p[51] & 0x0f) | (d & 0x00f0)); p[53] = (uint8_t) ((p[53] & 0x0f) | ((d & 0x0f00) >> 4)); p[55] = (uint8_t) ((p[55] & 0x0f) | ((d & 0xf000) >> 8)); }
        else std::memcpy(p, &d, 2);
    }
    uint8_t* dblob = up(hb);
    std::vector<int8_t> xq(2 * E), hq((size_t) NEnt * FF); std::vector<float> xd(2 * E / 32), hd((size_t) NEnt * FF / 32);
    for (auto& v : xq) v = (int8_t) ((int) (rng() % 255) - 127);
    for (auto& v : hq) v = (int8_t) ((int) (rng() % 255) - 127);
    for (auto& v : xd) v = 0.002f + 0.0001f * (float) (rng() % 100);
    for (auto& v : hd) v = 0.002f + 0.0001f * (float) (rng() % 100);
    iq::Args a{};
    a.n_embd = E; a.n_ff = FF; a.limit = 10.0f; a.xq = up(xq); a.xd = up(xd); a.hq = up(hq); a.hd = up(hd); a.tg = a.td = type;
    for (int r = 0; r < 3; ++r) a.tr[r] = off[r];
    float *dh, *dp; device_alloc(dh, (size_t) NEnt * FF * 4); device_alloc(dp, (size_t) NEnt * E * 4); a.h = dh; a.part = dp;
    std::vector<unsigned long long> plan = {(unsigned long long) dblob, (unsigned long long) (dblob + blob), 0ull, (unsigned long long) dblob};
    a.plan_ptr = up(plan);
    queue().memset(dh, 0, (size_t) NEnt * FF * 4).wait_and_throw(); queue().memset(dp, 0, (size_t) NEnt * E * 4).wait_and_throw();
    if (!iq::gate_up(a, 4, 1, queue()) || !iq::down(a, 4, 1, queue())) { std::printf("real shape: refused\n"); return; }
    queue().wait_and_throw();
    { auto h = down_(dh, (size_t) NEnt * FF); bits("real plan h", type, h.data(), h.size() * 4); auto p = down_(dp, (size_t) NEnt * E); bits("real plan part", type, p.data(), p.size() * 4); }
    std::vector<unsigned long long> gp = {(unsigned long long) dblob, (unsigned long long) (dblob + blob)};
    std::vector<int> gs = {0, 3, 5}, ge = {0, 1, 2, 3, 4}, ng = {NG};
    a.plan_ptr = nullptr; a.grp_ptr = up(gp); a.grp_start = up(gs); a.grp_ent = up(ge); a.n_groups = up(ng);
    queue().memset(dh, 0, (size_t) NEnt * FF * 4).wait_and_throw(); queue().memset(dp, 0, (size_t) NEnt * E * 4).wait_and_throw();
    if (!iq::gate_up(a, NG, 3, queue()) || !iq::down(a, NG, 3, queue())) { std::printf("real shape grouped: refused\n"); return; }
    queue().wait_and_throw();
    { auto h = down_(dh, (size_t) NEnt * FF); bits("real grp h", type, h.data(), h.size() * 4); auto p = down_(dp, (size_t) NEnt * E); bits("real grp part", type, p.data(), p.size() * 4); }
    // more windows: a bound of 7 groups with 3 live (dynamic count), 5/1/3 entries (ne_max 5: the 8-entry kernels), then 9 groups of
    // one entry with 4 live; entries index the h/hq rows, their input row is entry >> 3
    {
        std::vector<unsigned long long> gp2 = {(unsigned long long) dblob, (unsigned long long) (dblob + blob), (unsigned long long) dblob, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull};
        std::vector<int> gs2 = {0, 5, 6, 9, 9, 9, 9, 9, 9, 9}, ge2 = {3, 1, 4, 0, 2, 7, 5, 6, 8, 11, 10, 9}, ng2 = {3};
        a.grp_ptr = up(gp2); a.grp_start = up(gs2); a.grp_ent = up(ge2); a.n_groups = up(ng2);
        queue().memset(dh, 0, (size_t) NEnt * FF * 4).wait_and_throw(); queue().memset(dp, 0, (size_t) NEnt * E * 4).wait_and_throw();
        if (!iq::gate_up(a, 7, 5, queue()) || !iq::down(a, 7, 5, queue())) { std::printf("real shape window 2: refused\n"); return; }
        queue().wait_and_throw();
        { auto h = down_(dh, (size_t) NEnt * FF); bits("real win2 h", type, h.data(), h.size() * 4); auto p = down_(dp, (size_t) NEnt * E); bits("real win2 part", type, p.data(), p.size() * 4); }
        std::vector<int> gs3 = {0, 1, 2, 3, 4, 4, 4, 4, 4, 4}, ng3 = {4};
        std::vector<unsigned long long> gp3 = {(unsigned long long) (dblob + blob), (unsigned long long) dblob, (unsigned long long) (dblob + blob), (unsigned long long) dblob, 0ull, 0ull, 0ull, 0ull, 0ull};
        a.grp_ptr = up(gp3); a.grp_start = up(gs3); a.n_groups = up(ng3);
        queue().memset(dh, 0, (size_t) NEnt * FF * 4).wait_and_throw(); queue().memset(dp, 0, (size_t) NEnt * E * 4).wait_and_throw();
        if (!iq::gate_up(a, 9, 1, queue()) || !iq::down(a, 9, 1, queue())) { std::printf("real shape window 3: refused\n"); return; }
        queue().wait_and_throw();
        { auto h = down_(dh, (size_t) NEnt * FF); bits("real win3 h", type, h.data(), h.size() * 4); auto p = down_(dp, (size_t) NEnt * E); bits("real win3 part", type, p.data(), p.size() * 4); }
    }
    device_free(dblob);
}

// the same DeepSeek decode token over a working set far past L2 (>= 640 MiB of distinct expert blobs, rotated launch by launch),
// so the rate is DRAM's, not L2's: gate/up and down timed apart; ne = entries a group (1: the decode token)
static void bench_big(int type, int mib) {
    if (!iq::supported(type) || mib <= 0) throw std::invalid_argument("invalid benchmark type or size");
    const int E = 5120, FF = 2304, NE = 6;
    const size_t rb_g = iq::row_bytes(type, E), rb_d = iq::row_bytes(type, FF);
    auto al = [](size_t n) { return (n + 255) / 256 * 256; };
    const size_t off[3] = {0, al(FF * rb_g), al(FF * rb_g) * 2}, blob = off[2] + al((size_t) E * rb_d);
    if (blob == 0) throw std::invalid_argument("invalid benchmark block size");
    const int L = std::max(2, (int) (((size_t) mib << 20) / (NE * blob)) + 1);   // layers rotated
    std::mt19937_64 rng(5);
    std::vector<uint8_t> one(blob);
    for (auto& b : one) b = (uint8_t) rng();
    std::vector<unsigned long long> plan((size_t) L * NE);
    for (auto& p : plan) { uint8_t* d; device_alloc(d, blob); queue().memcpy(d, one.data(), blob).wait_and_throw(); p = (unsigned long long) d; }
    const int NW = std::getenv("DS4_BENCH_WIN") ? std::max(1, std::min(16, std::atoi(std::getenv("DS4_BENCH_WIN")))) : 0;   // window path: NW entries a group
    const int NEN = std::max(1, NW) * NE;   // entries
    std::vector<int8_t> xq((size_t) 8 * E, 3), hq((size_t) NEN * FF, 3); std::vector<float> xd((size_t) 8 * E / 32, 0.01f), hd((size_t) NEN * FF / 32, 0.01f);
    for (size_t i = 0; i < xq.size(); ++i) xq[i] = (int8_t) ((int) (rng() % 255) - 127);
    for (size_t i = 0; i < hq.size(); ++i) hq[i] = (int8_t) ((int) (rng() % 255) - 127);
    if (const char* e = std::getenv("DS4_BENCH_ACTIVE")) { const int na = std::atoi(e); for (int l = 0; l < L; ++l) for (int k = na; k < NE; ++k) plan[(size_t) l * NE + k] = 0; }   // (empty plan entries)
    unsigned long long* dplan = up(plan);
    iq::Args a{};
    a.n_embd = E; a.n_ff = FF; a.limit = 10.0f; a.xq = up(xq); a.xd = up(xd); a.hq = up(hq); a.hd = up(hd); a.tg = a.td = type;
    for (int r = 0; r < 3; ++r) a.tr[r] = off[r];
    float *dh, *dp; device_alloc(dh, (size_t) NEN * FF * 4); device_alloc(dp, (size_t) NEN * E * 4); a.h = dh; a.part = dp;
    std::vector<int> gstart(NE + 1), gent(NEN), ngrp(1, NE);
    for (int g = 0; g <= NE; ++g) gstart[g] = g * std::max(1, NW);
    for (int i = 0; i < NEN; ++i) gent[i] = i;
    a.grp_start = up(gstart); a.grp_ent = up(gent); a.n_groups = up(ngrp);
    std::chrono::steady_clock::time_point e0, e1;
    double us[2] = {0, 0};
    for (int part = 0; part < 2; ++part) {
        auto run = [&](int reps) {
            queue().wait_and_throw(); e0 = std::chrono::steady_clock::now();
            for (int rep = 0; rep < reps; ++rep) {
                if (NW) a.grp_ptr = dplan + (size_t) (rep % L) * NE; else a.plan_ptr = dplan + (size_t) (rep % L) * NE;
                if (part == 0) iq::gate_up(a, NE, std::max(1, NW), queue()); else iq::down(a, NE, std::max(1, NW), queue());
            }
            queue().wait_and_throw(); e1 = std::chrono::steady_clock::now();
            float ms; ms = std::chrono::duration<float, std::milli>(e1 - e0).count();
            return (double) ms;
        };
        double ms = 0; int reps = 2 * L;
        while (ms < 300.0) { ms = run(reps); reps *= 2; }   // warm-up: clocks ramp, ~300 ms
        const int n = std::max(L, (int) (reps / 2 * 40.0 / ms));   // ~40 ms a window
        double best = 1e30;
        for (int w = 0; w < 7; ++w) best = std::min(best, run(n) * 1e3 / n);   // the best of 7 windows (clocks/thermals wander)
        us[part] = best;
    }
    const double bg = (double) NE * (off[2]) , bd = (double) NE * al((size_t) E * rb_d);   // bytes read a launch
    std::printf("big %-7s ws %4.0f MiB: gate/up %6.1f us (%5.0f GB/s)  down %6.1f us (%5.0f GB/s)  layer %6.1f us -> %5.0f GB/s, %.2f Tw/s\n",
                iq::name(type), (double) L * NE * blob / 1048576.0, us[0], bg / us[0] * 1e-3, us[1], bd / us[1] * 1e-3, us[0] + us[1],
                (bg + bd) / (us[0] + us[1]) * 1e-3, (double) NE * 3 * FF * E / ((us[0] + us[1]) * 1e-6) / 1e12);
    for (auto p : plan) if (p) device_free((void*) p);
}

// the light sets (experts of <= 16 rows: DSpark verify windows, short prompts): n_exp experts of `rows` rows each, gate/up and down
static void bench_light(int type, int nexp, int rows, bool tiles = false) {
    const int E = 5120, FF = 2304;
    const size_t rb_g = iq::row_bytes(type, E), rb_d = iq::row_bytes(type, FF);
    auto al = [](size_t n) { return (n + 255) / 256 * 256; };
    const size_t off[3] = {0, al(FF * rb_g), al(FF * rb_g) * 2}, blob = off[2] + al((size_t) E * rb_d);
    std::mt19937_64 rng(7);
    std::vector<uint8_t> one(blob);
    for (auto& b : one) b = (uint8_t) rng();
    uint8_t* w; device_alloc(w, blob * nexp);
    for (int e = 0; e < nexp; ++e) queue().memcpy(w + (size_t) e * blob, one.data(), blob).wait_and_throw();
    const int NR = nexp * rows, T = 8;
    if (tiles && rows <= 16) { std::printf("tiles: rows > 16 only\n"); return; }
    std::vector<int> bounds(nexp + 1), idx(NR);
    for (int e = 0; e <= nexp; ++e) bounds[e] = e * rows;
    for (int i = 0; i < NR; ++i) idx[i] = i % T;
    std::vector<float> x((size_t) T * E);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : x) v = nd(rng);
    std::vector<uint16_t> h((size_t) NR * FF, 0x3c00);
    iq::SetArgs q{};
    q.wbase = w; q.stride = blob; for (int r = 0; r < 3; ++r) q.tr[r] = off[r];
    q.bounds = up(bounds); q.row0 = 0; q.max_rows = rows; q.skip_le = 0; q.n_exp = nexp; q.n_embd = E; q.n_ff = FF; q.limit = 7.0f;
    q.tg = q.td = type; q.xf = up(x); q.idx = up(idx); q.ldx = E;
    uint16_t* dh; device_alloc(dh, (size_t) NR * FF * 2); q.h = dh; q.ldo = FF;
    q.xh = up(h);
    float* dy; device_alloc(dy, (size_t) NR * E * 4);
    std::chrono::steady_clock::time_point e0, e1;
    double us[2] = {0, 0};
    for (int part = 0; part < 2; ++part) {
        auto run = [&](int reps) {
            queue().wait_and_throw(); e0 = std::chrono::steady_clock::now();
            for (int rep = 0; rep < reps; ++rep) {
                if (part == 0) { q.h = dh; q.ldo = FF; if (tiles) iq::set_gate_up(q, queue()); else iq::light_gate_up(q, queue()); }
                else { q.y = dy; q.ldo = E; q.ldx = FF; if (tiles) iq::set_down(q, queue()); else iq::light_down(q, queue()); }
            }
            queue().wait_and_throw(); e1 = std::chrono::steady_clock::now();
            float ms; ms = std::chrono::duration<float, std::milli>(e1 - e0).count();
            return (double) ms;
        };
        double ms = 0; int reps = 4;
        while (ms < 200.0) { ms = run(reps); reps *= 2; }
        const int n = std::max(2, (int) (reps / 2 * 40.0 / ms));
        double best = 1e30;
        for (int wnd = 0; wnd < 5; ++wnd) best = std::min(best, run(n) * 1e3 / n);
        us[part] = best;
    }
    const double bytes = (double) nexp * (off[2] + al((size_t) E * rb_d));
    std::printf("%s %-7s %3d experts x %2d rows: gate/up %7.1f us  down %7.1f us  -> %6.0f GB/s of weights, %.2f Tw/s, %.1f TMAC/s\n", tiles ? "tiles" : "light", iq::name(type), nexp, rows,
                us[0], us[1], bytes / ((us[0] + us[1]) * 1e-6) / 1e9, (double) nexp * 3 * FF * E / ((us[0] + us[1]) * 1e-6) / 1e12, (double) NR * 3 * FF * E / ((us[0] + us[1]) * 1e-6) / 1e12);
    device_free(w);
}

int main(int argc, char** argv) try {
    if (sycl::device::get_devices(sycl::info::device_type::gpu).empty()) {
        std::puts("SKIP: no SYCL GPU"); return 77;
    }
    // Append new types: the existing fixtures share one RNG stream in this order.
    std::vector<int> types = {16, 17, 22, 18, 21, 23, 29, 19};
    std::vector<int> devs;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--bench-tiles") {   // [types] [n_exp] [rows list]: the prompt sets' optional matrix-engine tiles
            std::vector<int> bt, rl = {32, 128};
            int nexp = 48;
            if (i + 1 < argc && argv[i + 1][0] != '-') { for (const char* p = argv[++i]; *p;) { bt.push_back(std::atoi(p)); while (*p && *p != ',') ++p; if (*p) ++p; } }
            else bt = types;
            if (i + 1 < argc && argv[i + 1][0] != '-') nexp = std::atoi(argv[++i]);
            if (i + 1 < argc && argv[i + 1][0] != '-') { rl.clear(); for (const char* p = argv[++i]; *p;) { rl.push_back(std::atoi(p)); while (*p && *p != ',') ++p; if (*p) ++p; } }
            for (int t : bt) for (int r : rl) bench_light(t, nexp, r, true);
            return 0;
        } else if (std::string(argv[i]) == "--bench-light") {   // [types] [n_exp] [rows list]
            std::vector<int> bt, rl = {1, 2, 4};
            int nexp = 24;
            if (i + 1 < argc && argv[i + 1][0] != '-') { for (const char* p = argv[++i]; *p;) { bt.push_back(std::atoi(p)); while (*p && *p != ',') ++p; if (*p) ++p; } }
            else bt = types;
            if (i + 1 < argc && argv[i + 1][0] != '-') nexp = std::atoi(argv[++i]);
            if (i + 1 < argc && argv[i + 1][0] != '-') { rl.clear(); for (const char* p = argv[++i]; *p;) { rl.push_back(std::atoi(p)); while (*p && *p != ',') ++p; if (*p) ++p; } }
            for (int t : bt) for (int r : rl) bench_light(t, nexp, r);
            return 0;
        } else if (std::string(argv[i]) == "--bench-big") {   // [types] [MiB]
            std::vector<int> bt; int mib = 640;
            if (i + 1 < argc && argv[i + 1][0] != '-') { for (const char* p = argv[++i]; *p;) { bt.push_back(std::atoi(p)); while (*p && *p != ',') ++p; if (*p) ++p; } }
            else bt = types;
            if (i + 1 < argc && argv[i + 1][0] != '-') mib = std::atoi(argv[++i]);
            for (int t : bt) bench_big(t, mib);
            return 0;
        } else if (std::string(argv[i]) == "--bench") {
            std::vector<int> bt;
            if (i + 1 < argc && argv[i + 1][0] != '-') { for (const char* p = argv[++i]; *p;) { bt.push_back(std::atoi(p)); while (*p && *p != ',') ++p; if (*p) ++p; } }
            else bt = types;
            for (int t : bt) bench(t);
            return 0;
        } else if (std::string(argv[i]) == "--devices" && i + 1 < argc) {   // every check on each device in turn, in this process;
            devs.clear(); for (const char* p = argv[++i]; *p;) { devs.push_back(std::atoi(p)); while (*p && *p != ',') ++p; if (*p) ++p; }   // the later ones bit-compared to the first
        } else if (std::string(argv[i]) == "--dump" && i + 1 < argc) {
            g_dump = std::fopen(argv[++i], "wb"); if (!g_dump) { std::printf("cannot write %s\n", argv[i]); return 2; }
        } else if (std::string(argv[i]) == "--cmp" && i + 1 < argc) {
            g_cmp = std::fopen(argv[++i], "rb"); if (!g_cmp) { std::printf("cannot read %s\n", argv[i]); return 2; }
        } else if (std::string(argv[i]) == "--type" && i + 1 < argc) {
            types.clear(); for (const char* p = argv[++i]; *p;) { types.push_back(std::atoi(p)); while (*p && *p != ',') ++p; if (*p) ++p; }
        } else if (std::string(argv[i]) == "--iq256" && i + 1 < argc) {
            gg::gg_cpu_set_iq256(std::atoi(argv[++i]));
        }
    }
    int fails = 0;
    if (devs.size() > 1) {   // the first device writes its outputs, the others compare to them
        if (g_dump || g_cmp) { std::printf("--devices takes neither --dump nor --cmp\n"); return 2; }
    }
    for (size_t di = 0; di < std::max<size_t>(1, devs.size()); ++di) {
    if (!devs.empty()) {
        select_device(devs[di]);
        if (devs.size() > 1) {
            if (g_dump) { std::fclose(g_dump); g_dump = nullptr; }
            if (di == 0) g_dump = std::fopen("iq_devices.bin", "wb"); else { g_cmp = std::fopen("iq_devices.bin", "rb"); g_cmp_n = g_cmp_bad = g_cmp_rms_n = 0; }
        }
        std::printf("---- device %d\n", devs[di]);
    }
    std::mt19937_64 rng(11);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto report = [&](const char* what, int type, double worst, double tol) {
        const bool ok = worst <= tol;
        std::printf("%-7s %-22s worst %.2e (tol %.0e) %s\n", iq::name(type), what, worst, tol, ok ? "ok" : "FAIL");
        fails += !ok;
    };
    mx_contract(report);
    for (int type : types) {
        if (!iq::supported(type) && !iq::known(type)) { std::printf("%-7s not a type this engine knows\n", iq::name(type)); ++fails; continue; }
        if (type == iq::kIQ2_XXS) matrix_precision_boundary(report);
        const bool dev = iq::supported(type);   // (IQ1_S / IQ1_M / IQ4_NL: a known layout, no device kernel - CPU lane only)
        // IQ1 exercises a complete K32 subgroup in the changed light path.
        // Other types retain their existing fixture and exact comparisons.
        const bool lane_light = type == 19 || type == 29;
        const int E = lane_light ? 32 * 32 : 512, FF = lane_light ? E : 256, NEXP = 3, T = 4;
        const float lim = 7.0f;
        const size_t rb_g = iq::row_bytes(type, E), rb_d = iq::row_bytes(type, FF);
        auto al = [](size_t n) { return (n + 255) / 256 * 256; };
        const size_t off[3] = {0, al(FF * rb_g), al(FF * rb_g) * 2}, blob = off[2] + al((size_t) E * rb_d);
        std::vector<Mat> G(NEXP), U(NEXP), D(NEXP);
        auto refh = [&](int e, int r, const float* xv, double& scale) {
            double sg, su;
            const double g = dotd(&G[e].w[(size_t) r * E], xv, E, &sg), u = dotd(&U[e].w[(size_t) r * E], xv, E, &su);
            scale = sg + su + 1e-30;
            return swiglu(g, u, lim);
        };
        std::vector<uint8_t> blobs((size_t) NEXP * blob, 0);
        for (int e = 0; e < NEXP; ++e) {
            G[e] = make_mat(rng, type, FF, E); U[e] = make_mat(rng, type, FF, E); D[e] = make_mat(rng, type, E, FF);
            std::memcpy(&blobs[e * blob + off[0]], G[e].bytes.data(), G[e].bytes.size());
            std::memcpy(&blobs[e * blob + off[1]], U[e].bytes.data(), U[e].bytes.size());
            std::memcpy(&blobs[e * blob + off[2]], D[e].bytes.data(), D[e].bytes.size());
        }
        uint8_t* dblob = up(blobs);
        std::vector<float> x((size_t) T * E);
        for (auto& v : x) v = nd(rng) * (1.0f + 3.0f * (rng() % 7 == 0));
        std::vector<Q8> qx(T);
        std::vector<int8_t> xq((size_t) T * E); std::vector<float> xd((size_t) T * E / 32);
        for (int t = 0; t < T; ++t) { qx[t] = quant(&x[(size_t) t * E], E); std::memcpy(&xq[(size_t) t * E], qx[t].q.data(), E); std::memcpy(&xd[(size_t) t * E / 32], qx[t].d.data(), E / 32 * 4); }
        const int8_t* dxq = up(xq); const float* dxd = up(xd);

        // ---- the reference h of every expert's rows for token 0 (the CPU lane and the device share it)
        std::vector<float> href((size_t) NEXP * FF);
        std::vector<Q8> qh(NEXP);
        for (int e = 0; e < NEXP; ++e)
            for (int r = 0; r < FF; ++r) {
                double sc;
                href[(size_t) e * FF + r] = (float) refh(e, r, qx[0].xhat.data(), sc);
            }
        for (int e = 0; e < NEXP; ++e) qh[e] = quant(&href[(size_t) e * FF], FF);

        // ---- token path (device; skipped for a type with no device kernel)
        if (dev) {
            std::vector<unsigned long long> plan(NEXP);
            for (int e = 0; e < NEXP; ++e) plan[e] = (unsigned long long) (dblob + e * blob);
            iq::Args a{};
            a.plan_ptr = up(plan); a.n_embd = E; a.n_ff = FF; a.limit = lim; a.xq = dxq; a.xd = dxd; a.tg = a.td = type;
            for (int r = 0; r < 3; ++r) a.tr[r] = off[r];
            float* dh; device_alloc(dh, (size_t) NEXP * FF * 4); a.h = dh;
            if (!iq::gate_up(a, NEXP, 1, queue())) { std::printf("gate_up refused\n"); return 1; }
            queue().wait_and_throw();
            const auto h = down_(dh, (size_t) NEXP * FF);
            bits("token h", type, h.data(), h.size() * 4);
            double worst = 0;
            for (int e = 0; e < NEXP; ++e)
                for (int r = 0; r < FF; ++r) {
                    double sc;
                    const double ref = refh(e, r, qx[0].xhat.data(), sc);
                    worst = std::max(worst, std::fabs(h[(size_t) e * FF + r] - ref) / sc);
                }
            report("token gate_up", type, worst, 1e-5);
            std::vector<int8_t> hq((size_t) NEXP * FF); std::vector<float> hd((size_t) NEXP * FF / 32);
            for (int e = 0; e < NEXP; ++e) { std::memcpy(&hq[(size_t) e * FF], qh[e].q.data(), FF); std::memcpy(&hd[(size_t) e * FF / 32], qh[e].d.data(), FF / 32 * 4); }
            a.hq = up(hq); a.hd = up(hd);
            float* dp; device_alloc(dp, (size_t) NEXP * E * 4); a.part = dp;
            if (!iq::down(a, NEXP, 1, queue())) { std::printf("down refused\n"); return 1; }
            queue().wait_and_throw();
            const auto part = down_(dp, (size_t) NEXP * E);
            bits("token part", type, part.data(), part.size() * 4);
            worst = 0;
            for (int e = 0; e < NEXP; ++e)
                for (int r = 0; r < E; ++r) {
                    double sa; const double ref = dotd(&D[e].w[(size_t) r * FF], qh[e].xhat.data(), FF, &sa);
                    worst = std::max(worst, std::fabs(part[(size_t) e * E + r] - ref) / (sa + 1e-30));
                }
            report("token down", type, worst, 1e-6);
            token_contract(type, G, U, D, x, dblob, blob, off, lim, report);
            window_contract(type, x, E, FF, dblob, blob, off, lim, report);
            light_contract(type, G, U, D, x, dblob, blob, off, lim, report);

            // ---- the prompt sets: expert 0 owns rows [0, 37), expert 2 [37, 42)
            const int NR = 42;
            std::vector<int> bounds = {0, 37, 37, 42}, idx(NR);
            for (int i = 0; i < NR; ++i) idx[i] = i % T;
            iq::SetArgs q{};
            q.wbase = dblob; q.stride = blob; for (int r = 0; r < 3; ++r) q.tr[r] = off[r];
            q.bounds = up(bounds); q.row0 = 0; q.max_rows = 37; q.n_exp = 3; q.n_embd = E; q.n_ff = FF; q.limit = lim;
            q.tg = q.td = type; q.xf = up(x); q.idx = up(idx); q.ldx = E;
            uint16_t* dh16; device_alloc(dh16, (size_t) NR * FF * 2); q.h = dh16; q.ldo = FF;
            if (!iq::set_gate_up(q, queue())) { std::printf("set_gate_up refused\n"); return 1; }
            queue().wait_and_throw();
            const auto h16 = down_(dh16, (size_t) NR * FF);
            // Compact routing must preserve every output bit, including an
            // empty expert between two populated experts and the light split.
            q.host_bounds = bounds.data();
            if (!iq::set_gate_up(q, queue())) throw std::runtime_error("compact gate/up refused");
            const auto compact_h = down_(dh16, h16.size());
            if (std::memcmp(h16.data(), compact_h.data(), h16.size() * sizeof(uint16_t)) != 0)
                throw std::runtime_error("compact gate/up differs from rectangular grid");
            q.host_bounds = nullptr;
            if ((type == 19 || type == 29) && g_cmp) {
                std::vector<double> reference; std::vector<size_t> positions;
                for (int e = 0; e < NEXP; ++e) if (bounds[e + 1] - bounds[e] <= 16)
                    for (int lr = bounds[e]; lr < bounds[e + 1]; ++lr) {
                        const Q8 input = quant(x.data() + size_t(idx[lr]) * E, E);
                        for (int col = 0; col < FF; ++col) {
                            reference.push_back(swiglu(dot_q8_double(G[e].w.data() + size_t(col) * E, input, E),
                                                       dot_q8_double(U[e].w.data() + size_t(col) * E, input, E), lim));
                            positions.push_back(size_t(lr) * FF + col);
                        }
                    }
                bits_rms("set h16", type, h16, reference, positions, FF);
            } else bits("set h16", type, h16.data(), h16.size() * 2);
            double w2 = 0;
            for (int e = 0; e < 3; ++e)
                for (int lr = bounds[e]; lr < bounds[e + 1]; ++lr)
                    for (int r = 0; r < FF; ++r) {
                        double sc;
                        const double ref = refh(e, r, &x[(size_t) idx[lr] * E], sc);
                        w2 = std::max(w2, std::fabs((double) h2f(h16[(size_t) lr * FF + r]) - ref) / sc);
                    }
            // (the prompt sets quantise the activations, so this worst case is data dependent: IQ1_S measured 7.1e-3 .. 1.08e-2 over
            // different random-data orders, the same on the older build, with every other check far inside; IQ1_S gets 1.5e-2)
            report("set gate_up (f16)", type, w2, type == 19 ? 1.5e-2 : 1e-2);
            // Keep down inputs independent of gate/up rounding, so both
            // kernels' golden comparisons use the same quantized operands.
            std::vector<uint16_t> down_input = h16;
            if (lane_light) for (size_t i = 0; i < down_input.size(); ++i)
                down_input[i] = strata::ds4::sycl_detail::f16_bits(x[i % x.size()]);
            q.xh = lane_light ? up(down_input) : dh16; q.ldx = FF;
            float* dy; device_alloc(dy, (size_t) NR * E * 4); q.y = dy; q.ldo = E;
            if (!iq::set_down(q, queue())) { std::printf("set_down refused\n"); return 1; }
            queue().wait_and_throw();
            const auto y = down_(dy, (size_t) NR * E);
            q.host_bounds = bounds.data();
            if (!iq::set_down(q, queue())) throw std::runtime_error("compact down refused");
            const auto compact_y = down_(dy, y.size());
            if (std::memcmp(y.data(), compact_y.data(), y.size() * sizeof(float)) != 0)
                throw std::runtime_error("compact down differs from rectangular grid");
            q.host_bounds = nullptr;
            if ((type == 19 || type == 29) && g_cmp) {
                std::vector<double> reference; std::vector<size_t> positions;
                for (int e = 0; e < NEXP; ++e) if (bounds[e + 1] - bounds[e] <= 16)
                    for (int lr = bounds[e]; lr < bounds[e + 1]; ++lr) {
                        std::vector<float> hidden(FF);
                        for (int col = 0; col < FF; ++col) hidden[col] = h2f(down_input[size_t(lr) * FF + col]);
                        const Q8 input = quant(hidden.data(), FF);
                        for (int col = 0; col < E; ++col) {
                            reference.push_back(dot_q8_double(D[e].w.data() + size_t(col) * FF, input, FF));
                            positions.push_back(size_t(lr) * E + col);
                        }
                    }
                bits_rms("set y", type, y, reference, positions, E);
            } else bits("set y", type, y.data(), y.size() * 4);
            real_shape_bits(type);
            w2 = 0;
            for (int e = 0; e < 3; ++e)
                for (int lr = bounds[e]; lr < bounds[e + 1]; ++lr) {
                    std::vector<float> hf(FF);
                    for (int c = 0; c < FF; ++c) hf[c] = h2f(down_input[(size_t) lr * FF + c]);
                    for (int r = 0; r < E; ++r) {
                        double sa; const double ref = dotd(&D[e].w[(size_t) r * FF], hf.data(), FF, &sa);
                        w2 = std::max(w2, std::fabs(y[(size_t) lr * E + r] - ref) / (sa + 1e-30));
                    }
                }
            // (the prompt sets' tiles take F16 operands for the i-quants: the tile's own rounding, not the token path's.
            //  Q2_0's codes are not symmetric about 0 (value = code - 1, so a block's |w| runs to 2 d): against the
            //  unquantized reference this fixture's worst row measured 6.75e-3, of which |y - q8ref| / sa was 3.9e-8 -
            //  the activation's per-32 int8 rounding, as with IQ1_S above, not a decode error)
            report("set down", type, w2, type == 42 ? 1e-2 : 3e-3);

            // ---- optional matrix-engine tiles forced on, against the scalar tiles forced on the same inputs: each within the
            // tolerances above of the CPU reference, the two within the same tolerance of each other, and the forced-on run must really
            // have launched them (STRATA_DS4_HIPK is not needed)
            if (iq::hipk_force(1)) {
                const long long n0 = iq::hipk_launches();
                iq::SetArgs qm = q;
                uint16_t* mh; device_alloc(mh, (size_t) NR * FF * 2); float* my; device_alloc(my, (size_t) NR * E * 4);
                qm.h = mh; qm.ldx = E; qm.ldo = FF; qm.xh = nullptr; qm.y = nullptr;
                if (!iq::set_gate_up(qm, queue())) { std::printf("matrix-core set_gate_up refused\n"); return 1; }
                qm.xh = mh; qm.ldx = FF; qm.y = my; qm.ldo = E;
                if (!iq::set_down(qm, queue())) { std::printf("matrix-core set_down refused\n"); return 1; }
                queue().wait_and_throw();
                const long long taken = iq::hipk_launches() - n0;
                iq::hipk_force(0);
                const long long n1 = iq::hipk_launches();
                uint16_t* sh; device_alloc(sh, (size_t) NR * FF * 2); float* sy; device_alloc(sy, (size_t) NR * E * 4);
                iq::SetArgs qs = q;
                qs.h = sh; qs.ldx = E; qs.ldo = FF; qs.xh = nullptr; qs.y = nullptr;
                if (!iq::set_gate_up(qs, queue())) { std::printf("scalar set_gate_up refused\n"); return 1; }
                qs.xh = sh; qs.ldx = FF; qs.y = sy; qs.ldo = E;
                if (!iq::set_down(qs, queue())) { std::printf("scalar set_down refused\n"); return 1; }
                queue().wait_and_throw();
                const long long scal = iq::hipk_launches() - n1;
                iq::hipk_force(-1);
                const auto hm = down_(mh, (size_t) NR * FF), hs = down_(sh, (size_t) NR * FF);
                const auto ym = down_(my, (size_t) NR * E), ys = down_(sy, (size_t) NR * E);
                double wg = 0, wd = 0, wx = 0, wy = 0;
                for (int e = 0; e < 3; ++e)
                    for (int lr = bounds[e]; lr < bounds[e + 1]; ++lr) {
                        std::vector<float> hf(FF);
                        for (int c = 0; c < FF; ++c) hf[c] = h2f(hm[(size_t) lr * FF + c]);
                        for (int r = 0; r < FF; ++r) {
                            double sc;
                            const double ref = refh(e, r, &x[(size_t) idx[lr] * E], sc);
                            wg = std::max(wg, std::fabs((double) h2f(hm[(size_t) lr * FF + r]) - ref) / sc);
                            wx = std::max(wx, std::fabs((double) h2f(hm[(size_t) lr * FF + r]) - (double) h2f(hs[(size_t) lr * FF + r])) / sc);
                        }
                        for (int r = 0; r < E; ++r) {
                            double sa; const double ref = dotd(&D[e].w[(size_t) r * FF], hf.data(), FF, &sa);
                            wd = std::max(wd, std::fabs(ym[(size_t) lr * E + r] - ref) / (sa + 1e-30));
                            wy = std::max(wy, std::fabs((double) ym[(size_t) lr * E + r] - (double) ys[(size_t) lr * E + r]) / (sa + 1e-30));
                        }
                    }
                std::printf("%-7s matrix-core tiles taken %lld (scalar run: %lld)\n", iq::name(type), taken, scal);
                if (taken < 2 || scal != 0) { std::printf("%-7s FAIL: the matrix-core tiles did not run as forced\n", iq::name(type)); ++fails; }
                report("wmma gate_up vs CPU", type, wg, type == 19 ? 1.5e-2 : 1e-2);
                report("wmma down vs CPU", type, wd, 3e-3);
                report("wmma vs scalar h", type, wx, type == 19 ? 1.5e-2 : 1e-2);
                report("wmma vs scalar y", type, wy, 3e-3);
            }
        }

        // ---- the CPU lane on the same experts and inputs (ggml-cpu's rows, or Strata's AVX2 ones with DS4_GG_IQ256=1)
        {
            gg::IqCpu fm; std::string err;
            if (!gg::iq_cpu_fmt(type, type, E, FF, fm, err)) { std::printf("%-7s cpu lane: %s\n", iq::name(type), err.c_str()); ++fails; }
            else {
                std::vector<uint8_t> act(fm.act_bytes), hact(fm.h_bytes);
                gg::iq_cpu_quant(fm.gu_act, x.data(), E, act.data());
                // (the CPU lane rounds the activation to its own type - Q8_K, one F32 scale per 256 values - so the
                //  reference is that quantized activation, not the device's per-32 one)
                const std::vector<float> xhat = act_values(fm.gu_act, act.data(), E);
                double wg = 0, wd = 0;
                for (int e = 0; e < NEXP; ++e) {
                    std::vector<float> hc(FF), yc(E);
                    float* hp[1] = {hc.data()}; float* yp[1] = {yc.data()};
                    const void* ap[1] = {act.data()};
                    gg::iq_cpu_gate_up(fm, &blobs[e * blob + off[0]], &blobs[e * blob + off[1]], ap, 1, hp, 0, FF, lim);
                    for (int r = 0; r < FF; ++r) {
                        double sg, su;
                        const double g = dotd(&G[e].w[(size_t) r * E], xhat.data(), E, &sg);
                        const double u = dotd(&U[e].w[(size_t) r * E], xhat.data(), E, &su);
                        wg = std::max(wg, std::fabs(hc[r] - swiglu(g, u, lim)) / (sg + su + 1e-30));
                    }
                    gg::iq_cpu_quant(fm.d_act, &href[(size_t) e * FF], FF, hact.data());
                    const std::vector<float> hhat = act_values(fm.d_act, hact.data(), FF);
                    const void* hpp[1] = {hact.data()};
                    gg::iq_cpu_down(fm, &blobs[e * blob + off[2]], hpp, 1, yp, 0, E);
                    for (int r = 0; r < E; ++r) { double sa; const double ref = dotd(&D[e].w[(size_t) r * FF], hhat.data(), FF, &sa); wd = std::max(wd, std::fabs(yc[r] - ref) / (sa + 1e-30)); }
                }
                report("cpu gate_up vs ref", type, wg, 1e-5);
                report("cpu down vs ref", type, wd, 1e-5);
                std::printf("%-7s cpu rows: %s\n", iq::name(type), gg::iq_cpu_own_rows(type) ? "Strata AVX2" : "ggml-cpu");
            }
        }
        device_free(dblob);
    }
    if (g_cmp && devs.size() > 1) { std::fclose(g_cmp); g_cmp = nullptr; std::printf("device %d vs device %d: %ld buffers, %ld differ\n", devs[di], devs[0], g_cmp_n, g_cmp_bad); fails += g_cmp_bad != 0; }
    }
    if (g_cmp) { std::printf("golden comparison: %ld buffers (%ld FP64/RMS), %ld failed -> %s\n", g_cmp_n, g_cmp_rms_n, g_cmp_bad, g_cmp_bad ? "FAIL" : "PASS"); fails += g_cmp_bad != 0; }
    if (g_dump) std::fclose(g_dump);
    std::printf("%s\n", fails ? "FAILED" : "ALL OK");
    return fails;
} catch (const std::exception& e) {
    std::fprintf(stderr, "iq_expert_test: %s\n", e.what());
    return 1;
} catch (...) {
    std::fputs("iq_expert_test: unknown failure\n", stderr);
    return 1;
}
