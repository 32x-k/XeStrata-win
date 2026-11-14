// SPDX-FileCopyrightText: 2026 Peasant Smith and the Project Maya contributors
// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// Strata. The CPU lane of the DeepSeek-V4.1 engine path for ggml's block types (IQ2_XXS .. Q4_K, the experts of a requantized model):
// ggml-cpu's own row dots (type traits: vec_dot over the Q8_K / Q8_0 activation of vec_dot_type), one row at a time.  Maya's lane for
// these types (native_expert split-base rows) is not in Strata's tree; this one is plain and complete, not tuned.
#include "ds4_gguf_cpu.hpp"

#include <cmath>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#if defined(STRATA_DSV41_TQ_CPU)
#include "ggml-cpu.h"
#include "ggml.h"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/iq_avx512.hpp"

#include <algorithm>
#include <chrono>
#include <mutex>
#include <vector>

namespace strata::ds4::gguf {
namespace {
void init_once() {
    static std::once_flag f;
    std::call_once(f, [] { ggml_cpu_init(); });
}
size_t row_size(int type, int64_t k) {
    if (type == 42) return (size_t) (k / 64) * 18;   // GSQ-RCO's Q2_0: not a ggml type (block below)
    return ggml_row_size((ggml_type) type, k);
}
float f16_at(const uint8_t* p) {
    ggml_fp16_t u;
    std::memcpy(&u, p, sizeof u);
    return ggml_fp16_to_fp32(u);
}
// A Q8_K (type 15) row's dot against a Q8_0 activation ({fp16 d; int8 codes} per 32 values): {fp32 d; int8
// qs[256]; u16 bsums[16]}, the sums unused - the activation keeps no min to correct.  An exact integer sum per 32
// values times the two scales, the device lane's decoding (ds4_iq_decode.hpp's kQ8_K) value for value.
float q8k_row_dot(const uint8_t* w, const uint8_t* a, int n) {
    float sum = 0.0f;
    const int nb = n / 256;
    for (int kb = 0; kb < nb; ++kb) {
        const uint8_t* b = w + (size_t) kb * 292;
        float d;
        std::memcpy(&d, b, 4);
        for (int sb = 0; sb < 8; ++sb) {
            const uint8_t* ab = a + (size_t) (8 * kb + sb) * 34;
            const int8_t* xq = (const int8_t*) (ab + 2);
            const int8_t* qs = (const int8_t*) (b + 4 + 32 * sb);
            int acc = 0;
            for (int i = 0; i < 32; ++i) acc += (int) qs[i] * (int) xq[i];
            sum += f16_at(ab) * (d * (float) acc);
        }
    }
    return sum;
}
// A Q2_0 (type 42, GSQ-RCO's routed experts) row's dot against a Q8_0 activation ({fp16 d; int8 codes} per 32
// values): value = d (code - 1), the 64-value block's u16 word j holding values 8j..8j+3 in its low byte and
// 8j+4..8j+7 in its high one, the lowest 2 bits first (src/kernels' Fmt<42>).  An exact integer sum per 32 values
// times the two fp16 scales - the device lane's decoding (ds4_iq_decode.hpp's kQ2_0) value for value.
float q20_row_dot(const uint8_t* w, const uint8_t* a, int n) {
    float sum = 0.0f;
    const int nb = n / 64;
    for (int kb = 0; kb < nb; ++kb) {
        const uint8_t* b = w + (size_t) kb * 18;
        const float d = f16_at(b);
        for (int sb = 0; sb < 2; ++sb) {   // the block's two 32-value sub-blocks, u16 words 4 sb .. 4 sb + 3
            const uint8_t* ab = a + (size_t) (2 * kb + sb) * 34;
            const int8_t* xq = (const int8_t*) (ab + 2);
            int acc = 0;
            for (int j = 0; j < 4; ++j) {
                const uint32_t q = (uint32_t) b[2 + 2 * (4 * sb + j)] | (uint32_t) b[3 + 2 * (4 * sb + j)] << 8;
                for (int m = 0; m < 4; ++m)
                    acc += ((int) ((q >> (2 * m)) & 3u) - 1) * (int) xq[8 * j + m] +
                           ((int) ((q >> (8 + 2 * m)) & 3u) - 1) * (int) xq[8 * j + 4 + m];
            }
            sum += f16_at(ab) * (d * (float) acc);
        }
    }
    return sum;
}
}  // namespace

bool gg_cpu_type(int type) {
    // every type ggml-cpu has a row dot for (ds4_gguf.cu's known types), plus Q2_0 (42, GSQ-RCO: the dot above) and
    // the plain Q8_0 / Q8_K rows; the i-quants with a device kernel are usually
    // placed in VRAM, but a pack may hand any of them to this lane
    switch (type) {
        case 8: case 15: case 42:
        case 10: case 11: case 12: case 13: case 14: case 16: case 17: case 18: case 19: case 20: case 21: case 22:
        case 23: case 29: return true;
        default: return false;
    }
}

static bool dbg() { static const bool on = std::getenv("DS4_DBG_GG") != nullptr; return on; }
bool gg_cpu_fmt(int tg, int td, int n_embd, int n_ff, IqCpu& f, std::string& err) {
    if (dbg()) std::fprintf(stderr, "gg_cpu_fmt %d %d\n", tg, td);
    init_once();
    if (!gg_cpu_type(tg) || !gg_cpu_type(td)) {
        err = "not ggml block types";
        return false;
    }
    const ggml_type_traits_cpu* a = ggml_get_type_traits_cpu((ggml_type) tg);
    const ggml_type_traits_cpu* b = ggml_get_type_traits_cpu((ggml_type) td);
    // Q2_0 (42) and Q8_K (15: an intermediate type - no CPU row dot in ggml-cpu) take the lane's own dots above,
    // over a Q8_0 activation (type 8's from_float)
    const auto own = [](int t) { return t == 42 || t == 15; };
    if ((!own(tg) && (a == nullptr || a->vec_dot == nullptr)) ||
        (!own(td) && (b == nullptr || b->vec_dot == nullptr))) {
        err = "ggml-cpu has no row dot for these types";
        return false;
    }
    f.tg = tg;
    f.td = td;
    f.n_embd = n_embd;
    f.n_ff = n_ff;
    f.gu_act = own(tg) ? 8 : (int) a->vec_dot_type;
    f.d_act = own(td) ? 8 : (int) b->vec_dot_type;
    f.gu_row = row_size(tg, n_embd);
    f.d_row = row_size(td, n_ff);
    f.act_bytes = row_size(f.gu_act, n_embd);
    f.h_bytes = row_size(f.d_act, n_ff);
    return true;
}

// x (k values) -> the activation type (Q8_K, Q8_0 ..) ggml-cpu's dot takes
void gg_cpu_quant(int act, const float* x, int k, void* dst) {
    init_once();
    if (dbg()) std::fprintf(stderr, "gg_cpu_quant act %d k %d\n", act, k);
    const ggml_type_traits_cpu* tc = ggml_get_type_traits_cpu((ggml_type) act);
    if (tc != nullptr && tc->from_float != nullptr) {
        tc->from_float(x, dst, k);
        return;
    }
    const ggml_type_traits* t = ggml_get_type_traits((ggml_type) act);
    if (dbg()) std::fprintf(stderr, "gg_cpu_quant: no cpu from_float, ref %p\n", (void*) (t ? t->from_float_ref : nullptr));
    t->from_float_ref(x, dst, k);
}

// Strata's multi-token i-quant rows decode each weight chunk once for every token.
// own_ok checks their arithmetic against this build's ggml-cpu before they can be selected.
// DS4_GG_IQ256=0 disables these rows; other settings still require CPU instruction support.
static std::atomic<int> g_iq256{-1};   // -1: environment, 0: disabled, positive: enabled
void gg_cpu_set_iq256(int mode) { g_iq256.store(mode, std::memory_order_relaxed); }
// `rows` rows of n random values of a type (the scales small and finite), and the xorshift state the caller goes on with
static std::vector<uint8_t> test_rows(int type, int n, int rows, uint64_t& st) {
    auto next = [&st] { st ^= st << 13; st ^= st >> 7; st ^= st << 17; return st; };
    const size_t rb = ggml_row_size((ggml_type) type, n), bb = ggml_type_size((ggml_type) type);
    std::vector<uint8_t> w(rb * (size_t) rows);
    for (auto& b : w) b = (uint8_t) next();
    for (size_t o = 0; o < w.size(); o += bb) {
        const ggml_fp16_t d = ggml_fp32_to_fp16(0.002f + 0.001f * (float) (next() % 16));
        if (type == 29) {   // IQ1_M keeps its f16 in the top nibbles of the four scale words
            uint16_t sc[4];
            std::memcpy(sc, &w[o + 48], 8);
            for (int j = 0; j < 4; ++j) sc[j] = (uint16_t) ((sc[j] & 0x0fff) | ((d >> (4 * j)) & 0xf) << 12);
            std::memcpy(&w[o + 48], sc, 8);
        } else {
            std::memcpy(&w[o], &d, sizeof d);
        }
    }
    return w;
}
// impl 1: the AVX-512 rows (iq512x_rows), 0: the AVX2 rows
static bool own_ok(int type, int impl) {
    static std::once_flag once[2][32];
    static bool ok[2][32];
    const int k = type & 31;
    std::call_once(once[impl][k], [&] {
        ok[impl][k] = false;
        const ggml_type_traits_cpu* tc = ggml_get_type_traits_cpu((ggml_type) type);
        const ggml_type_traits_cpu* q8 = tc ? ggml_get_type_traits_cpu(tc->vec_dot_type) : nullptr;
        if (tc == nullptr || tc->vec_dot == nullptr || q8 == nullptr || q8->from_float == nullptr) return;
        const int n = 1024, rows = 7;
        const size_t rb = ggml_row_size((ggml_type) type, n), ab = ggml_row_size(tc->vec_dot_type, n);
        uint64_t st = 0x9E3779B97F4A7C15ull + (uint64_t) type;
        auto next = [&st] { st ^= st << 13; st ^= st >> 7; st ^= st << 17; return st; };
        const std::vector<uint8_t> w = test_rows(type, n, rows, st);
        for (int nt : {1, 2, 3, 4, 5, 6, 8, 16, 33}) {
            std::vector<std::vector<uint8_t>> a((size_t) nt, std::vector<uint8_t>(ab));
            std::vector<const void*> ap((size_t) nt);
            std::vector<std::vector<float>> o0((size_t) nt, std::vector<float>(rows)), o1 = o0;
            std::vector<float*> p1((size_t) nt);
            for (int t = 0; t < nt; ++t) {
                std::vector<float> x((size_t) n);
                for (auto& v : x) v = ((float) (next() % 2001) - 1000.0f) * 1e-3f;
                q8->from_float(x.data(), a[(size_t) t].data(), n);
                ap[(size_t) t] = a[(size_t) t].data();
                p1[(size_t) t] = o1[(size_t) t].data();
                for (int r = 0; r < rows; ++r) tc->vec_dot(n, &o0[(size_t) t][(size_t) r], 0, w.data() + (size_t) r * rb, 0, ap[(size_t) t], 0, 1);
            }
            if (impl == 1) strata::kernels::cpu::iq512x_rows(type, w.data(), rb, n, ap.data(), nt, p1.data(), 0, rows);
            else strata::kernels::cpu::iq256_rows(type, w.data(), rb, n, ap.data(), nt, p1.data(), 0, rows);
            for (int t = 0; t < nt; ++t)
                if (std::memcmp(o0[(size_t) t].data(), o1[(size_t) t].data(), rows * sizeof(float)) != 0) {
                    if (dbg() || std::getenv("DS4_GG_VERBOSE"))
                        std::fprintf(stderr, "ds4: CPU lane type %d: %s rows differ from ggml-cpu's dot at %d tokens, not used\n",
                                     type, impl == 1 ? "AVX-512" : "AVX2", nt);
                    return;
                }
        }
        ok[impl][k] = true;
    });
    return ok[impl][k];
}
// One token: whether the AVX-512 rows beat ggml-cpu's dot on this CPU, timed once per type on the first call - the same
// cache-resident rows by both, interleaved rounds, the best round of each counting, and the rows taken only 3% faster or more.
// (Upstream Strata chose them by the CPU's maker and family, from its measurements on Zen 4 and Zen 5: one token IQ2_XXS 35 ->
// 44 GB/s, IQ2_XS 37 -> 48, IQ3_XXS 43 -> 45 on Zen 5.)
static bool faster_512(int type) {
    static std::once_flag once[32];
    static bool fast[32];
    const int k = type & 31;
    std::call_once(once[k], [&] {
        fast[k] = false;
        const ggml_type_traits_cpu* tc = ggml_get_type_traits_cpu((ggml_type) type);
        const ggml_type_traits_cpu* q8 = tc ? ggml_get_type_traits_cpu(tc->vec_dot_type) : nullptr;
        if (tc == nullptr || tc->vec_dot == nullptr || q8 == nullptr || q8->from_float == nullptr) return;
        const int n = 4096, rows = 128, passes = 8, rounds = 7;   // (about 140 KB of IQ2_XXS rows: the core's own caches)
        const size_t rb = ggml_row_size((ggml_type) type, n);
        uint64_t st = 0xD1B54A32D192ED03ull + (uint64_t) type;
        const std::vector<uint8_t> w = test_rows(type, n, rows, st);
        std::vector<float> x((size_t) n);
        for (size_t i = 0; i < x.size(); ++i) x[i] = 0.02f * std::sin(0.37f * (float) i);
        std::vector<uint8_t> a(ggml_row_size(tc->vec_dot_type, n));
        q8->from_float(x.data(), a.data(), n);
        const void* ap[1] = {a.data()};
        std::vector<float> o((size_t) rows);
        float* op[1] = {o.data()};
        double t_ggml = 1e30, t_rows = 1e30;
        for (int round = 0; round < rounds; ++round) {
            auto t0 = std::chrono::steady_clock::now();
            for (int p = 0; p < passes; ++p)
                for (int r = 0; r < rows; ++r) tc->vec_dot(n, &o[(size_t) r], 0, w.data() + (size_t) r * rb, 0, a.data(), 0, 1);
            t_ggml = std::min(t_ggml, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
            t0 = std::chrono::steady_clock::now();
            for (int p = 0; p < passes; ++p) strata::kernels::cpu::iq512x_rows(type, w.data(), rb, n, ap, 1, op, 0, rows);
            t_rows = std::min(t_rows, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        }
        fast[k] = t_rows * 1.03 <= t_ggml;
        if (dbg() || std::getenv("DS4_GG_VERBOSE"))
            std::fprintf(stderr, "ds4: CPU lane type %d, one token: ggml-cpu's dot %.1f us, the AVX-512 rows %.1f us: %s\n", type,
                         t_ggml * 1e6, t_rows * 1e6, fast[k] ? "the rows" : "ggml-cpu's dot");
    });
    return fast[k];
}
// The AVX-512 rows where the CPU has them (AVX-512 F/BW/VL/DQ/VNNI/VBMI and VPOPCNTDQ): from two tokens wherever they pass
// own_ok() (upstream measured x2.4-4.8 at 2-16 tokens on Zen 4 and Zen 5), at one token where faster_512() finds them faster.
// DS4_GG_IQ512: 1 takes them at every token count, 0 never.  The environment is read once: dots() asks for every 64 rows, and
// Windows' getenv takes the CRT's environment lock, which the lane's threads then wait on together.
static bool want_512(int type, int nt) {
    static const int mode = [] {
        const char* v = std::getenv("DS4_GG_IQ512");
        if (!strata::kernels::cpu::cpu_avx512_ok()) return 0;
        return v == nullptr ? 2 : std::strtol(v, nullptr, 10) != 0 ? 1 : 0;
    }();
    if (mode == 0 || !strata::kernels::cpu::iq512x_supported(type)) return false;
    return mode == 1 || nt >= 2 || faster_512(type);
}
// 0: ggml-cpu's dot, 1: the AVX2 rows, 2: the AVX-512 rows
static int use_iq256(int type, int nt) {
    if (!strata::kernels::cpu::cpu_avx2_ok()) return 0;
    static const int from_env = [] {
        const char* v = std::getenv("DS4_GG_IQ256");
        return v == nullptr || std::strtol(v, nullptr, 10) != 0 ? 1 : 0;
    }();
    const int mode = g_iq256.load(std::memory_order_relaxed);
    if ((mode < 0 ? from_env : mode) == 0) return 0;
    if (want_512(type, nt) && own_ok(type, 1)) return 2;
    if (!strata::kernels::cpu::iq256_supported(type)) return 0;
    return own_ok(type, 0) ? 1 : 0;
}

static void dots(int type, const uint8_t* w, size_t rb, int n, const void* const* act, int nt, float* const* out, int r0, int r1) {
    if (nt > 0) {
        const int impl = use_iq256(type, nt);
        if (impl == 2) {
            strata::kernels::cpu::iq512x_rows(type, w, rb, n, act, nt, out, r0, r1);
            return;
        }
        if (impl == 1) {
            strata::kernels::cpu::iq256_rows(type, w, rb, n, act, nt, out, r0, r1);
            return;
        }
    }
    if (type == 42 || type == 15) {   // the lane's own dots, a Q8_0 activation row per token
        for (int r = r0; r < r1; ++r)
            for (int t = 0; t < nt; ++t)
                out[t][r] = type == 42 ? q20_row_dot(w + (size_t) r * rb, (const uint8_t*) act[t], n)
                                       : q8k_row_dot(w + (size_t) r * rb, (const uint8_t*) act[t], n);
        return;
    }
    const ggml_type_traits_cpu* tc = ggml_get_type_traits_cpu((ggml_type) type);
    if (dbg()) std::fprintf(stderr, "gg dots type %d n %d rows %d..%d nt %d\n", type, n, r0, r1, nt);
    for (int r = r0; r < r1; ++r)
        for (int t = 0; t < nt; ++t) tc->vec_dot(n, &out[t][r], 0, w + (size_t) r * rb, 0, act[t], 0, 1);
}

void gg_cpu_gate_up(const IqCpu& f, const uint8_t* gate, const uint8_t* up, const void* const* act, int nt, float* const* ff, int r0,
                    int r1, float limit) {
    constexpr int kMaxTok = 64;   // (the prompt path: a chunk's tokens an expert)
    static thread_local std::vector<float> tmp[kMaxTok];
    float* uo[kMaxTok];
    if (nt > kMaxTok) {   // (more than a pass takes: in pieces)
        for (int t0 = 0; t0 < nt; t0 += kMaxTok)
            gg_cpu_gate_up(f, gate, up, act + t0, nt - t0 < kMaxTok ? nt - t0 : kMaxTok, ff + t0, r0, r1, limit);
        return;
    }
    for (int t = 0; t < nt; ++t) {
        tmp[t].resize((size_t) r1);
        uo[t] = tmp[t].data();
    }
    dots(f.tg, gate, f.gu_row, f.n_embd, act, nt, ff, r0, r1);
    dots(f.tg, up, f.gu_row, f.n_embd, act, nt, uo, r0, r1);
    for (int t = 0; t < nt; ++t)
        for (int r = r0; r < r1; ++r) {
            float g = ff[t][r], u = uo[t][r];
            if (limit > 0.0f) {
                g = g < limit ? g : limit;
                u = u < -limit ? -limit : (u > limit ? limit : u);
            }
            ff[t][r] = g / (1.0f + std::exp(-g)) * u;
        }
}

void gg_cpu_down(const IqCpu& f, const uint8_t* down, const void* const* hq, int nt, float* const* out, int r0, int r1) {
    dots(f.td, down, f.d_row, f.n_ff, hq, nt, out, r0, r1);
}

// a row of any ggml block type -> BF16 on the host (the embedding table's reader; Q5_K / Q4_K / Q6_K / Q8_0 rows of a
// GGUF whose dense tensors are not BF16): ggml's own to_float, then the same round-to-nearest-even to BF16 the Q8_0 path uses
bool gg_row_to_bf16(int type, const uint8_t* row, int k, uint16_t* out) {
    init_once();
    const ggml_type_traits* t = ggml_get_type_traits((ggml_type) type);
    if (t == nullptr || t->to_float == nullptr) return false;
    static thread_local std::vector<float> f;
    f.resize((size_t) k);
    t->to_float(row, f.data(), k);
    for (int i = 0; i < k; ++i) {
        uint32_t u;
        std::memcpy(&u, &f[(size_t) i], 4);
        u += 0x7fffu + ((u >> 16) & 1u);
        out[i] = (uint16_t) (u >> 16);
    }
    return true;
}
}  // namespace strata::ds4::gguf
#else
namespace strata::ds4::gguf {
bool gg_cpu_type(int) { return false; }
bool gg_cpu_fmt(int, int, int, int, IqCpu&, std::string& err) {
    err = "this build has no ggml-cpu lane (STRATA_NATIVE_EXPERTS=OFF)";
    return false;
}
void gg_cpu_quant(int, const float*, int, void*) {}
void gg_cpu_gate_up(const IqCpu&, const uint8_t*, const uint8_t*, const void* const*, int, float* const*, int, int, float) {}
void gg_cpu_down(const IqCpu&, const uint8_t*, const void* const*, int, float* const*, int, int) {}
bool gg_row_to_bf16(int, const uint8_t*, int, uint16_t*) { return false; }
}  // namespace strata::ds4::gguf
#endif
