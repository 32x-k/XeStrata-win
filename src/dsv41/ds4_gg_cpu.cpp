// Strata. The CPU lane of the DeepSeek-V4.1 engine path for ggml's block types (IQ2_XXS .. Q4_K, the experts of a requantized model):
// ggml-cpu's own row dots (type traits: vec_dot over the Q8_K / Q8_0 activation of vec_dot_type), one row at a time.  Maya's lane for
// these types (native_expert split-base rows) is not in Strata's tree; this one is plain and complete, not tuned.
#include "ds4_gguf_cpu.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#if defined(STRATA_DSV41_TQ_CPU)
#include "ggml-cpu.h"
#include "ggml.h"
#include "strata/kernels/cpu/iq_avx2.hpp"

#include <mutex>
#include <vector>

namespace strata::ds4::gguf {
namespace {
void init_once() {
    static std::once_flag f;
    std::call_once(f, [] { ggml_cpu_init(); });
}
size_t row_size(int type, int64_t k) { return ggml_row_size((ggml_type) type, k); }
}  // namespace

bool gg_cpu_type(int type) {
    // every type ggml-cpu has a row dot for (ds4_gguf.cu's known types); the i-quants with a device kernel are usually
    // placed in VRAM, but a pack may hand any of them to this lane
    switch (type) {
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
    if (a == nullptr || b == nullptr || a->vec_dot == nullptr || b->vec_dot == nullptr) {
        err = "ggml-cpu has no row dot for these types";
        return false;
    }
    f.tg = tg;
    f.td = td;
    f.n_embd = n_embd;
    f.n_ff = n_ff;
    f.gu_act = (int) a->vec_dot_type;
    f.d_act = (int) b->vec_dot_type;
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

// Strata's AVX2 multi-token rows for the i-quants (iq256x_rows: each 32-value chunk decoded once, every token applies it).  They
// are ggml-cpu's vec_dot bit for bit (the integer sums are ggml's and the float additions come in ggml's order: ds4_gg_bench --exact,
// and own_ok() below checks it once per type on this CPU against the ggml-cpu build in use), so taking them or not never changes an
// answer, at any number of tokens - the choice is speed only.  Measured on the rows alone: no faster than ggml's dot at one token
// (both wait for the codebook lookups), x1.7-2.9 at 2-4 tokens, x1.4 on IQ3_S already at one; so by default ggml's dot below
// min_tokens(type) and these rows from there.  DS4_GG_IQ256=0 never takes them, =1 always (the old opt-in).
static int g_iq256 = -1;   // -1: from the environment and the CPU, 0 off, 1 always, 2 from min_tokens(type)
void gg_cpu_set_iq256(int mode) { g_iq256 = mode; }
static int min_tokens(int type) { return type == 21 ? 1 : type == 29 ? 3 : 2; }
static bool own_ok(int type) {
    static std::once_flag once[32];
    static bool ok[32];
    const int k = type & 31;
    std::call_once(once[k], [&] {
        ok[k] = false;
        const ggml_type_traits_cpu* tc = ggml_get_type_traits_cpu((ggml_type) type);
        const ggml_type_traits_cpu* q8 = tc ? ggml_get_type_traits_cpu(tc->vec_dot_type) : nullptr;
        if (tc == nullptr || tc->vec_dot == nullptr || q8 == nullptr || q8->from_float == nullptr) return;
        const int n = 1024, rows = 7;
        const size_t rb = ggml_row_size((ggml_type) type, n), bb = ggml_type_size((ggml_type) type), ab = ggml_row_size(tc->vec_dot_type, n);
        uint64_t st = 0x9E3779B97F4A7C15ull + (uint64_t) type;
        auto next = [&st] { st ^= st << 13; st ^= st >> 7; st ^= st << 17; return st; };
        std::vector<uint8_t> w(rb * rows);
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
        for (int nt : {1, 2, 3, 4, 5, 6, 8}) {
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
            strata::kernels::cpu::iq256x_rows(type, w.data(), rb, n, ap.data(), nt, p1.data(), 0, rows);
            for (int t = 0; t < nt; ++t)
                if (std::memcmp(o0[(size_t) t].data(), o1[(size_t) t].data(), rows * sizeof(float)) != 0) {
                    if (dbg() || std::getenv("DS4_GG_VERBOSE")) std::fprintf(stderr, "ds4: CPU lane type %d: AVX2 rows differ from ggml-cpu's dot at %d tokens, not used\n", type, nt);
                    return;
                }
        }
        ok[k] = true;
    });
    return ok[k];
}
static bool use_iq256(int type, int nt) {
    if (g_iq256 < 0) {
        const char* v = std::getenv("DS4_GG_IQ256");
        bool avx2 = false;
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
        __builtin_cpu_init();
        avx2 = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");
#endif
        g_iq256 = !avx2 ? 0 : v == nullptr ? 2 : std::atoi(v) != 0 ? 1 : 0;
    }
    if (g_iq256 == 0 || !strata::kernels::cpu::iq256x_supported(type)) return false;
    if (g_iq256 == 2 && nt < min_tokens(type)) return false;
    return own_ok(type);
}

static void dots(int type, const uint8_t* w, size_t rb, int n, const void* const* act, int nt, float* const* out, int r0, int r1) {
    if (nt > 0 && use_iq256(type, nt)) {
        strata::kernels::cpu::iq256x_rows(type, w, rb, n, act, nt, out, r0, r1);
        return;
    }
    const ggml_type_traits_cpu* tc = ggml_get_type_traits_cpu((ggml_type) type);
    if (dbg()) std::fprintf(stderr, "gg dots type %d n %d rows %d..%d nt %d\n", type, n, r0, r1, nt);
    for (int r = r0; r < r1; ++r)
        for (int t = 0; t < nt; ++t) tc->vec_dot(n, &out[t][r], 0, w + (size_t) r * rb, 0, act[t], 0, 1);
}

void gg_cpu_gate_up(const IqCpu& f, const uint8_t* gate, const uint8_t* up, const void* const* act, int nt, float* const* ff, int r0,
                    int r1, float limit) {
    static thread_local std::vector<float> tmp[8];
    float* uo[8];
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
