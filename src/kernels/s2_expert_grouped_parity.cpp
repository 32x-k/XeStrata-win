// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/s2_expert_grouped_parity.cpp - the S2 expert kernels' rewrite (upstream e155b07b) against the previous
// kernels, byte for byte (strata/kernels/s2_expert_grouped.hpp, moe_grouped_select_old).  On synthetic S2 blobs (random
// codes, finite FP16 scales) and Q8_0 activations, for windows of 1, 2, 4 and 8 tokens:
//   1. moe_grouped_s2 (the verify window's groups), with FP32 activation scales;
//   2. moe_hit_grouped_s2_multi (a window's hits), the same;
//   3. moe_hit_grouped_s2 with the FP16 scales in the blocks;
//   4. an activation row 2 bytes off a 4-byte boundary takes the previous kernels (moe_grouped_last_path).
// Each call's path is checked, and every output must match the previous kernels' bit for bit.  Needs a GPU, no model.

#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"

#include "parity_device.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace k = strata::kernels;
namespace p = strata::parity;

namespace {

constexpr int H = 2560, FF = 640, K = 10, NE = 64, TMAX = 8;
constexpr size_t BLOB = (size_t) 3 * H * FF / 4 + (size_t) 3 * H * FF / 64 * 2;   // codes, then FP16 scales
constexpr size_t O_SCALES = (size_t) 3 * H * FF / 4;

int g_fail = 0;
uint32_t g_rng = 7;
uint32_t rnd() { return g_rng = g_rng * 1664525u + 1013904223u; }

// `run(out)` with the previous kernels, then the new ones; both paths checked, the outputs compared byte for byte
template <class Run>
void both(const char* what, int t, float* out_old, float* out_new, size_t n, int new_path, Run run) {
    for (int path = 0; path < 2; ++path) {
        float* o = path == 0 ? out_old : out_new;
        std::vector<float> zero(n, 0.0f);
        p::copy_bytes_in(o, zero.data(), n * 4);
        k::moe_grouped_select_old(path == 0 ? 1 : 0);
        run(o);
        p::sync();
        const int taken = k::moe_grouped_last_path();
        const int want = path == 0 ? 0 : new_path;
        if (taken != want) {
            std::printf("FAIL %s T=%d: path %d taken, %d expected\n", what, t, taken, want);
            ++g_fail;
        }
    }
    std::vector<float> a(n), b(n);
    p::copy_bytes_out(a.data(), out_old, n * 4);
    p::copy_bytes_out(b.data(), out_new, n * 4);
    size_t diff = 0;
    for (size_t i = 0; i < n; ++i) diff += std::memcmp(&a[i], &b[i], 4) != 0;
    if (diff != 0) {
        std::printf("FAIL %s T=%d: %zu of %zu outputs differ\n", what, t, diff, n);
        ++g_fail;
    }
}

}  // namespace

int main(int argc, char** argv) {
    (void) argc;
    (void) argv;   // --selftest: the checks above are the whole program
    void* s = p::stream();
    std::vector<uint8_t> hb(BLOB * NE);
    for (auto& b : hb) b = (uint8_t) (rnd() >> 24);
    for (int e = 0; e < NE; ++e)
        for (size_t o = O_SCALES; o < BLOB; o += 2) {
            const uint16_t h = (uint16_t) (0x2000 + (rnd() >> 22));   // 2^-7 and a little above
            hb[(size_t) e * BLOB + o] = (uint8_t) h;
            hb[(size_t) e * BLOB + o + 1] = (uint8_t) (h >> 8);
        }
    auto* base = (uint8_t*) p::alloc_bytes(hb.size());
    p::copy_bytes_in(base, hb.data(), hb.size());
    std::vector<float> hx((size_t) TMAX * H);
    for (auto& v : hx) v = (float) ((int) (rnd() >> 20) - 2048) / 1024.0f;
    auto* x = (float*) p::alloc_bytes(hx.size() * 4);
    p::copy_bytes_in(x, hx.data(), hx.size() * 4);
    // the activations twice: at an aligned row and 2 bytes past one (the new kernels' alignment fallback)
    const size_t xq_bytes = (size_t) TMAX * (H / 32) * 34;
    auto* xq_mem = (uint8_t*) p::alloc_bytes(xq_bytes + 64);
    uint8_t* xq = xq_mem;
    uint8_t* xq_odd = xq_mem + 2;
    auto* xs = (float*) p::alloc_bytes((size_t) TMAX * (H / 32) * 4);
    k::quantize_q8_0_scaled(x, xq, xs, (int64_t) TMAX * H, s);
    p::sync();
    void* scratch = p::alloc_bytes((size_t) 64 << 20);
    const size_t outn = (size_t) TMAX * K * H;
    auto* out_old = (float*) p::alloc_bytes(outn * 4);
    auto* out_new = (float*) p::alloc_bytes(outn * 4);
    auto* ids = (int32_t*) p::alloc_bytes(128 * 4);
    auto* gstart = (int32_t*) p::alloc_bytes(130 * 4);
    auto* counts = (int32_t*) p::alloc_bytes(8 * 4);
    auto* edst = (int32_t*) p::alloc_bytes(128 * 4);
    auto* etok = (int32_t*) p::alloc_bytes(128 * 4);
    auto* gp = (unsigned long long*) p::alloc_bytes(128 * 8);
    auto* dst = (int32_t*) p::alloc_bytes(128 * 4);
    auto* cnt = (int32_t*) p::alloc_bytes(4);
    for (int t : {1, 2, 4, 8}) {
        std::vector<int32_t> hid((size_t) t * K), hd((size_t) t * K);
        for (int i = 0; i < t * K; ++i) {
            hid[(size_t) i] = (int) ((i % K) * 5 + (rnd() >> 29) * (i / K)) % NE;   // shared between tokens
            hd[(size_t) i] = i;
        }
        p::copy_bytes_in(ids, hid.data(), hid.size() * 4);
        p::copy_bytes_in(dst, hd.data(), hd.size() * 4);
        const int32_t n = t * K;
        p::copy_bytes_in(cnt, &n, 4);
        k::moe_group_resident(ids, t * K, K, base, (int64_t) BLOB, gp, gstart, counts, edst, etok, s);
        p::sync();
        const size_t m = (size_t) t * K * H;
        both("moe_grouped_s2", t, out_old, out_new, m, 1, [&](float* o) {
            k::moe_grouped_s2(gp, gstart, counts, edst, etok, t * K, t * K, xq, xs, scratch, o, s);
        });
        both("moe_hit_grouped_s2_multi", t, out_old, out_new, m, 1, [&](float* o) {
            k::moe_hit_grouped_s2_multi(base, ids, dst, cnt, t * K, (int64_t) BLOB, xq, xs, K, scratch, o, s);
        });
        both("moe_grouped_s2 (2 bytes off)", t, out_old, out_new, m, 0, [&](float* o) {
            k::moe_grouped_s2(gp, gstart, counts, edst, etok, t * K, t * K, xq_odd, nullptr, scratch, o, s);
        });
    }
    both("moe_hit_grouped_s2 (FP16 scales)", 1, out_old, out_new, (size_t) K * H, 1, [&](float* o) {
        k::moe_hit_grouped_s2(base, ids, dst, K, (int64_t) BLOB, xq, scratch, o, s, nullptr);
    });
    k::moe_grouped_select_old(-1);
    if (g_fail != 0) {
        std::printf("s2_expert_grouped_parity: %d check(s) FAILED\n", g_fail);
        return 1;
    }
    std::printf("s2_expert_grouped_parity OK\n");
    return 0;
}
