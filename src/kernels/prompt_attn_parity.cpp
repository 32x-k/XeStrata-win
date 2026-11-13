// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/prompt_attn_parity.cpp - the plain models' prompt attention (`prompt_attn`, prompt_attn.hpp) against an
// FP64 host reference (GPU, synthetic, no model):
//
//   Qwen3.5-9B's 16/4 heads of 256 on FP16 pools behind a shuffled page table, a chunk of queries after cells already
//   in the pools, its length a whole number of neither kernel's query blocks; every query attends causally to the
//   cells up to its own.  The kernel is the device's choice: the WMMA one on RDNA3 and RDNA4, the FMA one elsewhere
//   and under STRATA_PROMPT_ATTN_MMA=0 (CTest runs both ways).  The output agrees with FP64 to 1e-4 of its scale
//   (the FMA kernel's FP32), or 5e-3 where the WMMA kernel rounds Q and the probabilities to FP16 (a wrong head
//   or cell is off by the output's whole scale).
#include "strata/kernels/prompt_attn.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "parity_device.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {
namespace par = strata::parity;

template <typename T> T* up(par::Device& dev, const std::vector<T>& h) {
    T* d = dev.alloc<T>(h.size() + 64 / sizeof(T));
    par::copy_in(d, h.data(), h.size());
    return d;
}

int check(const char* name, int64_t nh, int64_t nkv, int64_t hd, int64_t pos0, int64_t T) {
    par::Device dev;
    k::QsaShapes s = k::qsa_real_shapes();
    s.n_head = nh; s.n_head_kv = nkv; s.head_dim = hd;
    const int64_t PS = s.page_size, cells = pos0 + T, pages = (cells + PS - 1) / PS + 1, rows = pages * nkv * PS;
    const int64_t G = nh / nkv;
    std::mt19937 rng(91 + (unsigned) (nh + hd + T));
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<uint16_t> kh((size_t) (rows * hd)), vh((size_t) (rows * hd));
    for (auto& x : kh) x = k::f16_from_f32(nd(rng) * 1.5f);
    for (auto& x : vh) x = k::f16_from_f32(nd(rng));
    std::vector<int32_t> table((size_t) pages);
    for (int64_t i = 0; i < pages; ++i) table[(size_t) i] = (int32_t) i;
    std::shuffle(table.begin(), table.end(), rng);
    std::vector<float> q((size_t) (T * nh * hd));
    for (auto& x : q) x = nd(rng) * 2.0f;

    // FP64 reference: query t (position pos0 + t), head h reads KV head h / G over cells 0 .. pos0 + t
    std::vector<double> ref(q.size(), 0.0);
    double scale_out = 0.0;
    for (int64_t t = 0; t < T; ++t)
        for (int64_t h = 0; h < nh; ++h) {
            const int64_t kvh = h / G, n = pos0 + t + 1;
            const float* qr = q.data() + (t * nh + h) * hd;
            std::vector<double> sc((size_t) n);
            double m = -1e300;
            for (int64_t c = 0; c < n; ++c) {
                const int64_t row = ((int64_t) table[(size_t) (c / PS)] * nkv + kvh) * PS + c % PS;
                double d = 0.0;
                for (int64_t i = 0; i < hd; ++i) d += (double) qr[i] * k::f32_from_f16(kh[(size_t) (row * hd + i)]);
                sc[(size_t) c] = d / std::sqrt((double) hd);
                m = std::max(m, sc[(size_t) c]);
            }
            double l = 0.0;
            for (auto& v : sc) { v = std::exp(v - m); l += v; }
            double* out = ref.data() + (t * nh + h) * hd;
            for (int64_t c = 0; c < n; ++c) {
                const int64_t row = ((int64_t) table[(size_t) (c / PS)] * nkv + kvh) * PS + c % PS;
                for (int64_t i = 0; i < hd; ++i) out[i] += sc[(size_t) c] / l * k::f32_from_f16(vh[(size_t) (row * hd + i)]);
            }
        }
    for (double v : ref) scale_out = std::max(scale_out, std::fabs(v));

    const uint16_t* d_k = up(dev, kh);
    const uint16_t* d_v = up(dev, vh);
    const int32_t* d_table = up(dev, table);
    const float* d_q = up(dev, q);
    float* d_attn = dev.alloc<float>(q.size());
    if (!k::prompt_attn(d_q, d_k, d_v, d_table, pos0, T, s, d_attn, nullptr)) {
        std::printf("  %-34s the kernel does not take this shape: FAIL\n", name);
        return 1;
    }
    std::vector<float> attn(q.size());
    par::copy_out(attn.data(), d_attn, attn.size());

    double err = 0.0;
    for (size_t i = 0; i < attn.size(); ++i) err = std::max(err, std::fabs((double) attn[i] - ref[i]));
    const char* mma = std::getenv("STRATA_PROMPT_ATTN_MMA");
    const bool fma = mma != nullptr && std::strtol(mma, nullptr, 10) == 0;
    const double rel = err / scale_out, tol = fma ? 1e-4 : 5e-3;   // the device may take the WMMA kernel
    std::printf("  %-34s %lld queries after %lld cells, vs FP64 %.3e of the scale (%s, at most %.0e) %s\n", name,
                (long long) T, (long long) pos0, rel, fma ? "the FMA kernel" : "the device's kernel", tol,
                rel <= tol ? "ok" : "FAIL");
    return rel <= tol ? 0 : 1;
}
}  // namespace

int main() {
    int bad = 0;
    // 45 queries: not a whole number of the FMA kernel's 4 nor the WMMA kernel's 16; 37 cells before them
    bad += check("Qwen3.5-9B (16/4 x 256)", 16, 4, 256, 37, 45);
    std::printf("prompt_attn: %d failures\n", bad);
    return bad ? 1 : 0;
}
