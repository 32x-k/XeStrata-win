// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/iq_mmq_parity.cpp - the prompt's int8 expert products (iq_mmq.hpp) in the form this GPU takes (Intel's
// XMX, NVIDIA's tensor cores, AMD's WMMA, or dp4a) against the host: the activations as the device quantized them,
// the weights dequantized by ggml (to_float), multiplied in double.
//
//     build/iq_mmq_parity --selftest
//
// A type for each way the products read scales: a scale per 32 (Q8_0, at a K with a tail past the step; IQ3_XXS), a
// scale per 16 (IQ2_XS) and a min term (Q4_K); each over three experts of 5, 40 and 17 rows and one of 70 (the tile
// shapes of 16, 32 and 64 rows).  The weights are random block bytes with a sane d: every bit pattern of the grid
// indices and signs is a valid block.  Exit 77 (skipped) where the GPU runs neither form.
#include "strata/kernels/iq_mmq.hpp"
#include "parity_device.hpp"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <random>
#include <vector>

namespace {

namespace P = strata::parity;

// A fixed sequence (a std engine seeded with a constant draws a lint warning).
struct XorShift32 {
    using result_type = uint32_t;
    uint32_t state;
    static constexpr result_type min() { return 1; }
    static constexpr result_type max() { return 0xFFFFFFFFu; }
    result_type operator()() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return state;
    }
};
XorShift32& rng() {
    static XorShift32 g{20261013u};
    return g;
}

// Rows of `cols` values of type t: ggml's quantizer where it needs no importance matrix, else random block bytes whose
// fp16 scales (the block's first two bytes, and Q4_K's next two) are set to small positive values.
std::vector<uint8_t> weights(ggml_type t, int64_t rows, int64_t cols) {
    const size_t rb = ggml_row_size(t, cols);
    std::vector<uint8_t> w(rb * (size_t) rows);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    if (!ggml_quantize_requires_imatrix(t) && t != GGML_TYPE_IQ3_XXS) {
        std::vector<float> f((size_t) (rows * cols));
        for (float& v : f) v = u(rng());
        ggml_quantize_chunk(t, f.data(), w.data(), 0, rows, cols, nullptr);
        return w;
    }
    std::uniform_int_distribution<int> byte(0, 255);
    for (uint8_t& b : w) b = (uint8_t) byte(rng());
    const size_t bs = ggml_type_size(t);
    std::uniform_real_distribution<float> d(0.002f, 0.02f);
    for (size_t at = 0; at < w.size(); at += bs) {
        const ggml_fp16_t h = ggml_fp32_to_fp16(d(rng()));
        std::memcpy(&w[at], &h, 2);
    }
    return w;
}

int check(ggml_type t, int64_t N, int64_t K, const std::vector<int32_t>& counts) {
    sycl::queue& q = strata::core::Runtime::get().compute();
    const int n = (int) counts.size();
    std::vector<int32_t> bounds(1, 0);
    int64_t max_rows = 0;
    for (int32_t c : counts) {
        bounds.push_back(bounds.back() + c);
        max_rows = std::max<int64_t>(max_rows, c);
    }
    const int64_t R = bounds.back();
    std::vector<std::vector<uint8_t>> w;
    std::vector<uint8_t> all;
    for (int e = 0; e < n; ++e) {
        w.push_back(weights(t, N, K));
        all.insert(all.end(), w.back().begin(), w.back().end());
    }
    std::vector<float> x((size_t) (R * K));
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (float& v : x) v = nd(rng());
    std::vector<int32_t> ids((size_t) R);
    for (int64_t i = 0; i < R; ++i) ids[(size_t) i] = (int32_t) i;

    P::Device dev;
    const size_t xq_bytes = strata::kernels::iq_mmq_act_bytes(R, K);
    float* dx = dev.upload(x);
    uint8_t* dw = dev.upload(all);
    int32_t* db = dev.upload(bounds);
    int32_t* di = dev.upload(ids);
    uint8_t* dxq = dev.alloc<uint8_t>(xq_bytes);
    float* dy = dev.alloc<float>((size_t) (R * N));
    strata::kernels::iq_mmq_quantize(q, dx, nullptr, dxq, K, K, R, strata::kernels::iq_mmq_type_sums((int) t)).wait();
    strata::kernels::iq_mmq_grouped(q, (int) t, dw, w[0].size(), N, K, n, dxq, R, db, max_rows, di, dy, N).wait();
    std::vector<uint8_t> xq(xq_bytes);
    std::vector<float> y((size_t) (R * N));
    P::copy_out(xq.data(), dxq, xq_bytes);
    P::copy_out(y.data(), dy, y.size());

    // the reference: the device's int8 activations times their scales, the weights through ggml's dequantizer
    const auto* x8 = reinterpret_cast<const int8_t*>(xq.data());
    const auto* xs = reinterpret_cast<const float*>(xq.data() + R * K);
    const ggml_type_traits* tr = ggml_get_type_traits(t);
    std::vector<float> wf((size_t) (N * K));
    double worst = 0.0, scale = 0.0;
    for (int e = 0; e < n; ++e) {
        for (int64_t o = 0; o < N; ++o) tr->to_float(w[(size_t) e].data() + (size_t) o * ggml_row_size(t, K), &wf[(size_t) (o * K)], K);
        for (int64_t r = bounds[(size_t) e]; r < bounds[(size_t) e + 1]; ++r)
            for (int64_t o = 0; o < N; ++o) {
                double s = 0.0;
                for (int64_t k = 0; k < K; ++k)
                    s += (double) x8[r * K + k] * xs[r * (K / 32) + k / 32] * wf[(size_t) (o * K + k)];
                worst = std::max(worst, std::fabs(s - y[(size_t) (r * N + o)]));
                scale = std::max(scale, std::fabs(s));
            }
    }
    const double rel = worst / std::max(scale, 1e-30);
    const bool ok = rel < 1e-4 && std::isfinite(rel);
    std::printf("  %-8s K %4lld, %d experts of up to %3lld rows: rel %.2e  %s\n", ggml_type_name(t), (long long) K, n,
                (long long) max_rows, rel, ok ? "ok" : "*** FAIL ***");
    return ok ? 0 : 1;
}

}  // namespace

int main() try {
    sycl::queue& q = strata::core::Runtime::get().compute();
    const bool matrix = strata::kernels::iq_mmq_usable(q), dp4a = strata::kernels::iq_mmq_dp4a_usable(q);
    std::printf("iq_mmq_parity: the int8 expert products %s\n",
                matrix ? "on the matrix engines" : dp4a ? "by dp4a" : "(neither form runs here: skipped)");
    if (!matrix && !dp4a) return 77;
    const std::vector<int32_t> three = {5, 40, 17}, one = {70};
    int bad = 0;
    for (const auto& c : {three, one}) {
        bad += check(GGML_TYPE_Q8_0, 128, 160, c);
        bad += check(GGML_TYPE_IQ3_XXS, 128, 512, c);
        bad += check(GGML_TYPE_IQ2_XS, 128, 512, c);
        bad += check(GGML_TYPE_Q4_K, 128, 512, c);
    }
    std::printf(bad ? "FAIL\n" : "PASS\n");
    return bad ? 1 : 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "iq_mmq_parity: %s\n", e.what());
    return 1;
}
