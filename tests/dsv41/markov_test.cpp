// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "ds4_kernels.cuh"
#include "ds4_numerics.cuh"
#include "ds4_sycl.hpp"
#include "sycl_test.hpp"
#include "strata/core/readback.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <vector>

namespace {
namespace ds = strata::ds4;
namespace sd = ds::sycl_detail;
template <class T> T* upload(const std::vector<T>& v) {
    T* p;
    dsv41_test::device_alloc(p, v.size() * sizeof(T));
    strata::core::copy_async(dsv41_test::queue(), p, v.data(), v.size() * sizeof(T)).wait_and_throw();
    return p;
}
int check(int R, int V, bool q8, bool offset) {
    const size_t rank = size_t(R), vocab = size_t(V), pad = offset ? 1 : 0;
    std::vector<uint16_t> embedding(3 * rank), head(pad + vocab * rank);
    const size_t row_bytes = rank / 32 * 34;
    std::vector<uint8_t> quantized(pad + (q8 ? vocab * row_bytes : 1));
    std::vector<float> initial(vocab + 2, 12345.f), actual(initial.size());
    std::vector<double> expected(vocab);
    for (size_t i = 0; i < embedding.size(); ++i) embedding[i] = ds::bf16_rn(float(int(i % 37) - 18) / 32.f);
    for (size_t v = 0; v < vocab; ++v) {
        initial[v + 1] = float(int(v % 7) - 3) / 8.f;
        for (size_t i = 0; i < rank; ++i) head[pad + v * rank + i] = sd::f16_bits(float(int((v + i * 7) % 41) - 20) / 32.f);
        if (q8) for (size_t b = 0; b < rank / 32; ++b) {
            auto* block = quantized.data() + pad + v * row_bytes + b * 34;
            const uint16_t scale = sd::f16_bits(float(1 + (v + b) % 5) / 128.f);
            block[0] = uint8_t(scale); block[1] = uint8_t(scale >> 8);
            for (size_t j = 0; j < 32; ++j) block[2 + j] = uint8_t(int8_t(int((v + b * 11 + j * 13) % 255) - 127));
        }
        expected[v] = initial[v + 1];
        for (size_t i = 0; i < rank; ++i) {
            float weight = sd::f16_value(head[pad + v * rank + i]);
            if (q8) {
                const auto* block = quantized.data() + pad + v * row_bytes + (i / 32) * 34;
                weight = sd::f16_value(uint16_t(block[0] | uint16_t(block[1]) << 8)) * float(int8_t(block[2 + i % 32]));
            }
            expected[v] += double(sd::bf16_value(embedding[rank + i])) * weight;
        }
    }
    auto& queue = dsv41_test::queue();
    auto* e = upload(embedding);
    auto* h = upload(head);
    auto* hq = upload(quantized);
    auto* logits = upload(initial);
    auto* token = upload(std::vector<int>{1});
    if (q8) ds::markov_add_q8(logits + 1, e, hq + pad, token, V, R, queue);
    else ds::markov_add(logits + 1, e, h + pad, token, V, R, queue);
    strata::core::readback(queue, actual.data(), logits, actual.size() * sizeof(float));
    double worst = 0;
    for (size_t v = 0; v < vocab; ++v) {
        const double error = std::fabs(double(actual[v + 1]) - expected[v]) / (1 + std::fabs(expected[v]));
        worst = std::isfinite(error) ? std::max(worst, error) : 1;
    }
    bool ok = worst <= 1e-5 && actual.front() == initial.front() && actual.back() == initial.back();
    if (!q8) {
        strata::core::copy_async(queue, logits, initial.data(), initial.size() * sizeof(float)).wait_and_throw();
        ds::markov_share(logits + 1, h + pad, e + rank, V, R, queue);
        strata::core::readback(queue, actual.data(), logits, actual.size() * sizeof(float));
        for (size_t v = 0; v < vocab; ++v) {
            const double error = std::fabs(double(actual[v + 1]) - expected[v]) / (1 + std::fabs(expected[v]));
            worst = std::isfinite(error) ? std::max(worst, error) : 1;
        }
        ok = ok && worst <= 1e-5 && actual.front() == initial.front() && actual.back() == initial.back();
    }
    std::printf("Markov R=%d V=%d q8=%d offset=%d relative=%.3g %s\n", R, V, int(q8), int(offset), worst, ok ? "PASS" : "FAIL");
    dsv41_test::device().clear();
    return ok ? 0 : 1;
}
}
int main() try {
    if (sycl::device::get_devices(sycl::info::device_type::gpu).empty()) return 77;
    int errors = 0;
    for (const int rank : {32, 128, 256}) for (const bool q8 : {false, true})
        for (const bool offset : {false, true}) errors += check(rank, 257, q8, offset);
    errors += check(41, 17, false, true); // shape fallback, without assuming a subgroup width
    errors += check(256, 129280, true, false); // official drafter's vocabulary/rank
    return errors ? 1 : 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "Markov test: %s\n", e.what());
    return 1;
}
