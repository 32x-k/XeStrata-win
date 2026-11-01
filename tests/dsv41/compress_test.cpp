// SPDX-FileCopyrightText: 2026 Peasant Smith and the Project Maya contributors
// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/main/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
#include "ds4_kernels.cuh"
#include "ds4_sycl.hpp"
#include "strata/core/readback.hpp"
#include "sycl_test.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

namespace {
namespace ds = strata::ds4;
constexpr size_t prefix = 5, suffix = 11;
constexpr float sentinel = 12345.f, epsilon = .125f;

struct Memory {
    sycl::queue& queue;
    std::array<void*, 12> pointers{};
    size_t count = 0;
    explicit Memory(sycl::queue& q) : queue(q) {}
    ~Memory() {
        try { queue.wait_and_throw(); for (void* p : pointers) if (p) sycl::free(p, queue); }
        catch (...) { std::terminate(); }
    }
    Memory(const Memory&) = delete;
    Memory& operator=(const Memory&) = delete;
    float* upload(const std::vector<float>& values) {
        if (count == pointers.size()) throw std::logic_error("compress test allocation limit");
        auto* p = sycl::malloc_device<float>(values.size(), queue);
        if (!p) throw std::bad_alloc();
        pointers[count++] = p;
        strata::core::copy_async(queue, p, values.data(), values.size() * sizeof(float));
        return p;
    }
};

struct Buffer {
    std::vector<float> initial, actual;
    float* allocation = nullptr;
    explicit Buffer(size_t elements) : initial(prefix + elements + suffix, sentinel), actual(initial.size()) {}
    float* data() const { return allocation + prefix; }
    void upload(Memory& memory) { allocation = memory.upload(initial); }
    void read(sycl::queue& q) { strata::core::readback(q, actual.data(), allocation, actual.size() * sizeof(float)); }
};
struct State {
    Buffer kv, score, latent;
    State(size_t state_elements, size_t latent_elements) : kv(state_elements), score(state_elements), latent(latent_elements) {}
    void upload(Memory& memory) { kv.upload(memory); score.upload(memory); latent.upload(memory); }
    void read(sycl::queue& q) { kv.read(q); score.read(q); latent.read(q); }
};
struct Spec {
    int dim = 0, ratio = 0, position = 0, tokens = 0;
    bool norm = false;
    int split = 0;
    bool zeros = false;
};

// The pre-fusion three-stage implementation: state update, per-value softmax,
// then the existing finish entry point. Keep the softmax's three separate scans
// and ascending slot order so this tests the fusion's FP32 rounding contract.
void reference_compress(const float* kv, const float* score, float* kv_state, float* score_state,
                        int ratio, int p0, int tokens, int d, const float* w, float* latent, sycl::queue& q) {
    int group = 0;
    for (int t = 0; t < tokens; ++t) {
        const int slot = (p0 + t) % ratio;
        ds::sycl_detail::parallel(q, size_t(d), [=](size_t e) {
            kv_state[size_t(slot) * size_t(d) + e] = kv[size_t(t) * size_t(d) + e];
            score_state[size_t(slot) * size_t(d) + e] = score[size_t(t) * size_t(d) + e];
        });
        if ((p0 + t + 1) % ratio) continue;
        float* row = latent + size_t(group++) * size_t(d);
        ds::sycl_detail::parallel(q, size_t(d), [=](size_t e) {
            float maximum = -FLT_MAX, den = 0.f, value = 0.f;
            for (int j = 0; j < ratio; ++j) maximum = sycl::fmax(maximum, score_state[size_t(j) * size_t(d) + e]);
            for (int j = 0; j < ratio; ++j) den += sycl::exp(score_state[size_t(j) * size_t(d) + e] - maximum);
            for (int j = 0; j < ratio; ++j) value += kv_state[size_t(j) * size_t(d) + e] *
                (sycl::exp(score_state[size_t(j) * size_t(d) + e] - maximum) / den);
            row[e] = value;
        });
        ds::Finish f; f.w = w; f.eps = epsilon;
        ds::finish(row, 1, d, d, f, q);
    }
}

float value(size_t index, uint32_t salt) {
    uint32_t code = uint32_t(index) + salt * 0x9e3779b9U;
    code ^= code >> 16; code *= 0x7feb352dU; code ^= code >> 15;
    return float(int(code % 127U) - 63) / 32.f;
}
int completions(int position, int tokens, int ratio) { return (position + tokens) / ratio - position / ratio; }

int exact(const std::vector<float>& actual, const std::vector<float>& expected, const char* label) {
    if (actual.size() != expected.size()) throw std::logic_error("compress test vector size");
    int bad = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const uint32_t a = std::bit_cast<uint32_t>(actual[i]), b = std::bit_cast<uint32_t>(expected[i]);
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]) || a != b) {
            if (bad++ == 0) std::printf("compress %s at %zu: %.9g [0x%08x] expected %.9g [0x%08x]\n",
                                        label, i, actual[i], unsigned(a), expected[i], unsigned(b));
        }
    }
    return bad;
}

struct Mathematical {
    std::vector<float> kv, score;
    std::vector<double> latent;
};
Mathematical cpu_reference(const Spec& spec, const Buffer& input_kv, const Buffer& input_score,
                           const Buffer& weights, const State& state) {
    Mathematical result{state.kv.initial, state.score.initial,
                        std::vector<double>(state.latent.initial.begin(), state.latent.initial.end())};
    std::vector<double> row(size_t(spec.dim), 0.0);
    size_t group = 0;
    for (int t = 0; t < spec.tokens; ++t) {
        const int position = spec.position + t;
        const size_t base = prefix + size_t(position % spec.ratio) * size_t(spec.dim);
        const size_t input = prefix + size_t(t) * size_t(spec.dim);
        std::copy_n(input_kv.initial.data() + input, size_t(spec.dim), result.kv.data() + base);
        std::copy_n(input_score.initial.data() + input, size_t(spec.dim), result.score.data() + base);
        if ((position + 1) % spec.ratio) continue;
        if (spec.dim <= 512) {
            for (int e = 0; e < spec.dim; ++e) {
                double maximum = -std::numeric_limits<double>::infinity();
                for (int slot = 0; slot < spec.ratio; ++slot)
                    maximum = std::max(maximum, double(result.score[prefix + size_t(slot) * size_t(spec.dim) + size_t(e)]));
                double numerator = 0, denominator = 0;
                for (int slot = 0; slot < spec.ratio; ++slot) {
                    const size_t index = prefix + size_t(slot) * size_t(spec.dim) + size_t(e);
                    const double probability = std::exp(double(result.score[index]) - maximum);
                    numerator += double(result.kv[index]) * probability;
                    denominator += probability;
                }
                row[size_t(e)] = numerator / denominator;
            }
            double inverse = 1;
            if (spec.norm) {
                double squares = 0;
                for (double element : row) squares += element * element;
                inverse = 1 / std::sqrt(squares / double(spec.dim) + double(epsilon));
            }
            for (int e = 0; e < spec.dim; ++e) {
                const size_t index = size_t(e);
                result.latent[prefix + group * size_t(spec.dim) + index] = row[index] * inverse *
                    (spec.norm ? double(weights.initial[prefix + index]) : 1.0);
            }
        }
        ++group;
    }
    return result;
}

int check_cpu(const State& state, const Mathematical& reference, const Spec& spec, const char* label) {
    int bad = exact(state.kv.actual, reference.kv, "CPU kv state/guard") +
              exact(state.score.actual, reference.score, "CPU score state/guard");
    const size_t used = size_t(completions(spec.position, spec.tokens, spec.ratio)) * size_t(spec.dim);
    for (size_t i = 0; i < state.latent.actual.size(); ++i) {
        const bool active = i >= prefix && i - prefix < used;
        if (active && spec.dim > 512) continue; // Large rows use the old GPU bit-exact reference.
        const double expected = reference.latent[i], actual = state.latent.actual[i];
        // Independent double softmax/RMS reference; the exact GPU check above
        // separately enforces the original float evaluation/reduction order.
        const double tolerance = active ? 2e-5 + 2e-4 * std::abs(expected) : 0;
        if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
            if (bad++ == 0) std::printf("compress %s CPU latent/guard at %zu: %.12g expected %.12g\n", label, i, actual, expected);
        }
    }
    return bad;
}

int run(sycl::queue& q, const Spec& spec) {
    if (spec.dim <= 0 || spec.ratio <= 0 || spec.position < 0 || spec.tokens < 0 ||
        spec.split < 0 || spec.split >= std::max(1, spec.tokens)) throw std::logic_error("invalid compressor test shape");
    const size_t dim = size_t(spec.dim), inputs = size_t(spec.tokens) * dim;
    const size_t states = size_t(spec.ratio) * dim;
    const int groups = completions(spec.position, spec.tokens, spec.ratio);
    const size_t outputs = size_t(groups + 2) * dim; // Two unwritten rows must retain the sentinel.
    Buffer kv(inputs), score(inputs), weights(dim);
    State old(states, outputs), fused(states, outputs), split(states, outputs);
    for (size_t i = 0; i < inputs; ++i) {
        kv.initial[prefix + i] = spec.zeros ? 0.f : value(i, 1);
        score.initial[prefix + i] = value(i, 2) * 3.f;
    }
    for (size_t i = 0; i < states; ++i) {
        old.kv.initial[prefix + i] = spec.zeros ? 0.f : value(i, 3);
        old.score.initial[prefix + i] = value(i, 4) * 3.f;
    }
    for (size_t i = 0; i < dim; ++i) weights.initial[prefix + i] = 1.f + value(i, 5) / 4.f;
    fused.kv.initial = split.kv.initial = old.kv.initial;
    fused.score.initial = split.score.initial = old.score.initial;
    const Mathematical expected = cpu_reference(spec, kv, score, weights, old);
    Memory memory(q); // Drain before any host source/readback vector is destroyed on an exception.
    kv.upload(memory); score.upload(memory); weights.upload(memory);
    old.upload(memory); fused.upload(memory);
    if (spec.split) split.upload(memory);
    const float* x = spec.tokens ? kv.data() : nullptr;
    const float* s = spec.tokens ? score.data() : nullptr;
    const float* w = spec.norm ? weights.data() : nullptr;
    reference_compress(x, s, old.kv.data(), old.score.data(), spec.ratio, spec.position, spec.tokens,
                       spec.dim, w, old.latent.data(), q);
    ds::compress_win(x, s, fused.kv.data(), fused.score.data(), spec.ratio, spec.position, spec.tokens,
                     spec.dim, w, epsilon, fused.latent.data(), q);
    if (spec.split) {
        ds::compress_win(x, s, split.kv.data(), split.score.data(), spec.ratio, spec.position, spec.split,
                         spec.dim, w, epsilon, split.latent.data(), q);
        const size_t input_offset = size_t(spec.split) * dim;
        const size_t output_offset = size_t(completions(spec.position, spec.split, spec.ratio)) * dim;
        ds::compress_win(x + input_offset, s + input_offset, split.kv.data(), split.score.data(), spec.ratio,
                         spec.position + spec.split, spec.tokens - spec.split, spec.dim, w, epsilon,
                         split.latent.data() + output_offset, q);
    }
    old.read(q); fused.read(q);
    int bad = exact(fused.kv.actual, old.kv.actual, "fused kv state") +
              exact(fused.score.actual, old.score.actual, "fused score state") +
              exact(fused.latent.actual, old.latent.actual, "fused latent") +
              check_cpu(old, expected, spec, "old") + check_cpu(fused, expected, spec, "fused");
    if (spec.split) {
        split.read(q);
        bad += exact(split.kv.actual, old.kv.actual, "split kv state") +
               exact(split.score.actual, old.score.actual, "split score state") +
               exact(split.latent.actual, old.latent.actual, "split latent") + check_cpu(split, expected, spec, "split");
    }
    kv.read(q); score.read(q); weights.read(q);
    bad += exact(kv.actual, kv.initial, "input kv/guard") + exact(score.actual, score.initial, "input score/guard") +
           exact(weights.actual, weights.initial, "weights/guard");
    std::printf("compress d=%d ratio=%d p0=%d T=%d norm=%d split=%d zeros=%d groups=%d: %s\n",
                spec.dim, spec.ratio, spec.position, spec.tokens, int(spec.norm), spec.split, int(spec.zeros), groups,
                bad ? "FAIL" : "PASS");
    return bad;
}
} // namespace

int main() try {
    if (sycl::device::get_devices(sycl::info::device_type::gpu).empty()) { std::puts("compress test: no GPU, skipped"); return 77; }
    auto& queue = dsv41_test::queue();
    const std::array<Spec, 13> cases{{
        {1, 2, 1, 3, true, 1},
        {33, 4, 0, 3, true, 0}, // No completed group: only state changes.
        {255, 4, 3, 6, false, 2},
        {256, 2, 0, 4, true, 1},
        {257, 4, 2, 7, true, 3},
        {512, 2, 4095, 5, true, 2},
        {512, 128, 127, 3, true, 1},
        {512, 4, 1, 8, false, 5},
        {8192, 2, 1, 3, true, 1},
        {8193, 2, 1, 3, true, 1}, // Above the fused register bound: retained path.
        {33, 4, 3, 0, true, 0},
        {33, 1, 5, 3, false, 1},
        {256, 2, 1, 3, true, 1, true},
    }};
    int bad = 0;
    for (const auto& spec : cases) bad += run(queue, spec);
    std::printf("compress old-stage parity and independent semantics: %s (%d mismatches)\n", bad ? "FAIL" : "PASS", bad);
    return bad ? 1 : 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "compress test: %s\n", error.what()); return 1;
} catch (...) {
    std::fputs("compress test: unknown exception\n", stderr); return 1;
}
