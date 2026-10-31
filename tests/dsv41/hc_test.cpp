// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "ds4_kernels.cuh"
#include "strata/core/readback.hpp"
#include "sycl_test.hpp"
#include <algorithm>
#include <array>
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
constexpr float sentinel = 12345.f;
constexpr unsigned counter_guard = 0x5a5aa5a5U;
constexpr size_t suffix = 9;

struct Memory {
    sycl::queue& queue;
    std::array<void*, 48> pointers{};
    size_t count = 0;
    explicit Memory(sycl::queue& q) : queue(q) {}
    ~Memory() {
        // Host sources/destinations are declared before this owner, including on exceptions.
        try { queue.wait_and_throw(); for (void* p : pointers) if (p) sycl::free(p, queue); }
        catch (...) { std::terminate(); }
    }
    Memory(const Memory&) = delete;
    Memory& operator=(const Memory&) = delete;
    template <class T> T* upload(const std::vector<T>& values) {
        if (count == pointers.size()) throw std::logic_error("too many HC test allocations");
        auto* p = sycl::malloc_device<T>(values.size(), queue);
        if (!p) throw std::bad_alloc();
        pointers[count++] = p;
        strata::core::copy_async(queue, p, values.data(), values.size() * sizeof(T));
        return p;
    }
};

struct Buffer {
    size_t prefix;
    std::vector<float> initial, actual;
    std::vector<double> expected;
    std::vector<uint8_t> active;
    float* allocation = nullptr;
    Buffer(size_t elements, size_t offset)
        : prefix(offset), initial(offset + elements + suffix, sentinel), actual(initial.size()),
          expected(initial.size(), double(sentinel)), active(initial.size(), 0) {}
    float* device() const { return allocation + prefix; }
    float input(size_t i) const { return initial[prefix + i]; }
    void input(size_t i, float value) { initial[prefix + i] = value; }
    void want(size_t i, double value) { expected[prefix + i] = value; active[prefix + i] = 1; }
    void upload(Memory& memory) { allocation = memory.upload(initial); }
    int check(sycl::queue& queue, const char* name) {
        strata::core::readback(queue, actual.data(), allocation, actual.size() * sizeof(float));
        int bad = 0;
        double worst = 0;
        for (size_t i = 0; i < actual.size(); ++i) {
            const double ref = active[i] ? expected[i] : double(initial[i]);
            const double error = std::abs(double(actual[i]) - ref);
            // Double reference is independent of lane order; include float reduction roundoff.
            const double tolerance = active[i] ? 2e-5 + 2e-4 * std::abs(ref) : 0;
            worst = std::max(worst, error);
            if (!std::isfinite(actual[i]) || error > tolerance) {
                if (bad++ == 0) std::printf("HC %s mismatch at %zu: %.9g expected %.12g\n", name, i, actual[i], ref);
            }
        }
        if (bad) std::printf("HC %s mismatches=%d max_abs=%.9g\n", name, bad, worst);
        return bad;
    }
};

struct Outputs {
    Buffer mixes, pre, post, comb;
    explicit Outputs(int tokens)
        : mixes(size_t(tokens) * 32, 7), pre(size_t(tokens) * 8, 7),
          post(size_t(tokens) * 8, 7), comb(size_t(tokens) * 64, 7) {}
    void upload(Memory& memory) { mixes.upload(memory); pre.upload(memory); post.upload(memory); comb.upload(memory); }
    int check(sycl::queue& queue, const char* name) {
        std::printf("HC check %s\n", name);
        return mixes.check(queue, "mixes") + pre.check(queue, "pre") + post.check(queue, "post") + comb.check(queue, "comb");
    }
};

struct Spec {
    int hc = 0, dim = 0, tokens = 0, iters = 0, mixes = 0;
    size_t input_offset = 0;
    bool zeros = false;
};

// Deterministic dyadic inputs, with independent streams for each tensor.
float value(size_t index, uint32_t stream) {
    uint32_t x = uint32_t(index) + stream * 0x9e3779b9U;
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
    return float(int(x & 1023U) - 512) / 1024.f;
}

// Mathematical reference: RMS-scaled projection, sigmoid gates, row softmax and
// alternating column/row normalization. No device reduction or shuffle is reused.
void split_reference(const std::vector<double>& mixes, const Buffer& scale, const Buffer& base,
                     const Spec& s, double eps, Outputs& out) {
    for (int t = 0; t < s.tokens; ++t) {
        std::array<double, 16> matrix{};
        for (int j = 0; j < s.hc; ++j) {
            const size_t row = size_t(t) * 32;
            const auto sigmoid = [](double x) { return 1.0 / (1.0 + std::exp(-x)); };
            out.pre.want(size_t(t) * 8 + j, sigmoid(mixes[row + j] * scale.input(0) + base.input(size_t(j))) + eps);
            out.post.want(size_t(t) * 8 + j, 2.0 * sigmoid(mixes[row + s.hc + j] * scale.input(1) + base.input(size_t(s.hc) + j)));
            double maximum = -std::numeric_limits<double>::infinity();
            for (int k = 0; k < s.hc; ++k) {
                const size_t cell = size_t(j) * s.hc + k;
                const size_t m = size_t(2 * s.hc) + cell;
                matrix[cell] = mixes[row + m] * scale.input(2) + base.input(m);
                maximum = std::max(maximum, matrix[cell]);
            }
            double sum = 0;
            for (int k = 0; k < s.hc; ++k) {
                auto& cell = matrix[size_t(j) * s.hc + k]; cell = std::exp(cell - maximum); sum += cell;
            }
            for (int k = 0; k < s.hc; ++k) matrix[size_t(j) * s.hc + k] = matrix[size_t(j) * s.hc + k] / sum + eps;
        }
        // Even a nonpositive iteration count includes the first column normalization.
        for (int iteration = 0; iteration < std::max(1, s.iters); ++iteration) {
            if (iteration) for (int j = 0; j < s.hc; ++j) {
                double sum = 0;
                for (int k = 0; k < s.hc; ++k) sum += matrix[size_t(j) * s.hc + k];
                for (int k = 0; k < s.hc; ++k) matrix[size_t(j) * s.hc + k] /= sum + eps;
            }
            for (int k = 0; k < s.hc; ++k) {
                double sum = 0;
                for (int j = 0; j < s.hc; ++j) sum += matrix[size_t(j) * s.hc + k];
                for (int j = 0; j < s.hc; ++j) matrix[size_t(j) * s.hc + k] /= sum + eps;
            }
        }
        for (int j = 0; j < s.hc; ++j) for (int k = 0; k < s.hc; ++k)
            out.comb.want(size_t(t) * 64 + size_t(j) * s.hc + k, matrix[size_t(j) * s.hc + k]);
    }
}

void mix_reference(const Buffer& h, const Buffer& fn, const Buffer& scale, const Buffer& base,
                   const Spec& s, double eps, double hc_eps, Outputs& out) {
    const int length = s.hc * s.dim;
    std::vector<double> mixes(size_t(s.tokens) * 32, 0);
    for (int t = 0; t < s.tokens; ++t) {
        double squares = 0;
        for (int k = 0; k < length; ++k) { const double v = h.input(size_t(t) * length + k); squares += v * v; }
        const double inv = 1.0 / std::sqrt(squares / length + eps);
        for (int m = 0; m < s.mixes; ++m) {
            double dot = 0;
            for (int k = 0; k < length; ++k) dot += double(h.input(size_t(t) * length + k)) * fn.input(size_t(m) * length + k);
            mixes[size_t(t) * 32 + m] = dot * inv;
            out.mixes.want(size_t(t) * 32 + m, dot * inv);
        }
    }
    split_reference(mixes, scale, base, s, hc_eps, out);
}

void norm_reference(const Buffer& h, const Buffer& mix, const Buffer& w, const Spec& s,
                    double eps, Buffer& normalized, Buffer& unnormalized) {
    std::vector<double> row(size_t(s.dim), 0);
    for (int t = 0; t < s.tokens; ++t) {
        double squares = 0;
        for (int e = 0; e < s.dim; ++e) {
            double sum = 0;
            for (int j = 0; j < s.hc; ++j) sum += double(mix.input(size_t(t) * 8 + j)) * h.input((size_t(t) * s.hc + j) * s.dim + e);
            row[size_t(e)] = sum; squares += sum * sum;
            unnormalized.want(size_t(t) * s.dim + e, sum);
        }
        const double inv = 1.0 / std::sqrt(squares / s.dim + eps);
        for (int e = 0; e < s.dim; ++e) normalized.want(size_t(t) * s.dim + e, row[size_t(e)] * inv * w.input(size_t(e)));
    }
}

int run_case(sycl::queue& queue, const Spec& s) {
    constexpr float norm_eps = 1e-6f, hc_eps = 1e-5f;
    const int length = s.hc * s.dim;
    Buffer h(size_t(s.tokens) * length, s.input_offset), h2(size_t(s.tokens) * length, s.input_offset);
    Buffer fn(size_t(s.mixes) * length, s.input_offset), scale(3, 4), base(32, 4);
    Buffer premix(size_t(s.tokens) * 8, 5), w(size_t(s.dim), 5), split_mixes(size_t(s.tokens) * 32, 5);
    Buffer normalized(size_t(s.tokens) * s.dim, 7), unnormalized(size_t(s.tokens) * s.dim, 7);
    Outputs separate(s.tokens), combined(s.tokens), counted1(s.tokens), counted2(s.tokens), null_count(s.tokens), split_only(s.tokens);
    std::vector<unsigned> counter_initial(size_t(s.tokens) + 4 + suffix, counter_guard), counter_actual(counter_initial.size());
    std::vector<double> split_input(size_t(s.tokens) * 32, 0);
    for (int t = 0; t < s.tokens; ++t) {
        counter_initial[size_t(t) + 4] = 0;
        for (int k = 0; k < length; ++k) {
            const size_t at = size_t(t) * length + k;
            h.input(at, s.zeros ? 0.f : value(at, 1)); h2.input(at, value(at, 2));
        }
        for (int j = 0; j < s.hc; ++j) premix.input(size_t(t) * 8 + j, .5f + value(size_t(t) * 8 + j, 3));
        for (int m = 0; m < s.mixes; ++m) {
            const size_t at = size_t(t) * 32 + m;
            split_mixes.input(at, value(at, 4) * 8.f); split_input[at] = split_mixes.input(at);
        }
    }
    for (size_t i = 0; i < size_t(s.mixes) * length; ++i) fn.input(i, value(i, 5) / 32.f);
    scale.input(0, .625f); scale.input(1, -.75f); scale.input(2, .5f);
    for (size_t i = 0; i < 32; ++i) base.input(i, value(i, 6));
    for (int e = 0; e < s.dim; ++e) w.input(size_t(e), 1.f + value(size_t(e), 7));
    for (auto* output : {&separate, &combined, &counted1, &null_count}) mix_reference(h, fn, scale, base, s, norm_eps, hc_eps, *output);
    mix_reference(h2, fn, scale, base, s, norm_eps, hc_eps, counted2);
    split_reference(split_input, scale, base, s, hc_eps, split_only);
    norm_reference(h, premix, w, s, norm_eps, normalized, unnormalized);

    // No host vector used by a copy is destroyed until this owner has drained the queue.
    Memory memory(queue);
    for (auto* buffer : {&h, &h2, &fn, &scale, &base, &premix, &w, &split_mixes, &normalized, &unnormalized}) buffer->upload(memory);
    for (auto* output : {&separate, &combined, &counted1, &counted2, &null_count, &split_only}) output->upload(memory);
    unsigned* counter = memory.upload(counter_initial);
    ds::hc_mix(h.device(), fn.device(), length, s.mixes, s.tokens, norm_eps, separate.mixes.device(), queue);
    ds::hc_split(separate.mixes.device(), scale.device(), base.device(), s.hc, s.iters, hc_eps, s.tokens,
                 separate.pre.device(), separate.post.device(), separate.comb.device(), queue);
    ds::hc_split(split_mixes.device(), scale.device(), base.device(), s.hc, s.iters, hc_eps, s.tokens,
                 split_only.pre.device(), split_only.post.device(), split_only.comb.device(), queue);
    ds::hc_mix_split(h.device(), fn.device(), length, s.mixes, s.tokens, norm_eps, combined.mixes.device(),
                     scale.device(), base.device(), s.hc, s.iters, hc_eps, combined.pre.device(), combined.post.device(), combined.comb.device(), queue);
    // Same counter, different inputs and outputs: the second launch cannot pass as a no-op.
    // There is deliberately no host wait, upload or counter reset between these invocations.
    ds::hc_mix_split(h.device(), fn.device(), length, s.mixes, s.tokens, norm_eps, counted1.mixes.device(), counter + 4,
                     scale.device(), base.device(), s.hc, s.iters, hc_eps, counted1.pre.device(), counted1.post.device(), counted1.comb.device(), queue);
    ds::hc_mix_split(h2.device(), fn.device(), length, s.mixes, s.tokens, norm_eps, counted2.mixes.device(), counter + 4,
                     scale.device(), base.device(), s.hc, s.iters, hc_eps, counted2.pre.device(), counted2.post.device(), counted2.comb.device(), queue);
    ds::hc_mix_split(h.device(), fn.device(), length, s.mixes, s.tokens, norm_eps, null_count.mixes.device(), nullptr,
                     scale.device(), base.device(), s.hc, s.iters, hc_eps, null_count.pre.device(), null_count.post.device(), null_count.comb.device(), queue);
    ds::hc_pre_norm(h.device(), premix.device(), w.device(), s.dim, s.hc, s.tokens, norm_eps, normalized.device(), queue);
    ds::hc_pre_norm(h.device(), premix.device(), nullptr, s.dim, s.hc, s.tokens, norm_eps, unnormalized.device(), queue);
    int bad = 0;
    bad += separate.check(queue, "separate"); bad += combined.check(queue, "combined");
    bad += counted1.check(queue, "counter_first"); bad += counted2.check(queue, "counter_reuse");
    bad += null_count.check(queue, "null_counter"); bad += split_only.check(queue, "split_only");
    bad += normalized.check(queue, "pre_norm"); bad += unnormalized.check(queue, "pre_norm_null_weight");
    for (auto* input : {&h, &h2, &fn, &scale, &base, &premix, &w, &split_mixes}) bad += input->check(queue, "unchanged_input");
    strata::core::readback(queue, counter_actual.data(), counter, counter_actual.size() * sizeof(unsigned));
    for (size_t i = 0; i < counter_actual.size(); ++i) if (counter_actual[i] != counter_initial[i]) {
        if (bad++ == 0) std::printf("HC counter/guard mismatch at %zu: %u expected %u\n", i, counter_actual[i], counter_initial[i]);
    }
    std::printf("HC hc=%d dim=%d T=%d iters=%d n_mix=%d offset=%zu zeros=%d: %s\n",
                s.hc, s.dim, s.tokens, s.iters, s.mixes, s.input_offset, int(s.zeros), bad ? "FAIL" : "PASS");
    return bad;
}
}  // namespace

int main() try {
    if (sycl::device::get_devices(sycl::info::device_type::gpu).empty()) {
        std::puts("HC test: no GPU, skipped"); return 77;
    }
    auto& queue = dsv41_test::queue();
    const std::array<Spec, 12> cases{{
        {1, 1, 3, -2, 3, 4, false},
        {2, 17, 3, 0, 8, 4, false},
        {3, 33, 3, 1, 15, 4, false},
        {4, 257, 3, 20, 24, 4, false},
        {4, 260, 2, 2, 24, 5, false},
        {4, 5120, 1, 20, 24, 4, false},
        {4, 8193, 1, 1, 32, 4, false},
        {4, 64, 2, 20, 24, 4, true},
        {4, 64, 1, 0, 24, 4, false},
        {4, 64, 1, -2, 24, 4, false},
        {1, 16, 1, 0, 32, 4, false},
        {4, 5120, 0, 20, 24, 4, false},
    }};
    int bad = 0;
    for (const Spec& spec : cases) bad += run_case(queue, spec);
    std::printf("HC independent-reference test: %s (%d mismatches)\n", bad ? "FAIL" : "PASS", bad);
    return bad ? 1 : 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "HC test: %s\n", error.what()); return 1;
} catch (...) {
    std::fputs("HC test: unknown exception\n", stderr); return 1;
}
