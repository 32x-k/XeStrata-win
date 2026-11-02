// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "ds4_kernels.cuh"
#include "strata/core/readback.hpp"
#include "sycl_test.hpp"
#include <algorithm>
#include <array>
#include <bit>
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
constexpr size_t guard = 7;
constexpr float sentinel = 12345.f;

struct Memory {
    sycl::queue& queue;
    std::array<void*, 5> pointers{};
    size_t used = 0;
    explicit Memory(sycl::queue& q) : queue(q) {}
    ~Memory() {
        try { queue.wait_and_throw(); for (void* p : pointers) if (p) sycl::free(p, queue); }
        catch (...) { std::terminate(); }
    }
    Memory(const Memory&) = delete;
    Memory& operator=(const Memory&) = delete;
    template <class T> T* upload(const std::vector<T>& values) {
        if (used == pointers.size()) throw std::logic_error("index test allocation limit");
        auto* p = sycl::malloc_device<T>(values.size(), queue);
        if (!p) throw std::bad_alloc();
        pointers[used++] = p;
        strata::core::copy_async(queue, p, values.data(), values.size() * sizeof(T));
        return p;
    }
};
struct Spec {
    int heads = 0, dim = 0, tokens = 0, count = 0;
    bool mapped = false, generated = false;
};

float input(size_t i, unsigned salt) {
    const auto code = uint32_t(i) * 1664525U + salt * 1013904223U;
    return float(int((code >> 16) % 127U) - 63) / 127.f;
}

// Independent format interpretation: integer BF16 bits or the enumerated E2M1
// alphabet and a binary exponent. No production decoder is used by the oracle.
double key_value(const uint8_t* row, int e, int dim, int format) {
    if (format == ds::kKvBf16) {
        const uint32_t bits = uint32_t(row[size_t(e) * 2]) | (uint32_t(row[size_t(e) * 2 + 1]) << 8);
        return double(std::bit_cast<float>(bits << 16));
    }
    constexpr std::array<double, 8> magnitude{0, .5, 1, 1.5, 2, 3, 4, 6};
    const unsigned code = (row[e / 2] >> (4 * (e % 2))) & 15U;
    const double value = (code & 8U) ? -magnitude[code & 7U] : magnitude[code & 7U];
    const unsigned scale = row[dim / 2 + e / 32];
    return scale ? std::ldexp(value, int(scale) - 127) : std::copysign(0.0, value);
}

double gamma(int operations) {
    const double error = double(operations) * (double(std::numeric_limits<float>::epsilon()) / 2);
    return error / (1 - error);
}
struct Reference { double value = 0, error = 0; };
Reference reference(const float* query, const float* weights, const uint8_t* row, const Spec& spec, int format) {
    Reference result;
    double magnitude = 0;
    const double unit = double(std::numeric_limits<float>::epsilon()) / 2;
    for (int h = 0; h < spec.heads; ++h) {
        double dot = 0, absolute = 0;
        for (int e = 0; e < spec.dim; ++e) {
            const double term = double(query[size_t(h) * size_t(spec.dim) + size_t(e)]) *
                                key_value(row, e, spec.dim, format);
            dot += term; absolute += std::abs(term);
        }
        // At most two FP32 roundings per dot term, followed by a 1-Lipschitz
        // ReLU, one product and the head sum. Fixture values stay normal/bounded.
        const double dot_error = gamma(2 * spec.dim) * absolute;
        const double weight = double(weights[h]), term = std::max(0.0, dot) * weight;
        const double term_error = std::abs(weight) * dot_error +
                                  unit * std::abs(weight) * (std::max(0.0, dot) + dot_error);
        result.value += term; result.error += term_error;
        magnitude += std::abs(term) + term_error;
    }
    result.error += gamma(spec.heads) * magnitude;
    return result;
}

template <class T> void unchanged(sycl::queue& q, const std::vector<T>& original,
                                  std::vector<T>& actual, const T* device) {
    strata::core::readback(q, actual.data(), device, actual.size() * sizeof(T));
    if (actual != original) throw std::runtime_error("index input or input guard changed");
}

int run(sycl::queue& q, const Spec& spec, int format, bool aligned, std::vector<float>& captured) {
    constexpr int block = 8, keys = 384;
    const int columns = std::max(1, spec.count), rows = std::max(1, spec.tokens);
    const int candidate_stride = (columns + block - 1) / block + 3, output_stride = columns + 9;
    const size_t key_offset = aligned ? 16 : format == ds::kKvBf16 ? 18 : 17;
    const size_t row_bytes = ds::kv_row_bytes(format, spec.dim);
    std::vector<float> query(2 * guard + size_t(rows) * spec.heads * spec.dim, sentinel);
    std::vector<float> weights(2 * guard + size_t(rows) * 64, sentinel);
    std::vector<uint8_t> key(key_offset + size_t(keys) * row_bytes + 13, uint8_t(0xa5));
    std::vector<int> candidates(2 * guard + size_t(rows) * candidate_stride, -12345);
    std::vector<float> output(2 * guard + size_t(rows) * output_stride, sentinel), actual(output.size());
    for (size_t i = guard; i < query.size() - guard; ++i) query[i] = input(i, 3);
    for (int t = 0; t < rows; ++t) {
        for (int h = 0; h < spec.heads; ++h) weights[guard + size_t(t) * 64 + size_t(h)] = input(size_t(h), unsigned(t) + 7);
        for (int c = 0; c < candidate_stride; ++c)
            candidates[guard + size_t(t) * candidate_stride + size_t(c)] = (c * 5 + t) % (keys / block);
    }
    for (int p = 0; p < keys; ++p) {
        const size_t offset = key_offset + size_t(p) * row_bytes;
        for (int e = 0; e < spec.dim; ++e) {
            if (format == ds::kKvBf16) {
                const uint16_t bits = uint16_t(0x3f00U + (unsigned(p * 13 + e * 7) & 255U) +
                                               ((unsigned(p + e) & 1U) << 15));
                key[offset + size_t(e) * 2] = uint8_t(bits & 255U);
                key[offset + size_t(e) * 2 + 1] = uint8_t(bits >> 8);
            } else if (!(e % 2)) {
                key[offset + size_t(e / 2)] = uint8_t((p + e) % 16 + 16 * ((p * 3 + e + 1) % 16));
            }
        }
        if (format == ds::kKvFp4p)
            for (int b = 0; b < spec.dim / 32; ++b)
                key[offset + size_t(spec.dim / 2 + b)] = uint8_t(125 + (p + b) % 5);
    }
    ds::Rows limits{};
    for (int t = 0; t < spec.tokens; ++t) {
        limits.n[t] = spec.mapped ? keys - 3 - t : std::max(0, spec.count - 3 - t);
        limits.k[t] = std::max(0, (spec.count + block - 1) / block - 1);
    }
    if (spec.generated) {
        limits.gen = 2; limits.R = 2; limits.p0 = (keys - 11) * 2 - 1;
        limits.ccap = std::max(0, (spec.count + block - 1) / block - 1); limits.blk = block;
    }
    // All host sources and destinations outlive the queue drain on exceptions.
    auto query_after = query, weights_after = weights;
    auto key_after = key;
    auto candidates_after = candidates;
    Memory memory(q);
    auto* dq = memory.upload(query); auto* dw = memory.upload(weights);
    auto* dk = memory.upload(key); auto* dc = memory.upload(candidates); auto* dy = memory.upload(output);
    const bool empty = spec.tokens <= 0 || spec.count <= 0;
    ds::index_score(empty ? nullptr : dq + guard, empty ? nullptr : dw + guard,
                    empty ? nullptr : dk + key_offset, spec.heads, spec.dim, spec.tokens, spec.count,
                    spec.mapped ? dc + guard : nullptr, candidate_stride, block, limits,
                    dy + guard, output_stride, q, format);
    strata::core::readback(q, actual.data(), dy, actual.size() * sizeof(float));
    int errors = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const size_t at = i >= guard ? i - guard : 0;
        const int t = int(at / size_t(output_stride)), column = int(at % size_t(output_stride));
        const bool active = !empty && i >= guard && t < spec.tokens && column < spec.count;
        bool valid = actual[i] == sentinel;
        double expected = sentinel, tolerance = 0;
        if (active) {
            const int visible = spec.generated ? int((limits.p0 + t + 1) / limits.R) : limits.n[t];
            const int blocks = spec.generated ? std::min(limits.ccap, (visible + block - 1) / block) : limits.k[t];
            const bool mapped = !spec.mapped || column < blocks * block;
            const int p = spec.mapped ? candidates[guard + size_t(t) * candidate_stride + size_t(column / block)] * block + column % block : column;
            if (!mapped || p >= visible) {
                expected = -std::numeric_limits<double>::infinity();
                valid = std::isinf(actual[i]) && actual[i] < 0;
            } else {
                const auto ref = reference(query.data() + guard + size_t(t) * spec.heads * spec.dim,
                                           weights.data() + guard + size_t(t) * 64,
                                           key.data() + key_offset + size_t(p) * row_bytes, spec, format);
                expected = ref.value; tolerance = ref.error;
                valid = std::isfinite(actual[i]) && std::abs(double(actual[i]) - expected) <= tolerance;
            }
        }
        if (!valid && errors++ == 0)
            std::printf("index h%d d%d t%d n%d map%d gen%d fmt%d aligned%d at %zu: %.9g expected %.12g tolerance %.3g\n",
                        spec.heads, spec.dim, spec.tokens, spec.count, int(spec.mapped), int(spec.generated),
                        format, int(aligned), i, actual[i], expected, tolerance);
    }
    unchanged(q, query, query_after, dq); unchanged(q, weights, weights_after, dw);
    unchanged(q, key, key_after, dk); unchanged(q, candidates, candidates_after, dc);
    captured = actual;
    return errors;
}

int alphabet(sycl::queue& q, int heads, std::vector<float>& captured) {
    constexpr int dim = 128;
    constexpr std::array<int, 8> positions{0, 7, 8, 31, 32, 63, 64, 127};
    // Encoded scales represent zero, 1/8, 1, 8 and 8192. All nonzero products
    // are normal and exact, independent of device subnormal support.
    constexpr std::array<uint8_t, 5> scales{0, 124, 127, 130, 140};
    constexpr int tokens = int(positions.size()), count = int(scales.size()) * 16;
    constexpr int stride = count + 3;
    const size_t row_bytes = ds::kv_row_bytes(ds::kKvFp4p, dim);
    std::vector<float> query(2 * guard + size_t(tokens) * size_t(heads) * dim, sentinel);
    std::vector<float> weights(2 * guard + size_t(tokens) * 64, sentinel);
    std::vector<uint8_t> keys(16 + size_t(count) * row_bytes + 13, uint8_t(0xa5));
    std::vector<float> output(2 * guard + size_t(tokens) * stride, sentinel), actual(output.size());
    std::fill(query.begin() + guard, query.end() - guard, 0.f);
    std::fill(weights.begin() + guard, weights.end() - guard, 0.f);
    ds::Rows limits{};
    for (int t = 0; t < tokens; ++t) {
        const size_t at = guard + size_t(t) * size_t(heads) * dim + size_t(positions[size_t(t)]);
        query[at] = 1.f; query[at + dim] = -1.f;
        weights[guard + size_t(t) * 64] = 1.f;
        weights[guard + size_t(t) * 64 + 1] = -1.f;
        limits.n[t] = count; limits.k[t] = count;
    }
    for (int p = 0; p < count; ++p) {
        const size_t at = 16 + size_t(p) * row_bytes;
        std::fill_n(keys.data() + at, dim / 2, uint8_t(0));
        const auto code = uint8_t(p % 16);
        for (const int e : positions)
            keys[at + size_t(e / 2)] |= uint8_t(code << (4 * (e % 2)));
        for (int b = 0; b < dim / 32; ++b)
            keys[at + size_t(dim / 2 + b)] = scales[size_t(p / 16)];
    }
    auto query_after = query, weights_after = weights;
    auto keys_after = keys;
    Memory memory(q);
    auto* dq = memory.upload(query); auto* dw = memory.upload(weights);
    auto* dk = memory.upload(keys); auto* dy = memory.upload(output);
    ds::index_score(dq + guard, dw + guard, dk + 16, heads, dim, tokens, count,
                    nullptr, 0, 1, limits, dy + guard, stride, q, ds::kKvFp4p);
    strata::core::readback(q, actual.data(), dy, actual.size() * sizeof(float));
    int errors = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const size_t at = i >= guard ? i - guard : 0;
        const int t = int(at / stride), p = int(at % stride);
        double expected = sentinel;
        if (i >= guard && t < tokens && p < count) {
            // Two heads with opposite query/weight signs recover both signs
            // through ReLU, without deriving the oracle from the new decoder.
            expected = key_value(keys.data() + 16 + size_t(p) * row_bytes,
                                 positions[size_t(t)], dim, ds::kKvFp4p);
        }
        if (double(actual[i]) != expected && errors++ == 0)
            std::printf("index E2M1 h%d at %zu: %.9g expected %.12g\n", heads, i, actual[i], expected);
    }
    unchanged(q, query, query_after, dq); unchanged(q, weights, weights_after, dw);
    unchanged(q, keys, keys_after, dk);
    captured = actual;
    return errors;
}
} // namespace

int main() try {
    if (sycl::device::get_devices(sycl::info::device_type::gpu).empty()) return 77;
    auto& q = dsv41_test::queue();
    const std::array<Spec, 8> cases{{
        {32, 128, 2, 257, true, false}, {32, 128, 1, 255, false, false},
        {31, 128, 2, 257, true, false}, {33, 128, 1, 129, false, false},
        {4, 64, 2, 17, true, false}, {32, 128, 2, 257, true, true},
        {32, 128, 0, 17, false, false}, {32, 128, 1, 0, false, false}
    }};
    int errors = 0;
    for (const auto& spec : cases) {
        for (const int format : {ds::kKvBf16, ds::kKvFp4p}) {
            std::vector<float> aligned, unaligned;
            errors += run(q, spec, format, false, unaligned);
            errors += run(q, spec, format, true, aligned);
            if (aligned.size() != unaligned.size()) throw std::logic_error("index alignment output size");
            for (size_t i = 0; i < aligned.size(); ++i) {
                if (std::bit_cast<uint32_t>(aligned[i]) != std::bit_cast<uint32_t>(unaligned[i])) {
                    if (errors++ == 0) std::printf("index aligned/fallback bits differ at %zu\n", i);
                }
            }
        }
    }
    std::vector<float> specialized, generic;
    errors += alphabet(q, 32, specialized);
    errors += alphabet(q, 33, generic);
    if (specialized.size() != generic.size()) throw std::logic_error("index alphabet output size");
    for (size_t i = 0; i < specialized.size(); ++i) {
        if (std::bit_cast<uint32_t>(specialized[i]) != std::bit_cast<uint32_t>(generic[i])) {
            if (errors++ == 0) std::printf("index E2M1 specialized/generic bits differ at %zu\n", i);
        }
    }
    std::printf("index CPU reference, aligned/fallback bits, formats, map, tails and guards: %s (%d errors)\n",
                errors ? "FAIL" : "PASS", errors);
    return errors ? 1 : 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "index parity: %s\n", e.what());
    return 1;
}
