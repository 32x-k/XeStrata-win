// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "ds4_kernels.cuh"
#include "strata/core/readback.hpp"
#include "sycl_test.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <functional>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

namespace {
namespace ds = strata::ds4;
constexpr size_t guard = 7;
constexpr int sentinel = -1234567;
constexpr float source_guard = 12345.f;
constexpr int block = 8;

struct Memory {
    sycl::queue& queue;
    std::array<void*, 3> pointers{};
    size_t used = 0;
    explicit Memory(sycl::queue& q) : queue(q) {}
    ~Memory() {
        try { queue.wait_and_throw(); for (void* p : pointers) if (p) sycl::free(p, queue); }
        catch (...) { std::terminate(); }
    }
    Memory(const Memory&) = delete;
    Memory& operator=(const Memory&) = delete;
    template <class T> T* upload(const std::vector<T>& values) {
        if (used == pointers.size()) throw std::logic_error("topk test allocation limit");
        auto* p = sycl::malloc_device<T>(values.size(), queue);
        if (!p) throw std::bad_alloc();
        pointers[used++] = p;
        strata::core::copy_async(queue, p, values.data(), values.size() * sizeof(T));
        return p;
    }
};
struct Row { int n = 0, k = 0; };
enum class Pattern { unique, constant, threshold_ties, signed_zero_inf, minus_inf };
struct Spec {
    const char* name = nullptr;
    std::vector<Row> rows;
    Pattern pattern = Pattern::unique;
    bool mapped = false;
    int generated = 0;
};

float score(int i, int row, Pattern pattern) {
    switch (pattern) {
    case Pattern::constant: return .25f;
    case Pattern::threshold_ties: return i % 17 == 0 ? 4.f : i % 3 == 0 ? 1.f : -2.f;
    case Pattern::signed_zero_inf: {
        constexpr std::array<uint32_t, 8> bits{
            0xff800000u, 0xbf800000u, 0x80000000u, 0u,
            0x3f800000u, 0x7f800000u, 0x80000000u, 0u};
        return std::bit_cast<float>(bits[size_t(i + row) % bits.size()]);
    }
    case Pattern::minus_inf: return -std::numeric_limits<float>::infinity();
    default:
        return float((uint32_t(i) * 4051u + uint32_t(row) * 31u) & 65535u) / 256.f - 128.f;
    }
}

// Interpret the current ordered-bit key contract. The oracle sorts keys to
// find the threshold; it does not copy the device radix or collective scans.
uint32_t key(float value) {
    const uint32_t bits = std::bit_cast<uint32_t>(value);
    return bits >> 31 ? ~bits : bits + 0x80000000u;
}
std::vector<int> reference(const float* values, Row row, const int* candidates) {
    std::vector<int> positions;
    if (row.n <= 0 || row.k <= 0) return positions;
    if (row.k >= row.n) {
        for (int i = 0; i < row.n; ++i) positions.push_back(i);
    } else {
        std::vector<uint32_t> ordered;
        ordered.reserve(size_t(row.n));
        for (int i = 0; i < row.n; ++i) ordered.push_back(key(values[i]));
        std::sort(ordered.begin(), ordered.end(), std::greater<uint32_t>());
        const uint32_t threshold = ordered[size_t(row.k - 1)];
        // Preserve the raw order of the current SYCL and fixed upstream:
        // greater positions first in index order, then lowest equal positions.
        for (int i = 0; i < row.n; ++i) if (key(values[i]) > threshold) positions.push_back(i);
        for (int i = 0; i < row.n && positions.size() < size_t(row.k); ++i)
            if (key(values[i]) == threshold) positions.push_back(i);
    }
    if (candidates)
        for (int& position : positions) position = candidates[position / block] * block + position % block;
    return positions;
}

template <class T> void unchanged(sycl::queue& q, const std::vector<T>& original,
                                  std::vector<T>& actual, const T* device) {
    strata::core::readback(q, actual.data(), device, actual.size() * sizeof(T));
    // Compare bytes so changing -0 to +0 in read-only scores is also detected.
    if (std::memcmp(original.data(), actual.data(), original.size() * sizeof(T)) != 0)
        throw std::runtime_error("topk input or input guard changed");
}

int run(sycl::queue& q, const Spec& spec) {
    const int tokens = int(spec.rows.size()), storage_rows = std::max(1, tokens);
    if (tokens > ds::kWin) throw std::logic_error("topk explicit test rows exceed public capacity");
    ds::Rows limits{};
    std::vector<Row> expected_rows = spec.rows;
    for (int t = 0; t < tokens; ++t) {
        limits.n[t] = spec.rows[size_t(t)].n;
        limits.k[t] = spec.rows[size_t(t)].k;
    }
    if (spec.generated) {
        // Exercise documented generated lengths independently of the unused
        // explicit arrays: (p0+t+1)/R is 1023,1024,1024,1025 here.
        limits.gen = spec.generated; limits.p0 = 2046; limits.R = 2;
        limits.kcap = 513; limits.ccap = 127; limits.blk = block;
        for (int t = 0; t < tokens; ++t) {
            const int visible = (2047 + t) / 2;
            expected_rows[size_t(t)] = spec.generated == 1 ? Row{visible, std::min(513, visible)} :
                Row{std::min(127, (visible + 7) / 8) * 8, std::min(513, visible)};
        }
    }
    int widest = 1, output_width = 1;
    for (const auto row : expected_rows) {
        widest = std::max(widest, row.n);
        output_width = std::max(output_width, std::max(row.n, row.k));
    }
    const int value_stride = widest + 9, output_stride = output_width + 11;
    const int candidate_stride = (widest + block - 1) / block + 5;
    std::vector<float> values(2 * guard + size_t(storage_rows) * value_stride, source_guard);
    std::vector<int> candidates(2 * guard + size_t(storage_rows) * candidate_stride, sentinel);
    std::vector<int> output(2 * guard + size_t(storage_rows) * output_stride, sentinel);
    std::vector<int> expected = output, actual(output.size());
    for (int t = 0; t < tokens; ++t) {
        const auto row = expected_rows[size_t(t)];
        for (int i = 0; i < row.n; ++i)
            values[guard + size_t(t) * value_stride + size_t(i)] = score(i, t, spec.pattern);
        for (int i = 0; i < (row.n + block - 1) / block; ++i) {
            // Deliberate permutation and duplicate source blocks: output IDs
            // may repeat. Selection is on logical score positions, not IDs.
            candidates[guard + size_t(t) * candidate_stride + size_t(i)] = (i * 5 + t) % 19;
        }
        const auto selected = reference(values.data() + guard + size_t(t) * value_stride, row,
            spec.mapped ? candidates.data() + guard + size_t(t) * candidate_stride : nullptr);
        std::copy(selected.begin(), selected.end(), expected.begin() +
                  static_cast<std::ptrdiff_t>(guard + size_t(t) * output_stride));
    }
    // Every host source/readback target outlives Memory's exception drain.
    auto values_after = values;
    auto candidates_after = candidates;
    Memory memory(q);
    auto* dv = memory.upload(values);
    auto* dc = memory.upload(candidates);
    auto* dy = memory.upload(output);
    ds::topk(tokens ? dv + guard : nullptr, value_stride, tokens, limits,
             spec.mapped ? dc + guard : nullptr, candidate_stride, block,
             dy + guard, output_stride, q);
    strata::core::readback(q, actual.data(), dy, actual.size() * sizeof(int));
    int errors = 0;
    for (size_t i = 0; i < actual.size(); ++i)
        if (actual[i] != expected[i] && errors++ == 0)
            std::printf("topk %s at %zu: %d expected %d\n", spec.name, i, actual[i], expected[i]);
    unchanged(q, values, values_after, dv);
    unchanged(q, candidates, candidates_after, dc);
    return errors;
}
} // namespace

int main() try {
    if (sycl::device::get_devices(sycl::info::device_type::gpu).empty()) return 77;
    auto& q = dsv41_test::queue();
    const std::vector<Spec> cases{
        {"small_unique", {{31, 1}, {32, 17}, {33, 32}}, Pattern::unique},
        {"small_ties", {{255, 63}, {256, 127}, {257, 129}}, Pattern::threshold_ties},
        {"wg_tail_unique", {{1023, 512}, {1024, 512}, {1025, 512}}, Pattern::unique},
        {"wg_tail_ties", {{1023, 512}, {1024, 512}, {1025, 512}}, Pattern::threshold_ties},
        {"real_and_tail", {{2048, 512}, {2049, 512}}, Pattern::unique},
        {"ties_across_steps", {{2049, 512}, {2049, 1537}}, Pattern::threshold_ties},
        {"all_equal", {{33, 17}, {1025, 513}, {2049, 1025}}, Pattern::constant},
        {"signed_zero_inf", {{33, 17}, {1025, 513}, {2049, 1025}}, Pattern::signed_zero_inf},
        {"all_minus_inf", {{1025, 512}}, Pattern::minus_inf},
        {"mapped_ties", {{33, 17}, {1025, 513}, {2049, 512}}, Pattern::threshold_ties, true},
        {"mapped_all", {{33, 33}, {33, 37}}, Pattern::unique, true},
        {"zero_counts", {{0, 7}, {17, 0}, {1, 1}}, Pattern::unique},
        {"take_all", {{1, 9}, {33, 33}, {33, 37}}, Pattern::threshold_ties},
        {"empty_tokens", {}, Pattern::unique},
        {"generated_all_rows", {{}, {}, {}, {}}, Pattern::threshold_ties, false, 1},
        {"generated_candidates", {{}, {}, {}, {}}, Pattern::threshold_ties, true, 3}
    };
    int errors = 0;
    for (const auto& spec : cases) errors += run(q, spec);
    std::printf("topk independent threshold, raw order, ties, tails, mapping and guards: %s (%d errors)\n",
                errors ? "FAIL" : "PASS", errors);
    return errors ? 1 : 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "topk parity: %s\n", error.what());
    return 1;
}
