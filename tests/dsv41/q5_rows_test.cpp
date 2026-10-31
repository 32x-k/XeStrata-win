// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "ds4_gguf.cuh"
#include "ds4_numerics.cuh"
#include "ds4_sycl.hpp"
#include "strata/core/readback.hpp"
#include "sycl_test.hpp"
#include "ggml.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

namespace {
namespace gg = strata::ds4::gguf;
namespace ds = strata::ds4;

struct Memory {
    sycl::queue& queue;
    std::array<void*, 3> allocations{};
    size_t count = 0;
    explicit Memory(sycl::queue& q) : queue(q) {}
    ~Memory() {
        // This object follows every host copy buffer, so exceptions also drain before those buffers are destroyed.
        try { queue.wait_and_throw(); for (void* p : allocations) if (p) sycl::free(p, queue); }
        catch (...) { std::terminate(); }
    }
    Memory(const Memory&) = delete;
    Memory& operator=(const Memory&) = delete;
    template <class T> T* get(size_t size) {
        if (count == allocations.size()) throw std::logic_error("too many test allocations");
        T* p = sycl::malloc_device<T>(size, queue);
        if (!p) throw std::bad_alloc();
        allocations[count++] = p;
        return p;
    }
};

void half_bytes(uint8_t* dst, uint16_t value) {
    dst[0] = uint8_t(value); dst[1] = uint8_t(value >> 8);
}

float round_even(float value) {
    const float lower = std::floor(value), part = value - lower;
    return lower + float(part > .5f || (part == .5f && std::fmod(lower, 2.f) != 0.f));
}

int check_mv(sycl::queue& queue, const gg::x3::Mat& matrix, const float* weights) {
    constexpr size_t prefix = 7, suffix = 11;
    constexpr float guard = 12345.f;
    const size_t k = size_t(matrix.k), rows = size_t(matrix.n), xcs = k + 3, ycs = rows + 5;
    std::vector<float> input(prefix + 3 * xcs + suffix, std::numeric_limits<float>::quiet_NaN());
    std::vector<float> bias(rows), actual(prefix + 3 * ycs + suffix), expected(actual.size());
    std::vector<uint8_t> active(actual.size());
    for (size_t col = 0; col < 3; ++col)
        for (size_t j = 0; j < k; ++j)
            input[prefix + col * xcs + j] = col == 0 && j < 32 ? 0.f : float(int((j * 37 + col * 11) % 65) - 32) / 32.f;
    for (size_t row = 0; row < rows; ++row) bias[row] = float(int(row % 9) - 4) / 8.f;
    Memory memory(queue);
    auto* device_input = memory.get<float>(input.size());
    auto* device_bias = memory.get<float>(bias.size());
    auto* device_output = memory.get<float>(actual.size());
    queue.memcpy(device_input, input.data(), input.size() * sizeof(float)).wait_and_throw();
    queue.memcpy(device_bias, bias.data(), bias.size() * sizeof(float)).wait_and_throw();
    int errors = 0;
    for (const int columns : {2, 3}) for (const bool with_bias : {false, true}) {
        const float alpha = columns == 2 ? -.75f : 1.25f;
        std::fill(actual.begin(), actual.end(), guard);
        std::fill(expected.begin(), expected.end(), guard);
        std::fill(active.begin(), active.end(), 0);
        for (int col = 0; col < columns; ++col) for (size_t row = 0; row < rows; ++row) {
            float sum = 0.f;
            const float* x = input.data() + prefix + size_t(col) * xcs;
            for (size_t first = 0; first < k; first += 32) {
                float amax = 0.f;
                for (size_t j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(x[first + j]));
                const float scale = amax / 127.f;
                float dot = 0.f;
                for (size_t j = 0; j < 32; ++j) {
                    const float code = amax > 0.f ? round_even(x[first + j] / scale) : 0.f;
                    dot += code * weights[row * k + first + j];
                }
                sum = std::fma(scale, dot, sum);
            }
            const size_t at = prefix + size_t(col) * ycs + row;
            expected[at] = alpha * sum + (with_bias ? bias[row] : 0.f);
            active[at] = 1;
        }
        queue.memcpy(device_output, actual.data(), actual.size() * sizeof(float)).wait_and_throw();
        gg::x3::MvJob job{};
        job.m = &matrix; job.xf = device_input + prefix; job.y = device_output + prefix;
        job.bias = with_bias ? device_bias : nullptr; job.alpha = alpha;
        job.ncol = columns; job.xf_cs = xcs; job.y_cs = ycs;
        if (!gg::mv(&job, 1, queue)) throw std::runtime_error("Q5_K MV launch rejected");
        strata::core::readback(queue, actual.data(), device_output, actual.size() * sizeof(float));
        size_t mismatch = actual.size();
        for (size_t i = 0; i < actual.size(); ++i) {
            const bool pass = active[i] ? std::isfinite(actual[i]) &&
                std::fabs(actual[i] - expected[i]) <= 1e-4f + 1e-4f * std::fabs(expected[i]) : actual[i] == guard;
            if (!pass) { mismatch = i; break; }
        }
        std::printf("Q5_K MV k=%d n=%d columns=%d bias=%d: %s\n", matrix.k, matrix.n, columns,
                    int(with_bias), mismatch == actual.size() ? "PASS" : "FAIL");
        if (mismatch != actual.size()) {
            ++errors;
            std::printf("  first mismatch at %zu: actual=%.9g expected=%.9g\n", mismatch,
                        double(actual[mismatch]), double(expected[mismatch]));
        }
    }
    return errors;
}

int check_shape(sycl::queue& queue, int k, int rows, bool padded) {
    constexpr size_t raw_offset = 5, prefix = 7, suffix = 11;
    constexpr uint16_t guard = 0x5a5a;
    const size_t row_bytes = ggml_row_size(GGML_TYPE_Q5_K, k);
    const size_t stride = row_bytes + (padded ? 3 : 0);
    const size_t block_bytes = ggml_type_size(GGML_TYPE_Q5_K);
    std::vector<uint8_t> raw(raw_offset + size_t(rows) * stride + 13, 0xa7), raw_after(raw.size());
    std::vector<uint16_t> actual(prefix + size_t(rows) * k + suffix, guard), expected(actual.size(), guard);
    std::vector<float> decoded(size_t(rows) * k);
    // ggml's typed block pointer needs alignment; only the GPU input deliberately has odd byte alignment/stride.
    std::vector<uint64_t> aligned((row_bytes + sizeof(uint64_t) - 1) / sizeof(uint64_t));
    const auto* traits = ggml_get_type_traits(GGML_TYPE_Q5_K);
    if (!traits || !traits->to_float) throw std::runtime_error("Q5_K CPU decoder missing");
    for (int row = 0; row < rows; ++row) {
        uint8_t* encoded = raw.data() + raw_offset + size_t(row) * stride;
        for (size_t j = 0; j < row_bytes; ++j)
            encoded[j] = uint8_t((j * 73 + size_t(row) * 29 + (j >> 2) * 17) & 255u);
        for (size_t block = 0; block < row_bytes / block_bytes; ++block) {
            // Finite non-power-of-two scales exercise the scale/minimum products as well as all packed code bits.
            half_bytes(encoded + block * block_bytes, uint16_t(0x2e35 + (size_t(row) * 11 + block * 17) % 96));
            half_bytes(encoded + block * block_bytes + 2, uint16_t(0x2831 + (size_t(row) * 7 + block * 13) % 128));
        }
        std::memcpy(aligned.data(), encoded, row_bytes);
        traits->to_float(aligned.data(), decoded.data() + size_t(row) * k, k);
    }
    Memory memory(queue);
    auto* device_raw = memory.get<uint8_t>(raw.size());
    auto* device_output = memory.get<uint16_t>(actual.size());
    queue.memcpy(device_raw, raw.data(), raw.size()).wait_and_throw();
    gg::x3::Mat matrix{};
    matrix.fmt = gg::kFmtGG; matrix.K = int(GGML_TYPE_Q5_K);
    matrix.raw = device_raw + raw_offset; matrix.k = k; matrix.n = rows;
    matrix.ld = padded ? int(stride) : 0;
    int errors = 0;
    for (const bool view : {false, true}) {
        if (view && rows < 3) continue;
        const int first = view ? 1 : 0, count = view ? rows - 2 : rows;
        const auto selected = view ? gg::view(matrix, first, count) : matrix;
        errors += check_mv(queue, selected, decoded.data() + size_t(first) * k);
        for (const bool bf16 : {false, true}) {
            std::fill(actual.begin(), actual.end(), guard);
            std::fill(expected.begin(), expected.end(), guard);
            for (size_t i = 0; i < size_t(count) * k; ++i) {
                const float value = decoded[size_t(first) * k + i];
                expected[prefix + i] = bf16 ? ds::bf16_rn(value) : ds::sycl_detail::f16_bits(value);
            }
            queue.memcpy(device_output, actual.data(), actual.size() * sizeof(uint16_t)).wait_and_throw();
            const int errors_before = gg::launch_errors();
            gg::rows(selected, device_output + prefix, bf16, queue);
            strata::core::readback(queue, actual.data(), device_output, actual.size() * sizeof(uint16_t));
            const size_t mismatch = size_t(std::mismatch(actual.begin(), actual.end(), expected.begin()).first - actual.begin());
            const bool pass = mismatch == actual.size() && gg::launch_errors() == errors_before;
            std::printf("Q5_K rows k=%d n=%d stride=%s view=%d format=%s: %s\n", k, rows,
                        padded ? "odd" : "default", int(view), bf16 ? "BF16" : "F16", pass ? "PASS" : "FAIL");
            if (!pass) {
                ++errors;
                if (mismatch < actual.size())
                    std::printf("  first mismatch at %zu: actual=%04x expected=%04x\n", mismatch,
                                unsigned(actual[mismatch]), unsigned(expected[mismatch]));
            }
        }
    }
    strata::core::readback(queue, raw_after.data(), device_raw, raw_after.size());
    if (raw != raw_after) { std::fputs("Q5_K rows modified its source or padding\n", stderr); ++errors; }
    return errors;
}
}

int main() try {
    if (sycl::device::get_devices(sycl::info::device_type::gpu).empty()) {
        std::puts("SKIP: no SYCL GPU"); return 77;
    }
    ggml_context* context = ggml_init({1u << 20, nullptr, true});
    if (!context) throw std::runtime_error("ggml initialization failed");
    ggml_free(context);
    auto& queue = dsv41_test::queue();
    int errors = 0;
    for (const int k : {256, 512})
        for (const int rows : {1, 3, 17})
            for (const bool padded : {false, true}) errors += check_shape(queue, k, rows, padded);
    const auto device = queue.get_device();
    const size_t max_group = std::min(device.get_info<sycl::info::device::max_work_group_size>(),
                                      device.get_info<sycl::info::device::max_work_item_sizes<1>>()[0]);
    size_t tail_rows = 1;
    for (const size_t width : device.get_info<sycl::info::device::sub_group_sizes>())
        if (width <= max_group) tail_rows = std::max(tail_rows, width);
    if (tail_rows < size_t(std::numeric_limits<int>::max())) errors += check_shape(queue, 256, int(tail_rows + 1), true);
    // Choose a valid quantized row whose activation cache exceeds this device's reported local memory.
    const uint64_t local_words = device.get_info<sycl::info::device::local_mem_size>() / sizeof(float);
    const uint64_t fallback_k = (local_words / 256 + 1) * 256;
    const uint64_t max_alloc = device.get_info<sycl::info::device::max_mem_alloc_size>();
    if (fallback_k <= uint64_t(std::numeric_limits<int>::max()) && (3 * fallback_k + 32) * sizeof(float) <= max_alloc)
        errors += check_shape(queue, int(fallback_k), 1, true);
    else std::puts("SKIP: scalar MV fallback fixture exceeds this device's allocation limits");
    std::printf("Q5_K rows/MV: %s (%d errors)\n", errors ? "FAIL" : "PASS", errors);
    return errors ? 1 : 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "Q5_K rows/MV: %s\n", error.what());
    return 1;
} catch (...) {
    std::fputs("Q5_K rows/MV: unknown failure\n", stderr);
    return 1;
}
