// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "ds4_gguf.cuh"
#include "strata/core/readback.hpp"
#include "sycl_test.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
namespace gg = strata::ds4::gguf;
constexpr float guard = 12345.f;
constexpr size_t output_prefix = 7, suffix = 11;

struct Memory {
    sycl::queue& queue;
    std::array<void*, 32> pointers{};
    size_t count = 0;
    explicit Memory(sycl::queue& q) : queue(q) {}
    ~Memory() {
        // All host copy buffers precede this owner, including on an exceptional return.
        try { queue.wait_and_throw(); for (void* p : pointers) if (p) sycl::free(p, queue); }
        catch (...) { std::terminate(); }
    }
    Memory(const Memory&) = delete;
    Memory& operator=(const Memory&) = delete;
    template <class T> T* get(size_t count_elements) {
        if (count == pointers.size()) throw std::logic_error("too many MV test allocations");
        auto* p = sycl::malloc_device<T>(count_elements, queue);
        if (!p) throw std::bad_alloc();
        pointers[count++] = p;
        return p;
    }
};

struct Spec {
    int format = 0, k = 0, n = 0, columns = 0;
    size_t input = 0, xcs = 0, row_padding = 0, raw_offset = 0, x_offset = 0;
    float alpha = 1.f;
    bool bias = false, default_strides = false;
};

struct Job {
    Spec spec{};
    size_t row_bytes = 0, xcs = 0, ycs = 0;
    gg::x3::Mat matrix{};
    std::vector<uint8_t> weights, weights_after;
    std::vector<float> bias, actual;
    std::vector<double> expected;
    std::vector<uint8_t> active;
    float* device_output = nullptr;
    uint8_t* device_weights = nullptr;
};

void put16(uint8_t* dst, uint16_t bits) {
    dst[0] = uint8_t(bits); dst[1] = uint8_t(bits >> 8);
}

uint16_t get16(const uint8_t* src) { return uint16_t(unsigned(src[0]) | (unsigned(src[1]) << 8)); }

float round_even(float value) {
    const float lower = std::floor(value), part = value - lower;
    return lower + float(part > .5f || (part == .5f && std::fmod(lower, 2.f) != 0.f));
}

// The reference decodes stored bytes independently, then accumulates in double.
// Q8_0 uses per-32 activation rounding; BF16 consumes the original F32 input.
double weight(const Job& j, int row, int col) {
    const auto* p = j.weights.data() + j.spec.raw_offset + size_t(row) * j.row_bytes;
    if (j.spec.format == gg::kFmtBF16) return std::bit_cast<float>(uint32_t(get16(p + size_t(col) * 2)) << 16);
    p += size_t(col / 32) * 34;
    const unsigned bits = get16(p);
    const double scale = std::ldexp(double(1024 + (bits & 1023)), int((bits >> 10) & 31) - 25);
    const int code = p[2 + col % 32] < 128 ? int(p[2 + col % 32]) : int(p[2 + col % 32]) - 256;
    return scale * code;
}

int run_case(sycl::queue& queue, const char* name, const std::vector<Spec>& specs) {
    std::vector<Job> data;
    std::array<std::vector<float>, 4> inputs, inputs_after;
    std::array<float*, 4> device_inputs{};
    data.reserve(specs.size());
    for (size_t ji = 0; ji < specs.size(); ++ji) {
        const Spec s = specs[ji];
        Job j{}; j.spec = s;
        j.row_bytes = s.format == gg::kFmtBF16 ? (size_t(s.k) + s.row_padding) * 2
                                              : size_t(s.k / 32) * 34 + s.row_padding;
        j.xcs = s.xcs ? s.xcs : size_t(s.k);
        j.ycs = size_t(s.n) + (s.default_strides ? 0 : 5);
        j.weights.assign(s.raw_offset + size_t(s.n) * j.row_bytes + suffix, 0xa5);
        j.weights_after.resize(j.weights.size());
        j.bias.resize(size_t(s.n));
        j.actual.assign(output_prefix + size_t(s.columns) * j.ycs + suffix, guard);
        j.expected.assign(j.actual.size(), guard);
        j.active.resize(j.actual.size());
        for (int row = 0; row < s.n; ++row) {
            auto* p = j.weights.data() + s.raw_offset + size_t(row) * j.row_bytes;
            j.bias[size_t(row)] = float(row % 9 - 4) / 16.f;
            if (s.format == gg::kFmtBF16) {
                for (int col = 0; col < s.k; ++col) {
                    const int code = int((size_t(row) * 17 + size_t(col) * 13 + ji * 31) % 257) - 128;
                    const float value = float(code) / 256.f;
                    put16(p + size_t(col) * 2, uint16_t(std::bit_cast<uint32_t>(value) >> 16));
                }
            } else {
                for (int block = 0; block < s.k / 32; ++block) {
                    const size_t block_offset = s.raw_offset + size_t(row) * j.row_bytes + size_t(block) * 34;
                    put16(j.weights.data() + block_offset, uint16_t(0x1800 + ((size_t(row + block) + ji) % 4) * 0x400));
                    for (int col = 0; col < 32; ++col)
                        j.weights[block_offset + 2 + size_t(col)] = uint8_t(int((size_t(row) * 11 + size_t(block) * 7 + size_t(col) * 19 + ji * 3) % 127) - 63);
                }
            }
        }
        const size_t input_size = s.x_offset + size_t(s.columns - 1) * j.xcs + size_t(s.k) + suffix;
        inputs.at(s.input).resize(std::max(inputs.at(s.input).size(), input_size));
        data.push_back(std::move(j));
    }
    for (size_t group = 0; group < inputs.size(); ++group) {
        auto& input = inputs[group];
        for (size_t i = 0; i < input.size(); ++i)
            input[i] = i < 32 ? 0.f : float(int((i * 29 + group * 43) % 257) - 128) / 128.f;
        // A complete zero block and positive/negative halfway values exercise both quantizer branches.
        if (input.size() >= 64) {
            input[32] = 127.f / 64.f;
            for (size_t i = 33; i < 64; ++i) input[i] = (float(int(i) - 48) + .5f) / 64.f;
        }
        inputs_after[group].resize(input.size());
    }
    for (auto& j : data) {
        const Spec& s = j.spec;
        std::vector<double> reference_input(size_t(s.k), 0.0);
        for (int col = 0; col < s.columns; ++col) {
            const float* x = inputs[s.input].data() + s.x_offset + size_t(col) * j.xcs;
            for (int i = 0; i < s.k; ++i) reference_input[size_t(i)] = x[i];
            if (s.format == gg::kFmtQ8_0) {
                for (int base = 0; base < s.k; base += 32) {
                    float amax = 0.f;
                    for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[base + i]));
                    const float scale = amax / 127.f;
                    for (int i = 0; i < 32; ++i)
                        reference_input[size_t(base) + size_t(i)] = scale == 0.f ? 0.0 : double(round_even(x[base + i] / scale)) * scale;
                }
            }
            for (int row = 0; row < s.n; ++row) {
                double sum = 0.0;
                for (int i = 0; i < s.k; ++i) sum += weight(j, row, i) * reference_input[size_t(i)];
                const size_t out = output_prefix + size_t(col) * j.ycs + size_t(row);
                j.expected[out] = double(s.alpha) * sum + (s.bias ? j.bias[size_t(row)] : 0.f);
                j.active[out] = 1;
            }
        }
    }
    std::array<gg::x3::MvJob, 4> jobs{};
    Memory memory(queue);
    for (size_t group = 0; group < inputs.size(); ++group) if (!inputs[group].empty()) {
        device_inputs[group] = memory.get<float>(inputs[group].size());
        strata::core::copy_async(queue, device_inputs[group], inputs[group].data(), inputs[group].size() * sizeof(float)).wait_and_throw();
    }
    for (size_t ji = 0; ji < data.size(); ++ji) {
        auto& d = data[ji]; const Spec& s = d.spec;
        d.device_weights = memory.get<uint8_t>(d.weights.size());
        d.device_output = memory.get<float>(d.actual.size());
        auto* device_bias = memory.get<float>(d.bias.size());
        strata::core::copy_async(queue, d.device_weights, d.weights.data(), d.weights.size()).wait_and_throw();
        strata::core::copy_async(queue, d.device_output, d.actual.data(), d.actual.size() * sizeof(float)).wait_and_throw();
        strata::core::copy_async(queue, device_bias, d.bias.data(), d.bias.size() * sizeof(float)).wait_and_throw();
        d.matrix.fmt = s.format; d.matrix.k = s.k; d.matrix.n = s.n;
        d.matrix.raw = d.device_weights + s.raw_offset;
        d.matrix.ld = s.default_strides ? 0 : int(s.format == gg::kFmtBF16 ? d.row_bytes / 2 : d.row_bytes);
        auto& j = jobs[ji]; j.m = &d.matrix; j.xf = device_inputs[s.input] + s.x_offset;
        j.y = d.device_output + output_prefix; j.bias = s.bias ? device_bias : nullptr;
        j.alpha = s.alpha; j.ncol = s.columns;
        j.xf_cs = s.default_strides ? 0 : d.xcs; j.y_cs = s.default_strides ? 0 : d.ycs;
    }
    const int errors_before = gg::launch_errors();
    int errors = gg::mv(jobs.data(), int(data.size()), queue) ? 0 : 1;
    for (size_t ji = 0; ji < data.size(); ++ji) {
        auto& d = data[ji];
        strata::core::readback(queue, d.actual.data(), d.device_output, d.actual.size() * sizeof(float));
        double worst = 0.0;
        size_t bad = 0;
        for (size_t i = 0; i < d.actual.size(); ++i) {
            const double error = std::fabs(double(d.actual[i]) - d.expected[i]);
            worst = std::max(worst, error);
            if (!std::isfinite(d.actual[i]) || (d.active[i] ? error > 2e-5 + 2e-5 * std::fabs(d.expected[i]) : d.actual[i] != guard)) ++bad;
        }
        strata::core::readback(queue, d.weights_after.data(), d.device_weights, d.weights.size());
        const bool pass = !bad && d.weights_after == d.weights;
        errors += !pass;
        std::printf("GGUF MV %s job=%zu fmt=%d k=%d n=%d cols=%d: %s max_abs=%.9g bad=%zu\n",
                    name, ji, d.spec.format, d.spec.k, d.spec.n, d.spec.columns, pass ? "PASS" : "FAIL", worst, bad);
    }
    for (size_t group = 0; group < inputs.size(); ++group) if (!inputs[group].empty()) {
        strata::core::readback(queue, inputs_after[group].data(), device_inputs[group], inputs[group].size() * sizeof(float));
        if (inputs_after[group] != inputs[group]) { ++errors; std::puts("GGUF MV changed its input"); }
    }
    errors += gg::launch_errors() != errors_before;
    return errors;
}

int check_empty(sycl::queue& queue) {
    const uint16_t weight_bits = 0x3f80;
    const float input = 1.f;
    std::array<float, 3> output{guard, guard, guard};
    Memory memory(queue);
    auto* w = memory.get<uint16_t>(1);
    auto* x = memory.get<float>(1);
    auto* y = memory.get<float>(output.size());
    strata::core::copy_async(queue, w, &weight_bits, sizeof(weight_bits)).wait_and_throw();
    strata::core::copy_async(queue, x, &input, sizeof(input)).wait_and_throw();
    strata::core::copy_async(queue, y, output.data(), sizeof(output)).wait_and_throw();
    gg::x3::Mat valid{}; valid.fmt = gg::kFmtBF16; valid.k = valid.n = 1; valid.raw = w;
    auto empty = valid; empty.n = 0;
    std::array<gg::x3::MvJob, 2> jobs{};
    for (auto& j : jobs) { j.m = &valid; j.xf = x; j.y = y + 1; }
    const int before = gg::launch_errors();
    // The existing public contract rejects empty inputs, before submitting any job.
    const bool empty_count = !gg::mv(nullptr, 0, queue);
    jobs[1].m = &empty;
    const bool empty_shape = !gg::mv(jobs.data(), int(jobs.size()), queue);
    jobs[1].m = &valid; jobs[1].ncol = 0;
    const bool empty_columns = !gg::mv(jobs.data(), int(jobs.size()), queue);
    strata::core::readback(queue, output.data(), y, sizeof(output));
    const bool pass = empty_count && empty_shape && empty_columns && gg::launch_errors() == before + 3 &&
                      std::all_of(output.begin(), output.end(), [](float value) { return value == guard; });
    std::printf("GGUF MV empty inputs reject before partial submission: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
}  // namespace

int main() try {
    if (sycl::device::get_devices(sycl::info::device_type::gpu).empty()) {
        std::puts("SKIP: no SYCL GPU"); return 77;
    }
    auto& queue = dsv41_test::queue();
    int errors = 0;
    std::vector<Spec> layer{{gg::kFmtQ8_0, 5120, 1280, 1}, {gg::kFmtQ8_0, 5120, 512, 1},
                            {gg::kFmtBF16, 5120, 512, 1}, {gg::kFmtBF16, 5120, 512, 1}};
    for (auto& s : layer) s.default_strides = true;
    errors += run_case(queue, "layer shared x", layer);
    for (size_t i = 0; i < layer.size(); ++i) layer[i].input = i;
    errors += run_case(queue, "layer distinct x", layer);
    for (const int nc : {1, 3, 17}) {
        std::vector<Spec> boundary{{gg::kFmtQ8_0, 96, 17, nc}, {gg::kFmtQ8_0, 96, 33, nc},
                                  {gg::kFmtBF16, 259, 17, nc}, {gg::kFmtBF16, 264, 33, nc}};
        for (size_t i = 0; i < boundary.size(); ++i) {
            auto& s = boundary[i]; s.input = i < 2 ? 0 : 1;
            s.xcs = size_t(s.k) + (i % 2 ? 7 : 3); s.row_padding = 1;
            s.raw_offset = s.format == gg::kFmtBF16 ? 2 : 1; s.x_offset = 1;
            s.alpha = i % 2 ? -.75f : 1.25f; s.bias = i % 2 != 0;
        }
        errors += run_case(queue, "tail/unaligned/shared pointer distinct stride", boundary);
    }
    errors += check_empty(queue);
    std::printf("GGUF BF16/Q8_0 MV: %s (%d errors)\n", errors ? "FAIL" : "PASS", errors);
    return errors ? 1 : 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "GGUF BF16/Q8_0 MV: %s\n", error.what()); return 1;
} catch (...) {
    std::fputs("GGUF BF16/Q8_0 MV: unknown failure\n", stderr); return 1;
}
