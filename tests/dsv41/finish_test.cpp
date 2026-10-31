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
constexpr size_t prefix = 5, suffix = 11;
constexpr float float_guard = 12345.f;
constexpr uint16_t bf_guard = 0x4e55;
constexpr uint8_t byte_guard = 0xa5;

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
    template<class T> T* upload(const std::vector<T>& values) {
        if (count == pointers.size()) throw std::logic_error("finish test allocation limit");
        auto* p = sycl::malloc_device<T>(values.size(), queue);
        if (!p) throw std::bad_alloc();
        pointers[count++] = p;
        strata::core::copy_async(queue, p, values.data(), values.size() * sizeof(T));
        return p;
    }
};
template<class T> void read(sycl::queue& queue, std::vector<T>& out, const T* in) {
    strata::core::readback(queue, out.data(), in, out.size() * sizeof(T));
}
template<class T> int equal(const std::vector<T>& actual, const std::vector<T>& expected, const char* name) {
    int bad = 0;
    for (size_t i = 0; i < actual.size(); ++i) if (actual[i] != expected[i]) {
        if (bad++ == 0) std::printf("%s mismatch at %zu: %.12g expected %.12g\n", name, i, double(actual[i]), double(expected[i]));
    }
    return bad;
}

float from_bf16(uint16_t code) { return std::bit_cast<float>(uint32_t(code) << 16); }
uint16_t to_bf16(float value) {
    // Compare the two adjacent finite representable magnitudes, including ties.
    const uint32_t bits = std::bit_cast<uint32_t>(value);
    const uint16_t sign = uint16_t((bits >> 16) & 0x8000U), low = uint16_t((bits & 0x7fffffffU) >> 16);
    const double magnitude = std::abs(double(value));
    const double dl = magnitude - from_bf16(low), dh = from_bf16(uint16_t(low + 1)) - magnitude;
    return uint16_t(sign | uint16_t(low + unsigned(dh < dl || (dh == dl && (low & 1U)))));
}
float fp8(unsigned code) {
    const unsigned exp = (code >> 3) & 15U, mantissa = code & 7U;
    return exp ? std::ldexp(float(8 + mantissa), int(exp) - 10) : float(mantissa) / 512.f;
}
float fp4(unsigned code) {
    constexpr std::array<float, 8> table{0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    return table[code];
}
uint8_t nearest_code(float value, bool four_bit) {
    const unsigned count = four_bit ? 8U : 127U;
    unsigned best = 0;
    double error = std::numeric_limits<double>::infinity();
    for (unsigned i = 0; i < count; ++i) {
        const double distance = std::abs(std::abs(double(value)) - (four_bit ? fp4(i) : fp8(i)));
        if (distance < error || (distance == error && !(i & 1U))) { error = distance; best = i; }
    }
    return uint8_t(best | (std::signbit(value) ? (four_bit ? 8U : 128U) : 0U));
}
float decode_code(uint8_t code, bool four_bit) {
    const unsigned mask = four_bit ? 7U : 127U, sign = four_bit ? 8U : 128U;
    const float value = four_bit ? fp4(code & mask) : fp8(code & mask);
    return code & sign ? -value : value;
}
float power2_ceil(float x) {
    int exponent = 0;
    const double mantissa = std::frexp(x, &exponent);
    return float(std::ldexp(1.f, exponent - int(mantissa == .5f)));
}
size_t row_bytes(int quant, int length) {
    const size_t payload = quant == 1 ? size_t(length) + size_t(length) / 32
                                    : size_t(length) / 2 + size_t(length) / (quant == 2 ? 16 : 32);
    return (payload + 15) / 16 * 16;
}

// CPU FP32 order model of the fixed upstream 256-item launch: item-strided
// accumulation, 32-lane XOR tree, then XOR of eight warp totals. This is separate
// from the double mathematical reference used for unquantized output accuracy.
float ordered_squares(const std::vector<float>& row, bool portable) {
    if (portable) {
        float sum = 0;
        for (float v : row) sum += v * v;
        return sum;
    }
    std::array<float, 256> partial{};
    for (size_t tid = 0; tid < partial.size(); ++tid)
        for (size_t i = tid; i < row.size(); i += partial.size()) partial[tid] += row[i] * row[i];
    for (size_t step = 16; step; step /= 2) {
        const auto old = partial;
        for (size_t tid = 0; tid < partial.size(); ++tid) partial[tid] = old[tid] + old[tid ^ step];
    }
    std::array<float, 32> warps{};
    for (size_t i = 0; i < 8; ++i) warps[i] = partial[i * 32];
    for (size_t step = 16; step; step /= 2) {
        const auto old = warps;
        for (size_t i = 0; i < warps.size(); ++i) warps[i] = old[i] + old[i ^ step];
    }
    return warps[0];
}

struct Spec {
    int length = 0, vectors = 0, quant = 0, output = 0, rotary = 0;
    bool norm = false, inverse = false, zeros = false;
};
struct Reference {
    std::vector<float> fp32;
    std::vector<double> mathematical;
    std::vector<uint8_t> packed;
};

Reference reference(const std::vector<float>& row, const std::vector<float>& weights,
                    const std::vector<float>& frequencies, const Spec& spec, int vector, bool portable) {
    Reference result{row, std::vector<double>(row.begin(), row.end()), {}};
    constexpr float eps = .125f;
    if (spec.norm) {
        const float inverse = 1.f / std::sqrt(ordered_squares(row, portable) / float(spec.length) + eps);
        double squares = 0;
        for (double x : result.mathematical) squares += x * x;
        const double inverse_double = 1.0 / std::sqrt(squares / spec.length + double(eps));
        for (int e = 0; e < spec.length; ++e) {
            result.fp32[size_t(e)] = result.fp32[size_t(e)] * inverse * weights[prefix + size_t(e)];
            result.mathematical[size_t(e)] *= inverse_double * weights[prefix + size_t(e)];
        }
    }
    const int position = 3 + (vector / 2) * 5;
    for (int e = spec.length - spec.rotary; e < spec.length; e += 2) {
        const size_t fi = prefix + size_t(e - spec.length + spec.rotary) / 2;
        const float angle = float(position) * frequencies[fi];
        const float c = std::cos(angle), s = (spec.inverse ? -1.f : 1.f) * std::sin(angle);
        const float a = result.fp32[size_t(e)], b = result.fp32[size_t(e) + 1];
        result.fp32[size_t(e)] = a * c - b * s; result.fp32[size_t(e) + 1] = a * s + b * c;
        const double ad = result.mathematical[size_t(e)], bd = result.mathematical[size_t(e) + 1];
        const double cd = std::cos(double(angle)), sd = (spec.inverse ? -1.0 : 1.0) * std::sin(double(angle));
        result.mathematical[size_t(e)] = ad * cd - bd * sd;
        result.mathematical[size_t(e) + 1] = ad * sd + bd * cd;
    }
    if (!spec.quant) return result;
    result.packed.assign(row_bytes(spec.quant, spec.length), byte_guard);
    const int block = spec.quant == 2 ? 16 : 32;
    for (int start = 0; start < spec.length; start += block) {
        const int stop = std::min(start + block, spec.length);
        float maximum = 0;
        for (int e = start; e < stop; ++e) maximum = std::max(maximum, std::abs(result.fp32[size_t(e)]));
        float scale = 0;
        if (spec.quant == 1) scale = power2_ceil(portable ? std::max(maximum, 1e-4f) / 448.f
                                                       : std::max(maximum, 1e-4f) * (1.f / 448.f));
        else if (spec.quant == 2) scale = decode_code(nearest_code(std::max(maximum, 6.f / 512.f) / 6.f, false), false);
        else {
            const float maximum_clamped = std::max(maximum, 6.f * std::numeric_limits<float>::min());
            // The retained portable path uses division, while the port uses the
            // upstream reciprocal product. Keep their boundary contracts separate.
            scale = power2_ceil(portable ? maximum_clamped / 6.f : maximum_clamped * (1.f / 6.f));
        }
        for (int e = start; e < stop; ++e) {
            const uint8_t code = nearest_code(result.fp32[size_t(e)] / scale, spec.quant != 1);
            result.fp32[size_t(e)] = decode_code(code, spec.quant != 1) * scale;
            if (spec.length % 32 == 0) {
                if (spec.quant == 1) result.packed[size_t(e)] = code;
                else if (!(e & 1)) result.packed[size_t(e) / 2] = code;
                else result.packed[size_t(e) / 2] |= uint8_t(code << 4);
            }
        }
        if (spec.length % 32 == 0) {
            const size_t at = size_t(spec.quant == 1 ? spec.length : spec.length / 2) + size_t(start / block);
            result.packed[at] = spec.quant == 2 ? nearest_code(scale, false)
                                               : uint8_t(std::ilogb(scale) + 127);
        }
    }
    return result;
}

bool portable_shape(sycl::queue& queue, int length) {
    const auto d = queue.get_device();
    const auto widths = d.get_info<sycl::info::device::sub_group_sizes>();
    const auto local = d.get_info<sycl::info::device::local_mem_size>();
    const bool sg = (local >= 40 && std::find(widths.begin(), widths.end(), 32) != widths.end()) ||
                    (local >= 1060 && std::find(widths.begin(), widths.end(), 16) != widths.end());
    return length > 8192 || !sg || d.get_info<sycl::info::device::max_work_group_size>() < 256 ||
           d.get_info<sycl::info::device::max_work_item_sizes<1>>()[0] < 256;
}

int finish_case(sycl::queue& queue, const Spec& spec) {
    const size_t stride = size_t(spec.length) + 7, dst_stride = size_t(spec.length) + 9;
    const size_t packed_stride = row_bytes(std::max(1, spec.quant), spec.length) + 13;
    std::vector<float> x(prefix + size_t(spec.vectors) * stride + suffix, float_guard), actual(x.size()), expected = x;
    std::vector<float> w(prefix + size_t(spec.length) + suffix, float_guard), w_after(w.size());
    std::vector<float> freq(prefix + size_t(spec.rotary) / 2 + suffix, float_guard), freq_after(freq.size());
    std::vector<uint16_t> dst(prefix + size_t(spec.vectors) * dst_stride + suffix, bf_guard), dst_after(dst.size()), dst_expected = dst;
    std::vector<uint8_t> packed(prefix + size_t(spec.vectors) * packed_stride + suffix, byte_guard), packed_after(packed.size()), packed_expected = packed;
    std::vector<Reference> references;
    references.reserve(size_t(spec.vectors));
    for (int e = 0; e < spec.length; ++e) w[prefix + size_t(e)] = float(12 + e % 9) / 16.f;
    for (int i = 0; i < spec.rotary / 2; ++i) freq[prefix + size_t(i)] = float(i + 1) / 1024.f;
    for (int v = 0; v < spec.vectors; ++v) {
        std::vector<float> row(size_t(spec.length), 0);
        for (int e = 0; e < spec.length; ++e) {
            // Dyadic inputs keep multiplication rounding distinct from reduction order.
            row[size_t(e)] = spec.zeros || ((e / 32 + v) % 7 == 0) ? 0.f : float((e * 17 + v * 23) % 63 - 31) / 32.f;
            x[prefix + size_t(v) * stride + e] = row[size_t(e)];
        }
        references.push_back(reference(row, w, freq, spec, v, portable_shape(queue, spec.length)));
    }
    expected = x;
    for (int v = 0; v < spec.vectors; ++v) {
        const auto& ref = references[size_t(v)];
        for (int e = 0; e < spec.length; ++e) {
            if (spec.output == 0) expected[prefix + size_t(v) * stride + e] = ref.fp32[size_t(e)];
            if (spec.output == 1 || spec.output == 3) dst_expected[prefix + size_t(v) * dst_stride + e] = to_bf16(ref.fp32[size_t(e)]);
        }
        if (spec.output >= 2) std::copy(ref.packed.begin(), ref.packed.end(), packed_expected.begin() + std::ptrdiff_t(prefix + size_t(v) * packed_stride));
    }
    Memory memory(queue); // All host buffers, including comparison destinations, outlive queue drain.
    float* dx = memory.upload(x); float* dw = memory.upload(w); float* df = memory.upload(freq);
    uint16_t* dd = memory.upload(dst); uint8_t* dp = memory.upload(packed);
    ds::Finish f;
    f.w = spec.norm ? dw + prefix : nullptr; f.eps = .125f; f.rd = spec.rotary;
    f.pos = 3; f.vpp = 2; f.pos_step = 5; f.inverse = int(spec.inverse); f.freqs = df + prefix;
    f.quant = spec.quant;
    if (spec.output == 1 || spec.output == 3) { f.dst = dd + prefix; f.dst_stride = static_cast<long long>(dst_stride); }
    if (spec.output >= 2) { f.pdst = dp + prefix; f.pdst_stride = static_cast<long long>(packed_stride); }
    ds::finish(dx + prefix, spec.vectors, spec.length, static_cast<long long>(stride), f, queue);
    read(queue, actual, dx); read(queue, dst_after, dd); read(queue, packed_after, dp);
    read(queue, w_after, dw); read(queue, freq_after, df);
    int bad = equal(w_after, w, "finish weights/guard") + equal(freq_after, freq, "finish frequencies/guard");
    bad += equal(packed_after, packed_expected, "finish packed/guard");
    for (size_t i = 0; i < actual.size(); ++i) {
        const bool in_row = i >= prefix && (i - prefix) / stride < size_t(spec.vectors) && (i - prefix) % stride < size_t(spec.length);
        const bool write = in_row && spec.output == 0;
        const double ref = write && !spec.quant ? references[(i - prefix) / stride].mathematical[(i - prefix) % stride] : double(expected[i]);
        const double tolerance = write && !spec.quant ? 3e-5 * std::max(1.0, std::abs(ref)) : 0.0;
        if (!std::isfinite(actual[i]) || std::abs(double(actual[i]) - ref) > tolerance) {
            if (bad++ == 0) std::printf("finish F32/input mismatch at %zu: %.9g expected %.12g\n", i, actual[i], ref);
        }
    }
    for (size_t i = 0; i < dst_after.size(); ++i) {
        const bool in_row = i >= prefix && (i - prefix) / dst_stride < size_t(spec.vectors) && (i - prefix) % dst_stride < size_t(spec.length);
        const bool write = in_row && (spec.output == 1 || spec.output == 3);
        bool match = dst_after[i] == dst_expected[i];
        if (!match && write && !spec.quant) {
            const double ref = references[(i - prefix) / dst_stride].mathematical[(i - prefix) % dst_stride];
            const double half_ulp = ref != 0 ? std::ldexp(1.0, std::ilogb(std::abs(ref)) - 8) : 0;
            match = std::isfinite(from_bf16(dst_after[i])) && std::abs(double(from_bf16(dst_after[i])) - ref) <= half_ulp + 3e-5 * std::max(1.0, std::abs(ref));
        }
        if (!match && bad++ == 0) std::printf("finish BF16/guard mismatch at %zu: %04x expected %04x\n", i, unsigned(dst_after[i]), unsigned(dst_expected[i]));
    }
    std::printf("finish len=%d vectors=%d quant=%d dst=%d rd=%d norm=%d inverse=%d zero=%d: %s\n",
                spec.length, spec.vectors, spec.quant, spec.output, spec.rotary, int(spec.norm), int(spec.inverse), int(spec.zeros), bad ? "FAIL" : "PASS");
    return bad;
}

int ring_case(sycl::queue& queue, int length, int slots, int tokens, int64_t position, bool backup_enabled) {
    const size_t rb = row_bytes(1, length);
    std::vector<uint16_t> ext(prefix + size_t(slots + tokens) * length + suffix, bf_guard), ext_after(ext.size());
    std::vector<uint8_t> ring(prefix + size_t(slots) * rb + suffix, byte_guard), ring_after(ring.size());
    std::vector<uint8_t> backup(prefix + size_t(tokens) * rb + suffix, byte_guard), backup_after(backup.size()), want_backup = backup;
    for (int r = 0; r < slots; ++r) for (size_t j = 0; j < rb; ++j) ring[prefix + size_t(r) * rb + j] = uint8_t((j * 37 + size_t(r) * 19) % 251);
    auto want_ring = ring;
    for (int t = 0; t < tokens; ++t) for (int e = 0; e < length; ++e) {
        const int block = e / 32;
        const unsigned code = unsigned((e * 11 + t * 7) % 126);
        const float magnitude = (block + t) % 5 ? fp8(code) : 0.f;
        const float value = std::ldexp((e & 1) ? -magnitude : magnitude, (block + t) % 7 - 3);
        ext[prefix + size_t(slots + t) * length + e] = to_bf16(value);
    }
    for (int t = std::max(0, tokens - slots); t < tokens; ++t) {
        const size_t dest = prefix + size_t((position + t) % slots) * rb;
        if (backup_enabled) std::copy_n(ring.begin() + std::ptrdiff_t(dest), rb, want_backup.begin() + std::ptrdiff_t(prefix + size_t(t) * rb));
        for (int start = 0; start < length; start += 32) {
            float maximum = 0;
            for (int j = 0; j < 32; ++j) maximum = std::max(maximum, std::abs(from_bf16(ext[prefix + size_t(slots + t) * length + start + j])));
            // Minimal power-of-two scale whose finite E4M3 range contains the block.
            float scale = 1.f;
            if (maximum != 0) {
                while (448.f * scale < maximum) scale *= 2.f;
                while (448.f * (scale * .5f) >= maximum) scale *= .5f;
            }
            for (int j = 0; j < 32; ++j) want_ring[dest + start + j] = nearest_code(from_bf16(ext[prefix + size_t(slots + t) * length + start + j]) / scale, false);
            want_ring[dest + length + start / 32] = uint8_t(std::ilogb(scale) + 127);
        }
    }
    Memory memory(queue);
    uint16_t* de = memory.upload(ext); uint8_t* dr = memory.upload(ring); uint8_t* db = memory.upload(backup);
    ds::ext_to_ring(de + prefix, dr + prefix, position, slots, tokens, length, queue, backup_enabled ? db + prefix : nullptr);
    read(queue, ring_after, dr); read(queue, backup_after, db); read(queue, ext_after, de);
    const int bad = equal(ring_after, want_ring, "ring bytes/padding") + equal(backup_after, want_backup, "ring backup/skipped rows") + equal(ext_after, ext, "ring input/guard");
    std::printf("ext_to_ring dim=%d n=%d T=%d p0=%lld backup=%d: %s\n", length, slots, tokens, static_cast<long long>(position), int(backup_enabled), bad ? "FAIL" : "PASS");
    return bad;
}
} // namespace

int main() try {
    if (sycl::device::get_devices(sycl::info::device_type::gpu).empty()) { std::puts("finish test: no GPU, skipped"); return 77; }
    auto& queue = dsv41_test::queue();
    int bad = 0;
    for (int length : {128, 512, 1280, 5120}) {
        bad += finish_case(queue, {length, 3, 0, 0, 64, true, false, false});
        bad += finish_case(queue, {length, 3, 0, 1, 64, false, true, false});
        for (int quant : {1, 2, 3}) {
            bad += finish_case(queue, {length, 3, quant, 2, 64, true, quant == 3, false});
            bad += finish_case(queue, {length, 1, quant, 0, 0, false, false, false});
        }
    }
    bad += finish_case(queue, {512, 3, 2, 3, 64, true, false, false}); // Both packed and BF16 destinations.
    bad += finish_case(queue, {512, 2, 1, 1, 64, false, false, false});
    for (int quant : {0, 1, 2, 3}) {
        bad += finish_case(queue, {33, 2, quant, 0, 0, true, false, false});
        bad += finish_case(queue, {128, 1, quant, quant ? 2 : 1, 0, true, false, true});
    }
    bad += finish_case(queue, {1, 1, 0, 0, 0, false, false, false});
    bad += finish_case(queue, {258, 3, 0, 1, 66, true, true, false});
    bad += finish_case(queue, {8224, 1, 0, 0, 64, true, false, false});
    bad += finish_case(queue, {8224, 1, 3, 2, 64, false, true, false});
    bad += finish_case(queue, {512, 0, 2, 3, 64, true, false, false});
    for (int length : {128, 512, 1280, 5120}) {
        bad += ring_case(queue, length, 7, 3, 6, true);
        bad += ring_case(queue, length, 3, 8, 11, false);
    }
    bad += ring_case(queue, 512, 3, 8, 11, true);
    bad += ring_case(queue, 512, 1, 5, 17, true);
    bad += ring_case(queue, 32, 4, 0, 3, true);
    std::printf("finish/ring independent references: %s (%d mismatches)\n", bad ? "FAIL" : "PASS", bad);
    return bad ? 1 : 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "finish/ring test: %s\n", error.what()); return 1;
} catch (...) {
    std::fputs("finish/ring test: unknown exception\n", stderr); return 1;
}
