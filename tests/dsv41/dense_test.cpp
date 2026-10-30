// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "ds4_runtime.hpp"
#include "ds4_sycl.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {
namespace rt = strata::ds4::rt;

void require(rt::Result result) {
    if (result != rt::Result::success) throw std::runtime_error(rt::error_string(result));
}

struct Buffer {
    void* data = nullptr;
    explicit Buffer(size_t bytes) { require(rt::alloc_device(&data, bytes)); }
    ~Buffer() { rt::free(data); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

struct Stream {
    rt::Queue queue;
    Stream() { require(rt::stream_create(&queue)); }
    ~Stream() { rt::stream_destroy(queue); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
};

template <class E> E input(float f) {
    if constexpr (std::is_same_v<E, uint16_t>) return strata::ds4::sycl_detail::f16_bits(f);
    else return f;
}
template <class E> float value(E e) {
    if constexpr (std::is_same_v<E, uint16_t>) return strata::ds4::sycl_detail::f16_value(e);
    else return e;
}

template <class E>
rt::Result project(rt::Dense& dense, int rows, int n, int k, float alpha, const E* w, int ldw,
                    const E* x, int ldx, float beta, float* y, int ldy) {
    if constexpr (std::is_same_v<E, uint16_t>)
        return rt::dense_f16(dense, n, rows, k, alpha, w, ldw, x, ldx, beta, y, ldy);
    else return rt::dense_f32(dense, n, rows, k, alpha, w, ldw, x, ldx, beta, y, ldy);
}

template <class E>
bool check_product(rt::Dense& dense, rt::Queue queue, bool padded, int k, float alpha, float beta) {
    constexpr int rows = 3, n = 5;
    constexpr size_t offset = 2;
    constexpr float guard = 12345.0f;
    const int ldx = k + (padded ? 3 : 0), ldw = k + (padded ? 5 : 0), ldy = n + 4;
    const bool empty_product = k == 0 || alpha == 0.0f;
    std::vector<E> x((size_t) rows * ldx, input<E>(std::numeric_limits<float>::quiet_NaN()));
    std::vector<E> w((size_t) n * ldw, input<E>(std::numeric_limits<float>::quiet_NaN()));
    for (int t = 0; t < rows; ++t)
        for (int j = 0; j < k; ++j) x[(size_t) t * ldx + j] = input<E>((float) (t - j + 2) / 4.0f);
    for (int c = 0; c < n; ++c)
        for (int j = 0; j < k; ++j) w[(size_t) c * ldw + j] = input<E>((float) (c + j - 3) / 8.0f);
    std::vector<float> y(offset + (size_t) rows * ldy + 3, guard), expected = y;
    for (int t = 0; t < rows; ++t) {
        for (int c = 0; c < n; ++c) {
            const size_t at = offset + (size_t) t * ldy + c;
            y[at] = beta == 0.0f ? std::numeric_limits<float>::quiet_NaN() : (float) (t + c - 2) / 2.0f;
            float sum = 0.0f;
            if (!empty_product)
                for (int j = 0; j < k; ++j)
                    sum = std::fma(value(x[(size_t) t * ldx + j]), value(w[(size_t) c * ldw + j]), sum);
            expected[at] = beta == 0.0f ? alpha * sum : std::fma(beta, y[at], alpha * sum);
        }
    }
    Buffer dx(std::max<size_t>(1, x.size() * sizeof(E))), dw(std::max<size_t>(1, w.size() * sizeof(E)));
    Buffer dy(y.size() * sizeof(float));
    if (!empty_product) {
        require(rt::copy_async(dx.data, x.data(), x.size() * sizeof(E), queue));
        require(rt::copy_async(dw.data, w.data(), w.size() * sizeof(E), queue));
    }
    require(rt::copy_async(dy.data, y.data(), y.size() * sizeof(float), queue));
    require(project<E>(dense, rows, n, k, alpha, empty_product ? nullptr : static_cast<const E*>(dw.data), ldw,
                        empty_product ? nullptr : static_cast<const E*>(dx.data), ldx, beta,
                        static_cast<float*>(dy.data) + offset, ldy));
    require(rt::copy_async(y.data(), dy.data, y.size() * sizeof(float), queue));
    require(rt::stream_sync(queue));
    for (size_t i = 0; i < y.size(); ++i) {
        const bool padding = i < offset || i >= offset + (size_t) rows * ldy || (i - offset) % ldy >= n;
        const bool ok = padding ? y[i] == guard :
            std::isfinite(y[i]) && std::abs(y[i] - expected[i]) <= 1e-5f * std::max(1.0f, std::abs(expected[i]));
        if (!ok) {
            std::fprintf(stderr, "%s padded=%d K=%d alpha=%g beta=%g at %zu: %g != %g%s\n",
                         std::is_same_v<E, uint16_t> ? "FP16" : "FP32", padded ? 1 : 0, k, alpha, beta,
                         i, y[i], expected[i], padding ? " (output guard)" : "");
            return false;
        }
    }
    return true;
}

template <class E> int check_type(rt::Dense& dense, rt::Queue queue) {
    int failed = 0;
    failed += !check_product<E>(dense, queue, false, 7, 1.0f, 0.0f);
    failed += !check_product<E>(dense, queue, true, 7, 1.0f, 0.0f);
    failed += !check_product<E>(dense, queue, true, 7, 1.25f, -0.5f);
    failed += !check_product<E>(dense, queue, false, 0, 1.0f, 0.0f);
    for (float beta : {0.0f, 0.5f, 1.0f}) {
        failed += !check_product<E>(dense, queue, true, 0, 1.0f, beta);
        failed += !check_product<E>(dense, queue, true, 7, 0.0f, beta);
    }
    // Invalid strides must fail before submitting work and retain the adapter's failure reason.
    const auto bad = project<E>(dense, 3, 5, 7, 1.0f, nullptr, 7, nullptr, 6, 0.0f, nullptr, 5);
    if (bad != rt::Result::error || std::string(rt::error_string(bad)).find("stride") == std::string::npos) {
        std::fputs("invalid stride was not reported by the dense adapter\n", stderr);
        ++failed;
    }
    return failed;
}
}  // namespace

int main() {
    try {
        strata::core::restrict_hip_to_selector();
        strata::core::no_hip_resource_cache();
        if (!strata::core::Runtime::count()) {
            std::puts("SKIP: no SYCL GPU");
            return 77;
        }
        const auto& device = strata::core::Runtime::get().device();
        if (!device.has(sycl::aspect::fp16) || !device.has(sycl::aspect::usm_device_allocations)) {
            std::puts("SKIP: dense parity requires FP16 and device USM");
            return 77;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "SKIP: no usable SYCL GPU: %s\n", e.what());
        return 77;
    }
    try {
        Stream stream;
        rt::Dense dense;
        require(rt::dense_create(&dense));
        require(rt::dense_stream(dense, stream.queue));
        auto& runtime = strata::core::Runtime::get();
        if (&static_cast<sycl::queue&>(stream.queue) != &runtime.stream(stream.queue.handle))
            throw std::runtime_error("dense adapter and runtime selected different queues");
        const int failed = check_type<uint16_t>(dense, stream.queue) + check_type<float>(dense, stream.queue);
        std::printf("dsv41 strided dense parity: %s (%d failures)\n", failed ? "FAIL" : "PASS", failed);
        return failed ? 1 : 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "dsv41 strided dense parity: %s\n", e.what());
        return 1;
    }
}
