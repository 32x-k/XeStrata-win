// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "strata/core/readback.hpp"
#include "strata/core/runtime.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace strata::core {
namespace {

void wait_readback(sycl::queue& q, sycl::event event, const char* what) {
    if (auto* owner = Runtime::owner_of_stream(&q)) owner->wait(event, what);
    else event.wait_and_throw();
}

void prepare_readback(sycl::queue& q) {
    if (!q.has_property<sycl::property::queue::in_order>())
        throw std::invalid_argument("readback requires an in-order producer queue");
    if (q.ext_oneapi_get_state() == sycl::ext::oneapi::experimental::queue_state::recording)
        throw std::invalid_argument("readback cannot run during graph recording");
#if defined(_WIN32)
    // Windows HIP's DMA copy can observe stale data after a kernel despite an in-order queue. Complete the
    // producer before submitting D2H; waiting after the copy cannot repair the data it already transferred.
    // See https://github.com/ROCm/rocm-systems/issues/12571 and https://github.com/ROCm/rocm-systems/pull/12952.
    if (q.get_backend() == sycl::backend::ext_oneapi_hip)
        wait_readback(q, q.ext_oneapi_submit_barrier(), "readback producer");
#endif
}

void check_buffers(void* dst, const void* src) {
    if (!dst || !src) throw std::invalid_argument("readback requires non-null buffers");
}

bool has_bytes(std::span<const ReadbackSpan> copies) {
    return std::any_of(copies.begin(), copies.end(), [](const ReadbackSpan& c) { return c.bytes != 0; });
}

}  // namespace

sycl::event readback_async(sycl::queue& q, void* dst, const void* src, size_t bytes) {
    if (bytes == 0) return {};
    check_buffers(dst, src);
    prepare_readback(q);
    return q.memcpy(dst, src, bytes);
}

sycl::event readback_async(sycl::queue& q, std::span<const ReadbackSpan> copies) {
    if (!has_bytes(copies)) return {};
    for (const auto& c : copies) if (c.bytes) check_buffers(c.dst, c.src);
    prepare_readback(q);
    sycl::event last;
    for (const auto& c : copies) if (c.bytes) last = q.memcpy(c.dst, c.src, c.bytes);
    return last;
}

void readback(sycl::queue& q, void* dst, const void* src, size_t bytes) {
    if (bytes) wait_readback(q, readback_async(q, dst, src, bytes), "readback copy");
}

void readback(sycl::queue& q, std::span<const ReadbackSpan> copies) {
    if (has_bytes(copies)) wait_readback(q, readback_async(q, copies), "readback copies");
}

sycl::event readback2d_async(sycl::queue& q, void* dst, size_t dpitch, const void* src, size_t spitch,
                            size_t width, size_t height) {
    if (width == 0 || height == 0) return {};
    check_buffers(dst, src);
    if (width > dpitch || width > spitch)
        throw std::invalid_argument("readback2d width exceeds a pitch");
    const size_t room = std::numeric_limits<size_t>::max() - width;
    if (height - 1 > room / dpitch || height - 1 > room / spitch)
        throw std::invalid_argument("readback2d offset overflows size_t");
    prepare_readback(q);
    auto* d = static_cast<std::byte*>(dst);
    const auto* s = static_cast<const std::byte*>(src);
    sycl::event last;
    for (size_t row = 0; row < height; ++row) last = q.memcpy(d + row * dpitch, s + row * spitch, width);
    return last;
}

void readback2d(sycl::queue& q, void* dst, size_t dpitch, const void* src, size_t spitch,
                size_t width, size_t height) {
    if (width && height)
        wait_readback(q, readback2d_async(q, dst, dpitch, src, spitch, width, height), "readback2d copies");
}

}  // namespace strata::core
