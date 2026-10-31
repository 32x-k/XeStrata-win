// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <sycl/sycl.hpp>
#include <cstddef>
#include <span>
#include <vector>

namespace strata::core {

// Copies on the supplied queue, retaining dependencies from other queues. On Windows HIP, kernel-accessible
// USM endpoints use a compute copy to preserve stream order without blocking a CPU service thread. Other host
// memory uses a producer wait before DMA, and cannot be copied while recording a graph. Other backends and Linux
// keep their native memcpy path. The caller owns the buffers until completion; overlapping ranges are invalid.
// Windows HIP requires an in-order queue. A zero-byte copy marks prior queue work and explicit dependencies,
// without touching either buffer. This differs from the explicit readback API's empty no-op below.
sycl::event copy_async(sycl::queue& queue, void* dst, const void* src, size_t bytes,
                       const std::vector<sycl::event>& dependencies = {});
// The same ordering without allocating a completion event. Use a later queue event/drain before consuming data.
void copy_async_no_event(sycl::queue& queue, void* dst, const void* src, size_t bytes);

// Copy-only host registration. Use these together: Windows HIP reports imported host memory as host USM too,
// so the copy helper tracks these ranges and keeps them out of kernel copies. Registration does not grant
// kernel access. Ranges must not overlap, as required by the native registration API. The caller must finish all
// uses before releasing the registration or its storage. Register through these wrappers before passing it here.
void prepare_for_device_copy(const void* ptr, size_t bytes, const sycl::context& context);
void release_from_device_copy(const void* ptr, const sycl::context& context);

struct ReadbackSpan {
    void* dst;
    const void* src;
    size_t bytes;
};

// Explicit GPU-produced data -> host copies on the producer's in-order queue, outside graph recording.
// Host destinations may be ordinary, registered or USM memory; GPU-produced host USM is also a valid source.
// On Windows HIP these use compute copies where possible, otherwise a wait before DMA. The async forms return the final
// copy event; the caller must keep both buffers alive and wait before reading or reusing them. Empty copies are
// no-ops and return a default event. Queues owned by Runtime retain its wait watchdog; other queues use SYCL waits.
sycl::event readback_async(sycl::queue& producer, void* dst, const void* src, size_t bytes);
sycl::event readback_async(sycl::queue& producer, std::span<const ReadbackSpan> copies);
void readback(sycl::queue& producer, void* dst, const void* src, size_t bytes);
void readback(sycl::queue& producer, std::span<const ReadbackSpan> copies);

// Pitches and width are bytes. Empty rectangles are no-ops; nonempty rows must fit both pitches and size_t.
sycl::event readback2d_async(sycl::queue& producer, void* dst, size_t dpitch, const void* src, size_t spitch,
                            size_t width, size_t height);
void readback2d(sycl::queue& producer, void* dst, size_t dpitch, const void* src, size_t spitch,
                size_t width, size_t height);

}  // namespace strata::core
