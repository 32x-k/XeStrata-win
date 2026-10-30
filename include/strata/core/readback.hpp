// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <sycl/sycl.hpp>
#include <cstddef>
#include <span>

namespace strata::core {

struct ReadbackSpan {
    void* dst;
    const void* src;
    size_t bytes;
};

// Explicit GPU-produced data -> host copies on the producer's in-order queue, outside graph recording.
// Host destinations may be ordinary, registered or USM memory; GPU-produced host USM is also a valid source.
// On Windows HIP these calls wait for the producer before submitting copies. The async forms return the final
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
