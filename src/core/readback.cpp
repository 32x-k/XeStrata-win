// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "strata/core/readback.hpp"
#include "strata/core/runtime.hpp"
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <mutex>
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
}

void check_buffers(void* dst, const void* src) {
    if (!dst || !src) throw std::invalid_argument("readback requires non-null buffers");
}

bool has_bytes(std::span<const ReadbackSpan> copies) {
    return std::any_of(copies.begin(), copies.end(), [](const ReadbackSpan& c) { return c.bytes != 0; });
}

#if defined(_WIN32)
struct CopyRegistration { size_t bytes; size_t references; };
std::mutex registration_mutex;
std::map<uintptr_t, CopyRegistration>& copy_registrations() {
    static std::map<uintptr_t, CopyRegistration> registrations;
    return registrations;
}

bool copy_registered(const void* p) {
    const auto address = reinterpret_cast<uintptr_t>(p);
    std::lock_guard lock(registration_mutex);
    const auto& registrations = copy_registrations();
    auto it = registrations.upper_bound(address);
    if (it == registrations.begin()) return false;
    --it;
    return address - it->first < it->second.bytes;
}

bool kernel_accessible(const void* p, const sycl::queue& q) {
    // HIP's allocation query cannot distinguish malloc_host from a copy-only imported host range.
    if (copy_registered(p)) return false;
    const auto type = sycl::get_pointer_type(p, q.get_context());
    const auto device = q.get_device();
    if (type == sycl::usm::alloc::host) return device.has(sycl::aspect::usm_host_allocations);
    if (type == sycl::usm::alloc::device)
        return device.has(sycl::aspect::usm_device_allocations) &&
               sycl::get_pointer_device(p, q.get_context()) == device;
    if (type == sycl::usm::alloc::shared)
        return device.has(sycl::aspect::usm_shared_allocations) &&
               sycl::get_pointer_device(p, q.get_context()) == device;
    // Registering ordinary host memory for DMA does not grant SYCL kernel access to it.
    return false;
}

bool prepare_copy(sycl::queue& q, void* dst, const void* src, const std::vector<sycl::event>& dependencies) {
    if (q.get_backend() != sycl::backend::ext_oneapi_hip) return false;
    if (!q.has_property<sycl::property::queue::in_order>())
        throw std::invalid_argument("ordered copy requires an in-order queue on Windows HIP");
    check_buffers(dst, src);
    // PAL may lose a compute -> SDMA dependency after an event record, for H2D as well as D2H. A compute copy
    // retains the GPU ordering and lets the issuer return to CPU service work on which earlier kernels depend.
    // See https://github.com/ROCm/rocm-systems/issues/12571 and https://github.com/ROCm/rocm-systems/pull/12952.
    if (kernel_accessible(dst, q) && kernel_accessible(src, q)) return true;
    if (q.ext_oneapi_get_state() == sycl::ext::oneapi::experimental::queue_state::recording)
        throw std::invalid_argument("Windows HIP graph copies require kernel-accessible USM buffers");
    // Only this queue and the caller's dependencies: draining every runtime queue can deadlock CPU/GPU handoffs.
    // Waiting after DMA cannot repair bytes it transferred before the producer finished.
    wait_readback(q, q.ext_oneapi_submit_barrier(dependencies), "copy producer");
    return false;
}

void compute_copy(sycl::handler& h, void* dst, const void* src, size_t bytes) {
    auto* d = static_cast<unsigned char*>(dst);
    const auto* s = static_cast<const unsigned char*>(src);
    // Match the object-representation vector's alignment at both ends. Only complete vectors are loaded;
    // unequal alignments, the leading bytes and the tail retain byte access. The device chooses the WG size.
    constexpr size_t width = sizeof(sycl::vec<unsigned char, 16>);
    const auto da = reinterpret_cast<uintptr_t>(d), sa = reinterpret_cast<uintptr_t>(s);
    const size_t prefix = (da % width == sa % width) ? std::min(bytes, (width - sa % width) % width) : bytes;
    const size_t blocks = (bytes - prefix) / width, tail = bytes - prefix - blocks * width;
    h.parallel_for(sycl::range<1>(blocks + prefix + tail), [=](sycl::id<1> at) {
        const size_t i = at[0];
        if (i < blocks) {
            sycl::vec<unsigned char, 16> value;
            value.load(i, sycl::address_space_cast<sycl::access::address_space::global_space,
                       sycl::access::decorated::no>(s + prefix));
            value.store(i, sycl::address_space_cast<sycl::access::address_space::global_space,
                        sycl::access::decorated::no>(d + prefix));
        } else {
            const size_t j = i - blocks;
            const size_t offset = j < prefix ? j : blocks * width + j;
            d[offset] = s[offset];
        }
    });
}

// Readback batches contain copies only: after one fallback fence they can all use DMA, without adding a host
// round trip per span or row. A batch with entirely accessible endpoints remains asynchronous compute work.
bool native_readback_batch(sycl::queue& q, std::span<const ReadbackSpan> copies) {
    if (q.get_backend() != sycl::backend::ext_oneapi_hip) return true;
    for (const auto& c : copies) {
        if (c.bytes && (!kernel_accessible(c.dst, q) || !kernel_accessible(c.src, q))) {
            (void)prepare_copy(q, c.dst, c.src, {});
            return true;
        }
    }
    return false;
}
#endif

}  // namespace

void prepare_for_device_copy(const void* ptr, size_t bytes, const sycl::context& context) {
    sycl::ext::oneapi::experimental::prepare_for_device_copy(ptr, bytes, context);
#if defined(_WIN32)
    if (context.get_backend() == sycl::backend::ext_oneapi_hip) {
        try {
            std::lock_guard lock(registration_mutex);
            auto& registrations = copy_registrations();
            auto [it, inserted] = registrations.try_emplace(reinterpret_cast<uintptr_t>(ptr), CopyRegistration{bytes, 0});
            (void)inserted;
            it->second.bytes = std::max(it->second.bytes, bytes);
            ++it->second.references;
        } catch (...) {
            sycl::ext::oneapi::experimental::release_from_device_copy(ptr, context);
            throw;
        }
    }
#endif
}

void release_from_device_copy(const void* ptr, const sycl::context& context) {
    sycl::ext::oneapi::experimental::release_from_device_copy(ptr, context);
#if defined(_WIN32)
    if (context.get_backend() == sycl::backend::ext_oneapi_hip) {
        std::lock_guard lock(registration_mutex);
        auto& registrations = copy_registrations();
        const auto it = registrations.find(reinterpret_cast<uintptr_t>(ptr));
        if (it != registrations.end() && --it->second.references == 0) registrations.erase(it);
    }
#endif
}

sycl::event copy_async(sycl::queue& q, void* dst, const void* src, size_t bytes,
                       const std::vector<sycl::event>& dependencies) {
    if (!bytes) {
#if defined(_WIN32)
        // A HIP barrier can block its issuer on an outstanding host-task dependency. An empty compute command
        // retains that dependency asynchronously, including when a caller only wants a completion marker.
        if (q.get_backend() == sycl::backend::ext_oneapi_hip)
            return q.submit([&](sycl::handler& h) {
                h.depends_on(dependencies);
                h.single_task([] {});
            });
#endif
        return q.ext_oneapi_submit_barrier(dependencies);
    }
#if defined(_WIN32)
    if (prepare_copy(q, dst, src, dependencies))
        return q.submit([&](sycl::handler& h) {
            h.depends_on(dependencies);
            compute_copy(h, dst, src, bytes);
        });
#endif
    return q.memcpy(dst, src, bytes, dependencies);
}

void copy_async_no_event(sycl::queue& q, void* dst, const void* src, size_t bytes) {
    if (!bytes) return;
#if defined(_WIN32)
    if (prepare_copy(q, dst, src, {})) {
        sycl::ext::oneapi::experimental::submit(q, [&](sycl::handler& h) { compute_copy(h, dst, src, bytes); });
        return;
    }
#endif
    sycl::ext::oneapi::experimental::memcpy(q, dst, src, bytes);
}

sycl::event readback_async(sycl::queue& q, void* dst, const void* src, size_t bytes) {
    if (bytes == 0) return {};
    check_buffers(dst, src);
    prepare_readback(q);
    return copy_async(q, dst, src, bytes);
}

sycl::event readback_async(sycl::queue& q, std::span<const ReadbackSpan> copies) {
    if (!has_bytes(copies)) return {};
    for (const auto& c : copies) if (c.bytes) check_buffers(c.dst, c.src);
    prepare_readback(q);
    sycl::event last;
    bool native = true;
#if defined(_WIN32)
    native = native_readback_batch(q, copies);
#endif
    for (const auto& c : copies) if (c.bytes)
        last = native ? q.memcpy(c.dst, c.src, c.bytes) : copy_async(q, c.dst, c.src, c.bytes);
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
#if defined(_WIN32)
    if (prepare_copy(q, dst, src, {}))
        return q.parallel_for(sycl::range<2>(height, width), [=](sycl::id<2> at) {
            d[at[0] * dpitch + at[1]] = s[at[0] * spitch + at[1]];
        });
#endif
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
