// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include "strata/core/readback.hpp"
#include <sycl/sycl.hpp>
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <vector>
#include <new>
#include <stdexcept>

namespace dsv41_test {
struct Device {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    std::vector<void*> allocations;
    void clear() {
        q.wait_and_throw();
        for (void* p : allocations) sycl::free(p, q);
        allocations.clear();
    }
    ~Device() {
        // Destruction must remain safe while an exception from a failed submission unwinds.
        try { clear(); }
        catch (...) { std::fputs("dsv41 test: device cleanup failed\n", stderr); }
    }
};
inline Device& device() {
    // An inline accessor owns one lazily initialized queue, shared across translation units.
    static Device d;  // NOLINT(bugprone-dynamic-static-initializers)
    return d;
}
inline sycl::queue& queue() { return device().q; }
template <typename T> void device_alloc(T*& ptr, size_t bytes) {
    ptr = static_cast<T*>(sycl::malloc_device(std::max<size_t>(1, bytes), queue()));
    if (!ptr) throw std::bad_alloc();
    device().allocations.push_back(ptr);
}
inline void device_free(void* p) {
    queue().wait_and_throw();
    auto& a = device().allocations;
    a.erase(std::remove(a.begin(), a.end(), p), a.end());
    sycl::free(p, queue());
}
inline void select_device(int index) {
    const auto ds = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (index < 0 || (size_t) index >= ds.size()) throw std::out_of_range("GPU index");
    device().clear();
    device().q = sycl::queue(ds[(size_t) index], sycl::property::queue::in_order{});
}
}  // namespace dsv41_test
