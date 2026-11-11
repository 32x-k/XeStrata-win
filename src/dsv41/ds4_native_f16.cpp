// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "ds4_native_f16.hpp"
#include "strata/core/per_device.hpp"
#include "strata/dsv41/exl3.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>

namespace strata::ds4::native_f16 {
bool qualify(sycl::queue& queue) {
    static core::PerDevice<bool> results;
    return results.get(queue.get_device(), [&] {
        if (!queue.get_device().has(sycl::aspect::fp16)) return false;
        const auto begin = std::chrono::steady_clock::now();
        auto* mismatch = sycl::malloc_device<uint32_t>(1, queue);
        if (!mismatch) return false;
        uint32_t host = 0;
        try {
            queue.memset(mismatch, 0, sizeof(uint32_t));
            // The item index supplies the high 24 bits, and the loop supplies
            // the low eight. Every float pattern is checked exactly once.
            queue.parallel_for(sycl::range<1>(size_t(1) << 24), [=](sycl::id<1> id) {
                uint32_t failed = 0;
                for (uint32_t low = 0; low < 256; ++low) {
                    const float value = sycl::bit_cast<float>(uint32_t(id[0]) * 256u + low);
                    failed |= uint32_t(bits(value) != sycl_detail::f16_bits(value));
                }
                if (failed) sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                    sycl::access::address_space::global_space>(mismatch[0]).fetch_or(failed);
            });
            queue.memcpy(&host, mismatch, sizeof(host)).wait_and_throw();
        } catch (...) { sycl::free(mismatch, queue); throw; }
        sycl::free(mismatch, queue);
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        const bool enabled = host == 0 && seconds <= 1.0;
        std::fprintf(stderr, "ds4: native f16 qualification enabled=%d mismatch=%u checked=4294967296 seconds=%.9f time_limit=%d\n",
                     int(enabled), host, seconds, int(seconds > 1.0));
        return enabled;
    });
}

void rows(const float* x, int ldx, const int* idx, int count, int cols,
          uint16_t* out, sycl::queue& queue, bool qualified) {
    if (!qualified || count <= 0 || cols <= 0 || (reinterpret_cast<uintptr_t>(out) & 15u) || ldx % 8 || cols % 8) {
        dsv41k::exl3::rows_f16(x, ldx, idx, count, cols, out, queue);
        return;
    }
    sycl_detail::require_in_order(queue);
    const auto device = queue.get_device();
    const size_t width = std::min(device.get_info<sycl::info::device::max_work_group_size>(),
                                  device.get_info<sycl::info::device::max_work_item_sizes<1>>()[0]);
    const size_t group = std::min(width, size_t(cols / 8));
    queue.parallel_for(sycl::nd_range<1>(size_t(count) * group, group), [=](sycl::nd_item<1> it) {
        const size_t row = it.get_group(0), source_row = idx ? size_t(idx[row]) : row;
        for (size_t col = it.get_local_id(0) * 8; col < size_t(cols); col += it.get_local_range(0) * 8) {
            sycl::vec<uint16_t, 8> values;
            for (int j = 0; j < 8; ++j) values[j] = bits(x[source_row * ldx + col + j]);
            __builtin_memcpy(out + row * cols + col, &values, sizeof(values));
        }
    });
}
}  // namespace strata::ds4::native_f16
