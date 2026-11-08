// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

// One scratch per host thread and in-order queue. Growth retains old blocks:
// recorded graphs may still reference them, and dispatch must never wait.
struct GridScratch {
    sycl::queue queue;
    std::vector<sycl::int2*> blocks;
    size_t capacity = 0, bytes = 0;
    explicit GridScratch(const sycl::queue& s) : queue(s) {}
    ~GridScratch() {
        if (blocks.empty()) return;
        try { queue.wait_and_throw(); for (auto* p : blocks) sycl::free(p, queue); }
        catch (...) { std::terminate(); }
    }
    sycl::int2* reserve(size_t count) {
        if (count <= capacity) return blocks.back();
        const auto device = queue.get_device();
        const size_t limit = std::min(size_t(device.get_info<sycl::info::device::max_mem_alloc_size>()),
                                     size_t(device.get_info<sycl::info::device::global_mem_size>()) - bytes);
        if (count > limit / sizeof(sycl::int2)) return nullptr;
        const size_t next = std::max(count, std::min(capacity, limit / sizeof(sycl::int2) / 2) * 2);
        // Reserve host bookkeeping first; no exception may orphan a USM block.
        blocks.reserve(blocks.size() + 1);
        sycl::int2* p = sycl::malloc_device<sycl::int2>(next, queue);
        if (!p) return nullptr;
        blocks.push_back(p); capacity = next; bytes += next * sizeof(sycl::int2);
        return p;
    }
};
struct MatrixGrid { const sycl::int2* jobs = nullptr; size_t groups = 0; };

MatrixGrid matrix_grid(const SetArgs& a, sycl::queue& s, size_t rows, size_t columns, int skip) {
    const size_t nr = (size_t(a.max_rows) + rows - 1) / rows;
    const MatrixGrid fallback{nullptr, size_t(a.n_exp) * nr * columns};
    if (!a.host_bounds) return fallback;
    if (!s.get_device().has(sycl::aspect::usm_device_allocations)) return fallback;
    size_t count = 0;
    for (int e = 0; e < a.n_exp; ++e) {
        const int64_t n = int64_t(a.host_bounds[e + 1]) - a.host_bounds[e];
        if (n < 0) return fallback;
        if (n > skip && !(a.skip_ge > 0 && n >= a.skip_ge))
            count += (size_t(std::min(n, int64_t(a.max_rows))) + rows - 1) / rows;
    }
    const int64_t total = int64_t(a.host_bounds[a.n_exp]) - a.host_bounds[0];
    if (total < 0 || count > size_t(total) / rows + size_t(a.n_exp) || count > size_t(a.n_exp) * nr ||
        count > std::numeric_limits<size_t>::max() / columns) return fallback;
    if (!count) return {nullptr, 0};
    sycl::int2* jobs = nullptr;
    try {
        thread_local std::vector<std::unique_ptr<GridScratch>> entries;
        GridScratch* scratch = nullptr;
        for (auto& entry : entries) if (entry->queue == s) { scratch = entry.get(); break; }
        if (!scratch) { entries.push_back(std::make_unique<GridScratch>(s)); scratch = entries.back().get(); }
        jobs = scratch->reserve(count);
    } catch (const std::bad_alloc&) { return fallback; }
      catch (const sycl::exception& e) {
        if (e.code() == sycl::make_error_code(sycl::errc::memory_allocation)) return fallback;
        throw;
    }
    if (!jobs) return fallback;
    // The host bounds already arrived for routing; only the exact grid length
    // is taken from them. Build pairs from the matching device bounds, avoiding
    // a transfer source lifetime, another readback, or a host/device wait.
    const int* bounds = a.bounds;
    const int ne = a.n_exp, hi = a.skip_ge, max_rows = a.max_rows;
    sycl_detail::submit(s, [&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(size_t(ne)), [=](sycl::id<1> id) {
            const int e = int(id[0]), n = bounds[e + 1] - bounds[e];
            if (n <= skip || (hi > 0 && n >= hi)) return;
            size_t offset = 0;
            for (int i = 0; i < e; ++i) {
                const int m = bounds[i + 1] - bounds[i];
                if (m > skip && !(hi > 0 && m >= hi))
                    offset += (size_t(sycl::min(m, max_rows)) + rows - 1) / rows;
            }
            for (size_t r = 0; r < size_t(sycl::min(n, max_rows)); r += rows)
                jobs[offset++] = sycl::int2(e, int(r));
        });
    });
    return {jobs, count * columns};
}
