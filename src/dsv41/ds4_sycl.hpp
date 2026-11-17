// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/group_local_memory.hpp>
#if defined(SYCL_EXT_ONEAPI_ENQUEUE_FUNCTIONS)
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#endif
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

namespace strata::ds4::sycl_detail {

inline void require_in_order(const sycl::queue& q) {
    if (!q.has_property<sycl::property::queue::in_order>())
        throw std::invalid_argument("ds4 kernels require an in-order SYCL queue");
}

// DS4_KPROF (a diagnostic, with STRATA_KERNEL_PROFILE for the queues' timing): each kernel of a decoded token is
// enqueued with an event while `on` is set, named by its function object's type, and `collect` adds its GPU time
// (command_end - command_start) to that name's after the token's sync; `report` prints them by time, a token's.
struct KernelProfile {
    struct Sum { double ms = 0; long long n = 0; };
    std::atomic<bool> on{false};
    std::mutex mu;
    std::vector<std::pair<sycl::event, const char*>> events;
    std::map<std::string, Sum> sums;
    long long tokens = 0;
    void add(sycl::event e, const char* name) {
        std::lock_guard<std::mutex> lk(mu);
        events.emplace_back(std::move(e), name);
    }
    void collect() {
        std::lock_guard<std::mutex> lk(mu);
        if (events.empty()) return;
        for (auto& [e, name] : events) {
            try {
                e.wait();
                const auto t0 = e.get_profiling_info<sycl::info::event_profiling::command_start>();
                const auto t1 = e.get_profiling_info<sycl::info::event_profiling::command_end>();
                Sum& s = sums[name];
                s.ms += t1 > t0 ? double(t1 - t0) * 1e-6 : 0.0;
                ++s.n;
            } catch (const std::exception&) {
                ++sums[std::string("(no timing) ") + name].n;
            }
        }
        events.clear();
        ++tokens;
    }
    void report(std::FILE* f) {
        std::lock_guard<std::mutex> lk(mu);
        if (tokens == 0) return;
        std::vector<std::pair<std::string, Sum>> v(sums.begin(), sums.end());
        std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.second.ms > b.second.ms; });
        double ms = 0;
        long long n = 0;
        for (const auto& x : v) ms += x.second.ms, n += x.second.n;
        const double t = double(tokens);
        std::fprintf(f, "kernels per token (DS4_KPROF, %lld tokens): %.3f ms in %.1f kernels\n", tokens, ms / t, double(n) / t);
        for (const auto& [name, s] : v)
            std::fprintf(f, "  %8.3f ms %7.1f x %8.2f us  %s\n", s.ms / t, double(s.n) / t,
                         s.n ? 1e3 * s.ms / double(s.n) : 0.0, name.c_str());
    }
};
inline KernelProfile& kernel_profile() {
    static KernelProfile p;
    return p;
}
inline bool kernel_profile_on() { return kernel_profile().on.load(std::memory_order_relaxed); }

// (DS4_GRAPH) while a decode token's host pass runs against a recorded graph, the submissions it would make are
// the graph's own: they are dropped here (the pass only builds the run's staged words), then one graph_launch
// replays the recorded stream.  Set by ds4_model's decode loop only.
inline std::atomic<bool>& submit_suppressed() {
    static std::atomic<bool> v{false};
    return v;
}
inline std::atomic<long long>& submit_suppress_count() {
    static std::atomic<long long> v{0};
    return v;
}
inline bool submit_suppress_on() { return submit_suppressed().load(std::memory_order_relaxed); }
// A stream write the suppressed pass skips: counted so the pass's submission shape can be checked against the
// recorded graph's node count.  The queue argument is ignored.
inline bool suppressed_submit(sycl::queue&) {
    if (!submit_suppress_on()) return false;
    submit_suppress_count().fetch_add(1, std::memory_order_relaxed);
    return true;
}

// Kernels are enqueued without an event (sycl_ext_oneapi_enqueue_functions) where the compiler has the extension:
// the in-order queue keeps their order, and no event is made a kernel.  Measured on the R9700 (HIP backend,
// Windows), 50 small kernels took 1.7 ms enqueued with events and 0.19 ms without; without the extension, the
// queue's own calls.
#if defined(SYCL_EXT_ONEAPI_ENQUEUE_FUNCTIONS)
namespace sx = sycl::ext::oneapi::experimental;
template <class CGF>
inline void submit(sycl::queue& q, CGF&& cgf) {
    if (suppressed_submit(q)) return;
    if (kernel_profile_on()) return kernel_profile().add(q.submit(std::forward<CGF>(cgf)), typeid(CGF).name());
    sx::submit(q, std::forward<CGF>(cgf));
}
template <int D, class K>
inline void launch(sycl::queue& q, sycl::nd_range<D> range, const K& k) {
    if (suppressed_submit(q)) return;
    if (kernel_profile_on()) return kernel_profile().add(q.parallel_for(range, k), typeid(K).name());
    sx::nd_launch(q, range, k);
}
template <int D, class K>
inline void launch(sycl::queue& q, sycl::range<D> range, const K& k) {
    if (suppressed_submit(q)) return;
    if (kernel_profile_on()) return kernel_profile().add(q.parallel_for(range, k), typeid(K).name());
    sx::parallel_for(q, range, k);
}
template <class K>
inline void single(sycl::queue& q, const K& k) {
    if (suppressed_submit(q)) return;
    if (kernel_profile_on()) return kernel_profile().add(q.single_task(k), typeid(K).name());
    sx::single_task(q, k);
}
#else
template <class CGF>
inline void submit(sycl::queue& q, CGF&& cgf) { if (!suppressed_submit(q)) q.submit(std::forward<CGF>(cgf)); }
template <int D, class K>
inline void launch(sycl::queue& q, sycl::nd_range<D> range, const K& k) { if (!suppressed_submit(q)) q.parallel_for(range, k); }
template <int D, class K>
inline void launch(sycl::queue& q, sycl::range<D> range, const K& k) { if (!suppressed_submit(q)) q.parallel_for(range, k); }
template <class K>
inline void single(sycl::queue& q, const K& k) { if (!suppressed_submit(q)) q.single_task(k); }
#endif

// The runtime chooses the work-group shape; these portable kernels need neither
// a particular sub-group size nor a matrix extension.
// BoundBody's counterpart for bodies without a subgroup requirement, of any work-group size: the kernel's entry
// inlines the body instead of copying the functor's captures to private memory (R9700, HIP backend).
template <int WG, class Body>
struct BoundBodyX {
    Body body;
    [[sycl::reqd_work_group_size(WG)]] [[gnu::always_inline]]
    void operator()(sycl::nd_item<1> item) const { body(item); }
};
template <class F>
inline void parallel(sycl::queue& q, size_t n, F f) {
    require_in_order(q);
    if (!n) return;
    launch(q, sycl::range<1>(n), [=](sycl::id<1> i) { f(i[0]); });
}

inline float bf16_value(uint16_t v) {
    return sycl::bit_cast<float>(uint32_t(v) << 16);
}

// Integer conversion also works on devices without the optional fp16 aspect. Zeros and subnormals come out
// of one float subtraction, exact (both operands and the result are normal), instead of a normalizing loop.
inline float f16_value(uint16_t v) {
    constexpr uint32_t exponent = 0x7c00u << 13;
    uint32_t bits = uint32_t(v & 0x7fffu) << 13;
    const uint32_t e = bits & exponent;
    bits += (127u - 15u) << 23;
    float value = sycl::bit_cast<float>(bits);
    if (e == exponent) value = sycl::bit_cast<float>(bits + ((128u - 16u) << 23));
    else if (e == 0) value = sycl::bit_cast<float>(bits + (1u << 23)) - sycl::bit_cast<float>(113u << 23);
    return sycl::bit_cast<float>(sycl::bit_cast<uint32_t>(value) | (uint32_t(v & 0x8000u) << 16));
}

inline uint16_t f16_bits(float v) {
    const uint32_t u = sycl::bit_cast<uint32_t>(v), sign = (u >> 16) & 0x8000u;
    const uint32_t a = u & 0x7fffffffu;
    if (a >= 0x7f800000u) return uint16_t(sign | 0x7c00u | (a > 0x7f800000u ? 0x0200u : 0u));
    const int e = int(a >> 23) - 127;
    if (e > 15) return uint16_t(sign | 0x7c00u);
    if (e < -25) return uint16_t(sign);
    if (e < -14) {
        const uint32_t m = (a & 0x7fffffu) | 0x800000u;
        const int shift = -e - 1;
        const uint32_t r = m + ((1u << (shift - 1)) - 1u) + ((m >> shift) & 1u);
        return uint16_t(sign | (r >> shift));
    }
    const uint32_t r = a + 0xfffu + ((a >> 13) & 1u);
    return uint16_t(sign | ((r - 0x38000000u) >> 13));
}

}  // namespace strata::ds4::sycl_detail
