// GPU-to-host doorbell store without a host-USM atomic: does the host see a kernel's store to host USM while that
// kernel is still running?  Mode 0: system release fence, then a system-scope atomic store (the engine before 2026-10-11).
// Mode 1: system release fence, then a volatile store.  Mode 2: volatile store, then a system release fence.
// usage: store_probe <gpu index>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <vector>

namespace sx = sycl::ext::oneapi::experimental;
using clk = std::chrono::steady_clock;
using sys_ref = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::system,
                                 sycl::access::address_space::global_space>;

inline void ring_store(int M, uint32_t* p, uint32_t v) {
    if (M == 0) {
        sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
        sys_ref(*p).store(v);
    } else if (M == 1) {
        sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
        *reinterpret_cast<volatile uint32_t*>(p) = v;
    } else {
        *reinterpret_cast<volatile uint32_t*>(p) = v;
        sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
    }
}

void kernel(int M, sycl::queue& q, uint32_t* ring, const uint32_t* answer, const uint32_t* seqp) {
    q.single_task([=] {
        const uint32_t v = *reinterpret_cast<const volatile uint32_t*>(seqp);
        sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
        ring_store(M, ring, v);
        for (uint64_t spins = 0; spins < 50000000ull; ++spins) {
            const uint32_t w = *reinterpret_cast<const volatile uint32_t*>(answer);
            sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
            if (w == v) break;
        }
    });
}

int run(int M, sycl::queue& q, bool graph) {
    auto* ring = sycl::malloc_host<uint32_t>(1, q);
    auto* answer = sycl::malloc_host<uint32_t>(1, q);
    auto* seqp = sycl::malloc_host<uint32_t>(1, q);
    std::optional<sx::command_graph<sx::graph_state::executable>> exec;
    if (graph) {
        sx::command_graph g(q.get_context(), q.get_device());
        g.begin_recording(q);
        kernel(M, q, ring, answer, seqp);
        g.end_recording(q);
        exec.emplace(g.finalize());
    }
    std::vector<double> us;
    int missed = 0;
    for (uint32_t seq = 1; seq <= 2000; ++seq) {
        *reinterpret_cast<volatile uint32_t*>(ring) = 0;
        *reinterpret_cast<volatile uint32_t*>(answer) = 0;
        *reinterpret_cast<volatile uint32_t*>(seqp) = seq;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        const auto t0 = clk::now();
        if (graph) q.ext_oneapi_graph(*exec);
        else kernel(M, q, ring, answer, seqp);
        bool seen = false;
        while (!(seen = *reinterpret_cast<volatile uint32_t*>(ring) == seq) && clk::now() - t0 < std::chrono::seconds(1)) {}
        us.push_back(std::chrono::duration<double, std::micro>(clk::now() - t0).count());
        if (!seen) ++missed;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        *reinterpret_cast<volatile uint32_t*>(answer) = seq;
        q.wait();
        if (missed > 5) break;
    }
    std::sort(us.begin(), us.end());
    std::printf("  mode %d %-6s: %zu runs, %d not seen while the kernel ran; ring seen median %.1f us p99 %.1f\n", M,
                graph ? "graph" : "direct", us.size(), missed, us[us.size() / 2], us[us.size() * 99 / 100]);
    sycl::free(ring, q);
    sycl::free(answer, q);
    sycl::free(seqp, q);
    return missed;
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const auto gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
    const int i = argc > 1 ? (int) std::strtol(argv[1], nullptr, 10) : 0;
    if (i < 0 || i >= (int) gpus.size()) { std::fprintf(stderr, "no GPU %d\n", i); return 2; }
    sycl::queue q{gpus[(size_t) i], sycl::property::queue::in_order{}};
    std::printf("%s (%s), usm_atomic_host_allocations %d\n", gpus[(size_t) i].get_info<sycl::info::device::name>().c_str(),
                gpus[(size_t) i].get_platform().get_info<sycl::info::platform::name>().c_str(),
                (int) gpus[(size_t) i].has(sycl::aspect::usm_atomic_host_allocations));
    int bad = 0;
    for (bool graph : {false, true}) {
        bad += run(0, q, graph);
        bad += run(1, q, graph);
        bad += run(2, q, graph);
    }
    return bad != 0;
}
