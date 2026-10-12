// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/core/gpu.cpp - see include/strata/core/gpu.hpp.
#include "strata/core/gpu.hpp"
#include "strata/core/device.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/runtime.hpp"

#if !defined(STRATA_ADAPTIVECPP)
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <sycl/ext/oneapi/experimental/profiling_tag.hpp>
#include <sycl/ext/oneapi/virtual_mem/physical_mem.hpp>
#include <sycl/ext/oneapi/virtual_mem/virtual_mem.hpp>
#endif

#include <algorithm>
#include <map>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#if defined(STRATA_ADAPTIVECPP)
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#endif

namespace strata::gpu {

#if !defined(STRATA_ADAPTIVECPP)
namespace sx = sycl::ext::oneapi::experimental;
#endif

struct Event {
    std::optional<sycl::event> ev;
    bool timing = false;
};

struct Graph {
#if !defined(STRATA_ADAPTIVECPP)
    sx::command_graph<sx::graph_state::executable> exec;
#endif
    size_t nodes = 0;
#if defined(STRATA_ADAPTIVECPP)
    void* hip_exec = nullptr;   // hipGraphExec_t
#endif
};

struct VmemSegment {
#if !defined(STRATA_ADAPTIVECPP)
    sx::physical_mem mem;
#endif
};

namespace {

thread_local std::string g_error;

// The live alloc_device blocks, so a refusal can say how much this process holds (the refusals seen so far came
// with ~26 GiB free in the runs that passed, cause unknown: bench/results/2026-09-30-xe-iq3xxs)
std::mutex g_live_mu;
std::map<const void*, size_t> g_live;
size_t g_live_bytes = 0;

std::string refusal_state() {
    size_t held = 0, n = 0;
    {
        std::lock_guard<std::mutex> lock(g_live_mu);
        held = g_live_bytes;
        n = g_live.size();
    }
    std::string free_mib = "unknown";
#if !defined(STRATA_ADAPTIVECPP)
    free_mib += " (set ZES_ENABLE_SYSMAN=1)";
    try {
        const auto& d = strata::core::Runtime::get().device();
        if (d.has(sycl::aspect::ext_intel_free_memory))
            free_mib = std::to_string(d.get_info<sycl::ext::intel::info::device::free_memory>() >> 20) + " MiB";
    } catch (const std::exception& e) { free_mib = std::string("unreadable: ") + e.what(); }
#endif
    return "; device free " + free_mib + ", this process holds " + std::to_string(held >> 20) + " MiB in " +
           std::to_string(n) + " blocks";
}

bool fail(const char* what, const std::exception& e) {
    g_error = std::string(what) + ": " + e.what();
    return false;
}

core::Runtime& rt() { return core::Runtime::get(); }

// A single memset of more than 4 GiB never completes on the B70; the same bytes in 1 GiB pieces do, and a 5 GiB memcpy
// and kernels addressing past 4 GiB work normally (bench/results/2026-09-30-xe-large-alloc).
constexpr size_t kPiece = size_t(1) << 30;
sycl::event marker(sycl::queue& queue, const std::vector<sycl::event>& dependencies = {}) {
    return queue.submit([&](sycl::handler& h) {
        if (!dependencies.empty()) h.depends_on(dependencies);
        h.single_task([] {});
    });
}

sycl::event memset_pieces(sycl::queue& queue, void* dst, int value, size_t bytes) {
    sycl::event e;
    for (size_t o = 0; o < bytes; o += kPiece) e = queue.memset((uint8_t*) dst + o, value, std::min(kPiece, bytes - o));
    return e;
}
// A stream is a queue of the device it was created on, whichever device is current (null: the current device's
// compute queue).
sycl::queue& q(Stream s) {
    if (!s) return rt().compute();
    core::Runtime* owner = core::Runtime::owner_of_stream(s);
    if (!owner) throw core::DeviceError("the stream is not a queue of the Xe runtime");
    return owner->stream(s);
}
core::Runtime& rt_of(Stream s) {
    if (!s) return rt();
    core::Runtime* owner = core::Runtime::owner_of_stream(s);
    return owner ? *owner : rt();
}

#if defined(STRATA_ADAPTIVECPP)
// The few HIP calls SYCL has no word for (free memory, stream capture and graphs), from the HIP runtime the
// AdaptiveCpp backend has already loaded, so the build does not link HIP.  Handles are opaque pointers.
struct HipApi {
    int (*get_device)(int*) = nullptr;
    int (*set_device)(int) = nullptr;
    int (*mem_get_info)(size_t*, size_t*) = nullptr;
    int (*begin_capture)(void* stream, int mode) = nullptr;
    int (*end_capture)(void* stream, void** graph) = nullptr;
    int (*graph_nodes)(void* graph, void** nodes, size_t* count) = nullptr;
    int (*instantiate)(void** exec, void* graph, void** error_node, char* log, size_t log_size) = nullptr;
    int (*graph_launch)(void* exec, void* stream) = nullptr;
    int (*graph_destroy)(void* graph) = nullptr;
    int (*exec_destroy)(void* exec) = nullptr;
    const char* (*error_string)(int) = nullptr;
    bool memory() const { return get_device && set_device && mem_get_info; }
    bool graphs() const {
        return begin_capture && end_capture && graph_nodes && instantiate && graph_launch && graph_destroy &&
               exec_destroy && error_string;
    }
};
constexpr int kHipCaptureModeRelaxed = 2;   // hipStreamCaptureModeRelaxed

const HipApi& hip() {
    static const HipApi api = [] {
        HipApi a;
#if defined(_WIN32)
        HMODULE lib = ::GetModuleHandleA("amdhip64_7.dll");
        if (lib == nullptr) lib = ::GetModuleHandleA("amdhip64_6.dll");
        if (lib == nullptr) return a;
        auto sym = [&](const char* name) { return reinterpret_cast<void*>(::GetProcAddress(lib, name)); };
#else
        void* lib = nullptr;
        for (const char* name : {"libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6"})
            if ((lib = ::dlopen(name, RTLD_NOW | RTLD_NOLOAD)) != nullptr) break;
        if (lib == nullptr) return a;
        auto sym = [&](const char* name) { return ::dlsym(lib, name); };
#endif
        auto get = [&](auto& fn, const char* name) { fn = reinterpret_cast<std::decay_t<decltype(fn)>>(sym(name)); };
        get(a.get_device, "hipGetDevice");
        get(a.set_device, "hipSetDevice");
        get(a.mem_get_info, "hipMemGetInfo");
        get(a.begin_capture, "hipStreamBeginCapture");
        get(a.end_capture, "hipStreamEndCapture");
        get(a.graph_nodes, "hipGraphGetNodes");
        get(a.instantiate, "hipGraphInstantiate");
        get(a.graph_launch, "hipGraphLaunch");
        get(a.graph_destroy, "hipGraphDestroy");
        get(a.exec_destroy, "hipGraphExecDestroy");
        get(a.error_string, "hipGetErrorString");
        return a;
    }();
    return api;
}

std::string hip_error(const char* what, int code) {
    return std::string(what) + ": HIP error " + std::to_string(code) +
           (hip().error_string ? std::string(" (") + hip().error_string(code) + ")" : std::string());
}

// The HIP stream of a queue: for a native in-order queue (a capture stream) its one stream.
void* hip_stream(sycl::queue& queue) {
    void* stream = nullptr;
    queue.AdaptiveCpp_enqueue_custom_operation([&stream](sycl::interop_handle& h) {
        stream = h.get_native_queue<sycl::backend::hip>();
    }).wait();
    return stream;
}

// HIP's own free-memory answer for a HIP device.
bool hip_free_memory(const sycl::device& d, uint64_t& free, uint64_t& total) {
    if (d.get_backend() != sycl::backend::hip || !hip().memory()) return false;
    const int id = sycl::get_native<sycl::backend::hip>(d);
    int previous = -1;
    if (hip().get_device(&previous) != 0 || (previous != id && hip().set_device(id) != 0)) return false;
    size_t f = 0, t = 0;
    const bool ok = hip().mem_get_info(&f, &t) == 0;
    if (previous != id) hip().set_device(previous);
    if (!ok || t == 0) return false;
    free = f;
    total = t;
    return true;
}

// the recordings in progress: queue -> its HIP stream being captured
std::mutex g_capture_mutex;
std::map<const sycl::queue*, void*> g_captures;
#endif

// the recordings in progress, by queue
#if !defined(STRATA_ADAPTIVECPP)
std::mutex g_capture_mutex;
std::map<const sycl::queue*, sx::command_graph<sx::graph_state::modifiable>> g_captures;
#endif

}  // namespace

const char* last_error() { return g_error.c_str(); }

Stream stream_create() {
    try { return rt().create_stream(); }
    catch (const std::exception& e) { fail("stream_create", e); return nullptr; }
}

Stream stream_create_capture() {
    try { return rt().create_capture_stream(); }
    catch (const std::exception& e) { fail("stream_create_capture", e); return nullptr; }
}

void stream_destroy(Stream s) {
    try { rt_of(s).destroy_stream(static_cast<sycl::queue*>(s)); }
    catch (const std::exception& e) { fail("stream_destroy", e); }
}

bool stream_sync(Stream s) {
    try { rt_of(s).finish(q(s)); return true; }
    catch (const std::exception& e) { return fail("stream_sync", e); }
}

bool stream_idle(Stream s) {
    try {
#if defined(STRATA_ADAPTIVECPP)
        return marker(q(s)).get_info<sycl::info::event::command_execution_status>() ==
               sycl::info::event_command_status::complete;
#else
        return q(s).ext_oneapi_empty();
#endif
    } catch (const std::exception& e) { return fail("stream_idle", e); }
}

bool device_sync() {
    try { rt().finish(); return true; }
    catch (const std::exception& e) { return fail("device_sync", e); }
}

void* alloc_device(size_t bytes) {
    try {
        void* p = sycl::malloc_device(bytes, rt().device(), rt().context());
        if (!p) {
            g_error = "alloc_device: " + std::to_string(bytes) + " bytes refused" + refusal_state();
            return nullptr;
        }
        std::lock_guard<std::mutex> lock(g_live_mu);
        g_live[p] = bytes;
        g_live_bytes += bytes;
        return p;
    } catch (const std::exception& e) { fail("alloc_device", e); return nullptr; }
}

void* alloc_host(size_t bytes) {
    try {
        void* p = sycl::malloc_host(bytes, rt().context());
        if (!p) g_error = "alloc_host: " + std::to_string(bytes) + " bytes refused";
        return p;
    } catch (const std::exception& e) { fail("alloc_host", e); return nullptr; }
}

void free(const void* p) {
    {
        std::lock_guard<std::mutex> lock(g_live_mu);
        if (auto it = g_live.find(p); it != g_live.end()) {
            g_live_bytes -= it->second;
            g_live.erase(it);
        }
    }
    // freed by the device that allocated it, whichever is current
    core::Runtime* owner = core::Runtime::owner(p);
    (owner ? *owner : rt()).free(const_cast<void*>(p));
}

bool vmem_supported() {
#if defined(STRATA_ADAPTIVECPP)
    return false;
#else
    try { return rt().device().has(sycl::aspect::ext_oneapi_virtual_mem); }
    catch (const std::exception& e) { return fail("vmem_supported", e); }
#endif
}

size_t vmem_granularity() {
#if defined(STRATA_ADAPTIVECPP)
    g_error = "vmem_granularity: virtual memory is unavailable with AdaptiveCpp";
    return 0;
#else
    try { return sx::get_mem_granularity(rt().device(), rt().context(), sx::granularity_mode::recommended); }
    catch (const std::exception& e) { fail("vmem_granularity", e); return 0; }
#endif
}

void* vmem_reserve(size_t bytes) {
#if defined(STRATA_ADAPTIVECPP)
    (void) bytes;
    g_error = "vmem_reserve: virtual memory is unavailable with AdaptiveCpp";
    return nullptr;
#else
    try { return reinterpret_cast<void*>(sx::reserve_virtual_mem(bytes, rt().context())); }
    catch (const std::exception& e) { fail("vmem_reserve", e); return nullptr; }
#endif
}

void vmem_free(void* va, size_t bytes) {
#if defined(STRATA_ADAPTIVECPP)
    (void) va;
    (void) bytes;
#else
    try { sx::free_virtual_mem(reinterpret_cast<uintptr_t>(va), bytes, rt().context()); }
    catch (const std::exception& e) { fail("vmem_free", e); }
#endif
}

VmemSegment* vmem_map(void* va, size_t bytes) {
#if defined(STRATA_ADAPTIVECPP)
    (void) va;
    (void) bytes;
    g_error = "vmem_map: virtual memory is unavailable with AdaptiveCpp";
    return nullptr;
#else
    try {
        auto* seg = new VmemSegment{sx::physical_mem(rt().device(), rt().context(), bytes)};
        try {
            seg->mem.map(reinterpret_cast<uintptr_t>(va), bytes, sx::address_access_mode::read_write);
        } catch (...) {
            delete seg;
            throw;
        }
        return seg;
    } catch (const std::exception& e) { fail("vmem_map", e); return nullptr; }
#endif
}

void vmem_unmap(void* va, size_t bytes, VmemSegment* seg) {
#if defined(STRATA_ADAPTIVECPP)
    (void) va;
    (void) bytes;
#else
    try { sx::unmap(va, bytes, rt().context()); }
    catch (const std::exception& e) { fail("vmem_unmap", e); }
#endif
    delete seg;
}

VmemSegment* vmem_new(size_t bytes) {
#if defined(STRATA_ADAPTIVECPP)
    (void) bytes;
    g_error = "vmem_new: virtual memory is unavailable with AdaptiveCpp";
    return nullptr;
#else
    try { return new VmemSegment{sx::physical_mem(rt().device(), rt().context(), bytes)}; }
    catch (const std::exception& e) { fail("vmem_new", e); return nullptr; }
#endif
}

bool vmem_attach(VmemSegment* seg, void* va, size_t bytes) {
#if defined(STRATA_ADAPTIVECPP)
    (void) seg;
    (void) va;
    (void) bytes;
    g_error = "vmem_attach: virtual memory is unavailable with AdaptiveCpp";
    return false;
#else
    try {
        seg->mem.map(reinterpret_cast<uintptr_t>(va), bytes, sx::address_access_mode::read_write);
        return true;
    } catch (const std::exception& e) { return fail("vmem_attach", e); }
#endif
}

bool vmem_detach(void* va, size_t bytes) {
#if defined(STRATA_ADAPTIVECPP)
    (void) va;
    (void) bytes;
    g_error = "vmem_detach: virtual memory is unavailable with AdaptiveCpp";
    return false;
#else
    try {
        sx::unmap(va, bytes, rt().context());
        return true;
    } catch (const std::exception& e) { return fail("vmem_detach", e); }
#endif
}

void vmem_delete(VmemSegment* seg) { delete seg; }

bool is_host_usm(const void* p) {
    if (!p) return false;
    try {
        const core::Runtime* owner = core::Runtime::owner(p);
        return owner && sycl::get_pointer_type(p, owner->context()) == sycl::usm::alloc::host;
    }
    catch (const std::exception& e) { return fail("is_host_usm", e); }
}

bool copy(void* dst, const void* src, size_t bytes) {
    if (bytes == 0) return true;
    try { rt().wait(rt().compute().memcpy(dst, src, bytes), "synchronous copy"); return true; }
    catch (const std::exception& e) { return fail("copy", e); }
}

bool copy_async(void* dst, const void* src, size_t bytes, Stream s) {
    if (bytes == 0) return true;
    try { q(s).memcpy(dst, src, bytes); return true; }
    catch (const std::exception& e) { return fail("copy_async", e); }
}

bool copy_async_after(void* dst, const void* src, size_t bytes, Stream s, const Event* after, Event* done) {
    try {
        auto& queue = q(s);
        const sycl::event e = queue.submit([&](sycl::handler& h) {
            if (after != nullptr && after->ev) h.depends_on(*after->ev);
            if (bytes > 0) h.memcpy(dst, src, bytes);
        });
        if (done != nullptr) done->ev = e;
        return true;
    } catch (const std::exception& e) { return fail("copy_async_after", e); }
}

bool copy2d_async(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, Stream s) {
    if (width == 0 || height == 0) return true;
    // a kernel rather than ext_oneapi_memcpy2d, which a SYCL graph recording does not accept: the verify window and
    // the drafter record this copy
    try {
        auto& queue = q(s);
        const bool words = ((uintptr_t) dst | (uintptr_t) src | dpitch | spitch | width) % 4 == 0;
        if (words) {
            const size_t w = width / 4, dp = dpitch / 4, sp = spitch / 4;
            auto* d = static_cast<uint32_t*>(dst);
            const auto* x = static_cast<const uint32_t*>(src);
            queue.parallel_for(sycl::range<2>(height, w), [=](sycl::id<2> id) { d[id[0] * dp + id[1]] = x[id[0] * sp + id[1]]; });
        } else {
            auto* d = static_cast<uint8_t*>(dst);
            const auto* x = static_cast<const uint8_t*>(src);
            queue.parallel_for(sycl::range<2>(height, width),
                               [=](sycl::id<2> id) { d[id[0] * dpitch + id[1]] = x[id[0] * spitch + id[1]]; });
        }
        return true;
    } catch (const std::exception& e) { return fail("copy2d_async", e); }
}

bool memset(void* dst, int value, size_t bytes) {
    if (bytes == 0) return true;
    try { rt().wait(memset_pieces(rt().compute(), dst, value, bytes), "synchronous memset"); return true; }
    catch (const std::exception& e) { return fail("memset", e); }
}

bool memset_async(void* dst, int value, size_t bytes, Stream s) {
    if (bytes == 0) return true;
    try { memset_pieces(q(s), dst, value, bytes); return true; }
    catch (const std::exception& e) { return fail("memset_async", e); }
}

bool mem_info(size_t* free_bytes, size_t* total_bytes) {
    try {
        const auto& d = rt().device();
#if defined(STRATA_ADAPTIVECPP)
        // SYCL has no portable free-memory query. On the HIP backend ask HIP itself; elsewhere, or when HIP
        // cannot answer, count the device's total less what this process holds (other processes unseen).
        uint64_t total = d.get_info<sycl::info::device::global_mem_size>();
        uint64_t free = 0;
        if (!hip_free_memory(d, free, total)) {
            std::lock_guard<std::mutex> lock(g_live_mu);
            free = total > g_live_bytes ? total - g_live_bytes : 0;
        }
#if defined(_WIN32)
        // WDDM charges every device allocation against the system commit limit as well (a 4 GiB hipMalloc took
        // 4.1 GiB of commit on the R9700), so what can be allocated is also bounded by the commit still free.
        // 4 GiB is left for the host allocations that follow and for the other programs.
        MEMORYSTATUSEX m{};
        m.dwLength = sizeof m;
        constexpr uint64_t kCommitReserve = uint64_t(4) << 30;
        if (::GlobalMemoryStatusEx(&m))
            free = std::min<uint64_t>(free, m.ullAvailPageFile > kCommitReserve ? m.ullAvailPageFile - kCommitReserve : 0);
#endif
        core::apply_vram_limit(free, total);
        if (free_bytes) *free_bytes = (size_t) free;
        if (total_bytes) *total_bytes = (size_t) total;
        return true;
#else
        if (!d.has(sycl::aspect::ext_intel_free_memory)) {
            g_error = "mem_info: the device does not report free memory (set ZES_ENABLE_SYSMAN=1)";
            return false;
        }
        uint64_t free = d.get_info<sycl::ext::intel::info::device::free_memory>();
        uint64_t total = d.get_info<sycl::info::device::global_mem_size>();
        if (core::integrated_gpu(d)) core::apply_uma_limit(free);   // its memory is the system RAM
        core::apply_vram_limit(free, total);
        if (free_bytes) *free_bytes = (size_t) free;
        if (total_bytes) *total_bytes = (size_t) total;
        return true;
#endif
    } catch (const std::exception& e) { return fail("mem_info", e); }
}

bool device_speed(int device, int* units, int* khz) {
    try {
        const auto& d = core::Runtime::at(device < 0 ? core::current_device() : device).device();
        *units = (int) d.get_info<sycl::info::device::max_compute_units>();
        *khz = (int) d.get_info<sycl::info::device::max_clock_frequency>() * 1000;
        return true;
    } catch (const std::exception& e) { return fail("device_speed", e); }
}

bool host_register(void* p, size_t bytes) {
#if defined(STRATA_ADAPTIVECPP)
    (void) p;
    (void) bytes;
    g_error = "host_register: arbitrary host registration is unavailable with AdaptiveCpp";
    return false;
#else
    try { sx::prepare_for_device_copy(p, bytes, rt().context()); return true; }
    catch (const std::exception& e) { return fail("host_register", e); }
#endif
}

void host_unregister(void* p) {
#if defined(STRATA_ADAPTIVECPP)
    (void) p;
#else
    try { sx::release_from_device_copy(p, rt().context()); }
    catch (const std::exception& e) { fail("host_unregister", e); }
#endif
}

bool launch_host(Stream s, std::function<void()> fn) {
    try {
#if defined(STRATA_ADAPTIVECPP)
        q(s).AdaptiveCpp_enqueue_custom_operation([fn = std::move(fn)](auto&) { fn(); });
#else
        q(s).submit([&](sycl::handler& h) { h.host_task([fn = std::move(fn)] { fn(); }); });
#endif
        return true;
    } catch (const std::exception& e) { return fail("launch_host", e); }
}

bool event_create(Event** e, bool timing) {
    *e = new Event();
    (*e)->timing = timing;
    return true;
}

void event_destroy(Event* e) { delete e; }

bool event_record(Event* e, Stream s) {
    // an event without timing is a barrier where the runtime measured timestamps to slow the waits (measure_marks)
    try {
        auto& queue = q(s);
#if defined(STRATA_ADAPTIVECPP)
        e->ev = marker(queue);
#else
        e->ev = e->timing || !rt_of(s).marks_by_barrier() ? sx::submit_profiling_tag(queue)
                                                          : queue.ext_oneapi_submit_barrier();
#endif
        return true;
    }
    catch (const std::exception& ex) { return fail("event_record", ex); }
}

bool event_query(const Event* e) {
    if (!e->ev) return true;
    try {
        return e->ev->get_info<sycl::info::event::command_execution_status>() ==
               sycl::info::event_command_status::complete;
    } catch (const std::exception& ex) { return fail("event_query", ex); }
}

bool event_sync(const Event* e) {
    if (!e->ev) return true;
    try { rt().wait(*e->ev, "event sync"); return true; }
    catch (const std::exception& ex) { return fail("event_sync", ex); }
}

bool stream_wait_event(Stream s, const Event* e) {
    if (!e->ev) return true;
    try {
#if defined(STRATA_ADAPTIVECPP)
        (void) marker(q(s), {*e->ev});
#else
        q(s).ext_oneapi_submit_barrier({*e->ev});
#endif
        return true;
    } catch (const std::exception& ex) { return fail("stream_wait_event", ex); }
}

bool event_elapsed_ms(float* ms, const Event* from, const Event* to) {
    if (!from->ev || !to->ev) { g_error = "event_elapsed_ms: an event was not recorded"; return false; }
    if (!from->timing || !to->timing) { g_error = "event_elapsed_ms: an event was not created for timing"; return false; }
    try {
        const auto a = from->ev->get_profiling_info<sycl::info::event_profiling::command_end>();
        const auto b = to->ev->get_profiling_info<sycl::info::event_profiling::command_end>();
        *ms = (float) ((double) (int64_t) (b - a) / 1e6);
        return true;
    } catch (const std::exception& ex) { return fail("event_elapsed_ms", ex); }
}

bool begin_capture(Stream s) {
#if defined(STRATA_ADAPTIVECPP)
    // HIP's stream capture on the queue's one HIP stream: `s` must come from stream_create_capture.  Relaxed, as
    // AdaptiveCpp itself records and queries events around what it submits.
    try {
        if (!hip().graphs()) { g_error = "begin_capture: this runtime has no HIP graphs"; return false; }
        auto& queue = q(s);
        std::lock_guard<std::mutex> lock(g_capture_mutex);
        if (g_captures.count(&queue)) { g_error = "begin_capture: the stream is already recording"; return false; }
        void* stream = hip_stream(queue);
        if (stream == nullptr) { g_error = "begin_capture: the stream has no HIP stream"; return false; }
        if (const int rc = hip().begin_capture(stream, kHipCaptureModeRelaxed)) {
            g_error = hip_error("begin_capture", rc);
            return false;
        }
        g_captures.emplace(&queue, stream);
        return true;
    } catch (const std::exception& e) { return fail("begin_capture", e); }
#else
    try {
        auto& queue = q(s);
        std::lock_guard<std::mutex> lock(g_capture_mutex);
        if (g_captures.count(&queue)) { g_error = "begin_capture: the stream is already recording"; return false; }
        auto it = g_captures.emplace(&queue, sx::command_graph<sx::graph_state::modifiable>(queue.get_context(),
                                                                                             queue.get_device())).first;
        it->second.begin_recording(queue);
        return true;
    } catch (const std::exception& e) { return fail("begin_capture", e); }
#endif
}

bool end_capture(Stream s, Graph** out) {
    *out = nullptr;
#if defined(STRATA_ADAPTIVECPP)
    try {
        auto& queue = q(s);
        void* stream = nullptr;
        {
            std::lock_guard<std::mutex> lock(g_capture_mutex);
            auto it = g_captures.find(&queue);
            if (it == g_captures.end()) { g_error = "end_capture: the stream is not recording"; return false; }
            stream = it->second;
            g_captures.erase(it);
        }
        void* graph = nullptr;
        if (const int rc = hip().end_capture(stream, &graph)) { g_error = hip_error("end_capture", rc); return false; }
        size_t nodes = 0;
        int rc = hip().graph_nodes(graph, nullptr, &nodes);
        void* exec = nullptr;
        if (rc == 0 && nodes > 0) rc = hip().instantiate(&exec, graph, nullptr, nullptr, 0);
        hip().graph_destroy(graph);
        if (rc != 0) { g_error = hip_error("end_capture: instantiate", rc); return false; }
        if (nodes == 0) { g_error = "end_capture: the recording holds ZERO nodes - the body submitted nothing"; return false; }
        auto* g = new Graph;
        g->nodes = nodes;
        g->hip_exec = exec;
        *out = g;
        return true;
    } catch (const std::exception& e) { return fail("end_capture", e); }
#else
    try {
        auto& queue = q(s);
        std::lock_guard<std::mutex> lock(g_capture_mutex);
        auto it = g_captures.find(&queue);
        if (it == g_captures.end()) { g_error = "end_capture: the stream is not recording"; return false; }
        auto graph = std::move(it->second);
        g_captures.erase(it);
        graph.end_recording(queue);
        const size_t nodes = graph.get_nodes().size();
        if (nodes == 0) { g_error = "end_capture: the recording holds ZERO nodes - the body submitted nothing"; return false; }
        *out = new Graph{graph.finalize(), nodes};
        return true;
    } catch (const std::exception& e) { return fail("end_capture", e); }
#endif
}

void abandon_capture(Stream s) {
#if defined(STRATA_ADAPTIVECPP)
    try {
        auto& queue = q(s);
        std::lock_guard<std::mutex> lock(g_capture_mutex);
        auto it = g_captures.find(&queue);
        if (it == g_captures.end()) return;
        void* graph = nullptr;
        if (hip().end_capture(it->second, &graph) == 0 && graph != nullptr) hip().graph_destroy(graph);
        g_captures.erase(it);
    } catch (const std::exception& e) { fail("abandon_capture", e); }
#else
    try {
        auto& queue = q(s);
        std::lock_guard<std::mutex> lock(g_capture_mutex);
        auto it = g_captures.find(&queue);
        if (it == g_captures.end()) return;
        const auto graph = it->second;
        for (auto b = g_captures.begin(); b != g_captures.end();) {   // the queue and its open branches
            if (b->second == graph) {
                b->second.end_recording(*const_cast<sycl::queue*>(b->first));
                b = g_captures.erase(b);
            } else {
                ++b;
            }
        }
    } catch (const std::exception& e) { fail("abandon_capture", e); }
#endif
}

bool stream_fork(Stream from, Stream to) {
    try {
        auto& a = q(from);
        auto& b = q(to);
#if !defined(STRATA_ADAPTIVECPP)
        {
            std::lock_guard<std::mutex> lock(g_capture_mutex);
            auto it = g_captures.find(&a);
            if (it != g_captures.end()) {
                if (g_captures.count(&b)) { g_error = "stream_fork: the branch is already recording"; return false; }
                auto graph = it->second;   // a handle: the branch records into the same graph
                graph.begin_recording(b);
                g_captures.emplace(&b, graph);
            }
        }
        const sycl::event e = a.ext_oneapi_submit_barrier();
        b.ext_oneapi_submit_barrier({e});
#else
        const sycl::event e = marker(a);
        (void) marker(b, {e});
#endif
        return true;
    } catch (const std::exception& e) { return fail("stream_fork", e); }
}

bool stream_join(Stream into, Stream from) {
    try {
        auto& a = q(into);
        auto& b = q(from);
#if !defined(STRATA_ADAPTIVECPP)
        const sycl::event e = b.ext_oneapi_submit_barrier();
        a.ext_oneapi_submit_barrier({e});
        std::lock_guard<std::mutex> lock(g_capture_mutex);
        auto it = g_captures.find(&b);
        if (it != g_captures.end()) {
            it->second.end_recording(b);
            g_captures.erase(it);
        }
#else
        const sycl::event e = marker(b);
        (void) marker(a, {e});
#endif
        return true;
    } catch (const std::exception& e) { return fail("stream_join", e); }
}

bool is_capturing(Stream s) {
#if defined(STRATA_ADAPTIVECPP)
    try {
        auto& queue = q(s);
        std::lock_guard<std::mutex> lock(g_capture_mutex);
        return g_captures.count(&queue) != 0;
    } catch (const std::exception& e) { return fail("is_capturing", e); }
#else
    try {
        auto& queue = q(s);
        std::lock_guard<std::mutex> lock(g_capture_mutex);
        return g_captures.count(&queue) != 0;
    } catch (const std::exception& e) { return fail("is_capturing", e); }
#endif
}

bool graph_launch(const Graph* g, Stream s) {
    if (!g) { g_error = "graph_launch: null graph"; return false; }
#if defined(STRATA_ADAPTIVECPP)
    // Launched from a custom operation, so AdaptiveCpp orders it on `s` like a kernel: after what `s` was given
    // before, and before what it is given next (it records its completion event after the launch).
    try {
        void* exec = g->hip_exec;
        q(s).AdaptiveCpp_enqueue_custom_operation([exec](sycl::interop_handle& h) {
            if (const int rc = hip().graph_launch(exec, h.get_native_queue<sycl::backend::hip>()))
                throw std::runtime_error(hip_error("graph_launch", rc));
        });
        return true;
    } catch (const std::exception& e) { return fail("graph_launch", e); }
#else
    try { q(s).ext_oneapi_graph(g->exec); return true; }
    catch (const std::exception& e) { return fail("graph_launch", e); }
#endif
}

size_t graph_nodes(const Graph* g) { return g ? g->nodes : 0; }

void graph_destroy(Graph* g) {
    if (!g) return;
    try { rt().finish(); } catch (const std::exception& e) { fail("graph_destroy", e); }
#if defined(STRATA_ADAPTIVECPP)
    if (g->hip_exec != nullptr) hip().exec_destroy(g->hip_exec);
#endif
    delete g;
}

}  // namespace strata::gpu
