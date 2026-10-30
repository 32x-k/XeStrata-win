// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include "strata/core/gpu.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/runtime.hpp"
#include "strata/prefill/gemm.hpp"
#include <memory>

constexpr bool ds4_dbg(const char*) { return false; }

namespace strata::ds4::rt {

enum class Result { success, pending, error };
inline thread_local std::string local_error;
inline Result result(bool ok) { local_error.clear(); return ok ? Result::success : Result::error; }
inline Result result(gpu::Status s) {
    local_error.clear();
    return s == gpu::Status::Done ? Result::success : s == gpu::Status::Pending ? Result::pending : Result::error;
}
template <class F> Result checked(const char* operation, F&& call) {
    try { return call(); }
    catch (const std::exception& e) {
        local_error = std::string(operation) + ": " + e.what();
        return Result::error;
    }
}
inline const char* error_string(Result) { return local_error.empty() ? gpu::last_error() : local_error.c_str(); }
// SYCL reports submission failures as exceptions and asynchronous failures at waits. The underlying GPU wrapper
// retains the last failed operation's message; there is no separate CUDA-style launch-error state to clear here.
inline Result last_error() { return Result::error; }

// A model owns the handle; the runtime owns its in-order queue and context. Kernel APIs receive the queue itself.
struct Queue {
    gpu::Stream handle = nullptr;
    Queue() = default;
    Queue(std::nullptr_t) {}
    explicit Queue(gpu::Stream s) : handle(s) {}
    operator sycl::queue&() const {
        auto* owner = handle ? core::Runtime::owner_of_stream(handle) : &core::Runtime::get();
        if (!owner) throw core::DeviceError("DeepSeek queue has no owning runtime");
        return owner->stream(handle);
    }
    explicit operator bool() const { return handle != nullptr; }
    bool operator==(std::nullptr_t) const { return handle == nullptr; }
    bool operator!=(std::nullptr_t) const { return handle != nullptr; }
};
using Event = gpu::Event*;

inline Result set_device(int id) {
    return checked("set_device", [&] { core::set_current_device(id); return result(true); });
}
inline Result get_device(int* id) { *id = core::current_device(); return result(true); }
inline Result device_count(int* n) {
    return checked("device_count", [&] { *n = core::Runtime::count(); return result(true); });
}
inline Result device_sync() { return result(gpu::device_sync()); }
inline Result stream_create(Queue* q) { q->handle = gpu::stream_create(); return result(q->handle != nullptr); }
inline Result stream_destroy(Queue q) { gpu::stream_destroy(q.handle); return Result::success; }
inline Result stream_sync(Queue q) { return result(gpu::stream_sync(q.handle)); }
inline Result stream_status(Queue q) { return result(gpu::stream_status(q.handle)); }
inline Result event_create(Event* e, bool timing = false) { return result(gpu::event_create(e, timing)); }
inline Result event_destroy(Event e) { gpu::event_destroy(e); return Result::success; }
inline Result event_record(Event e, Queue q) { return result(gpu::event_record(e, q.handle)); }
inline Result event_sync(Event e) { return result(gpu::event_sync(e)); }
inline Result event_status(Event e) { return result(gpu::event_status(e)); }
inline Result event_elapsed(float* ms, Event a, Event b) { return result(gpu::event_elapsed_ms(ms, a, b)); }
inline Result stream_wait_event(Queue q, Event e) { return result(gpu::stream_wait_event(q.handle, e)); }
template <class T> Result alloc_device(T** p, size_t n) {
    *p = nullptr;
    return checked("alloc_device", [&] {
        if (n > core::max_alloc_bytes()) {
            local_error = "allocation exceeds the device's largest allocation limit";
            return Result::error;
        }
        return result(gpu::alloc_device(p, n));
    });
}
template <class T> Result alloc_host(T** p, size_t n) { return result(gpu::alloc_host(p, n)); }
inline Result free(const void* p) { gpu::free(p); return Result::success; }
inline Result mem_info(size_t* available, size_t* total) { return result(gpu::mem_info(available, total)); }
inline Result copy(void* d, const void* s, size_t n) { return result(gpu::copy(d, s, n)); }
inline Result copy_async(void* d, const void* s, size_t n, Queue q) {
    return result(gpu::copy_async(d, s, n, q.handle));
}
inline Result copy2d_async(void* d, size_t dp, const void* s, size_t sp, size_t w, size_t h, Queue q) {
    return result(gpu::copy2d_async(d, dp, s, sp, w, h, q.handle));
}
inline Result copy2d(void* d, size_t dp, const void* s, size_t sp, size_t w, size_t h) {
    return result(gpu::copy2d_async(d, dp, s, sp, w, h, nullptr) && gpu::stream_sync(nullptr));
}
inline Result memset(void* d, int value, size_t n) { return result(gpu::memset(d, value, n)); }
inline Result memset_async(void* d, int value, size_t n, Queue q) {
    return result(gpu::memset_async(d, value, n, q.handle));
}
inline Result host_register(void* p, size_t n) { return result(gpu::host_register(p, n)); }
inline Result host_unregister(void* p) { gpu::host_unregister(p); return Result::success; }
inline Result device_pointer(void** d, void* h) {
    if (gpu::kernels_read_host(h)) { *d = h; return result(true); }
    *d = nullptr;
    local_error = "host memory is not accessible by the current device's kernels";
    return Result::error;
}
struct DeviceProperties { int integrated = 0; };
inline Result device_properties(DeviceProperties* p, int id) {
    return checked("device_properties", [&] {
        p->integrated = core::device_info(id).integrated ? 1 : 0;
        return result(true);
    });
}
inline Result device_integrated(int* v, int id) {
    return checked("device_integrated", [&] {
        *v = core::device_info(id).integrated ? 1 : 0;
        return result(true);
    });
}

using Dense = std::unique_ptr<prefill::Gemm>;
inline Result dense_create(Dense* h) {
    return checked("dense_create", [&] { *h = std::make_unique<prefill::Gemm>(); return result(true); });
}
inline Result dense_destroy(Dense& h) { h.reset(); return Result::success; }
inline Result dense_stream(Dense& h, Queue q) {
    h->init_external(q.handle, nullptr, 0);
    return Result::success;
}
inline Result dense_f16(Dense& h, int n, int rows, int k, float alpha, const uint16_t* w, int ldw,
                         const uint16_t* x, int ldx, float beta, float* y, int ldy) {
    return checked("dense_f16", [&] {
        h->f16_strided(x, w, y, rows, n, k, ldx, ldw, ldy, alpha, beta);
        return result(true);
    });
}
inline Result dense_f32(Dense& h, int n, int rows, int k, float alpha, const float* w, int ldw,
                         const float* x, int ldx, float beta, float* y, int ldy) {
    return checked("dense_f32", [&] {
        h->f32_strided(x, w, y, rows, n, k, ldx, ldw, ldy, alpha, beta);
        return result(true);
    });
}

}  // namespace strata::ds4::rt
