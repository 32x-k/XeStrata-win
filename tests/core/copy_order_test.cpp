// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "strata/core/readback.hpp"
#include "strata/core/runtime.hpp"
#include "strata/core/device.hpp"
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#if defined(SYCL_EXT_ONEAPI_GRAPH) && __has_include(<sycl/ext/oneapi/experimental/graph.hpp>)
#include <sycl/ext/oneapi/experimental/graph.hpp>
#define STRATA_COPY_TEST_GRAPH 1
#endif

namespace {
namespace core = strata::core;
void check(bool ok, const char* what) { if (!ok) throw std::runtime_error(what); }

struct Memory {
    sycl::queue& q;
    sycl::queue* other;
    std::vector<void*> allocations;
    explicit Memory(sycl::queue& queue, sycl::queue* second = nullptr) : q(queue), other(second) {
        allocations.reserve(16);
    }
    ~Memory() {
        try {
            q.wait_and_throw();
            if (other) other->wait_and_throw();
            for (void* p : allocations) sycl::free(p, q);
        } catch (...) {
            std::fputs("copy order test: cannot safely release buffers\n", stderr);
            std::_Exit(1);
        }
    }
    template<class T> T* get(size_t count, sycl::usm::alloc kind = sycl::usm::alloc::device) {
        auto* p = static_cast<T*>(sycl::malloc(count * sizeof(T), q, kind));
        if (!p) throw std::bad_alloc();
        allocations.push_back(p);
        return p;
    }
    Memory(const Memory&) = delete;
    Memory& operator=(const Memory&) = delete;
};

// Initialization and independent verification are outside the dependency chain
// under test. Host vectors are declared before Memory so exceptions also drain
// the queue before any asynchronous copy's ordinary host storage is destroyed.
void safe_read(sycl::queue& q, void* dst, const void* src, size_t bytes) {
    q.wait_and_throw();
    q.memcpy(dst, src, bytes).wait_and_throw();
}

void byte_ranges(sycl::queue& q, bool host_usm) {
    for (size_t bytes : {size_t{37}, (size_t{1} << 20) + 3}) {
        constexpr size_t source_offset = 1, destination_offset = 3;
        const size_t size = bytes + 19;
        std::vector<uint8_t> source(size), actual(size, 0xa5), expected = actual;
        for (size_t i = 0; i < size; ++i) source[i] = static_cast<uint8_t>((i * 29 + 7) & 255);
        std::copy_n(source.data() + source_offset, bytes, expected.data() + destination_offset);
        Memory memory(q);
        auto* a = memory.get<uint8_t>(size);
        auto* b = memory.get<uint8_t>(size);
        const uint8_t* upload = source.data();
        uint8_t* download = actual.data();
        if (host_usm) {
            auto* host_source = memory.get<uint8_t>(size, sycl::usm::alloc::host);
            download = memory.get<uint8_t>(size, sycl::usm::alloc::host);
            std::copy(source.begin(), source.end(), host_source);
            upload = host_source;
        }
        q.fill(a, uint8_t{0xa5}, size);
        q.fill(b, uint8_t{0xa5}, size).wait_and_throw();
        core::copy_async(q, a + destination_offset, upload + source_offset, bytes);
        core::copy_async_no_event(q, b + destination_offset, a + destination_offset, bytes);
        core::copy_async(q, download, b, size).wait_and_throw();
        if (host_usm) std::copy_n(download, size, actual.begin());
        check(actual == expected, "unaligned/large copy or guard bytes differ");
        core::copy_async(q, nullptr, nullptr, 0).wait_and_throw();
        core::copy_async_no_event(q, nullptr, nullptr, 0);
        safe_read(q, actual.data(), b, size);
        check(actual == expected, "zero-byte copy changed data");
    }
    std::printf("copy byte ranges host_usm=%d: PASS\n", host_usm);
}

constexpr size_t columns = 256, inner = 512, rows = 16, inputs = 4, input_stride = 515;
constexpr size_t active = rows * columns * 2, prefix = 3, output_size = prefix + active + 5;
constexpr int guard = 0x55555555;
struct DotArgs { const float* x; const float* w; const float* u; int* out; int bias; };

template<class F> sycl::event parallel(sycl::queue& q, size_t count, F f) {
    return q.parallel_for(sycl::range<1>(count), [=](sycl::id<1> id) { f(id[0]); });
}
void produce(sycl::queue& q, const DotArgs& a) {
    // Aggregate pointer capture and per-block private arrays preserve the
    // dependency shape of the copy-order reproducer. No fixed work-group size.
    (void)parallel(q, rows * columns, [=](size_t id) {
        const size_t col = id % columns, row = id / columns;
        const size_t input = (32 - row) % inputs;
        float sum = 0.0f, usum = 0.0f;
        for (size_t block = 0; block < inner / 32; ++block) {
            float values[32], amax = 0.0f;
            for (size_t j = 0; j < 32; ++j) {
                values[j] = a.x[input * input_stride + block * 32 + j];
                amax = sycl::fmax(amax, sycl::fabs(values[j]));
            }
            const float scale = amax / 127.0f;
            int8_t codes[32];
            for (size_t j = 0; j < 32; ++j)
                codes[j] = static_cast<int8_t>(amax > 0.0f ? sycl::rint(values[j] / scale) : 0.0f);
            const size_t shift = (col + 1 + block) & 31;
            float dot = 0.0f, udot = 0.0f;
            for (size_t j = 0; j < 32; ++j) {
                const float x = static_cast<float>(codes[(j + shift) & 31]);
                dot = sycl::fma(x, a.w[col * inner + block * 32 + j], dot);
                udot = sycl::fma(x, a.u[col * inner + block * 32 + j], udot);
            }
            sum = sycl::fma(scale, dot, sum);
            usum = sycl::fma(scale, udot, usum);
        }
        a.out[2 * id] = static_cast<int>(sum * 2048.0f) + a.bias;
        a.out[2 * id + 1] = static_cast<int>(usum * 2048.0f) - a.bias;
    });
}

void structured_order(sycl::queue& q, bool host_usm) {
    std::vector<int> xi(inputs * input_stride), wi(columns * inner), ui(wi.size());
    std::vector<float> old_x(xi.size()), new_x(xi.size()), w(wi.size()), u(wi.size()), actual_x(xi.size());
    std::vector<int> initial(output_size, guard), expected = initial, actual(output_size), replacement = initial;
    for (size_t t = 0; t < inputs; ++t) for (size_t j = 0; j < inner; ++j) {
        const size_t at = t * input_stride + j;
        xi[at] = j % 32 == 31 ? 127 : static_cast<int>((j * 29 + t * 47) % 255) - 127;
        old_x[at] = static_cast<float>(xi[at]) / 128.0f;
        new_x[at] = -old_x[at];
    }
    for (size_t i = 0; i < wi.size(); ++i) {
        wi[i] = static_cast<int>((i * 13 + 3) % 15) - 7;
        ui[i] = static_cast<int>((i * 7 + 11) % 15) - 7;
        w[i] = static_cast<float>(wi[i]) / 16.0f;
        u[i] = static_cast<float>(ui[i]) / 16.0f;
    }
    // Each block has scale=1/128 exactly. Full integer sums stay below
    // 512*127*7 < 2^24, so the independent integer reference is exact in FP32.
    for (size_t row = 0; row < rows; ++row) for (size_t col = 0; col < columns; ++col) {
        int sum = 0, usum = 0;
        for (size_t block = 0; block < inner / 32; ++block) for (size_t j = 0; j < 32; ++j) {
            const size_t shift = (col + 1 + block) & 31;
            const int x = xi[((32 - row) % inputs) * input_stride + block * 32 + ((j + shift) & 31)];
            sum += x * wi[col * inner + block * 32 + j];
            usum += x * ui[col * inner + block * 32 + j];
        }
        const size_t at = prefix + 2 * (row * columns + col);
        expected[at] = sum; expected[at + 1] = usum;
    }
    for (size_t i = 0; i < active; ++i) replacement[prefix + i] = 10000000 + static_cast<int>(i);
    Memory memory(q);
    auto* x = memory.get<float>(old_x.size());
    auto* weights = memory.get<float>(w.size());
    auto* up = memory.get<float>(u.size());
    auto* out = memory.get<int>(output_size);
    auto* copied = memory.get<int>(output_size);
    const float* next_x = new_x.data();
    const int* next_out = replacement.data();
    int* result = actual.data();
    if (host_usm) {
        auto* hx = memory.get<float>(new_x.size(), sycl::usm::alloc::host);
        auto* hy = memory.get<int>(output_size, sycl::usm::alloc::host);
        std::copy(new_x.begin(), new_x.end(), hx);
        std::copy(replacement.begin(), replacement.end(), hy);
        next_x = hx; next_out = hy;
        result = memory.get<int>(output_size, sycl::usm::alloc::host);
    }
    q.memcpy(weights, w.data(), w.size() * sizeof(float));
    q.memcpy(up, u.data(), u.size() * sizeof(float)).wait_and_throw();
    const auto reset = [&] {
        q.wait_and_throw();
        q.memcpy(x, old_x.data(), old_x.size() * sizeof(float));
        q.memcpy(out, initial.data(), initial.size() * sizeof(int));
        q.memcpy(copied, initial.data(), initial.size() * sizeof(int)).wait_and_throw();
    };
    for (int repeat = 0; repeat < 8; ++repeat) {
        auto want = expected;
        for (size_t i = 0; i < active; i += 2) { want[prefix + i] += repeat; want[prefix + i + 1] -= repeat; }
        const DotArgs args{x, weights, up, out + prefix, repeat};
        reset();
        produce(q, args);
        core::copy_async(q, x, next_x, new_x.size() * sizeof(float)).wait_and_throw();
        safe_read(q, actual.data(), out, output_size * sizeof(int));
        safe_read(q, actual_x.data(), x, new_x.size() * sizeof(float));
        check(actual == want && actual_x == new_x, "H2D overwrite changed a preceding reader's input");

        reset();
        produce(q, args);
        core::copy_async(q, out + prefix, next_out + prefix, active * sizeof(int)).wait_and_throw();
        safe_read(q, actual.data(), out, output_size * sizeof(int));
        check(actual == replacement, "H2D overwrite did not follow a preceding writer");

        reset();
        std::fill_n(result, output_size, -12345);
        produce(q, args);
        core::copy_async(q, result, out, output_size * sizeof(int)).wait_and_throw();
        // Inspect the first readback before any independent queue drain/rescue.
        check(std::equal(want.begin(), want.end(), result), "D2H readback did not observe the producer");

        reset();
        produce(q, args);
        core::copy_async_no_event(q, copied + prefix, out + prefix, active * sizeof(int));
        core::copy_async(q, result, copied, output_size * sizeof(int)).wait_and_throw();
        check(std::equal(want.begin(), want.end(), result), "D2D copy did not observe the producer");
    }
    std::printf("structured copy order host_usm=%d: PASS\n", host_usm);
}

// A finite host task makes blocking submission observable without deadlocking:
// every exit releases it before joining the submission thread or draining USM.
// These timeouts bound the test protocol, not a device tuning parameter.
struct Gate {
    std::mutex mutex;
    std::condition_variable cv;
    bool entered = false, released = false, expired = false;
    void run() {
        std::unique_lock lock(mutex);
        entered = true; cv.notify_all();
        expired = !cv.wait_for(lock, std::chrono::seconds(10), [&] { return released; });
    }
    bool wait_entered() {
        std::unique_lock lock(mutex);
        return cv.wait_for(lock, std::chrono::seconds(3), [&] { return entered; });
    }
    void release() { std::lock_guard lock(mutex); released = true; cv.notify_all(); }
    bool timed_out() { std::lock_guard lock(mutex); return expired; }
};
struct Release {
    std::shared_ptr<Gate> gate;
    explicit Release(std::shared_ptr<Gate> value) : gate(std::move(value)) {}
    ~Release() { gate->release(); }
    Release(const Release&) = delete;
    Release& operator=(const Release&) = delete;
};
sycl::event block(sycl::queue& q, const std::shared_ptr<Gate>& gate) {
    return q.submit([&](sycl::handler& h) { h.host_task([gate] { gate->run(); }); });
}

void submission(sycl::queue& q, bool no_event, bool download, bool unrelated) {
    constexpr size_t count = 1027;
    std::vector<uint8_t> source(count, 0x7b), actual(count);
    sycl::queue other(q.get_context(), q.get_device(), sycl::property::queue::in_order{});
    Memory memory(q, &other);
    auto* device = memory.get<uint8_t>(count);
    uint8_t* host = unrelated ? source.data() : memory.get<uint8_t>(count, sycl::usm::alloc::host);
    std::fill_n(host, count, download ? uint8_t{0} : uint8_t{0x7b});
    q.fill(device, download ? uint8_t{0x7b} : uint8_t{0}, count).wait_and_throw();
    auto gate = std::make_shared<Gate>();
    Release release(gate);
    (void)block(unrelated ? other : q, gate);
    check(gate->wait_entered(), "host task did not enter the nonblocking protocol");
    std::promise<void> submitted;
    auto ready = submitted.get_future();
    sycl::event copied;
    std::thread worker([&] {
        try {
            if (no_event) core::copy_async_no_event(q, download ? host : device, download ? device : host, count);
            else copied = core::copy_async(q, download ? host : device, download ? device : host, count);
            submitted.set_value();
        } catch (...) { submitted.set_exception(std::current_exception()); }
    });
    bool nonblocking = false;
    try { nonblocking = ready.wait_for(std::chrono::seconds(3)) == std::future_status::ready; }
    catch (...) { gate->release(); worker.join(); throw; }
    gate->release();
    worker.join();
    ready.get();
    if (no_event) q.ext_oneapi_submit_barrier().wait_and_throw();
    else copied.wait_and_throw();
    other.wait_and_throw();
    check(nonblocking && !gate->timed_out(), "copy submission waited for a held host task");
    if (download) check(std::equal(source.begin(), source.end(), host), "nonblocking D2H copy differs");
    else {
        safe_read(q, actual.data(), device, count);
        check(actual == source, "nonblocking H2D copy differs");
    }
    std::printf("copy submission no_event=%d download=%d unrelated=%d: PASS\n", no_event, download, unrelated);
}

void dependencies(sycl::queue& q) {
    constexpr size_t count = 4099;
    std::vector<int> actual(count), expected(count);
    for (size_t i = 0; i < count; ++i) expected[i] = static_cast<int>(i * 7 + 11);
    sycl::queue other(q.get_context(), q.get_device(), sycl::property::queue::in_order{});
    Memory memory(q, &other);
    auto* source = memory.get<int>(count);
    auto* destination = memory.get<int>(count);
    for (bool zero : {false, true}) {
        q.fill(source, -1234, count);
        q.fill(destination, -4321, count).wait_and_throw();
        auto gate = std::make_shared<Gate>();
        Release release(gate);
        (void)block(other, gate);
        check(gate->wait_entered(), "dependency host task did not enter");
        const auto producer = other.parallel_for(sycl::range<1>(count), [=](sycl::id<1> at) {
            source[at[0]] = static_cast<int>(at[0] * 7 + 11);
        });
        // A separate thread also catches an erroneous host wait in the zero-byte
        // barrier path. Unknown-host copies are tested separately, not here.
        std::promise<sycl::event> promise;
        auto ready = promise.get_future();
        std::thread worker([&] {
            try { promise.set_value(core::copy_async(q, zero ? nullptr : destination,
                zero ? nullptr : source, zero ? 0 : count * sizeof(int), {producer})); }
            catch (...) { promise.set_exception(std::current_exception()); }
        });
        bool nonblocking = false, incomplete = false;
        sycl::event copied;
        try {
            nonblocking = ready.wait_for(std::chrono::seconds(3)) == std::future_status::ready;
            if (nonblocking) {
                copied = ready.get();
                incomplete = copied.get_info<sycl::info::event::command_execution_status>() != sycl::info::event_command_status::complete;
            }
        } catch (...) { gate->release(); worker.join(); throw; }
        gate->release(); worker.join();
        if (!nonblocking) copied = ready.get();
        copied.wait_and_throw();
        std::printf("dependency zero=%d nonblocking=%d incomplete=%d expired=%d\n",
                    zero, nonblocking, incomplete, gate->timed_out());
        check(nonblocking && incomplete && !gate->timed_out(), "copy dropped or synchronously waited for a dependency");
        safe_read(q, actual.data(), zero ? source : destination, count * sizeof(int));
        check(actual == expected, "cross-queue copy dependency did not carry producer data");
        other.wait_and_throw();
    }
    std::puts("cross-queue and zero-byte dependencies: PASS");
}

void registered_copies(sycl::queue& q) {
#if defined(SYCL_EXT_ONEAPI_COPY_OPTIMIZE)
    constexpr size_t size = 8192 + 19;
    std::vector<uint8_t> host(size), expected(size), device_result(size);
    Memory memory(q);
    struct Registration {
        sycl::queue& q;
        const void* ptr;
        Registration(sycl::queue& queue, const void* data, size_t bytes) : q(queue), ptr(data) {
            core::prepare_for_device_copy(ptr, bytes, q.get_context());
        }
        ~Registration() {
            try {
                q.wait_and_throw();
                core::release_from_device_copy(ptr, q.get_context());
            } catch (const std::exception& e) {
                std::fprintf(stderr, "registered copy cleanup: %s\n", e.what());
                std::_Exit(1);
            } catch (...) {
                std::fputs("registered copy cleanup: unknown error\n", stderr);
                std::_Exit(1);
            }
        }
        Registration(const Registration&) = delete;
        Registration& operator=(const Registration&) = delete;
    };
    // This is a fresh ordinary vector, never malloc_host storage. The owner is
    // destroyed before Memory and the vector, and drains before unregistering.
    std::unique_ptr<Registration> registration;
    try { registration = std::make_unique<Registration>(q, host.data(), host.size()); }
    catch (const sycl::exception& e) {
        if (e.code() != sycl::make_error_code(sycl::errc::feature_not_supported)) throw;
        std::printf("SKIP: host registration unsupported: %s\n", e.what());
        return;
    }
    auto* device = memory.get<uint8_t>(size);
    for (bool interior : {false, true}) {
        const size_t from = interior ? 1 : 0, to = interior ? 3 : 0;
        const size_t bytes = interior ? 8192 : 4096;
        std::fill(host.begin(), host.end(), uint8_t{0xa5});
        for (size_t i = 0; i < bytes; ++i) host[from + i] = static_cast<uint8_t>((i * 29 + 17) & 255);
        expected = host;
        for (size_t i = 0; i < bytes; ++i) expected[to + i] = static_cast<uint8_t>(host[from + i] ^ 0x6b);
        q.fill(device, uint8_t{0xa5}, size).wait_and_throw();
        core::copy_async(q, device + to, host.data() + from, bytes);
        q.parallel_for(sycl::range<1>(bytes), [=](sycl::id<1> at) {
            device[to + at[0]] = static_cast<uint8_t>(device[to + at[0]] ^ 0x6b);
        });
        core::copy_async_no_event(q, host.data() + to, device + to, bytes);
        q.wait_and_throw();
        check(host == expected, "registered roundtrip or host guard bytes differ");
        safe_read(q, device_result.data(), device, size);
        for (size_t i = 0; i < size; ++i) {
            const uint8_t want = i >= to && i < to + bytes ? expected[i] : uint8_t{0xa5};
            check(device_result[i] == want, "registered copy changed device guard bytes");
        }
    }
#if defined(_WIN32) && defined(STRATA_COPY_TEST_GRAPH)
    if (q.get_backend() == sycl::backend::ext_oneapi_hip) {
        if (q.get_device().has(sycl::aspect::ext_oneapi_graph) ||
            q.get_device().has(sycl::aspect::ext_oneapi_limited_graph)) {
            namespace sx = sycl::ext::oneapi::experimental;
            for (bool download : {false, true}) for (bool no_event : {false, true}) {
                sx::command_graph<sx::graph_state::modifiable> graph(q.get_context(), q.get_device());
                graph.begin_recording(q);
                bool refused = false;
                try {
                    auto* dst = download ? host.data() + 3 : device + 3;
                    const auto* src = download ? device + 1 : host.data() + 1;
                    if (no_event) core::copy_async_no_event(q, dst, src, size_t{37});
                    else (void)core::copy_async(q, dst, src, size_t{37});
                } catch (const std::invalid_argument&) { refused = true; }
                catch (...) { graph.end_recording(q); throw; }
                graph.end_recording(q);
                check(refused, "registered interior pointer was accepted in Windows HIP graph capture");
            }
            std::puts("registered interior graph rejection: PASS");
        } else std::puts("SKIP: registered graph rejection requires graph aspect");
    }
#endif
    std::puts("registered roundtrip/interior copy: PASS");
#else
    (void)q;
    std::puts("SKIP: compiler has no host registration extension");
#endif
}

void graphs(sycl::queue& q) {
#if defined(STRATA_COPY_TEST_GRAPH)
    namespace sx = sycl::ext::oneapi::experimental;
    if (!q.get_device().has(sycl::aspect::ext_oneapi_graph) &&
        !q.get_device().has(sycl::aspect::ext_oneapi_limited_graph)) {
        std::puts("SKIP: copy graph replay requires graph aspect"); return;
    }
    constexpr size_t count = 1031;
    std::vector<int> ordinary(count);
    Memory memory(q);
    auto* input = memory.get<int>(count, sycl::usm::alloc::host);
    auto* output = memory.get<int>(count, sycl::usm::alloc::host);
    auto* device = memory.get<int>(count);
    sx::command_graph<sx::graph_state::modifiable> graph(q.get_context(), q.get_device());
    graph.begin_recording(q);
    try {
        core::copy_async(q, device, input, count * sizeof(int));
        q.parallel_for(sycl::range<1>(count), [=](sycl::id<1> at) { device[at[0]] = 3 * device[at[0]] + 1; });
        core::copy_async_no_event(q, output, device, count * sizeof(int));
        graph.end_recording(q);
    } catch (...) { graph.end_recording(q); throw; }
    auto executable = graph.finalize();
    for (int replay = 0; replay < 2; ++replay) {
        for (size_t i = 0; i < count; ++i) input[i] = static_cast<int>(i) - replay * 17;
        std::fill_n(output, count, -12345);
        q.ext_oneapi_graph(executable).wait_and_throw();
        for (size_t i = 0; i < count; ++i) check(output[i] == 3 * input[i] + 1, "copy graph replay returned stale data");
    }
#if defined(_WIN32)
    if (q.get_backend() == sycl::backend::ext_oneapi_hip) {
        sx::command_graph<sx::graph_state::modifiable> rejected(q.get_context(), q.get_device());
        rejected.begin_recording(q);
        bool refused = false;
        try { core::copy_async(q, device, ordinary.data(), count * sizeof(int)); }
        catch (const std::exception&) { refused = true; }
        rejected.end_recording(q);
        check(refused, "pageable copy was accepted in Windows HIP graph capture");
    }
#endif
    std::puts("copy graph record/replay: PASS");
#else
    (void)q;
    std::puts("SKIP: compiler has no command graph extension");
#endif
}
} // namespace

int main() try {
    core::restrict_hip_to_selector();
    core::no_hip_resource_cache();
    if (!core::Runtime::count()) { std::puts("SKIP: no SYCL GPU"); return 77; }
    auto& runtime = core::Runtime::get();
    if (!runtime.device().has(sycl::aspect::usm_device_allocations)) {
        std::puts("SKIP: copy order test requires device USM"); return 77;
    }
    sycl::queue q(runtime.context(), runtime.device(), [](const sycl::exception_list& errors) {
        if (errors.size()) std::rethrow_exception(*errors.begin());
    }, sycl::property::queue::in_order{});
    byte_ranges(q, false);
    structured_order(q, false);
    registered_copies(q);
    dependencies(q);
    // A pageable fallback may wait for its own queue/dependencies, but must not
    // drain an unrelated queue waiting for CPU service.
    submission(q, false, false, true);
    if (q.get_device().has(sycl::aspect::usm_host_allocations)) {
        byte_ranges(q, true);
        structured_order(q, true);
        for (bool no_event : {false, true}) for (bool download : {false, true})
            submission(q, no_event, download, false);
        graphs(q);
    } else std::puts("SKIP: host USM submission and graph tests require host USM aspect");
    std::puts("core copy order: PASS");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "core copy order: %s\n", e.what());
    return 1;
}
