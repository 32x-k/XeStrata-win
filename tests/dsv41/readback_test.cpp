// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "ds4_runtime.hpp"
#include "ds4_kernels.cuh"
#include "ds4_mailbox.hpp"
#include "ds4_sycl.hpp"
#include "strata/core/readback.hpp"
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
namespace rt = strata::ds4::rt;
namespace ds4 = strata::ds4;

void require(rt::Result status) {
    if (status != rt::Result::success) throw std::runtime_error(rt::error_string(status));
}
void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }

struct Stream {
    rt::Queue value;
    Stream() { require(rt::stream_create(&value)); }
    ~Stream() { rt::stream_destroy(value); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
};
struct Memory {
    sycl::queue& q;
    std::vector<void*> pointers;
    explicit Memory(sycl::queue& queue) : q(queue) {}
    ~Memory() {
        try {
            q.wait_and_throw();
            for (void* p : pointers) sycl::free(p, q);
        } catch (...) {
            std::fputs("readback test: cannot safely release GPU buffers\n", stderr);
            std::_Exit(1);
        }
    }
    template <class T> T* get(size_t n, sycl::usm::alloc kind = sycl::usm::alloc::device) {
        auto* p = static_cast<T*>(sycl::malloc(n * sizeof(T), q, kind));
        if (!p) throw std::bad_alloc();
        pointers.push_back(p);
        return p;
    }
    Memory(const Memory&) = delete;
    Memory& operator=(const Memory&) = delete;
};

void pitched(rt::Queue stream, bool odd, bool host_usm) {
    auto& q = static_cast<sycl::queue&>(stream);
    const size_t rows = 17, width = odd ? 37 : 36, sp = odd ? 43 : 44, dp = odd ? 49 : 52;
    const size_t offset = odd ? 3 : 4;
    std::vector<uint8_t> source(offset + rows * sp + 8, 0x13);
    std::vector<uint8_t> actual(offset + rows * dp + 8, 0xa5), expected = actual;
    for (size_t r = 0; r < rows; ++r) for (size_t c = 0; c < width; ++c) {
        const auto value = static_cast<uint8_t>((r * 19 + c * 7) & 255);
        source[offset + r * sp + c] = value;
        expected[offset + r * dp + c] = value;
    }
    Memory memory(q);  // Drain before host vectors are destroyed, including on failure.
    auto* device = memory.get<uint8_t>(actual.size());
    require(rt::copy_async(device, actual.data(), actual.size(), stream));
    const uint8_t* input = source.data();
    if (host_usm) {
        auto* host = memory.get<uint8_t>(source.size(), sycl::usm::alloc::host);
        std::copy(source.begin(), source.end(), host);
        input = host;
    }
    require(rt::copy2d_async(device + offset, dp, input + offset, sp, width, rows, stream));
    require(rt::readback_async(actual.data(), device, actual.size(), stream));
    require(rt::stream_sync(stream));
    check(actual == expected, "pitched upload or guard bytes differ");
    auto* copied = memory.get<uint8_t>(actual.size());
    q.fill(copied, uint8_t{0xa5}, actual.size());
    require(rt::copy2d_async(copied + offset, dp, device + offset, dp, width, rows, stream));
    std::fill(actual.begin(), actual.end(), 0xa5);
    uint8_t* output = actual.data();
    if (host_usm) {
        output = memory.get<uint8_t>(actual.size(), sycl::usm::alloc::host);
        std::copy(actual.begin(), actual.end(), output);
    }
    require(rt::readback2d_async(output + offset, dp, copied + offset, dp, width, rows, stream));
    require(rt::stream_sync(stream));
    if (host_usm) std::copy_n(output, actual.size(), actual.begin());
    check(actual == expected, "pitched readback or guard bytes differ");
    require(rt::copy2d_async(nullptr, 0, nullptr, 0, 0, rows, stream));
    require(rt::readback2d_async(nullptr, 0, nullptr, 0, width, 0, stream));
    std::printf("pitched odd=%d host_usm=%d: PASS\n", odd, host_usm);
}

void ordered(sycl::queue& q, rt::Queue* runtime_stream) {
    constexpr size_t count = 4096, inner = 512;
    std::vector<float> input(inner), weights(inner), expected(count), output(count);
    for (size_t j = 0; j < inner; ++j) {
        input[j] = static_cast<float>(static_cast<int>(j % 31) - 15) / 32.0f;
        weights[j] = static_cast<float>(static_cast<int>(j % 7) - 3) / 8.0f;
    }
    float dot = 0;
    for (size_t j = 0; j < inner; ++j) dot += input[j] * weights[j];
    Memory memory(q);
    auto* x = memory.get<float>(inner);
    auto* w = memory.get<float>(inner);
    auto* y = memory.get<float>(count);
    auto* host = memory.get<float>(count, sycl::usm::alloc::host);
    auto* flag = memory.get<unsigned>(1, sycl::usm::alloc::host);
    q.memcpy(x, input.data(), inner * sizeof(float));
    q.memcpy(w, weights.data(), inner * sizeof(float));
    q.wait_and_throw();
    auto control = std::make_shared<ds4::MailboxControl>();
    ds4::bind_mailbox(q, control);
    struct Unbind {
        sycl::queue& q;
        ~Unbind() { ds4::unbind_mailbox(&q); }
    } unbind{q};
    sycl::queue other(q.get_context(), q.get_device(), sycl::property::queue::in_order{});
    for (int variant = 0; variant < 4; ++variant) {
        for (int rep = 0; rep < 16; ++rep) {
            const float bias = static_cast<float>(rep + variant * 16);
            std::fill(expected.begin(), expected.end(), dot + bias);
            q.fill(y, -12345.0f, count).wait_and_throw();
            if (variant == 1) q.submit([](sycl::handler& h) { h.host_task([] {}); });
            if (variant == 2) {
                const auto dependency = other.single_task([] {});
                q.ext_oneapi_submit_barrier({dependency});
            }
            if (variant == 3) (void) ds4::sycl_detail::parallel(q, size_t{0}, [](size_t) {});
            const auto produce = [&](float next_bias) {
                struct Args { const float* x; const float* w; float* y; float bias; };
                const Args args{x, w, y, next_bias};
                q.parallel_for(sycl::range<1>(count), [=](sycl::id<1> id) {
                    float sum = 0;
                    for (size_t j = 0; j < inner; ++j) sum = sycl::fma(args.x[j], args.w[j], sum);
                    args.y[id[0]] = sum + args.bias;
                });
            };
            produce(bias);
            if (runtime_stream) {
                require(rt::readback_async(output.data(), y, count * sizeof(float), *runtime_stream));
                require(rt::stream_sync(*runtime_stream));
            } else strata::core::readback(q, output.data(), y, count * sizeof(float));
            check(output == expected, "readback did not observe the producer");
            std::fill_n(host, count, -12345.0f);
            std::fill(expected.begin(), expected.end(), dot + bias + 64.0f);
            produce(bias + 64.0f);
            std::atomic_ref<unsigned>(*flag).store(0, std::memory_order_relaxed);
            ds4::Xfer transfer;
            transfer.n = 1; transfer.src[0] = y; transfer.dst[0] = host;
            transfer.bytes[0] = count * sizeof(float);
            ds4::xfer_put(transfer, flag, 1, nullptr, q);
            q.wait_and_throw();
            check(std::atomic_ref<unsigned>(*flag).load(std::memory_order_acquire) == 1, "xfer flag missing");
            check(std::equal(expected.begin(), expected.end(), host), "xfer put payload differs");
            q.fill(y, -54321.0f, count);
            transfer.src[0] = host; transfer.dst[0] = y;
            ds4::xfer_get(transfer, flag, 1, q);
            strata::core::readback(q, output.data(), y, count * sizeof(float));
            check(output == expected, "xfer get payload differs");
        }
    }
    std::printf("ordered runtime=%d: PASS\n", runtime_stream != nullptr);
}
}  // namespace

int main() try {
    strata::core::restrict_hip_to_selector();
    strata::core::no_hip_resource_cache();
    if (!strata::core::Runtime::count()) { std::puts("SKIP: no SYCL GPU"); return 77; }
    Stream stream;
    auto& q = static_cast<sycl::queue&>(stream.value);
    if (!q.get_device().has(sycl::aspect::usm_device_allocations) ||
        !q.get_device().has(sycl::aspect::usm_host_allocations)) {
        std::puts("SKIP: device and host USM required"); return 77;
    }
    for (bool odd : {false, true}) for (bool host : {false, true}) pitched(stream.value, odd, host);
    ordered(q, &stream.value);
    sycl::queue raw(q.get_context(), q.get_device(), sycl::property::queue::in_order{});
    ordered(raw, nullptr);
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "dsv41 readback: %s\n", e.what());
    return 1;
}
