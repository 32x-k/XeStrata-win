// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

sycl::vec<uint16_t, 2> packed_half_product(uint16_t scale_bits, int c0, int c1) {
    // Preserve the software conversion's NaN payload/sign and infinity rules.
    if ((scale_bits & 0x7c00u) == 0x7c00u) {
        const float scale = matrix_half_value(scale_bits);
        return {matrix_half_bits(scale * float(c0)), matrix_half_bits(scale * float(c1))};
    }
    const auto scale = sycl::bit_cast<sycl::half>(scale_bits);
    const sycl::vec<sycl::half, 2> left{scale, scale}, right{sycl::half(float(c0)), sycl::half(float(c1))};
    return (left * right).template as<sycl::vec<uint16_t, 2>>();
}

[[maybe_unused]] sycl::vec<uint16_t, 8> packed_expand(uint16_t scale_bits, uint32_t grid, bool negative) {
    sycl::vec<uint16_t, 8> result;
#pragma unroll
    for (int j = 0; j < 8; j += 2) {
        const int a = 8 * int((grid >> ((j & 3) * 8 + (j >> 2) * 4)) & 15u) - 7 - (negative ? 2 : 0);
        const int b = 8 * int((grid >> (((j + 1) & 3) * 8 + ((j + 1) >> 2) * 4)) & 15u) - 7 - (negative ? 2 : 0);
        auto values = packed_half_product(scale_bits, a, b);
        result[j] = values[0]; result[j + 1] = values[1];
    }
    return result;
}

struct HalfProbeState { std::mutex mutex; bool done = false, passed = false; };
class HalfProbeKernel;

bool half_decode_qualified(sycl::queue& s) {
    if (!s.get_device().has(sycl::aspect::fp16)) return false;
    static core::PerDevice<std::shared_ptr<HalfProbeState>> per_device;
    const auto& state = per_device.get(s.get_device(), [] { return std::make_shared<HalfProbeState>(); });
    std::lock_guard<std::mutex> lock(state->mutex);
    if (state->done) return state->passed;
#if defined(SYCL_EXT_ONEAPI_GRAPH)
    // Defer the one-time probe until an ordinary submission can run it.
    if (s.ext_oneapi_get_state() == sycl::ext::oneapi::experimental::queue_state::recording) return false;
#endif
    sycl::queue probe(s.get_context(), s.get_device(), sycl::property::queue::in_order{});
    uint32_t* result = sycl::malloc_device<uint32_t>(2, probe);
    if (!result) { state->done = true; std::fprintf(stderr, "ds4: IQ1 native-half probe allocation unavailable; fallback\n"); return false; }
    uint32_t host[2] = {0, 0xffffffffu};
    constexpr size_t codes = 227, pairs = (codes + 1) / 2, scales = 65536;
    try {
        probe.memcpy(result, host, sizeof(host)).wait_and_throw();
        probe.parallel_for<HalfProbeKernel>(sycl::range<1>(scales * pairs), [=](sycl::id<1> id) {
            const size_t at = id[0];
            const uint16_t bits = uint16_t(at / pairs);
            const int c0 = int(at % pairs) * 2 - 113, c1 = c0 + 1;
            auto values = packed_half_product(bits, c0, c1);
            const float scale = matrix_half_value(bits);
            for (int j = 0; j < 2; ++j) if (c0 + j <= 113) {
                const uint16_t original = matrix_half_bits(scale * float(c0 + j));
                if (values[j] != original) {
                    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                        sycl::access::address_space::global_space>(result[0]).fetch_add(1);
                    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                        sycl::access::address_space::global_space>(result[1]).fetch_min(uint32_t(size_t(bits) * codes + size_t(c0 + j + 113)));
                }
            }
        }).wait_and_throw();
        probe.memcpy(host, result, sizeof(host)).wait_and_throw();
        state->passed = host[0] == 0;
    } catch (...) { sycl::free(result, probe); throw; }
    sycl::free(result, probe);
    state->done = true;
    std::fprintf(stderr, "ds4: IQ1 native-half probe passed=%d mismatches=%u checked=%zu\n", int(state->passed), host[0], scales * codes);
    return state->passed;
}

template <bool Prefetch>
void native_half_used(const sycl::device& device) {
    static core::PerDevice<bool> reported;
    (void)reported.get(device, [] {
        std::fprintf(stderr, "ds4: IQ1 native-half used GU %s\n", Prefetch ? "prefetch" : "original");
        return true;
    });
}
