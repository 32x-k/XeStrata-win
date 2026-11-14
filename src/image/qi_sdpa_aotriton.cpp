// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/qi_sdpa_aotriton.cpp - see qi_sdpa.hpp: aotriton's flash attention on AMD GPUs (MIT; its library and its
// kernels' images beside the engine), on the queue's HIP stream.  A module of its own (libstrata_sdpa_aotriton.so,
// strata_sdpa_aotriton.dll), which the image engine opens on an AMD GPU only: aotriton needs AMD's HIP runtime, which
// a machine with other makers' GPUs need not have, and the engine must start there.

#include <aotriton/flash.h>
#include <sycl/ext/oneapi/backend/hip.hpp>

#include "qi_sdpa.hpp"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdio>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace strata::image::sdpa {

namespace {

std::mutex g_mu;

struct Aot {
    bool failed = false;
    int32_t* counter = nullptr;   // the persistent kernels' work counter, zeroed before each call
};
std::unordered_map<sycl::device, Aot> g_aot;

}  // namespace

bool aotriton(sycl::queue& q, const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t q0, int64_t q1,
              int64_t T, int64_t heads, int64_t hd, uint16_t* out16, bool causal) {
    namespace ao = aotriton;
    std::lock_guard<std::mutex> lock(g_mu);
    Aot& d = g_aot[q.get_device()];
    if (d.failed) return false;
    if (d.counter == nullptr) d.counter = sycl::malloc_device<int32_t>(1, q);
    const uint64_t row = (uint64_t) (heads * hd);
    auto view = [&](const uint16_t* p, int64_t rows) {
        return ao::TensorView<4>(reinterpret_cast<intptr_t>(p),
                                 std::array<uint64_t, 4>{1, (uint64_t) heads, (uint64_t) rows, (uint64_t) hd},
                                 std::array<uint64_t, 4>{(uint64_t) rows * row, (uint64_t) hd, row, 1}, ao::kFloat16);
    };
    ao::v3::flash::attn_fwd_params p;
    p.Q = view(q16 + q0 * (int64_t) row, q1 - q0);
    p.K = view(k16, T);
    p.V = view(v16, T);
    p.Out = view(out16, q1 - q0);
    p.B = ao::TensorView<4>::get_null_tensor(ao::kFloat16);
    p.A = ao::TensorView<2>::get_null_tensor(ao::kFloat16);
    p.L = ao::TensorView<2>::get_null_tensor(ao::kFloat32);
    p.Sm_scale = 1.0f / std::sqrt((float) hd);
    p.dropout_p = 0.0f;
    p.philox_seed_ptr = ao::TensorView<0>::get_null_tensor(ao::kUInt64);
    p.philox_offset1 = ao::TensorView<0>::get_null_tensor(ao::kUInt64);
    p.philox_offset2 = 0;
    p.philox_seed_output = ao::TensorView<0>::get_null_tensor(ao::kUInt64);
    p.philox_offset_output = ao::TensorView<0>::get_null_tensor(ao::kUInt64);
    p.encoded_softmax = ao::TensorView<4>::get_null_tensor(ao::kFloat16);
    p.persistent_atomic_counter = ao::TensorView<0>(reinterpret_cast<intptr_t>(d.counter), ao::kInt32);
    // causal: the windowed form, both edges the top-left aligned diagonal (the binary kernels have no other causal)
    p.causal_type = causal ? ao::v3::flash::CausalType::WindowedAttention : ao::v3::flash::CausalType::None;
    p.window_left = causal ? ao::v3::flash::WindowValue::TopLeftAligned : 0;
    p.window_right = causal ? ao::v3::flash::WindowValue::TopLeftAligned : 0;
    p.Max_seqlen_q = (int32_t) (q1 - q0);
    p.Max_seqlen_k = (int32_t) T;
    q.memset(d.counter, 0, sizeof(int32_t));
    const hipStream_t hs = sycl::get_native<sycl::backend::ext_oneapi_hip>(q);
    const hipError_t e = ao::v3::flash::attn_fwd(p, ao::v3::flash::attn_fwd_params::kVersion, ao::Stream(hs));
    if (e != hipSuccess) {
        std::fprintf(stderr, "strata-image: aotriton's attention is not taken on this GPU (HIP error %d): the engine's "
                             "own kernel\n", (int) e);
        d.failed = true;
        return false;
    }
    return true;
}

}  // namespace strata::image::sdpa

// the module's entry, which qi_sdpa.cpp looks up by this name: q a sycl::queue of the same SYCL runtime
#if defined(_WIN32)
#define STRATA_SDPA_EXPORT extern "C" __declspec(dllexport)
#else
#define STRATA_SDPA_EXPORT extern "C" __attribute__((visibility("default")))
#endif
STRATA_SDPA_EXPORT bool strata_sdpa_aotriton(void* q, const uint16_t* q16, const uint16_t* k16, const uint16_t* v16,
                                             int64_t q0, int64_t q1, int64_t T, int64_t heads, int64_t hd,
                                             uint16_t* out16, bool causal) {
    return strata::image::sdpa::aotriton(*static_cast<sycl::queue*>(q), q16, k16, v16, q0, q1, T, heads, hd, out16,
                                         causal);
}
