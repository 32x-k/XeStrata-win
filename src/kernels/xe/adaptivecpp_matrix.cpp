// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Matrix-extension fallbacks for AdaptiveCpp, which does not expose oneAPI joint_matrix.
#include "strata/kernels/iq_mmq.hpp"
#include "strata/kernels/matrix_report.hpp"
#include "strata/kernels/prompt_attn.hpp"
#include "strata/kernels/qsa_prompt_attn.hpp"
#include "strata/kernels/xmx_gemm.hpp"

#include "strata/core/runtime.hpp"
#include "dp4a_gemm.hpp"
#include "mma_gemm.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

void finish(void* stream, const sycl::event& event, const char* what) {
    if (!stream) core::Runtime::get().wait(event, what);
}

sycl::event scalar_gemm(sycl::queue& q, XmxType type, const uint16_t* x, const uint16_t* w, float* y,
                        int64_t rows, int64_t n, int64_t k, int64_t ldy, bool accumulate) {
    const bool f16 = type == XmxType::f16;
    return q.parallel_for(sycl::range<2>((size_t) rows, (size_t) n), [=](sycl::id<2> id) {
        const int64_t r = (int64_t) id[0], col = (int64_t) id[1];
        float sum = 0.0f;
        for (int64_t i = 0; i < k; ++i) {
            const auto value = [f16](uint16_t bits) {
                return f16 ? (float) sycl::bit_cast<sycl::half>(bits)
                           : sycl::bit_cast<float>((uint32_t) bits << 16);
            };
            sum = sycl::fma(value(x[r * k + i]), value(w[col * k + i]), sum);
        }
        y[r * ldy + col] = accumulate ? y[r * ldy + col] + sum : sum;
    });
}

}  // namespace

bool xmx_available(XmxType) { return false; }
bool f16_matrix_engines() { return false; }
bool f16_only_matrix_engines() { return false; }
const char* gemm_path(XmxType) { return "DP4a"; }

bool xmx_gemm_ok(const void* x, const void* w, const float* y, int64_t n, int64_t k, int64_t ldy) noexcept {
    const auto aligned = [](const void* p) { return ((uintptr_t) p & 63) == 0; };
    return n >= 16 && n % 16 == 0 && k >= 32 && k % 32 == 0 && ldy >= n && ldy % 16 == 0 &&
           aligned(x) && aligned(w) && aligned(y);
}

void xmx_gemm(XmxType type, const uint16_t* x, const uint16_t* w, float* y, int64_t rows, int64_t n, int64_t k,
              int64_t ldy, void* stream, bool accumulate) {
    if (rows <= 0 || n <= 0) return;
    auto& q = core::Runtime::get().stream(stream);
    const sycl::event event = accumulate || k % 32 != 0
        ? scalar_gemm(q, type, x, w, y, rows, n, k, ldy, accumulate)
        : xe::dp4a_gemm(q, type == XmxType::bf16, x, w, y, rows, n, k, ldy);
    finish(stream, event, "xmx_gemm fallback");
}

void gemm_rows(XmxType type, const uint16_t* x, const uint16_t* w, float* y, int64_t rows, int64_t n, int64_t k,
               int64_t ldy, void* stream, bool accumulate) {
    xmx_gemm(type, x, w, y, rows, n, k, ldy, stream, accumulate);
}

void xmx_gemm_grouped(const uint16_t* x, const uint16_t* w, int64_t w_stride, float* y,
                      const int32_t* bounds, int groups, int64_t max_rows, int64_t n, int64_t k, void* stream) {
    if (groups <= 0 || max_rows <= 0) return;
    auto& q = core::Runtime::get().stream(stream);
    finish(stream, xe::dp4a_gemm_grouped(q, x, w, w_stride, y, bounds, groups, max_rows, n, k),
           "xmx_gemm_grouped fallback");
}

bool no_bf16_mma() { return false; }

bool iq_mmq_usable(sycl::queue&) { return false; }
bool iq_mmq_dp4a_usable(sycl::queue&) { return false; }
bool iq_mmq_type_ok(int) { return false; }
size_t iq_mmq_row_bytes(int, int64_t) { return 0; }
size_t iq_mmq_act_bytes(int64_t, int64_t) { return 0; }
bool iq_mmq_type_sums(int) { return false; }
bool iq_mmq_type_dup(int) { return false; }
sycl::event iq_mmq_quantize(sycl::queue& q, const float*, const int32_t*, void*, int64_t, int64_t, int64_t, bool, bool) {
    return q.single_task([] {});
}
sycl::event iq_mmq_grouped(sycl::queue& q, int, const void*, size_t, int64_t, int64_t, int, const void*, int64_t,
                           const int32_t*, int64_t, const int32_t*, float*, int64_t, const uint8_t* const*, size_t,
                           int64_t) {
    return q.single_task([] {});
}

bool prompt_attn(const float*, const uint16_t*, const uint16_t*, const int32_t*, int64_t, int64_t, const QsaShapes&,
                 float*, void*) {
    return false;
}

bool qsa_prompt_attn_batch(const float*, const QsaAttnPools&, const int32_t*, const int32_t*, int64_t,
                           const QsaShapes&, float*, int64_t, void*) {
    return false;
}

namespace xe {
bool mma_usable(const sycl::queue&, bool) { return false; }
bool mma_forced() { return false; }
bool mma_shape_is(const sycl::queue&, int, int, int, int) { return false; }
sycl::event mma_gemm(sycl::queue& q, bool, const uint16_t*, const uint16_t*, float*, int64_t, int64_t, int64_t,
                     int64_t, bool) {
    return q.single_task([] {});
}
sycl::event mma_gemm_grouped(sycl::queue& q, const uint16_t*, const uint16_t*, int64_t, float*, const int32_t*, int,
                             int64_t, int64_t, int64_t) {
    return q.single_task([] {});
}
}  // namespace xe

}  // namespace strata::kernels
