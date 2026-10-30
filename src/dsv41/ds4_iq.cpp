// SPDX-FileCopyrightText: 2026 Peasant Smith and the Project Maya contributors
// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// SPDX-FileCopyrightText: 2023-2026 The ggml authors

// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// src/ds4/ds4_iq.cu - a GGUF's low-bit routed experts on the device (see ds4_iq.cuh).  The block layouts and grids are
// ggml's (ggml-common.h, MIT, third_party/ggml/LICENSE); the sign and grid tricks are llama.cpp's vecdotq.cuh.
#include "ds4_quant_sycl.hpp"
#include "ds4_numerics.cuh"
#include "ds4_iq_decode.hpp"

namespace strata::ds4::gguf::iq {
using namespace sycl_decode;

bool gate_up(const Args& a, int groups, int ne_max, sycl::queue& s) {
    if (is_tq(a.tg)) return tq::gate_up(a, groups, ne_max, s);
    return dispatch(a.tg, [&](auto t) {
        portable::token<true>(a, groups, ne_max, row_bytes_t<t.value>(a.n_embd), Decoder<t.value>{}, s);
    });
}
bool down(const Args& a, int groups, int ne_max, sycl::queue& s) {
    if (is_tq(a.td)) return tq::down(a, groups, ne_max, s);
    return dispatch(a.td, [&](auto t) {
        portable::token<false>(a, groups, ne_max, row_bytes_t<t.value>(a.n_ff), Decoder<t.value>{}, s);
    });
}
bool set_gate_up(const SetArgs& a, sycl::queue& s) {
    if (is_tq(a.tg)) return tq::set_gate_up(a, s);
    return dispatch(a.tg, [&](auto t) { portable::set<true>(a, row_bytes_t<t.value>(a.n_embd), Decoder<t.value>{}, s); });
}
bool set_down(const SetArgs& a, sycl::queue& s) {
    if (is_tq(a.td)) return tq::set_down(a, s);
    return dispatch(a.td, [&](auto t) { portable::set<false>(a, row_bytes_t<t.value>(a.n_ff), Decoder<t.value>{}, s); });
}
// The portable set path handles every row count. No light launch means the caller leaves skip_le at zero.
bool light_gate_up(const SetArgs&, sycl::queue&) { return false; }
bool light_down(const SetArgs&, sycl::queue&) { return false; }
bool hipk_force(int) { return false; }
long long hipk_launches() { return -1; }

bool mv(const MvJob* jobs, int n, sycl::queue& s) {
    for (int i = 0; i < n; ++i) if (!supported(jobs[i].type)) return false;
    for (int i = 0; i < n; ++i) {
        const MvJob j = jobs[i];
        if (is_tq(j.type)) { if (!tq::mv(&j, 1, s)) return false; }
        else if (!dispatch(j.type, [&](auto t) { portable::mv(j, Decoder<t.value>{}, s); })) return false;
    }
    return true;
}
bool dequant_rows(int type, const void* raw, size_t ld, int k, int n, uint16_t* dst, bool bf16, sycl::queue& s) {
    return dispatch(type, [&](auto t) {
        sycl_detail::parallel(s, (size_t) n * k, [=](size_t i) {
            const float v = Decoder<t.value>{}.value((const uint8_t*) raw + (i / k) * ld, (int) (i % k));
            dst[i] = bf16 ? bf16_rn(v) : sycl_detail::f16_bits(v);
        });
    });
}
bool f16_rows_ok(int type) { return supported(type) && !is_tq(type) && type != kQ5_K && type != kQ6_K; }
bool f16_rows(int type, const uint8_t* w, int k, int n, uint16_t* dst, int ldd, sycl::queue& s) {
    if (!f16_rows_ok(type)) return false;
    return dispatch(type, [&](auto t) {
        const size_t rb = row_bytes_t<t.value>(k);
        sycl_detail::parallel(s, (size_t) n * k, [=](size_t i) {
            dst[(i / k) * ldd + i % k] = sycl_detail::f16_bits(Decoder<t.value>{}.value(w + (i / k) * rb, (int) (i % k)));
        });
    });
}
void gather_f16(const uint16_t* x, int ldx, const int* idx, int n, int k, uint16_t* dst, sycl::queue& s) {
    sycl_detail::parallel(s, (size_t) n * k, [=](size_t i) { dst[i] = x[(size_t) idx[i / k] * ldx + i % k]; });
}
void swiglu_f16(const float* gu, int ld, int n, int ff, float limit, uint16_t* h, int ldh, sycl::queue& s) {
    sycl_detail::parallel(s, (size_t) n * ff, [=](size_t i) {
        const size_t row = i / ff, col = i % ff;
        h[row * ldh + col] = sycl_detail::f16_bits(portable::swiglu(gu[row * ld + col], gu[row * ld + ff + col], limit));
    });
}
}  // namespace strata::ds4::gguf::iq
