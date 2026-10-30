// SPDX-FileCopyrightText: 2026 Peasant Smith and the Project Maya contributors
// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// Split-KV sparse attention: Strata, used with the DeepSeek attention of Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// (the surrounding types, AttnRows and the packed-row formats, are theirs; see third_party/main/project-maya/NOTICE.md).
#pragma once
#include "ds4_kernels.cuh"
#include <stdexcept>

namespace strata::ds4::attnsplit {

struct AttnSplitScratch {
    float* part = nullptr;
    float* ml = nullptr;
    int nsplit = 0, max_t = 0;
};

// The public split-attention call has a 512-component model head. Its portable
// implementation shares sparse_attn's device-sized reduction and does not need
// the old architecture-specific split tiles or their temporary storage.
template <int G, int R>
void sparse_attn_split(const float* q, const uint16_t* win, const uint8_t* ckv, size_t ckv_row_bytes,
                       const int* sel, int sel_stride, const AttnRows& ar, const float* sink, int heads, int T,
                       float scale, float* out, const AttnSplitScratch& sc, int total_max, sycl::queue& s) {
    (void)sc;
    (void)total_max;
    static_assert(G > 0 && R > 0);
    constexpr int d = 512;
    if (ckv_row_bytes != kv_row_bytes(kKvFp4, d)) throw std::invalid_argument("ds4 attention: invalid packed row stride");
    sparse_attn(q, win, ckv, sel, sel_stride, ar, sink, heads, d, T, scale, out, s, kKvFp4);
}

}  // namespace strata::ds4::attnsplit
