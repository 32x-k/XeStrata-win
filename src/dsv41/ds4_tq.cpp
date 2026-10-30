// SPDX-FileCopyrightText: 2026 Peasant Smith and the Project Maya contributors
// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// Strata. The trellis routed experts, retaining the file's circular bit stream and codebooks.
// The portable SYCL path uses the same int8 activations and F32 scales as the CPU lane.
#include "ds4_quant_sycl.hpp"
#include "ds4_tq_decode.hpp"
namespace strata::ds4::gguf::iq::tq {
using namespace sycl_decode;

bool gate_up(const Args& a, int groups, int ne_max, sycl::queue& s) {
    if (!is_tq(a.tg)) return false;
    portable::token<true>(a, groups, ne_max, row_bytes(a.tg, a.n_embd), decoder(a.tg), s);
    return true;
}
bool down(const Args& a, int groups, int ne_max, sycl::queue& s) {
    if (!is_tq(a.td)) return false;
    portable::token<false>(a, groups, ne_max, row_bytes(a.td, a.n_ff), decoder(a.td), s);
    return true;
}
bool set_gate_up(const SetArgs& a, sycl::queue& s) {
    if (!is_tq(a.tg)) return false;
    portable::set<true>(a, row_bytes(a.tg, a.n_embd), decoder(a.tg), s);
    return true;
}
bool set_down(const SetArgs& a, sycl::queue& s) {
    if (!is_tq(a.td)) return false;
    portable::set<false>(a, row_bytes(a.td, a.n_ff), decoder(a.td), s);
    return true;
}
bool mv(const MvJob* jobs, int n, sycl::queue& s) {
    for (int i = 0; i < n; ++i) if (!is_tq(jobs[i].type)) return false;
    for (int i = 0; i < n; ++i) portable::mv(jobs[i], decoder(jobs[i].type), s);
    return true;
}
}  // namespace strata::ds4::gguf::iq::tq
