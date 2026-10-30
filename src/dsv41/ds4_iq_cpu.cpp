// SPDX-FileCopyrightText: 2026 Peasant Smith and the Project Maya contributors
// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// The CPU lane dispatches to XeStrata's ggml and trellis row implementations.
#include "ds4_gguf_cpu.hpp"

namespace strata::ds4::gguf {
// (no ggml-cpu lane for the low-bit types here; the trellis types have their own: ds4_tq_cpu.cpp)
bool iq_cpu_fmt(int tg, int td, int n_embd, int n_ff, IqCpu& f, std::string& err) {
    if (tq_cpu_type(tg) && tq_cpu_type(td)) return tq_cpu_fmt(tg, td, n_embd, n_ff, f, err);
    if (gg_cpu_type(tg) && gg_cpu_type(td)) return gg_cpu_fmt(tg, td, n_embd, n_ff, f, err);
    err = "this build has no ggml-cpu (STRATA_NATIVE_EXPERTS=OFF): the low-bit experts CPU lane is not in it";
    return false;
}
void iq_cpu_quant(int act, const float* x, int k, void* dst) {
    if (act == tq_cpu_act()) tq_cpu_quant(x, k, dst);
    else gg_cpu_quant(act, x, k, dst);
}
void iq_cpu_gate_up(const IqCpu& f, const uint8_t* gate, const uint8_t* up, const void* const* act, int nt, float* const* ff,
                    int r0, int r1, float limit) {
    if (tq_cpu_type(f.tg)) tq_cpu_gate_up(f, gate, up, act, nt, ff, r0, r1, limit);
    else gg_cpu_gate_up(f, gate, up, act, nt, ff, r0, r1, limit);
}
void iq_cpu_down(const IqCpu& f, const uint8_t* down, const void* const* hq, int nt, float* const* out, int r0, int r1) {
    if (tq_cpu_type(f.td)) tq_cpu_down(f, down, hq, nt, out, r0, r1);
    else gg_cpu_down(f, down, hq, nt, out, r0, r1);
}
bool iq_cpu_own_rows(int) { return false; }

}  // namespace strata::ds4::gguf
