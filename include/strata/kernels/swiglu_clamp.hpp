// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/kernels/swiglu_clamp.hpp - the routed experts' SwiGLU limit.
//
// GLM-5.3-Flash clamps its experts' SwiGLU (glm5-next.swiglu_clamp_exp, ggml_swiglu_clamp): with a limit L > 0 the gate
// is cut above at L and the up value to [-L, L] before silu(gate) * up.  Every other model has none (0).  A process
// setting, as the GDN output gate (gdn_set_silu_gate): set at load, before any graph is recorded; the GPU kernels read
// it when they are launched, so a recorded graph keeps the value it was recorded with.
#pragma once

#include <atomic>
#include <cmath>

namespace strata::kernels {

inline std::atomic<float> g_expert_swiglu_limit{0.0f};

inline void expert_swiglu_set_limit(float limit) {
    g_expert_swiglu_limit.store(limit > 0.0f ? limit : 0.0f, std::memory_order_relaxed);
}
inline float expert_swiglu_limit() { return g_expert_swiglu_limit.load(std::memory_order_relaxed); }

/// gate <= lim, -lim <= up <= lim when lim > 0 (host and device).
inline void swiglu_clamp(float& gate, float& up, float lim) {
    if (lim > 0.0f) {
        gate = gate < lim ? gate : lim;
        up = up < -lim ? -lim : (up > lim ? lim : up);
    }
}

/// The CPU kernels' silu(gate) * up after the clamp: with lim 0, the expression they always used.
inline float swiglu_host(float gate, float up, float lim) {
    swiglu_clamp(gate, up, lim);
    return (gate / (1.f + std::exp(-gate))) * up;
}

}  // namespace strata::kernels
