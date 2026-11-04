// SPDX-FileCopyrightText: 2026 Peasant Smith and the Project Maya contributors
// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// Prompt path selection from the caller's window capacity. Explicit DS4_PICK_MIN wins.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <limits>

namespace strata::ds4 {

inline int prompt_flow_min(const char* env, int window_capacity = 1) {
    if (env == nullptr || *env == 0) return std::max(1, window_capacity);
    const long requested = std::strtol(env, nullptr, 10);
    return (int) std::max(1L, std::min((long) std::numeric_limits<int>::max(), requested));
}
struct PromptPathFacts {
    int window_capacity = 1, attention_window = 0, expert_width = 0;
    size_t max_group = 0, subgroup = 0;
    bool single_device = false, resident = false, matrix = true, one_bit = false;
};

inline int prompt_flow_min(const char* env, const PromptPathFacts& f) {
    const int capacity = std::max(1, f.window_capacity);
    if (env != nullptr && *env != 0) return prompt_flow_min(env, capacity);
    if (!f.single_device || !f.resident || !f.one_bit ||
        f.attention_window <= 0 || f.expert_width <= 0 || !f.subgroup) return capacity;
    // A conservative coverage limit: work-items per row of a complete window,
    // aligned to a supported subgroup and bounded by the model's attention window
    // and expert width. The kernels still choose their own tiles.
    // Measured on R9700 with resident IQ1_S/IQ1_M experts (2026-11-04): windows
    // gained most at 16/32/48, were close at 64/96, and lost from 128. No measured
    // batch size is stored here. Generic FP16 matrix support does not describe
    // this IQ1 path. Streamed, other-format and unknown configurations keep the
    // capacity default; other GPUs' crossover points remain to be measured.
    const size_t cover = std::min({f.max_group / size_t(capacity), size_t(f.attention_window), size_t(f.expert_width)});
    const size_t aligned = cover / f.subgroup * f.subgroup;
    return int(std::max(size_t(capacity), std::min(aligned, size_t(std::numeric_limits<int>::max()))));
}
inline bool prompt_uses_flow(int tokens, int flow_min) { return tokens >= flow_min; }

}  // namespace strata::ds4
