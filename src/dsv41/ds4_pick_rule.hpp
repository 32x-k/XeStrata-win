// SPDX-FileCopyrightText: 2026 Peasant Smith and the Project Maya contributors
// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// Prompt path selection from the caller's window capacity. Explicit DS4_PICK_MIN wins.
#pragma once

#include <algorithm>
#include <cstdlib>
#include <limits>

namespace strata::ds4 {

inline int prompt_flow_min(const char* env, int window_capacity = 1) {
    if (env == nullptr || *env == 0) return std::max(1, window_capacity);
    const long requested = std::strtol(env, nullptr, 10);
    return (int) std::max(1L, std::min((long) std::numeric_limits<int>::max(), requested));
}
inline bool prompt_uses_flow(int tokens, int flow_min) { return tokens >= flow_min; }

}  // namespace strata::ds4
