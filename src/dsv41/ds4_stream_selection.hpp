// SPDX-FileCopyrightText: 2026 Peasant Smith and the Project Maya contributors
// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
#pragma once
#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <vector>

namespace strata::ds4 {
inline int stream_selection_limit(const char* enabled, const char* rows, bool files) {
    if (enabled != nullptr) return std::strtol(enabled, nullptr, 10) != 0 ? INT_MAX : 0;
    if (rows != nullptr) return int(std::clamp(std::strtol(rows, nullptr, 10), 0L, long(INT_MAX)));
    return files ? INT_MAX : 0;
}

// The chunk policy is immutable while read-ahead runs. Only the compute thread
// publishes routing and calls the staging pool, whose jobs finish before return.
struct StreamSelection {
    std::vector<uint8_t> chunks, slots;
    int layer = -1, chunk = -1;
    bool selective(int k) const { return k >= 0 && size_t(k) < chunks.size() && chunks[size_t(k)] != 0; }
    bool ready(int il, int k) const { return !selective(k) || (layer == il && chunk == k); }
    void routed(int il, int k, const std::vector<int>& bounds) {
        slots.resize(bounds.empty() ? 0 : bounds.size() - 1);
        for (size_t v = 0; v < slots.size(); ++v) slots[v] = bounds[v + 1] > bounds[v];
        layer = il;
        chunk = k;
    }
    bool selected(size_t v) const { return v < slots.size() && slots[v] != 0; }
};
}  // namespace strata::ds4
