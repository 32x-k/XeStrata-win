// SPDX-FileCopyrightText: 2026 Peasant Smith and the Project Maya contributors
// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// Host-only checks for explicit and capacity-derived prompt path selection.
#include "ds4_pick_rule.hpp"
#include <cstdio>

int main() {
    using namespace strata::ds4;
    int fails = 0;
    for (int capacity : {1, 4, 16, 32}) {
        const int n = prompt_flow_min(nullptr, capacity);
        if (n != capacity || prompt_uses_flow(n - 1, n) || !prompt_uses_flow(n, n)) ++fails;
        if (prompt_flow_min("", capacity) != capacity || prompt_flow_min("7", capacity) != 7) ++fails;
    }
    if (prompt_flow_min("0") != 1 || prompt_flow_min("-1") != 1 || prompt_flow_min(nullptr, 0) != 1) ++fails;
    PromptPathFacts f;
    f.window_capacity = 16; f.attention_window = 128; f.expert_width = 2304;
    f.max_group = 1024; f.subgroup = 32;
    f.single_device = true; f.resident = true; f.matrix = false; f.one_bit = true;
    if (prompt_flow_min(nullptr, f) != 64 || prompt_flow_min("", f) != 64 ||
        prompt_flow_min("7", f) != 7 || prompt_flow_min("0", f) != 1) ++fails;
    f.matrix = true;
    if (prompt_flow_min(nullptr, f) != 64) ++fails;
    for (int missing = 0; missing < 4; ++missing) {
        auto fallback = f;
        if (missing == 0) fallback.resident = false;
        if (missing == 1) fallback.single_device = false;
        if (missing == 2) fallback.one_bit = false;
        if (missing == 3) fallback.subgroup = 0;
        if (prompt_flow_min(nullptr, fallback) != 16 || prompt_flow_min("7", fallback) != 7) ++fails;
    }
    f.max_group = 256;
    if (prompt_flow_min(nullptr, f) != 16) ++fails;
    f.max_group = 1024; f.attention_window = 32;
    if (prompt_flow_min(nullptr, f) != 32) ++fails;
    f.attention_window = 128; f.expert_width = 16;
    if (prompt_flow_min(nullptr, f) != 16) ++fails;
    std::printf("prompt path: %s\n", fails ? "FAIL" : "OK");
    return fails ? 1 : 0;
}
