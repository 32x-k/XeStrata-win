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
    std::printf("prompt path: %s\n", fails ? "FAIL" : "OK");
    return fails ? 1 : 0;
}
