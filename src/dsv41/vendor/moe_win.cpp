// SPDX-FileCopyrightText: 2026 Peasant Smith and the Project Maya contributors
// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/main/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// Extracted from Project Maya's src/kernels/cuda/glm_fast.cu (MIT).
#include "strata/dsv41/moe_win.hpp"
#include "../ds4_sycl.hpp"
#include <stdexcept>

namespace strata::dsv41k::glmf {
void moe_win_group(const MoeWin& w, sycl::queue& s) {
    strata::ds4::sycl_detail::require_in_order(s);
    if (w.T < 0 || w.k < 0 || w.k > 8) throw std::invalid_argument("dsv41: invalid expert window");
    // Scratch is supplied by the caller. A single task preserves first-appearance order without private tables.
    s.single_task([=] {
        int ng = 0, entries = 0;
        for (int t = 0; t < w.T; ++t)
            for (int i = 0; i < w.k; ++i) {
                const unsigned long long p = w.plan_ptr[t * 8 + i];
                if (!p) continue;
                int g = 0;
                while (g < ng && w.grp_ptr[g] != p) ++g;
                if (g != ng) continue;
                w.grp_ptr[ng] = p;
                w.grp_start[ng++] = entries;
                for (int tt = 0; tt < w.T; ++tt)
                    for (int ii = 0; ii < w.k; ++ii)
                        if (w.plan_ptr[tt * 8 + ii] == p) w.grp_ent[entries++] = tt * 8 + ii;
            }
        w.grp_start[ng] = entries;
        *w.n_groups = ng;
    });
}
}  // namespace strata::dsv41k::glmf
