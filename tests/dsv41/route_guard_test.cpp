// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// Upstream tests/dsv41/ds4_route_guard_test.cu: the router on a non-finite gate input - route() and route_publish()
// must give valid, distinct picks (no read past the expert tables) whatever the logits hold, and the same picks as
// the host's top-k when they are finite.  The SYCL port also checks router_nonfinite_take() - the device's flag the
// fallback pick sets, upstream's g_router_bad - once per run and cleared by the take.
#include "ds4_kernels.cuh"
#include "sycl_test.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

namespace {
namespace ds = strata::ds4;

int fails = 0;
void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++fails;
}

}  // namespace

int main() try {
    if (sycl::device::get_devices(sycl::info::device_type::gpu).empty()) return 77;
    sycl::queue& q = dsv41_test::queue();
    const int n = 384, k = 6, T = 16;
    std::vector<float> bias(1024, 0.0f), logits((size_t) T * 1024, 0.0f);
    for (int i = 0; i < n; ++i) bias[(size_t) i] = 0.01f * (float) ((i * 37) % 17);
    std::vector<unsigned long long> vram(1024, 0ull);
    for (int i = 0; i < n; i += 2) vram[(size_t) i] = 0x1000ull + (unsigned long long) i;

    float* d_logits = nullptr;
    float* d_bias = nullptr;
    float* d_w = nullptr;
    int* d_idx = nullptr;
    unsigned long long* d_vram = nullptr;
    unsigned long long* d_plan = nullptr;
    dsv41_test::device_alloc(d_logits, logits.size() * 4);
    dsv41_test::device_alloc(d_bias, 1024 * 4);
    dsv41_test::device_alloc(d_w, (size_t) T * ds::kPicks * 4);
    dsv41_test::device_alloc(d_idx, (size_t) T * ds::kPicks * 4);
    dsv41_test::device_alloc(d_vram, 1024 * 8);
    dsv41_test::device_alloc(d_plan, (size_t) T * ds::kPicks * 8);
    q.memcpy(d_bias, bias.data(), 1024 * 4);
    q.memcpy(d_vram, vram.data(), 1024 * 8);
    q.wait();

    const float nan = std::nanf(""), inf = INFINITY;
    // STRATA_DS4_ROUTE_FINITE_ONLY=1: skip the non-finite cases (an old build faults in them, and the context with it)
    const int modes = std::getenv("STRATA_DS4_ROUTE_FINITE_ONLY") != nullptr ? 1 : 6;
    for (int mode = 0; mode < modes; ++mode) {
        const char* names[] = {"finite", "all NaN", "+inf everywhere", "-inf everywhere", "half NaN",
                               "fewer than k finite"};
        unsigned seed = 12345u;
        for (int t = 0; t < T; ++t)
            for (int i = 0; i < 1024; ++i) {
                seed = seed * 1664525u + 1013904223u;
                float v = ((float) (seed >> 8) / (float) (1 << 24) - 0.5f) * 8.0f;
                if (i >= n) v = 0.0f;
                if (mode == 1) v = nan;
                if (mode == 2) v = inf;
                if (mode == 3) v = -inf;
                if (mode == 4 && (i & 1)) v = nan;
                if (mode == 5 && i >= 3) v = nan;
                logits[(size_t) t * 1024 + (size_t) i] = v;
            }
        q.memcpy(d_logits, logits.data(), logits.size() * 4);
        q.wait();
        for (int which = 0; which < 2; ++which) {
            std::vector<int> prefill((size_t) T * ds::kPicks, -7);
            q.memcpy(d_idx, prefill.data(), prefill.size() * 4);
            if (which == 0)
                ds::route(d_logits, d_bias, n, k, T, 1.5f, d_idx, d_w, q);
            else
                ds::route_publish(d_logits, d_bias, n, k, T, 1.5f, d_vram, nullptr, 0, d_idx, d_w, d_plan, nullptr, 0, q);
            char what[160];
            try {
                q.wait_and_throw();
                std::snprintf(what, sizeof what, "%s: %s runs without a device error", names[mode],
                              which ? "route_publish" : "route");
                check(true, what);
            } catch (...) {
                std::snprintf(what, sizeof what, "%s: %s runs without a device error", names[mode],
                              which ? "route_publish" : "route");
                check(false, what);
                return 1;
            }
            std::vector<int> idx((size_t) T * ds::kPicks);
            q.memcpy(idx.data(), d_idx, idx.size() * 4);
            q.wait();
            bool valid = true, distinct = true, same = true;
            for (int t = 0; t < T; ++t) {
                for (int r = 0; r < k; ++r) {
                    const int e = idx[(size_t) t * ds::kPicks + r];
                    valid = valid && e >= 0 && e < n;
                    for (int p = 0; p < r; ++p) distinct = distinct && idx[(size_t) t * ds::kPicks + p] != e;
                }
                if (mode == 0) {   // the host's top-k of the same scores
                    std::vector<std::pair<float, int>> sc;
                    for (int i = 0; i < n; ++i) {
                        const float l = logits[(size_t) t * 1024 + (size_t) i];
                        const float sp = l > 20.0f ? l : std::log1p(std::exp(l));
                        sc.push_back({std::sqrt(sp) + bias[(size_t) i], i});
                    }
                    std::stable_sort(sc.begin(), sc.end(),
                                     [](const auto& a, const auto& b) { return a.first > b.first; });
                    for (int r = 0; r < k; ++r) same = same && sc[(size_t) r].second == idx[(size_t) t * ds::kPicks + r];
                }
            }
            if (!valid || !distinct) {
                std::printf("  token 0 picks:");
                for (int r = 0; r < ds::kPicks; ++r) std::printf(" %d", idx[(size_t) r]);
                std::printf("\n");
            }
            std::snprintf(what, sizeof what, "%s: %s picks are all in [0, %d)", names[mode],
                          which ? "route_publish" : "route", n);
            check(valid, what);
            // (with fewer than k finite scores the picked entries' -FLT_MAX ties with the start value, so the router re-takes
            //  the lowest of them: a repeat, harmless to the tables - and the weights are NaN there anyway - so distinct is
            //  checked only when at least k scores are usable)
            if (mode != 1 && mode != 5) {
                std::snprintf(what, sizeof what, "%s: %s picks are distinct", names[mode],
                              which ? "route_publish" : "route");
                check(distinct, what);
            }
            if (mode == 0) {
                std::snprintf(what, sizeof what, "finite: %s picks equal the host's top-%d",
                              which ? "route_publish" : "route", k);
                check(same, what);
            }
            if (which == 1) {
                std::vector<unsigned long long> pl((size_t) T * ds::kPicks);
                q.memcpy(pl.data(), d_plan, pl.size() * 8);
                q.wait();
                bool plan_ok = true;
                for (int t = 0; t < T; ++t)
                    for (int r = 0; r < k; ++r)
                        plan_ok = plan_ok && pl[(size_t) t * ds::kPicks + r] == vram[(size_t) idx[(size_t) t * ds::kPicks + r]];
                std::snprintf(what, sizeof what,
                              "%s: route_publish's plan pointers are the table's entries of the picks", names[mode]);
                check(plan_ok, what);
            }
            // The fallback pick sets the device's router flag once a pick could not be made (a pick needs a finite
            // score: all-NaN and fewer-than-k-finite inputs leave sentinels; +inf, -inf and half-NaN rows pick).
            const bool expect_bad = mode == 1 || mode == 5;
            std::snprintf(what, sizeof what, "%s: %s %s router_nonfinite", names[mode],
                          which ? "route_publish" : "route", expect_bad ? "sets" : "does not set");
            check(ds::router_nonfinite_take(q) == expect_bad, what);
            std::snprintf(what, sizeof what, "%s: %s the take clears the flag", names[mode],
                          which ? "route_publish" : "route");
            check(!ds::router_nonfinite_take(q), what);
        }
    }

    std::printf("%s\n", fails ? "route_guard_test: FAILED" : "route_guard_test: all ok");
    return fails ? 1 : 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "route_guard_test: %s\n", error.what());
    return 1;
}
