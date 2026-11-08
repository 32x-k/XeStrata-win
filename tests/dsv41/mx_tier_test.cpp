// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Host-only MXFP4 parity and startup tier selection. No device or model is opened.
#include "ds4_gguf_cpu.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using namespace strata::ds4::gguf;
namespace {
int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

void parity() {
    uint32_t seed = 0x9e3779b9u;
    const auto next = [&] { seed = seed * 1664525u + 1013904223u; return seed; };
    constexpr int rows = 7;
    constexpr float guard = -123456.25f;
    for (int k : {256, 5120}) {
        const size_t rb = size_t(k / 32) * 17;
        std::vector<uint8_t> w(rb * rows);
        for (auto& b : w) b = uint8_t(next() >> 24);
        for (size_t b = 0; b < w.size() / 17; ++b) w[b * 17] = uint8_t(118 + b % 6);
        std::vector<float> x(k);
        std::vector<std::vector<uint8_t>> act(kMxCpuMaxN, std::vector<uint8_t>(mx_act_bytes(k)));
        const void* ap[kMxCpuMaxN];
        for (int i = 0; i < kMxCpuMaxN; ++i) {
            for (int j = 0; j < k; ++j) x[j] = i == 1 ? 0.f : float(int(next() >> 16) - 32768) / 32768.f;
            mx_quant(x.data(), k, act[i].data());
            ap[i] = act[i].data();
        }
        for (int n : {1, 2, 4, kMxCpuMaxN}) {
            std::vector<float> ref(size_t(n) * rows, guard), out(ref.size());
            float* rp[kMxCpuMaxN];
            float* yp[kMxCpuMaxN];
            for (int i = 0; i < n; ++i) { rp[i] = ref.data() + size_t(i) * rows; yp[i] = out.data() + size_t(i) * rows; }
            // Nonzero row offset, a tail row count, and guard values outside the written range.
            mx_rows_tier(kMxScalar, w.data(), rb, k, 1, rows - 1, ap, n, rp);
            for (int tier = kMxScalar; tier < kMxTiers; ++tier) {
                if (!mx_tier_ok(tier)) continue;
                std::fill(out.begin(), out.end(), guard);
                mx_rows_tier(tier, w.data(), rb, k, 1, rows - 1, ap, n, yp);
                CHECK(std::memcmp(ref.data(), out.data(), ref.size() * sizeof(float)) == 0);
                CHECK(std::all_of(out.begin(), out.end(), [](float v) { return std::isfinite(v); }));
                for (int i = 0; i < n; ++i) for (int j = rows - 2; j < rows; ++j) CHECK(yp[i][j] == guard);
            }
        }
    }
    for (int t = 0; t < kMxTiers; ++t)
        std::printf("PARITY tier=%s supported=%d\n", mx_tier_name(t), int(mx_tier_ok(t)));
}

void selection(const std::string& expected) {
    const auto start = std::chrono::steady_clock::now();
    const int chosen = mx_tier();
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    const std::string report = mx_tier_report();
    CHECK(mx_tier_ok(chosen));
    CHECK(mx_tier() == chosen);
    CHECK(report == mx_tier_report());
    int override_tier = -1;
    if (const char* manual = std::getenv("DS4_MX_TIER"))
        for (int t = 0; t < kMxTiers; ++t)
            if (manual == std::string(mx_tier_name(t)) && mx_tier_ok(t)) override_tier = t;
    if (override_tier >= 0) {
        CHECK(chosen == override_tier);
        CHECK(report == std::string("DS4_MX_TIER=") + mx_tier_name(chosen));
    } else {
        const char* force = std::getenv("STRATA_FORCE_AVX2");
        const bool limited = force && std::strtol(force, nullptr, 10) != 0;
        std::vector<int> candidates;
        for (int t = kMxAvx2; t <= (limited ? kMxAvxVnni : kMxAvx512Vnni); ++t)
            if (mx_tier_ok(t)) candidates.push_back(t);
        if (candidates.size() < 2) {
            CHECK(chosen == (candidates.empty() ? kMxScalar : candidates.front()));
        } else {
            std::string fields = report;
            std::replace(fields.begin(), fields.end(), ',', ' ');
            std::istringstream input(fields);
            double minimum = std::numeric_limits<double>::infinity(), selected_us = minimum;
            for (int t : candidates) {
                std::string name, unit;
                double us = -1;
                CHECK(bool(input >> name >> us >> unit));
                CHECK(name == mx_tier_name(t) && unit == "us" && std::isfinite(us) && us >= 0);
                minimum = std::min(minimum, us);
                if (chosen == t) selected_us = us;
            }
            CHECK(selected_us == minimum);   // The report rounds to whole microseconds, preserving minimum order.
            std::string extra;
            CHECK(!(input >> extra));
        }
    }
    if (!expected.empty()) {
        for (int t = 0; t < kMxTiers; ++t)
            if (expected == mx_tier_name(t) && mx_tier_ok(t)) CHECK(chosen == t);
    }
    std::printf("SELECTION tier=%s report=[%s] elapsed_ms=%.6f\n", mx_tier_name(chosen), report.c_str(), ms);
}
}  // namespace

int main(int argc, char** argv) try {
    bool selection_only = false;
    std::string expected;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--selection-only") == 0) selection_only = true;
        else if (std::strcmp(argv[i], "--expect") == 0 && i + 1 < argc) expected = argv[++i];
        else return 2;
    }
    if (!selection_only) parity();
    selection(expected);
    std::printf("mx_tier_test: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL exception: %s\n", e.what());
    return 1;
}
