// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "ds4_model.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

int main(int argc, char** argv) try {
    if (argc != 1 && (argc != 3 || std::strcmp(argv[1], "--dump") != 0)) return 1;
    const char* pack = std::getenv("STRATA_DSV41_TEST_PACK");
    if (!pack || !*pack) {
        std::puts("SKIP: set STRATA_DSV41_TEST_PACK to a synthetic model pack");
        return 77;
    }
    strata::ds4::Options options;
    options.pack = pack;
    options.max_context = 128;
    options.prefill_chunk = 32;
    if (const char* share = std::getenv("STRATA_DSV41_TEST_VRAM_EXPERTS")) {
        options.vram_experts_auto = false;
        options.vram_experts = std::strtod(share, nullptr);
    }
    options.ssd_cache_gb = 0;
    options.dspark_auto = false;
    options.profile_readonly = true;
    if (const char* profile = std::getenv("STRATA_DSV41_TEST_PROFILE")) options.profile = profile;
    if (const char* threads = std::getenv("STRATA_DSV41_TEST_THREADS"))
        options.cpu_threads = std::max(1, int(std::strtol(threads, nullptr, 10)));
    strata::ds4::Model model;
    std::string error;
    if (!model.load(options, error)) {
        std::fprintf(stderr, "chunk repeat load: %s\n", error.c_str());
        return 1;
    }
    if (model.vocab() < 64) {
        std::fputs("chunk repeat: the synthetic pack needs at least 64 tokens\n", stderr);
        return 1;
    }
    std::array<int, 64> tokens{};
    for (size_t i = 0; i < tokens.size(); ++i)
        tokens[i] = int((i * 1001) % size_t(model.vocab()));
    std::vector<float> expected, logits;
    for (int repeat = 0; repeat < 6; ++repeat) {
        // Repeat each layout after reset, including one-slot reuse after the
        // two-slot layout has expanded both capacities.
        const auto sizes = repeat / 2 == 1 ? std::vector<int>{64} : std::vector<int>{17, 30, 17};
        if (repeat % 2 == 0) expected.clear();
        model.reset();
        size_t offset = 0;
        // Grow one slot from 17 to 30 rows, then reuse it after reset. A
        // 64-row feed also needs the second slot with this 32-row chunk.
        for (const int rows : sizes) {
            if (!model.feed(tokens.data() + offset, rows, rows, error) || !model.logits(logits)) {
                std::fprintf(stderr, "chunk repeat feed at %zu: %s\n", offset, error.c_str());
                return 1;
            }
            const size_t count = size_t(rows) * size_t(model.vocab());
            if (logits.size() != count ||
                !std::all_of(logits.begin(), logits.end(), [](float x) { return std::isfinite(x); })) {
                std::fputs("chunk repeat: invalid logits\n", stderr);
                return 1;
            }
            if (repeat % 2 == 0) expected.insert(expected.end(), logits.begin(), logits.end());
            else if (std::memcmp(expected.data() + offset * size_t(model.vocab()), logits.data(), count * sizeof(float)) != 0) {
                std::fprintf(stderr, "chunk repeat: logits differ at row %zu\n", offset);
                return 1;
            }
            offset += size_t(rows);
        }
        if (model.pos() != int64_t(tokens.size())) return 1;
    }
    if (argc == 3) {
        std::FILE* file = std::fopen(argv[2], "wb");
        if (!file) return 1;
        const bool written = std::fwrite(expected.data(), sizeof(float), expected.size(), file) == expected.size();
        const int closed = std::fclose(file);
        if (!written || closed) return 1;
    }
    std::puts("chunk repeat: all 64 logit rows are bit equal after reset");
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "chunk repeat: %s\n", error.what());
    return 1;
}
