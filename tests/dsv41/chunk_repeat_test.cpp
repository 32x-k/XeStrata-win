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
    if (argc != 1 && (argc != 3 || std::strcmp(argv[1], "--dump"))) return 1;
    const char* pack = std::getenv("STRATA_DSV41_TEST_PACK");
    if (!pack || !*pack) {
        std::puts("SKIP: set STRATA_DSV41_TEST_PACK to a synthetic model pack");
        return 77;
    }
    strata::ds4::Options options;
    options.pack = pack;
    options.max_context = 128;
    options.prefill_chunk = 16;
    if (const char* share = std::getenv("STRATA_DSV41_TEST_VRAM_EXPERTS")) {
        options.vram_experts_auto = false;
        options.vram_experts = std::strtod(share, nullptr);
    }
    options.ssd_cache_gb = 0;
    options.dspark_auto = false;
    if (const char* threads = std::getenv("STRATA_DSV41_TEST_THREADS"))
        options.cpu_threads = std::max(1, std::atoi(threads));
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
    for (int repeat = 0; repeat < 2; ++repeat) {
        model.reset();
        for (size_t offset = 0; offset < tokens.size(); offset += 16) {
            if (!model.feed(tokens.data() + offset, 16, 16, error) || !model.logits(logits)) {
                std::fprintf(stderr, "chunk repeat feed at %zu: %s\n", offset, error.c_str());
                return 1;
            }
            const size_t count = size_t(16) * size_t(model.vocab());
            if (logits.size() != count ||
                !std::all_of(logits.begin(), logits.end(), [](float x) { return std::isfinite(x); })) {
                std::fputs("chunk repeat: invalid logits\n", stderr);
                return 1;
            }
            if (repeat == 0) expected.insert(expected.end(), logits.begin(), logits.end());
            else if (std::memcmp(expected.data() + offset * size_t(model.vocab()), logits.data(), count * sizeof(float))) {
                std::fprintf(stderr, "chunk repeat: logits differ at chunk %zu\n", offset / 16);
                return 1;
            }
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
