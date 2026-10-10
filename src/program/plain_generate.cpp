// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/program/plain_generate.cpp - `strata generate` for the models of plain_model.hpp (Ornith-1.5, llm-jp-4.1).
//
//     strata --native MODEL.gguf --tokens 1,2,3 [--n 32] [--max-context N] [--max-t T] [--threads N]
//            [--dump-logits FILE] [--adapt-every N] [--mmap-experts] [--no-pcie-split]
//            [--expert-profile FILE] [--expert-profile-save FILE] [--rope-scaling yarn --rope-scale F --yarn-orig-ctx N]
//
// The prompt runs in windows of up to T tokens, then one token a window, greedily.  stdout gets one line,
// `tokens: a,b,c`, the generated ids; --dump-logits writes the logits each generated token was chosen from, n_vocab
// floats a row.  The report goes to stderr.
#include "strata/program/plain_generate.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/plain_model.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

namespace strata::program {
namespace {

using Clock = std::chrono::steady_clock;

bool parse_ids(const std::string& text, std::vector<int32_t>& out) {
    out.clear();
    const char* p = text.c_str();
    while (*p != '\0') {
        while (*p == ',' || *p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') ++p;
        if (*p == '\0') break;
        char* end = nullptr;
        const long v = std::strtol(p, &end, 10);
        if (end == p || v < 0 || v > 0x7fffffff) return false;
        out.push_back((int32_t) v);
        p = end;
    }
    return !out.empty();
}

// The model's architecture, from its first shard's metadata only; empty when it cannot be read.
std::string architecture(const std::string& path) {
    try {
        const GgufModel m = GgufModel::open(path);
        const MetaValue* a = m.meta().get("general.architecture");
        return a != nullptr && a->type == MetaType::STRING ? a->s : std::string();
    } catch (const std::exception&) {
        return std::string();
    }
}

}  // namespace

int plain_main(int argc, char** argv) {
    std::string gguf;
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], "--native") == 0) gguf = argv[i + 1];
    if (gguf.empty()) return -1;
    const std::string arch = architecture(gguf);
    if (arch.empty() || arch == "qwen4exp") return -1;

    strata::core::PlainOptions opt;
    opt.gguf = gguf;
    std::vector<int32_t> prompt;
    int64_t n_new = 32;
    std::string dump_path, profile_save;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", what); std::exit(2); }
            return argv[++i];
        };
        if (a == "--native") (void) next("--native");
        else if (a == "--tokens") {
            if (!parse_ids(next("--tokens"), prompt)) { std::fprintf(stderr, "--tokens: a list of token ids\n"); return 2; }
        } else if (a == "--tokens-file") {
            std::ifstream in(next("--tokens-file"));
            const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            if (!in.good() && !in.eof()) { std::fprintf(stderr, "cannot read the token file\n"); return 2; }
            if (!parse_ids(text, prompt)) { std::fprintf(stderr, "--tokens-file: a list of token ids\n"); return 2; }
        }
        else if (a == "--n") n_new = std::strtoll(next("--n"), nullptr, 10);
        else if (a == "--max-context") opt.max_cells = std::strtoll(next("--max-context"), nullptr, 10);
        else if (a == "--max-t") opt.max_t = (int) std::strtol(next("--max-t"), nullptr, 10);
        else if (a == "--threads") opt.threads = (int) std::strtol(next("--threads"), nullptr, 10);
        else if (a == "--dump-logits") dump_path = next("--dump-logits");
        else if (a == "--adapt-every") opt.adapt_every = (int) std::strtol(next("--adapt-every"), nullptr, 10);
        else if (a == "--gpu") (void) next("--gpu");
        else if (a == "--mmap-experts") opt.mmap_experts = true;
        else if (a == "--no-pcie-split") opt.pcie_split = false;
        else if (a == "--expert-profile") opt.profile = next("--expert-profile");
        else if (a == "--rope-scaling") opt.rope_scaling = next("--rope-scaling");
        else if (a == "--rope-scale") opt.rope_scale = std::strtod(next("--rope-scale"), nullptr);
        else if (a == "--yarn-orig-ctx") opt.yarn_orig_ctx = std::strtod(next("--yarn-orig-ctx"), nullptr);
        else if (a == "--expert-profile-save") profile_save = next("--expert-profile-save");
        else {
            std::fprintf(stderr, "strata (%s): unknown option %s\n", arch.c_str(), a.c_str());
            return 2;
        }
    }
    if (prompt.empty() || n_new < 1) { std::fprintf(stderr, "strata (%s): needs --tokens and --n >= 1\n", arch.c_str()); return 2; }
    if ((int64_t) prompt.size() + n_new > opt.max_cells) opt.max_cells = (int64_t) prompt.size() + n_new;

    strata::core::PlainModel model;
    std::string err;
    const Clock::time_point t0 = Clock::now();
    if (!model.load(opt, err)) { std::fprintf(stderr, "strata (%s): %s\n", arch.c_str(), err.c_str()); return 1; }
    const double load_s = std::chrono::duration<double>(Clock::now() - t0).count();
    std::unique_ptr<std::FILE, int (*)(std::FILE*)> dump_file(nullptr, &std::fclose);
    if (!dump_path.empty()) {
        dump_file.reset(std::fopen(dump_path.c_str(), "wb"));
        if (!dump_file) {
            std::fprintf(stderr, "cannot write %s\n", dump_path.c_str());
            return 1;
        }
    }
    std::FILE* const dump = dump_file.get();
    const int T = opt.max_t;
    const int64_t V = model.vocab();
    std::vector<int32_t> out((size_t) T);
    std::vector<float> logits((size_t) T * (size_t) V);
    std::vector<int32_t> produced;
    auto emit = [&](int row) -> bool {
        produced.push_back(out[(size_t) row]);
        if (dump != nullptr &&
            std::fwrite(logits.data() + (size_t) row * (size_t) V, sizeof(float), (size_t) V, dump) != (size_t) V) {
            std::fprintf(stderr, "cannot write the logits\n");
            return false;
        }
        return true;
    };
    float* lg = dump != nullptr ? logits.data() : nullptr;
    const Clock::time_point t1 = Clock::now();
    int64_t pos = 0;
    while (pos < (int64_t) prompt.size()) {
        const int n = (int) std::min<int64_t>(T, (int64_t) prompt.size() - pos);
        if (!model.forward(prompt.data() + pos, n, pos, out.data(), lg, err)) {
            std::fprintf(stderr, "strata (%s): %s\n", arch.c_str(), err.c_str());
            return 1;
        }
        pos += n;
        if (pos == (int64_t) prompt.size() && !emit(n - 1)) return 1;
    }
    const Clock::time_point t2 = Clock::now();
    while ((int64_t) produced.size() < n_new) {
        const int32_t x = produced.back();
        if (!model.forward(&x, 1, pos, out.data(), lg, err)) {
            std::fprintf(stderr, "strata (%s): %s\n", arch.c_str(), err.c_str());
            return 1;
        }
        ++pos;
        if (!emit(0)) return 1;
    }
    const Clock::time_point t3 = Clock::now();
    if (!profile_save.empty() && !model.save_profile(profile_save, err)) {
        std::fprintf(stderr, "strata (%s): %s\n", arch.c_str(), err.c_str());
        return 1;
    }
    std::printf("tokens: ");
    for (size_t i = 0; i < produced.size(); ++i) std::printf(i ? ",%d" : "%d", produced[i]);
    std::printf("\n");
    const double pp = std::chrono::duration<double>(t2 - t1).count(), tg = std::chrono::duration<double>(t3 - t2).count();
    const int64_t routed = model.hits() + model.misses();
    std::fprintf(stderr, "strata (%s): load %.1f s, prompt %zu tokens %.2f s (%.1f tok/s), %zu generated %.2f s (%.2f tok/s); "
                         "VRAM tier %lld of %lld slots filled, %.1f%% of routed experts hit, %lld swaps\n",
                 arch.c_str(), load_s, prompt.size(), pp, (double) prompt.size() / pp, produced.size() - 1, tg,
                 (double) (produced.size() - 1) / (tg > 0 ? tg : 1), (long long) model.resident(),
                 (long long) model.slots(), routed > 0 ? 100.0 * (double) model.hits() / (double) routed : 0.0,
                 (long long) model.swaps());
    return 0;
}

}  // namespace strata::program
