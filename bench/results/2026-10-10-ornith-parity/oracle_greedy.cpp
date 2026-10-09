// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// llama.cpp's greedy decode of one prompt, as the reference XeStrata's plain engine is compared against.
//
//   oracle_greedy MODEL PROMPT_FILE N OUT [threads=8] [gpu_layers=0] [cpu_moe=0]
//
// The prompt is tokenized with the model's special tokens; OUT.prompt gets its ids, OUT.tokens the N generated ids
// (each the argmax of its logits, the lowest id on a tie), comma-separated, and OUT.logits the N rows of logits each
// was chosen from, n_vocab floats a row.  cpu_moe=1 keeps the routed experts in host memory (llama.cpp's --cpu-moe):
// with every layer on a GPU, the placement XeStrata's own decode compares with.
#include "llama.h"
#include "ggml-cpu.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

void write_ids(const std::string& path, const std::vector<llama_token>& ids) {
    std::FILE* f = std::fopen(path.c_str(), "w");
    if (f == nullptr) { std::fprintf(stderr, "oracle_greedy: cannot write %s\n", path.c_str()); std::exit(1); }
    for (size_t i = 0; i < ids.size(); ++i) std::fprintf(f, i ? ",%d" : "%d", ids[i]);
    std::fprintf(f, "\n");
    std::fclose(f);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: %s MODEL PROMPT_FILE N OUT [threads] [gpu_layers] [cpu_moe]\n", argv[0]);
        return 2;
    }
    const int n_new = (int) std::strtol(argv[3], nullptr, 10);
    const std::string out = argv[4];
    const int threads = argc > 5 ? (int) std::strtol(argv[5], nullptr, 10) : 8;
    const int gpu_layers = argc > 6 ? (int) std::strtol(argv[6], nullptr, 10) : 0;
    const bool cpu_moe = argc > 7 && (int) std::strtol(argv[7], nullptr, 10) != 0;
    std::ifstream in(argv[2], std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string prompt = ss.str();

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = gpu_layers;   // 0: the CPU backend alone (a build with a GPU backend for the noise floor)
    const llama_model_tensor_buft_override moe_cpu[2] = {{"\\.ffn_(up|down|gate)_exps", ggml_backend_cpu_buffer_type()},
                                                         {nullptr, nullptr}};
    if (cpu_moe) mp.tensor_buft_overrides = moe_cpu;
    llama_model* model = llama_model_load_from_file(argv[1], mp);
    if (model == nullptr) return 1;
    const llama_vocab* vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> toks(prompt.size() + 16);
    const int np = llama_tokenize(vocab, prompt.data(), (int32_t) prompt.size(), toks.data(), (int32_t) toks.size(),
                                  true, true);
    if (np <= 0) { std::fprintf(stderr, "oracle_greedy: tokenize failed\n"); return 1; }
    toks.resize((size_t) np);
    write_ids(out + ".prompt", toks);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) (np + n_new + 16);
    cp.n_batch = (uint32_t) np;
    cp.n_ubatch = 512;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    llama_context* ctx = llama_init_from_model(model, cp);
    if (ctx == nullptr) return 1;

    std::FILE* lf = std::fopen((out + ".logits").c_str(), "wb");
    if (lf == nullptr) return 1;
    std::vector<llama_token> gen;
    llama_batch batch = llama_batch_init(np, 0, 1);
    batch.n_tokens = np;
    for (int i = 0; i < np; ++i) {
        batch.token[i] = toks[(size_t) i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = i == np - 1;
    }
    int pos = np;
    for (int step = 0; step < n_new; ++step) {
        if (llama_decode(ctx, batch) != 0) { std::fprintf(stderr, "oracle_greedy: decode failed\n"); return 1; }
        const float* lg = llama_get_logits_ith(ctx, -1);
        int best = 0;
        for (int v = 1; v < n_vocab; ++v)
            if (lg[v] > lg[best]) best = v;
        std::fwrite(lg, sizeof(float), (size_t) n_vocab, lf);
        gen.push_back(best);
        batch.n_tokens = 1;
        batch.token[0] = best;
        batch.pos[0] = pos++;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = 1;
    }
    std::fclose(lf);
    write_ids(out + ".tokens", gen);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
