// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Writes the routed experts of every layer and token of one conversation turn, read from llama.cpp's
// `ffn_moe_topk-<layer>` tensors: the prompt's tokens, then each generated token (sampled, so the trace follows
// what the model would answer).
//
//   dump_routing MODEL PROMPT_FILE OUT [max_new_tokens=2048] [threads=8] [seed=1]
//
// OUT: "XRT1", uint32 n_layer, uint32 k, then per token: uint8 phase (0 prompt, 1 generated), int32 token,
// n_layer * k uint16 expert ids.  OUT.txt gets the generated text.
#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Capture {
    int n_layer = 0;
    int k = 0;
    int64_t n_tokens = 0;
    std::vector<uint16_t> ids;   // [token][layer][k] of the last decode
    bool seen_any = false;
};

bool on_tensor(struct ggml_tensor* t, bool ask, void* user) {
    auto* c = static_cast<Capture*>(user);
    if (std::strncmp(t->name, "ffn_moe_topk-", 13) != 0) return !ask;
    if (ask) return true;
    const int layer = std::atoi(t->name + 13);
    const int k = (int) t->ne[0];
    const int64_t n = t->ne[1];
    if (t->type != GGML_TYPE_I32 || layer < 0 || layer >= c->n_layer) return true;
    if (c->k == 0) c->k = k;
    if (k != c->k) { std::fprintf(stderr, "dump_routing: k changed (%d, %d)\n", c->k, k); std::exit(1); }
    if (n != c->n_tokens) {
        c->n_tokens = n;
        c->ids.assign((size_t) (n * c->n_layer * k), 0xffff);
    }
    std::vector<int32_t> buf((size_t) (k * n));
    ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(int32_t));
    for (int64_t i = 0; i < n; ++i)
        for (int j = 0; j < k; ++j)
            c->ids[(size_t) ((i * c->n_layer + layer) * k + j)] = (uint16_t) buf[(size_t) (i * k + j)];
    c->seen_any = true;
    return true;
}

void write_tokens(std::FILE* f, const Capture& c, const llama_token* tokens, int64_t n, uint8_t phase) {
    if (n != c.n_tokens) { std::fprintf(stderr, "dump_routing: %lld tokens, captured %lld\n", (long long) n, (long long) c.n_tokens); std::exit(1); }
    const size_t row = (size_t) c.n_layer * (size_t) c.k;
    for (int64_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < row; ++j)
            if (c.ids[(size_t) i * row + j] == 0xffff) { std::fprintf(stderr, "dump_routing: a layer was not captured\n"); std::exit(1); }
        std::fwrite(&phase, 1, 1, f);
        std::fwrite(&tokens[i], 4, 1, f);
        std::fwrite(c.ids.data() + (size_t) i * row, 2, row, f);
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s MODEL PROMPT_FILE OUT [max_new_tokens] [threads] [seed]\n", argv[0]);
        return 2;
    }
    const int max_new = argc > 4 ? std::atoi(argv[4]) : 2048;
    const int threads = argc > 5 ? std::atoi(argv[5]) : 8;
    const uint32_t seed = argc > 6 ? (uint32_t) std::atoi(argv[6]) : 1;

    std::ifstream in(argv[2], std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string prompt = ss.str();

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    llama_model* model = llama_model_load_from_file(argv[1], mp);
    if (model == nullptr) return 1;
    const llama_vocab* vocab = llama_model_get_vocab(model);

    Capture cap;
    cap.n_layer = llama_model_n_layer(model);

    std::vector<llama_token> toks(prompt.size() + 16);
    const int np = llama_tokenize(vocab, prompt.data(), (int32_t) prompt.size(), toks.data(), (int32_t) toks.size(),
                                  false, true);
    if (np <= 0) { std::fprintf(stderr, "dump_routing: tokenize failed\n"); return 1; }
    toks.resize((size_t) np);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) (np + max_new + 16);
    cp.n_batch = 512;
    cp.n_ubatch = 512;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    cp.cb_eval = on_tensor;
    cp.cb_eval_user_data = &cap;
    llama_context* ctx = llama_init_from_model(model, cp);
    if (ctx == nullptr) return 1;

    std::FILE* f = std::fopen(argv[3], "wb");
    if (f == nullptr) return 1;
    std::fwrite("XRT1", 1, 4, f);

    // the prompt in batches of n_ubatch, so each decode is one graph and the capture holds exactly its tokens.
    // Every token is an output: llama.cpp cuts the last layer to the output rows before its FFN.
    llama_batch batch = llama_batch_init((int32_t) cp.n_ubatch, 0, 1);
    for (int i = 0; i < np; i += (int) cp.n_ubatch) {
        const int n = std::min((int) cp.n_ubatch, np - i);
        batch.n_tokens = n;
        for (int j = 0; j < n; ++j) {
            batch.token[j] = toks[(size_t) (i + j)];
            batch.pos[j] = i + j;
            batch.n_seq_id[j] = 1;
            batch.seq_id[j][0] = 0;
            batch.logits[j] = 1;
        }
        cap.n_tokens = -1;
        if (llama_decode(ctx, batch) != 0) return 1;
        if (cap.k != 0 && ftell(f) == 4) {
            const uint32_t hdr[2] = {(uint32_t) cap.n_layer, (uint32_t) cap.k};
            std::fwrite(hdr, 4, 2, f);
        }
        write_tokens(f, cap, toks.data() + i, n, 0);
    }
    llama_batch_free(batch);

    llama_sampler* smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.95f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.6f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));

    std::string text;
    int generated = 0;
    llama_token tok = llama_sampler_sample(smpl, ctx, -1);
    while (generated < max_new && !llama_vocab_is_eog(vocab, tok)) {
        char piece[256];
        const int m = llama_token_to_piece(vocab, tok, piece, sizeof piece, 0, true);
        if (m > 0) text.append(piece, (size_t) m);
        cap.n_tokens = -1;
        if (llama_decode(ctx, llama_batch_get_one(&tok, 1)) != 0) return 1;
        write_tokens(f, cap, &tok, 1, 1);
        ++generated;
        tok = llama_sampler_sample(smpl, ctx, -1);
    }
    std::fclose(f);

    std::ofstream(std::string(argv[3]) + ".txt", std::ios::binary) << text;
    std::fprintf(stderr, "dump_routing: %d prompt tokens, %d generated\n", np, generated);

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
