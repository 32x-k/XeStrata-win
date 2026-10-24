// Strata. Experts per word: the "faster mode" rule of the DeepSeek V4.1 engine.  Header only, no GPU code (tests/dsv41/experts_rule_test.cpp).
//
// The model routes every word to its top 6 experts.  On a PC with a small graphics card the CPU lane and the SSD set the speed, and
// each expert dropped saves their bytes: the router's top N are kept and their weights renormalised (the same selection the full
// model makes, a shorter list).  The default is 5 where the graphics memory is 16 GiB or less, the model's own count elsewhere;
// --experts N says it outright (--experts 6: full quality), DS4_TOPK=N is the test override over everything.
#pragma once

#include <cstdint>
#include <cstdlib>

namespace strata::ds4 {

constexpr uint64_t kFasterModeVramMax = 16ull << 30;   // GiB of graphics memory in all, at most
constexpr int kFasterModeExperts = 5;

// requested: --experts (-1: not given, the rule decides); vram_bytes: the total memory of the cards the engine uses;
// model_k: the experts a word takes in the model.  The result is in 1 .. model_k.
inline int experts_per_word(int requested, uint64_t vram_bytes, int model_k) {
    if (model_k <= 1) return model_k;
    int n = requested;
    if (n < 1) n = vram_bytes != 0 && vram_bytes <= kFasterModeVramMax ? kFasterModeExperts : model_k;
    if (n > model_k) n = model_k;
    return n < 1 ? 1 : n;
}

// Prompt chunk (--prefill-chunk) of the single-card default, decided once at load from bytes (no timing).  The chunk's buffers come out of the VRAM
// the expert tier would have: 1024 -> 4096 tokens takes 83 of 469 resident experts on a 12 GB card, which cost nothing measurable in decode
// (RTX 3060: x0.995 code, x1.002 agent) and made 8K prompts 20% faster (RTX 3060 175 -> 210, RTX 5070 186 -> 223, 32 GB class 285 -> 353 tok/s);
// 8192 returns up to 4% of agent decode.  Cards under 10 GiB keep 1024 (their tier is the first thing a bigger buffer would eat).
constexpr uint64_t kChunk4096MinVram = 10ull << 30;
// fallback: the value the case had before (1024: an explicit --tp 1 single card; 2048: a GGUF pack with --mmap-experts off, the experts copied into RAM; the experts
// keep 8192, which is not decided here).  hip: a HIP build keeps the fallback until a one-card Radeon sweep shows 4096 is no slower.
// The result does not depend on --max-context: the chunk's buffers and the KV are allocated whole at load for the full context, and a
// chunk the card cannot hold is halved there (4096 -> 2048 -> 1024 -> ...), so a session never runs out later.
inline int prefill_chunk_default(uint64_t min_vram_bytes, bool hip = false, int fallback = 1024) {
    return !hip && min_vram_bytes >= kChunk4096MinVram ? 4096 : fallback;
}

// DS4_TOPK=N (a test, not for users): over the flag and the rule; returns the count to use
inline int experts_test_override(int chosen, int model_k) {
    if (const char* v = std::getenv("DS4_TOPK")) {
        const int n = std::atoi(v);
        if (n >= 1 && n <= model_k) return n;
    }
    return chosen;
}

}  // namespace strata::ds4
