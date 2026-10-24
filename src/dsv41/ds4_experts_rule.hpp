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

// DS4_TOPK=N (a test, not for users): over the flag and the rule; returns the count to use
inline int experts_test_override(int chosen, int model_k) {
    if (const char* v = std::getenv("DS4_TOPK")) {
        const int n = std::atoi(v);
        if (n >= 1 && n <= model_k) return n;
    }
    return chosen;
}

}  // namespace strata::ds4
