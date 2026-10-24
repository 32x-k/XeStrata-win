// Strata. The experts-per-word rule of the DeepSeek V4.1 engine (src/dsv41/ds4_experts_rule.hpp): 5 on a card with 16 GiB or less, 6
// elsewhere, --experts overrides, DS4_TOPK over both.  No GPU needed.
#include "ds4_experts_rule.hpp"

#include <cstdio>
#include <cstdlib>

using strata::ds4::experts_per_word;
using strata::ds4::experts_test_override;

static int fails = 0;
#define CHECK_EQ(a, b)                                                                              \
    do {                                                                                            \
        const long long x = (a), y = (b);                                                           \
        if (x != y) {                                                                               \
            std::printf("FAIL %s:%d: %s = %lld, expected %lld\n", __FILE__, __LINE__, #a, x, y);    \
            ++fails;                                                                                \
        }                                                                                           \
    } while (0)

int main() {
    const uint64_t GiB = 1ull << 30;
    // the owner's 12 GB-class cards (RTX 5070 12227 MiB, RTX 3060 12 GB), 16 GiB cards: faster mode by default
    CHECK_EQ(experts_per_word(-1, 12227ull << 20, 6), 5);
    CHECK_EQ(experts_per_word(-1, 12 * GiB, 6), 5);
    CHECK_EQ(experts_per_word(-1, 16 * GiB, 6), 5);
    CHECK_EQ(experts_per_word(-1, 8 * GiB, 6), 5);
    // 24 GB and bigger, and several cards: the model's own 6
    CHECK_EQ(experts_per_word(-1, 16 * GiB + 1, 6), 6);
    CHECK_EQ(experts_per_word(-1, 24 * GiB, 6), 6);
    CHECK_EQ(experts_per_word(-1, 4 * 32 * GiB, 6), 6);
    CHECK_EQ(experts_per_word(-1, 2 * 12 * GiB, 6), 6);   // two 12 GB cards hold 24 GiB in all
    // unknown memory (0): the model's count, never a silent cut
    CHECK_EQ(experts_per_word(-1, 0, 6), 6);
    // the flag decides, both ways
    CHECK_EQ(experts_per_word(6, 12 * GiB, 6), 6);
    CHECK_EQ(experts_per_word(4, 12 * GiB, 6), 4);
    CHECK_EQ(experts_per_word(5, 96 * GiB, 6), 5);
    CHECK_EQ(experts_per_word(9, 12 * GiB, 6), 6);   // never more than the model has
    CHECK_EQ(experts_per_word(1, 12 * GiB, 6), 1);
    // a model that routes to 1 (or no) expert: left alone
    CHECK_EQ(experts_per_word(-1, 12 * GiB, 1), 1);
    CHECK_EQ(experts_per_word(-1, 12 * GiB, 0), 0);
    // a model that routes to fewer than 5: the rule never raises it
    CHECK_EQ(experts_per_word(-1, 12 * GiB, 4), 4);
    // the test override wins over the rule and the flag
#if defined(_WIN32)
    _putenv("DS4_TOPK=4");
#else
    setenv("DS4_TOPK", "4", 1);
#endif
    CHECK_EQ(experts_test_override(experts_per_word(-1, 96 * GiB, 6), 6), 4);
    CHECK_EQ(experts_test_override(experts_per_word(6, 12 * GiB, 6), 6), 4);
#if defined(_WIN32)
    _putenv("DS4_TOPK=9");
#else
    setenv("DS4_TOPK", "9", 1);
#endif
    CHECK_EQ(experts_test_override(5, 6), 5);   // out of range: ignored
#if defined(_WIN32)
    _putenv("DS4_TOPK=");
#else
    unsetenv("DS4_TOPK");
#endif
    CHECK_EQ(experts_test_override(5, 6), 5);
    // prompt chunk: 4096 from 10 GiB of VRAM (12 GB cards, 16, 24, 32, 80 GB), 1024 below
    using strata::ds4::prefill_chunk_default;
    CHECK_EQ(prefill_chunk_default(12227ull << 20), 4096);
    CHECK_EQ(prefill_chunk_default(12 * GiB), 4096);
    CHECK_EQ(prefill_chunk_default(32 * GiB), 4096);
    CHECK_EQ(prefill_chunk_default(80 * GiB), 4096);
    CHECK_EQ(prefill_chunk_default(10 * GiB), 4096);
    CHECK_EQ(prefill_chunk_default(10 * GiB - 1), 1024);
    CHECK_EQ(prefill_chunk_default(8 * GiB), 1024);
    CHECK_EQ(prefill_chunk_default(0), 1024);
    CHECK_EQ(prefill_chunk_default(16 * GiB), 4096);
    CHECK_EQ(prefill_chunk_default(24 * GiB), 4096);
    CHECK_EQ(prefill_chunk_default(12 * GiB, true), 1024);   // HIP builds: 1024 until measured
    CHECK_EQ(prefill_chunk_default(32 * GiB, true), 1024);
    // the experts-copied-into-RAM case (fallback 2048): 4096 on 12/16/24/32 GB cards, 2048 below 10 GiB and on HIP builds
    CHECK_EQ(prefill_chunk_default(12227ull << 20, false, 2048), 4096);
    CHECK_EQ(prefill_chunk_default(16 * GiB, false, 2048), 4096);
    CHECK_EQ(prefill_chunk_default(24 * GiB, false, 2048), 4096);
    CHECK_EQ(prefill_chunk_default(32 * GiB, false, 2048), 4096);
    CHECK_EQ(prefill_chunk_default(8 * GiB, false, 2048), 2048);
    CHECK_EQ(prefill_chunk_default(0, false, 2048), 2048);
    CHECK_EQ(prefill_chunk_default(32 * GiB, true, 2048), 2048);
    // the rule takes no --max-context: a 128K session on a 12 GB card gets the same 4096, because the chunk's buffers and the KV are allocated
    // whole at load for that context and a chunk that does not fit there is halved before the first prompt (4096 -> 2048 -> 1024 -> ...)
    if (fails == 0) std::printf("experts rule: all checks passed\n");
    return fails == 0 ? 0 : 1;
}
