// SPDX-FileCopyrightText: 2026 Peasant Smith and the Project Maya contributors
// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// Host-only checks for resource budgets, explicit overrides and unsupported capability fallbacks.
#include "ds4_experts_rule.hpp"

#include <cstdio>

using namespace strata::ds4;
static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

int main() {
    const uint64_t mib = 1048576;
    const uint64_t GiB = 1ull << 30;
    // 16 GiB or less of graphics memory in all: faster mode by default
    CHECK(experts_per_word(-1, 12227ull << 20, 6) == 5);
    CHECK(experts_per_word(-1, 12 * GiB, 6) == 5);
    CHECK(experts_per_word(-1, 16 * GiB, 6) == 5);
    CHECK(experts_per_word(-1, 8 * GiB, 6) == 5);
    // 24 GB and bigger, and several cards: the model's own 6
    CHECK(experts_per_word(-1, 16 * GiB + 1, 6) == 6);
    CHECK(experts_per_word(-1, 24 * GiB, 6) == 6);
    CHECK(experts_per_word(-1, 4 * 32 * GiB, 6) == 6);
    CHECK(experts_per_word(-1, 2 * 12 * GiB, 6) == 6);   // two 12 GB cards hold 24 GiB in all
    // a GPU that shares the PC's RAM: faster mode whatever the size; discrete cards by size only
    CHECK(experts_per_word(-1, 91 * GiB, 6, true) == 5);
    CHECK(experts_per_word(-1, 24 * GiB, 6, false) == 6);   // a discrete 24 GB card
    CHECK(experts_per_word(-1, 12 * GiB, 6, false) == 5);   // a discrete 12 GB card
    CHECK(experts_per_word(6, 91 * GiB, 6, true) == 6);     // --experts 6 wins on the unified-memory GPU too
    // multi_rdna4 (the 5th argument): two or more matrix cards that hold at least half of the experts
    CHECK(experts_per_word(-1, 4 * 32 * GiB, 6, false, true) == 5);
    CHECK(experts_per_word(-1, 3 * 32 * GiB, 6, false, true) == 5);
    CHECK(experts_per_word(-1, 4 * 32 * GiB, 6, false, false) == 6);   // under half (or not matrix cards): the model's 6
    CHECK(experts_per_word(6, 4 * 32 * GiB, 6, false, true) == 6);     // --experts 6 wins
    // single_rdna4 (the 7th argument): one matrix card, whatever its size
    CHECK(experts_per_word(-1, 32 * GiB, 6, false, false, false, true) == 5);
    CHECK(experts_per_word(-1, 24 * GiB, 6, false, false, false, true) == 5);
    CHECK(experts_per_word(-1, 32 * GiB, 6, false, false, false, false) == 6);   // the same size when it is not one matrix card
    CHECK(experts_per_word(6, 32 * GiB, 6, false, false, false, true) == 6);     // --experts 6 still wins
    CHECK(experts_per_word(4, 32 * GiB, 6, false, false, false, true) == 4);     // and so does any other --experts N
    CHECK(experts_per_word(-1, 32 * GiB, 6, false, false, true, true) == 6);     // DS4_DEFAULTS=026: back to the model's 6
    CHECK(experts_per_word(4, 4 * 32 * GiB, 6, false, true) == 4);     // any other --experts N wins too
    CHECK(experts_per_word(5, 4 * 32 * GiB, 6, false, false) == 5);
    CHECK(experts_per_word(-1, 4 * 32 * GiB, 1, false, true) == 1);    // a model of one expert a word has nothing to drop
    CHECK(experts_test_override(experts_per_word(-1, 4 * 32 * GiB, 6, false, true), 6) == 5);
    CHECK(experts_per_word(-1, 91 * GiB, 4, true) == 4);    // never raised
    // unknown memory (0): the model's count, never a silent cut
    CHECK(experts_per_word(-1, 0, 6) == 6);
    // the flag decides, both ways
    CHECK(experts_per_word(6, 12 * GiB, 6) == 6);
    CHECK(experts_per_word(4, 12 * GiB, 6) == 4);
    CHECK(experts_per_word(5, 96 * GiB, 6) == 5);
    CHECK(experts_per_word(9, 12 * GiB, 6) == 6);   // never more than the model has
    CHECK(experts_per_word(1, 12 * GiB, 6) == 1);
    // a model that routes to 1 (or no) expert: left alone
    CHECK(experts_per_word(-1, 12 * GiB, 1) == 1);
    CHECK(experts_per_word(-1, 12 * GiB, 0) == 0);
    // a model that routes to fewer than 5: the rule never raises it
    CHECK(experts_per_word(-1, 12 * GiB, 4) == 4);
    CHECK(prefill_chunk_default(false, 0, false, true, false, 64 * mib, 1024) == 1);
    CHECK(prefill_chunk_default(false, 0, false, true, false, 0, 1024, false, false, mib) == 1);
    CHECK(prefill_chunk_default(false, 0, false, true, false, 64 * mib, 1024, false, false, mib) == 64);
    CHECK(prefill_chunk_default(true, 1, true, false, true, 64 * mib, 17, false, false, mib) == 17);
    CHECK(prefill_chunk_default(false, 0, false, true, false, 64, 1024, false, false, mib) == 1);
    // Aggregate budget can exceed the allocation limit when arrays fit singly.
    CHECK(prefill_chunk_capacity(32ULL * 1024 * mib, 1024 * mib, 1024 * mib,
                                 784000, 131072, 262144, 4096, 684) == 4096);
    CHECK(prefill_chunk_capacity(32ULL * 1024 * mib, 1024 * mib, 128 * mib,
                                 784000, 131072, 262144, 4096, 684) == 1022);
    CHECK(prefill_chunk_capacity(0, 0, mib, 1, 1, 0, 4096, 17) == 17);
    CHECK(prefill_chunk_capacity(mib, mib, mib, 1, 1, 0, 4096, 17) == 17);
    CHECK(prefill_chunk_capacity(mib, 0, 0, 1, 1, 0, 4096, 17) == 17);
    CHECK(prefill_chunk_capacity(mib, 0, mib, 0, 1, 0, 4096, 17) == 17);
    CHECK(prefill_chunk_capacity(mib, 0, mib, 1, 1, mib, 4096, 17) == 17);
    CHECK(prefill_chunk_capacity(64, 0, 64, 8, 8, 0, 4096, 17) == 1);
    CHECK(copy_all_fits(100, 120, 20, 100));
    CHECK(!copy_all_fits(100, 119, 20, 100));
    CHECK(!copy_all_fits(100, 120, 20, 99));
    CHECK(!copy_all_fits(100, ~uint64_t{0}, 0, 100));
    CHECK(!copy_all_fits(0, 120, 0, 100));
    CHECK(store_gpu_shares_ram(true, ~uint64_t{0}, ~uint64_t{0}));
    CHECK(!store_gpu_shares_ram(false, 10000000000ull, 0));
    CHECK(store_gpu_shares_ram(true, 10000000000ull, 5000000000ull, 4));
    CHECK(!store_gpu_shares_ram(true, 10000000000ull, 7000000000ull, 4));
    CHECK(attn_slab_rows(-1, 100, 4, 24, 8) == 7);
    CHECK(attn_slab_rows(-1, 100, 4, 99999, 8) == 100);
    CHECK(attn_slab_rows(0, 100, 4, 99999, 8) == 4);
    CHECK(attn_slab_rows(25, 100, 4, 0, 8, 0, false, 8) == 24);
    CHECK(attn_slab_rows(-1, 3, 8, 0, 8, 0, false, 8) == 3);
    CHECK(attn_slab_rows(-1, 100, 4, 24, 0) == 4);
    CHECK(attn_slab_rows(-1, 100, 4, 24, 8, 1, true) == 4);
    CHECK(staging_ring_mb(1, 64 * mib, 1, false, true) == 0);
    CHECK(staging_ring_mb(0, 64 * mib, 2, false, true) == 32);
    CHECK(staging_ring_mb(0, 64 * mib, 2, true, false) == 32);
    CHECK(staging_ring_mb(0, 0, 1, false, false) == 0);
    CHECK(staging_ring_mb(0, 64 * mib, 0, false, false) == 0);
    CHECK(attn_slab_base_mb(false, 0) == 1);
    CHECK(attn_slab_base_mb(true, 3, false, 8 * mib) == 3);
    CHECK(attn_slab_base_mb(false, 0, false, 8 * mib) == 8);
    CHECK(!copy_pin_default(1, true, false, 1, true));
    CHECK(!copy_pin_default(1, true, true, 1, false));
    CHECK(copy_pin_default(-1, false, true, 0, true));
    CHECK(!copy_pin_default(0, true, true, 1, true));
    CHECK(copy_hugetlb_default(-1, 100, 100));
    CHECK(!copy_hugetlb_default(-1, 99, 100));
    CHECK(!copy_hugetlb_default(-1, 0, 0));
    CHECK(!fast_attn_default(1, false));
    CHECK(fast_attn_default(-1, true));
    CHECK(!fast_attn_default(0, true));
    CHECK(!fast_attn_default(-1, true, true));
    CHECK(dspark_cum_value(false, 0.5f, true) == 0);
    CHECK(dspark_cum_value(true, 0.5f, false) == 0.5f);
    CHECK(store_feature_default(-1));
    CHECK(!store_feature_default(-1, true));
    CHECK(!store_feature_default(0));
    CHECK(store_feature_default(1, true));
    CHECK(settle_store_default() == 0 && settle_store_default(true) == 1);
    CHECK(settle_skips_copy_experts() && !settle_skips_copy_experts(true));
    CHECK(dq_old_default(-1, false));
    CHECK(!dq_old_default(-1, true));
    CHECK(dq_old_default(1, true));
    CHECK(head_per_default() == 1 && head_per_default(false, 7) == 7);
    CHECK(markov_head_q8_default() && !markov_head_q8_default(true));
    CHECK(gate_up_x16_default() && !gate_up_x16_default(true));
    CHECK(dense_pfgemm_default(-1) && !dense_pfgemm_default(0));
    CHECK(is_026_token("026") && is_026_token("0.2.6") && is_026_token("v0.2.6"));
    CHECK(!is_026_token(nullptr) && !is_026_token("0.2.7"));
    // the test override wins over the rule and the flag
#if defined(_WIN32)
    _putenv("DS4_TOPK=4");
#else
    setenv("DS4_TOPK", "4", 1);
#endif
    CHECK(experts_test_override(experts_per_word(-1, 96 * GiB, 6), 6) == 4);
    CHECK(experts_test_override(experts_per_word(6, 12 * GiB, 6), 6) == 4);
    std::printf("resource rules: %s\n", fails ? "FAIL" : "OK");
    return fails ? 1 : 0;
}
