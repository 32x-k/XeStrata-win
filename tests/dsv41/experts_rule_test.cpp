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
    for (uint64_t memory : {0ull, 1ull << 30, 8ull << 30, 32ull << 30})
        for (bool shared : {false, true})
            for (bool matrix : {false, true}) {
                CHECK(experts_per_word(-1, memory, 6, shared, matrix) == 6);
                CHECK(experts_per_word(4, memory, 6, shared, matrix) == 4);
                CHECK(experts_per_word(9, memory, 6, shared, matrix) == 6);
            }
    CHECK(experts_per_word(1, 0, 1) == 1);
    CHECK(experts_per_word(-1, 0, 0) == 0);
    CHECK(prefill_chunk_default(false, 0, false, true, false, 64 * mib, 1024) == 1);
    CHECK(prefill_chunk_default(false, 0, false, true, false, 0, 1024, false, false, mib) == 1);
    CHECK(prefill_chunk_default(false, 0, false, true, false, 64 * mib, 1024, false, false, mib) == 64);
    CHECK(prefill_chunk_default(true, 1, true, false, true, 64 * mib, 17, false, false, mib) == 17);
    CHECK(prefill_chunk_default(false, 0, false, true, false, 64, 1024, false, false, mib) == 1);
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
    std::printf("resource rules: %s\n", fails ? "FAIL" : "OK");
    return fails ? 1 : 0;
}
