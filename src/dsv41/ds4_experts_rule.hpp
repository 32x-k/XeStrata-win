// SPDX-FileCopyrightText: 2026 Peasant Smith and the Project Maya contributors
// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// Strata. Experts per word: the "faster mode" rule of the DeepSeek V4.1 engine.
// DeepSeek's resource rules. The caller supplies capacities obtained from the device and the model's
// tensor shapes. A missing estimate takes the smallest working batch, with no product-name defaults.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace strata::ds4 {

inline bool is_026_token(const char* v) {
    return v != nullptr && (std::strcmp(v, "026") == 0 || std::strcmp(v, "0.2.6") == 0 || std::strcmp(v, "v0.2.6") == 0);
}
inline bool defaults_026() {
    static const bool on = is_026_token(std::getenv("DS4_DEFAULTS"));
    return on;
}

// A smaller expert count changes the model's answer, so it is selected only by the explicit request.
inline int experts_per_word(int requested, uint64_t vram_bytes, int model_k, bool integrated = false,
                            bool matrix_engines = false, bool v026 = false) {
    (void) vram_bytes, (void) integrated, (void) matrix_engines, (void) v026;
    if (model_k <= 1) return model_k;
    return requested > 0 ? std::min(requested, model_k) : model_k;
}

// budget_bytes is the available scratch budget after persistent tensors and memory reserves.
// bytes_per_token includes every temporary array live during a chunk. big is the caller's row capacity.
inline int prefill_chunk_default(bool own_default, double gpu_share, bool gguf_copy, bool file_experts,
                                 bool matrix_engines, uint64_t budget_bytes, int big = 1,
                                 bool multi_tp = false, bool v026 = false, size_t bytes_per_token = 0) {
    (void) own_default, (void) gpu_share, (void) gguf_copy, (void) file_experts;
    (void) matrix_engines, (void) multi_tp, (void) v026;
    if (bytes_per_token == 0 || budget_bytes == 0) return 1;
    return (int) std::max<uint64_t>(1, std::min<uint64_t>((uint64_t) std::max(1, big), budget_bytes / bytes_per_token));
}

// Total temporary storage and a single allocation are independent limits.
// The one-eighth policy leaves the rest for weights and experts; it is not a
// throughput threshold. Actual allocation failure retries the old default.
inline int prefill_chunk_capacity(uint64_t free_bytes, uint64_t reserve_bytes, uint64_t largest,
                                   size_t row_bytes, size_t allocation_row_bytes, size_t padding_bytes,
                                   int context_rows, int fallback) {
    if (free_bytes <= reserve_bytes || !largest || !row_bytes || !allocation_row_bytes ||
        largest <= padding_bytes || context_rows <= 0) return fallback;
    const uint64_t total_rows = (free_bytes - reserve_bytes) / 8 / row_bytes;
    const uint64_t single_rows = (largest - padding_bytes) / allocation_row_bytes;
    return (int) std::max<uint64_t>(1, std::min({uint64_t(context_rows), total_rows, single_rows}));
}

inline bool copy_all_fits(double lane_bytes, uint64_t ram_avail, double ram_margin, double commit_room) {
    if (!(lane_bytes > 0.0) || ram_avail == ~uint64_t{0}) return false;
    return lane_bytes + ram_margin <= (double) ram_avail && lane_bytes <= commit_room;
}

// Shared memory is a device property, not inferred from an unrelated fall in the OS's free RAM.
// A nonzero threshold is retained for explicit probes of the change in available RAM.
inline bool store_gpu_shares_ram(bool shared_memory, uint64_t avail_before, uint64_t avail_now, double min_drop_gb = 0.0) {
    if (!shared_memory) return false;
    if (min_drop_gb <= 0.0) return true;
    if (avail_before == ~uint64_t{0} || avail_now == ~uint64_t{0}) return false;
    return avail_before > avail_now && (double) (avail_before - avail_now) / 1e9 >= min_drop_gb;
}

// A slab grows only into scratch bytes the caller made available. tile_rows is the selected kernel's
// actual row multiple; the scalar fallback has a multiple of one.
inline int attn_slab_rows(int env, int chunk, int base, double spare_bytes, double bytes_per_row,
                          double expert_share = 0.0, bool v026 = false, int tile_rows = 1) {
    (void) expert_share;
    chunk = std::max(1, chunk);
    tile_rows = std::max(1, std::min(tile_rows, chunk));
    auto clampn = [&](int n) { return std::max(tile_rows, std::min(chunk, n) / tile_rows * tile_rows); };
    base = clampn(base);
    if (env > 0) return clampn(env);
    if (env == 0 || v026 || !(spare_bytes > 0.0) || !(bytes_per_row > 0.0)) return base;
    const double extra = std::min<double>(chunk - base, spare_bytes / bytes_per_row);
    return clampn(base + (int) extra);
}

// room_bytes is a staging-only budget, after persistent tensors and reserves, summed across devices.
// A fully resident expert set does not require a staging ring; a sub-MiB budget uses the direct path.
inline int staging_ring_mb(double share, double room_bytes, int n_dev, bool integrated,
                            bool matrix_engines, bool v026 = false) {
    (void) integrated, (void) matrix_engines, (void) v026;
    if (share >= 1.0 || !(room_bytes > 0.0) || n_dev < 1) return 0;
    return (int) std::min<double>(std::numeric_limits<int>::max(), room_bytes / n_dev / 1048576.0);
}

inline int experts_test_override(int chosen, int model_k) {
    if (const char* v = std::getenv("DS4_TOPK")) {
        const int n = std::atoi(v);
        if (n >= 1 && n <= model_k) return n;
    }
    return chosen;
}

// The one-MiB fallback is the smallest amount representable by this interface, not a tuned slab.
inline size_t attn_slab_base_mb(bool unified, size_t env_mb, bool v026 = false, size_t budget_bytes = 0) {
    (void) unified, (void) v026;
    if (env_mb != 0) return env_mb;
    return std::max<size_t>(1, budget_bytes / 1048576);
}

// host_mapping is the runtime's support for making this allocation visible to the device.
inline bool copy_pin_default(int env, bool arena_all_huge, bool host_mapping, double share,
                             bool have_chunks, bool v026 = false) {
    (void) arena_all_huge, (void) share;
    if (!host_mapping || !have_chunks) return false;
    return env >= 0 ? env != 0 : !v026;
}
inline bool copy_hugetlb_default(int env, double pool_free_bytes, double copy_bytes, bool v026 = false) {
    if (env >= 0) return env != 0;
    return !v026 && copy_bytes > 0.0 && pool_free_bytes >= copy_bytes;
}
inline bool fast_attn_default(int env, bool matrix_engines, bool v026 = false) {
    if (!matrix_engines) return false;
    return env >= 0 ? env != 0 : !v026;
}

// Confidence changes are explicit; matrix engines do not change a sampling decision.
inline float dspark_cum_value(bool has_env, float env_value, bool matrix_engines, bool v026 = false) {
    (void) matrix_engines, (void) v026;
    return has_env ? env_value : 0.0f;
}
inline bool store_feature_default(int env, bool v026 = false) {
    return env >= 0 ? env != 0 : !v026;
}
inline float settle_store_default(bool v026 = false) { return v026 ? 1.0f : 0.0f; }
inline bool settle_skips_copy_experts(bool v026 = false) { return !v026; }

// coalesced_supported says whether the selected kernel can use the coalesced path.
inline bool dq_old_default(int env, bool coalesced_supported, bool v026 = false) {
    return env >= 0 ? env != 0 : v026 || !coalesced_supported;
}
inline int head_per_default(bool v026 = false, int row_capacity = 1) {
    (void) v026;
    return std::max(1, row_capacity);
}
inline bool markov_head_q8_default(bool v026 = false) { return !v026; }
inline bool gate_up_x16_default(bool v026 = false) { return !v026; }
inline bool dense_pfgemm_default(int env, bool v026 = false) {
    return env >= 0 ? env != 0 : !v026;
}

}  // namespace strata::ds4
