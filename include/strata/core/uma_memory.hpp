#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>

namespace strata::core {
inline uint64_t parse_uma_gib_env(const char* val, uint64_t fallback) {
    if (!val || !*val) return fallback;
    uint64_t n = 0;
    for (const char* p = val; *p; ++p) {
        if (*p < '0' || *p > '9') return fallback;
        n = n * 10 + unsigned(*p - '0');
        if (n > 1024) return fallback;
    }
    return n;
}
inline uint64_t uma_commit_margin_bytes(uint64_t total_ram, const char* env_gib = nullptr) {
    const uint64_t invalid = std::numeric_limits<uint64_t>::max();
    const uint64_t n = parse_uma_gib_env(env_gib, invalid);
    return n == invalid ? std::max(uint64_t(8) << 30, total_ram / 10) : n << 30;
}
struct UmaMemoryInput {
    bool runtime_free_known = false;
    uint64_t runtime_free = 0, runtime_total = 0, os_available = 0, gpu_reclaimable = 0, headroom = 0;
    bool has_commit_limit = false;
    uint64_t commit_available = 0, commit_margin = 0;
};
struct UmaMemoryResult { uint64_t effective_free; bool os_capped; bool commit_capped; };
inline UmaMemoryResult compute_uma_free(const UmaMemoryInput& i) {
    const auto max = std::numeric_limits<uint64_t>::max();
    const uint64_t sum = i.gpu_reclaimable > max - i.os_available ? max : i.os_available + i.gpu_reclaimable;
    const uint64_t os = std::min(i.runtime_total, sum > i.headroom ? sum - i.headroom : 0);
    uint64_t free = std::min(i.runtime_free_known ? i.runtime_free : i.runtime_total, i.runtime_total);
    const bool os_capped = os < free;
    free = std::min(free, os);
    const uint64_t commit = i.commit_available > i.commit_margin ? i.commit_available - i.commit_margin : 0;
    const bool commit_capped = i.has_commit_limit && commit < free;
    if (i.has_commit_limit) free = std::min(free, commit);
    return {free, os_capped, commit_capped};
}
}
