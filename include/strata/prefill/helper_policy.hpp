// SPDX-License-Identifier: MIT
#pragma once
#include <algorithm>
#include <cstdint>
#include <climits>

namespace strata::prefill {
inline int uma_host_helper_cpu_threads(int default_threads, bool helper_integrated, bool helper_host_mode,
                                       bool user_specified_threads, const char* env_override = nullptr) {
    if (user_specified_threads || !helper_integrated || !helper_host_mode) return default_threads;
    if (env_override && *env_override) {
        unsigned long long value = 0;
        bool valid = true;
        for (const char* p = env_override; *p; ++p) {
            if (*p < '0' || *p > '9') { valid = false; break; }
            const unsigned digit = *p - '0';
            if (value > (ULLONG_MAX - digit) / 10) { valid = false; break; }
            value = value * 10 + digit;
        }
        if (valid) {
            if (value == 0) return default_threads;
            return (int) std::min<unsigned long long>(value, (unsigned) std::max(1, default_threads));
        }
    }
    return std::min(default_threads, std::max(2, default_threads / 2));
}
inline int64_t uma_helper_default_cap(bool integrated, bool host_mode) {
    (void) integrated; (void) host_mode;
    return 8192;
}
inline int64_t prompt_helper_max_rows(double blob, int64_t n_embd, double primary, double helper, bool integrated,
                                      bool fp16_exchange = false) {
    const double exchange_bytes = 2.0 * n_embd * (fp16_exchange ? 2.0 : 4.0);
    if (integrated) return primary > 0 ? (int64_t) std::min(256.0, blob / exchange_bytes) : 0;
    return primary > 0 && helper > 0 ? (int64_t) (blob / exchange_bytes * helper / primary) : 0;
}
}
