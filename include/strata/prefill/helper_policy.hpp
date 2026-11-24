// SPDX-License-Identifier: MIT
#pragma once
#include <algorithm>
#include <cstdint>

namespace strata::prefill {
inline int64_t uma_helper_default_cap(bool integrated, bool host_mode) {
    return integrated && host_mode ? 32768 : 8192;
}
inline int64_t prompt_helper_max_rows(double blob, int64_t n_embd, double primary, double helper, bool integrated,
                                      bool fp16_exchange = false) {
    const double exchange_bytes = 2.0 * n_embd * (fp16_exchange ? 2.0 : 4.0);
    if (integrated) return primary > 0 ? (int64_t) std::min(256.0, blob / exchange_bytes) : 0;
    return primary > 0 && helper > 0 ? (int64_t) (blob / exchange_bytes * helper / primary) : 0;
}
}
