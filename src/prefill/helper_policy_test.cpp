// SPDX-License-Identifier: MIT
#include "strata/prefill/helper_policy.hpp"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
int main() {
    using namespace strata::prefill;
    assert(uma_helper_default_cap(true, true) == 32768);
    assert(uma_helper_default_cap(true, false) == 8192);
    assert(uma_helper_default_cap(false, true) == 8192);
    assert(prompt_helper_max_rows(8000, 100, 10, 5, false) == 5);
    assert(prompt_helper_max_rows(8000, 100, 10, 0, false) == 0);
    assert(prompt_helper_max_rows(8000, 100, 10, 0, true) == 10);
    assert(prompt_helper_max_rows(8000000, 100, 10, 0, true) == 256);
    assert(prompt_helper_max_rows(8000, 100, 0, 10, true) == 0);
}
