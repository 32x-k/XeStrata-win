// SPDX-License-Identifier: MIT
#include "strata/prefill/helper_policy.hpp"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
int main() {
    using namespace strata::prefill;
    assert(uma_host_helper_cpu_threads(16, true, true, false) == 8);
    assert(uma_host_helper_cpu_threads(1, true, true, false) == 1);
    assert(uma_host_helper_cpu_threads(3, true, true, false) == 2);
    assert(uma_host_helper_cpu_threads(16, false, true, false) == 16);
    assert(uma_host_helper_cpu_threads(16, true, false, false) == 16);
    assert(uma_host_helper_cpu_threads(16, true, true, true, "3") == 16);
    assert(uma_host_helper_cpu_threads(16, true, true, false, "0") == 16);
    assert(uma_host_helper_cpu_threads(16, true, true, false, "3") == 3);
    assert(uma_host_helper_cpu_threads(16, true, true, false, "999") == 16);
    assert(uma_host_helper_cpu_threads(16, true, true, false, "-1") == 8);
    assert(uma_host_helper_cpu_threads(16, true, true, false, "3x") == 8);
    assert(uma_host_helper_cpu_threads(16, true, true, false, "999999999999999999999999") == 8);
    assert(uma_helper_default_cap(true, true) == 32768);
    assert(uma_helper_default_cap(true, false) == 8192);
    assert(uma_helper_default_cap(false, true) == 8192);
    assert(prompt_helper_max_rows(8000, 100, 10, 5, false) == 5);
    assert(prompt_helper_max_rows(8000, 100, 10, 5, false, true) == 10);
    assert(prompt_helper_max_rows(8000, 100, 10, 0, true, true) == 20);
    assert(prompt_helper_max_rows(8000000, 100, 10, 0, true, true) == 256);
    assert(prompt_helper_max_rows(8000, 100, 10, 0, false) == 0);
    assert(prompt_helper_max_rows(8000, 100, 10, 0, true) == 10);
    assert(prompt_helper_max_rows(8000000, 100, 10, 0, true) == 256);
    assert(prompt_helper_max_rows(8000, 100, 0, 10, true) == 0);
}
