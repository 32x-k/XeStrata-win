// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
#include "strata/core/conversation_memory.hpp"

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <sstream>

using namespace strata::core;
namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
}

int main() {
    for (const char* text : {"", "MemFree: 100 kB\n", "MemAvailable: -1 kB\n",
                            "MemAvailable: +1 kB\n", "MemAvailable: 1 MB\n",
                            "MemAvailable: 1\n", "MemAvailable: 1x kB\n",
                            "MemAvailable: 18446744073709551615 kB\n",
                            "MemAvailable: 18446744073709551616 kB\n",
                            "MemAvailable: 1 kB trailing\n",
                            "MemAvailable: 1 kB\nMemAvailable: 2 kB\n"}) {
        std::istringstream input(text);
        check(!conversation_mem_available(input), "missing/malformed telemetry is unknown");
    }
    std::istringstream normal("MemTotal: 999999 kB\nMemAvailable:    12345 kB\nSwapFree: 777 kB\n");
    check(conversation_mem_available(normal) == 12345ULL * 1024, "only MemAvailable is counted");
    std::istringstream zero("MemAvailable: 0 kB");
    check(conversation_mem_available(zero) == 0, "zero available memory is known");
    std::istringstream broken("MemAvailable: 123 kB\n");
    broken.setstate(std::ios::badbit);
    check(!conversation_mem_available(broken), "I/O failure declines admission");
    check(!conversation_memory_admit({}, 0, 0), "unknown fails closed even with zero floor");
    check(conversation_memory_admit(100, 40, 60), "exact allocation plus floor fits");
    check(!conversation_memory_admit(99, 40, 60), "one byte below required memory rejected");
    check(!conversation_memory_admit(59, 0, 60), "floor subtraction cannot underflow");
    check(conversation_memory_admit(60, 0, 60), "post-capture floor check");
    check(!conversation_memory_admit(100, std::numeric_limits<uint64_t>::max(), 1), "allocation arithmetic cannot overflow");
    check(conversation_memory_admit(std::numeric_limits<uint64_t>::max(),
                                   std::numeric_limits<uint64_t>::max(), 0), "maximal exact bound");
    check(!conversation_memory_admit(std::numeric_limits<uint64_t>::max(),
                                    std::numeric_limits<uint64_t>::max(), 1), "maximal sum overflow rejected");
    // #3493: the budget used when --conversation-cache-mib is not given
    const auto GiB = [](uint64_t n) { return n << 30; };
    check(auto_conversation_cache_mib({}, 2560) == 0, "unknown telemetry: no cache");
    check(auto_conversation_cache_mib(GiB(2), 2560) == 0, "below the parking floor: off");
    check(auto_conversation_cache_mib(GiB(4), 2560) == 0, "4 GiB free: 40% of 1.5 GiB is under 1 GiB: off");
    check(auto_conversation_cache_mib(GiB(8), 2560) == 2048, "8 GiB free: 40% of 5.5 GiB, in steps of 256");
    check(auto_conversation_cache_mib(GiB(20), 2560) == 7168, "20 GiB free: 40% of 17.5 GiB");
    check(auto_conversation_cache_mib(GiB(40), 2560) == 8192, "capped at 8 GiB");
    check(auto_conversation_cache_mib(GiB(1000), 0) == 8192, "a big server: still the cap");
    check(auto_conversation_cache_mib(GiB(40), 1000000) == 0, "a floor above the RAM: off, no underflow");
    // Test the real provider without assuming any particular amount of free RAM.
    const auto available = available_host_bytes();
    check(!available || conversation_memory_admit(available, 0, 0), "provider returns bytes or unknown");
    // #4518: after the load, does the PLE table's lock leave the automatic conversation cache no room?  (min free 2560 MiB,
    // cache 40% of the rest, 8192 MiB at most, 1024 MiB at least)
    {
        const uint64_t table = GiB(26) + 870ull * (1ull << 20);   // IQ3_S's table, 26.8 GiB
        const int64_t floor_mib = 2560;
        check(!ple_lock_blocks_cache(std::nullopt, table, floor_mib), "unknown telemetry: leave the lock");
        check(!ple_lock_blocks_cache(GiB(40), 0, floor_mib), "unknown table size: leave the lock");
        check(ple_lock_blocks_cache(GiB(4), table, floor_mib), "4 GiB left with the table locked (cache off), 30 GiB without");
        check(ple_lock_blocks_cache(GiB(10), table, floor_mib), "10 GiB left: cache 1.9 GiB with the lock, 8 GiB without");
        check(!ple_lock_blocks_cache(GiB(40), table, floor_mib), "40 GiB left: the cache is already at its 8 GiB cap");
        check(!ple_lock_blocks_cache(GiB(23), table, floor_mib), "23 GiB: the cap is just reached with the lock");
        check(ple_lock_blocks_cache(GiB(22), table, floor_mib), "22 GiB: the lock costs the cache some of its 8 GiB");
        check(!ple_lock_blocks_cache(GiB(1), 1ull << 20, floor_mib), "a tiny table cannot make a cache");
        check(!ple_lock_blocks_cache(std::numeric_limits<uint64_t>::max(), table, floor_mib), "no overflow");
    }
    std::printf("conversation_memory_test: %d checks passed\n", checks);
}
