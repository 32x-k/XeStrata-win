// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/program/done_line.hpp - the DONE line both engines' --serve loops end a request with (generate.cpp,
// plain_generate.cpp), written in one place so the server (serve/, tools/bench_core.py) reads the same fields from
// either.
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>

namespace strata::program {

// DONE <generated> <prompt> <prompt ms> <decode ms> <finish> <drafts accepted> <drafts offered> <reused> [hits] [lookups]
//      [RAM blobs] [file blobs] [file MB]   (CS-T tiers; appended, so an older server reads the rest)
//      [prompt tokens read]   (#471: fewer than <prompt> - <reused> when a cancel stopped the read)
//      [offloaded]   (#588: the decode's routed experts the GPU read over PCIe or another GPU computed;
//                    not in [lookups])
//      [lookup-chain accepted] [lookup-chain offered] [suffix accepted] [suffix offered] [windows]
//      (only with --lookup-chain: `tail`)
struct DoneLine {
    int64_t generated = 0, prompt = 0;
    double prompt_ms = 0, decode_ms = 0;
    const char* finish = "stop";
    int64_t accepted = 0, offered = 0, reused = 0, hits = 0, lookups = 0, ram_blobs = 0, file_blobs = 0;
    double file_mb = 0;
    int64_t read = 0, offloaded = 0;
    std::string tail;   ///< more fields, each after a space
};

inline void print_done(const DoneLine& d) {
    std::printf("DONE %lld %lld %.1f %.1f %s %lld %lld %lld %lld %lld %lld %lld %.1f %lld %lld%s\n",
                (long long) d.generated, (long long) d.prompt, d.prompt_ms, d.decode_ms, d.finish, (long long) d.accepted,
                (long long) d.offered, (long long) d.reused, (long long) d.hits, (long long) d.lookups,
                (long long) d.ram_blobs, (long long) d.file_blobs, d.file_mb, (long long) d.read, (long long) d.offloaded,
                d.tail.c_str());
}

}  // namespace strata::program
