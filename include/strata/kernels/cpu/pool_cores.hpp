// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// include/strata/kernels/cpu/pool_cores.hpp - `--pool-cores` (experimental, off by default) and the start-up line about
// where the CPU expert workers run: reading the list, checking it against the processors the process may use, and
// describing the placement.  Pure functions (no OS calls), unit tested in pool_cores_test.cpp; the code is upstream's
// (pool_plan.hpp, 7a5599581, c4f1e05db), the pool's use of it is XeStrata's (pool.cpp).
#pragma once

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace strata::kernels::cpu {

/// One logical processor the process may use and the physical core it belongs to (`core`: any number that is the
/// same for the SMT siblings of one core and different between cores, across sockets too; -1: the OS did not say).
/// `cpu` is the OS's number: Linux CPU id, on Windows group * 64 + the processor number inside the group (Task
/// Manager's CPU number on a PC with one processor group).
struct LogicalCpuInfo {
    int cpu = -1;
    long core = -1;
};

/// "2,4,6,8,10,12" -> "2,4,...,12" when the numbers step by a constant from the second on and there are at least four of
/// them; "0,1,2,3" -> "0-3"; anything else is written out in full (at most 64 numbers, then "... and N more").
inline std::string format_cpu_list(const std::vector<int>& v) {
    if (v.empty()) return "none";
    auto num = [](int x) { return std::to_string(x); };
    if (v.size() >= 4) {
        const int step = v[1] - v[0];
        bool even = step != 0;
        for (size_t i = 2; even && i < v.size(); ++i) even = v[i] - v[i - 1] == step;
        if (even) return step == 1 ? num(v.front()) + "-" + num(v.back())
                                   : num(v[0]) + "," + num(v[1]) + ",...," + num(v.back());
    }
    std::string s;
    for (size_t i = 0; i < v.size() && i < 64; ++i) s += (i ? "," : "") + num(v[i]);
    if (v.size() > 64) s += ",... and " + std::to_string(v.size() - 64) + " more";
    return s;
}

/// `--pool-cores` / STRATA_POOL_CORES: "0,2,4,6" or "2-14:2" (first-last, optional :step) or a mix, in the order given.
/// False with a plain `err` for anything else (empty items, letters, a reversed range, a step of 0); `out` is then empty.
/// A number that is listed twice is kept once (the second is dropped).
inline bool parse_pool_cores(const std::string& text, std::vector<int>& out, std::string& err) {
    out.clear();
    err.clear();
    size_t i = 0;
    auto skip = [&] { while (i < text.size() && (text[i] == ' ' || text[i] == '\t')) ++i; };
    auto number = [&](long& v) {
        skip();
        if (i >= text.size() || text[i] < '0' || text[i] > '9') return false;
        v = 0;
        while (i < text.size() && text[i] >= '0' && text[i] <= '9') {
            v = v * 10 + (text[i] - '0');
            if (v > 100000) return false;
            ++i;
        }
        return true;
    };
    skip();
    if (i >= text.size()) { err = "the list is empty (write it like 0,2,4,6 or 2-14:2)"; return false; }
    for (;;) {
        long lo = 0, hi = 0, step = 1;
        if (!number(lo)) { err = "'" + text + "' is not a list of logical processor numbers (write it like 0,2,4,6 or 2-14:2)"; out.clear(); return false; }
        hi = lo;
        skip();
        if (i < text.size() && text[i] == '-') {
            ++i;
            if (!number(hi) || hi < lo) { err = "'" + text + "' has a range that does not go up (write it like 2-14:2)"; out.clear(); return false; }
            skip();
            if (i < text.size() && text[i] == ':') {
                ++i;
                if (!number(step) || step < 1) { err = "'" + text + "' has a step that is not a positive number"; out.clear(); return false; }
            }
        }
        for (long c = lo; c <= hi; c += step) {
            if (std::find(out.begin(), out.end(), (int) c) == out.end()) out.push_back((int) c);
            if (out.size() > 4096) { err = "the list has more than 4096 logical processors"; out.clear(); return false; }
        }
        skip();
        if (i == text.size()) break;
        if (text[i] != ',') { err = "'" + text + "' is not a list of logical processor numbers (write it like 0,2,4,6 or 2-14:2)"; out.clear(); return false; }
        ++i;
    }
    return true;
}

/// What `--pool-cores` made of the person's list: the processors the workers get, in the order given, and what to tell
/// them.  Recommend, never force: the only entries dropped are those that cannot be used at all (a processor the process
/// may not run on, or that does not exist, cannot be pinned to).  Two entries that are SMT siblings of one core, or the
/// host thread's own processor, are kept with a warning: the workers then share a core, which is what the person asked for.
struct PoolCoresCheck {
    std::vector<int> cpus;
    std::vector<std::string> warnings;
};

inline PoolCoresCheck check_pool_cores(const std::vector<int>& list, const std::vector<LogicalCpuInfo>& layout, int host_cpu) {
    PoolCoresCheck r;
    auto find = [&](int cpu) -> const LogicalCpuInfo* {
        for (const auto& l : layout)
            if (l.cpu == cpu) return &l;
        return nullptr;
    };
    std::vector<std::pair<long, int>> seen;   // (core, the first listed cpu of it)
    std::vector<int> dropped;
    for (int c : list) {
        const LogicalCpuInfo* info = find(c);
        if (info == nullptr) { dropped.push_back(c); continue; }
        r.cpus.push_back(c);
        if (info->core < 0) continue;
        bool shared = false;
        for (const auto& s : seen)
            if (s.first == info->core) {
                r.warnings.push_back("logical processors " + std::to_string(s.second) + " and " + std::to_string(c) +
                                     " are the two hardware threads of one core: two workers on it share the core's memory "
                                     "bandwidth and slow each other down (one per physical core is the default)");
                shared = true;
                break;
            }
        if (!shared) seen.emplace_back(info->core, c);
    }
    if (!dropped.empty())
        r.warnings.push_back("left out: " + format_cpu_list(dropped) +
                             (dropped.size() == 1 ? " is not a logical processor this process may run on"
                                                  : " are not logical processors this process may run on"));
    if (host_cpu >= 0) {
        const LogicalCpuInfo* h = find(host_cpu);
        for (int c : r.cpus) {
            const LogicalCpuInfo* info = find(c);
            if (c == host_cpu) {
                r.warnings.push_back("logical processor " + std::to_string(c) + " is also the host thread's: a worker there "
                                     "competes with the thread that waits for the graphics card");
                break;
            }
            if (h != nullptr && info != nullptr && h->core >= 0 && info->core == h->core) {
                r.warnings.push_back("logical processor " + std::to_string(c) + " shares a core with the host thread's " +
                                     std::to_string(host_cpu));
                break;
            }
        }
    }
    return r;
}

/// The start-up line about where the workers run, without the "strata generate: " prefix:
///   "CPU workers on logical processors 2,4,...,30 (15 workers, one per physical core; host on 0)".
/// `layout` empty or a worker/host processor not in it: the OS did not say which processors are SMT siblings, and the
/// line says so instead of claiming "one per physical core".  `list_given`: the workers are the person's --pool-cores.
inline std::string describe_worker_placement(const std::vector<int>& workers, int host_cpu,
                                             const std::vector<LogicalCpuInfo>& layout, bool list_given) {
    if (workers.empty()) return "";
    auto core_of = [&](int cpu) -> long {
        for (const auto& l : layout)
            if (l.cpu == cpu) return l.core;
        return -1;
    };
    bool known = true;
    std::vector<long> cores;
    for (int w : workers) {
        const long c = core_of(w);
        if (c < 0) known = false;
        cores.push_back(c);
    }
    const long host_core = host_cpu >= 0 ? core_of(host_cpu) : -1;
    std::string s = "CPU workers on logical processors " + format_cpu_list(workers) + " (" + std::to_string(workers.size()) +
                    (workers.size() == 1 ? " worker" : " workers");
    if (list_given) s += ", your --pool-cores list";
    if (!known) {
        s += ", the OS did not say which logical processors share a core";
    } else {
        std::vector<long> distinct = cores;
        std::sort(distinct.begin(), distinct.end());
        distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());
        if (distinct.size() == workers.size()) {
            s += ", one per physical core";
        } else {
            size_t doubled = 0;   // cores that carry more than one worker
            for (long d : distinct) doubled += std::count(cores.begin(), cores.end(), d) > 1;
            s += " on " + std::to_string(distinct.size()) + " physical core" + (distinct.size() == 1 ? "" : "s") +
                 "; " + std::to_string(doubled) + (doubled == 1 ? " core carries" : " cores carry") + " more than one worker (they slow each other down)";
        }
        if (host_core >= 0 && std::find(cores.begin(), cores.end(), host_core) != cores.end())
            s += "; a worker shares the host's core";
    }
    s += host_cpu >= 0 ? "; host on " + std::to_string(host_cpu) : "; host not pinned";
    return s + ")";
}

}  // namespace strata::kernels::cpu
