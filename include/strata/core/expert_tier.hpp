// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// include/strata/core/expert_tier.hpp - the rules of the expert tiers that Qwen3.8-Flash-Next's engine
// (generate.cpp) and the other models' (PlainModel) share: which experts an adaptive round swaps into the VRAM tier,
// which ones STRATA_RAM_ADAPT moves into the RAM copy, and the copies back into the RAM copy's exchange buffers.
// When a round runs, how many swaps it may take and how its copies are issued stay with each engine: those were
// measured apart (plain_model.hpp's adapt_mib_per_token, generate.cpp's adapt_lag and --adapt-async).
#pragma once

#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/gpu.hpp"
#include "strata/core/on_device.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace strata::core {

/// One swap of a round: `in` (not in VRAM) takes the slot of `out` (in VRAM), or a free slot when `out` < 0.
struct TierSwap {
    float gain;
    int32_t layer, in, out;
};

/// The adaptive tier's rule.  In each layer the missing experts routed at least twice since the last round (their
/// decayed count; with free slots, any routed one for each free slot first) pair with the least routed experts in
/// VRAM, the most routed with the least, while the one coming in leads by `min_gain` (--adapt-min-gain); then every
/// layer's pairs by gain, the largest first.  `usage` and `res` are layer-major (`res` the slot of an expert in VRAM,
/// -1 when it is not); `free_slots(layer)` the layer's free slots; `skip(layer, expert)` leaves an expert out of the
/// candidates (STRATA_DISJOINT_ADAPT: one a helper GPU holds).
template <class FreeSlots, class Skip>
std::vector<TierSwap> choose_swaps(const float* usage, const int32_t* res, int64_t layers, int64_t n_expert,
                                   float min_gain, FreeSlots free_slots, Skip skip) {
    constexpr float kRouted = 2.0f;
    std::vector<TierSwap> swaps;
    std::vector<std::pair<float, int32_t>> cand, vict;
    for (int64_t l = 0; l < layers; ++l) {
        cand.clear();
        vict.clear();
        const float* u = usage + l * n_expert;
        const int32_t* r = res + l * n_expert;
        for (int32_t e = 0; e < (int32_t) n_expert; ++e) {
            if (r[e] < 0) { if (u[e] > 0.0f && !skip(l, e)) cand.emplace_back(u[e], e); }
            else vict.emplace_back(u[e], e);
        }
        // a free slot takes any expert routed since (an empty slot only costs a copy); one held, a missing expert
        // routed at least twice
        const size_t n_free = (size_t) free_slots(l);
        for (size_t i = 0; i < n_free; ++i) vict.emplace_back(-1.0f, -1);
        if (cand.empty() || vict.empty()) continue;
        std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.first > b.first; });
        size_t keep = 0;
        while (keep < cand.size() && (keep < n_free || cand[keep].first >= kRouted)) ++keep;
        cand.resize(keep);
        const size_t nc = std::min(cand.size(), vict.size());
        std::partial_sort(vict.begin(), vict.begin() + (ptrdiff_t) nc, vict.end(),
                          [](auto& a, auto& b) { return a.first < b.first; });
        for (size_t i = 0; i < nc; ++i) {
            if (cand[i].first < vict[i].first + min_gain) break;
            swaps.push_back({cand[i].first - vict[i].first, (int32_t) l, cand[i].second, vict[i].second});
        }
    }
    std::sort(swaps.begin(), swaps.end(), [](const TierSwap& a, const TierSwap& b) { return a.gain > b.gain; });
    return swaps;
}

/// choose_swaps for a tier without free slots and with every missing expert a candidate.
inline std::vector<TierSwap> choose_swaps(const float* usage, const int32_t* res, int64_t layers, int64_t n_expert,
                                          float min_gain) {
    return choose_swaps(usage, res, layers, n_expert, min_gain, [](int64_t) { return 0; },
                        [](int64_t, int32_t) { return false; });
}

/// STRATA_RAM_ADAPT's rule (upstream d7298883): the RAM copy is chosen once, at start, so an expert it does not hold is
/// read from the file every time the conversation routes to it.  In each layer the expert the windows route to most
/// (a decayed count of 2 or more) of those neither in VRAM nor in the RAM copy takes the RAM place of the copy's least
/// routed one (below 0.5): one pair a layer, the `n` with the largest difference.  `res` as choose_swaps'.
inline std::vector<TierSwap> choose_ram_swaps(const float* usage, const int32_t* res, int64_t layers, int64_t n_expert,
                                              const FileExpertSource& src, int n) {
    constexpr float kRouted = 2.0f, kUnused = 0.5f;
    std::vector<TierSwap> swaps;
    for (int64_t l = 0; l < layers; ++l) {
        const float* u = usage + l * n_expert;
        const int32_t* r = res + l * n_expert;
        int32_t in = -1, out = -1;
        for (int32_t e = 0; e < (int32_t) n_expert; ++e) {
            if (r[e] >= 0) continue;   // in VRAM: the RAM copy does not hold it
            if (src.has_resident(l, e)) { if (out < 0 || u[e] < u[out]) out = e; }
            else if (u[e] >= kRouted && (in < 0 || u[e] > u[in])) in = e;
        }
        if (in >= 0 && out >= 0 && u[out] < kUnused) swaps.push_back({u[in] - u[out], (int32_t) l, in, out});
    }
    std::sort(swaps.begin(), swaps.end(), [](const TierSwap& a, const TierSwap& b) { return a.gain > b.gain; });
    if ((int) swaps.size() > n) swaps.resize((size_t) std::max(n, 0));
    return swaps;
}

/// How many slots of a tier of `budget` bytes each layer takes, from its profile (`prior`, (layer, expert) pairs in
/// rank order): Qwen3.8-Flash-Next's fill (generate.cpp), the pairs in their order over all the layers while the bytes
/// last (`bytes(l)` a layer's expert, `local(l)` the layers this tier holds), then the rest of the bytes the layers in
/// turn; `alike`: every layer the same count instead (PlainModel's for Ornith-1.5, below).  `fill` the pairs that get
/// a slot, in rank order.
template <class Pairs>
struct ProfileFill {
    std::vector<int64_t> quota;   ///< per layer
    Pairs fill;                   ///< the profile's pairs within the quotas
};
template <class Pairs, class Bytes, class Local>
ProfileFill<Pairs> plan_profile_fill(const Pairs& prior, int64_t layers, int64_t n_expert, uint64_t budget, Bytes bytes,
                                     Local local, bool alike) {
    ProfileFill<Pairs> f;
    f.quota.assign((size_t) layers, 0);
    uint64_t left = budget;
    auto take = [&](int64_t l) {
        if (l < 0 || l >= layers || !local(l) || f.quota[(size_t) l] >= n_expert || bytes(l) > left) return false;
        ++f.quota[(size_t) l];
        left -= bytes(l);
        return true;
    };
    if (!alike)
        for (const auto& p : prior) take((int64_t) p.first);
    for (bool more = true; more;) {   // the rest (or, alike, all of it) the layers in turn
        more = false;
        for (int64_t l = 0; l < layers; ++l) more |= take(l);
    }
    std::vector<int64_t> used((size_t) layers, 0);
    for (const auto& p : prior)
        if ((int64_t) p.first >= 0 && (int64_t) p.first < layers && used[(size_t) p.first] < f.quota[(size_t) p.first]) {
            ++used[(size_t) p.first];
            f.fill.push_back(p);
        }
    return f;
}

/// #463 without the stall (upstream d69424c9, #764): a verify window takes the adaptive tier's previous swaps only once
/// they are this many windows old, and then waits for their copies (which have landed by then).  The window that first
/// computes a swapped-in expert on the GPU depends only on the window count, as with #463's wait, so the answers are as
/// reproducible.  Default 2 (upstream 2725c23f7 on sm_86 and sm_60; on XeStrata's GPUs the RTX 4070's decode 59.1 ->
/// 64.2 tok/s, IQ2_XS, and the B70's equal, IQ3_XXS 1K and 32K within 0.7%, 2026-10-20); STRATA_ADAPT_LAG=1 is #463's
/// (the very next window waits for the copies).  Both engines' (generate.cpp, PlainModel); `windows` the engine's
/// default, STRATA_ADAPT_LAG over it.
inline int adapt_lag(int windows = 2) {
    static const int v = [] {
        const char* e = std::getenv("STRATA_ADAPT_LAG");
        return e ? std::max(1, (int) std::strtol(e, nullptr, 10)) : 0;
    }();
    return v > 0 ? v : windows;
}

/// After a prompt, the adaptive tier's counts start from the prompt's routing instead of from nothing.  A count is the
/// rows routed to an expert, decayed by `decay` once a period of `period` tokens, so an expert the prompt routed with
/// probability p holds p * period / (1 - decay) rows at the tier's steady state: that is added for the routings
/// `routed` holds beyond `seen` (a request's own prompt in serve, where the counts run on from request to request).
/// `routed` [layers][n_expert], `routed_k` the experts a row takes.  The tier's rounds then swap by the counts as before.
/// STRATA_USAGE_SEED=0: the counts start from nothing.
inline void seed_usage(std::vector<float>& usage, const std::vector<int64_t>& routed, int64_t routed_k,
                       std::vector<int64_t>& seen, int64_t n_expert, int64_t period, double decay) {
    static const bool on = [] { const char* v = std::getenv("STRATA_USAGE_SEED"); return v == nullptr || *v != '0'; }();
    if (!on || usage.empty() || routed.empty() || routed_k <= 0 || usage.size() != routed.size() || n_expert <= 0 ||
        period <= 0)
        return;
    if (seen.size() != routed.size()) seen.assign(routed.size(), 0);
    const double rows = (double) period / std::max(1e-3, 1.0 - decay);
    for (size_t l0 = 0; l0 + (size_t) n_expert <= routed.size(); l0 += (size_t) n_expert) {
        int64_t total = 0;
        for (size_t i = l0; i < l0 + (size_t) n_expert; ++i) total += routed[i] - seen[i];
        if (total <= 0) continue;   // not a MoE layer, or nothing new
        for (size_t i = l0; i < l0 + (size_t) n_expert; ++i)
            usage[i] += (float) ((double) (routed[i] - seen[i]) * (double) routed_k / (double) total * rows);
    }
    seen = routed;
}

/// The decode's counts from the rows a window kept (the drafts the model accepted and the token after them), not from
/// every row it computed: `rec` [layers][max_t][k] the window's routing (ExpertDispatch::route_rec), its first `rows`
/// rows counted for the layers in `layers` (Qwen3.8's Verifier::add_routing does the same from its own record).
template <class Layers>
void add_route_rows(std::vector<float>& usage, const std::vector<int32_t>& rec, const Layers& layers, int64_t n_expert,
                    int64_t max_t, int64_t k, int rows) {
    rows = (int) std::min<int64_t>(rows, max_t);
    for (int64_t l : layers) {
        if ((size_t) ((l + 1) * max_t * k) > rec.size()) continue;
        for (int t = 0; t < rows; ++t)
            for (int64_t j = 0; j < k; ++j) {
                const int32_t e = rec[(size_t) ((l * max_t + t) * k + j)];
                if (e >= 0 && e < n_expert) usage[(size_t) (l * n_expert + e)] += 1.0f;
            }
    }
}

/// A prompt chunk's layer, for the CPU's share of the experts the GPU does not hold: `missed` the routed experts not in
/// VRAM, `cand` (rows, expert) those the CPU may take, fewest rows first - at most MAXT rows (one multi-token job) and a
/// blob the CPU reads where it lies, page-locked or not assembled into a short-lived buffer (the CPU's jobs keep their
/// pointers while the GPU's groups stage the rest; upstream 92bce20e).  How many of them the CPU takes is each engine's
/// (a share of `missed`).  `in_vram(e)`, `skip(e)`: an expert another part computes (a helper GPU's).
struct CpuShareCandidates {
    int64_t missed = 0;
    std::vector<std::pair<int32_t, int32_t>> cand;
};
template <class InVram, class Skip>
CpuShareCandidates cpu_share_candidates(const int32_t* counts, int64_t n_expert, int64_t layer, const ExpertSource& src,
                                        InVram in_vram, Skip skip) {
    CpuShareCandidates r;
    for (int32_t e = 0; e < (int32_t) n_expert; ++e) {
        const int32_t c = counts[e];
        if (c == 0 || in_vram(e) || skip(e)) continue;
        ++r.missed;
        if (c <= kernels::cpu::MAXT && (src.pinned(layer, e) || !src.transient(layer, e))) r.cand.emplace_back(c, e);
    }
    std::sort(r.cand.begin(), r.cand.end());
    return r;
}

// The resident RAM mode and the adaptive tier.  A swap copies `in` (held in RAM) into the slot of `out` (held only
// by that slot).  Before the slot is overwritten, `out`'s bytes are copied back from it into an exchange buffer, so
// the CPU computes `out` from RAM while the swap is in flight; when the swap has landed, `commit_exchanges` moves
// them into `in`'s place in RAM.  The RAM copy then again holds exactly the experts no core slot does, and no swap
// reads the file.  Swaps that need no exchange (`out` in the lend region is held in RAM already; or `in` is not; or
// a free slot) go on as before; ones beyond the buffers' room wait for a later round.  Runs on the adaptive tier's
// thread while the GPU commits and drafts: the copies back are on its stream, and waited for before the refills are
// queued.
//
// With a layer split, `host_res` holds slot numbers of WHICHEVER cache owns the layer, so the copy back reads the
// owning card's cache on that card's stream: `locate(layer)` names them (upstream fe9c10ca: reading a later stage's
// slot number out of the first card's cache put another expert's bytes in RAM, and the answers turned to garbage).
struct SwapHome {
    ExpertCache* cache;
    gpu::Stream stream;
    int dev;   ///< -1: the first card
};

template <class Swap, class Locate>
bool resident_stage_swaps(FileExpertSource& src, const std::vector<int32_t>& host_res, int64_t n_expert,
                          std::vector<Swap>& swaps, Locate locate) {
    if (!src.complement_ready() || swaps.empty()) return true;
    struct Staged { int32_t layer, in, out; int64_t q; };
    std::vector<Staged> staged;
    std::vector<SwapHome> used;   // the streams the copies back went on
    std::vector<Swap> kept;
    kept.reserve(swaps.size());
    for (const Swap& s : swaps) {
        if (s.out < 0 || !src.has_resident(s.layer, s.in) || src.has_resident(s.layer, s.out)) {
            kept.push_back(s);
            continue;
        }
        const int64_t q = (int64_t) staged.size();
        if (q >= src.exchange_capacity()) continue;
        const int32_t slot = host_res[(size_t) s.layer * (size_t) n_expert + (size_t) s.out];
        if (slot < 0) continue;
        const SwapHome home = locate(s.layer);
        const OnDevice on(home.dev);
        if (gpu::copy_async(src.exchange_buffer(q, s.layer), home.cache->device_slot(slot),
                            (size_t) kernels::cpu::expert_layout().blob_bytes(s.layer), home.stream) != true)
            return false;
        if (std::find_if(used.begin(), used.end(), [&](const SwapHome& h) { return h.stream == home.stream; }) == used.end())
            used.push_back(home);
        staged.push_back({s.layer, s.in, s.out, q});
        kept.push_back(s);
    }
    if (!staged.empty()) {
        for (const SwapHome& h : used) {
            const OnDevice on(h.dev);
            if (gpu::stream_sync(h.stream) != true) return false;
        }
        for (const Staged& x : staged)
            if (!src.stage_exchange(x.layer, x.in, x.out, x.q)) return false;
    }
    swaps.swap(kept);
    return true;
}

}  // namespace strata::core
