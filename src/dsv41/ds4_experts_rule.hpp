// Strata. Experts per word: the "faster mode" rule of the DeepSeek V4.1 engine.  Header only, no GPU code (tests/dsv41/experts_rule_test.cpp).
//
// The model routes every word to its top 6 experts.  On a PC with a small graphics card the CPU lane and the SSD set the speed, and
// each expert dropped saves their bytes: the router's top N are kept and their weights renormalised (the same selection the full
// model makes, a shorter list).  The default is 5 where the graphics memory is 16 GiB or less, or the GPU shares the PC's RAM, the model's own count elsewhere;
// --experts N says it outright (--experts 6: full quality), DS4_TOPK=N is the test override over everything.
#pragma once

#include <cstdint>
#include <algorithm>
#include <cstdlib>

namespace strata::ds4 {

constexpr uint64_t kFasterModeVramMax = 16ull << 30;   // GiB of graphics memory in all, at most
constexpr int kFasterModeExperts = 5;

// requested: --experts (-1: not given, the rule decides); vram_bytes: the total memory of the cards the engine uses;
// model_k: the experts a word takes in the model; integrated: a GPU that shares the PC's RAM (Strix Halo: part of the experts live only on
// the SSD, each expert dropped is less read - measured 5 against 6 on a Radeon 8065S: decode +11..13%, prompts +5..7%).  The result is in 1 .. model_k.
inline int experts_per_word(int requested, uint64_t vram_bytes, int model_k, bool integrated = false) {
    if (model_k <= 1) return model_k;
    int n = requested;
    if (n < 1) n = integrated || (vram_bytes != 0 && vram_bytes <= kFasterModeVramMax) ? kFasterModeExperts : model_k;
    if (n > model_k) n = model_k;
    return n < 1 ? 1 : n;
}

// Prompt chunk (--prefill-chunk) default, one rule decided once at load from bytes (no timing).  The chunk's buffers come out of the VRAM the
// expert tier would have, and every chunk streams the experts the cards do not hold once, so a bigger chunk is fewer passes.  Measured (8K prompt):
//   - the GPUs hold most of the experts (4 x R9700, share 0.86): 1024 370 tok/s, 8192 214  -> small chunk;
//   - the experts are read in place and the CPU lane streams most of them (RTX 5070 / 3060 12 GB, 1 x R9700): 8192 best (5070 244 against 186
//     at 1024, 3060 228 against 175, 1 x R9700 431 against 200) -> the big chunk (the GGUF pack's long-standing default);
//   - the experts copied into RAM (--mmap-experts off): H100 32 GiB 2048 502, 4096 538, 8192 495 -> 4096 (not measured on HIP: no).
// Reference KL against the original model does not move with the chunk (1024 0.1705, 2048 0.1692, 4096 0.1688, 8192 0.1695).
// Alloc failure halves the chunk at load (4096 -> 2048 -> 1024 ...), KV and score buffers are allocated whole at the full --max-context.
constexpr uint64_t kChunk4096MinVram = 10ull << 30;
constexpr uint64_t kChunkInPlace4096Below = 16ull << 30;
// own_default: a run with tp 1 and no --threads (`strata launch` writes --tp 1 for 4 or more AMD cards); multi_tp: two or more cards, tp above 1
// and no `--threads on` (the engine's own default tp is 2, and `strata launch` writes --tp 2 for 2-3 cards); gpu_share: the share of the
// routed experts the cards can hold (1 = all);
// gguf_copy: a GGUF pack with --mmap-experts off; file_experts: the experts stay in the files; hip: a HIP build; min_vram_bytes: the
// smallest card; big: the chunk of the streamed case (the engine's kVlRows, 8192)
inline int prefill_chunk_default(bool own_default, double gpu_share, bool gguf_copy, bool file_experts, bool hip, uint64_t min_vram_bytes,
                                 int big = 8192, bool multi_tp = false) {
    // The cards hold most of the experts: the small chunk is the measured one.  Two datapoints, both on 4 x R9700: with `--tp 1` (what
    // `strata launch` writes for 4 or more AMD cards), share 0.86: 1024 370 against 214 tok/s at 8192; and with the engine's own default tp
    // (2, pairs of cards, nothing written), where the rule used to fall through to 2048/8192: 2.25 bpw, --dspark auto, ids 5705e1164a, 8K
    // warm prompts 1024 -> 1247 ppts against 690 at 8192, decode unchanged.  So a multi-card tp run takes it too.  One card with the default
    // tp, and `--threads on` (it stalled at the first 8K prompt in the 4-card runs), keep their old defaults: not measured with 1024.
    // An explicit --prefill-chunk N wins over all of this (the call site takes the rule only when none was given).
    if ((own_default || multi_tp) && gpu_share >= 0.5) return 1024;
    if (gguf_copy && !hip && min_vram_bytes >= kChunk4096MinVram) return 4096;
    // the experts read in place on a card under 16 GiB (the 12 GB PCs; not measured on HIP): 4096 keeps ~120 more experts resident than 8192 (RTX 5070
    // 287 against 165, RTX 3060 386 against 277) - 5 interleaved pairs, 4096 against 8192, decode: 5070 agent x1.020 (5/5) code x1.004 (3/5), 3060
    // agent x1.018 (4/5) code x1.091 (5/5); an agent turn of 3K new tokens on a 16K context takes the same time (5070 18.9-19.2 s against 19.1-19.4 s,
    // 3060 21.4 s against 20.9-21.4 s) - for 9% less speed on a cold 8K prompt (5070 223 against 244 tok/s)
    if (file_experts && !hip && min_vram_bytes != 0 && min_vram_bytes < kChunkInPlace4096Below) return 4096;
    return file_experts ? big : 2048;
}

// --mmap-experts auto copies all the CPU lane's experts into RAM at load (no demand store) only when they fit: lane_bytes of experts,
// ram_avail the RAM this process can still get (~0: unknown -> no), ram_margin left to the rest, commit_room what Windows still lets it
// commit beside its margin (1e300: no such limit, Linux).  All or none: a lane that does not fit both leaves the demand store as it was.
// Measured 1 x R9700 + 251 GB, 2.00 pack: decode +10-25% (the copies sit on huge pages, the store's page-locked arena does not).
inline bool copy_all_fits(double lane_bytes, uint64_t ram_avail, double ram_margin, double commit_room) {
    if (!(lane_bytes > 0.0) || ram_avail == ~uint64_t{0}) return false;
    return lane_bytes + ram_margin <= (double) ram_avail && lane_bytes <= commit_room;
}

// The chunked prompt's attention slab (rows a pass: the dense projections q_b / wo_a / wo_b are dequantized and multiplied once a slab, so
// a 256-row slab dequantizes them 32 times for an 8K prompt).  `base`: the 0.2.6 rule (8 MB of rows / heads width, 256 on this model).
// DS4_ATTN_SLAB=N (`env` >= 0): 0 gives the 0.2.6 rule back, N > 0 asks for that many rows (64..chunk, a multiple of 64).  Unset (`env` < 0):
// automatic - up to 2048 rows, but only where the bigger buffers take VRAM no expert would have used: `spare_bytes` is the card's room
// beyond what its expert tier can use (huge when the tier is off, e.g. --vram-experts 0), `bytes_per_row` the buffers' cost of one row.
// The answers do not change with the slab (a row's arithmetic is its own), but the experts a card holds do: measured 2026-10-26 on the RTX
// 5070 (12 GB, 2.25 bpw, 8K prompt): 1024 rows +2.0% prompt, 2048 rows +2.2%, at 29 / 83 fewer experts in VRAM, a -3% decode and different
// greedy ids (placement); so a card whose room is all experts keeps the 0.2.6 slab, and the automatic size is for cards with VRAM to spare.
// `expert_share` (the share of the routed experts the cards hold, 0 when unknown): the second measured case.  Where the cards hold at least
// half the experts the whole small chunk's slab is worth its VRAM: 4 x R9700 (2.25 bpw, share 0.77, chunk 1024), 8K warm 1204 / 1236 / 1250
// ppts with the 256-row rule against 1324 / 1353 / 1363 with a 1024-row slab (+9%), decode 77.7 against 78.1 tok/s, 268 MB a device (1% of
// the card).  That branch asks for min(chunk, 1024) rows whatever the spare room (the spare of such a box is negative: its room is all
// experts), and never less than the base; the automatic growth on top of it still applies where the room is there.  A unified-memory GPU
// (the caller passes env 0 there) and DS4_ATTN_SLAB=0 never take either branch.
inline int attn_slab_rows(int env, int chunk, int base, double spare_bytes, double bytes_per_row, double expert_share = 0.0) {
    auto clampn = [&](int a) { return std::max(64, std::min(chunk, a) / 64 * 64); };
    if (env == 0) return base;
    if (env > 0) return clampn(env);
    int rows = base;
    if (expert_share >= 0.5) rows = std::max(rows, clampn(1024));
    const int target = clampn(2048);
    if (target > rows) {
        const double extra = (double) (target - base) * bytes_per_row;
        if (spare_bytes >= extra * 1.25 + (256.0 * 1048576.0)) rows = target;
    }
    return rows;
}

// The prompt path's expert staging ring (MB).  The prompt streams the experts the cards do not hold through it.  Three measured cases and
// the rest left as it was:
//   - the cards hold the experts (share >= 0.82): 16 MB.  4 x R9700, share 0.86 (the 2.00 pack): a bigger ring costs decode after a prompt
//     (-1.8% at 16 MB against none, -4.4% at 64, -4.6% at 400); everything fitting needs no ring at all;
//   - they hold most of them (0.5 <= share < 0.82, discrete cards): a small ring from each DEVICE's room, 8 MB a GB of it, 128..512 MB
//     (4 x R9700, 2.25 bpw, share 0.72-0.77, chunk 1024, 8K prompt + six 128-token decodes a load: 2000 MB 1243 ppts / 73.2 tok/s, 1000 MB
//     1287 / 77.5, 500 MB 1314 / 78.5, 256 MB 1324 / 78.4, 128 MB 1338 / 78.3; the room per device there is 31 GB -> 248 MB);
//   - a unified-memory GPU (Strix Halo) that holds under 1/1.3 of them, and every discrete card that holds under half: a share of the
//     summed room, 64 MB a GB, 400..1200 MB.  Measured on Strix Halo only (400 -> 1200: the queue 3,200 -> 8,087 experts, 4K prompt 27.3
//     -> 21.3 s, decode unchanged); a discrete card in this band is not measured and takes the same, VRAM-bounded, rule.
// A unified-memory GPU never takes the small ring (its datapoint is the 1200 MB one): it keeps the 16 MB "fit" rule at 1/1.3.
inline int staging_ring_mb(double share, double room_bytes, int n_dev, bool integrated) {
    const double GB = 1e9;
    const double per_dev = n_dev > 0 ? room_bytes / (double) n_dev : 0.0;
    const double streamed = std::min(1200.0, std::max(400.0, room_bytes / GB * 64.0));
    if (integrated) return share >= 1.0 / 1.3 ? 16 : (int) streamed;
    if (share >= 0.82) return 16;
    if (share >= 0.5) return (int) std::min(512.0, std::max(128.0, per_dev / GB * 8.0));
    return (int) streamed;
}

// DS4_TOPK=N (a test, not for users): over the flag and the rule; returns the count to use
inline int experts_test_override(int chosen, int model_k) {
    if (const char* v = std::getenv("DS4_TOPK")) {
        const int n = std::atoi(v);
        if (n >= 1 && n <= model_k) return n;
    }
    return chosen;
}

}  // namespace strata::ds4
