# Community benchmark on 4x Intel Arc Pro B70 (SYCL, layer split)

English | [日本語](README.ja.md)

Measured on 2026-10-12 by JP-devv. **Official upstream source**, commit `8607e667` (xe0.1.43.1), clean
tree, **no local engine patches** — the only difference from the release is the run configuration
documented below. Unsloth **UD-Q4_K_XL**, 131,072-token context, layer split across all four cards,
MTP draft layer. Headline, medians of three runs (prompt 4,154 / 32,702 / 127,531 tokens):
**PP 1,111 / 1,869 / 2,233 tok/s** and **TG 78.5 / 72.0 / 72.5 tok/s**.

`docs/MULTIGPU.md` measures this same file at 20.8 tok/s TG on one B70 and 38.3–39.0 on a B70 + RTX
4070; this is ~3.5x one card. Main limitations: one model on one box; no quality check beyond draft
acceptance and the CTest suite; one prompt shape per size (see Limitations).

## Hardware and software

- GPU: **4x Intel Arc Pro B70** (Xe2), 31.9 GiB usable each, all four used, 275 W cap each (hwmon
  `power1_cap`). Four cards behind PCIe switch branches of one root complex.
  Reading the link at the card or its upstream port gives `x1 2.5GT/s`, which is an internal port, not
  the data path: the engine's own auto-split probe measures **26.7 GB/s** for a 64 KiB host→device
  copy, and [p2p-bw.cpp](p2p-bw.cpp) measures **12.6 GB/s** end to end through host memory (two legs,
  so ~25 GB/s a leg). [system.json](system.json), [environment.json](environment.json).
- CPU: Intel Core i7-13700K (24 threads, AVX2, **no AVX-512**, so the AVX-2 expert kernels ran).
  RAM: 64 GB DDR5 (61 GiB visible). Models, pack and MTP head on one NVMe partition.
- OS: Ubuntu 26.04.1, kernel 7.2.10, `xe` driver, Level Zero `libze-intel-gpu1` 26.18.38308.1.
- Compiler: **intel/llvm v7.1.1** (`504366f4`) built from source by `tools/intel_llvm_build.py`,
  libsycl 9. oneMKL 2026.1. oneAPI 2026.1.1 is installed but unused (that would be the contrib-icpx
  mode, which this record does not cover).
- Background workloads: an unrelated service that also uses these four GPUs was stopped for all runs;
  the engine log shows only this benchmark's requests, one at a time.

## Build

Built and tested in **both** `free` and `contrib-llvm` with `STRATA_BUILD_TESTS=ON`: **92/92 CTest
cases pass in each**. The measured binary is the contrib-llvm one (oneMKL for the prompt path's dense
matrix products, XeStrata's own XMX kernels elsewhere); the free binary was built and passed its tests
but was not measured. `contrib-icpx` was not built. [build.json](build.json) carries the compiler
record; [BUILD.md](BUILD.md) has the one build-environment note worth passing on.

## Model and configuration

- Model: [`unsloth/Qwen3.8-Flash-Next-GGUF`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF),
  folder `UD-Q4_K_XL`, at revision `38bb39ee97821de2c9009abb7e93950eec396e66`. Four shards, all
  SHA-256-verified against the table `setup.py` pins (111,334,654,784 bytes total; per-file sizes and
  hashes in [configurations.json](configurations.json)'s sibling record —
  `10,946,624`, `49,859,583,136`, `49,376,141,504`, `12,087,983,520` bytes).
- Pack: `iq_pack.py --compat-bf16` (its Q8_0 hyper-connection projections become the BF16 the engine
  reads); the experts stay in the GGUF and are read in place, so a RAM copy of the whole arena is not
  taken. The n-gram table is read from the SSD (`--ple-io direct`; the engine's probe reported
  139,373 rows/s).
- MTP: built from the BF16 checkpoint by `tools/mtp_fetch.py` (31 `mtp.*` tensors at
  `Qwen/Qwen3.8-Flash-Next@de4b8e4d43b917e7706784d8bb445c9af86a3540`, each verified against its pinned
  SHA-256; `verify` exits 0), packed to q2_0 and built into the runtime head.
- Configuration: [configurations.json](configurations.json). On top of what `setup.sh` writes:
  **`--layer-split 5,24,43`** (pinned; see below), `--pipeline-windows 2`,
  `--expert-cache-per-layer`, `--pcie-frac 0`, `--adapt-every 0`, `--spec 3`, `--resident-budget-gib 8`
  (down from setup's 35), `--kv-resident` **removed** (0 = the whole KV in VRAM; at 131072 int8 it is
  ~1.8 GB over four cards), and in the engine's environment `STRATA_SPLIT_CPU_GBPS=5` and
  `STRATA_SPLIT_MTP_BATCH=0`.
- Resolved placement: layers 0–4 / 5–23 / 24–42 / 43–47; **21,952 resident experts** in the token
  graph, per-stage caches 2,560 of 2,560, 9,565 of 9,728, 9,529 of 9,728 and 2,560 of 2,560 profiled
  pairs ([engine-startup.txt](engine-startup.txt)). Decode expert-cache hit rate 93–94% during the
  runs (the residue is the MTP draft layer's own experts and the prompt path's streaming, which the
  per-layer cache does not cover).

## Results

House method: one discarded warm-up, then three runs per size, temperature 0, 256-token cap, one
request at a time. **PP** is prompt (prefill) tok/s, **TG** is decode tok/s. Every measured run is a
full prefill — the harness varies the prompt's leading tokens so the conversation cache cannot serve
it — which is the cold case; a continuation in real use reads only the new tokens.

| Prompt tokens | PP tok/s | TG tok/s | drafts accepted |
| ---: | ---: | ---: | ---: |
| 4,154 | 1,111.4 (1,109.1–1,111.9) | 78.5 (76.3–79.0) | 132–138 of 177–183 |
| 32,702 | 1,868.6 (1,868.3–1,868.7) | 72.0 (71.9–84.9) | 128–143 of 176–177 |
| 127,531 | 2,233.0 (2,232.3–2,233.4) | 72.5 (71.7–73.1) | 129–137 of 169–191 |

Per-run rows: [results.csv](results.csv) and [matrix.jsonl](matrix.jsonl), both parsed from the
engine's own request lines ([engine-requests.txt](engine-requests.txt)); medians and ranges:
[summary.json](summary.json).

**Recall check.** A unique 5-digit code buried at 10%, 50% and 90% of a filler prompt, then asked
back (temperature 0, thinking off, one fresh-context request per cell, so each cell is a full cold
prefill): **9 of 9 recalled** — at 4K (3,630-token prompts), 32K (28,786) and 128K (115,016). Cells
and replies in [needles.json](needles.json), the check itself in [needles.py](needles.py). Recall
says nothing about answer quality in general.

## What the numbers show

**TG is flat to 128K** (72–79, no meaningful context penalty): the KV cache is in VRAM and only 12 of
the 48 layers are attention. **PP rises with prompt length** (1,111 → 2,233 tok/s), because the batched
prompt path amortises per-request startup and chunk overhead on longer prompts — so short prompts are
this engine's worst PP case, not long ones. PP is very stable (ranges above span <0.3%); TG carries
±8% run to run, which is why the table gives ranges.

**The auto split is not stable across restarts on this machine.** With identical settings,
`--layer-split auto` plus `STRATA_SPLIT_CPU_GBPS=5` chose **5 / 19 / 19 / 5** (21,952 pairs resident)
on one start and **4 / 23 / 30** (**24,303** pairs, ~99.7% of routed mass) on the next. The candidate
placements are close enough that the planner's own measured rates decide between them run to run, so
this record pins the placement where the numbers were taken and states it. For reference, setup's
default CPU rate (88 GB/s measured here) picks 3 / 30 / 3 / 12 — 18,240 pairs, with two cards holding
~40 GB of unusable VRAM.

**Lowering `STRATA_SPLIT_CPU_GBPS` improves the pick, and a fully resident split is not the fastest
one.** Five runs at 1K each:

| 1K, median of 5 | auto + `CPU_GBPS=5` | forced `--layer-split 12,24,36` |
| --- | ---: | ---: |
| PP tok/s | **475.1** (475.0–479.1) | 448.2 (437.9–449.9) |
| TG tok/s | **78.0** (69.6–80.8) | 72.5 (63.9–78.6) |
| pairs resident | 21,952 (5/19/19/5) | 24,576 (12/12/12/12) |

The PP ranges do not overlap; the TG difference is inside the noise. `CPU_GBPS=1` is worse again
(10 / 3 / 17, 20,618 pairs, TG 77.2 / 73.0), and 3 stages (`--layer-split 16,32`, one card idle) are
clearly worse (TG 67.2 / 69.4). The likely reason the less-resident split wins: the first stage
carries the embedding gather, the hyper-connection broadcast and the PLE block, the last carries the
output head and the MTP drafter, so those stages cost more per layer than the middle ones — balancing
per-stage *time* beats balancing layer counts. Per-card work share during a 400-token generation
agrees ([power-balance.jsonl](power-balance.jsonl)): card0 19%, card1 29%, card2 26%, card3 26% of the
499 W total, so nvtop's differing percentages are expected and are not a fault.

**P2P works on this rig and the engine does not use it.** `zeDeviceCanAccessPeer` answers YES for all
12 ordered pairs, and measured 256 MiB copies give **25.6–28.7 GB/s** device→device against **12.6–12.7
GB/s** through host memory, the path the layer hand-off takes today ([p2p.json](p2p.json),
[p2p-bw.cpp](p2p-bw.cpp), [l0-peer.py](l0-peer.py)). It does not help this configuration — the
hand-off is ~50 KB a token — but it is the fabric a helper tier or expert-parallel placement would use
for expert blobs. The engine has no peer path at all, so that design is not measured here.

**setup's per-model settings are sized from one card.** `--kv-resident 32768` streams a KV cache to
host RAM that fits in VRAM (removing it was free); `--resident-budget-gib 35` reserves RAM for experts
already resident (8 GB is kept as insurance for the residual misses); `--expert-cache-per-layer` is off
by default although its own help text records why the default hurts; and the start prints
`WARNING: RAM is tight - the model's experts take 71.7 GB of this PC's 61 GB`, which reads as a fault
on a rig whose GPUs hold the model and copy only ~8 GB into RAM.

## Files

| file | what it is |
| --- | --- |
| [system.json](system.json), [environment.json](environment.json), [build.json](build.json) | hardware, software versions, engine provenance |
| [configurations.json](configurations.json) | the exact engine args and environment |
| [results.csv](results.csv), [matrix.jsonl](matrix.jsonl), [summary.json](summary.json) | per-run results, medians and ranges |
| [engine-startup.txt](engine-startup.txt), [engine-requests.txt](engine-requests.txt) | the engine's own startup diagnostics (split, residency, cache, probes) and per-request lines |
| [bench.py](bench.py) | the harness that produced the results (it varies the prompt's leading tokens so no run is served from the conversation cache) |
| [p2p-bw.cpp](p2p-bw.cpp), [l0-peer.py](l0-peer.py), [p2p.json](p2p.json) | the peer-access measurement |
| [gpu-balance.py](gpu-balance.py), [power-balance.jsonl](power-balance.jsonl) | per-card work share during one generation |
| [needles.py](needles.py), [needles.json](needles.json) | the recall check (a code buried at three depths, asked back) |
| [BUILD.md](BUILD.md) | how the toolchain and engine were built, including one package worth knowing about |

## Limitations

- One model (UD-Q4_K_XL) on one box; one repetitive English prompt shape per size. TG moves a few
  percent with the text and with draft acceptance, so neighbouring cells are not meaningful.
- No quality check beyond the project's CTest suite, the draft-acceptance counts and the recall
  check above: no perplexity, no task scores.
- Not measured: contexts beyond 127,531 tokens (the 262,144 window was not exercised); temperature
  0.7 / serve-path / everyday-use numbers; the free-mode binary's speed (built and tested only);
  `contrib-icpx`; images (off); the AVX-512 expert paths (this CPU has none, and Intel SDE was not
  run); the Q2_0 AVX-512 pack; 2-stage splits; `STRATA_SPLIT_ROWS` / `STRATA_SPLIT_MASS_EXP`.
- The P2P figures are 256 MiB copies; the small-copy latency behaviour (what a hand-off or a single
  expert blob is) was not measured.
