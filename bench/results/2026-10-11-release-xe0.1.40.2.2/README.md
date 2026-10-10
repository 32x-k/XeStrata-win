# The release table of xe0.1.40.2.2 against upstream v0.1.42 — 2026-10-11

XeStrata xe0.1.40.2.2 against upstream Strata's newest release, v0.1.42, on the AMD GPU only, by the release's fixed method (`.claude/skills/release/scripts/bench.sh`): IQ3_XXS, the six tiers of `bench/results/2026-10-04-speed-matrix` (1K to 262K) with what setup writes for each context, 256 greedy tokens, each run a new process.
Each tier ran twice per engine, the engines alternating, after a discarded warm-up per engine; the tables are the medians.
Beside it, the smaller serve and everyday-use tables by the upstream merge's scripts (`tools/bench_core.py`): `strata --serve` at 32K, sampled at temperature 0.7, and each engine's `serve/server.py` with the Japanese chat and the coding agent, two rounds each, each engine writing its own text.

The B70 and the RTX 4070 were not measured for this release: their runs were stopped part way (other work on the development machine overlapped them) and left out.

## The machine

| GPU | Machine |
| --- | --- |
| AMD Radeon RX 9060 XT (16 GB) | a Ryzen 7 5700X (16 threads), 62 GiB of RAM, in a Fedora 44 container |

## The engines

| | XeStrata | upstream |
| --- | --- | --- |
| Version | xe0.1.40.2.2 | v0.1.42 (`61b3fb5d`) |
| Build | the contrib-llvm mode with HIP (`STRATA_HIP_ARCHS=gfx1200`; no CUDA in the container) of `3e0a662c` | its HIP build (`-DSTRATA_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1200 -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PREFILL_MMQ=ON`) |

The tag was moved from `3e0a662c` to `e32b8b05` before it was pushed; the commits between change the README, the documents, AGENTS.md and the skills only, so the engine is the same.
Both engines built ggml from llama.cpp `3cf03257`, the commit both pin.
The logs have the container's paths written as `<repo>` (XeStrata's tree), `<upstream>` (upstream's) and `<data>` (the data folder), and leave out the prompt's token list.

## The tables

### RX9060XT: Prefill (tokens/s)

| Engine | 1K | 4K | 32K | 64K | 128K | 262K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| xe0.1.40.2.2 | 582 | 774 | 842 | 845 | 830 | 798 |
| v0.1.42 | 279 | 378 | 385 | 374 | 362 | 339 |

### RX9060XT: Decode (tokens/s)

| Engine | 1K | 4K | 32K | 64K | 128K | 262K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| xe0.1.40.2.2 | 49.1 | 47.3 | 51.4 | 50.2 | 39.9 | 33.5 |
| v0.1.42 | 48.0 | 44.7 | 45.9 | 38.9 | 31.9 | 36.3 |

## serve and everyday use

### RX9060XT serve t07: Prefill (tokens/s)

| Engine | 32K |
| --- | ---: |
| xe0.1.40.2.2 | 836 |
| v0.1.42 | 381 |

### RX9060XT serve t07: Follow-ups (ms)

| Engine | 32K |
| --- | ---: |
| xe0.1.40.2.2 | 528 |
| v0.1.42 | 436 |

### RX9060XT serve t07: Decode (tokens/s)

| Engine | 32K |
| --- | ---: |
| xe0.1.40.2.2 | 32.9 |
| v0.1.42 | 40.4 |

### RX9060XT everyday use

| Engine | chat-ja first token (s) | chat-ja prompt (tok/s) | chat-ja output (tok/s) | agent first token (s) | agent prompt (tok/s) | agent output (tok/s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| xe0.1.40.2.2 | 0.92 | 148 | 32.2 | 5.80 | 673 | 33.6 |
| v0.1.42 | 1.49 | 107 | 29.3 | 11.67 | 338 | 27.9 |

## What the numbers say

- generate: the prefill is 2.0 to 2.4 times as fast as upstream's at every length; the decode is 2% to 29% faster up to 128K and 8% slower at 262K (33.9 and 33.2 tok/s against 36.4 and 36.3, both runs).
- serve at 32K: the first prefill is 2.2 times as fast, but the wait for a follow-up is longer (528 against 436 ms) and the decode slower (32.9 against 40.4 tok/s), beyond the ±5% that counts as equal.
- everyday use: the first token comes 38% (chat-ja) and 50% (agent) sooner, the agent's prefill is 1.99 times as fast, the decode 10% (chat-ja) and 20% (agent) faster.
- No cell stopped upstream at 10 tok/s or less, and every cell fit both engines.
- The 128K decode varied between the runs for both engines (XeStrata 43.4 and 36.4, upstream 37.8 and 26.0 tok/s). The causes were not looked into.
