# The release table of xe0.1.40.2.1 against upstream v0.1.41 — 2026-10-10

XeStrata xe0.1.40.2.1 against upstream Strata's newest release, v0.1.41, on an Intel, an NVIDIA and an AMD GPU, by the release's fixed method (`.claude/skills/release/scripts/bench.sh`): IQ3_XXS, the six tiers of [the speed-matrix record](../2026-10-04-speed-matrix/README.md) (1K to 262K, what setup writes for each context), 256 greedy tokens, each engine's own shipped expert profile.
Each tier ran twice per engine, the engines alternating and a new process each run, after a discarded warm-up per engine; the tables are the medians.
Each engine writes its own text (no `--spec-follow`), so the output speed carries the difference in how each text drafts.

## The machines

| GPU | Machine |
| --- | --- |
| Intel Arc Pro B70 (32 GB, PCIe 5.0 x16) | the development machine: Core i7-14700 (28 threads, no AVX-512), DDR4-3200 in two channels, 123 GiB of RAM, Ubuntu 26.04 |
| NVIDIA GeForce RTX 4070 (12 GB, PCIe 4.0 x4, 6.4 GB/s measured) | the same machine |
| AMD Radeon RX 9060 XT (16 GB, 28.3 GB/s measured) | a Ryzen 7 5700X (16 threads), 62 GiB of RAM, in a Fedora 44 container |

## The engines

| | XeStrata | upstream |
| --- | --- | --- |
| Version | xe0.1.40.2.1 (`94287ef0`) | v0.1.41 (`fb58e0db`) |
| B70 | contrib-llvm build of the tag (intel/llvm 7.1.1 with its CUDA target, `-fsycl-targets=spir64,nvidia_gpu_sm_89`), `ONEAPI_DEVICE_SELECTOR=level_zero:*` | its `sycl/` port, built and run in its `strata-sycl-dev` Docker image (oneAPI 2026.1), ahead-of-time for the B70 (`AOT=bmg-g31`), as its docs/INTEL.md says |
| RTX 4070 | the same build, `ONEAPI_DEVICE_SELECTOR=cuda:*` | its CUDA build (`-DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89`, CUDA 13.1) |
| RX 9060 XT | the contrib-llvm build of the tag with HIP (`STRATA_HIP_ARCHS=gfx1200`) | its HIP build |

Both engines built ggml from llama.cpp `3cf03257`, the commit both pin.
The logs in `runs/` have this machine's paths written as `<repo>` (XeStrata's tree), `<upstream>` (upstream's) and `<data>` (the data folder), and leave out the prompt's token list (`prompts.py` of the speed-matrix record writes the prompts again).

## The tables

### B70: Prompt (tokens/s)

| Engine | 1K | 4K | 32K | 64K | 128K | 262K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| xe0.1.40.2.1 | 844 | 1,501 | 2,310 | 1,967 | 1,975 | 1,903 |
| v0.1.41 | 547 | 732 | 735 | 1,129 | 1,018 | 884 |

### B70: Output (tokens/s)

| Engine | 1K | 4K | 32K | 64K | 128K | 262K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| xe0.1.40.2.1 | 82.8 | 82.4 | 79.5 | 74.3 | 75.8 | 72.1 |
| v0.1.41 | 34.4 | 40.6 | 28.7 | 36.4 | 30.3 | 34.3 |

### RTX4070: Prompt (tokens/s)

| Engine | 1K | 4K | 32K | 64K | 128K | 262K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| xe0.1.40.2.1 | 297 | 533 | 1,142 | 1,068 | 1,042 | 997 |
| v0.1.41 | 258 | 526 | 1,143 | 1,070 | 1,043 | 997 |

### RTX4070: Output (tokens/s)

| Engine | 1K | 4K | 32K | 64K | 128K | 262K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| xe0.1.40.2.1 | 53.9 | 58.0 | 58.5 | 53.4 | 47.1 | 50.1 |
| v0.1.41 | 46.7 | 46.4 | 50.9 | 49.5 | 41.2 | 43.7 |

### RX9060XT: Prompt (tokens/s)

| Engine | 1K | 4K | 32K | 64K | 128K | 262K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| xe0.1.40.2.1 | 509 | 768 | 847 | 829 | 830 | 803 |
| v0.1.41 | 295 | 372 | 382 | 370 | 356 | 337 |

### RX9060XT: Output (tokens/s)

| Engine | 1K | 4K | 32K | 64K | 128K | 262K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| xe0.1.40.2.1 | 48.8 | 48.9 | 50.3 | 46.2 | 50.5 | 44.1 |
| v0.1.41 | 46.8 | 42.8 | 45.7 | 38.2 | 43.3 | 38.3 |

## What the numbers say

- B70: XeStrata reads the prompt 1.5 to 3.1 times as fast as upstream's `sycl/` port and writes the output 2.0 to 2.8 times as fast, at every length.
- RTX 4070: the prompt is the same from 32K on (within 0.2%), 15% and 1% faster at 1K and 4K; the output is 8% to 25% faster.
- RX 9060 XT: the prompt is 1.7 to 2.4 times as fast, the output 4% to 21% faster.
- No cell stopped upstream at 10 tok/s or less, and every cell fit both engines.
- XeStrata's output varied more than upstream's between the two runs in places (the B70 at 64K: 79.6 and 69.1 tok/s); the RX 9060 XT's first 1K prompt read at 586 and the second at 432 tok/s. The causes were not looked into.
