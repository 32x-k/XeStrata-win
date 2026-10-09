<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# Ornith-1.5 on XeStrata against llama.cpp — 2026-10-10

The first runs of Ornith-1.5-35B-A3B (`qwen35moe`, the Q6_K GGUF) on XeStrata's plain engine (`PlainModel`,
branch moe-models), compared with llama.cpp `de7fa0a3` on five short chat prompts, greedy, 32 tokens each.

Machine: the AMD container (Ryzen 7 5700X, 62 GiB RAM, Radeon RX 9060 XT 16 GB, gfx1200), XeStrata's free build
(intel/llvm, HIP).  The experts went through the engine's tiers: the VRAM slots filled and swapped by the adaptive
rule, the GGUF read in place (FileExpertSource), the CPU pool for the rest, in one MoeWindow per token.

## Files

| File | What |
| --- | --- |
| [oracle_greedy.cpp](oracle_greedy.cpp) | llama.cpp's greedy decode: the prompt's ids, the generated ids, the logits each was chosen from |
| [compare.py](compare.py) | leading tokens that agree, and per position the logit difference, the KL divergence and the top-2 margin |
| [run_oracle.sh](run_oracle.sh), [run_xs.sh](run_xs.sh) | the runs (`MODEL`, `STRATA`, `PROMPTS`, `STRATA_VRAM_LIMIT_MIB`) |
| [prompts/](prompts) | the five prompts, in Ornith's chat format |
| [results.txt](results.txt) | every comparison and every run's report line |

`oracle_greedy` was built twice: on llama.cpp's CPU backend (the reference) and on its ROCm backend with every
layer on the GPU and the routed experts in host memory (`cpu_moe=1`, llama.cpp's `--cpu-moe`), the placement
XeStrata's own decode has.  The second measures how far llama.cpp's own GPU arithmetic moves from its CPU's: the
noise floor of the comparison.

## Results

Leading tokens of 32 that agree with llama.cpp on the CPU:

| Prompt | XeStrata | llama.cpp ROCm (noise floor) |
| --- | ---: | ---: |
| en | 32 | 32 |
| ja | 24 | 32 |
| code | 32 | 21 |
| math | 32 | 32 |
| story | 32 | 19 |

- Four of five prompts agree on all 32 tokens; llama.cpp's own GPU path agrees on three.
- ja: at position 24 three tokens lie within about one logit of each other (llama.cpp CPU: 1919 at 21.08, 9764 at
  20.71, 760 at 20.03).  XeStrata picks 760 (20.59 against 1919's 20.26); llama.cpp's GPU path moved 760 by 0.86
  from its CPU value, XeStrata by 0.55.  A near tie decided by rounding, not a different computation.
- The logits differ from the CPU's by up to 0.1-0.25 of the row's largest |logit| (in the tail of the vocabulary),
  with a KL divergence of the distribution up to 0.12 before the first difference; llama.cpp's GPU path: up to 0.18
  and 0.12.

With the engine's VRAM capped (`STRATA_VRAM_LIMIT_MIB`), every prompt gave the same 32 tokens as uncapped:

| Cap | VRAM slots (a layer) | Swaps (en) | Prompts run |
| --- | ---: | ---: | --- |
| none (16 GB) | 5,440 (136) | 768 | all five |
| 16,384 MiB | 5,440 (136) | 768 | all five |
| 8,192 MiB | 2,120 (53) | 768 | all five |
| 4,096 MiB, a round every window | 480 (12) | 839 | en, ja, code |

At 4 GiB the slots fill and the adaptive rule evicts: more swaps than slots.

Speed is not measured here: 32 tokens with a cold tier, before the speed work (MTP, the prompt path, a profile
for the tier).  The runs decoded 11-13 tokens/s (one token a window) and read the prompt at 15-16 tokens/s (windows
of up to 8 tokens).
