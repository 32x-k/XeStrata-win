<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# Ornith-1.5 and llm-jp-4.1 against llama.cpp — 2026-10-11

The plain engine (`PlainModel`, branch moe-models) on the development machine, compared with llama.cpp `de7fa0a3`'s
CPU backend on the prompts of [2026-10-10-ornith-parity](../2026-10-10-ornith-parity) and
[2026-10-10-llmjp-parity](../2026-10-10-llmjp-parity): greedy, 32 tokens, the same prompt ids.  `oracle_greedy` and
`compare.py` are that record's; the CPU reference built here gives the same bits as the one built in the AMD container.

## Runs

| Tag | Build and device |
| --- | --- |
| `b70`, `b70q6` | the free mode, the Arc Pro B70 |
| `rtx` | the contrib-llvm mode, the RTX 4070 |
| `small` | the free mode, the B70 as a small card: `STRATA_VRAM_LIMIT_MIB=8192 STRATA_MAX_ALLOC_MIB=4096 STRATA_NO_XMX=1`, `--pool-workers 4` |

Files: Ornith-1.5 Q6_K, Q4_K_M, BF16 and pfeifferj's GSQ-RCO 3-bit; llm-jp-4.1 Q4_K_M.
[run_xs.sh](run_xs.sh) runs one model's prompts; [results.txt](results.txt) has every comparison's first line and its
last compared position (the token, the logit difference, the KL divergence, llama.cpp's top-2 margin).

## Results

- llm-jp-4.1 Q4_K_M: 5 of 5 prompts agree for 32 tokens on the B70 and the RTX 4070, and both prompts tried as a small card.
- Ornith-1.5: every prompt agrees for 32 tokens or parts where llama.cpp's own two candidates are close: Q6_K 3 of 5
  (the others at top-2 margins 0.114 and 0.042), Q4_K_M 3 of 5 (code at 0.183), GSQ-RCO 3-bit 3 of 5 (0.136, 0.060),
  BF16 3 of 5 (code at 0.082).
- The `ja` prompt parts where llama.cpp's own GPU backend parts from its CPU backend: on Q4_K_M at position 26 to the
  same token XeStrata gives, on BF16 at position 24 with the same 32 tokens as XeStrata.  On Q4_K_M switching off
  XeStrata's new CPU kernels, the CPU share of the prompt, the int8 products, the prompt path or the GPU experts
  leaves the divergence where it is.
- The GSQ-RCO 3-bit file has 7 layers whose gate and up differ in type (Q2_K, Q3_K, Q4_K); they run since this day's
  change.  The prompt takes the FP16 path for that file.

Not checked here: the speeds, Ornith Q5_K_M, Q8_0 and GSQ-RCO 3.5-bit, llm-jp-4.1 BF16, other GPUs.
