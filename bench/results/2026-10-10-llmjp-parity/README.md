<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# llm-jp-4.1 on XeStrata against llama.cpp — 2026-10-10

The first runs of llm-jp-4.1-32b-a3b-thinking (`qwen3moe`, the official Q4_K_M GGUF) on XeStrata's plain engine
(`PlainModel`, branch moe-models), compared with llama.cpp `de7fa0a3` on its CPU backend: five prompts, greedy, 32
tokens each.  Machine, build and tools as in [2026-10-10-ornith-parity](../2026-10-10-ornith-parity/README.md)
(`oracle_greedy`, `compare.py`); the experts in pinned RAM, the CPU / PCIe split measured at load (it kept every
missed expert on the CPU here), the VRAM tier starting empty.

[prompts/](prompts) are five of the conversations of [2026-10-09-moe-routing](../2026-10-09-moe-routing/README.md)
rendered with the model's chat template (llm-jp-harmony-v1) by its `prompts.py`; llama.cpp adds the BOS the GGUF
asks for.  [results.txt](results.txt) holds every comparison.

| Prompt | Prompt tokens | Leading tokens that agree | Worst KL before a difference |
| --- | ---: | ---: | ---: |
| ja-explain | 105 | 32 of 32 | 9.5e-03 |
| en-essay | 118 | 32 of 32 | 6.6e-04 |
| math | 114 | 32 of 32 | 9.8e-03 |
| code-python | 130 | 32 of 32 | 5.9e-02 |
| ja-summary | 202 | 32 of 32 | 2.1e-02 |

Every prompt agrees on all 32 tokens.  The KL divergence stays below Ornith-1.5's (up to 0.12, the same size as
llama.cpp's own GPU path against its CPU), and the largest logit difference lies in the tail of the vocabulary as
there.  YaRN (`--rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 65536`, the 256K context) is not exercised by
prompts this short.
