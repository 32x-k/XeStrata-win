<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# Ornith-1.5 decode and prompt speed against llama.cpp — 2026-10-10

Ornith-1.5-35B-A3B Q6_K on the AMD container (Radeon RX 9060 XT 16 GB, Ryzen 7 5700X, 62 GiB RAM): a 515-token
prompt (a wikitext-2 passage to summarize, in Ornith's chat format, written by
[make_prompt.py](make_prompt.py)), 256 tokens generated
greedily.  llama.cpp `de7fa0a3` with ROCm (`llama-completion`, 8 threads) in the placements it offers for experts
that do not fit; XeStrata's free build, branch moe-models.  [run.sh](run.sh) runs it all; [results.txt](results.txt)
holds the lines it printed.

| Engine and placement | Prompt (tok/s) | Decode (tok/s) |
| --- | ---: | ---: |
| llama.cpp `-ngl 99 -ncmoe 99` (experts on the CPU) | 318 | 25.2 |
| llama.cpp `-ngl 99 -ncmoe 99 --moe-cache-mib 12000` | 325 | **38.4** |
| llama.cpp `-ngl 99 -ncmoe 26` | 426 | 27.3 |
| llama.cpp `-ngl 99 -ncmoe 22` | 481 | 32.2 |
| llama.cpp `-fit on` | **535** | 31.9 |
| XeStrata, the GGUF read in place, CPU only (`25b3db46`) | 22 | 27.4 |
| XeStrata, experts in pinned RAM, CPU / PCIe split (`061a04a6`) | 66 | 44.3 |
| XeStrata, the same with a profile from another text | 115 | **51.1** |

`-ncmoe 18` and `-ncmoe 14` did not fit in the card's VRAM.  XeStrata's rows are the mean of two runs (they agreed
within 1%).

- Decoding, XeStrata is 1.33x llama.cpp's fastest placement (`--moe-cache-mib`, the GPU expert cache).  The profile
  was learned from the story prompt of [2026-10-10-ornith-parity](../2026-10-10-ornith-parity/README.md), 512
  tokens generated (`--expert-profile-save`), and fills the VRAM tier at load: 86.0% of the routed experts hit
  against 52.7% starting empty.
- The split measured an expert at 0.064 ms on the CPU pool, 0.091 ms over PCIe and 0.010 ms on the GPU.
- The prompt is still 0.21x llama.cpp's: XeStrata reads it in windows of 8 tokens through the decode's kernels;
  the batched prompt path is not written for this model yet.
- The CPU pool ran on 7 workers; nothing else ran on the container during the runs.
