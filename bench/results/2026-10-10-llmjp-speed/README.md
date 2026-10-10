<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# llm-jp-4.1 decode and prompt speed against llama.cpp — 2026-10-10

llm-jp-4.1-32b-a3b-thinking Q4_K_M, measured as Ornith-1.5 was in
[2026-10-10-ornith-speed](../2026-10-10-ornith-speed/README.md): the same machine (RX 9060 XT 16 GB, Ryzen 7 5700X),
the same prompt text (554 tokens with llm-jp's tokenizer and BOS), 256 tokens generated greedily, its `run.sh` with
`MODEL` this GGUF and the profile learned from the en-essay prompt of
[2026-10-10-llmjp-parity](../2026-10-10-llmjp-parity/README.md).  [results.txt](results.txt) holds the lines printed.

| Engine and placement | Prompt (tok/s) | Decode (tok/s) |
| --- | ---: | ---: |
| llama.cpp `-ngl 99 -ncmoe 99` (experts on the CPU) | 318 | 22.1 |
| llama.cpp `-ngl 99 -ncmoe 99 --moe-cache-mib 12000` | 288 | 33.1 |
| llama.cpp `-ngl 99 -ncmoe 26` | 373 | 25.6 |
| llama.cpp `-ngl 99 -ncmoe 22` | 421 | 28.5 |
| llama.cpp `-ngl 99 -ncmoe 18` | 489 | 31.9 |
| llama.cpp `-ngl 99 -ncmoe 14` | 579 | 36.0 |
| llama.cpp `-fit on` | **786** | 42.1 |
| XeStrata, experts in pinned RAM, the VRAM tier starting empty | 48 | 49.0 |
| XeStrata, the VRAM tier filled from a profile (`91ac1a5f`) | 124 | **54.8** |

- Decoding, XeStrata is 1.30x llama.cpp's fastest placement (`-fit on` here: llm-jp's experts are smaller than
  Ornith-1.5's, so more of its layers fit on the card).  The profile raises the VRAM hits from 47.9% to 84.1%.
- The split measured an expert at 0.149 ms on the CPU and 0.189 ms over PCIe: it sends none over PCIe.  Before
  `91ac1a5f` a run with a profile could not time it (too few experts left outside the VRAM); timed, the speed is
  the same.
- The prompt is 0.16x llama.cpp's, for the reason Ornith-1.5's is: no batched prompt path for this model yet.
