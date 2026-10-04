<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# Expert routing of llm-jp-4.1-32b-a3b-thinking — 2026-10-04

Whether XeStrata's split (every expert in RAM, a VRAM cache of the most-used ones, the CPU computing the misses)
would pay for [llm-jp-4.1-32b-a3b-thinking](https://huggingface.co/llm-jp/llm-jp-4.1-32b-a3b-thinking) on a 16 GB Arc,
in Q6_K.
The question is the cache's hit rate: this record measures the model's routing and scores caches against it.
Nothing here runs on the GPU, and XeStrata's engine does not run this model.

## The model

`Qwen3MoeForCausalLM`: 32 layers, each with 128 experts of which 8 are routed per token, no shared expert,
full attention only (GQA 40/4, head 128), no MTP layer.
The router was trained with a load-balancing loss (`router_aux_loss_coef` 0.01).

"Q6_K" made with `llama-quantize --imatrix` from the published BF16 GGUF and imatrix
([llm-jp/llm-jp-4.1-32b-a3b-thinking-gguf](https://huggingface.co/llm-jp/llm-jp-4.1-32b-a3b-thinking-gguf)):
26.9 GiB, 7.18 bits per weight.
`ffn_down_exps` cannot take Q6_K (its 960 columns do not divide into 256-wide K-quant blocks) and falls back to Q8_0,
so an expert is 6.34 MiB (gate and up Q6_K, down Q8_0) and the experts are 25.3 GiB of the file.
A token routes 256 experts, 1.59 GiB.

| On the GPU | Size |
| --- | ---: |
| attention, router, norms (Q6_K) | 0.71 GiB |
| output head (Q6_K) | 0.38 GiB |
| K/V, FP16, 32K tokens (64 KiB a token) | 2 GiB |

The token embedding is a lookup and can stay in RAM.
On a 16 GB card, with about 2 GiB for buffers and the driver, about 10 GiB is left for experts: 1,600 slots,
39% of the 4,096.

## The traces

[dump_routing.cpp](dump_routing.cpp), built against llama.cpp at the commit setup pins (CPU backend), reads
`ffn_moe_topk-<layer>` through the eval callback: the prompt's tokens, then each sampled token (top-p 0.95,
temperature 0.6, seed 1, at most 2,048).
llama.cpp cuts the last layer to the output rows before its FFN, so every prompt token is marked as an output.
[prompts.py](prompts.py) renders eight one-turn conversations with the model's own chat template
(`chat_template.jinja` from the model repository; reasoning "medium"):

| Conversation | Prompt | Generated |
| --- | ---: | ---: |
| code-python (chunked CSV aggregation, with tests) | 129 | 2,048 |
| code-review (a C++ snippet's bugs) | 213 | 1,946 |
| en-essay | 117 | 498 |
| ja-business (meeting rules) | 120 | 886 |
| ja-creative (a short story) | 105 | 2,048 |
| ja-explain (quantum computing for a high-school student) | 104 | 1,025 |
| ja-summary | 201 | 388 |
| math | 113 | 1,379 |

10,218 generated tokens in all.
On the i7-14700 (16 threads, DDR4-3200 in two channels) with the callback, the CPU alone generated 8.4 tokens/s.

## Scoring

[simulate.py](simulate.py) scores each conversation's generated tokens; "LOO" ranks experts on the other seven
conversations (leave one out), as the engine's `--expert-profile` would come from other runs.

- **layers**: whole layers' experts on the GPU (llama.cpp's `--n-cpu-moe`)
- **profile**: the slots hold the experts the other conversations routed most (static)
- **adaptive**: profile, then the engine's adaptive tier (`adapt` in `src/program/generate.cpp`: every 4 rounds,
  per layer, a missing expert routed at least twice, decayed, replaces the least-routed resident one when it leads by
  1.5; at most 96 swaps a round; counts × 0.7).
  A round is taken as one token (this model has no MTP), and a swap as landing before the next token.
- **lru**: per-layer LRU, every miss copied in (for reference)
- **oracle**: the experts this conversation itself routed most (static; an upper bound for a static cache)

## Results

How concentrated the routing is: the most-used 25% of the experts take 32.1% of the routing, the top half 58.2%.

| Slots | GiB | layers | profile | adaptive | swap MiB/token | lru | lru MiB/token | oracle |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 3.2 | 0.125 | 0.133 | 0.255 | 108.3 | 0.278 | 1171.0 | 0.249 |
| 1,024 | 6.3 | 0.250 | 0.267 | 0.413 | 134.8 | 0.454 | 885.7 | 0.415 |
| **1,600** | **9.9** | **0.375** | **0.412** | **0.560** | **119.1** | 0.594 | 658.2 | 0.571 |
| 2,048 | 12.7 | 0.500 | 0.523 | 0.661 | 93.8 | 0.687 | 507.9 | 0.678 |
| 2,560 | 15.8 | 0.625 | 0.648 | 0.765 | 64.5 | 0.781 | 355.8 | 0.786 |
| 3,072 | 19.0 | 0.750 | 0.769 | 0.859 | 37.9 | 0.864 | 219.9 | 0.878 |

At 1,600 slots with fewer swaps a round ([swaps.py](swaps.py)):

| Swaps a round | Hit | Swap MiB/token | Per conversation |
| ---: | ---: | ---: | --- |
| 8 | 0.510 | 12.7 | 0.46–0.59 |
| 24 | 0.537 | 38.0 | 0.50–0.61 |
| 48 | 0.551 | 74.9 | 0.51–0.62 |
| 96 | 0.560 | 119.1 | 0.51–0.63 |

## What it means

- The routing is nearly flat across the whole run: a profile from other conversations does hardly better than whole
  layers (0.41 against 0.375 at 1,600 slots).
  Qwen3.8-Flash-Next's held-out profile reaches 0.645 with 17% of its experts in VRAM
  ([include/strata/core/expert_cache.hpp](../../../include/strata/core/expert_cache.hpp)); this model's 39% gives 0.41.
- What concentrates is the routing within one conversation: the adaptive tier reaches the conversation's own static
  optimum (0.56 against the oracle's 0.57).
  The engine's 96 swaps a round cost 119 MiB of copies a token with 6.3 MiB experts; 8 swaps keep most of the gain
  (0.51) for 12.7 MiB.
- What the CPU reads per token at 1,600 slots: whole layers 1,014 MiB; the adaptive tier with 8 swaps 795 MiB plus
  12.7 MiB of swap reads, about 20% less.
  When the CPU sets the pace, that is about 1.25× the speed of llama.cpp's layer split; scaling the CPU-only
  8.4 tokens/s (2,739 MiB read a token: the experts and 1.1 GiB of attention and output head) by the bytes the CPU
  reads gives roughly 23 tokens/s for the layer split and 28 for the adaptive tier.
  These speeds are an estimate from bytes, not a measurement: the GPU's share, the per-layer synchronisation and a real
  16 GB card are `unverified`.

## The local copy of the BF16 file

Quantizing first failed in `blk.23.ffn_down_exps.weight` (an inf in Q8_0): one element read 0x7ba0 (1.66e36) where the
published file and the safetensors both hold 0x3ba0 (0.00488), one flipped bit.
The file's SHA-256 had matched the published one when it was computed right after the download; after a reboot the
element read flipped, and a write of the right bytes read back flipped again several times within seconds before it
held.
No process had the file open and the kernel logged no NVMe or ext4 error; the cause is not known.
Read with O_DIRECT, the whole file then matched the published SHA-256, and the quantization ran.
