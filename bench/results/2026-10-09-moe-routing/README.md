<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# Expert routing of other MoE models — 2026-10-09

The tools of [2026-10-04-llmjp-routing](../2026-10-04-llmjp-routing/README.md), changed so that they also trace
models whose layers do not all route (GLM-5.3-Flash), and llm-jp-4.1's traces taken again with its BOS.
Nothing here runs on the GPU, and XeStrata's engine does not run these models.

## What changed in the tools

| File | Change |
| --- | --- |
| [dump_routing.cpp](dump_routing.cpp) | The prompt is tokenized with the model's special tokens (`add_special`), so llm-jp-4.1 gets the BOS its GGUF asks for; the 10-04 traces ran without it. Only the layers that route are written: the first decode decides which (a dense layer, or an MTP layer the plain decode does not run, has no `ffn_moe_topk`), the header's `n_layer` counts them, and `OUT.layers` lists their model layer numbers. Every later decode must route the same layers. |
| [prompts.py](prompts.py) | `NAME=VALUE` arguments go to the chat template as variables (GLM-5.3-Flash: `reasoning_effort=high`), and Jinja's loop controls are on (GLM's template uses `{% break %}`). |
| [simulate.py](simulate.py) | Unchanged; `--n-expert` and `--expert-mib` already set the shape. |
| [swaps.py](swaps.py) | The expert count and size are arguments (`N_EXPERT`, `EXPERT_MIB`); 10-04 fixed both to llm-jp-4.1's Q6_K. |

`dump_routing` is built against llama.cpp `de7fa0a3` (2026-10-08), CPU backend only, which has GLM-5.3-Flash
(`glm5-next`, #27773) and the LLM-jp-4.1 output parser (#29681); 10-04 used the commit setup pins (`3cf0325`).

## llm-jp-4.1 again, with its BOS

The same Q6_K, the same eight conversations, sampling and limits as 10-04, on the i7-14700 (16 threads):

| Conversation | Prompt | Generated |
| --- | ---: | ---: |
| code-python | 130 | 2,048 |
| code-review | 214 | 2,019 |
| en-essay | 118 | 2,048 |
| ja-business | 121 | 899 |
| ja-creative | 106 | 2,048 |
| ja-explain | 105 | 1,089 |
| ja-summary | 202 | 331 |
| math | 114 | 1,419 |

11,901 generated tokens (10,218 on 10-04: with the BOS and the newer llama.cpp the same seed samples different text).
Every prompt is one token longer: the BOS.

`simulate.py` over the eight traces (expert 6.34 MiB), with 10-04's numbers in brackets:

| Slots | GiB | Layers | Profile | Adaptive | Swap MiB/token | LRU | Oracle |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 3.2 | 0.125 | 0.145 (0.133) | 0.266 (0.255) | 105.4 | 0.272 | 0.246 |
| 1,024 | 6.3 | 0.250 | 0.278 (0.267) | 0.427 (0.413) | 132.0 | 0.461 | 0.412 |
| 1,600 | 9.9 | 0.375 | 0.423 (0.412) | 0.574 (0.560) | 114.6 | 0.604 | 0.569 |
| 2,048 | 12.7 | 0.500 | 0.533 (0.523) | 0.673 (0.661) | 89.6 | 0.698 | 0.675 |
| 2,560 | 15.8 | 0.625 | 0.656 (0.648) | 0.774 (0.765) | 61.2 | 0.791 | 0.783 |
| 3,072 | 19.0 | 0.750 | 0.775 (0.769) | 0.864 (0.859) | 36.1 | 0.872 | 0.876 |

The top 25% of the experts take 0.326 of the routing (0.321), the top 50% 0.585 (0.582).
The BOS moves every hit rate up by 0.5 to 1.5 points and changes none of 10-04's conclusions.

`swaps.py` (the adaptive tier with fewer swaps a round):

| Slots | Swaps a round | Hit | Swap MiB/token |
| ---: | ---: | ---: | ---: |
| 450 | 8 | 0.207 | 12.6 |
| 450 | 96 | 0.243 | 98.1 |
| 1,600 | 8 | 0.519 | 12.6 |
| 1,600 | 24 | 0.550 | 37.9 |
| 1,600 | 96 | 0.574 | 114.6 |

## GLM-5.3-Flash

The eight conversations rendered with GLM-5.3-Flash's own template (`reasoning_effort=high`).
Its traces were not taken: the CPU was needed for other work.

## Reproducing

```sh
S=build/llama-oracle-src   # llama.cpp de7fa0a3
cmake -S $S -B build/llama-oracle -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF
cmake --build build/llama-oracle -j --target llama
g++ -O2 -std=c++17 -I$S/include -I$S/ggml/include bench/results/2026-10-09-moe-routing/dump_routing.cpp \
    -Lbuild/llama-oracle/bin -lllama -lggml -lggml-base -Wl,-rpath,$PWD/build/llama-oracle/bin \
    -o build/dump_routing_oracle
python prompts.py CHAT_TEMPLATE.jinja PROMPT_DIR [reasoning_effort=high]
build/dump_routing_oracle MODEL.gguf PROMPT_DIR/NAME.txt TRACE_DIR/NAME.bin 2048 16 1
python simulate.py TRACE_DIR/*.bin [--n-expert N --expert-mib MIB]
python swaps.py TRACE_DIR SLOTS [N_EXPERT EXPERT_MIB]
```
