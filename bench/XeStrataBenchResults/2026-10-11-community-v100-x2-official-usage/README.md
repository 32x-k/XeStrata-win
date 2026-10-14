<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# Dual V100 32 GB: current releases with the official usage benchmark

XeStrata xe0.1.40.2.2 and Strata v0.1.42 were measured alternately on the same PC.
Unlike the short custom tasks in PR #3, this uses the new `tools/bench_core.py`
`ab_usage`, conversations, sampling and aggregation without changes.
Upstream was faster in Japanese, English and coding on this machine.

[日本語](README.ja.md)

## Results

Medians of two rounds. Each conversation's throughput is total tokens / total time within a round,
then aggregated across rounds. TTFT is the median of each round's turn medians.

| Measurement | Strata v0.1.42 | XeStrata xe0.1.40.2.2 |
| --- | ---: | ---: |
| Japanese decode (tok/s) | 79.2 | 68.2 |
| English decode (tok/s) | 88.5 | 80.5 |
| Agent decode (tok/s) | 87.2 | 75.6 |
| Japanese TTFT (s) | 0.64 | 0.96 |
| English TTFT (s) | 0.66 | 0.84 |
| Agent TTFT (s) | 4.53 | 5.29 |
| Agent prefill (tok/s) | 876 | 743 |

## Conditions and reproduction

- Ryzen 9 9950X, 128 GB RAM, Ubuntu 24.04.5, NVIDIA 580.178.04.
- Two Tesla V100 PCIe 32 GB cards, 150 W each. PCIe probes approximately 13.1 / 3.2 GB/s.
- IQ3_S GSQ-RCO, 32K, INT8 KV, MTP spec 4 / min-p 0.5, expert-cache auto, prefill auto.
- Both engines: 24 layers per card, two slots, one actual request at a time.
- Same frozen expert profile; no additional V100 optimization switches enabled.
- XeStrata: contrib-llvm, CUDA 12.8 / sm_70. Upstream also built for CUDA 12.8 / sm_70.
- One warm-up response per engine discarded. Order: Xe then upstream, upstream then Xe.
- Temperature 0.7, top_p 0.8, top_k 20, maximum 1,024 tokens. Japanese 5, English 5, agent 7 turns.
- All 68 measured turns completed. Maximum GPU core temperature: 52 degrees C.

`conditions.json` records the commits; `configs.json` contains the actual launch configurations.
`<HOME>` and `<GPU_UUID>` are publication placeholders: replace them for your machine.
Model, pack, MTP and expert-profile files and their checksums are not included.

Use `tools/bench_core.py` from the measured tag, construct each `Engine` from its configuration,
and call
`ab_usage(out, "V100x2", engines, config_callback, 32768, load_corpus()["agent"], rounds=2, temperature=0.7)`.
Each engine uses its own `serve/server.py`. Then aggregate with `collect`.
The configuration's port is replaced by a free port on every launch.

## Differences from the official CLI and limitations

At this tag, `bench.py:usage_cfg` does not preserve `split_pci`, `layer_split`, `parallel` or `env`
when `--original-strata` is supplied. Upstream preparation also selects the first GPU.
This is therefore not an unmodified `--bench --original-strata` invocation.
We passed full engine-specific configurations to the official `ab_usage` to retain the dual-GPU setup.
Measurement, prompts and aggregation implementations are unchanged.

Like the official CLI, no standalone generate/serve tiers were measured for the split configuration.
This is not a concurrent-load test, a quality evaluation or a full 32K-input speed measurement.
Sampling produces different answers, so this is not an identical-output kernel comparison.
Fixed prefill overhead has a large effect on the short Japanese prompts.

## Diagnostic files

- [Aggregates](matrix-usage.json), [CSV](comparison.csv), [turn values and warm-ups](runs.json).
- [XeStrata engine log](xe-engine.log), [upstream engine log](up-engine.log).
- Logs contain synthetic benchmarks only, with no personal conversations, credentials, usernames or GPU UUIDs.
- XeStrata reports the two-GPU split and oneMath (cuBLAS) / joint_matrix paths.
  This alone does not establish individual kernel efficiency or the cause of the slowdown.

Production services were restored after testing. This PR contains records only, with no engine changes.
