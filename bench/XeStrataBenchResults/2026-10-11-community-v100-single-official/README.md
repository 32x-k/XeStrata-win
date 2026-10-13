<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# Single V100 32 GB: follow-up measurement

Single-GPU scores requested by the maintainer on PR #8.
[日本語](README.ja.md)

## Results

Medians of two rounds using official `ab_usage` and `ab_generate`.

| Measurement | Strata v0.1.42 | XeStrata xe0.1.40.2.2 |
| --- | ---: | ---: |
| Japanese decode (tok/s) | 65.1 | 64.9 |
| English decode (tok/s) | 78.8 | 76.6 |
| Agent decode (tok/s) | 77.5 | 72.2 |
| Japanese TTFT (s) | 0.54 | 0.68 |
| English TTFT (s) | 0.54 | 0.72 |
| Agent TTFT (s) | 3.87 | 4.25 |
| Agent prefill (tok/s) | 1029 | 937 |
| 1K generate decode (tok/s) | 78.9 | 80.0 |
| 1K generate prefill (tok/s) | 693 | 679 |

Japanese decode is nearly equal. The gap is smaller than in the dual-GPU report, but slot count also changed:
this does not isolate GPU count alone.

## Conditions

CPU, RAM, driver, builds and model match the [dual-GPU report](../2026-10-11-community-v100-x2-official-usage/README.md).
Only the faster PCIe GPU0 (01:00.0) V100 was used, at 150 W. GPU1 had 0 MiB allocated.

- Usage: 32K, INT8 KV, one slot. Japanese 5 / English 5 / agent 7 turns, two rounds: 68 turns.
- The dual-GPU report used two slots. Actual concurrency is one in both reports.
- Same expert profile, IQ3_S, MTP spec 4 / min-p 0.5, expert-cache auto, prefill auto.
- One discarded warm-up per engine. Usage order: Xe then upstream, upstream then Xe.
- Usage temperature 0.7, top_p 0.8, top_k 20, maximum 1,024 tokens.
- Generate: official 1K tier, 1,016 input tokens, 256 generated, 8K context.
  Warm-ups excluded. Remaining measured runs use `--spec-follow` on upstream's first measured answer.
- Upstream default route tail skip and adaptive PCIe share were not disabled.
  Logs show the share changing from 0.36 to 0.10.
- Standalone serve tiers, fixed inputs of 4K or longer, concurrent load and quality evaluation were not tested.

## Evidence for reviewing the Volta path

XeStrata reports 7 of 256 tokens differing from argmax in each followed run; upstream's followed run reports 0.
These are different engines and defaults: this does not by itself establish a Volta bug or quality regression.
Successful execution and speed are not proof of numerical correctness.

The dense-bf16 diagnostic measured approximately 2,726 us default, 22,512 us no-xmx and 2,909 us no-bf16-mma.
The labels report oneMath (cuBLAS) / joint_matrix by default and DP4a for no-xmx.
These are reported path names, not verification of executed instructions by disassembly.
Expert-gate-up/down report off and were not measured.

## Files

- [Usage aggregates](matrix-usage.json), [turn records](usage-turns.json), [configs](configs.json).
- [Generate aggregates](matrix.json), [generate logs](generate-logs.txt), [kernel diagnostics](kernel-bench.txt).
- [XeStrata log](xe-engine.log), [upstream log](up-engine.log).

Personal paths and GPU UUIDs are replaced. Logs contain synthetic benchmarks only.
A helper-script syntax error before measurement was corrected and excluded from the results.
Production dual-GPU chat and LINE services were restored after measurement.
