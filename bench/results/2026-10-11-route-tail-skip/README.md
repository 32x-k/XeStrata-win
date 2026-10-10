<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# STRATA_ROUTE_TAIL_SKIP on XeStrata's GPUs — 2026-10-11

upstream 0.1.42 turned `STRATA_ROUTE_TAIL_SKIP=7` on by default on CUDA serve (b4e3407b, 7bbfab81): in a verify window, a
missed expert (not in VRAM) that every token routes only at rank 7 or lower is neither copied nor computed.
It changes the answers, so XeStrata takes it as a default only when upstream's own checks, redone here, stay within
upstream's figures and it is faster (the upstream-sync skill's rule for approximations): the teacher-forced KL of the
decode path, a task check, and the serve decode speed, on the three GPUs.

- Engine: the workflow-upstream branch at `578818681` (the opt-in port), the contrib-llvm build on the development
  machine (B70, RTX 4070) and the free build with HIP in the AMD container (RX 9060 XT). IQ3_XXS, the shipped expert
  profile, MTP draft layer, `--expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5`.
- Machines: `LOCAL.md` (not in the repository). Paths in the logs are written `<repo>` and `<data>`.

## KL of the decode path ([kl_tail.py](kl_tail.py), [kl/](kl/))

21 chat prompts (code, reasoning, chat, Japanese, Chinese, Russian, German, French, Spanish, Korean, two source-file
summaries), 220 greedy tokens each, 4,620 scored tokens per GPU. The text is the one the skip (=7) writes; =7, =0 and =0
again follow it with `--spec-follow` and dump the logits of every emitted position (`--dump-logits`), so the rows come
through the real verify windows with the MTP drafts. KL(off ‖ on) per position; the floor is KL(off ‖ off again).

| | B70 | RTX 4070 | RX 9060 XT | upstream, RTX 3060 |
| --- | --- | --- | --- | --- |
| KL mean | 0.0027 | 0.0053 | 0.0049 | 0.0058 |
| p99 / max | 0.046 / 0.18 | 0.062 / 0.43 | 0.060 / 0.62 | 0.061 / 0.29 |
| top-1 agreement | 99.1% | 97.8% | 98.4% | 97.7% |
| perplexity of the forced text, on / off | 1.0020 | 1.0017 | 1.0015 | 1.011 / 1.007 |
| first token's KL | 0.0073 | 0.0197 | 0.0146 | 0.024 |
| floor (=0 twice) | 0.0000 | 0.0009 | 0.0012 | 0.0006-0.0007 |

As upstream found, code and reasoning prompts move least and chat in other languages most.

## Task check ([task_tail.py](task_tail.py), [task/](task/))

Greedy, thinking off, through `strata --serve`: every second HumanEval problem (82; the answer's code run against the
problem's tests) and 30 generated arithmetic word problems (an `Answer: <integer>` line), with =0 and =7.

| | B70 | RTX 4070 | RX 9060 XT |
| --- | --- | --- | --- |
| HumanEval, =0 / =7 | 75 / 76 | 77 / 79 | 77 / 77 |
| arithmetic, =0 / =7 | 25 / 24 | 25 / 25 | 24 / 24 |

One run each: a change of one or two tasks is noise.

## Serve decode ([ab/](ab/), [ab-rx9060xt/](ab-rx9060xt/))

`.claude/skills/upstream-sync/scripts/ab_pair.py serve` with =0 (A) and =7 (B), the chat of four turns at 1K and 32K,
greedy and temperature 0.7, five rounds alternating; medians, =7 against =0.

| | 1K greedy | 1K t0.7 | 32K greedy | 32K t0.7 |
| --- | --- | --- | --- | --- |
| B70 decode | -1.0% | +12.7% | +3.0% | +7.9% |
| RTX 4070 decode | +24.5% | +7.5% | +22.7% | +18.8% |
| RX 9060 XT decode | +16.1% | +6.3% | +42.0% | +55.4% |
| RTX 4070 follow-up wait | -17.6% | -12.9% | -17.5% | -22.2% |
| RX 9060 XT follow-up wait | -7.5% | -20.3% | -4.1% | -11.1% |
| B70 follow-up wait | -0.4% | +0.4% | +0.4% | +1.8% |

The first prompt is read at the same speed (the skip works in verify windows only). One cell says otherwise: the RX 9060
XT's 1K first prompt at temperature 0.7, 347 against 445 tokens/s. Its rounds fall in two groups in both arms (=0 340-503,
=7 326-522; [ab.txt](ab-rx9060xt/ab.txt)) and the medians landed in different ones: noise of that prompt, not the
skip. Each engine writes its own text, so the cells move up to about 10% from round to round at temperature 0.7.

## What it says

- The answers move as little as upstream measured: the mean KL is 0.003-0.005 against upstream's 0.0058 on the RTX
  3060, the forced text's perplexity rises by 0.15-0.2%, and the tasks pass as often with the skip as without.
- Decode is faster where experts miss VRAM: the RX 9060 XT (16 GB) most, at 32K by 42-55%, the RTX 4070 (12 GB) by
  8-25%; the B70 (32 GB, few misses) by -1 to +13%, no cell slower beyond the noise.
- So it is on by default on serve (rank 7), on every GPU: upstream keeps it off on HIP, where XeStrata measured its
  largest gain. Off when every expert is in VRAM, with `--batch` and with a helper cache; `STRATA_ROUTE_TAIL_SKIP=0`
  turns it off.
