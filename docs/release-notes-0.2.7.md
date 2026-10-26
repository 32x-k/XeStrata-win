# Strata 0.2.7 (draft): a faster Qwen3.8 on NVIDIA

> **DRAFT for the owner.** Written by the CUDA speed work on branch `f-027-cuda`; no engine version is claimed here
> because the owner decides what else 0.2.7 carries. Every number below was measured on the borrowed boxes named in
> each line, same day, against the same build with the new default turned off.

## What is faster, by default

- **The hyper-connection read on RTX 50 and RTX PRO Blackwell cards (compute capability 12.0).** The model's
  hyper-connection mixing runs 96 small projections per token; its "up" half now runs one warp per row instead of
  eight rows per warp, which fills a modern card's SMs instead of 160 blocks on 192 of them. Measured on an RTX PRO
  5000 Blackwell 72 GB (Qwen3.8-Flash-Next IQ3_S, every expert in VRAM), five interleaved pairs of fresh engines:
  **story 197.6 -> 201.3 tokens/s (+1.9%), code 243.5 -> 245.1 (+0.7%)**. The answers are unchanged - the new
  kernel does the same arithmetic in the same order, the produced tokens were identical in all six pairs and the
  engine's own kernel check is bitwise.
  - Off switch: `STRATA_GR_UP_WIDE=0`. `STRATA_SM120_PERF=0` still turns off the whole Blackwell set.
  - Other cards are untouched: the old kernel runs everywhere else.

## What is faster if your PC is set up for it

These need no engine change; they are settings. Each was measured on the same boxes.

- **`--ple-io ram`** - the 28.8 GB n-gram table in RAM instead of read from the SSD row by row. On the 72 GB card
  (251 GB RAM): prompt **+16.5% at 1K, +30.6% at 4K, +16.8% at 8K**, -2.6% at 16K. Setup already chooses it on
  Linux PCs with RAM to spare.
- **`--mtp-draft-vocab draft_vocab_en.bin`** - the helper that guesses the next words considers 162 KB of the
  vocabulary instead of 425 KB, so it reads less per guess: **+2.7..3.1% story and +2.2..3.0% code** decode on both
  big boxes. The big model checks every guess, so the answers are unchanged (identical tokens in every run where
  the machine is deterministic). A CJK or Cyrillic user should keep the wider subset.
- **`--slots 2`** - when the engine serves more than one conversation at a time (an agent with several sessions), decoding two
  requests together returns **+10% aggregate throughput on the RTX PRO 5000 and +12% on the RTX 3060**, for a slightly higher
  latency per token. Three and four slots add nothing measurable (+8.4% and +11.7%): the gain is all in the second slot, and
  more slots cost KV memory. A single conversation gains nothing from it.
- **`--prefill-pipe <R>`** - on a two-card layer split, plan the prompt's chunks for the pipeline: **+15.4% at 4K,
  +22.5% at 8K, +2.7% at 16K** prompt on 2x RTX A4000 (5 interleaved pairs). `R` is the PC's own ratio; `setup --calibrate` measures it
  (1,500 on that box).
- **`--kv q4_0` on a 12 GB card** - the KV cache at 4 bits leaves room for ~1,000 more experts. On the RTX 3060
  12 GB (IQ3_XXS) the gain did **not** reproduce: +12.0% story over three pairs, -2.9% over five, with that box's
  own baseline moving 41.6 -> 49.0 tok/s between sessions, so no speed claim is made for it. It costs KV precision
  on long documents (document perplexity +8% at 1K, +12% at 8K in `bench/results/2026-09-27-kv-q4`), so it is a
  choice, not a default.

## The 12 GB default: residency-biased routing

**On by default at 2.0 on a card with 13 GiB of memory or less** (`STRATA_ROUTE_PRIOR=0` turns it off). The router prefers
experts that are already resident, so a window stops waiting on a page-in that the next window would need again. Measured
on the RTX 3060 12 GB (IQ3_XXS), five interleaved pairs of fresh engines:

| axis | before | after |
|---|---|---|
| decode story | 45.9 | **61.6 (+34.1%)** |
| decode code | 43.6 | **73.1 (+67.6%)** |
| prompt 1K | 333.6 | **451.5 (+35.3%)** |
| prompt 4K, in a 4K context | 248.5 | **291.0 (+17.1%)** |
| decode behind a 4K document | 45.1 | **62.2 (+37.9%)** |

It holds where it matters most: with **two conversations in flight** the whole recipe reads **31.2 -> 62.0 tok/s aggregate
(+99%)** and both answers arrive in 4.8 s instead of 9.7 s, and the routing default alone is **+67% aggregate** (59.7 against
35.6 tok/s) -
and on content the tuning was never fitted to it **+29.0%** (43.5 -> 55.0 tok/s on a
document prompt), and the whole recipe - routing, a workload-fitted expert profile, `--adapt-every 0`, `--vram-reserve-mib
896` and the English draft vocabulary - gives **+26.5%** there and **+80%** on code in a sustained 80-request session.

It works by choosing a slightly different set of experts, so replies change: the owner approved that trade for 12 GB cards.
**Do not raise the bonus beyond 2.0.** 3.0 measures +17.7% / +24.0% on the test prompts but **-4.5% on held-out content**,
and 4.0 breaks outright - 2.0 is the setting that generalises.

## What was tried and did not make it

Recorded so nobody repeats it: deeper or wider MTP windows, `STRATA_QFUSE`, `STRATA_STAGE_TRIM`, `STRATA_HC_Q8`,
DFlash2 as the drafter, 4-bit draft copies, `pcie_frac` changes on the 3060, `--pipeline-windows` on a two-card
split, the K-split and the shared-memory-free variants of the hyper-connection down projection, `--prefill` chunks
above 8K, and every Blackwell default switch turned off one at a time. The numbers are in
`CUDA Plan/RESULTS.md` outside this repository.

## Updating

- The engine package that carries this is chosen by the owner; the app and the Terminal keep their own versions.
- From 0.2.6: the usual update path for the part that changed.

---

Strata is free and open source. If it runs well on your PC, a coffee keeps the work on it going:

<a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a>




