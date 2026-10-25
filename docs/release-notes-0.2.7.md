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
- **`--prefill-pipe <R>`** - on a two-card layer split, plan the prompt's chunks for the pipeline: **+12.0% at 4K,
  +22.3% at 8K, +2.9% at 16K** prompt on 2x RTX A4000. `R` is the PC's own ratio; `setup --calibrate` measures it
  (1,500 on that box).
- **`--kv q4_0` on a 12 GB card** - the KV cache at 4 bits leaves room for ~1,000 more experts: **+12.0% story
  decode** on an RTX 3060 12 GB (IQ3_XXS). It costs KV precision on long documents (document perplexity +8% at 1K,
  +12% at 8K in `bench/results/2026-09-27-kv-q4`), so it is a choice, not a default.

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
