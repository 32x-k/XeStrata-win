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

- **Two more kernel switches are on by default where they were measured.** Both sum in another order (about 1e-7), so a
  tie between two near-equal words can flip and a reply can differ from 0.2.6 here and there.
  - The prompt attention on int8 tensor cores (NVIDIA Ampere and newer): **+1.6% / +1.8% prompt at 8K / 16K on the RTX PRO 5000
    and +1.1% / +1.3% on 2x RTX A4000**, decode unchanged, five pairs each. Off switch: `STRATA_PROMPT_ATTN_IMMA=0`.
  - The hyper-connection read split over two kernels (a discrete RTX 50 / RTX PRO Blackwell card with more than 13 GiB): **+1.2%
    decode on the RTX PRO 5000 with identical tokens**, about +1.5% together with the switch above. On 2x RTX A4000 and the
    RTX 3060 it was slower, so those cards keep the old read. Off switch: `STRATA_GR_V3=0`.

## Going back to 0.2.6

Two switches bring back every default that 0.2.7 changed, in one step, for a test or for a PC where something looks wrong. The app's
"off" for its speed switches sends them.

- **`DS4_DEFAULTS=026`** (DeepSeek): the staging ring, the prompt chunk rule, the attention slab (its base and its growth), the pinned
  copy and the huge-page arena, the matrix-core attention, the drafter's cumulative cut, the 5-experts default on several Radeon cards,
  the RAM store's spare slots and put and its settle after a prompt, the dense dequantizer, the F16 gate/up operand, the drafter's
  markov head and the head's column count, and the RDNA3 dense GEMM all go back to what 0.2.6 did.
- **`STRATA_DEFAULTS=026`** (Qwen): the residency-biased routing, the wide and the stream-split hyper-connection reads, the int8
  prompt attention, the four gfx12 decode switches, the PLE residency policy and the gfx12 decode split of 0.15 (back to 0.55).
  Long prompts on the approximate read (`STRATA_KVA`) are off by default and stay as `STRATA_KVA` says.

A variable or flag you set yourself (`DS4_ATTN_SLAB=N`, `STRATA_ROUTE_PRIOR=2`, `--experts 5`, `--prefill-chunk N`, `--pcie-frac`,
`--ple-io`) still wins over the switch. The older single switches (`DS4_FAST_ATTN=0`, `DS4_ATTN_SLAB=0`, `DS4_COPY_PIN=0`,
`DS4_COPY_HUGETLB=0`, `DS4_STORE_SPARE=0`, `DS4_STORE_PUT=0`, `STRATA_ROUTE_PRIOR=0`, ...) work as before.

- **Strix Halo (`gfx1151`) on Windows: the shared expert no longer runs on its own stream.** `STRATA_SH_STREAM` was on by default on
  `gfx1151`; on a Windows PC it is slower. Qwen3.8-Flash-Next IQ3_S, three interleaved pairs: chat decode **43.6 -> 46.6 tokens/s (+6.9%)**
  with it off, prompts unchanged. The answers are the same: five fixed greedy prompts (up to 16,383 tokens) gave identical token ids with
  the switch on and off, in four runs. Linux `gfx1151` keeps it on, other cards are untouched. Turn it back on with
  `STRATA_SH_STREAM=1`; `STRATA_DEFAULTS=026` restores the 0.2.6 behaviour. Details and the other numbers of that PC:
  [docs/STRIX_HALO_WINDOWS.md](STRIX_HALO_WINDOWS.md).

## What is faster if your PC is set up for it

These need no engine change; they are settings. Each was measured on the same boxes.

- **`--ple-io ram`** - the 28.8 GB n-gram table in RAM instead of read from the SSD row by row. On the 72 GB card
  (251 GB RAM): prompt **+16.5% at 1K, +30.6% at 4K, +16.8% at 8K**, -2.6% at 16K. Setup chooses it on a Linux PC
  with RAM to spare - check `setup --plan` on yours before trusting that.
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

**On by default at 1.5 on a discrete NVIDIA card with 13 GiB of memory or less** (`STRATA_ROUTE_PRIOR=0` turns it off, an explicit value wins;
not a default on a Radeon, a Strix Halo or any integrated GPU, where it was not measured). The router prefers
experts that are already resident, so a window stops waiting on a page-in that the next window would need again. The first
0.2.7 candidates used a bonus of 2.0, and the table below was measured at 2.0 on the RTX 3060 12 GB (IQ3_XXS), five interleaved
pairs of fresh engines. The default was lowered to 1.5 by the owner on 26 October after the copy and code check on that card
(IQ3_XXS) found real defects in the answers at 2.0 and **no defect in 19 tasks at 1.5** (or at 1.0). 1.5 keeps most of the speed:
story **+15%** and code **+37%** against the routing off. That is 19 tasks, not a proof; a longer 61-task check of 0, 1.5 and 2.0
is still running, and if it shows defects at 1.5 the default falls back to 1.0. The 2.0 measurements:

| axis | before | after |
|---|---|---|
| decode story | 45.9 | **61.6 (+34.1%)** |
| decode code | 43.6 | **73.1 (+67.6%)** |
| prompt | **not claimed** - this box's prompt read swings between 218 and 466 tok/s across runs, and a repeat of the same five-pair A/B read +254% and -20.5% where the first read +35% and +17% |
| decode behind a 4K document | 45.1 | **62.2 (+37.9%)** |

It holds where it matters most: with **two conversations in flight** the whole recipe reads **31.2 -> 62.0 tok/s aggregate
(+99%)** on that card and +22% on the 2x A4000, both answers arriving in half the wall time on the 12 GB box, and the routing
default alone is **+67% aggregate** there (59.7 against 35.6 tok/s) -
and on content the tuning was never fitted to it **+29.0%** (43.5 -> 55.0 tok/s on a
document prompt), and the whole recipe - routing, a workload-fitted expert profile, `--adapt-every 0`, `--vram-reserve-mib
896` and the English draft vocabulary - gives **+26.5%** there and **+80%** on code in a sustained 80-request session.

It works by choosing a slightly different set of experts, so replies change: the owner approved that trade for 12 GB cards.
**Do not raise the bonus to 2.0 or more.** 2.0 showed defects in the copy and code check; 3.0 measures +17.7% / +24.0% on the test
prompts but **-4.5% on held-out content**, and 4.0 breaks outright. `STRATA_ROUTE_PRIOR=2` still works if you want the old value.

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




