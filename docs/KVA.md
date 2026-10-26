# KVA: approximate prompt processing for the late layers (opt-in, research)

This is an **experimental, opt-in** feature. Nothing in it runs unless `STRATA_KVA=1` is set, and a build without
that variable behaves exactly as before.

KVA ("approximate late-layer state") reads a long prompt exactly through layer 23, then for the late layers
24–47 builds **only their recurrent state** — the GDN recurrence and convolution history, the QSA K/V pages and
the indexer keys — instead of running the whole layer. The attention output, the output projection, the MoE half
(router, experts, shared expert) and the late layers' expert transfers are skipped, because nothing reads their
results. The last positions of the prompt are then read fully, exactly as before, and decode runs the ordinary
full path against the state KVA built.

## Turning it on

```sh
STRATA_KVA=1 \
STRATA_KVA_SUFFIX=2112 \
STRATA_KVA_MIN_PROMPT=4096 \
  strata --serve ... --prefill 1024 --mtp-window 2048
```

| variable | default | what it does |
|---|---:|---|
| `STRATA_KVA` | off | `1` enables the feature for qualifying requests |
| `STRATA_KVA_SPLIT` | 24 | first approximated layer (zero-based) |
| `STRATA_KVA_SUFFIX` | 2048 | fully computed tail positions; rounded **up** to a whole chunk |
| `STRATA_KVA_MIN_PROMPT` | 4096 | shorter prompts stay on the full path |
| `STRATA_KVA_ONLY` | all | `gdn` or `qsa`: approximate only that layer type |
| `STRATA_KVA_MODE` | identity | `identity`, `oracle`, `linear` (linear is not implemented) |
| `STRATA_KVA_CAPTURE` | — | directory: write every late layer's real input for calibration |
| `STRATA_KVA_CAPTURE_FP32` | off | write fp32 instead of bf16 records |
| `STRATA_KVA_ORACLE` | — | directory: replay a capture instead of a projector |
| `STRATA_KVA_GDN_NORM` | off | keep the full GDN recurrence (the A/B control for the state-only one) |
| `STRATA_LOGITS_DUMP` | — | debug (not KVA-specific): write every verify window's sampler logits to `path`, for KL / top-1 between two runs |

Requirements and limits for this first version:

- One cold read of one prompt, one device (`pos0 == 0`, one stage). A follow-up read, a resumed prompt or a
  layer split stays on the full path.
- The fully computed suffix must cover the MTP drafter's window: use `--mtp-window N` with
  `STRATA_KVA_SUFFIX >= N + 64`, or the drafter reads state built from approximated positions.
- **The session state after a KVA read is approximate.** It must not be reused as an exact prefix: no
  checkpoint of it, no follow-up prompt that reuses it. The engine prints a warning when this happens.
- The approximation is a quality tradeoff, not a lossless transformation. Measure before using it.
- Every run says what it did:
  - applied: `strata prefill: KVA on: split 24, suffix 2112, approximated 5120 of 7533 positions in 5 of 8 chunks, mode identity`
  - configured but not applied: `strata prefill: KVA configured but NOT applied: <reason> (prompt … positions, min_prompt …, suffix …, split …)`

## Measured (Tesla P100 16 GB, Qwen3.8-Flash-Next IQ3_XXS, chunk 1024, expert cache 1500, MTP window 2048)

| prompt tokens | control prompt tok/s | KVA prompt tok/s | speedup | approximated | residual error on the suffix |
|---:|---:|---:|---:|---:|---:|
| 5,642 | 203.3 | 265.7 | 1.31× | 54 % | 0.357 |
| 14,581 | 202.5 | 311.2 | 1.54× | 85 % | not measured |
| 29,997 | 199.7 | 338.3 | 1.69× | 92 % | 0.451 |

- Decode throughput was unchanged within the run-to-run spread (24–28 tok/s) in every arm: KVA changes prompt
  processing only.
- Quality: on a 14.6K-token prompt with a fact inserted at 10/50/90 %, the KVA arm retrieved the fact at all
  three positions and produced the identical answer string. On a "summarise this documentation" prompt the
  answers stayed coherent and semantically identical, with different wording after 4–12 tokens.
- **Likelihood** (`STRATA_LOGITS_DUMP`): at the first generated token the approximation moves the next-token
  distribution by KL 0.018 nats at a 5.6K prompt (two control runs differ by 0.0013) and 0.020 nats at 30K
  (control floor 0.0032). The top-1 token is the same and 9 of the top 10 are shared in both cases. The effect
  does not grow with context even though the residual error does.
- Layer type matters more than the count: approximating only the 6 late QSA layers cost a 0.153 residual error
  at 1.08×; approximating 18 GDN layers cost 0.339 at 1.18×. A gated recurrence integrates its input over the
  whole prefix, so a wrong input keeps biasing the state, while a wrong QSA K/V only misplaces attention keys.
- On IQ2_XS the same 14.6K-token three-needle test gives 208.0 → 328.7 prompt tok/s (1.58×) with all three
  codes retrieved, against 201.6 → 320.7 (1.59×) on IQ3_XXS: the feature is not tied to one quantization.
- The control is not bit-reproducible on this box (identical runs differ in the last chunks by ~0.85 %), so the
  residual figures are quoted against that measured floor, never as bit equality.

## What the block input is

The late layers' input is the 10,240-value hyper-connection residual (`hc * n_embd`) that the prompt path
carries. In `identity` mode every late layer reuses the residual entering layer 24 — no fitted weights, no extra
work. Capturing the *real* per-layer inputs (`STRATA_KVA_CAPTURE`) and replaying them (`STRATA_KVA_ORACLE`)
reproduces the full path's answer exactly; that is the mode a fitted projector would replace.

A per-position linear projector was tested and rejected: fitted on 5,636 positions, a rank-256 map from the
boundary to each late layer's input is *worse* than reusing the boundary for the first nine late layers and never
gets below 0.80 relative error (`bench/2026-10-25-predictability.md` in the research folder). The residual is
moved by routing and attention decisions a linear map cannot follow.

## Capturing inputs

```sh
STRATA_KVA_CAPTURE=/data/kva-cap STRATA_KVA_CAPTURE_FP32=1 \
  strata --serve ... --prefill 1024 --tokens-file ids.txt --max-new 1
```

writes `L24.bin … L47.bin` in that directory: one row of `hc * n_embd` floats per prompt position per late
layer, in position order. A second prompt overwrites them. `STRATA_KVA_ORACLE=<dir>` replays such a directory in
place of a projector; the replay must use the same prompt and the same chunking.

## Implementation notes

- `src/prefill/kva.hpp` — the plan and the environment variables.
- `src/prefill/prefill.cpp` — the chunk plan (`kva_approx`), the capture/oracle taps, the state-only layer
  exits, the transfer-plan skip, and `give_back()` on a skipped MoE half: a layer that consumes no expert must
  still release its stream entries, or the issuer thread fills the ring and the next fully computed layer waits
  forever.
- `src/prefill/kernels.cu` — `gdn_recurrence_state_only()`, the same recurrence dispatch without the output
  normalization.
- CUDA and HIP builds (`src/`); the SYCL engine (`sycl/`) does not have the feature.
