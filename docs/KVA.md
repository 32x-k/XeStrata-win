# KVA: the faster prompt read for long cold prompts (on by default since 0.2.7)

KVA ("approximate late-layer state") reads a long prompt exactly through layer 23, then for the late layers
24–47 builds **only their recurrent state** — the GDN recurrence and convolution history, the QSA K/V pages and
the indexer keys — instead of running the whole layer. The attention output, the output projection, the MoE half
(router, experts, shared expert) and the late layers' expert transfers are skipped, because nothing reads their
results. The last positions of the prompt are then read fully, exactly as before, and decode runs the ordinary
full path against the state KVA built.

It is an approximation: the answers can differ from the exact read (see "Measured" for how much). **`STRATA_KVA=0`
turns it off and gives the exact 0.2.6 prompt path.** The Strata app shows this as the "Faster long prompts" switch
in Expert settings; it sends `STRATA_KVA=0` when the switch is off.

## When it runs

KVA runs when all of these hold, and says so in the engine log:

- Qwen3.8-Flash-Next on CUDA or HIP (not SYCL, not Metal; Qwen3.6 and the other models read exactly);
- one GPU, no `--batch` slots, no `--peer-device`, no DFlash2 drafter;
- the read is cold (it starts at position 0 of the session) and the text has no pictures;
- the prompt (or the part of it read in one go, see below) has at least `STRATA_KVA_MIN_PROMPT` tokens (4096) and
  more than the fully computed tail plus 64.

Otherwise the prompt is read exactly. An engine that was started with `STRATA_KVA=1` logs the reason for every
read it turned away (`strata prefill: KVA configured but NOT applied: <reason>`); with the default it logs only
a long prompt turned away for an unusual reason (a layer split, pictures), and once at start-up when the engine
configuration rules KVA out for the whole run.

## The fully computed tail and the draft layer

The last `suffix` positions (2048, or `STRATA_KVA_SUFFIX`) are always read exactly.

The MTP draft layer builds its K/V from the final residuals of the prompt. An approximated chunk has no final
residual of its own, so its draft cells are built from the rows that chunk does have. By default nothing about the
draft layer changes: its window is whatever it was (`--mtp-window`, default 32768 cells), it builds the same cells and
attends to the same cells as with `STRATA_KVA=0`. The target model verifies every token, so the draft only decides
how many tokens a verify window keeps (decode speed), never the answer.

Measured on the P100 (IQ3_XXS, auto cache and prefill, 14K and 30K prompts, 256 tokens), the draft layer attends to
those cells with acceptance within noise of the exact read. A forced small window (the first default-on build set
`--mtp-window 2048` for every session) and the research option below both cost acceptance on faster cards.

`STRATA_KVA_DRAFT_FLOOR=1` (research): the draft layer gets a *cell floor* per session, the first cell built from a real
row. It builds no draft cell below it and attends to `[max(floor, pos - window), pos)`, the exact tail growing with
the conversation. The floor follows the tokens of the state a request continues from (`kva_floor.hpp`); a state no KVA
read produced has floor 0. It is off by default because it costs draft acceptance after long prompts.

## Chats, follow-up turns and checkpoints (`--serve`)

The server reads a prompt in parts: up to the end of a long system prompt (the shared checkpoint), up to the last
turn header, then the new turn's header and the last token. Each part that starts at position 0 and is long enough
can take the approximate path; its own last `suffix` positions are exact, and every later part starts at a position
greater than 0 and is read exactly.

- **Turn 2 and later.** The server continues from the live session or from a conversation checkpoint (the state
  after the earlier turns) and reads only the new tokens, at positions greater than 0, so they are read in full. They
  are read exactly *on top of* the cells the first read left: nothing is approximated twice. Questions about the
  old document in turn 5 are answered from the same cells turn 1 had, so their quality is that of the first read.
- **Checkpoints and parked conversations** hold whatever cells the session held, approximated or not, and give
  them back as they were. A checkpoint taken inside an approximated stretch is a consistent state: the read
  continues from it exactly. Restoring a checkpoint and re-reading the same prompt from 0 (the `STRATA_CKPT_REREAD`
  debug check) differ with KVA on, as they would between any two reads that take different paths.
- **A cold read again** (the cache evicted the conversation, a new chat with the same system prompt) approximates
  its own prefix again.
- **`STRATA_KVA=0`** restores 0.2.6: every prompt is read exactly, the draft window is 32768 cells, nothing else
  changes.

## Layer splits, batch slots, SYCL, Metal

These never take the approximate path and say why: a layer split reports "a layer split (not the whole trunk)", the
batch slots and a peer device switch KVA off for the process (one line at start-up with `STRATA_KVA=1`), the SYCL
engine has its own prompt path without KVA, and the Metal build has no state-only kernels (`STRATA_KVA` is ignored
there, with a line when it is set).

## Settings

| variable | default | what it does |
|---|---:|---|
| `STRATA_KVA` | on | `0` turns KVA off (the exact 0.2.6 prompt read). `1` is the same as the default but logs every read it turned away |
| `STRATA_KVA_SPLIT` | 24 | first approximated layer (zero-based) |
| `STRATA_KVA_SUFFIX` | 2048 | fully computed tail positions; see above |
| `STRATA_KVA_MIN_PROMPT` | 4096 | shorter prompts stay on the full path |
| `STRATA_KVA_DRAFT_FLOOR` | off | `1`: the draft layer skips the draft cells of approximated chunks (research) |
| `STRATA_KVA_SKIP` | state | `moe` (opt-in): the late layers of an approximated chunk run their whole attention half (GDN and QSA outputs into the residual) and skip only the MoE half and its expert transfers; the log line says `mode moe-skip`. Slower than the default state-only form, closer to the exact residual |
| `STRATA_KVA_ONLY` | all | `gdn` or `qsa`: approximate only that layer type |
| `STRATA_KVA_MODE` | identity | `identity`, `oracle`, `linear` (linear is not implemented) |
| `STRATA_KVA_CAPTURE` | — | directory: write every late layer's real input for calibration (the read is exact while capturing) |
| `STRATA_KVA_CAPTURE_FP32` | off | write fp32 instead of bf16 records |
| `STRATA_KVA_ORACLE` | — | directory: replay a capture instead of a projector |
| `STRATA_KVA_GDN_NORM` | off | keep the full GDN recurrence (the A/B control for the state-only one) |
| `STRATA_LOGITS_DUMP` | — | debug (not KVA-specific): write every verify window's sampler logits to `path`, for KL / top-1 between two runs |

Every read says what it did:

- applied: `strata prefill: KVA on: split 24, suffix 2048, approximated 5120 of 7533 positions in 5 of 8 chunks, mode identity`
- turned away: `strata prefill: KVA configured but NOT applied: <reason> (prompt … positions, min_prompt …, suffix …, split …)`

A quality comparison needs the control arm to say `STRATA_KVA=0` (an unset variable is now KVA).

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

- `include/strata/prefill/kva.hpp` — the plan, the environment variables and the reasons a read is turned away (`src/prefill/kva_test.cpp` tests them).
- `src/prefill/prefill.cpp` — the chunk plan (`kva_approx`), the capture/oracle taps, the state-only layer
  exits, the transfer-plan skip, and `give_back()` on a skipped MoE half: a layer that consumes no expert must
  still release its stream entries, or the issuer thread fills the ring and the next fully computed layer waits
  forever.
- `src/prefill/kernels.cu` — `gdn_recurrence_state_only()`, the same recurrence dispatch without the output
  normalization.
- CUDA and HIP builds (`src/`); the SYCL engine (`sycl/`) does not have the feature.
