# Split serving: read the prompt on one card, make the answer on another

> **Status: opt-in, first version, same make of cards on one PC.** Nothing changes unless you turn it on. It needs two cards
> that can each run the whole model, and it only pays on some workloads. Read [When it helps](#when-it-helps) first.

The idea, the hand-made setup and the first measurements are by @MarcoTool in issue #4000: a router in front of two
engines, an RTX 3070 reading the prompt and two MI50 cards making the answer, with the session handed over through
`/dev/shm`. This page describes the built-in version of the same idea.

> **On this page:** [What it does](#what-it-does) · [Turning it on](#turning-it-on) · [When it helps](#when-it-helps) ·
> [Measured](#measured) · [How it works](#how-it-works) · [What must match](#what-must-match) · [Limits](#limits) ·
> [NVIDIA with AMD, and other PCs](#nvidia-with-amd-and-other-pcs)

## What it does

A normal two-card setup ([MULTI_GPU.md](MULTI_GPU.md)) cuts the model into layers and every card works on both stages. Split
serving does not cut the model. It starts two engines, each on one card, each with the whole model:

- the **prefill engine** reads the prompt and stops just before the first answer token;
- the engine saves that session into RAM, the **decode engine** loads it, and makes the answer.

The two engines share one copy of the experts in RAM (`--shared-expert-arena`, Linux). The client still talks to one
OpenAI-compatible endpoint; the server hands the session over between the two.

A prompt whose new part is short (a chat follow-up, an agent step) never leaves the decode engine: it is read in place, as
without the option. Only a long new part goes to the prefill engine.

## Turning it on

In the run config (`strata-<model>.json`), next to the other keys:

```json
{
  "split_serve": {"prefill_gpu": 0, "decode_gpu": 1}
}
```

or `strata serve --split-serve 0,1` (prefill card first). The cards are numbered as `nvidia-smi` numbers them. The option
replaces `gpu`; the config must not also name `replicas`, `--layer-split`, `--batch`/`--slots` or `--prompt-cache 0` (the
server stops at start and says which).

| Key | Default | Meaning |
| --- | --- | --- |
| `prefill_gpu`, `decode_gpu` | required | The two cards. Different cards. |
| `min_prompt_tokens` | 2048 | A request is handed over when the decode engine would otherwise read at least this many new tokens. Below it the decode engine reads them in place. |
| `dir` | `/dev/shm` if it has room, else the temp folder | Where the session file lives for the seconds of a hand-over. |
| `shared_arena` | `true` | One copy of the experts in RAM for both engines (Linux, a file in `/dev/shm` when the experts fit there). `false`: each engine keeps its own copy (two copies of the experts in RAM). A file name: that file backs the arena (a file on disk works; it is written once). |
| `prefill_args`, `decode_args` | none | Extra engine arguments for one role, for example a bigger expert cache for the decode engine. They must not change the [session settings](#what-must-match). |

The startup line says what was chosen, including when the arena cannot be shared:

```
[strata] split serving: the prompt is read on card 0, the answer is made on card 1 (prompts with 2048+ new tokens are handed over); one shared expert arena in /dev/shm (32.0 GiB)
```

`/metrics` has a `split_serve` block: requests, hand-overs, fallbacks, bytes moved and the time spent saving and restoring.

## When it helps

Prefill needs matrix units and compute. Decode is limited by memory bandwidth and by how much of the model sits in VRAM. If one
card is much better at the first and another at the second, each can do what it is best at. That was the case in #4000: an
RTX 3070 reads a 32K prompt at about 1,600 tokens a second, the two MI50 cards read it at about 390 but decode at 40 tokens a
second with every expert in VRAM. The hand-over costs 2 seconds at 32K.

It does **not** pay when:

- the two cards are alike: a layer split over both cards reads the prompt on both and keeps twice the experts in VRAM, which
  split serving gives up (each engine has one card's worth of VRAM for experts);
- the hand-over (the session file's size over RAM bandwidth, written and read) costs more than reading the new tokens on the
  decode card would have;
- the new part of each request is short, which is most chat and agent use: nothing is handed over.

[Measured](#measured) shows the numbers on two identical cards.

## Measured

SPLIT-RESULTS-PLACEHOLDER

## How it works

1. The server sees a request with at least `min_prompt_tokens` new tokens for the decode engine (it tracks what the decode
   engine's session holds).
2. It sends the prompt to the prefill engine as `GEN 0 prefill=1 <ids>`. The engine reads the prompt as a request would, in
   the same chunks, with the same checkpoints, up to the place where the first answer window starts, and stops. Its session
   then holds all but the last prompt token. That is exactly the state a following turn resumes from, so the decode engine
   needs no new code path.
3. `SAVE <file>`: the engine's existing session file (see [DETAILS.md](DETAILS.md#session-files-disk)): the running state, the
   K/V of every layer and the draft layer's K/V, the deepest checkpoint.
4. `RESTORE <file>` on the decode engine, which checks the file against its own model files and settings.
5. The request goes to the decode engine as usual. It finds the restored session as the start of the prompt, reads the last
   token, and makes the answer. The file is deleted.

If anything on the way fails (the prefill engine died, the file was refused, the card is out of memory), the decode engine reads
the prompt itself, as without the option, and the log says why. A prefill engine that died is started again in the background.

The figures the server reports for the request (prompt time, tokens read, tokens reused) count the prefill engine's work and
the two file steps, so a client sees one request.

## What must match

The session file is bound to everything that changes what its bytes mean. Both engines start from one config, so these match;
they would not if `prefill_args`/`decode_args` changed them:

- the model files (sampled by size and first and last MiB) and the engine version;
- the backend (CUDA or HIP): **no cross-backend restore**;
- `--kv`, `--max-context`, `--kv-resident`, `--mtp-window`, `STRATA_KV_ROT`, the resolved rope settings, a loaded control vector;
- the arithmetic switches that change the computed state (the engine lists them in the file's fingerprint).

These may differ between the two engines: the expert cache size, `--prefill` chunk, sampling, drafting depth, the expert tier.
That is the point of the two roles: the prefill engine can spend its VRAM on prompt buffers and the decode engine on experts.

## Limits

- One request at a time, as for a single engine. The prefill engine could read the next request's prompt while the decode engine
  answers the current one; this version does not overlap them.
- The decode engine holds one conversation. A hand-over replaces it (the engine's restore does not park the session it
  replaces). A request that continues the conversation the decode engine holds is read in place.
- No parallel slots (`--batch`), no layer split inside an engine, no pictures (a request with pictures goes to the decode
  engine alone).
- Needs RAM for the experts once (shared arena, Linux) or twice (otherwise), plus the session file while it exists:
  about 20 MiB per 1,000 tokens at the default `--kv int8` on the Qwen3.8 model.
- Exact token replay across a hand-over is as good as across any session restore: see [the check](#measured).

## NVIDIA with AMD, and other PCs

This version refuses nothing it cannot run: it runs on any two cards of one make that each hold the model. What the cases beyond
it need:

CROSS-VENDOR-PLACEHOLDER

Credit: the idea, the hand-built router and the first measurements are @MarcoTool's (issue #4000, repository
`MarcoTool/PDHyperMix-Strata-MI50`).
