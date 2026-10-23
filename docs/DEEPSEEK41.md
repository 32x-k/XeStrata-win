# DeepSeek-V4.1-Flash

DeepSeek-V4.1-Flash is a large mixture-of-experts model: 40 layers, 384 experts in each (6 are used for every word), and a 203 GB
n-gram table next to them. Strata runs it with its own engine, `strata-dsv41` (`src/dsv41`, ported from Project Maya's
`maya-ds4`, MIT; the notice is in `third_party/project-maya`). The file is Unsloth's UD-Q8_K_XL, pinned in `setup.py`: nine
shards, 508.4 GB, the experts as MXFP4 and the rest in BF16. The two official 2-bit sets (below) are what a normal PC runs.

The 508 GB set is for a computer with several graphics cards and a lot of RAM. It was run on four Radeon AI PRO R9700 (32 GB each) with
251 GB of RAM. Strata does not refuse a smaller computer; it tells you what to expect first.

## Where it goes

| | size | where |
|---|---|---|
| routed experts (15,360 of 18.8 MB) | 288.8 GB | as many as fit in graphics memory; the rest are computed on the CPU, read in place from the model files |
| attention and other dense weights | 16.8 GB | graphics memory, one block of layers on each card (about 3.9 GB a card on four) |
| n-gram table (Engram) | 202.8 GB | the SSD. A few rows are read for each word it writes. It is not counted in RAM, but a slow disk slows every word, so keep the model files on an NVMe drive |

Measured on the four R9700 cards: 6,197 experts (116.5 GB) in graphics memory, 9,163 (172.3 GB) on the CPU, which needs about that
much free RAM to stay cached. `strata launch --plan` and `setup.py --check` work this out for your cards and RAM
(`strata_control/dsv41_fit.py`; it predicted 6,240 experts on the R9700 box).

With less memory than that, part of the CPU's experts are read from the SSD while the model answers. It runs, slowly. A card
that cannot hold its share of the dense weights (about 4 GB each on four cards, 16 GB on one) can make the engine stop at the start
with an out-of-memory message. Both are said before anything is downloaded.

## The official 2-bit versions (the ones for a normal PC)

Setup offers two official sets (helcig/tmp-oct26-2, pinned): **IST-DAS 2.00 bpw** (138 GB) and **IST-DAS 2.25 bpw** (155 GB). Both use the
same 111 GB n-gram table (downloaded once into `models/deepseek-v4.1-flash-shared`) and the 8 GB DSpark drafter: 257 GB of disk for
2.00, 274 GB for 2.25, 412 GB for both. Quality against the original model: perplexity +16% on 2.00, +8% on 2.25 (code and agent work
about +3.5% / +2.5% on 2.25).

Measured: an RTX 5070 (12 GB) with 64 GB of RAM and a 3.6 GB/s NVMe drive answers at about 4.7 tokens/s and reads prompts at 121
tokens/s (4K) to 231 tokens/s (8K, layer-major path); an RTX 3060 (12 GB) with 46 GB of RAM about 7.2 tokens/s; four R9700 about 31
tokens/s. Under about 40 GB of RAM it is untested and slow; setup says so and still installs.

Setup downloads with resume and SHA-256, packs the model with the plain recipe (`pack_gguf.py --model SET --shards ... --engram
engram-IQ4_NL.gguf --engram-meta <UD shard 1> --dspark-gguf dspark-...-Q8_0.gguf`; do not add `--skip-tensors` or `--dense-override`:
that packs without an error and gives garbage text), copies the shipped hot-expert profile (`data/dsv41/profile-2.00.bin` or `-2.25.bin`)
into the pack as `expert-profile.bin`, checks `strata-dsv41 --pack-info` (every file present) and answers a 16-token test question
(`tools/dsv41/smoke.py`, the text must name Paris). The start script passes `--dspark auto` (never `off`: `off` still loads the drafter
and its 7 GB of experts take the graphics memory); the RAM store sizes itself. Long prompts use the engine's fast chunked path on its own (no flag is written: the engine picks the chunk size for your cards).

## Install

```
./setup.sh --family deepseek41 --model IST-DAS-2.00bpw   # 257 GB: the official 2.00 set + Engram + drafter (resumable, SHA-256 checked)
./setup.sh --family deepseek41 --model IST-DAS-2.25bpw   # 274 GB (the Engram is shared with 2.00)
./setup.sh --family deepseek41 --model UD-Q8_K_XL        # Unsloth's 508 GB set, for hundreds of GB of RAM or several cards
./setup.sh --family deepseek41 --gguf-dir DIR            # files you already have: the shards (or the folder with 2.00bpw / 2.25bpw / UD-Q8_K_XL inside); the Engram and drafter are looked for beside them
strata launch --model ist-das-2.00bpw --plan             # where it would go on this computer, nothing changed
strata launch --model-path DIR --model ist-das-2.00bpw   # start from files you already have
```

Setup needs `strata-dsv41` next to the engine (or `STRATA_DSV41_EXE`) and says so before it downloads anything. Build it with
`cmake -DSTRATA_BUILD_DSV41=ON`. The default context is 8192 tokens, the size it was checked at; `--context` asks for more.
Loading takes a few minutes (the config sets `"engine_ready_s": 3600`).

## Through the API

The server speaks to the engine as it does for the other models, so `/v1/chat/completions`, `/v1/messages`, streaming, stop and
`/metrics` work as usual (the answer speed is the tokens the model produced over the engine's decode time).

- Thinking is on by default, as the model's template has it. `reasoning_effort: "none"` or `chat_template_kwargs.enable_thinking: false` turns it off.
  Effort words: `low` (50), `medium` (62), `high` (75, the default), `max` (100), or a whole number 1 to 100.
- Prompts are the GGUF's own chat template, the same bytes as DeepSeek's `encoding.py` on the cases checked. Project Maya's own template lost the
  tool associations and later system instructions (their issue #299); Strata does not use it.
- Tool calls: the model writes DSML (`<｜DSML｜ calls>` ... `<｜DSML｜ invoke name="...">` ... `<｜DSML｜ parameter name="..." string="true">`). The
  server turns them into OpenAI `tool_calls` and Anthropic `tool_use`, including `tool_choice`, streamed arguments and calls written inside the thinking.
  A value marked `string="true"` stays a string.
- Defaults when a request names none: temperature 1.0, top_p 1.0.
- Agents and sub-agents that take turns do not read their prompts again. When another request takes the model, the engine sets the
  conversation aside (about 17 MB per 6.5K tokens) and takes it back when the next prompt continues it; the system prompt and tools shared by
  an agent and its sub-agents are kept as a checkpoint too. Slots survive a restart, live in `ds4-slots` next to the run log, are capped at
  2 GB and are removed after a week. `STRATA_DS4_SLOT_KEEP=0` turns it off, `STRATA_DS4_SLOT_GB` changes the cap, `STRATA_DS4_AUTO_PIN=0`
  turns off the shared-start checkpoint. The answers are the same as a fresh read of the prompt.

Pictures and the DSpark drafter are in the engine (strata-dsv41 reads them from the unsloth mmproj and dspark files beside the model; the server side is not wired yet). Not in this version: several requests at once (`--parallel` is ignored with a note), tuning
(`--tune` is skipped with a note).

## Compressed experts (trellis), for testing

`strata-dsv41` reads the routed experts in the trellis types as well as the source MXFP4: TQK6 and TQK7 (agentionai's files), and the
experimental TQK5X / TQK4X (computed codebook) and TQK5T / TQK4T (table codebook), see `tools/trellis/README.md`. A layer may be converted
while the others stay MXFP4. On the GPU they run on the tensor cores of gfx12 (and in plain FMAs elsewhere), on the CPU lane on Strata's
AVX-512 / AVX2 kernels; the build needs `-DSTRATA_NATIVE_EXPERTS=ON` for the CPU lane.

```
# 1. the pack of the source model (also takes the DSpark and mmproj files that sit beside the model folder)
python tools/dsv41/pack_gguf.py --model DIR/UD-Q8_K_XL --out PACK
# 2. re-encode some layers (HIP encoder; one layer at K5 is 2.3 GB, the whole model 93 GB)
python tools/dsv41/tq_convert.py --pack PACK --out CONV --layers 1,7,14 --type K5      # K4 K5 K4T K5T K6 K7
# 3. the pack that reads them
python tools/dsv41/pack_gguf.py --model DIR/UD-Q8_K_XL --out PACK2 --tq-experts CONV
strata-dsv41 --pack PACK2 --score data/quant_quality/v1/set.ids.json --score-out RUN.tsv      # teacher-forced top-64 rows
```

The encode is plain error (no rotation, no activation weights). The tests are `ds4_tq_test` (device and CPU lane against a double
reference, needs one GPU) and `tqw_hip_test`.
