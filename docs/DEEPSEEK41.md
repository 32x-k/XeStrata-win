# DeepSeek-V4.1-Flash

DeepSeek-V4.1-Flash is a large mixture-of-experts model: 40 layers, 384 experts in each (6 are used for every word), and a 203 GB
n-gram table next to them. Strata runs it with its own engine, `strata-dsv41` (`src/dsv41`, ported from Project Maya's
`maya-ds4`, MIT; the notice is in `third_party/project-maya`). The file is Unsloth's UD-Q8_K_XL, pinned in `setup.py`: nine
shards, 508.4 GB, the experts as MXFP4 and the rest in BF16.

This is for a computer with several graphics cards and a lot of RAM. It was run on four Radeon AI PRO R9700 (32 GB each) with
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

## Install

```
./setup.sh --family deepseek41                    # downloads 508 GB (resumable, checked against pinned sizes and SHA-256)
./setup.sh --family deepseek41 --gguf-dir DIR     # files you already have: the nine shards, or the repository folder with UD-Q8_K_XL inside
strata launch --model ud-q8_k_xl --plan           # where it would go on this computer, nothing changed
strata launch --model-path DIR --model ud-q8_k_xl # start from files you already have
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
