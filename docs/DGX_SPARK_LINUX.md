# Strata on the NVIDIA DGX Spark (Linux, arm64)

One desktop box with a GB10 GPU, 20 Arm CPU cores and 128 GB of memory shared by the GPU and the CPU. This page is the
arm64 Linux build of Strata from source: what works, how to build it, how to install a model and how the speed was
measured. It is experimental: one Spark, one driver, one toolkit.

## What works, and what does not

- **Works:** the CUDA engine (`strata`), Qwen3.8-Flash-Next IQ3_S and the NVFP4 GPTQ pack, 64K context, pictures off,
  the DeepSeek-V4.1 engine built from source (`strata-dsv41`).
- **Not checked here:** the picture reader (`strata-vision` builds, not exercised in these runs), 128K context,
  DeepSeek serving (the engine builds; no model was downloaded in this session), the app, the Terminal and the
  ready-made engine feed, which are x86_64 only.
- **CPU experts:** Strata's own CPU expert kernels are x86 AVX/AVX-512; the arm64 build leaves them out and a native
  pack's CPU experts run on ggml-cpu's NEON rows. A canonical Q2_0 pack (no `native_experts.txt`) still needs AVX-512
  and is refused at start.

## Build the engine

Needs an arm64 Linux with the CUDA 13.0 toolkit (`/usr/local/cuda`; DGX OS has it) and Python 3.10+ with venv.

```sh
python3 -m venv .venv-build
.venv-build/bin/pip install -r requirements.txt -r requirements-build.txt
PATH="$PWD/.venv-build/bin:/usr/local/cuda/bin:$PATH" PYTHON="$PWD/.venv-build/bin/python" \
  JOBS=20 sh spark/build_engine.sh
```

About 3 minutes on the Spark with 20 jobs (the first run also fetches llama.cpp at the pinned commit). It leaves
`out-spark/engine/` with `strata`, `libstrata_kq.so`, the CUDA libraries it needs and a `BUILD.json` that says
`linux-aarch64`, arch `121`.

`BUILD_VISION=0` skips the picture reader; `BUILD_DSV41=0` skips the DeepSeek engine; `ARCHS=121` is the default and
is what a Spark needs.

Put it where setup looks for an engine:

```sh
cp -r out-spark/engine ./engine
```

## Install a model

```sh
./setup.sh --setup --family qwen --model IQ3_S  --context 65536 --vision no --yes --no-start --data-dir ~/Strata-data
./setup.sh --setup --family nvfp4 --model NVFP4  --context 65536 --vision no --yes --no-start --data-dir ~/Strata-data
```

`--data-dir` must be a folder you can write (on a DGX Spark the default `../Strata-data` next to the checkout is under
`/home`, which `dev` cannot create; `~/Strata-data` works). Setup finds the GB10, reports it as unified memory, and
passes `--mmap-experts`: the experts are read from the model files in place, so the GPU's cache holds all 24,576 of them
instead of a second copy in RAM.

Start a model with its run script (`./run-iq3_s.sh`) or `./setup.sh`. `./bench.sh` runs the quick speed check.

## What the Spark does with the memory

| | IQ3_S | NVFP4 GPTQ |
|---|---|---|
| Expert bytes | 50.3 GB | 68.0 GB |
| Expert cache | 24,576 of 24,576 slots, 46.84 GiB | 24,576 of 24,576 slots, 63.29 GiB |
| n-gram table | 28.8 GB, locked in RAM (`--ple-io ram`) | 51.2 GB, read from the SSD (`direct`) |
| Context | 64K, 8-bit KV, 32,768 cells streamed | same |
| Free VRAM at start | ~1.0-1.2 GiB | ~1.2 GiB |

The NVFP4 table does not fit in RAM beside the page cache the 68 GB expert file needs: locking it measured 137 tok/s
on a 4K prompt and 26 tok/s decode against 1,252 and 39 without. Leave it on the SSD.

## Speeds (medians, one Spark, 26 October 2026)

Measured with `tools/dgx_bench.py`: fresh 4K / 16K / 32K prompts at temperature 0, 256-token replies, the engine's own
`timings`, three paired rounds against steddyman's 0.1.49.2 engine built on the same machine.

| Model | prompt tok/s (4K / 16K / 32K) | decode tok/s |
|---|---|---|
| Qwen3.8-Flash-Next IQ3_S, this build | 1,416 / 1,597 / 1,581 | 45.5 / 45.0 / 44.0 |
| Qwen3.8-Flash-Next IQ3_S, the fork | 1,237 / 1,331 / 1,402 | 44.4 / 43.4 / 43.1 |
| Qwen3.8-Flash-Next NVFP4, this build | see HANDOVER.md | see HANDOVER.md |
| Qwen3.8-Flash-Next NVFP4, the fork | see HANDOVER.md | see HANDOVER.md |

On IQ3_S that is +16% prompts over the fork and +3% decode. The prompt gain is `--prefill auto:32768` plus the fused
IQ prompt path (`STRATA_PF_FUSED=1`, opt-in on this card because the fused kernels round differently from MMQ).
Decode is GPU-work-bound on the 48-SM GB10 (109 kernel launches per token, ~99% GPU busy); the remaining decode work is
kernel work, not configuration - see the handover's backlog.

## Rules for this machine

- Don't shut it down or restart it without agreement.
- Never publish the SSH key or put it in a repository.
- Ask the machine's owner for extra rights or installs.
