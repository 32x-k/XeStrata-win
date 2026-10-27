# Strata on the NVIDIA DGX Spark (Linux, ARM64)

The DGX Spark is a small desktop box with a GB10 chip: a Blackwell GPU (compute capability 12.1, 48 SMs) and 20 Arm CPU cores that share
128 GB of memory. Strata 0.2.7.1 runs Qwen3.8-Flash-Next and DeepSeek-V4.1-Flash on it. This page says what works, how to install it, what
the Spark does with its memory, and how the speeds were measured. It was tested on one Spark (Ubuntu 24.04, driver 580.173, CUDA 13.0).
Nothing here is known to work on another ARM machine.

## What works, and what does not

- **Works:** Qwen3.8-Flash-Next IQ3_S and the NVFP4 GPTQ pack (64K context, every expert in the GPU's cache), DeepSeek-V4.1-Flash 2.00 bpw
  (a short chat), the Terminal (`strata launch`, `strata update`, `strata serve`) and the engine packages of the signed feed.
- **Not built for ARM:** the desktop app (no AppImage; use the Terminal), the HIP, SYCL and CPU-only engines, and the Q2_0 and Gyro packs,
  whose CPU kernels are x86-only (the engine refuses them at start). Picture reading is compiled in and was not tried on the Spark.
- **Not tested:** contexts above 128K, batches of more than four chats, a long soak. `strata doctor --live` lists the GB10 with no memory
  figure, because `nvidia-smi` reports none for it.

## Install with the Terminal

```sh
curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
strata launch
```

`install.sh` unpacks the Terminal into `~/.local/share/strata` (no administrator rights) and checks the release's signed feed. `strata launch`
fetches the `linux-aarch64` engine package, shows what each model needs, downloads the one you choose and starts it. Pick Qwen3.8-Flash-Next
IQ3_S (84 GB download) or the NVFP4 pack, or DeepSeek-V4.1-Flash 2.00 bpw (257 GB). The engine package needs glibc 2.39 or newer
(Ubuntu 24.04, DGX OS 7) and NVIDIA driver 580.65 or newer.

## Install from a source checkout

```sh
./setup.sh --setup --family qwen --model IQ3_S --context 65536 --vision no --yes --no-start --data-dir ~/Strata-data
./setup.sh --setup --family nvfp4 --model NVFP4 --context 65536 --vision no --yes --no-start --data-dir ~/Strata-data
```

`--data-dir` must be a folder you can write (the default, next to the checkout, may be under `/home`, which a normal user cannot create).
Setup finds the GB10, calls it unified memory, takes the engine from the feed (or compiles it with `spark/build_engine.sh` when the release
has none) and writes the run config with the settings measured on this part:

| Setting | Why |
|---|---|
| `--mmap-experts` | the experts are read from the model files in place, so the GPU's cache holds all 24,576 of them instead of a second copy in RAM |
| `--prefill auto:32768` | a 32,768-token prompt chunk; the automatic 8,192 measured 2 to 6% slower here |
| `--mtp-q4 all` | the draft layer as 4-bit copies: +3.5% decode, the answers do not change because the target checks every draft |
| `--mtp-hnorm stream` | one norm per hyper-connection stream in the draft layer: draft acceptance 0.60 to 0.63, +4.3% decode |

Setup prints `--draft-vocab en` as a tip and does not set it: it measured +3.1% decode on English prompts, and it would cut the Chinese,
Japanese and Korean drafts.

Start a model with its run script (`./run-iq3_s.sh`) or `./setup.sh`.

## Build the engine yourself

You need the CUDA 13.0 toolkit (`/usr/local/cuda`; DGX OS has it), cmake, ninja and Python 3.10 or newer.

```sh
python3 -m venv .venv-build
.venv-build/bin/pip install -r requirements.txt -r requirements-build.txt
PATH="$PWD/.venv-build/bin:/usr/local/cuda/bin:$PATH" PYTHON="$PWD/.venv-build/bin/python" JOBS=16 sh spark/build_engine.sh
cp -r out-spark/engine ./engine
```

About 3 minutes for the Qwen engine and 6 with the DeepSeek engine and the picture reader (the first run also fetches llama.cpp at the
pinned commit). `out-spark/engine/` holds `strata`, `strata-dsv41`, `strata-vision`, `libstrata_kq.so`, the CUDA libraries and a `BUILD.json`.
`BUILD_VISION=0` and `BUILD_DSV41=0` leave a part out. `packaging/build_cuda_linux_aarch64.sh` runs this build and makes the engine packages
for the feed ([packaging/README.md](../packaging/README.md)). A model on a Q2_0 pack cannot run: its CPU kernels need AVX-512.

## What the Spark does with its memory

The GPU's memory and the CPU's are the same 121.6 GB, so a model's experts are held once. With `--mmap-experts` there is one copy: the model
file's pages, which the GPU cache uses.

| | IQ3_S | NVFP4 GPTQ |
|---|---|---|
| Expert bytes | 50.3 GB | 68.0 GB |
| Expert cache | 24,576 of 24,576 slots, 46.84 GiB | 24,576 of 24,576 slots, 63.29 GiB |
| n-gram table | 28.8 GB, locked in RAM (`--ple-io ram`) | 51.2 GB, read from the SSD |
| Context | 64K, 8-bit KV, 32,768 cells streamed | same |
| Free memory at start | about 1 GiB | about 1.2 GiB |

The NVFP4 table does not fit in RAM beside the page cache the 68 GB expert file needs. Locking it measured 137 tok/s on a 4K prompt and 26 tok/s
decode, against 1,252 and 39 without. Leave it on the SSD (setup does).

DeepSeek-V4.1-Flash 2.00 bpw places 12,163 of its 15,744 experts in the GPU's share and runs the rest on the CPU lane from the model's files.
The engine sizes that from the memory Linux can still give back (`MemAvailable` less 6 GiB, `DS4_UMA_HEADROOM_GIB`), because the model files' page
cache is most of the memory `cudaMemGetInfo` calls used.

## Speeds

[[SPEEDS]]

## The four speed defaults of this part

On a GB10 the engine switches on four settings that were measured here, and prints them at every start. `STRATA_SM121_PERF=0` (or
`STRATA_DEFAULTS=026`, which gives back everything 0.2.7.1 changed) turns the whole set off; a switch you set yourself is kept.

| Switch | What it does | Measured on the Spark |
|---|---|---|
| `STRATA_PF_FUSED=1` | the fused IQ prompt kernels | IQ3_S prompt +10 to 27% |
| `STRATA_PREFILL_NVFP4=w4a4` | NVFP4 prompt matrix product with FP4 activations | NVFP4 prompt +16% |
| `STRATA_HC_UPMIX=1` | the hyper-connection read's up projection and mix in one kernel | decode +5.4% |
| `STRATA_SPEC_COST=1` | the draft window follows what a draft costs | decode +4.7% |

[[QUALITY]]

With `"parallel": 4` (the default) the set adds up: 32 clients at 2,000-token prompts read 41.3 tok/s with all four on and 35.7 with all off. At
`"parallel": 8` the two decode switches cost about 3.5% each; set `STRATA_HC_UPMIX=0 STRATA_SPEC_COST=0` there.

## What this does not cover

Decode on the 48-SM GB10 is limited by the GPU's work, about 930 kernel launches per token. No configuration switch measured here moved it more
than a few percent. A DFlash2 drafter is slower here than the MTP drafter (37 against 45 tok/s) and is not used. Teacher-forced quality over
long text and task suites was run only for the prompt path (above).
