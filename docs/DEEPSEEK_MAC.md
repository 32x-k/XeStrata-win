# DeepSeek-V4.1-Flash on a Mac

DeepSeek-V4.1-Flash runs on Apple Silicon Macs through its own Metal engine, `strata-dsv41`. It is a separate download
from the Qwen engine: the app and `strata launch` fetch it the first time you pick DeepSeek. This page says what it
needs, what it does on the machine it was measured on, and what it cannot do yet.

> **Short version:** a Mac with 64 GB of memory or more, macOS 15 or newer, and about 260 GB of free disk space on a fast
> drive. Text only (no pictures). The model does not fit in memory, so part of it is read from the SSD for every word.
> On an M5 Max with 128 GB, DeepSeek 2.00 answers at 15-19 tokens a second and reads a 3,000-token prompt in under 8 seconds.

## What it needs

| | |
|---|---|
| Mac | Apple Silicon (M1 or newer), macOS 15 or newer. Measured on an M5 Max with 128 GB. |
| Memory | 64 GB or more is a good fit. Less still starts, but most of the model is read from the SSD for every word, so it is much slower; setup says so and asks before it goes on. |
| Disk | 2.00 set: 257 GB. 2.25 set: 274 GB (the 111 GB n-gram table is shared). Put the model on the internal SSD or an NVMe drive, not a USB or network drive. |
| Pictures | Off. DeepSeek's picture reader has not been run on the Metal engine, so a Mac reads text only (Qwen on the Mac keeps its pictures). |

## Install it

In the app: Models, DeepSeek-V4.1-Flash, pick 2.00 bpw (the default) or 2.25 bpw, Download and prepare.

In the Terminal:

```sh
strata launch --family deepseek41 --model ist-das-2.00bpw --plan     # what it would do on this Mac
strata launch --family deepseek41 --model ist-das-2.00bpw
```

Setup downloads the set from Hugging Face (pinned revision, size and SHA-256 checked, resumable), builds the pack (a few
MB), downloads the DeepSeek engine package for macOS from the signed update feed, and runs a 16-word test question
before it starts. Files you already have: `--gguf-dir DIR`.

## What it does (M5 Max, 128 GB, 2026-10-25 and -26)

| | 2.00 bpw | 2.25 bpw |
|---|---|---|
| Answer speed, code prompt | 19.2 tokens/s | 14.4-15.8 |
| Answer speed, agent prompt | 19.7 | 12.4-12.6 |
| 3K prompt | 362-377 tokens/s | 380-427 |
| 32K prompt | 303 | 306-404 |
| Read from the SSD per word | 40 MB | 93 MB |
| Quality against the original model (KL, 4,995 positions) | 0.1709 | 0.1102 |

The 2.25 set is the better of the two (about 7% above the original model's perplexity, against about 15%), the 2.00 set is
faster and smaller. Both numbers come from the engine's own scoring run with a fixed expert placement
(`--profile-readonly`); a run that learns the placement as it goes moves the last digits.

How the memory is used: the engine keeps about 71 GB of the experts in memory on a 128 GB Mac (the rest of the memory is
left to macOS and its file cache; a Mac with less memory keeps less) and reads the remaining experts from the SSD in place.
A decode token is limited by those SSD reads and by a CPU lane that computes the experts the GPU does not hold, not by the
GPU.

## Speed setting

DeepSeek's Speed choice (Faster = 5 experts a word, Full quality = 6) works as on a PC. On a Mac the engine's own default
is Full quality: that is what was measured and gated.

## Limits

- Text only. A picture sent to a DeepSeek model on a Mac is refused.
- The drafter (DSpark) is off: it was exact but not faster on the Mac.
- Longer than 32K of context was not measured on a Mac.
- Smaller Macs: nothing under 128 GB was run with DeepSeek. The plan screen shows how much of the model your Mac keeps in
  memory and warns when it is under a quarter of it; that estimate is the engine's own rule, not a measurement.
- The 1.75 bpw set shares the 2.00 set's kernel but was not run on a Mac, so it is not offered there.

## Building and packaging the engine (developers)

```sh
bash packaging/mac/build_dsv41_engine.sh            # configure, build, stage, package and verify the macOS package
```

It needs the Xcode Command Line Tools, cmake and ninja (a pip venv is enough), Apple's metal-cpp and the llama.cpp tree that
setup pins (`METAL_CPP_DIR`, `STRATA_GGML_DIR`). The result is `pkg/engine/<version>-macos-arm64-metal/` with
`strata-dsv41`, the Metal sources it compiles at start-up and `engine-component.json`; the package is unsigned, the feed key
signs it (`packaging/UPDATES.md`). The Qwen Metal engine is not rebuilt by this: the DeepSeek runtime rules sit behind
`STRATA_DS4_RUNTIME`, compiled only into `strata_ds4`.

Diagnostics, all off by default: `DS4_CPU_LOG=1` (per-token phase times), `DS4_PROF=1` (per-kernel GPU time),
`DS4_GPU_TOTAL=1`, `DS4_ENC_TIMES=1`, `DS4_TLOG=1`, `DS4_TRACE_DISP=1`.
