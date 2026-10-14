<!--
SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# XeStrata

English | [日本語](README.ja.md)

## About this branch (Windows and AMD GPUs)

> This section was added on the `feature/windows-r9700-sycl` branch. Everything below the rule is the upstream XeStrata README, unchanged.

This branch ports XeStrata to Windows 11 with AMD GPUs.
It builds intel/llvm (DPC++) on Windows with the HIP target only, and compiles XeStrata in the free mode with it.
It keeps taking in upstream's releases; it now includes everything up to xe0.1.43.1.

### What it does

- Fixes for building intel/llvm v7.1.1 on Windows (90-92 in `third_party/main/intel-llvm/patches`, numbered from 90 so they do not collide with upstream's patches)
- oneMath's rocBLAS backend built on Windows (`third_party/main/oneMath/patches/windows-gnu-driver.patch`), for the dense matrix products of prefill
- Engine changes for Windows: the GPU memory budget (read from DXGI by LUID), the available RAM, signals and abnormal exits, file reading, and falling back to XeStrata's own kernels when rocBLAS has no files for the GPU, among others
- Speed: the native pack, which reads the GGUF's experts in place, and a RAM copy of the experts the GPU cache does not hold (`--resident-experts`)

### Supported GPUs

- On Windows, only AMD GPUs run (HIP). Only the Radeon AI PRO R9700 (gfx1201, 32 GB) has been tried.
- On Windows, it is not built for Intel Arc or NVIDIA GPUs.
- On Linux, it is the same as upstream XeStrata (Intel Arc, AMD and NVIDIA).

### What works on Windows

- Text generation with `strata.exe` (`--tokens-file`)
- The OpenAI-compatible API with `--serve` and `serve/server.py`
- The native pack (`tools/iq_pack.py`), speculative decoding with MTP, `--resident-experts`, and prefill on rocBLAS

These four things differ from upstream.

- The installer (`xestrata --setup`) and the deb and rpm packages are for Linux. On Windows, build by hand as below.
- On Windows, the RAM copy of the experts is page-locked by default only when HIP sees one GPU (and the RAM has room for it). With two or more GPUs seen by HIP (the processor's graphics included), the GPU can fail at an allocation after a large page-locked one. Choosing the GPU with `ONEAPI_DEVICE_SELECTOR=hip:N` makes XeStrata narrow `HIP_VISIBLE_DEVICES` to the same GPU.
- The adaptive cache is slow to swap experts, and `--adapt-every 0` runs faster.
- `--gpu-stages` does not work with the native pack.

### Building

The full steps are in [docs/DEVTOOLS.md](docs/DEVTOOLS.md), under "Windows (AMD GPUs, HIP)".

1. You need: Visual Studio 2022 Build Tools (the C++ workload), CMake, Ninja, Python, the ROCm SDK for Windows (AMD's TheRock Python wheel, with clang, the HIP runtime and rocBLAS), and the Linux device libraries of ROCm 7.2 (`amdgcn/bitcode`).
2. Get intel/llvm v7.1.1 and apply the patches in `third_party/main/intel-llvm/patches` in the order of their numbers.
3. In Visual Studio's `x64 Native Tools Command Prompt`, configure and build intel/llvm:

   ```bat
   python buildbot\configure.py -o build -t Release --hip --hip-platform AMD ^
     "--cmake-opt=-DCMAKE_INSTALL_PREFIX=<intel-llvm>/install" ^
     "--cmake-opt=-DUR_HIP_ROCM_DIR=<ROCm SDK>" "--cmake-opt=-DUR_HIP_LIB_DIR=<ROCm SDK>/lib" ^
     "--cmake-opt=-DLIBCLC_TARGETS_TO_BUILD=amdgcn--amdhsa" "--cmake-opt=-DSYCL_ENABLE_BACKENDS=hip" ^
     "--cmake-opt=-DLLVM_ENABLE_DIA_SDK=OFF" ^
     "--cmake-opt=-DCMAKE_C_FLAGS=/DWIN32 /D_WINDOWS /utf-8" ^
     "--cmake-opt=-DCMAKE_CXX_FLAGS=/DWIN32 /D_WINDOWS /GR /EHsc /utf-8"
   cmake --build build --target deploy-sycl-toolchain
   ```

4. Rewrite the device libraries' `wchar_size` for Windows:

   ```bat
   python tools\windows\device_libs_wchar.py <intel-llvm>\build\bin <ROCm 7.2>\amdgcn\bitcode <device-libs>
   ```

5. Configure and build XeStrata (`STRATA_HIP_ARCHS` is the GPU's architecture; gfx1201 for the R9700):

   ```bat
   cmake -S . -B build\free-win -G Ninja -DCMAKE_BUILD_TYPE=Release ^
     -DCMAKE_C_COMPILER=<intel-llvm>/install/bin/clang.exe ^
     -DCMAKE_CXX_COMPILER=<intel-llvm>/install/bin/clang++.exe ^
     -DSTRATA_LICENSE=free -DSTRATA_HIP_ARCHS=gfx1201 -DSTRATA_CUDA_ARCHS= -DSTRATA_ONEMKL=OFF ^
     -DSTRATA_ROCM_DEVICE_LIBS=<device-libs> -DSTRATA_ROCM_ROOT=<ROCm SDK> ^
     -DSTRATA_ENABLE_ONEMATH=ON -DCMAKE_PREFIX_PATH=<ROCm SDK> -DHIP_PATH=<ROCm SDK> ^
     -DSTRATA_BUILD_TESTS=OFF -DSTRATA_PACKAGE=OFF
   cmake --build build\free-win --target strata onemath_blas_rocblas
   ```

6. Make the native pack:

   ```bat
   python tools\iq_pack.py --gguf <model>-00001-of-00002.gguf --out <pack>
   ```

7. To run it, put intel/llvm's `install\bin`, the build's `bin` (oneMath's DLLs) and the ROCm SDK's `bin` on `PATH`, and choose the GPU with `ONEAPI_DEVICE_SELECTOR=hip:0`:

   ```bat
   build\free-win\strata.exe --pack <pack> --native <model>-00001-of-00002.gguf ^
     --ple-gguf <model>-00002-of-00002.gguf --expert-profile data\expert-profile.bin --expert-cache auto ^
     --resident-experts --prefill auto --spec 4 --spec-min-p 0.5 --mtp <the MTP draft layer> ^
     --adapt-every 0 --pcie-frac 0 --tokens-file <prompt.ids> --max-new 256 --stats
   ```

8. For the tests, configure another folder with `-DSTRATA_BUILD_TESTS=ON` (the arguments of step 5, another folder), build all of it and run CTest.
   `expert_parity` and `pool_test` read the AVX-512 pack (`experts.bin`) at `pack\full` in the repository root, so put a junction to that pack there.
   Run them with the `PATH` and `ONEAPI_DEVICE_SELECTOR=hip:0` of step 7.

   ```bat
   cmake --build build\free-win-test
   mklink /J pack\full <the AVX-512 pack>
   cd build\free-win-test && ctest --output-on-failure
   ```

### Current measurements (for reference)

Measured on 2026-10-14 on a PC with a Radeon AI PRO R9700, a Ryzen AI 9 HX 370, 28 GB of RAM and Windows 11.
The model is Qwen3.8-Flash-Next's Q2_0 and the arguments are those above (4 tokens of speculative decoding with MTP).
The comparison is the HIP build of Strata v0.1.43.1, XeStrata's upstream, on the same PC with the same model and prompts and Strata's default arguments.
Values are tok/s; those measured twice are averages.

| Prompt | Output | decode (this branch) | decode (Strata) | prefill (this branch) | prefill (Strata) |
| --- | --- | --- | --- | --- | --- |
| 50 tokens | 300 | 96 | 107 | 125 | 46 |
| 190 tokens | 128 | 90 | 92 | 228 | 101 |
| 2,301 tokens | 200 | 103 | 107 | 522 | 499 |
| 18,801 tokens | 200 | 101 | 107 | 976 | 948 |
| 40 tokens | 2,000 | 84 | 96 | 99 | 37 |

Sent 20 questions in a row, `--serve` answered all 20, at 93 tok/s overall (Strata: 95 tok/s).

decode is slower than Strata's mainly because computing on the CPU the experts the GPU cache does not hold takes longer: more of them go to the CPU than in Strata, while the GPU's verify and MTP's drafting take about as long as Strata's.

---

Run a 125-billion-parameter AI model on one Intel Arc, AMD Radeon or NVIDIA GPU and an ordinary PC.

XeStrata runs **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)**, a large AI model that
normally runs on a server, on your own PC.
It uses one Intel Arc, AMD Radeon (RDNA2 or later) or NVIDIA GPU and Linux; Ubuntu 26.04 and Fedora 44 have deb and rpm packages.
It is free software, and on an Intel or AMD GPU it also builds and runs with free software only.

> **Contents:** [About XeStrata](#about-xestrata) · [What you need](#what-you-need) ·
> [Choosing a model](#choosing-a-model) · [Install](#install) · [Using it](#using-it) · [When something goes wrong](#when-something-goes-wrong) ·
> [How it works](#how-it-works) · [For developers](#for-developers) · [All the details](docs/DETAILS.md)

## About XeStrata

Software written in CUDA runs only on NVIDIA's GPUs, which ties its users to one maker's hardware.
XeStrata aims to run on Intel, AMD and NVIDIA GPUs alike, from one code base written in SYCL, an open standard.

A port of [Strata](https://github.com/Niko1221/Strata) (for NVIDIA GPUs) to Intel GPUs.
Its GPU code is rewritten from CUDA to SYCL and Level Zero, so that it runs on Intel's GPUs.
The same code is compiled for AMD GPUs (the free and contrib-llvm builds) and NVIDIA GPUs (the contrib-llvm build) as well.
The models, the install, the app and the API are much the same as Strata's.

How it differs from Strata:

- **For Intel Arc,** and it runs on AMD and NVIDIA GPUs too.
- **One GPU, as a rule.** The layers can also be spread over two or more GPUs in the same PC ([MULTIGPU](docs/MULTIGPU.md)).
  The GPUs may be of different makers, Intel, AMD and NVIDIA (for example an Intel Arc Battlemage and an NVIDIA Ada; with NVIDIA, in the contrib-llvm build).
- **Linux only.** Windows is not supported. WSL 2 may work (pull requests welcome).
- **Runs with free software only, too** (the free build, free enough for Debian main; [made from the source](docs/BUILD.md#the-deb-and-rpm-packages)). The contrib-llvm packages use
  Intel's oneMKL and NVIDIA's cuBLAS, which are not free software, when they are installed ([Install](#install)).

## What you need

| | |
| --- | --- |
| GPU | Intel Arc (A series, B series), AMD Radeon (RDNA2 or later, [AMD GPUs](#amd-gpus)), or an NVIDIA GPU (Pascal or later; Pascal has no tensor cores and runs the slower paths). 12 GB of VRAM or more is recommended (less works, but slower). |
| CPU | x86-64 with AVX2. With AVX-512 (F, BW, VL, VNNI, VBMI) the CPU's part runs on AVX-512. |
| RAM | 32-62 GB, depending on the model's size ([Choosing a model](#choosing-a-model)). |
| Disk | About 60-110 GB for the model, and about 6 GB for the MTP layer. An SSD (NVMe) is strongly recommended. |
| OS | Linux (WSL 2 may work; pull requests welcome). The packages are for Ubuntu 26.04 and Fedora 44; on other distributions, [install from the source](docs/BUILD.md#installing-from-the-source). |
| BIOS | For a discrete GPU: Above 4G Decoding and Re-Size BAR on, CSM off. |

- **Matrix engines:** Intel's XMX, AMD's WMMA (RDNA3 or later) and NVIDIA's tensor cores are used; a GPU without
  them computes with integer dot-product instructions such as DP4a. Which one is chosen from what the GPU reports. Without matrix engines, prefill is slower,
  but it runs.
- **GPUs checked:** development and checks are done on an Arc Pro B70 (Xe2, 32 GB). On NVIDIA, an RTX 4070 and an
  RTX 3070 are checked; on AMD, an RX 9060 XT (RDNA4). On an Arc A380 only the arithmetic is checked, without running a model.
  Smaller cards are checked by limiting the memory and the XMX the B70 may use.
- **The processor's own graphics:** checked up to starting and prefill. Its memory is shared with the RAM,
  and most have no XMX, so it is slow.

### AMD GPUs

On AMD GPUs, prefill's dense matrix products run in the distribution's ROCm rocBLAS and hipBLASLt.
So XeStrata runs only on the GPUs the distribution's rocBLAS has code for.

| GPU | gfx | Ubuntu 26.04 | Fedora 44 |
| --- | --- | :-: | :-: |
| RX 6800 / 6900 | gfx1030 | yes | yes |
| RX 6700 | gfx1031 | no | yes |
| Ryzen integrated graphics (RDNA2) | gfx1035, gfx1036 | no | yes |
| RX 7900 | gfx1100 | yes | yes |
| RX 7800 / 7700 | gfx1101 | yes | yes |
| RX 7600 | gfx1102 | no | yes |
| Ryzen integrated graphics (780M and others) | gfx1103 | no | yes |
| Ryzen AI integrated graphics | gfx1150 | no | yes |
| Ryzen AI Max integrated graphics | gfx1151 | yes | yes |
| RX 9060 | gfx1200 | yes | yes |
| RX 9070 | gfx1201 | yes | yes |

- **On an RX 6700 or an RX 7600, use Fedora 44.** Ubuntu 26.04's rocBLAS has no code for these GPUs.
- The RX 6600 (gfx1032) and the RX 6500 / 6400 (gfx1034) do not run: neither distribution's rocBLAS has code for them.
- gfx1152 and gfx1153 (some Ryzen AI integrated graphics) do not run: Fedora 44's rocBLAS has code for them, but the
  intel/llvm 7.1.1 XeStrata builds with has no target for them.
- `xestrata --check` shows your GPU's gfx after its name.
- The table is that of ROCm 7.1's rocBLAS on both distributions. Only the RX 9060 XT (gfx1200) has been checked.

## Choosing a model

| Model | Parameters | RAM (about) | VRAM | Good for |
| --- | --- | ---: | ---: | --- |
| **[Qwen3.8-Flash-Next](#qwen38-flash-next)** | 125B | 32-62 GB | 12 GB or more recommended | the smartest; for a PC with a lot of RAM |
| **[Ornith-1.5](#ornith-15-beta)** (BETA) | 35B (3B a token) | 22-80 GB | from 4 GB | a PC with less RAM |
| **[llm-jp-4.1](#llm-jp-41-beta)** (BETA) | 32B (3B a token) | 30-73 GB | from 4 GB | a thinking model for Japanese and English |

The two BETA models do not yet take images, two or more GPUs, or Intel's integrated GPUs
([details](docs/DETAILS.md#or-models-other-than-qwen38-flash-next-ornith-15-llm-jp-41)).

Not sure? On a PC with 48 GB of RAM or more, take Qwen3.8-Flash-Next's **IQ2_XS**.
With less RAM than that, Qwen3.8-Flash-Next's Coder (32 GB) or Ornith-1.5 is the one to choose.
To add another model later, run `xestrata --setup`.

### Qwen3.8-Flash-Next

**The version and the size** (the size: the same model, compressed more or less):

| Version | Size | Download | RAM (about) | Speed | Quality |
| --- | --- | ---: | ---: | --- | --- |
| the original, Swift 1.5 | **Q2_0** | 66 GB | 48 GB | fastest | good |
| the original, Swift 1.5 | **IQ2_XS** | 68 GB | 48 GB | fast | better (**recommended**) |
| the original, Swift 1.5 | **IQ3_XXS** | 76 GB | 60 GB | slower | great |
| the original, Swift 1.5 | **IQ3_S** | 84 GB | 62 GB | slow | as the original model |
| Coder | **IQ1_M** | 58 GB | 32 GB | fast | good at code, weaker elsewhere |
| Unsloth | **UD-IQ4_XS** | 94 GB | 48 GB | slow with less than about 80 GB of RAM | between IQ3_S and UD-Q4_K_XL |
| Unsloth (experimental) | **UD-Q4_K_XL** | 111 GB | 48 GB | slowest | best |

All of the model's experts ([How it works](#how-it-works)) are kept in RAM. With less RAM than that but a GPU with a
lot of VRAM, it runs in the **low-RAM mode**: the experts are read from the model files instead of copied into RAM
([details](docs/DETAILS.md#less-ram-than-the-model-the-low-ram-mode)).
The two Unsloth files read the experts that do not fit the RAM from the SSD while they answer.

**The version:**

- **Qwen3.8-Flash-Next:** the original.
- **[Swift 1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF):** UkisAI's fine-tune. It
  thinks for a shorter time before it answers, so the answer comes sooner, at about the same quality. Its own license
  applies.
- **[Coder](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF):** ISTA-DASLab's version for
  code. Of each layer's 512 experts it keeps the 256 that code uses. With half the experts it is weaker outside code
  and in languages other than English.
- **[Unsloth's UD-IQ4_XS](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF):** a ~4-bit i-quant of the
  original. Its experts take 59.5 GB; with about 80 GB of RAM or more, all of them stay in RAM
  ([details](docs/DETAILS.md#or-unsloths-ud-iq4_xs)).
- **[Unsloth's UD-Q4_K_XL](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)** (experimental): a 4-bit file
  of the original. Its experts take 77 GB, and it is slower than the 2-3-bit sizes
  ([details](docs/DETAILS.md#or-unsloths-ud-q4_k_xl-experimental)).

OrcaRouter's Flash-Next Uncensored IQ3_XXS is not in setup's menu.
How to convert it is in [docs/ORCA.md](docs/ORCA.md).

### Ornith-1.5 (BETA)

[Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF) (Ornith AI, MIT): a Qwen3.5 MoE, with a
context of up to 256K. Its official files (Q4_K_M to BF16) draft tokens with the model's MTP layer, which makes the
decode faster.

| Size | Download | RAM (about) | VRAM |
| --- | ---: | ---: | ---: |
| Q4_K_M | 21.7 GB | 30 GB | from 4 GB |
| Q5_K_M | 25.3 GB | 34 GB | from 6 GB |
| Q6_K | 29.2 GB | 38 GB | from 6 GB |
| Q8_0 | 37.8 GB | 46 GB | from 6 GB |
| BF16 | 71.1 GB | 80 GB | from 8 GB |
| GSQ-RCO-3BIT | 13.0 GB | 22 GB | from 4 GB |
| GSQ-RCO-3.5BIT | 15.2 GB | 24 GB | from 6 GB |

The two GSQ-RCO files are pfeifferj's [community reproduction](https://huggingface.co/pfeifferj/Ornith-1.5-35B-A3B-GSQ-RCO-GGUF)
and have no MTP layer.

### llm-jp-4.1 (BETA)

[llm-jp-4.1-32b-a3b-thinking](https://huggingface.co/llm-jp/llm-jp-4.1-32b-a3b-thinking-gguf) (LLM-jp, Apache-2.0):
a thinking model for Japanese and English, with a context of up to 256K.

| Size | Download | RAM (about) | VRAM |
| --- | ---: | ---: | ---: |
| Q4_K_M | 21.5 GB | 30 GB | from 4 GB |
| BF16 | 64.3 GB | 73 GB | from 8 GB |

For the two BETA models, the RAM is about the download plus 8 GB. With less, the experts are read from the model
file instead of loaded into RAM, which is slower.

## Install

1. Pick the package for your GPU from the table below and download it from [GitHub's Releases](https://github.com/MistVVK/XeStrata/releases) (with `SHA256SUMS`).
1. Check it: `sha256sum -c SHA256SUMS --ignore-missing`.
1. Install it: `sudo apt install ./xestrata-contrib-cuda13.1_*.deb` (Ubuntu) or `sudo dnf install ./xestrata-contrib-cuda13.4-*.rpm` (Fedora).
1. Run `xestrata --setup` in a terminal. If a GPU library or the render group is missing, it prints the command that installs it and stops; run that command, then `xestrata --setup` again.

> [!NOTE]
> Without an NVIDIA GPU, either CUDA version works the same.

| GPU | Ubuntu 26.04 | Fedora 44 |
| --- | --- | --- |
| **Recommended**: Intel, AMD (RDNA2 or later), NVIDIA (Turing or later) | `xestrata-contrib-cuda13.1` | `xestrata-contrib-cuda13.4` |
| Intel, AMD (RDNA2 or later), NVIDIA Pascal and Volta (P100, P40, GTX 10 series, V100 and others) | `xestrata-contrib-cuda12.4` | none ([from the source](docs/BUILD.md#installing-from-the-source)) |

- One package is installed at a time. The CUDA 13 packages have no code for Pascal and Volta, and NVIDIA's drivers for them end with the 580 series (CUDA 12.4 works with it).
- The package with free software only (`xestrata-free`, Intel and AMD GPUs) is not on Releases. Make it as [docs/BUILD.md](docs/BUILD.md#the-deb-and-rpm-packages) describes (`tools/package/build.sh ubuntu26.04 free`, for example).

In `xestrata --setup`'s questions, pressing Enter takes the recommended choice.

- **The model and its size:** see [Choosing a model](#choosing-a-model).
- **The context:** how much text it can keep in mind at once. It suggests one for your GPU.
- **Images:** whether it should also read pictures.
- **Experimental speed projection:** an experimental feature that changes how the model answers, off by default.
  Read [what it does](docs/DETAILS.md#experimental-speed-projection-experimental-off-by-default) before you turn it on.

`xestrata` checks the GPU, the RAM and the CPU, downloads the model and starts it. Your browser opens
`http://127.0.0.1:8095`.

- **Time:** the first time, the download (60-110 GB) takes a while. If you stop it, the next run continues where it
  stopped. The driver compiles the GPU code the first time it is used, so it ends by running the model once to get
  that done.
- **The PC is slow while the model starts:** it reads 23-50 GB into RAM. The first start takes longest, and the PC
  can respond slowly for 1-3 minutes. Don't close the window; wait.
- **Where things go:** the model files in `~/.local/share/xestrata` (`--data-dir` chooses another place), the settings
  in `~/.config/xestrata`, the logs in `~/.local/state/xestrata`.

**Next time**, run `xestrata`: it starts right away. Close its window to stop the model.

**Running it while you are logged in:** `systemctl --user enable --now xestrata` keeps the model set up last running
(no browser opens). Set it up with `xestrata` once first.

**Updating:** when a new release is out, XeStrata says so in its window and on the chat page's About tab (it checks
at the start and every six hours while it runs; `xestrata --check-update` checks now). Download the new package from
[Releases](https://github.com/MistVVK/XeStrata/releases) and install it as before, with `sudo apt install ./<file>.deb`
or `sudo dnf install ./<file>.rpm`. The model's settings are updated at the next start.

**Fitting it to your PC:** `xestrata --calibrate` measures some of the engine's settings on this PC and keeps the
fastest (about 5-10 minutes).

**Measuring its speed:** `xestrata --bench` measures the installed IQ3_XXS's speed on this PC (another model: `--model IQ2_XS`, for example) and where the time goes,
and writes a record to the data folder's `records/bench-<time>/` (about 40 minutes; stop a running xestrata first).
Each case runs twice; `--bench-rounds 3`, for example, changes that (about 20 minutes more a round).
`--bench-verbose` also takes each kernel's time and the GPU's temperature. With `--original-strata v0.1.41` it fetches that tag of
upstream Strata with git and measures the two in turns on the same model. The record is not sent anywhere: read
`summary.md` before you paste it into a report.
What it measures and what the record holds: [docs/BENCH.md](docs/BENCH.md).

**Changing the package:** on Ubuntu, `sudo apt install` the other one and it replaces the first.
On Fedora, name the installed one and the new one, for example `sudo dnf swap xestrata-contrib-cuda13.4 ./xestrata-free-*.rpm`.

**Removing it:** delete the model files, settings and logs with `xestrata --remove-data` (it shows their sizes and asks),
then remove the packages.

```bash
xestrata --remove-data
sudo apt purge xestrata-contrib-cuda13.1     # Ubuntu (the package you installed)
sudo dnf remove xestrata-contrib-cuda13.4    # Fedora (the same)
```

Removed without `xestrata --remove-data`, the packages leave the model files and settings in your home folder.

Installing from the source with `./setup.sh` is in [docs/BUILD.md](docs/BUILD.md#installing-from-the-source).

## Using it

- **In the browser:** `http://127.0.0.1:8095` (it opens by itself when the model starts): **Chat**, the **Monitor**
  of the model and the GPU, CPU and RAM, and **About** with the settings and addresses.
- **Chat in the terminal:** `xestrata chat`
- **Apps and coding agents:** add it as an "OpenAI-compatible" provider with the base URL
  **`http://127.0.0.1:8095/v1`**. Any API key and model name work. Apps that use Anthropic's API:
  `http://127.0.0.1:8095/v1/messages`.
- **Thinking:** the model thinks before it answers. Choose **Off, Low, Medium or High** in the chat page's menu, with
  `/think low` in `chat.py`, or with your app's "reasoning effort" setting. Off is fastest; High suits hard questions.
- **Pictures:** in the chat page, click **Picture**; in `chat.py`, type `/image <path>`; in apps, attach them.
- **From a phone or another PC:** run `xestrata --host 0.0.0.0 --api-key <secret>` and open the address
  the server window prints ([details](docs/DETAILS.md#streaming-and-connecting)).

**Good to know:** it answers one request at a time (several at once is opt-in: `"parallel": 2`, [BATCHING](docs/BATCHING.md)). The first message of a conversation is read in full; after that
it keeps the conversation and reads only what is new, so follow-ups start right away.

## When something goes wrong

**My PC froze or got very slow the first time it started.**
That is common while it starts, most of all the first time: it is loading the model into RAM and working out how many
experts the GPU holds. Don't close the window; wait. Still frozen after 10 minutes? Restart the PC, close other
programs (browsers above all) and try again. If it keeps happening, pick a smaller size (Q2_0 or IQ2_XS).

**It stopped while downloading or installing.**
Run `xestrata` again. It continues where it stopped.

**It says it cannot find or use the GPU.**
Check these three things:

- Check that Intel's GPU runtime (its Level Zero driver), for an AMD GPU the ROCm libraries (HIP, rocBLAS, hipBLASLt),
  or for an NVIDIA GPU its driver (whether `nvidia-smi` sees it), is installed. NVIDIA GPUs run with the contrib-llvm
  packages only. For an AMD GPU, check also that it is one [XeStrata runs on](#amd-gpus).
- Check that your user can open the GPU's device (`/dev/dri/renderD*`, and `/dev/kfd` for an AMD GPU; the `render`
  group).
- If the OS does not see a discrete GPU, check Above 4G Decoding and Re-Size BAR in the BIOS.

**It says XMX cannot be used.**
The current compiler cannot use that GPU's matrix engines. Choose one of the options setup offers. Without XMX it
still runs; prefill takes about 1.6 times as long.

**A VRAM allocation was refused at start (`alloc_device: ... refused`).**
This happens now and then on the B70. Starting again usually works.

**It says port 8095 is already in use.**
XeStrata is already running: look for its window. If another program uses the port, choose another one, for example
`xestrata --port 8081`.

**It is very slow and the disk light stays on.**
The RAM is not enough. Close other programs, or pick a smaller size (Q2_0 or IQ2_XS).

**An answer stopped with "the engine stopped unexpectedly".**
Usually too little RAM (Linux stops the engine). Send your message again: the engine starts again by itself. If it
keeps happening, close other programs or pick a smaller size.

**It says the prompt exceeds the context.**
The conversation is longer than the context you chose. Start a new conversation, or run `xestrata --setup` and
choose a longer context.

**Still stuck?**
See the [full troubleshooting table](docs/DETAILS.md#troubleshooting). If that does not help, open an
[issue](https://github.com/MistVVK/XeStrata/issues) and attach `~/.local/state/xestrata/xestrata-<model>.log`.

## How it works

Models like this one normally run on servers with hundreds of gigabytes of GPU memory. XeStrata fits it on one GPU by
**sharing the work across the whole PC**.

- **The model is a team of 24,576 small specialists ("experts"),** and each word needs only 10 of them. So they do not
  all have to be on the GPU at once.
- **The GPU** does the work every word needs, and holds the few thousand experts used most often. It keeps learning
  which ones those are while you use it.
- **The RAM** holds every expert. When a word needs one the GPU does not have, **the CPU** computes it, at the same
  time as the GPU, so neither waits for the other.
- **The SSD** holds a large table, of which only a few rows are read per word.
- **Guess, then check:** a small helper built into the model guesses the next few words, and the big model checks
  them all at once. It keeps the ones that are right and writes the next word itself. The big model always decides,
  so the quality of the answer does not change.
- **Long texts are read in large pieces** (up to 32,768 tokens at a time).

Matrix products run on XMX on Intel GPUs, on WMMA on AMD GPUs (RDNA3 or later) and on tensor cores on NVIDIA GPUs;
a GPU without matrix engines uses integer dot-product instructions such as DP4a.
Every part is explained in [docs/DETAILS.md](docs/DETAILS.md#how-it-works), and how it is done on Intel GPUs and
what has been checked in [docs/XE.md](docs/XE.md).

## For developers

- **Building:** installing from the source, the three build modes (free, contrib-llvm, contrib-icpx), what setup does and the
  packages it needs, and how the deb and rpm packages are made are in [docs/BUILD.md](docs/BUILD.md).
  The development tools (lints, Intel SDE, GPU profilers): [docs/DEVTOOLS.md](docs/DEVTOOLS.md).
- **Measuring the speed:** `xestrata --bench`, the upstream merges' measurements and the release table are in [docs/BENCH.md](docs/BENCH.md).
- **Rules:** [AGENTS.md](AGENTS.md) covers them: independence from the hardware (code paths chosen from what the GPU
  and CPU report), the lints (`tools/lint/run.sh`), the tests, and the license notices.
- **What goes in:** whoever wrote it (the maintainer's commits, pull requests, ports from upstream), a change goes in only when
  built and tested with intel/llvm in both the free and contrib-llvm modes, and depending on the hardware as little as it can
  (no CUDA or other maker's API relied on, no value measured on one card hard-coded). Others do not go in as they are: they
  are a reference, written again as XeStrata's code ([AGENTS.md](AGENTS.md#what-goes-in)).
  A pull request says which modes it was built in (free, contrib-llvm, contrib-icpx) and on which GPUs and CPUs it ran,
  and marks what was not checked `unverified`.
- **Upstream integration:** the changes of Strata up to 0.1.42 are carried into Xe, except the Windows
  implementation. Of the paths meant to be faster, those measured faster on the B70, the RTX 4070 or the RX 9060 XT
  are carried ([docs/XE.md](docs/XE.md#integration-through-strata-0142)).
- **The README and docs:** the Japanese ones (README.ja.md, docs/*.ja.md) are written first, and the English ones (README.md, docs/*.md) are their translations.

## Credits

- **The original software:** [Strata](https://github.com/Niko1221/Strata), by Niko1221 and the Strata contributors.
- **The models:**
   - [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), by the Qwen team;
   - compressed versions and the [Coder](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF) by
     [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF);
   - [Swift 1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF), by UkisAI;
   - UD-IQ4_XS and UD-Q4_K_XL, by [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF).

  Each model's own license applies to its files.
- **Parts it uses:**
   - parts of [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) (MIT);
   - [oneMath](https://github.com/uxlfoundation/oneMath) (Apache-2.0): the dense matrix products (the contrib-llvm and
     contrib-icpx modes, and the free mode for AMD GPUs), which CMake fetches from XeStrata's fork with XeStrata's
     changes ([MistVVK/oneMath](https://github.com/MistVVK/oneMath), under the same license);
   - [intel/llvm](https://github.com/intel/llvm)'s DPC++ (Apache-2.0 WITH LLVM-exception): the free and contrib-llvm
     modes' compiler, built with XeStrata's fixes (`third_party/main/intel-llvm/patches/`, under the same license);
     the packages carry its SYCL runtime;
   - the web app's font, [Outfit](https://github.com/Outfitio/Outfit-Fonts) (SIL Open Font License 1.1).
- **Ideas from:**
   - [Splash](https://github.com/incoai/splash), by Inco AI and the Splash contributors;
   - [ninfer](https://github.com/Neroued/ninfer), by Neroued;
   - [HyperQwen](https://github.com/syv-ai/HyperQwen), by syv.ai and the HyperQwen contributors;
   - [Project Maya](https://github.com/mw00/project-maya), by mw00 and the Project Maya contributors (how many of the
     experts missing from VRAM the CPU computes and how many are read over PCIe);
   - [nextsycl](https://github.com/maxfridbe/nextsycl), by maxfridbe (which kernels the decode of the Qwen3.5 family's
     models merges).
- More in [docs/DETAILS.md](docs/DETAILS.md#credits-and-licenses).

## License

XeStrata (copyright MistVVK and the XeStrata contributors) is free software; the whole of it is under the
[GNU Lesser General Public License, version 3 or later](COPYING.LESSER) (`COPYING.LESSER`, with `COPYING`).

- **Linking with the vendors' math libraries:** what is linked with a math or compute library a GPU or CPU maker
  publishes for its processors, and with the runtime libraries it needs (oneMKL and the runtime libraries of Intel's
  oneAPI compilers, cuBLAS and the CUDA runtime and driver, for example), may be distributed as well: an additional
  permission under section 7 of the GPL, version 3, worded in [NOTICE](NOTICE).
- **Code from Strata and ggml:** the code that comes from Strata and from ggml / llama.cpp, both MIT (the kernels
  transcribed into `src/` included), is part of XeStrata under the LGPL; as MIT asks, their copyright and permission
  notices are kept in [NOTICE](NOTICE).
- **Exceptions:** these, in `third_party/`, stay under their own licenses, one folder per project with its license text.
   - XeStrata's patches to intel/llvm (`third_party/main/intel-llvm/`): Apache-2.0 WITH LLVM-exception.
   - XeStrata's changes to oneMath (its fork, `third_party/main/oneMath/`): Apache-2.0.
   - ggml's `ggml-common.h` (`third_party/main/ggml/`, an unmodified copy): MIT.
   - The app's font Outfit (`third_party/main/outfit/`): SIL Open Font License 1.1.
   - `third_party/nonfree/`: what is not free software. The original model's chat template and the experimental
     speed projection's vector, both under the Qwen Community License 1.0.
- **Without `third_party/nonfree/`,** XeStrata still builds and runs. The packages leave it out.
- **What else the packages carry:** intel/llvm's SYCL runtime (Apache-2.0 WITH LLVM-exception), llama.cpp's gguf-py
  (MIT) and oneMath (Apache-2.0), each with its license text.
  Nothing that is not free software (oneMKL, cuBLAS, NVIDIA's driver) is in them.
- **The models** are not part of this repository; each model's own license applies to its files.
