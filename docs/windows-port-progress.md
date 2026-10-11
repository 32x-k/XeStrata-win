<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# Windows port progress

[日本語](windows-port-progress.ja.md) | English

This document records the work to run XeStrata's SYCL engine on Windows 11 and an AMD Radeon AI PRO R9700 (gfx1201). An item is not complete until it has been verified on the actual machine.

## Completion criteria

- [x] Build a native Windows 11 executable
- [x] Detect the R9700 (gfx1201) through the SYCL runtime
- [x] Run a SYCL kernel on the R9700
- [x] Load a real model in XeStrata
- [x] Complete prefill successfully
- [x] Complete decode successfully
- [x] Generate meaningful text
- [x] Confirm GPU use in the execution log

## 2026-10-11: toolchain check

The repository revision is `e32b8b0594095352365db0e140a68d5bf56d8b0f`. The worktree was unchanged at the start of this work.

The following were verified on the machine:

- Visual Studio Build Tools 2022 17.14, MSVC v143 and Windows SDK 10.0.28000.0
- CMake 4.4 and Ninja
- TheRock ROCm 7.1 series in Strata's private venv (AMD clang 24.0.0git)
- `hipInfo.exe` reports the R9700 as gfx1201 with 31.86 GiB and wave size 32
- The same runtime also reports the integrated Radeon 890M (gfx1150). The initial port explicitly selects the R9700 from enumeration and capabilities, not a hard-coded product path

AdaptiveCpp develop nightly `25.10.0+git.0ea863a.20261009` (LLVM 20.1.8) was unpacked under `.tools/`. That distribution did not contain the Windows HIP runtime backend. `rt-backend-hip.dll` was built from the same AdaptiveCpp revision with this configuration:

- ROCm: `<ROCM_SDK_ROOT>`
- `WITH_ROCM_BACKEND=ON`
- `ACPP_COMPILER_FEATURE_PROFILE=none`
- `-Wno-c++11-narrowing` for this external-tool build only, to accommodate the type difference between ROCm 7.1's `hipError_t` and AdaptiveCpp's error-code initializer

With that backend loaded, `acpp-info.exe` verified the following through AdaptiveCpp's SYCL HIP backend on the machine:

- device 0: `AMD Radeon AI PRO R9700`, architecture `gfx1201`
- 32 compute units, just under 32,768 MiB of global memory and 64 KiB of local memory
- subgroup size 32
- device, host and shared USM
- execution timestamps

This proves device enumeration, not kernel execution.

## 2026-10-11: minimal SYCL kernel execution on the R9700

The nightly's LLVM 20 revision, `87f0227cb60147a26a1eeb4fb06e3b505e9c7261`, was first built with `X86;AMDGPU` and AdaptiveCpp linked into it. That clang has the AMDGPU target, but TheRock ROCm 7.1's device libraries are LLVM 24 bitcode. Reading the explicitly specified `ocml.bc` stopped with `Producer: 'LLVM24.0.0git' Reader: 'LLVM 20.1.8'`.

The ROCm LLVM revision used by TheRock, `4f43f4746ede4cc49ab4129f473dbd54e0d9db4f` (LLVM 24.0.0git), was then built with `X86;AMDGPU`, AdaptiveCpp revision `0ea863a`, and the Windows HIP backend. AdaptiveCpp marks LLVM 24 as outside its supported range, so compatibility fixes for the moved LLVM plugin header and pass API, a clang preprocessor include, and stricter inline-attribute checking were applied only to the external source under `.tools/`.

`.tools/sycl-smoke.cpp` was compiled for `hip:gfx1201` and run on the physical R9700. The test exited with status zero after verifying all of the following:

- selecting the R9700 from SYCL device enumeration and creating its execution queue
- allocating device USM for 4,096 floats
- a host-to-device transfer
- a `parallel_for` that updates all 4,096 elements
- upload, kernel and download event dependencies, synchronized with `wait_and_throw()`
- a device-to-host transfer and validation of every result
- execution device `AMD Radeon AI PRO R9700`, global memory `34208743424` bytes, and subgroup size 32

The final output was `USM transfer/kernel/event verification passed`. The toolchain blocker for minimal SYCL execution is resolved. The next step is the Windows/AdaptiveCpp build port of XeStrata itself.

## 2026-10-11: engine build and real-model loading

`build/windows-acpp/strata.exe` compiled and linked successfully, and `--help` exited with status zero. This initial configuration disables oneMath, native experts, tests and packaging. Windows file I/O, allocation and CPU-affinity paths were added, along with alternatives to the oneAPI extensions missing from AdaptiveCpp. Matrix products use existing DP4a or scalar paths.

The first run terminated before GPU selection. After adding exception reporting, it became clear that AdaptiveCpp always returns false for `aspect::fp16` because its support is partial, causing XeStrata to reject the devices. The AdaptiveCpp path now validates the half conversions and multiplication it uses in a real kernel before selecting a device. This check passed on the R9700.

The run used existing data in `<MODEL_ROOT>/Q2_0/` and `packs/q2_0/`; no model data was downloaded. The execution log confirmed:

- execution GPU `AMD Radeon AI PRO R9700`, 32 compute units and 31.9 GiB
- loading 4,630 MiB of dense weights
- successful loading of every layer after fixing truncation of file offsets above 2 GiB
- opening the PLE IQ4_NL table (320,001,536 rows)
- opening `experts.bin` through Windows file mapping
- starting the CPU expert pool

This run stopped at the safety check explaining that `--no-capture` does not process CPU routed experts. Bypassing it with `--no-pool` changes the model, so that route will not be used. The next step is a standard queue/event execution path that includes the CPU expert pool. Successful prefill, decode and text generation have not been confirmed.

The build still reports Windows CRT deprecation and subgroup-attribute warnings, among others. Linux free and contrib-llvm, NVIDIA, smaller-configuration regression checks and lint have not been run.

## 2026-10-11: Q2_0 prefill, decode and generation completed

A graph-free `session_token_pool()` path was added. It transfers each layer's routing and input from device to host, waits for its event, and runs the CPU expert pool. It uploads that layer's expert output and updates the residual before advancing to the next layer. Embedding, PLE, layers and head run on the same in-order queue. Neither routed experts nor PLE were omitted.

The existing Q2_0 model's tokenizer encoded this conversation into 13 tokens:

```text
<|im_start|>user
Say hello in English.<|im_end|>
<|im_start|>assistant
```

Prefill and decode completed on the physical R9700, generating reasoning followed by a `Hello!` response, `<|im_end|>` and `<|endoftext|>`. The process exited with status zero. All 41 generated tokens matched across three independent runs. In the final-build log, prefill processed 12 tokens in 2,732.6 ms and decode processed 41 tokens in 9,027.6 ms. The last input token is counted in the first generated token's measurement. This prefill uses the token-by-token path; it does not establish native-pack batched-prefill support.

The CPU expert pool used three workers plus the host thread, reading 25,440 expert blobs across 48 layers at 53 positions. The second run also set `STRATA_VRAM_LIMIT_MIB=8192 STRATA_MAX_ALLOC_MIB=4096 STRATA_NO_XMX=1`, and the log reported 8.0 GiB of GPU memory. However, the current device allocator does not enforce the simulated largest-allocation limit, and the dense arena exceeds 4 GiB. This run does not establish support for a card whose largest allocation is 4 GiB.

Windows GPU Engine counters also recorded Copy and Compute activity for the `strata.exe` process ID. There were 14 nonzero snapshots, including Compute values of 58–76%. Together with the SYCL execution-device log, this confirms that generation was not CPU-only.

Local evidence is stored in:

- `build/windows-acpp/model-hello-final.txt`: final-build generation log
- `build/windows-acpp/model-hello-complete-1-decoded.txt`: tokenizer-decoded response
- `build/windows-acpp/model-hello-small.txt`: run with the reported memory limited to 8 GiB
- `build/windows-acpp/model-hello-small-gpu.txt`: process-specific GPU counters

The host syntax check for `session.cpp` passed with `-Wall -Wextra -Werror`, as did `git diff --check`. `tools/lint/run.sh` was invoked but exited with status one because the required lint tools are absent from this environment. In particular, the script labels failures from a missing gitleaks executable as “possible secret”; that output alone does not establish a secret finding. Linux free and contrib-llvm, NVIDIA, integrated-GPU, ASan/UBSan and native/batched-path regression checks have not been run.

REUSE was also run separately before committing. With `REUSE_ENCODING_MODULE=chardet`, none of the changed files, including the Japanese documents, were missing license information. However, the repository-wide check still reports the existing `.agents/skills`: it is a symlink in Git but is treated as a regular file in this Windows checkout. The full REUSE check exited with status one and has not passed.

## Features not started or explicitly deferred

Until the first correct generation, paths may explicitly disable:

- oneAPI experimental command graphs and graph replay
- the oneAPI virtual-memory elastic cache
- `prepare_for_device_copy` for arbitrary host mappings
- the adaptive cache
- the shared expert arena
- the two-stage multi-GPU pipeline

When command graphs are disabled, standard SYCL queues and events must preserve dependencies, memory lifetime and synchronization order. The engine will not be replaced wholesale with the HIP engine.
