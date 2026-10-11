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

## Current policy: establish correctness while protecting remote access

While the user cannot directly operate the PC, remote access and recoverability take priority over performance. Do not reboot or shut down Windows, update/restart/reset GPU drivers, or change system-wide environment variables or registry settings. Do not launch upstream Strata executables, benchmarks or GPU load tests. Static source reference is allowed, but performance comparisons and output matching must wait for renewed confirmation that the user can directly operate the PC.

Proceed through Phase 0 baseline protection, Phase 1 standalone correctness and stability, Phase 2 timing breakdowns, Phase 3 small measurement-driven optimizations, and Phase 4 generalization and maintainability. In Phase 3, change one major factor at a time and compare prefill, decode, numerical results and generation quality under matching conditions. Keep the working queue/event path; removing synchronization requires checking dependencies and buffer lifetimes. DPC++/HIP remains a separate follow-up investigation, limited for now to static inspection of public sources.

Run GPU work one process at a time using short inputs and existing models. Initial generation caps are 64 and 128 tokens; do not continue past EOS merely to reach a cap. Neither 256 tokens nor long stress tests are mandatory. Use timeouts and monitor Windows RAM, commit and GPU memory. Stop additional load if anomalies, driver unresponsiveness or sudden memory growth are suspected; obtain confirmation before recovery-risk operations.

## 2026-10-11: Phase 0 baseline protection

The current branch is `feature/windows-r9700-sycl`, with HEAD `e7d67ead89cdc308e737b05d8a6ab09675c7679c`. At the start of this check, the only uncommitted changes were the Japanese and English progress documents containing the future plan. Those changes were preserved as a patch too.

The local `build/windows-baseline-20261011T074825Z/` contains the working executable, AdaptiveCpp runtime/backend DLLs, HIP-related DLLs, run/build scripts, CMake configuration, input IDs, successful logs and external-toolchain compatibility patches. Executable and DLL copies were checked with SHA-256, and `source.bundle` passed `git bundle verify`. The copied executable also completed `--help` successfully. This directory is excluded from Git. No model data was copied or downloaded.

Its recovery `run-baseline.bat` selects the saved executable and runtime, using the existing Windows environment and model data. The bundle and `worktree.patch` preserve the source state. This is not a complete image of the OS, driver or original toolchain/ROCm resource tree. Keep this saved execution path without overwriting the existing environment.

The current compiler is clang 24.0.0git from ROCm LLVM revision `4f43f4746ede4cc49ab4129f473dbd54e0d9db4f`; AdaptiveCpp is `0ea863abe1af4422593583f91e5608ef75044a16`. HIP headers report 7.17.26391, and the DLL's `hipRuntimeGetVersion()` returns `71726391`. These current header/runtime values are preserved separately from the earlier “ROCm 7.1 series” description. Configuration uses `STRATA_ADAPTIVECPP=ON` and `hip:gfx1201`, with oneMath/native experts/tests/packaging disabled; generation uses CPU experts, PLE and mmap.

Keep these two saved logs distinct. The 32-token output matches the first 32 tokens of the completed 41-token log.

| Log | Generated tokens | Prefill | Decode | TTFT | CPU pool elapsed time |
| --- | --- | --- | --- | --- | --- |
| `model-hello-1.txt` | 32 | 2.00 tok/s | 3.11 tok/s | 6,308.0 ms | 224.633 ms/token |
| `model-hello-final.txt` | 41, through EOS | 4.39 tok/s | 4.54 tok/s | 2,903.8 ms | 97.727 ms/token |

The latter records a final answer. However, these are not a matched before/after experiment with identical generation caps and established cache conditions, so their difference is not attributed to optimization. The CPU pool figure is elapsed time that may include waits and synchronization, not pure arithmetic time. Numerical agreement with upstream Strata has not been checked.

Before load, Windows reported about 14,660 MiB of available RAM, 18,121 MiB of available commit, a 38,984-MiB commit limit, and 1,502 MiB of aggregate adapter dedicated usage. A separate process queried HIP device 0 without executing kernels: about 32,473 MiB free and 32,624 MiB total. HIP and Windows counters need not describe the same budget; monitor Windows values during execution too.

Commands and records:

- `python .tools/protect-windows-baseline.py`: snapshot creation and hash/bundle validation
- `cmd /d /c "call build\windows-baseline-20261011T074825Z\run-baseline.bat --help"`: saved executable check
- `build/windows-acpp/baseline-protection.txt`: protection results
- `build/windows-acpp/phase0-hip-memory.txt`: HIP memory query
- `build/windows-acpp/phase1-preflight.txt`: Windows RAM, commit and GPU counters

Phase 0 protection and checks are complete. Phase 1 first fixed asynchronous-I/O completion bounds and shutdown lifetime using host-only tests, then proceeded to small real-file and GPU checks.

## 2026-10-11: Phase 1 transfer ordering and numerical repeatability

### Host-only tests and Windows I/O

`tools/test_session_token_pool_host.py` extracts the actual `session_token_pool()` and related state-update functions into a fake FIFO queue. It checks 48-layer QSA/GDN indexing, state across tokens, callback/transfer ordering, staging lifetimes, invalid inputs and exception cleanup. The added layer dumps are checked by the same fixture. Its mock arithmetic does not establish real-kernel numerical correctness.

`tools/test_dequant_host_contract.py` checks all 65,536 half bit patterns, 384 Q2_0 blocks, and the IQ4_NL codebook and layout. The combined six host-only tests pass.

Windows `DirectFile` now respects the caller's completion capacity even when wake packets and real completions are mixed. Dequeued excess completions remain available for the next call. Shutdown stops issuer threads, cancels outstanding reads and collects terminal completions before ending request/buffer lifetimes. Persistent IOCP infrastructure errors retain live requests and continue waiting instead of freeing them.

Besides fake-Win32 bounds, cancellation and dequeue-failure tests, the real Windows SDK build passes `/W4 /WX` and a real-file test passes. It uses only an owned 32,785-byte file, 64-KiB buffer bank and at most eight reads in flight. Checks cover issuer counts 1/3, completion capacities 1/2, data and outside-buffer canaries, short reads/EOF, rejected misalignment, close/destruction and reopen. Windows may modify the unreturned tail of a short read, so data validation covers only reported bytes. The test does not establish that every cancellation found pending I/O, or recovery from persistent infrastructure errors.

### Native in-order failure and workaround

Two identical hello runs had bitwise-equal residual ladders and 41 generated tokens, but 455,360 logit values differed, with a maximum difference of 23.862265. They exceeded `atol=rtol=1e-5`, and some host-logit argmaxes disagreed with GPU-selected tokens. Matching generated text alone did not detect this problem.

Separate trials of host-USM staging, draining with `queue.wait_and_throw()`, and waiting on the D2H event itself still had argmax disagreements. Changes without demonstrated improvement were reverted.

A small model-free SYCL reproducer isolated the problem. Peak explicit GPU allocation is 993,344 bytes; the process runs with a 20-second Job Object timeout and memory monitoring.

| Check | Native dedicated in-order | In-order through DAG dependencies |
| --- | --- | --- |
| Actual subgroup size and XOR reduction | Size 32; all 1,024 items match | All items match |
| 248,320-element copy after kernel, followed by marker wait | Every item returns stale zero | Every item bitwise matches its reference |
| Wait kernel event, then wait copy event | All items match | All items match |
| Actual Q8 head kernel versus independent scalar dot, 64 rows | All rows match | All rows match |

Kernel-to-copy ordering in the native dedicated in-order path cannot be relied upon in the current Windows HIP environment. Only Windows with AdaptiveCpp now supplies `AdaptiveCpp_inorder_executor{nullptr}` when creating compute, transfer and additional streams. AdaptiveCpp uses its DAG path, where each operation depends on its predecessor event. This is an internal submission DAG, not enabling oneAPI command graphs or capture. Linux/DPC++ queue settings, arithmetic kernels and sampling settings are unchanged. The backend's internal cause has not been identified.

### Real-model recheck

The corrected engine was checked with the existing Q2_0 pack and PLE GGUF, context cap 256, at most 64 generated tokens, mmap experts and no capture.

- Two independent three-worker runs bitwise match all 26,593,280 residual floats and 13,160,960 logit floats across 53 positions. Maximum difference is zero; all values are finite.
- The 41 generated tokens match the previous hello result, and each sampled position's host-logit argmax matches the GPU selection. Decoding through the end confirms a final `Hello!` greeting and EOS.
- One worker also bitwise matches three workers in residuals, every logit and generated tokens.
- One run also used 51 input tokens and a 128-generated-token cap. It reached a hello answer and EOS after 55 tokens; all logits across 105 positions were finite, and all 55 sampled-position argmaxes matched GPU selection. This does not test actually generating 128 tokens or reaching a context boundary.
- Residuals before and after the workaround are bitwise equal. The previously incorrect logit readback changed.
- The two diagnostic runs measured prefill 2.00/2.24 tok/s and decode 2.01/2.37 tok/s. Layer/logit dumps and cache conditions differ, so these are not speed comparisons against the older logs.

Each run has its own monitored Job Object; GPU tests are not concurrent. Stop conditions include a 150-second deadline, 3 GiB available RAM, 6 GiB available commit, and 8 GiB caps for process private memory and aggregate GPU dedicated usage. None of these runs hit a stop condition, and post-exit GPU usage returned to approximately its starting level. No new models were fetched, upstream Strata was not launched, and drivers/system settings were not changed.

Main records are `build/windows-acpp/phase1-dag-numerical-dumps.json`, `phase1-dag-worker1-numerical.json`, `phase1-dag-longer128-argmax.json`, `phase1-sycl-minimal.txt`, `phase1-sycl-dag.txt`, `phase1-directfile-real-v2.txt`, and each run's monitor/JSON. The existing logit-dump header describes planned rows: 76 versus 53 actual for hello, and 178 versus 105 for the longer input. Validation derives the actual row count from token logs.

The corrected executable, runtime DLLs, changed sources/new tests, validation records, Git bundle and uncommitted patch are separately saved in `build/windows-correctness-baseline-20261011T094332Z/`. SHA-256 and the bundle are checked; the saved executable also passes `--help`. The Phase 0 baseline is unchanged. Model weights are not copied, and recovery still requires the existing driver, toolchain and model resources.

Transfer ordering and numerical repeatability are established for the short sequential path. Still longer inputs, context boundaries, other quantization formats, independent references for all internal states and cleanup under real GPU faults need further checking. No performance optimization has been adopted. The following section records the Phase 2 timing breakdown.

## 2026-10-11: Phase 2 timing breakdown and stopping point

`STRATA_POOL_TIMING=1` measures existing API boundaries in the sequential CPU-pool path using a host monotonic clock. It is disabled by default. No measurement events, copies, synchronization or GPU profiling were added; arithmetic and buffer lifetimes are unchanged. `PoolTiming` counts attempted calls and records successful explicit-copy payload bytes.

The runs use `.tools/bounded-windows-run.py` with `--pool-timing --diagnostics`, setting `STRATA_POOL_TIMING=1` only in the child. Other settings are `HIP_VISIBLE_DEVICES=0`, `STRATA_VRAM_LIMIT_MIB=8192`, `STRATA_NO_XMX=1` and `STRATA_SSD_KEEPALIVE=0`. This does not enforce a 4-GiB maximum allocation. The reproduction command below uses existing local toolchain/pack resources; it is not an instruction to perform extra runs after stopping.

```text
python .tools/bounded-windows-run.py --exe build/windows-acpp/strata.exe --label phase2-new-measurement --max-new 64 --pool-timing --diagnostics
```

The saved JSON's older environment list omits `STRATA_POOL_TIMING`; invocation options and the timing header/53 breakdown rows prove activation. This record-format limitation is deferred; no extra execution was performed to repair it.

Conditions match the Phase 1 hello: 13 input tokens, at most 64 generated tokens, context cap 256, three workers, mmap experts and no capture. Two independent runs included layer/logit dumps and used the same monitoring/stop conditions. Every logit and residual across 53 positions bitwise matches both the saved pre-instrumentation result and the other instrumented run. The 41 generated tokens and sampled-position argmaxes match, and all values are finite. All seven host-only tests, including the new timing contract, also pass.

These are mean host elapsed milliseconds across 53 positions. GPU submit means API time for embedding, pre, post and head, not GPU execution time. Waits include runtime scheduling, GPU work and transfers. CPU expert time includes computation, I/O and dispatch waits inside the callback; these are not individually separated.

| Boundary | Run A ms/position | Run B ms/position |
| --- | ---: | ---: |
| Routing/input event wait | 349.611 | 324.439 |
| CPU expert callback | 174.642 | 116.992 |
| Combined GPU submit | 23.269 | 26.295 |
| Routing/input D2H submit and event record | 1.430 | 1.760 |
| Expert parts H2D submit | 2.590 | 2.296 |
| Final session drain | 1.421 | 1.438 |
| Diagnostic residual D2H submit | 0.625 | 0.660 |

Routing/input D2H payload is 495,360 bytes/position; parts H2D is 4,915,200 bytes/position. These explicit payloads exclude kernel/driver staging, PLE and head-logit transfers and are not physical bus traffic. Normal settings have no independent wait immediately after post, so that timing field is zero. Completion waits are included in subsequent routing waits and the final drain. Diagnostic residual copies also affect later waits.

Prefill and decode are summarized separately. Decode averages are wait 352.349/318.846 ms, CPU callback 160.867/116.177 ms and GPU submit 22.938/25.144 ms. Cache temperature was not reset or controlled, so A cannot be declared cold and B warm. Instrumentation/dump overhead prevents speed comparisons with earlier ordinary runs.

The three largest observed boundaries and possible future investigations follow. No candidate implementation or optimization trial was performed.

1. **Routing/input waits.** The longest host interval. Separate GPU arithmetic, transfers and DAG scheduling before considering ordered submission batching or kernel fusion. Simply restoring the broken native in-order path or removing synchronization is not a candidate.
2. **CPU expert callbacks.** Also show substantial variation between runs. Separate paged I/O from computation before examining worker/task granularity and memory-budgeted fixed caches. Callback elapsed time is not pure CPU computation time.
3. **GPU submission host overhead.** Many small pre/post calls accumulate. Inspect launch counts and runtime work before considering batching that preserves dependencies and staging lifetimes.

Main records are `build/windows-acpp/phase2-timing-summary.json`, `phase2-timing-summary.txt`, `phase2-timing64-a-numerical.json`, `phase2-repeat-numerical.json`, and both execution/monitor logs. Peak process private memory was approximately 5,019 MiB, minimum available RAM 6,046 MiB and available commit 12,665 MiB; no stop threshold was reached.

The Phase 0 executable and the numerically checked Phase 1 recovery copies are preserved. Repository lint is incomplete because required tools are missing. The `possible secret` message when gitleaks is absent is not evidence of a detected secret. New-file REUSE findings are resolved; only the pre-existing Windows checkout state of `.agents/skills` remains. Linux free/contrib-llvm, NVIDIA, integrated graphics, ASan/UBSan, context boundaries, other quantization formats, real GPU fault cleanup, physical transfer timing and hard-fault attribution still need checking.

At the user's request, work stopped after Phase 2 records and validation. At that point, Phase 3 optimization, migration to DPC++, other-GPU support and major refactoring had not started.

## Before Phase 3: temporary AdaptiveCpp async-handler patch

In the AdaptiveCpp version in use, `0ea863abe1af4422593583f91e5608ef75044a16`, the node-bearing `sycl::event` constructor did not retain its supplied `async_handler`. Delivering a nonempty error list could produce `std::bad_function_call` instead of the original exception.

`third_party/main/adaptivecpp/patches/0001-event-async-handler.patch` changes only that constructor line, preserving the original BSD-2-Clause copyright/license notice. `tools/adaptivecpp_build.py` checks the commit SHA and complete header before applying the patch ahead of the build. It skips an already-applied patch and stops on inconsistency. It distinguishes LF/CRLF and refuses source/build/prefix overlap or overwriting the working installation. Installed-header contents and SHA-256 are recorded in the receipt. Instructions are in `third_party/main/adaptivecpp/README.md`.

Without overwriting the existing environment, a separate worktree at the same commit, `.tools/AdaptiveCpp-eventfix-src`, was built with only the CPU backend and installed into `.tools/adaptivecpp-eventfix`. The installed header matches the patched source. The initial standalone Windows build stopped because exceptions were disabled; `/EHsc` was then explicitly supplied in this separate build's compiler flags. Neither the XeStrata engine nor AdaptiveCpp queue/DAG/scheduler implementations were changed.

The isolated `tests/platform/adaptivecpp_event_async.cpp` uses real installed headers/runtime and the shared error list. Its nodes are null; it uses no device queue, device allocation, kernel, DMA or direct HIP call. Controlled synthetic errors verify original diagnostics and SYCL errc, handler counts, batching, event copies, no redelivery and throwing handlers. The negative control, using the same CPU runtime with the original header, reproduces an uncalled handler and `std::bad_function_call`. Both complete successfully within a 20-second timeout and report the new prefix's runtime DLL path. Windows does not guarantee `exception_ptr` identity across rethrows, so the test checks type/code/diagnostics instead.

All 26 host wrapper tests pass. Main records are `build/windows-acpp/async-eventfix-build-v2.log`, `async-eventfix-receipt.json`, `async-eventfix-receipt-v3.json` (the final wrapper's already-applied build/install rerun), `async-eventfix-installed-check.json`, `async-eventfix-wrapper-tests-final-v2.log`, `async-eventfix-installed-final-v2.json`, `async-eventfix-skip-final-v2.json`, `async-eventfix-fixed-test-v2.txt`, and `async-eventfix-broken-test-v2.txt`.

The working AdaptiveCpp source/header, HIP installation, generation executable and existing recovery copies remain intact. The new prefix is for isolated validation, not a switch of the working HIP environment. This CPU-only isolated check performed no GPU execution, intentional hang, driver update or Phase 3 optimization.

This fix covers notification by node-bearing events supplied with a handler. Default-constructed-event handler behavior, competing consumers of the process-global error list, cancelled nodes reporting complete, and cleanup after failed HIP waits or uncertain DMA remain separate issues. These passing tests do not establish GPU-fault buffer lifetime or real-model correctness. Work stops here.

## Phase 3B: Routing D2H marker reduction

### Baseline and change scope

At the start, HEAD was `ae1d3ac7be3b1e2eba6ccae75f32f8c9ca1c2968`, branch `feature/windows-r9700-sycl`, with a clean worktree. AdaptiveCpp is `0ea863abe1af4422593583f91e5608ef75044a16` and LLVM is 24. The baseline source was frozen in a separate worktree, and both `build/windows-phase3b-baseline` and `build/windows-phase3b-candidate` were built with the same free-mode configuration. Both use HIP-enabled headers copied from the working installation, overlaid only with the SHA-pinned async-handler fix. This does not replace the working installation.

Loaded modules were enumerated in the model process. `acpp-rt.dll` and the HIP backend came from `.tools/adaptivecpp-amd24/bin`; the HIP DLL was loaded from `C:\Windows\System32\amdhip64_7.dll`, not directly from the shared SDK. Each run's JSON records paths and SHA-256 hashes. This Windows DLL's FileVersion is `10.0.3683.0`; it is not treated as the HIP API or ROCm version.

In `src/core/session.cpp`, the weights D2H now uses the existing `copy_async_after(..., cs, nullptr, scratch.probe)`, removing only the immediately following empty marker and its failure branch. The pre → mixed D2H → IDs D2H → weights D2H order, final-copy event wait, CPU Expert callback, parts H2D, post, final drain and buffers remain unchanged. Windows/AdaptiveCpp DAG dependencies remain in place. Arithmetic, quantization, sampling, caches and capability-based device selection were not changed.

### Numerical, ordering and lifetime checks

Before Phase 3B, a small R9700 test verified all 6,144 values from one kernel and three D2H copies. Only the last copy event was waited on, without first waiting for the kernel or the whole queue. A controlled error injected into the shared error list after healthy GPU completion also reached the handler from that copy event, preserving exception type, code and diagnostics. The record is `async-eventfix-gpu-once.txt`. This was not a real GPU fault test.

The real-model comparison covers 13 input IDs, 41 generated tokens and 53 positions. All 13,160,960 logits and 26,593,280 residual floats for the input plus 48 layers are bitwise identical, with zero nonfinite values. Maximum logit difference is zero, generated tokens match, and all 41 sampled argmaxes in each run match the GPU-selected IDs. The new baseline build also matches the saved Phase 2 logits and residuals bitwise. EOS-shortened actual rows are used; the existing logit header's 76 planned rows are not mistaken for the 53 executed rows.

The FIFO host fixture checks last-copy event publication, completion of all three copies before callbacks, staging reuse after the preceding layer's parts DMA, and callback suppression/drain attempts when event publication fails. All seven existing session/dequant host tests pass. Successful mock drains are not evidence of physical DMA completion during real faults. Both builds have the same aggregate warning diagnostics, with no new warning messages. `git diff --check` and Japanese lint pass. Full repository lint remains incomplete because required tools are missing; the only remaining REUSE finding is the pre-existing `.agents/skills` checkout state.

Successful routing D2H API counts drop from 192 to 144 per position. Across 48 layers and 53 positions, 2,544 empty markers are removed; payload remains 495,360 bytes per position in both builds. Parts H2D remains 4,915,200 bytes per position. The existing `routing_D2H_submit+record` display label is retained. Bytes count payload for API calls that report success, not physical bus traffic.

### Repeated performance comparison under matching settings

The two numerical runs with diagnostic dumps were excluded from performance aggregation. Each build then ran three times without diagnostic dumps, sequentially in A1 → B1 → B2 → A2 → A3 → B3 order. A is the marker baseline and B waits on the last weights copy event. Both use the same Q2_0 pack, 13 input IDs, generation cap 64, context 256, greedy/EOS, three workers plus caller, mmap, fixed GPU cache 0, no-capture, no-XMX and 8 GiB VRAM reporting limit, with `--check-logits --stats` and host timing at existing boundaries. Each process runs in a private Job with a 150-second timeout and existing memory guards. Prefill covers 12 positions and decode 41.

| Run order | Prefill tok/s | Decode tok/s |
| --- | ---: | ---: |
| A1 | 2.083 | 2.073 |
| B1 | 2.265 | 2.282 |
| B2 | 2.275 | 2.525 |
| A2 | 2.471 | 2.636 |
| A3 | 2.546 | 2.684 |
| B3 | 2.517 | 2.689 |

Median prefill is A 2.471 and B 2.275 tok/s; median decode is A 2.636 and B 2.525 tok/s. Candidate medians are lower, but ranges overlap and performance also varies with run order. OS/file-cache temperature and external load were not controlled; no cold/warm designation or causal speedup/regression claim is made. The median of per-run mean routing D2H submit times is A 1.603 and B 1.094 ms per position. These are host API timings, not pure GPU arithmetic or transfer time.

All eight runs stay below memory guards, with no async-error diagnostics or generated-token differences. Peak process private memory is approximately 5,020 MiB and aggregate GPU dedicated usage 6,627 MiB. Across the six timing runs, minimum available RAM is approximately 6,002 MiB and available commit 12,456 MiB; post-exit aggregate GPU dedicated usage returns to approximately 1,522 MiB.

### Adoption decision and stopping point

An overall speedup was not established. The small change is adopted because it removes empty markers and a failure branch without changing dependencies, buffers or numerical processing. No performance benefit is guaranteed, including the observed lower medians above. No major new regression was observed. Cancellation/uncertain-DMA cleanup, competing global error consumers, long contexts, other quantization formats, Linux free/contrib-llvm, NVIDIA, integrated graphics and ASan/UBSan remain separate checks. No upstream SDMA fix was integrated, and DAG ordering was not removed.

The working AdaptiveCpp source/header, HIP/ROCm environment, `build/windows-acpp/strata.exe` and existing Phase 0/1 recovery copies remain unchanged. Baseline records are in `build/windows-acpp/phase3b-start-state.json`; numerical comparisons in `phase3b-numerical-comparison.json` and both argmax JSONs; performance in `phase3b-performance.json`; host tests in `phase3b-combined-host-contracts.log`. Run logs, monitors, module paths/hashes and final working-version integrity records are saved in the same directory. Generated artifacts are excluded from the commit.

Work stops after committing only Phase 3B records and the adopted change. Phase 3C GPU Expert caching, Phase 3D cleanup and upstream-update integration were not started.

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

## Future plan: stabilize AdaptiveCpp and investigate DPC++ with HIP

### Current baseline and public implementations

The port remains experimental Windows/AMD support. Its execution path is XeStrata → SYCL → a custom AdaptiveCpp build → LLVM 24/AMDGPU → HIP/ROCm → the R9700. Existing Q2_0 pack data and the PLE GGUF have already supported model loading, serial prefill, decode and text generation. This supersedes the early investigation's “model loading and generation not confirmed” status. However, the successful build covers the selected configuration, not native/batched paths or kernels in every configuration.

DPC++ also has an AMD HIP implementation. Unified Runtime is now developed under `unified-runtime/` in intel/llvm; oneapi-src/unified-runtime is a mirror. The HIP adapter's CMake contains Windows import-library settings. Nevertheless, the DPC++ Getting Started Guide states that Windows HIP is not supported. Windows-specific source branches and correct execution on the R9700 require separate verification.

Codeplay's oneAPI for AMD GPUs 2025.2.0 instructions target Linux. Intel/llvm issue #22606 also reports a gfx1200 case on Ubuntu where `single_task` works but `nd_range` fails to launch. Follow-up comments report successful launches and device linking from a static archive after fixes. These are not Windows/gfx1201 success evidence; they inform reproducer tests and compiler revision selection.

### Stage one: stabilize the AdaptiveCpp port

Keep the working Q2_0 serial path as the correctness baseline for further changes. First test asynchronous-I/O shutdown and cancellation, queue/event dependencies, and memory lifetime on failure. Investigate subgroup-size attribute warnings and check both the actual subgroup sizes and operations such as reduction and routing that depend on them. Compilation or a short generation run alone cannot establish correctness of every kernel.

Next document toolchain revisions, compatibility patches, DLL placement and build settings, then reproduce the same generation from an empty build directory. Add longer inputs, longer decode runs, context-boundary cases and different CPU worker counts. Enforce the largest-allocation limit, split arenas exceeding 4 GiB, and then test smaller configurations. Also verify affected tests in Linux free and contrib-llvm builds.

Completion requires reproducible generation without omitting model components, safe cleanup on failure, and build instructions and verification scope that can be checked in other environments. Add native experts and batched prefill after preparing comparison tests against the serial path. Restoring command graphs is not a prerequisite.

### Stage two: investigate Windows DPC++ with HIP separately

Use a separate branch or worktree and toolchain directory for DPC++ investigations; do not overwrite the working AdaptiveCpp environment. Build intel/llvm's DPC++ and the Unified Runtime HIP adapter from source, first checking LLVM compatibility with the chosen ROCm device bitcode. Keep compiler and runtime revisions aligned.

Proceed through adapter loading and GPU enumeration, USM allocation and transfers in both directions, `single_task`, `nd_range`, and event dependencies between multiple queues. Validate half and subgroup operations too, checking runtime sizes and computational results. Because XeStrata combines static libraries, also check device linking across multiple translation units and static archives. Only after output-validation tests pass should model loading and serial generation in XeStrata be attempted.

A compilable oneAPI extension API does not establish HIP-backend support. Check command graphs, virtual memory and matrix features individually, retaining queue/event or other fallbacks where unavailable. Select GPU paths from reported capabilities and actual kernel results, not product names.

Success requires executing and validating both the minimal tests and real-model generation on the Windows R9700. If a clear build or runtime blocker appears, record its minimal reproducer, revisions and errors, stop that investigation, and continue AdaptiveCpp development. Starting the investigation does not commit the project to switching to DPC++.

### Stage three: compare correctness and speed under matching conditions

Defer executing upstream Strata for comparison until the user confirms direct PC access. After that confirmation, compare the AdaptiveCpp port with Strata HIP using the same R9700, model revision and quantization, input token IDs, context length, KV settings, CPU worker count, memory budget and generation settings. Add DPC++ if it reaches real-model generation. Use existing models and packs, asking the user to provide any missing data rather than downloading models automatically.

For correctness, define numerical tolerances and compare logits and layer states as well as generated text. Measure prefill, decode and TTFT separately over repeated runs, distinguishing serial from batched execution and cold from warm runs. Record CPU expert work, GPU work, transfers, I/O, waits and memory use. CPU-only runs or runs omitting PLE or experts cannot establish success.

Use measurements to prioritize a fixed GPU expert cache, transfer overlap, batching and AMD matrix acceleration. Also compare whether DPC++ reduces oneAPI compatibility code and makes the toolchain easier to maintain. Choose paths by model correctness, reproducibility, speed and maintenance cost, not API counts or compiler names. Restrict non-free dependencies to the appropriate build modes, preserving XeStrata's license and Linux paths.

### Research references

- [DPC++ Getting Started Guide: HIP limitations](https://intel.github.io/llvm/GetStartedGuide.html#hip-back-end-limitations)
- [Unified Runtime's current development location](https://github.com/intel/llvm/tree/sycl/unified-runtime)
- [HIP adapter CMake](https://github.com/intel/llvm/blob/sycl/unified-runtime/source/adapters/hip/CMakeLists.txt)
- [Codeplay oneAPI for AMD GPUs 2025.2.0](https://developer.codeplay.com/products/oneapi/amd/2025.2.0/guides/get-started-guide-amd.html)
- [intel/llvm issue #22606](https://github.com/intel/llvm/issues/22606)

## Features not started or explicitly deferred

The initial generation path does not use the following. Until stabilization and comparative validation are complete, keep alternatives or explicit disabling rather than expanding support:

- oneAPI experimental command graphs and graph replay
- the oneAPI virtual-memory elastic cache
- `prepare_for_device_copy` for arbitrary host mappings
- the adaptive cache
- the shared expert arena
- the two-stage multi-GPU pipeline

When command graphs are disabled, standard SYCL queues and events must preserve dependencies, memory lifetime and synchronization order. The engine will not be replaced wholesale with the HIP engine.
