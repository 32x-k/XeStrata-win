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

Work stopped after committing only Phase 3B records and the adopted change. At that point, Phase 3C GPU Expert caching, Phase 3D cleanup and upstream-update integration had not started.

## Phase 3C: canonical GPU Expert cache investigation and withdrawal

### Starting baseline and existing implementation

Starting HEAD was `f4f2350ce9f50d65169644991dfbe898464f8702`, branch `feature/windows-r9700-sycl`, with a clean worktree. AdaptiveCpp remains `0ea863abe1af4422593583f91e5608ef75044a16`, compiled with LLVM 24. The separate executable validated in Phase 3B was reused unchanged, preserving the working version, Phase 0/1 recovery copies and shared ROCm environment. Patched async-handler headers, existing DAG ordering and Phase 3B D2H event reuse remain the same. The loaded core runtime DLL hashes match across all three model runs; the HIP DLL is still `C:\Windows\System32\amdhip64_7.dll`.

Existing `ExpertCache` and `expert_hit_run` are compiled on Windows. A positive cache count, enabled CPU pool, canonical geometry and complete hit buffers reach `moe_hit_grouped_s2`. Graphs, VMM and native layout are not required. The cache header's old claim of no consumer/kernel does not describe the current executable path.

`--expert-cache N` counts expert slots, not MiB. Canonical Q2_0 has H=2560, FF=640 and 1,382,400 bytes per expert. The uniform cache uses one device-USM arena of N×blob bytes, not N separate allocations. The initial test uses four slots: 5,529,600 bytes, approximately 5.27 MiB. Six hit allocations—scratch, slot/destination lists, input Q8, FP32 scales and output—total 165,440 bytes. SYCL reports both global memory and largest single allocation as 34,208,743,424 bytes, with subgroup size 32. The new cache arena is below that reported limit.

AdaptiveCpp `mem_info()` does not return free VRAM, so the cache's internal free-memory preflight is inactive. The existing allocator also does not enforce the largest-single-allocation limit. Auto sizing was therefore avoided; external aggregate GPU dedicated telemetry and the existing 8 GiB memory guard provided the safety margin check. This query does not resolve the existing dense-arena or simulated 4-GiB-limit issue.

Global admission fills consecutive slots with the first distinct (layer, expert) pairs; there is no eviction once full. A new admission queues blob H2D before the kernel on the same stream and computes on GPU on that call. CPU dispatch writes zero to hit rows and computes only miss rows. GPU hit output is added after parts H2D, with router weights applied in post; that existing path is unchanged. The per-layer policy assigns N/48 to each layer and all remainder to the final layer. With N=4, the first 47 layers would have zero quota, so only global policy was used.

The cache arena retains existing RAII ownership and runtime drains. Six raw hit buffers and the probe event have no individual final cleanup and remain until process exit, an existing ownership limitation. Failed filling after admission is also not established as safely reusable. These issues and cancellation/uncertain-DMA cleanup were not fixed in this work.

### Numerical contract fixed before comparison

Before the first cache-on run, every logit/residual value was required to satisfy `abs(a-b) <= 0.001 + 0.001 * max(abs(a), abs(b))`, with all values finite and sampled argmax/generated tokens matching. Bitwise equality was not mandatory because CPU/GPU reduction and quantization orders differ; tolerances were not relaxed after results. The contract is saved in `phase3c-predeclared-contract.json`.

Without source changes, cache-off and the default four-slot cache each ran once with the same Q2_0 pack and 13 input IDs, CPU pool with three workers plus caller, mmap, no-capture, context 256, greedy/EOS, cap 64, no-XMX, diagnostic dumps and host timing. Both explicitly use `--adapt-every 0 --pcie-frac 0`; no auto cache, profile, elastic cache, VMM or Command Graph was used.

### Numerical results and bounded investigation

Cache-off matches the saved Phase 3B logits and residuals bitwise. Default cache-on exits normally, generating the same 41 tokens and argmaxes at all 53 positions. However, 12,738,073 of 13,160,960 logits and 8,699,164 of 26,593,280 residual floats violate the predeclared rule. Maximum differences are approximately 0.827092 for logits and 0.391922 for residuals. The numerical gate fails; token agreement alone is not success.

The next run keeps four slots and the same tolerances, changing only the existing `--expert-cache-cpu-order` option. It also exits normally, but produces 40 tokens and 52 positions, with different tokens and argmaxes. Common-position comparisons still fail, with maximum differences approximately 34.5925 for logits and 15.0584 for residuals. The EOS row-count difference is recorded rather than treating 52-position payloads as the same shape as 53.

The first position's input checkpoint is exact in both cache-on variants. Maximum difference after one layer is approximately 2.24e-8 in both; the default path reaches approximately 0.001345 after four layers. This suggests possible amplification through subsequent quantization/state, but does not establish the root cause or correctness of all dispatches.

Further cache-on model repetitions were stopped in favor of a small raw Expert fixture. It links existing production CPU/GPU quantizers and Expert kernels, using one canonical layer-0/expert-0 blob and fixed runtime input. A small DAG-queue adapter replaces engine discovery/probes/teardown, so this fixture is not a model or complete-cache validation. Explicit device allocation is 1,413,120 bytes in one bounded process. Default and CPU-order each run once: all 2,560 input int8 values, 80 FP32 scales and 80 FP16 block headers match exactly. All 2,560 Expert output values meet the contract in both, with maximum differences approximately 3.35e-8 and 5.96e-8 and zero nonfinite values.

At the five-judgment review, progress includes confirming the cache path, rejecting CPU-order alone as a fix, locating small initial checkpoint differences and verifying local raw-kernel parity. Causes involving actual routing/input, multiple hits and subsequent state are not isolated, and no safe small Windows compatibility correction was identified. No tolerance increase, capacity escalation, new cache algorithm or GPU-kernel rewrite was used as a workaround.

### Hit rates, diagnostic timings and memory

The default four-slot run has 25,440 lookups, four already-resident hits, four admissions and 25,432 CPU misses. Its displayed resident hit rate is approximately 0.0157%; GPU execution including admissions is 0.0314%, and miss rate 99.9686%. The first routing fills all four slots, and global first-seen admission is biased. This is a small enablement check, not a useful cache-placement evaluation. `expert blobs` counts source requests including hits, not physical SSD reads.

| Diagnostic run | Prefill tok/s | Decode tok/s | Mean CPU callback ms/position | Mean routing wait ms/position | Total inference ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| cache-off | 1.71 | 1.87 | 177.260 | 330.578 | 28,952.0 |
| default four slots | 2.01 | 2.31 | 107.132 | 303.978 | 23,716.6 |
| CPU-order four slots | 1.65 | 1.20 | 203.666 | 529.283 | 40,598.0 |

These are single diagnostic-dump runs, not repeated performance comparisons. CPU-order also generates a different token count. OS page-cache temperature is uncontrolled; neither shorter CPU callbacks nor the default run's displayed throughput establish speedup. Existing instrumentation has no separate GPU hit launch/combine or cache-setup spans; initial fills occur inside hit Launch. These times are not inferred from other spans, and loading time is not called cache initialization time. Numerical gate failures prevented three-runs-per-side benchmarking, steady-state evaluation or capacity escalation.

All three model processes exit without async-error diagnostics or memory-guard violations. Peak private memory is approximately 5,020 MiB and aggregate GPU dedicated usage 6,716 MiB, returning to approximately 1,474–1,546 MiB after exit. These observations do not establish new real-fault synchronization or lifetime guarantees.

### Withdrawal decision and restart conditions

Cache-on is not adopted. The existing path executes on GPU but fails the predeclared full-model numerical contract, with no isolated compatibility correction. Sources and working binaries remain unchanged; cache-off and the Phase 3B baseline are retained. Restart requires comparing GPU hit/CPU miss intermediate outputs using the same experts and real routing inputs, locating the first amplifying quantization/state/dispatch, and validating a small correction under the same numerical contract. Cache-on performance evaluation follows only under a separate instruction.

Starting/protected state is in `build/windows-acpp/phase3c-start-state.json`; the contract in `phase3c-predeclared-contract.json`; comparisons in `phase3c-on4-initial-vs-phase3c-off-initial.json` and `phase3c-on4-cpuorder-vs-phase3c-off-initial-v2.json`; raw results in `phase3c-raw-default.txt` and `phase3c-raw-cpuorder.txt`; diagnostic metrics in `phase3c-diagnostic-metrics.json`; integrity in `phase3c-final-integrity.json`. Logs, models, executables and local fixtures are not committed. `git diff --check` and Japanese lint pass. Full repository lint remains incomplete because required tools are missing; REUSE retains only the existing `.agents/skills` checkout finding. Only bilingual withdrawal records are committed, then work stops. Phase 3D, WMMA, Command Graph, VMM, DPC++ migration and other-GPU support were not started.

## Phase 3P: performance accounting and restart point

### Baseline, methods and stopping decision

Starting HEAD was `570fc69a690d9a9a311fc943a1644a798553c281`, branch `feature/windows-r9700-sycl`, with a clean tree. Recorded hardware/software are Windows 11 Pro build 26200, Ryzen AI 9 HX 370 (12 cores/24 logical processors) and R9700 driver `32.0.32015.2008`. The free-mode LLVM 24 revision is `4f43f4746ede4cc49ab4129f473dbd54e0d9db4f`; AdaptiveCpp remains `0ea863abe1af4422593583f91e5608ef75044a16`. SDK headers identify HIP 7.17.26391. The actually loaded HIP DLL is System32 `amdhip64_7.dll`; its Windows FileVersion/ProductVersion `10.0.3683.0` is not called the HIP API version. Core DLL hashes also match in the two new model processes.

Eleven saved Phase 2/3B/3C cache-off runs were reaggregated. The current throughput reference uses only three no-dump Phase 3B candidate runs. Shared conditions are the existing Q2_0 pack and IQ4_NL PLE, 13 input IDs, 41 generated tokens, context 256, cap 64, greedy/EOS, three workers plus caller, mmap, cache zero, no-capture, DAG ordering, no-XMX and 8 GiB reported VRAM limit. Historical 3B base has 192 routing API calls/position; current candidate has 144. Phase 2 diagnostic-dump timings are not treated as an optimization comparison against no-dump 3B runs. OS page-cache temperature and external load are uncontrolled.

Additional measurement is limited to one healthy diagnostic run of the unchanged 3B executable, one model-free SYCL profiling check, and one ETW-recorded inference attempt. The healthy diagnostic has bitwise-equal 13,160,960 logits and 26,593,280 residual values against saved cache-off, all finite, with all 41 sampled argmaxes/tokens matching. Its diagnostic decode rate is 1.76 tok/s; the difference from historical no-dump median 2.525 tok/s is not treated as a regression.

No AMD profiler was found in the checked PATH/install locations. The [official Windows component table](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/conceptual/component-support.html) lists Radeon GPU Profiler, not proof of its installation here. Exactly two alternatives were considered: standard SYCL profiling and ETW. Standard profiling was queried only after the existing last-copy wait; all 6,144 readback values match. The HIP backend places stream events around operations and derives differences from `hipEventElapsedTime`. These intervals can include command-issue gaps; they are not hardware kernel-dispatch or DMA bus-occupancy durations. Absolute timestamps combine `steady_clock` with a host-anchored reference event, introducing cross-lane anchor uncertainty.

Walking the DAG from its last event cannot recover all submissions after weak predecessor nodes have been garbage-collected. A local queue-observer prototype was created, but requires a consistent header overlay and retention of every node. It was not built, run or adopted for the model. Working binaries and inference sources remain unchanged.

With explicit user UAC approval, WPR `10.0.26100` started a dedicated instance with `GPU.light`, `CPU.light` and `DiskIO.light` in memory mode. This avoided trace writes contaminating model-SSD measurement during inference. At 35.938 seconds, however, available RAM reached 3,022.6 MiB; the 3 GiB Job guard stopped inference. Thirty-one timing positions survive, not a complete decode run. Memory pressure during recording is confirmed, not ETW alone as its isolated cause. Additional GPU runs were cancelled. The owned WPR instance stopped with code zero and was confirmed not recording. The new healthy diagnostic has dumps, whereas ETW is no-dump and incomplete; their rate difference cannot quantify profiling overhead. A complete matched normal/profiled throughput comparison was not achieved.

WPAExporter `11.8.423.12582` analyzed the stopped trace offline. An initial JSON-config attempt failed before loading with `Invalid output format`; one changed-premise CLI attempt followed. Command-line metadata confirms PID 10020. CPU, process, disk and lost-event CSVs were exported, but the GPU preset had no data and failed export. Exit code zero is not complete table success. Failure to obtain GPU time from that preset does not prove absent GPU events in the trace. An empty lost-event CSV also does not prove zero loss. Export ran under a 4 GiB private-memory cap, 3 GiB available-RAM floor, 120-second deadline and kill-on-close Job; it completed in 23.984 seconds with approximately 2,468 MiB peak private memory.

### Measured accounting

The following is host accounting for 41 decode positions in Phase 3B candidate 2, the median-throughput run: 16,239.4 ms/41 = 396.083 ms/position. Independent per-span medians from different runs are not summed into one run's breakdown.

| Host window | ms/position | Wall share | Included work and limits |
| --- | ---: | ---: | --- |
| Routing/input wait | 288.519 | 72.84% | Queued pre, preceding post/H2D, transfers and DAG/runtime |
| CPU Expert callback | 72.724 | 18.36% | Mmap faults, input quantization, two-stage Expert work, dispatch/barriers |
| All GPU-operation submissions | 25.308 | 6.39% | Host kernel/copy calls, not GPU execution |
| Remaining measured drain windows | 0.781 | 0.20% | Existing completion waits; host-disjoint from the rows above |
| Unpartitioned token-timer remainder | 8.751 | 2.21% | PLE, head readback/finite scan/sampling, control work; not GPU idle |

Across the three current runs, routing/input wait is 267.042–330.200 ms/decode position, 71.81–75.34% of wall. CPU callback is 72.676–75.349 ms, 16.58–20.26%. These large host windows are established, but routing wait is not split into GPU arithmetic versus runtime. Pre/post/embedding/head API counts are not internal GPU kernel counts.

Explicit payload is 495,360 bytes/position for routing D2H (48 layers × three copies) and 4,915,200 bytes/position for parts H2D (48 copies). Representative host submissions average 1.152 ms D2H and 2.038 ms H2D. Forty-eight routing-event waits and final session drain are retained. Zero `parts_wait(post)` does not make transfer cost zero: the next routing gate protects preceding parts H2D and staging lifetime. Existing host staging uses `malloc_host`, mapped by HIP backend source to `hipHostMalloc`; Expert mmap is a separate ordinary pageable mapping. Actual DMA duration, transfer-only dependency waits and physical bus bytes remain unmeasured.

The CPU pool's all-53-position mean dr is 71.752 ms in 3B candidate 2, inside 71.916 ms outer callback. Default `run_split` charges intermediate quantization and both dispatch/barrier phases entirely to dr; it does not separately measure wp/rp. Printed zero wp/rp is not evidence of cheap worker waits or synchronization. Dequantization, arithmetic, hard faults, SSD service, memory bandwidth and task/worker times remain unpartitioned. Expert requests of 632.8125 MiB/position are logical blob bytes, not physical SSD/DRAM traffic.

The new healthy diagnostic's final process counters are 106.297 user plus 61.719 kernel = approximately 168.016 CPU seconds, averaging 4.79 logical-core equivalents between samples. Its 2,233,038 cumulative page faults combine soft and hard faults. Approximately 6.32 GB read in `GetProcessIoCounters` includes loading and is not physical mmap SSD traffic. The theoretical callback CPU-time ceiling for four compute threads is 9.759 callback seconds × four = 39.037 CPU seconds. That work region alone cannot explain whole-process 168 CPU seconds; assigning the remainder to runtime or worker idle requires thread-specific evidence.

Partial ETW sampled CPU is 169.763 CPU seconds/36.054-second process lifetime, approximately 4.71 logical-core equivalents. It is a different, incomplete run; sampling and exact counters are different quantities. The CSV is process-aggregated with blank Strata stack and no thread/module columns. Disk output is disk-aggregated with blank Process/Path. Neither identifies CPU Expert, driver, DAG scheduler or SSD as an individual cause.

GPU times and per-layer kernel counts for Attention/GDN/QSA, Router, Shared Expert, post, embedding/LM Head and quantization/GEMV remain unmeasured. Actual GPU idle intervals, HIP API durations, submission-to-execution latency and DAG scheduler time are also unmeasured. Neither roughly 320 ms waits nor whole-process 4.79-core CPU usage is labeled AdaptiveCpp overhead.

### Bottleneck ranking and improvement candidates

Ranking concerns established host windows and investigation value, not a confirmed GPU-arithmetic/runtime cause ranking. For one zero-cost window and otherwise unchanged work, the theoretical ceiling is `1/(1-f)`; it is not a predicted improvement and ceilings are not added.

| Rank/candidate | Evidence and confidence | Ceiling, effort, numerical risk and followability |
| --- | --- | --- |
| 1. Separate GPU/runtime inside routing critical window | High confidence in 71.81–75.34% window; internal cause unresolved | Representative ideal removal ceiling 3.68×; safely removable time unknown. Fusion/WMMA bring large effort, regression risk and upstream delta; no implementation choice before execution-occupancy evidence |
| 2. Isolated CPU-pool idle-spin setting A/B | Whole-process CPU approximately 4.8 cores; source default spin 20 ms, versus mean routing 5.56–6.88 ms/layer. Spin contribution is a source hypothesis without thread attribution | Throughput ceiling unmeasured. Existing `STRATA_POOL_SPIN_US` needs zero source delta, low effort/risk, but waking can hurt latency. Historical full environments were not saved, so actual 20-ms use is not asserted |
| 3. Separate CPU Expert work/fault/barrier | High confidence in 16.58–20.26% callback; subdivisions unresolved | Independent ideal callback-removal ceiling 1.20–1.25×. CPU layout/dequant optimization has moderate effort/numerical risk. GPU cache first requires 3C numerical recovery and additional GPU-cost validation; it stays disabled |
| 4. Small host-submission reduction | High confidence in representative 6.39% submit window, not all runtime cost | Ideal all-submit removal ceiling approximately 1.068×; D2H submit alone approximately 1.003×. Small API cleanup can follow upstream, without changing DAG ordering or necessary waits |

AdaptiveCpp/ROCm upstream fixes may improve scheduler/driver work, but unpartitioned times cannot predict their gains. SDMA ordering fixes do not justify removing the DAG workaround. CPU-pool settings/layout can mostly be shared with upstream rather than creating another GPU-specific engine. Upstream Strata was not executed or used as an overhead-performance oracle.

### Deliverables, restart point and stop

Adopted code is offline-only `tools/profile_pool_timing.py` and its standard-library tests. UTF-8 logs supply validated actual rows and JSON prefill/decode/all-position host accounting, payload/API counts, remainder and independent ideal ceilings. It starts no engine/GPU and infers no cache/profiling settings from absence. Nine tests and three current 3B logs plus the new healthy diagnostic validate it. `git diff --check` and Japanese lint pass; full lint is incomplete because tools are missing, and REUSE retains the existing `.agents/skills` checkout finding. The local observer and ETW controller are not adopted into production.

The smallest restart candidate is a single-factor, cache-off A/B of the existing spin setting after recording actual environment and worker threads. No source optimization has been selected. Bitwise numerical gates, matched repetitions, CPU-thread time and wake latency are needed before adoption. Separating the largest host window also requires a smaller-buffer dedicated ETW profile and verified GPU-table support. This needs a separate instruction; the current RAM guard is not weakened.

Reaggregation is `phase3p-saved-host-report.json/.txt`; protected start `phase3p-start-state.json`; healthy run `phase3p-normal-diag.*`; numerical gates `phase3p-normal-numerical.json` and `phase3p-normal-argmax.json`; counters `phase3p-process-counters.json`. Partial trace is `phase3p-etw-once.etl`, stop confirmation `phase3p-etw-once.done` and `-poststatus.txt`, and partial offline export `phase3p-offline/cli-once-report.md`. These remain under `build/windows-acpp/`; generated logs, traces, models, executables and toolchains are not committed. `phase3p-final-integrity.json` records unchanged working-engine/header/runtime hashes, matching loaded DLLs and inference sources. Shared SDK and recovery copies were not modified; that JSON does not claim revalidation of all their file hashes.

A single cause is not established. Phase 3P ends after organizing the largest host window, whole-process CPU burden, confirmed ceilings and missing evidence. No additional profiling GPU runs or optimization implementation follow. Only the adopted offline parser and bilingual documents are committed, then work stops. Phase 3D, cache re-enablement, WMMA, Command Graph, VMM, native in-order, DPC++ migration and driver/SDK updates were not started.

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

<a id="lane-affinity-measurement"></a>
## 2026-10-12: isolated AdaptiveCpp lane-affinity measurement

In AdaptiveCpp `0ea863abe1af4422593583f91e5608ef75044a16`, `multi_queue_executor::find_assigned_lane_index` compared the parent executor assigned to a DAG node against child executors, failing to find predecessor lanes. The candidate checks ownership, backend and device bounds, then matches the node's execution lane to the child queue. Kernel/copy lane separation and required cross-stream dependencies remain, as do XeStrata's DAG ordering and final-copy event wait. A separate all-lanes merge returned stale zero for all 248,320 D2H elements after a kernel; it was rejected without model performance runs.

On the Windows R9700, the isolated runtime (B) was compared with the working version (A) on the same 13 input IDs, 41 generated tokens and 53 positions. Diagnostic A/B runs bitwise matched all 13,160,960 logits and 26,593,280 residual floats, with zero nonfinite values. Three fresh no-dump processes per side ran A1→B1→A2→B2→A3→B3: median decode A 2.576→B 3.846 tok/s (+49.3%, 1.493×); median host Routing/Input wait 261.091→122.509 ms/position; CPU callback 100.624→101.673 ms/position. The wait includes GPU work, transfers and runtime; it is not a breakdown of GPU execution time. This is evidence for this fixed workload, not a 10× gain.

The change is scoped to `third_party/main/adaptivecpp/patches/0002-dag-lane-affinity.patch`, applied only to pinned AdaptiveCpp source. Reproduce and check it in separate source/build/prefix locations; do not overwrite the working installation. The working executable, SDK, driver and recovery copies remain intact. Local, Git-excluded evidence is in `build/windows-acpp/lane-affinity-20261012T084944Z/report.ja.md`, `result.json` and `.tools/lane-affinity-runtime/source.diff`. Other prompts, longer contexts, other GPUs/quantizations, real GPU-fault cleanup and a 4-GiB maximum-allocation limit are outside this check.

<a id="expert-cache-and-wsl"></a>
## 2026-10-13: VRAM expert cache enabled; the WSL 2 path does not work

### Why the expert cache was always 0

In the AdaptiveCpp build `strata::gpu::mem_info` always failed. `--expert-cache auto` read the free VRAM as 0 GiB and sized the cache at 0 slots. This is also why Phase 3C could try only 4 slots: the automatic sizing never worked. On a HIP-backend device the free VRAM now comes from HIP's `hipMemGetInfo`, loaded at run time (the build does not link HIP). On a backend HIP cannot answer for, the device total less what this process holds is used.

On Windows HIP, a device allocation takes the same amount of system commit. A 4 GiB `hipMalloc` lowered the free commit by 4.1 GiB, and `hipFree` gave it back; no physical RAM is used. So on Windows the free figure is the smaller of the free VRAM and the free commit less 4 GiB. Allocating 27 GiB of slots at once brought the free commit down to 98 MiB; with this bound, allocations stay within the free commit. A larger page file allows a larger cache.

On the same hello input (13 input IDs, `--max-new 64`, context 256, 3 workers, mmap, no VRAM limit or no-XMX setting), cache auto took 7,759 slots (9.99 GiB) and hit 19,091 of 25,440 requests (75.0%). The 41 generated tokens are the same as with the cache off. Decode was 3.22 to 4.87 tok/s, about the same as the cache-off 3.11 to 4.88 tok/s. This short run starts with an empty cache and reads 6,349 experts from the mmap into VRAM while it generates.

### The remaining floor: the GPU's time per layer

In cache-on Decode with `STRATA_POOL_TIMING=1`, the CPU pool fell to 0.09 ms per position, and the Routing/Input host wait stayed at about 115 ms per position (about 2.4 ms per layer). Decode with `--no-pool` (no CPU experts) is 10.60 tok/s (about 94 ms per token). The same engine decodes at about 49 tok/s on Linux on an RX 9060 XT (about half an R9700), so the Windows path loses mainly in the GPU's time per layer. The Linux path's command graphs, doorbell, native experts and batched prefill are absent from this path.

The native in-order queue (bypassing AdaptiveCpp's DAG scheduler) was also tried. Decode went from 4.88 to 6.02 tok/s, but the head's logits readback returned stale values and the argmax differed at 19 of 53 positions. The Routing wait was about 128 ms per position, the same as with the DAG path. It was rejected, and the DAG path stays.

### The WSL 2 path

XeStrata's Linux build was tried in WSL 2 (Ubuntu 24.04, ROCm 7.2.3, librocdxg). intel/llvm v7.1.1 with the HIP target and the engine built. On the R9700 (gfx1201), however, `hipMalloc` succeeds but the first queue creation, kernel launch or event record never completes. The same minimal HIP test passes on the 890M integrated GPU (gfx1150).

librocdxg's debug log shows that after ROCm's first internal kernel dispatch is converted to PM4 and submitted, the queue's read pointer on the R9700 never moves from 0. librocdxg's `src/wddm/cmd_util.cpp` (the same in its current home, ROCm/rocm-systems) has no gfx12 branch: the cache flush uses one bit layout for gfx10 and later, and `PRIV` is set for gfx11 only. librocdxg 1.2.2, the newest, behaves the same, and setting `PRIV` for gfx12 as well made Windows stop with bugcheck 0xD1 (`dxgmms2.sys`). The Windows driver is 32.0.32015.2008 (2026-09-28), newer than the pairing AMD supports for WSL (Adrenalin 26.2.2 with ROCm 7.2.1).

The R9700 cannot be used through WSL on this machine, so work continues on the native Windows path. The setup change that finds AMD GPUs from `rocminfo` under WSL 2 is not committed, because of this bugcheck risk. `xmx_probe` running on the CPU for about 1 h 45 min under WSL after the intel/llvm build also needs a separate fix.

<a id="pool-graphs"></a>
## 2026-10-13: the pool path replays its layers as HIP graphs

### What the time per layer was

A small SYCL test measured the launch cost on AdaptiveCpp's HIP backend. Even an empty kernel, submitted back to back, cost about 38 µs (DAG ordering) or 26 to 35 µs (native in-order). Plain HIP, without AdaptiveCpp, launched the same kernel in 1.7 µs, and in 34 µs with an event recorded after each kernel. AdaptiveCpp records a completion event for every operation, so on Windows HIP each kernel appears to take a submission of its own. Kernels with no event after them sometimes left no result even though the synchronization succeeded (a Windows HIP behaviour; possibly the same root as the stale readback of native in-order).

HIP's stream capture recorded SYCL kernels submitted to an AdaptiveCpp native in-order queue into a graph. Replaying a graph of 40 kernels cost 3 to 8 µs per kernel, with the right values.

### The change

- In the AdaptiveCpp build, the `strata::gpu` capture API is implemented with HIP's stream capture (`begin_capture`, `end_capture`, `abandon_capture`, `is_capturing`, `graph_launch`, `graph_destroy`). HIP is loaded at run time; the build does not link it. Recording is on the one HIP stream of a native in-order queue (`stream_create_capture`); a replay calls `hipGraphLaunch` from a custom operation on the engine's DAG queue, so AdaptiveCpp keeps it in order with what comes before and after.
- `session_capture` takes a `doorbell` argument and can record `pre` and `post` without the doorbell. Given graphs, `session_token_pool` replays `pre[l]` and `post[l]` for each layer and passes the position through `stage_token`. The routing D2H, the CPU pool, the hit path and the dumps stay outside the graphs.
- It is the default in the AdaptiveCpp build (`STRATA_POOL_GRAPHS=0` turns it off). Other builds use it only with `STRATA_POOL_GRAPHS=1`.

### Results

`pre` has 2,197 nodes over 48 layers (about 46 a layer), `post` 96. On the hello input with cache 0, runs with and without graphs alternated twice:

| | without graphs | with graphs |
| --- | --- | --- |
| Decode | 4.65, 4.61 tok/s | 7.15, 7.05 tok/s |
| Prefill | 4.08, 4.10 tok/s | 5.98, 6.00 tok/s |

All logits matched the run without graphs bitwise (53 positions of hello, 66 of the arithmetic input). The Routing wait fell from about 115 to about 95 ms per position and the CPU's submissions from about 14 to about 3 ms. The remaining ~2 ms per layer is most likely the kernels' own time (launching 46 nodes is about 0.4 ms).

On this PC the expert cache made it slower. The free commit limits it to about 4,900 slots (about 20% of the experts); filled from the profile at start it hit 48%, and Decode was 4.3 tok/s. The hits are computed outside the graphs, in series with the GPU's layers, and pay the launch cost per kernel. The CPU pool takes about 30 ms per position with graphs, so the cache adds more than it moves.

`--stage-timing` returns zeros in the AdaptiveCpp build, so the GPU time per kernel is not measured yet. Unverified: Linux free and contrib-llvm (`session_capture`'s change keeps the old behaviour by default), NVIDIA, integrated graphics, long contexts, other quantizations.
