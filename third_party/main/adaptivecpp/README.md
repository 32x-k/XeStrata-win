<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# Local AdaptiveCpp Windows patches

Only AdaptiveCpp commit `0ea863abe1af4422593583f91e5608ef75044a16` is supported. AdaptiveCpp itself is not bundled; use an existing local checkout. Nothing is downloaded. Patch 0001 fixes event handlers; 0002 fixes lane lookup.

`patches/0001-event-async-handler.patch` changes one line to retain the supplied `async_handler` in the node-bearing `sycl::event` constructor. The patch carries AdaptiveCpp's BSD-2-Clause license and leaves the original copyright/license header untouched. The original license is kept in `LICENSE`. XeStrata's Python build tooling and this document are LGPL-3.0-or-later.

## Apply before building

Use `tools/adaptivecpp_build.py` as the entry point for a standalone AdaptiveCpp build.

```text
python tools/adaptivecpp_build.py --source PATH_TO_ADAPTIVECPP --build build/acpp-eventfix --prefix .tools/acpp-eventfix --receipt build/acpp-eventfix-receipt.json --cmake-arg=-G --cmake-arg=Ninja
```

Specify compiler, CPU/GPU backends and LLVM/OpenMP locations with `--cmake-arg=-DNAME=VALUE`. On the current Windows machine, use a Visual Studio developer shell. Building as an LLVM component is separate from this standalone build; run `--apply-only` before configuring it, then `--verify-installed-only --prefix PATH` after installing.

The wrapper performs these steps:

1. Check that checkout HEAD equals the pinned commit and its committed `event.hpp` matches the pinned SHA-256.
2. Require the working header to match the complete original or patched file. Only the exact LF and CRLF variants are accepted; mixed endings or other edits are refused.
3. Apply after `git apply --check`, or reverse-check and skip if already applied. No user changes are reset, reverted or deleted.
4. Run CMake configure, build and install. Refuse overlapping source/build/prefix paths, an existing cache for another source/prefix, an existing unpatched installation, and receipt overwrite.
5. Verify the complete patched `PREFIX/include/AdaptiveCpp/hipSYCL/sycl/event.hpp`. Record commit SHA, patch/header SHA-256 values and disposition as JSON.

Stop on inconsistency or command failure. Direct CMake execution bypasses these checks and is not the update entry point. Preserve the working source/build/prefix for recovery and use separate locations.

The committed LF original's SHA-256 is `0576c693880346dcd6d85ec3934be332c8e1678af7f126c267d603916338b2ad`; the Windows CRLF original's is `0eb54a547e68eaf0055f75431fbc8af7186e7221988d0829c333acec76797526`. Distinguish Git blob and working-file hashes. The hunk omits trailing context to avoid whitespace-only context lines. The wrapper uses `--unidiff-zero` only after full-file hash validation and checks the entire postimage; it does not blindly apply to unknown headers.

## Validation and scope

`python -m unittest tools.test_adaptivecpp_build -v` uses local Git fixtures to test apply/skip, wrong SHA, modified headers, installation mismatch and protection checks.

`tests/platform/adaptivecpp_event_async.cpp` is an isolated test using real installed headers/runtime. Run with a separate CPU-only prefix; do not install or discover GPU backends. It uses no device queue, device allocation, kernel or DMA. Controlled synthetic runtime errors test handler counts, original diagnostics, SYCL errc, batching, event copies, no redelivery and throwing handlers. `--expect-broken`, using the same CPU runtime with the original event header, is the negative control.

Patch 0001 changes no queue ordering, DAG behavior, normal arithmetic or XeStrata engine source. Default-constructed-event handler behavior, the process-global error list, and cancellation/uncertain-DMA cleanup remain separate issues. This test does not establish GPU-fault cleanup or real-model correctness.

<a id="lane-affinity-patch-0002"></a>
## 0002: lane-lookup scope and isolated build

`patches/0002-dag-lane-affinity.patch` targets `src/runtime/multi_queue_executor.cpp` at the same pinned commit; its runtime behavior change is confined to `find_assigned_lane_index` (it also adds license attribution). For DAG nodes assigned to the parent multi_queue_executor, it identifies a predecessor lane by matching the node's execution lane to a child executor's queue, rather than asking whether the child owns the node. Ownership, backend and device bounds are checked first so foreign nodes' uninitialized lanes are not read. It does not merge kernel/copy lanes, remove necessary cross-stream dependencies or remove XeStrata's final-copy wait. The original-source difference is saved at `.tools/lane-affinity-runtime/source.diff`.

For reproduction, apply 0001 in a separate checkout, then run `python tools/adaptivecpp_lane_affinity.py --source PATH_TO_ISOLATED_ADAPTIVECPP --apply-only`. After building and installing, run `--verify-only` against the same source. The tool checks the pinned commit and SHA-256 of the complete original and patched source; it stops on a mismatch. It does not build or install. Use separate source/build/prefix locations; never overwrite the working installation, 0001 validation environment or recovery copies. The 0001 instructions alone do not establish application or verification of 0002. A newly built binary differs from the measured isolated DLL and needs a numerical gate before use.

On the Windows R9700, working A and isolated runtime B matched bitwise across 53 positions, 41 generated tokens, all 13,160,960 logits and 26,593,280 residual floats. Three alternating runs per side gave median decode 2.576→3.846 tok/s (+49.3%), host Routing/Input wait 261.091→122.509 ms/position and CPU callback 100.624→101.673 ms/position. A separate all-lanes merge was rejected after all 248,320 elements returned stale zero. This is neither a 10× gain nor evidence for other workloads.
