<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# Temporary AdaptiveCpp event patch

Only AdaptiveCpp commit `0ea863abe1af4422593583f91e5608ef75044a16` is supported. AdaptiveCpp itself is not bundled; use an existing local checkout. Nothing is downloaded.

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

The patch changes no queue ordering, DAG behavior, normal arithmetic or XeStrata engine source. Default-constructed-event handler behavior, the process-global error list, and cancellation/uncertain-DMA cleanup remain separate issues. This test does not establish GPU-fault cleanup or real-model correctness.
