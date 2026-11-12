# Project Maya (maya-ds4), DeepSeek-V4.1 engine

The files under `src/dsv41/`, `include/strata/dsv41/`, `tests/dsv41/` and `tools/dsv41/` come from
https://github.com/mw00/project-maya (release v1.1.17, commit 8d9a9b5d), its DeepSeek engine `src/ds4`, `tests/ds4_*`,
`tools/ds4` and the kernel sources it builds on (`exl3`, `sampler`). Project Maya is itself built on Strata.

Copyright (c) 2026 Peasant Smith (mw00) and the Project Maya contributors, including needmorevram.
MIT License, the full text is in `LICENSE` beside this file. The notice stays with every copy.

Changes made while porting into Strata are listed in the git history of the `src/dsv41` directory
(the first commit of each file is the unmodified upstream copy).

Later changes taken from Project Maya releases v1.1.18 to v1.1.26 (the same approach re-written onto Strata's engine,
for example the long-prompt layer skip, the thinking budget, conversation slots) are covered by the same copyright
and permission notice, in `LICENSE` beside this file.

The batches-path selection of streamed experts follows Project Maya #691 (needmorevram),
Strata commit 54eff4875, and the file-backed default in Project Maya commit 8563f6bba (mw00).
XeStrata rewrites that approach for its SYCL queues without changing virtual slots, batches or summation order.

The SYCL DeepSeek vision tower follows the geometry, BF16 rounding stages and image span contract of
Project Maya's vision tower and Strata's port. It uses XeStrata's shared dense products and its own SYCL kernels.
The transient workspace follows the lifetime described in Project Maya commit 6fba6be981:
weights remain on the host and GPU buffers are returned after each image. Buffer sizes and weight stripes
are derived from the image, model and device reports rather than the CUDA implementation's fixed budgets.

The device-side wait for a CPU lane reply follows the sequence wait of Strata's and Project Maya's DeepSeek kernels.
XeStrata selects it from the device's reported atomics and fences, keeps the host task elsewhere,
and adds a host-visible release word for cancellation.
