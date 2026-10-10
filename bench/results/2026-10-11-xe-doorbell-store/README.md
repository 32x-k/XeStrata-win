# GPU-to-host doorbell store without a host-USM atomic — 2026-10-11

The doorbell's ring (`doorbell_ring`, `doorbell_publish`) and the start-up probe (`doorbell_visible`) stored a host-USM word with a system-scope atomic.
No GPU on the development machine reports `usm_atomic_host_allocations`, and the xe driver serves a discrete GPU's atomic page fault on system memory only by moving the page to VRAM (`xe_vma_need_vram_for_atomic`); a read or a write is bound again.
Issue #2 (an Arc Pro B60) faulted with `AccessType: 2` (`XE_PAGEFAULT_ACCESS_TYPE_ATOMIC`) at a host virtual address.
This probe checks whether the host sees a plain volatile store from a kernel that is still running.

| Probe | What it checks | Result |
| --- | --- | --- |
| [store_probe.cpp](store_probe.cpp) ([output](store_probe.txt)) | A kernel stores a host-USM ring, then spins on a host-USM answer; the host polls the ring while the kernel runs (1 s limit), then answers. Mode 0: system release fence, then a system-scope atomic store. Mode 1: system release fence, then a volatile store. Mode 2: volatile store, then a system release fence. 2,000 direct launches and 2,000 graph replays a mode | B70: every mode seen in 2,000 of 2,000 runs, both ways. Median ring seen, direct: 11.0 µs (mode 0), 4.9 µs (mode 1), 4.8 µs (mode 2); graph: 7.4, 6.5, 6.4 µs. UHD 770: no mode seen while the kernel ran (its kernels do not see the host's writes either; the engine runs its windows as segments there), and the spinning kernel was then reset (`UR_RESULT_ERROR_DEVICE_LOST`) |

- Machine: the B70 (`0000:03:00.0`) and the UHD 770 of the i7-14700, kernel 7.0.0-38, compute runtime 26.05.37020.
- Built with intel/llvm 7.1 (`.tools/intel-llvm`): `clang++ -std=c++20 -O2 -fsycl -o store_probe store_probe.cpp`, run as `store_probe 0` (B70) and `store_probe 1` (UHD 770).
- A first version that chose the mode with a template crashed intel/llvm's `sycl-optimize-barriers` pass (an assertion in `CallBase::setArgOperand`), as did a release fence right after `read_host`'s acquire fence in `elementwise.cpp`.
- Whether the plain store avoids the B60's fault is `unverified`: there is no B60 here.
