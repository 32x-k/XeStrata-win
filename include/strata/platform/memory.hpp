// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// include/strata/platform/memory.hpp - plan v0.3 P0.1/P1: keep a large host region resident.
//
// If CUDA cannot register the expert arena, mlock can keep it resident while the CPU pool reads it.
#pragma once

#include <cstdint>
#include <string>

namespace strata::platform {

struct LockResult {
    bool ok = false;
    uint64_t locked_bytes = 0;   ///< may be less than requested; the rest stays pageable
    std::string note;            ///< what was done or why it failed, for the startup print
};

/// Lock [p, p + bytes) into physical memory. Partial success is reported, not hidden.
LockResult lock_resident(void* p, uint64_t bytes);

/// Undo lock_resident for the same region (best effort).
void unlock_resident(void* p, uint64_t bytes);

/// #243, Windows: the GPU's shared (non-local) memory budget and this process's use of it, from DXGI
/// (IDXGIAdapter3::QueryVideoMemoryInfo, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL) for the adapter whose LUID is the
/// 8 bytes at `luid` (the device LUID Level Zero reports).  Page-locked host memory the GPU maps is charged there.
/// False (and `why` says so) when the query is not possible - always elsewhere than Windows.
bool gpu_shared_memory_budget(const void* luid, uint64_t& budget, uint64_t& usage, std::string& why);

/// Windows: the GPU's own memory (VRAM) budget for this process and its use of it, from DXGI
/// (DXGI_MEMORY_SEGMENT_GROUP_LOCAL), for the adapter at the PCI address `pci` ("0000:0c:00.0", as the SYCL runtime
/// gives it; the adapter's bus, device and function from D3DKMT).  WDDM moves what a process holds past this budget
/// into system memory instead of refusing it, so a device's own free memory can say more than the process can take:
/// an RX 9060 XT (16 GB) reported 1.5 GB free through HIP while WDDM had moved 0.8 GB of the process out, and an
/// image's decode took 121 s instead of 12.6 (Windows 11, 2026-10-26).  False (and `why` says so) when the query is
/// not possible - always elsewhere than Windows.
bool gpu_local_memory_budget(const std::string& pci, uint64_t& budget, uint64_t& usage, std::string& why);

/// Windows: the same for the GPU's shared (non-local) segment, where WDDM charges the host memory a process pins for
/// that GPU (registered for its copies, or host USM): about half the RAM.  False (and `why`) as above.
bool gpu_nonlocal_memory_budget(const std::string& pci, uint64_t& budget, uint64_t& usage, std::string& why);

/// The machine's physical RAM in bytes (0 when unknown).
uint64_t total_physical_memory();

/// Linux: the free CMA memory (/proc/meminfo's CmaFree; 0 when there is none, and elsewhere).  MemAvailable counts it,
/// but a page-locked allocation cannot use it: the kernel moves long-term pinned pages out of CMA (upstream #3698: a
/// kernel 7.0 with KHO's ~17 GiB of scratch CMA passed the RAM checks, and the pinned memory froze the host).
uint64_t cma_free_bytes();

/// Linux under WSL 2 (Microsoft's kernel, "...-microsoft-standard-WSL2" in /proc/sys/kernel/osrelease): the GPU is
/// Windows' own, behind /dev/dxg.  Always false elsewhere.
bool under_wsl();

/// Whether `advise_willneed` asks the OS for anything: not on Windows, nor with STRATA_READ_AHEAD=0.
bool read_ahead_enabled();
/// Asks the OS to start reading [p, p + bytes) of a file mapping, without waiting.  Linux reads at most one
/// readahead window per request, so the range is asked for in 128 KiB steps.
void advise_willneed(const void* p, uint64_t bytes);
/// The same for [offset, offset + bytes) of an open file.
void advise_willneed(int fd, uint64_t offset, uint64_t bytes);
/// Asks the OS to drop the file at `path` from its file cache (the pages no process maps).  Nothing on Windows.
void drop_file_cache(const std::string& path);
/// The RAM the OS has free and the file cache holds, in bytes (Linux's MemFree and Cached); false elsewhere.
bool free_and_cached(uint64_t& free, uint64_t& cached);
/// Makes the OS free RAM until `bytes` are free (Linux's MemFree), as far as it can give them back (MemAvailable, less
/// 1 GiB): memory taken, touched and given back, so the file cache leaves it.  Nothing on Windows.
void make_free_ram(uint64_t bytes);
/// Waits until the kernel has stopped migrating host pages (Linux's /proc/vmstat pgmigrate_success: below `quiet_rate`
/// pages a second for `quiet_s` seconds), at most `cap_s` seconds; the seconds waited, and the highest rate seen in
/// `peak`.  0 at once where the counter cannot be read (Windows, a kernel without it).
double settle_page_migration(double cap_s, double quiet_s, double quiet_rate, double& peak);
/// The engine's wait for that before it says READY (STRATA_SETTLE_S, memory.cpp), with a line on stderr.
void settle_before_ready();

}  // namespace strata::platform
