// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Compile/link with src/platform/direct_file.cpp and the real Windows SDK (no fakeWindows).
// Run from the repository root, under an external 20-second job/watchdog. No GPU/model access.
// Only one CREATE_NEW-owned 32785-byte file, <=8 outstanding reads, issuer counts 1 and 3.
// At most 72 accepted reads in total; the one reusable _aligned_malloc bank is 64 KiB.
// wait is bounded; close deliberately is not: a timeout must never release live I/O buffers.
#ifndef _WIN32
#include <cstdio>
int main() {
    std::puts("SKIP: requires real Windows");
    return 77;
}
#else
#include "strata/platform/direct_file.hpp"
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <malloc.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <string>

namespace {
using strata::platform::Completion;
using strata::platform::DirectFile;
constexpr size_t kPage = DirectFile::alignment();
constexpr size_t kReads = 8;
constexpr size_t kFileBytes = kReads * kPage + 17;
constexpr unsigned char kUntouched = 0xcc;
constexpr unsigned char kGuard = 0xd3;
static_assert(kFileBytes <= 65536, "Keep the owned file tiny");

bool check(bool condition, const char* message) {
    if (!condition) std::fprintf(stderr, "FAIL: %s (Win32 last error=%lu)\n", message, GetLastError());
    return condition;
}

struct TinyFile {
    std::string path;
    std::array<unsigned char, kFileBytes> reference{};
    bool owned = false;

    TinyFile() = default;
    TinyFile(const TinyFile&) = delete;
    TinyFile& operator=(const TinyFile&) = delete;
    ~TinyFile() { remove(); }

    bool remove() noexcept {
        if (!owned) return true;
        if (!DeleteFileA(path.c_str())) return check(false, "delete owned tiny file");
        owned = false;
        return true;
    }

    bool create() {
        // ASCII relative name: both CreateFileA and DirectFile's UTF-8 open resolve the same file.
        // CREATE_NEW fails rather than overwriting any existing file, even on a name collision.
        path = "build/windows-acpp/direct-file-real/owned-" + std::to_string(GetCurrentProcessId()) +
               "-" + std::to_string(GetTickCount64()) + ".bin";
        char absolute[MAX_PATH]{};
        char volume[MAX_PATH]{};
        DWORD sectors = 0, bytes_per_sector = 0, free_clusters = 0, clusters = 0;
        const DWORD length = GetFullPathNameA(path.c_str(), MAX_PATH, absolute, nullptr);
        if (!check(length != 0 && length < MAX_PATH, "resolve owned file path") ||
            !check(GetVolumePathNameA(absolute, volume, MAX_PATH) != FALSE, "resolve volume") ||
            !check(GetDiskFreeSpaceA(volume, &sectors, &bytes_per_sector, &free_clusters, &clusters) != FALSE,
                   "query sector alignment") ||
            !check(bytes_per_sector != 0 && kPage % bytes_per_sector == 0,
                   "4096-byte DirectFile alignment must satisfy this volume's sector alignment")) return false;
        std::printf("sector=%lu; file=%zu bytes; buffers=65536 bytes; <=8 reads in flight\n",
                    bytes_per_sector, kFileBytes);
        // Independent, position-sensitive oracle, including distinct bytes on different pages.
        for (size_t i = 0; i < reference.size(); ++i)
            reference[i] = static_cast<unsigned char>(((i * 37u + 11u) ^ (i >> 7u) ^ (i >> 12u)) & 0xffu);
        HANDLE writer = CreateFileA(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW,
                                    FILE_ATTRIBUTE_NORMAL, nullptr);
        if (!check(writer != INVALID_HANDLE_VALUE, "CREATE_NEW owned file")) return false;
        owned = true;
        DWORD written = 0;
        const bool wrote = WriteFile(writer, reference.data(), static_cast<DWORD>(reference.size()),
                                     &written, nullptr) != FALSE;
        const bool flushed = FlushFileBuffers(writer) != FALSE;
        const bool closed = CloseHandle(writer) != FALSE;
        return check(wrote && written == reference.size() && flushed && closed, "write/flush/close reference file");
    }
};

struct Buffers {
    // Each payload has a whole sector-aligned guard page after it; all reads use distinct payloads.
    unsigned char* memory = static_cast<unsigned char*>(_aligned_malloc(kReads * 2 * kPage, kPage));
    Buffers() { if (memory) reset(); }
    ~Buffers() { _aligned_free(memory); }
    Buffers(const Buffers&) = delete;
    Buffers& operator=(const Buffers&) = delete;
    unsigned char* data(size_t slot) { return memory + slot * 2 * kPage; }
    void reset(unsigned char value = kUntouched) {
        for (size_t i = 0; i < kReads; ++i) {
            std::memset(data(i), value, kPage);
            std::memset(data(i) + kPage, kGuard, kPage);
        }
    }
    bool guards() {
        for (size_t i = 0; i < kReads; ++i)
            for (size_t j = 0; j < kPage; ++j)
                if (data(i)[kPage + j] != kGuard) return check(false, "read-buffer guard overwritten");
        return true;
    }
};

bool same(const Completion& a, const Completion& b) {
    return a.tag == b.tag && a.bytes == b.bytes && a.ok == b.ok;
}

template<size_t Capacity>
int guarded_wait(DirectFile& file, std::array<Completion, 2>& result, int timeout) {
    static_assert(Capacity == 1 || Capacity == 2, "Only small completion batches");
    constexpr Completion sentinel{0xfedcba9876543210ull, 0xabcdef12u, false};
    std::array<Completion, Capacity + 2> output;
    output.fill(sentinel);
    const int count = file.wait(output.data() + 1, static_cast<int>(Capacity), timeout);
    if (!check(count >= 0 && count <= static_cast<int>(Capacity), "wait count outside caller capacity") ||
        !check(same(output.front(), sentinel) && same(output.back(), sentinel), "completion canary overwritten"))
        return -1;
    for (size_t i = static_cast<size_t>(count) + 1; i <= Capacity; ++i)
        if (!check(same(output[i], sentinel), "wait wrote an unreported completion")) return -1;
    for (int i = 0; i < count; ++i) result[static_cast<size_t>(i)] = output[static_cast<size_t>(i) + 1];
    return count;
}

int wait(DirectFile& file, int capacity, std::array<Completion, 2>& result, int timeout) {
    return capacity == 1 ? guarded_wait<1>(file, result, timeout) : guarded_wait<2>(file, result, timeout);
}

bool open(DirectFile& file, const TinyFile& tiny) {
    std::string error;
    if (!file.open(tiny.path, error)) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return false; }
    return check(file.is_open() && file.size() == kFileBytes, "open/size");
}

bool submit(DirectFile& file, Buffers& buffers, size_t slot, size_t offset, uint64_t tag) {
    std::string error;
    if (file.submit(static_cast<uint64_t>(offset), buffers.data(slot), static_cast<uint32_t>(kPage), tag, error))
        return true;
    std::fprintf(stderr, "FAIL: aligned submit: %s\n", error.c_str());
    return false;
}

bool quiet(DirectFile& file, int capacity) {
    // EOF's immediate-error path can add a WAKE_TAG nudge. No other unsolicited result is legal.
    for (unsigned poll = 0; poll < 4; ++poll) {
        std::array<Completion, 2> result{};
        const int count = wait(file, capacity, result, 10);
        if (count < 0) return false;
        if (count == 0) return true;
        for (int i = 0; i < count; ++i) {
            const auto& c = result[static_cast<size_t>(i)];
            if (!check(c.tag == DirectFile::WAKE_TAG && c.ok && c.bytes == 0,
                       "duplicate/stale/unsolicited completion")) return false;
        }
    }
    return check(false, "too many unexpected wake packets");
}

bool collect(DirectFile& file, Buffers& buffers, const TinyFile& tiny, int capacity,
             uint64_t first_tag, size_t count, bool eof, ULONGLONG deadline) {
    std::array<bool, kReads> seen{};
    size_t retired = 0;
    unsigned wakes = 0;
    while (retired < count) {
        if (!check(GetTickCount64() < deadline, "bounded completion deadline expired; closing before buffer release"))
            return false;
        std::array<Completion, 2> result{};
        const int got = wait(file, capacity, result, 50);
        if (got < 0) return false;
        for (int i = 0; i < got; ++i) {
            const auto& c = result[static_cast<size_t>(i)];
            if (c.tag == DirectFile::WAKE_TAG) {
                if (!check(c.ok && c.bytes == 0 && ++wakes <= 4, "invalid/excess wake result")) return false;
                continue;
            }
            if (!check(c.tag >= first_tag && c.tag - first_tag < count, "unexpected completion tag")) return false;
            const size_t slot = static_cast<size_t>(c.tag - first_tag);
            if (!check(!seen[slot], "duplicate completion tag")) return false;
            seen[slot] = true;
            ++retired;
            const size_t offset = eof && slot >= 6 ? (slot + 2) * kPage : slot * kPage;
            const size_t bytes = offset >= kFileBytes ? 0 : (kFileBytes - offset < kPage ? kFileBytes - offset : kPage);
            if (!check(c.ok && c.bytes == bytes, "incorrect success/EOF status or byte count") ||
                !check(std::memcmp(buffers.data(slot), tiny.reference.data() + (bytes ? offset : 0), bytes) == 0,
                       "read differs from independent reference")) return false;
            // Unbuffered Windows reads may touch/zero-pad the requested capacity beyond c.bytes.
            // Only reported bytes are payload; guards() checks outside the full kPage request.
        }
    }
    return buffers.guards() && quiet(file, capacity);
}

bool invalid_alignment(DirectFile& file, Buffers& buffers, int capacity) {
    struct Invalid { uint64_t offset; void* buffer; uint32_t bytes; };
    const std::array<Invalid, 4> invalid{{
        {1, buffers.data(0), static_cast<uint32_t>(kPage)},
        {0, buffers.data(0) + 1, static_cast<uint32_t>(kPage)},
        {0, buffers.data(0), static_cast<uint32_t>(kPage - 1)},
        {0, buffers.data(0), 0}
    }};
    for (const auto& request : invalid) {
        std::string error;
        if (!check(!file.submit(request.offset, request.buffer, request.bytes, 9000, error) && !error.empty(),
                   "invalid alignment/zero length not rejected") || !quiet(file, capacity)) return false;
    }
    // The API rejects these before queueing, so retirement means *no* completion or buffer write,
    // not a fabricated native error packet. The following valid batch proves continued usability.
    for (size_t i = 0; i < kReads; ++i)
        for (size_t j = 0; j < kPage; ++j)
            if (!check(buffers.data(i)[j] == kUntouched, "rejected request wrote buffer")) return false;
    return buffers.guards();
}

bool reopen_read(DirectFile& file, Buffers& buffers, const TinyFile& tiny, int capacity,
                 uint64_t tag, ULONGLONG deadline) {
    // Called only after close/destruction drained: now poisoning/reusing the allocation is safe.
    buffers.reset();
    if (!open(file, tiny) || !quiet(file, capacity) || !submit(file, buffers, 0, 0, tag) ||
        !collect(file, buffers, tiny, capacity, tag, 1, false, deadline)) return false;
    file.close();
    for (size_t i = 1; i < kReads; ++i)
        for (size_t j = 0; j < kPage; ++j)
            if (!check(buffers.data(i)[j] == kUntouched, "late write from old handle after reopen")) return false;
    return buffers.guards();
}

bool batch(const TinyFile& tiny, int capacity, ULONGLONG deadline) {
    // Declaration order is intentional: file closes/drains before buffers on EVERY return/exception.
    Buffers buffers;
    if (!check(buffers.memory != nullptr, "_aligned_malloc")) return false;
    DirectFile file;
    if (!open(file, tiny) || !invalid_alignment(file, buffers, capacity)) return false;
    file.wake(); // Mix a real null IOCP packet with successful reads and EOF retirement.
    for (size_t i = 0; i < kReads; ++i) {
        const size_t offset = i >= 6 ? (i + 2) * kPage : i * kPage;
        if (!submit(file, buffers, i, offset, 100 + i)) return false;
    }
    if (!collect(file, buffers, tiny, capacity, 100, kReads, true, deadline)) return false;
    file.close();
    if (!check(!file.is_open() && file.size() == 0, "closed state") || !buffers.guards()) return false;
    return reopen_read(file, buffers, tiny, capacity, 200, deadline);
}

bool shutdown(const TinyFile& tiny, bool destroy, ULONGLONG deadline) {
    Buffers buffers; // Must also outlive unique_ptr reset and destruction on exception.
    if (!check(buffers.memory != nullptr, "_aligned_malloc")) return false;
    auto file = std::make_unique<DirectFile>();
    if (!open(*file, tiny)) return false;
    for (size_t i = 0; i < kReads; ++i)
        if (!submit(*file, buffers, i, i * kPage, 300 + i)) return false;
    // No wait/sleep between submit and close: queued, pending, or already terminal are all legal.
    // DirectFile::close invokes CancelIoEx internally. Cancellation may lose to success; neither
    // a particular cancellation status nor a pending kernel operation is guaranteed on a tiny file.
    if (destroy) file.reset(); else file->close();
    if (!buffers.guards()) return false;
    if (destroy) file = std::make_unique<DirectFile>();
    return reopen_read(*file, buffers, tiny, 2, 400, deadline);
}

bool run(const TinyFile& tiny) {
    const ULONGLONG deadline = GetTickCount64() + 10000; // One total collection budget, not per read.
    for (const char* issuers : {"1", "3"}) {
        // Process-local setting only; inherited stress settings cannot increase this test's pool.
        if (!check(_putenv_s("STRATA_IO_THREADS", issuers) == 0, "set bounded issuer count")) return false;
        std::printf("issuerThreads=%s: max_completions=1,2; close/destruction/reopen\n", issuers);
        if (!batch(tiny, 1, deadline) || !batch(tiny, 2, deadline) ||
            !shutdown(tiny, false, deadline) || !shutdown(tiny, true, deadline)) return false;
    }
    return true;
}
} // namespace

int main() {
    // Do not throw for assertions: early failure unwinds DirectFile before its buffer owner.
    // Unexpected C++ exceptions follow the same RAII ordering, then delete only our owned file.
    std::setvbuf(stdout, nullptr, _IONBF, 0); // Preserve phase diagnostics if the external job must kill a hang.
    bool passed = false;
    TinyFile tiny;
    try {
        passed = tiny.create() && run(tiny);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: exception after safe I/O unwind: %s\n", error.what());
    } catch (...) {
        std::fprintf(stderr, "FAIL: unknown exception after safe I/O unwind\n");
    }
    if (!tiny.remove()) passed = false;
    std::puts(passed ? "PASS: real Windows DirectFile; aligned data, max=1/2 canaries, EOF, rejected alignment, close/destruction/reopen"
                     : "FAIL: real Windows DirectFile");
    return passed ? 0 : 1;
}
#endif
