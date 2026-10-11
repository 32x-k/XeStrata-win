// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Link with the actual src/platform/direct_file.cpp, using direct_file_win32_fake/windows.h first.
// All file and IOCP calls below are simulations; no real asynchronous I/O is performed.
#include "strata/platform/direct_file.hpp"
#include "windows.h"
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <set>
#include <thread>
#include <vector>

using strata::platform::Completion;
using strata::platform::DirectFile;
static void check(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }

namespace fake {
constexpr ULONG_PTR pending_status = 0x103;
constexpr ULONG_PTR eof_status = 0xc0000011;
constexpr ULONG_PTR abort_status = 0xc0000120;
enum class Kind { File, Port };
enum class Mode { Pending, Sync, Eof, Fail };
enum class Cancel { Request, Fail, Race };
struct Handle {
    Kind kind;
    bool closed = false;
    Handle* port = nullptr;
    std::deque<OVERLAPPED_ENTRY> packets;
};
struct Op {
    Handle* file;
    void* buffer;
    bool terminal = false;
    bool dequeued = false;
    bool cancel = false;
    DWORD bytes = 0;
    ULONG_PTR status = pending_status;
};
std::mutex mu;
std::condition_variable cv;
std::vector<std::unique_ptr<Handle>> handles;
std::map<OVERLAPPED*, Op> ops;
std::map<std::uint64_t, Mode> modes;
std::set<OVERLAPPED*> requests;
std::function<void()> dequeue_hook;
Cancel cancellation = Cancel::Request;
unsigned reads = 0, nudges = 0, waits = 0, cancels = 0, close_waits = 0, dequeued = 0;
unsigned ex_errors_left = 0, single_errors_left = 0;
unsigned ex_errors = 0, single_errors = 0, single_waits = 0, single_failed_reads = 0;
thread_local DWORD error = 0;
Handle* get(HANDLE h, Kind kind) {
    auto* p = static_cast<Handle*>(h);
    check(p && !p->closed && p->kind == kind, "invalid fake handle");
    return p;
}
HANDLE make(Kind kind) {
    auto p = std::make_unique<Handle>();
    p->kind = kind;
    HANDLE h = p.get();
    handles.push_back(std::move(p));
    return h;
}
template<class Predicate> void await(Predicate pred) {
    std::unique_lock<std::mutex> lk(mu);
    check(cv.wait_for(lk, std::chrono::seconds(5), pred), "fake synchronization timeout");
}
void terminal(OVERLAPPED* ov, Op& op, ULONG_PTR status, DWORD bytes) {
    check(!op.terminal && !op.file->closed && !op.file->port->closed, "terminal lifetime violation");
    check(ov->hEvent == nullptr && ov->Internal == pending_status, "OVERLAPPED changed before terminal");
    // This is the simulated last buffer write. close must not return before it happens.
    if (bytes) static_cast<unsigned char*>(op.buffer)[0] = 0x5a;
    op.terminal = true;
    op.bytes = bytes;
    // Model IOCP's documented delayed status-block update: only dequeue publishes it.
    op.status = status;
    op.file->port->packets.push_back({0, ov, status, bytes});
    cv.notify_all();
}
void finish_all() {
    std::lock_guard<std::mutex> lk(mu);
    for (auto& [ov, op] : ops) if (!op.terminal) {
        // Cancellation may complete with success, cancellation or another terminal failure/EOF.
        const unsigned i = ov->Offset / 4096;
        const ULONG_PTR status = i % 3 == 0 ? 0 : (i % 3 == 1 ? abort_status : eof_status);
        terminal(ov, op, status, status == 0 ? 4096 : 0);
    }
}
OVERLAPPED_ENTRY pop(Handle* port) {
    auto entry = port->packets.front(); port->packets.pop_front();
    if (entry.lpOverlapped) {
        auto& op = ops.at(entry.lpOverlapped);
        check(op.terminal && !op.dequeued, "duplicate/nonterminal packet");
        entry.lpOverlapped->Internal = op.status;
        entry.lpOverlapped->InternalHigh = op.bytes;
        op.dequeued = true; ++dequeued;
    }
    return entry;
}
void clean() {
    std::lock_guard<std::mutex> lk(mu);
    for (const auto& h : handles) check(h->closed, "leaked fake handle");
    for (const auto& [ov, op] : ops) { (void) ov; check(op.terminal, "unresolved read"); }
    handles.clear(); ops.clear(); modes.clear(); requests.clear(); dequeue_hook = {};
    reads = nudges = waits = cancels = close_waits = dequeued = 0;
    ex_errors_left = single_errors_left = ex_errors = single_errors = single_waits = single_failed_reads = 0;
    cancellation = Cancel::Request;
}
} // namespace fake

void* VirtualAlloc(void*, std::size_t bytes, DWORD, DWORD) {
    return ::operator new(bytes, std::align_val_t(4096), std::nothrow);
}
BOOL VirtualFree(void* p, std::size_t, DWORD) {
    ::operator delete(p, std::align_val_t(4096)); return TRUE;
}
int MultiByteToWideChar(unsigned, DWORD, const char* s, int, wchar_t* out, int n) {
    const int len = static_cast<int>(std::strlen(s)) + 1;
    if (out) for (int i = 0; i < std::min(n, len); ++i) out[i] = static_cast<unsigned char>(s[i]);
    return len;
}
HANDLE CreateFileW(const wchar_t*, DWORD, DWORD, void*, DWORD, DWORD, HANDLE) {
    std::lock_guard<std::mutex> lk(fake::mu); return fake::make(fake::Kind::File);
}
BOOL GetFileSizeEx(HANDLE file, LARGE_INTEGER* sz) {
    std::lock_guard<std::mutex> lk(fake::mu);
    fake::get(file, fake::Kind::File); sz->QuadPart = 4096 * 1024; return TRUE;
}
HANDLE CreateIoCompletionPort(HANDLE file, HANDLE, ULONG_PTR, DWORD) {
    std::lock_guard<std::mutex> lk(fake::mu);
    auto* f = fake::get(file, fake::Kind::File);
    HANDLE p = fake::make(fake::Kind::Port);
    f->port = static_cast<fake::Handle*>(p); return p;
}
BOOL CloseHandle(HANDLE handle) {
    std::lock_guard<std::mutex> lk(fake::mu);
    auto* h = static_cast<fake::Handle*>(handle);
    check(h && !h->closed, "double close");
    for (const auto& [ov, op] : fake::ops) {
        (void) ov;
        if (h == op.file || h == op.file->port) {
            check(op.terminal && op.dequeued, "handle closed before terminal packet collected");
        }
    }
    h->closed = true; h->packets.clear(); fake::cv.notify_all(); return TRUE;
}
DWORD GetLastError() { return fake::error; }
BOOL ReadFile(HANDLE file, void* buffer, DWORD length, DWORD*, OVERLAPPED* ov) {
    std::lock_guard<std::mutex> lk(fake::mu);
    auto* f = fake::get(file, fake::Kind::File);
    check(ov->hEvent == nullptr, "event suppresses/changes IOCP completion");
    fake::requests.insert(ov);
    auto previous = fake::ops.find(ov);
    if (previous != fake::ops.end()) {
        check(previous->second.terminal, "OVERLAPPED reused before terminal");
        check(previous->second.dequeued || previous->second.file->port->closed, "OVERLAPPED reused with live packet");
        fake::ops.erase(previous);
    }
    const std::uint64_t offset = (std::uint64_t(ov->OffsetHigh) << 32) | ov->Offset;
    const auto mode = fake::modes.count(offset) ? fake::modes.at(offset) : fake::Mode::Pending;
    ++fake::reads;
    fake::cv.notify_all();
    if (mode == fake::Mode::Eof || mode == fake::Mode::Fail) {
        fake::error = mode == fake::Mode::Eof ? ERROR_HANDLE_EOF : ERROR_ACCESS_DENIED; return FALSE;
    }
    ov->Internal = fake::pending_status;
    auto [it, inserted] = fake::ops.emplace(ov, fake::Op{f, buffer});
    check(inserted, "duplicate operation");
    if (mode == fake::Mode::Sync) { fake::terminal(ov, it->second, 0, length); return TRUE; }
    fake::error = ERROR_IO_PENDING; return FALSE;
}
BOOL PostQueuedCompletionStatus(HANDLE port, DWORD bytes, ULONG_PTR key, OVERLAPPED* ov) {
    std::lock_guard<std::mutex> lk(fake::mu);
    fake::get(port, fake::Kind::Port)->packets.push_back({key, ov, 0, bytes});
    if (!ov) ++fake::nudges;
    fake::cv.notify_all(); return TRUE;
}
BOOL GetQueuedCompletionStatusEx(HANDLE port, OVERLAPPED_ENTRY* entries, ULONG want,
                               ULONG* got, DWORD timeout, BOOL alertable) {
    std::function<void()> hook;
    {
        std::lock_guard<std::mutex> lk(fake::mu);
        check(want > 0 && want <= 64 && !alertable, "invalid IOCP batch size");
        ++fake::waits; hook = std::move(fake::dequeue_hook); fake::dequeue_hook = {};
    }
    if (hook) hook(); // Insert new immediate results after production wait's initial drain.
    std::unique_lock<std::mutex> lk(fake::mu);
    auto* p = fake::get(port, fake::Kind::Port);
    if (timeout == INFINITE) { ++fake::close_waits; fake::cv.notify_all(); }
    if (timeout == INFINITE && fake::ex_errors_left) {
        --fake::ex_errors_left; ++fake::ex_errors;
        *got = 0; fake::error = ERROR_GEN_FAILURE; fake::cv.notify_all(); return FALSE;
    }
    if (p->packets.empty() && timeout) {
        const auto duration = std::chrono::milliseconds(timeout == INFINITE ? 5000 : timeout);
        fake::cv.wait_for(lk, duration, [&] { return !p->packets.empty(); });
    }
    *got = 0;
    while (*got < want && !p->packets.empty()) {
        entries[(*got)++] = fake::pop(p);
    }
    fake::error = *got ? 0 : WAIT_TIMEOUT; return *got ? TRUE : FALSE;
}
BOOL GetQueuedCompletionStatus(HANDLE port, DWORD* bytes, ULONG_PTR* key, OVERLAPPED** ov, DWORD timeout) {
    std::unique_lock<std::mutex> lk(fake::mu);
    auto* p = fake::get(port, fake::Kind::Port);
    check(timeout == INFINITE, "fallback must block for terminal completion");
    ++fake::single_waits; *ov = nullptr; fake::cv.notify_all();
    if (fake::single_errors_left) {
        --fake::single_errors_left; ++fake::single_errors;
        fake::error = ERROR_GEN_FAILURE; fake::cv.notify_all(); return FALSE;
    }
    check(fake::cv.wait_for(lk, std::chrono::seconds(5), [&] { return !p->packets.empty(); }), "fallback terminal gate timeout");
    const auto entry = fake::pop(p);
    *ov = entry.lpOverlapped; *bytes = entry.dwNumberOfBytesTransferred; *key = entry.lpCompletionKey;
    if (*ov && entry.Internal != 0) {
        ++fake::single_failed_reads;
        fake::error = entry.Internal == fake::eof_status ? ERROR_HANDLE_EOF : ERROR_OPERATION_ABORTED;
        return FALSE; // Non-null OVERLAPPED proves a dequeued terminal failed read.
    }
    fake::error = 0; return TRUE; // Includes null wake packets, which do not retire requests.
}
BOOL CancelIoEx(HANDLE file, OVERLAPPED* ov) {
    std::lock_guard<std::mutex> lk(fake::mu);
    fake::get(file, fake::Kind::File);
    auto& op = fake::ops.at(ov); ++fake::cancels;
    check(op.file == file, "cancel on wrong file");
    if (op.terminal) { fake::error = ERROR_NOT_FOUND; return FALSE; }
    if (fake::cancellation == fake::Cancel::Race) {
        fake::terminal(ov, op, 0, 4096); fake::error = ERROR_NOT_FOUND; return FALSE;
    }
    if (fake::cancellation == fake::Cancel::Fail) { fake::error = ERROR_ACCESS_DENIED; return FALSE; }
    op.cancel = true; return TRUE; // Intentionally does NOT complete the read.
}

namespace {
struct Buffer {
    void* p = DirectFile::alloc_aligned(4096);
    Buffer() { check(p != nullptr, "allocation failed"); }
    ~Buffer() { DirectFile::free_aligned(p); }
};
void submit(DirectFile& f, void* p, std::uint64_t offset, std::uint64_t tag) {
    std::string err; check(f.submit(offset, p, 4096, tag, err), "submit failed");
}
void open(DirectFile& f) { std::string err; check(f.open("fake-only", err), "open failed"); }
int guarded_wait(DirectFile& f, int max, std::vector<Completion>& result) {
    // Sentinels immediately before/after the exact caller capacity, not just a fixed 65-slot buffer.
    std::vector<Completion> memory(static_cast<std::size_t>(max) + 2, Completion{0x123456789abcdef0ull, 0xabcdef12u, false});
    const Completion sentinel = memory.front();
    int n = f.wait(memory.data() + 1, max, 0);
    check(n >= 0 && n <= max, "wait exceeded capacity");
    for (std::size_t i : {std::size_t(0), memory.size() - 1}) {
        check(memory[i].tag == sentinel.tag && memory[i].bytes == sentinel.bytes && !memory[i].ok, "output canary overwritten");
    }
    result.insert(result.end(), memory.begin() + 1, memory.begin() + 1 + n); return n;
}
void mixed_batch(int max) {
    Buffer b;
    {
        DirectFile f; open(f);
        constexpr unsigned real_count = 90;
        f.wake(); // A null packet BEFORE all real packets in the same batch.
        for (unsigned i = 0; i < real_count; ++i) submit(f, b.p, std::uint64_t(i) * 4096, i);
        fake::await([] { return fake::reads == real_count; });
        fake::finish_all();
        {
            std::lock_guard<std::mutex> lk(fake::mu);
            fake::modes[4096 * 100] = fake::Mode::Eof;
            fake::modes[4096 * 101] = fake::Mode::Fail;
            fake::dequeue_hook = [&] {
                submit(f, b.p, 4096 * 100, real_count);
                submit(f, b.p, 4096 * 101, real_count + 1);
                fake::await([] { return fake::nudges == 3; }); // Both immediate queues updated.
            };
        }
        std::vector<Completion> result;
        check(guarded_wait(f, max, result) > 0, "initial batch missing");
        // Force reuse while copied real completions are still buffered behind the output bound.
        constexpr unsigned reuse_count = 65;
        {
            std::lock_guard<std::mutex> lk(fake::mu);
            for (unsigned i = 0; i < reuse_count; ++i) fake::modes[std::uint64_t(500 + i) * 4096] = fake::Mode::Sync;
        }
        for (unsigned i = 0; i < reuse_count; ++i)
            submit(f, b.p, std::uint64_t(500 + i) * 4096, real_count + 2 + i);
        fake::await([] { return fake::reads == real_count + 2 + reuse_count; });
        for (unsigned i = 0; i < 200 && guarded_wait(f, max, result); ++i) {}
        std::vector<unsigned> seen(real_count + 2 + reuse_count);
        unsigned wakes = 0;
        for (const auto& c : result) {
            if (c.tag == DirectFile::WAKE_TAG) { ++wakes; check(c.ok && c.bytes == 0, "bad wake"); continue; }
            check(c.tag < seen.size() && ++seen[static_cast<std::size_t>(c.tag)] == 1, "lost/repeated tag");
            if (c.tag == real_count) check(c.ok && c.bytes == 0, "bad immediate EOF");
            else if (c.tag == real_count + 1) check(!c.ok && c.bytes == 0, "bad immediate failure");
            else if (c.tag > real_count + 1) check(c.ok && c.bytes == 4096, "recycled buffered result corrupted");
            else check(c.ok == (c.tag % 3 != 1) && c.bytes == (c.tag % 3 == 0 ? 4096u : 0u), "bad port status/bytes");
        }
        check(std::all_of(seen.begin(), seen.end(), [](unsigned n) { return n == 1; }), "completion lost");
        check(wakes == 3, "dequeued null packet lost");
        // Recycling may proceed after copying a packet even if its Completion was buffered.
        std::size_t created;
        {
            std::lock_guard<std::mutex> lk(fake::mu);
            created = fake::requests.size(); fake::modes[0] = fake::Mode::Sync;
        }
        submit(f, b.p, 0, 1000);
        fake::await([] { return fake::reads == real_count + 3 + reuse_count; });
        result.clear();
        check(guarded_wait(f, max, result) == 1 && result[0].tag == 1000 && result[0].ok, "sync/recycled completion failed");
        {
            std::lock_guard<std::mutex> lk(fake::mu);
            check(fake::requests.size() == created, "request not recycled");
        }
        f.close();
        open(f);
        check(guarded_wait(f, max, result) == 0, "stale completion after reopen");
    }
    fake::clean();
}
void invalid_capacity() {
    {
        DirectFile f; open(f); f.wake();
        check(f.wait(nullptr, 0, -1) == 0 && f.wait(nullptr, -1, -1) == 0 &&
              f.wait(nullptr, -2147483647 - 1, -1) == 0, "nonpositive max");
        {
            std::lock_guard<std::mutex> lk(fake::mu);
            check(fake::waits == 0, "nonpositive max touched port");
        }
        std::vector<Completion> result;
        check(guarded_wait(f, 1, result) == 1 && result[0].tag == DirectFile::WAKE_TAG, "nonpositive max consumed packet");
        f.close();
        check(f.wait(nullptr, 0, -1) == 0 && guarded_wait(f, 1, result) == 0, "closed wait");
    }
    fake::clean();
}
void shutdown(fake::Cancel behavior, bool destructor, bool dequeue_errors = false) {
    Buffer b;
    {
        auto f = std::make_unique<DirectFile>(); open(*f);
        {
            std::lock_guard<std::mutex> lk(fake::mu);
            fake::cancellation = behavior; fake::modes[4096 * 100] = fake::Mode::Sync;
            if (dequeue_errors) { fake::ex_errors_left = 32; fake::single_errors_left = 1; }
        }
        // Include a terminal synchronous read whose port packet has not been collected.
        submit(*f, b.p, 4096 * 100, 100);
        for (unsigned i = 0; i < 8; ++i) submit(*f, b.p, std::uint64_t(i) * 4096, i);
        fake::await([] { return fake::reads == 9; });
        f->wake(); // Null packets must not count toward the outstanding-read terminal gate.
        bool returned = false;
        std::thread closer([&] {
            if (destructor) f.reset(); else f->close();
            std::lock_guard<std::mutex> lk(fake::mu); returned = true; fake::cv.notify_all();
        });
        fake::await([&] { return (dequeue_errors ? fake::single_waits >= 4 : fake::close_waits > 0) || returned; });
        if (behavior != fake::Cancel::Race) {
            {
                std::lock_guard<std::mutex> lk(fake::mu);
                check(!returned, "close returned before terminal gate");
                check(fake::cancels == 9, "not all outstanding requests cancelled before waiting");
                for (const auto& h : fake::handles) check(!h->closed, "handle freed during terminal gate");
                check(fake::requests.size() == 9, "OVERLAPPED reused during terminal gate");
                if (dequeue_errors) check(fake::ex_errors >= 4 && fake::single_errors == 1, "IOCP infrastructure failures not exercised");
            }
            fake::finish_all();
        }
        closer.join();
        {
            std::lock_guard<std::mutex> lk(fake::mu);
            check(returned && fake::dequeued == 9, "close failed to collect all terminal packets");
            if (dequeue_errors) check(fake::single_failed_reads > 0, "FALSE/non-null terminal fallback not exercised");
            fake::ex_errors_left = fake::single_errors_left = 0;
        }
        if (!destructor) {
            open(*f);
            std::vector<Completion> result;
            check(guarded_wait(*f, 2, result) == 0, "old IOCP packet survived reopen");
            std::size_t created;
            {
                std::lock_guard<std::mutex> lk(fake::mu);
                created = fake::requests.size(); fake::modes[0] = fake::Mode::Sync;
            }
            // Repeat close/reopen cycles: all requests, not only collected ones, must be reusable.
            for (unsigned i = 0; i < 12; ++i) {
                submit(*f, b.p, 0, 1000 + i);
                fake::await([i] { return fake::reads == 10 + i; });
                f->close(); open(*f);
            }
            std::lock_guard<std::mutex> lk(fake::mu);
            check(fake::requests.size() == created, "close/reopen leaked requests");
        }
    }
    fake::clean();
}
} // namespace

int main(int argc, char** argv) {
    try {
        const std::string only = argc == 2 ? argv[1] : "";
        check(argc <= 2 && (only.empty() || only == "--batch2" || only == "--capacity" || only == "--shutdown" || only == "--iocp-errors"), "invalid test selector");
        if (only.empty()) for (int max : {1, 2, 64, 65}) mixed_batch(max);
        if (only == "--batch2") mixed_batch(2);
        if (only.empty() || only == "--capacity") invalid_capacity();
        if (only.empty() || only == "--shutdown")
            for (auto behavior : {fake::Cancel::Request, fake::Cancel::Fail, fake::Cancel::Race}) {
                shutdown(behavior, false); shutdown(behavior, true);
            }
        if (only.empty() || only == "--iocp-errors")
            for (auto behavior : {fake::Cancel::Request, fake::Cancel::Fail}) {
                shutdown(behavior, false, true); shutdown(behavior, true, true);
            }
        if (!only.empty()) {
            std::cout << "PASS: production Win32 branch " << only << '\n';
            return 0;
        }
        std::cout << "PASS: production Win32 branch; canaries/max=1,2,64,65; mixed wake/real/immediate arrival; "
                     "exactly-once/recycle/sync/EOF/failure; max<=0; delayed IOCP status; "
                     "close/destructor terminal gate, cancel success/failure/race, reopen/no leaks; "
                     "IOCP Ex/single dequeue errors, blocking fallback, FALSE/non-null terminal reads\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n'; return 1;
    }
}
