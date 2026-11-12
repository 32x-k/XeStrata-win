// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <sycl/sycl.hpp>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace strata::ds4 {

// Cancellation releases a queued wait only after the CPU response writer has
// finished. Subsequent device reads must never overlap a partially written reply.
struct MailboxControl {
    std::atomic<bool> stopped{false};
    std::atomic<bool> producer_done{true};
    mutable std::mutex mutex;
    std::string reason;
    // One word of host USM for each bound queue whose device waits for a reply itself (wait_seq): a wait
    // queued on the device cannot read this object, so cancellation is published to these words as well.
    std::vector<unsigned*> device_release;

    void stop(const char* message = nullptr) noexcept {
        if (message) {
            try {
                std::lock_guard<std::mutex> lock(mutex);
                if (reason.empty()) reason = message;
            } catch (...) {
                // Reporting may allocate; cancellation must still make progress.
                stopped.store(true);
                release_device_waits();
                return;
            }
        }
        stopped.store(true);
        release_device_waits();
    }
    // The CPU response writer has finished its last write (the end of the service thread, or none started).
    void producer_finished() noexcept {
        producer_done.store(true);
        release_device_waits();
    }
    // Called after either store above: whichever of the two comes last sees both and releases the device's
    // waits (the stores and these loads are sequentially consistent, so neither caller can miss the other).
    void release_device_waits() noexcept {
        if (!stopped.load() || !producer_done.load()) return;
        try {
            std::lock_guard<std::mutex> lock(mutex);
            for (unsigned* word : device_release) std::atomic_ref<unsigned>(*word).store(1u, std::memory_order_release);
        } catch (...) {
            // The device keeps waiting; the runtime's hang limit then ends the process.
            std::fputs("ds4: a mailbox lock failed; waits queued on the device were not released\n", stderr);
        }
    }
    bool may_release() const {
        return stopped.load(std::memory_order_acquire) && producer_done.load(std::memory_order_acquire);
    }
    bool ready(std::string& error) const {
        if (!stopped.load(std::memory_order_acquire)) return true;
        std::lock_guard<std::mutex> lock(mutex);
        error = reason.empty() ? "ds4 execution stopped after an earlier failure" : reason;
        return false;
    }
};

inline void wait_mailbox(const unsigned* pointer, unsigned sequence, bool at_least, const MailboxControl* control) {
    std::atomic_ref<unsigned> flag(*const_cast<unsigned*>(pointer));
    for (;;) {
        const unsigned value = flag.load(std::memory_order_acquire);
        if (at_least ? int32_t(value - sequence) >= 0 : value == sequence) return;
        if (control && control->may_release()) return;
        std::this_thread::yield();
    }
}

// The kernel API stays queue-based. Only queues owned by a model are registered;
// standalone callers retain the ordinary sequence-wait contract.
void bind_mailbox(sycl::queue& queue, const std::shared_ptr<MailboxControl>& control);
void unbind_mailbox(const sycl::queue* queue) noexcept;

}  // namespace strata::ds4
