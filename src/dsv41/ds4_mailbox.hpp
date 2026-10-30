// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <sycl/sycl.hpp>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace strata::ds4 {

// Cancellation releases a queued wait only after the CPU response writer has
// finished. Subsequent device reads must never overlap a partially written reply.
struct MailboxControl {
    std::atomic<bool> stopped{false};
    std::atomic<bool> producer_done{true};
    mutable std::mutex mutex;
    std::string reason;

    void stop(const char* message = nullptr) noexcept {
        if (message) {
            try {
                std::lock_guard<std::mutex> lock(mutex);
                if (reason.empty()) reason = message;
            } catch (...) {
                // Reporting may allocate; cancellation must still make progress.
                stopped.store(true, std::memory_order_release);
                return;
            }
        }
        stopped.store(true, std::memory_order_release);
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
