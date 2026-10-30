// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "ds4_mailbox.hpp"
#include <chrono>
#include <cstdio>
#include <future>

// Exercises the exact host wait used by queue host_tasks, without constructing
// a queue or asking the SYCL runtime to enumerate devices.
int main() try {
    using namespace strata::ds4;
    using namespace std::chrono_literals;
    int failures = 0;
    auto check = [&](bool ok, const char* text) {
        if (!ok) { std::fprintf(stderr, "%s\n", text); ++failures; }
    };
    for (bool at_least : {false, true}) {
        unsigned sequence = 0;
        int payload = 0;
        auto reader = std::async(std::launch::async, [&] {
            wait_mailbox(&sequence, 7, at_least, nullptr);
            return payload;
        });
        payload = 93;
        std::atomic_ref<unsigned>(sequence).store(at_least ? 8u : 7u, std::memory_order_release);
        check(reader.wait_for(2s) == std::future_status::ready, "published sequence did not release the wait");
        // Also releases an incorrectly implemented at-least wait, so the test can fail normally.
        std::atomic_ref<unsigned>(sequence).store(7, std::memory_order_release);
        check(reader.get() == 93, "sequence acquire did not expose the response payload");
    }
    for (bool at_least : {false, true}) {
        MailboxControl control;
        control.producer_done.store(false, std::memory_order_release);
        unsigned sequence = 0;
        std::atomic<int> payload{0};
        control.stop("CPU response failed");
        auto reader = std::async(std::launch::async, [&] {
            wait_mailbox(&sequence, 7, at_least, &control);
            return payload.load(std::memory_order_relaxed);
        });
        check(reader.wait_for(20ms) == std::future_status::timeout,
              "cancellation released a wait while the CPU producer could still write");
        payload.store(51, std::memory_order_relaxed);
        control.producer_done.store(true, std::memory_order_release);
        check(reader.wait_for(2s) == std::future_status::ready, "failed producer left a sequence wait blocked");
        // A regression must report failure rather than leave the test process hanging.
        std::atomic_ref<unsigned>(sequence).store(7, std::memory_order_release);
        check(reader.get() == 51, "producer completion did not publish its final write");
        std::string reason;
        check(!control.ready(reason) && reason == "CPU response failed", "the CPU failure reason was lost");
    }
    if (!failures) std::puts("dsv41 mailbox tests passed");
    return failures ? 1 : 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "mailbox test: %s\n", e.what());
    return 1;
} catch (...) {
    std::fprintf(stderr, "mailbox test: unexpected exception\n");
    return 1;
}
