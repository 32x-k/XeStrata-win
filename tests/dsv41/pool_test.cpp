// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "ds4_pool.hpp"
#include <array>
#include <cstdio>
#include <future>
#include <stdexcept>

namespace {
int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "%s\n", what); ++failures; }
}

void caller_exception() {
    strata::ds4::Pool pool(0);
    std::array<int, 7> calls{};
    const std::function<void(int)> job = [&](int i) {
        ++calls[(size_t) i];
        if (i == 0) throw std::runtime_error("caller job");
        if (i == 3) throw 3;
    };
    pool.start((int) calls.size(), job);
    try { pool.help(); }
    catch (...) { check(false, "help must record exceptions without throwing"); }
    for (int count : calls) check(count == 1, "caller must complete every job exactly once");
    bool caught = false;
    try { pool.wait(); }
    catch (const std::runtime_error& e) { caught = std::string(e.what()) == "caller job"; }
    catch (...) { caught = false; }
    check(caught, "wait must preserve the first caller exception");
    pool.wait();   // a drained batch does not throw twice
    pool.run((int) calls.size(), [&](int i) { ++calls[(size_t) i]; });
    for (int count : calls) check(count == 2, "pool must be reusable after an exception");
    pool.run(0, [&](int) { check(false, "an empty batch must not run"); });
}

void worker_exception() {
    constexpr int jobs = 9;
    strata::ds4::Pool pool(2);
    struct State {
        std::mutex mutex;
        std::condition_variable cv;
        std::array<int, jobs> calls{};
        bool release = false;
        int finished = 0;
    } state;
    struct Finish {
        State& state;
        ~Finish() {
            std::lock_guard<std::mutex> lock(state.mutex);
            ++state.finished;
            state.cv.notify_all();
        }
    };
    const std::function<void(int)> job = [&](int i) {
        Finish finished{state};
        {
            std::unique_lock<std::mutex> lock(state.mutex);
            ++state.calls[(size_t) i];
            if (i == 0) state.cv.wait(lock, [&] { return state.release; });
        }
        if (i == 1) throw std::runtime_error("worker job");
        if (i == 4) throw 4;
    };
    pool.start(jobs, job);   // no caller help: also covers a worker starting after the first batch was published
    std::promise<bool> result;
    auto done = result.get_future();
    std::thread waiter([&] {
        bool caught = false;
        try { pool.wait(); }
        catch (const std::runtime_error& e) { caught = std::string(e.what()) == "worker job"; }
        catch (...) { caught = false; }
        std::lock_guard<std::mutex> lock(state.mutex);
        result.set_value(caught && state.finished == jobs);
    });
    {
        std::unique_lock<std::mutex> lock(state.mutex);
        state.cv.wait(lock, [&] { return state.finished == jobs - 1; });
    }
    check(done.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout,
          "wait must not rethrow while another callback is blocked");
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.release = true;
        state.cv.notify_all();
    }
    waiter.join();
    check(done.get(), "worker exception must reach wait only after all callbacks have returned");
    for (int count : state.calls) check(count == 1, "workers must complete every job exactly once");
    pool.run(jobs, [&](int i) { ++state.calls[(size_t) i]; });
    for (int count : state.calls) check(count == 2, "workers must accept a batch after a failed batch");
}

void nonstandard_exception() {
    strata::ds4::Pool pool(1);
    bool caught = false;
    try { pool.run(1, [](int) { throw 42; }); }
    catch (int value) { caught = value == 42; }
    catch (...) { caught = false; }
    check(caught, "non-standard job exceptions must also propagate after draining");
}
}  // namespace

int main() try {
    caller_exception();
    worker_exception();
    nonstandard_exception();
    std::printf("dsv41 pool exceptions: %s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "dsv41 pool exceptions: %s\n", e.what());
    return 1;
} catch (...) {
    std::fputs("dsv41 pool exceptions: unexpected exception\n", stderr);
    return 1;
}
