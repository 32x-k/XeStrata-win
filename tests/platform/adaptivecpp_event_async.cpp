// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Manual real-header/runtime regression; not a mock and not a GPU fault test.
// Parent must supply matching headers/runtime from the separate CPU-only prefix
// .tools/adaptivecpp-eventfix, with NO HIP backend/plugin installed or discoverable.
// Null DAG nodes: runtime initialization only; no device/queue, device allocation,
// kernel, DMA, direct HIP calls, invalid memory, or deliberately hanging work.
// event.hpp needs the complete runtime type; no full sycl.hpp/GPU half machinery.
#include <hipSYCL/runtime/runtime.hpp>
#include <hipSYCL/sycl/event.hpp>
#include <hipSYCL/runtime/error.hpp>
#include <exception>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <filesystem>
#endif

namespace sycl = hipsycl::sycl;
namespace rt = hipsycl::rt;
namespace {
void check(bool ok, const char* why) {
    if (!ok) throw std::runtime_error(why);
}
void empty_errors() {
    // Never silently clear unknown runtime errors, including initialization errors.
    check(rt::application::errors().num_errors() == 0, "global async error list is not empty");
}
std::string inject(const char* sentinel) {
    // Synthetic, test-only message/type; NO actual allocation failure or GPU fault.
    rt::result error{__acpp_here(), rt::error_info{sentinel, rt::error_type::memory_allocation_error}};
    rt::register_error(error);
    return error.what();
}
void verify(const sycl::exception& error, const std::string& original, const char* sentinel) {
    check(std::string{error.what()} == original, "original runtime diagnostic was replaced");
    check(original.find(sentinel) != std::string::npos, "synthetic sentinel missing");
    check(error.code() == sycl::make_error_code(sycl::errc::memory_allocation), "original errc lost");
}
void verify(std::exception_ptr error, const std::string& original, const char* sentinel) {
    check(bool(error), "null exception_ptr");
    try { std::rethrow_exception(error); }
    catch (const sycl::exception& e) { verify(e, original, sentinel); return; }
    throw std::runtime_error("not a sycl::exception");
}
struct Capture {
    unsigned calls = 0;
    sycl::exception_list errors;
    void operator()(sycl::exception_list list) { ++calls; errors = list; }
};
void single(bool expect_broken) {
    empty_errors();
    Capture capture;
    sycl::event event{rt::dag_node_ptr{}, [&](sycl::exception_list list) { capture(list); }};
    empty_errors();
    const auto original = inject("TEST-ONLY-single-ACPP-71");
    bool bad_function = false;
    try { event.wait_and_throw(); }
    catch (const std::bad_function_call&) {
        if (!expect_broken) throw;
        bad_function = true;
    }
    if (expect_broken) {
        check(bad_function, "negative control did not catch bad_function_call");
        check(capture.calls == 0 && capture.errors.empty(), "broken control called handler");
    } else {
        check(capture.calls == 1 && capture.errors.size() == 1, "single error callback/count");
        verify(capture.errors.front(), original, "TEST-ONLY-single-ACPP-71");
    }
    empty_errors();
    event.wait_and_throw();
    check(capture.calls == (expect_broken ? 0u : 1u), "single error redelivered");
    empty_errors();
}
void batch() {
    empty_errors();
    Capture capture;
    sycl::event event{rt::dag_node_ptr{}, [&](sycl::exception_list list) { capture(list); }};
    empty_errors();
    const auto first = inject("TEST-ONLY-batch-first-ACPP-82");
    const auto second = inject("TEST-ONLY-batch-second-ACPP-93");
    check(rt::application::errors().num_errors() == 2, "two injections not registered");
    event.wait_and_throw();
    check(capture.calls == 1 && capture.errors.size() == 2, "batch must use one callback with both errors");
    auto it = capture.errors.begin();
    verify(*it++, first, "TEST-ONLY-batch-first-ACPP-82");
    verify(*it, second, "TEST-ONLY-batch-second-ACPP-93");
    empty_errors();
    event.wait_and_throw();
    check(capture.calls == 1, "batch redelivered");
    empty_errors();
}
void copied() {
    empty_errors();
    Capture capture;
    sycl::event copy = [&] {
        sycl::event original{rt::dag_node_ptr{}, [&](sycl::exception_list list) { capture(list); }};
        sycl::event copied_event{original}; // Explicit copy; original dies before delivery.
        return copied_event;
    }();
    empty_errors();
    const auto original = inject("TEST-ONLY-copy-ACPP-104");
    copy.wait_and_throw();
    check(capture.calls == 1 && capture.errors.size() == 1, "copy lost handler");
    verify(capture.errors.front(), original, "TEST-ONLY-copy-ACPP-104");
    empty_errors();
    copy.wait_and_throw();
    check(capture.calls == 1, "copy redelivered");
    empty_errors();
}
void throwing() {
    empty_errors();
    unsigned calls = 0;
    std::exception_ptr delivered;
    sycl::event event{rt::dag_node_ptr{}, [&](sycl::exception_list list) {
        ++calls;
        check(list.size() == 1, "throwing handler error count");
        delivered = list.front();
        std::rethrow_exception(delivered); // Standard direct propagation, no DMA.
    }};
    empty_errors();
    const auto original = inject("TEST-ONLY-throw-ACPP-115");
    bool caught = false;
    try { event.wait_and_throw(); }
    catch (const sycl::exception& e) {
        caught = true;
        verify(e, original, "TEST-ONLY-throw-ACPP-115");
        // current_exception may copy the exception on Windows; pointer identity is not portable.
        verify(delivered, original, "TEST-ONLY-throw-ACPP-115");
    }
    check(caught && calls == 1, "original handler exception not caught exactly once");
    empty_errors();
    event.wait_and_throw();
    check(calls == 1, "throwing handler error redelivered");
    empty_errors();
}
} // namespace
int main(int argc, char** argv) {
    try {
        check(argc == 1 || (argc == 2 && std::string{argv[1]} == "--expect-broken"),
              "usage: adaptivecpp_event_async [--expect-broken]");
        std::cout << "adaptivecpp_event_async: real event.hpp + runtime/error.hpp; C++=" << __cplusplus << '\n';
#ifdef _WIN32
        const auto module = GetModuleHandleW(L"acpp-rt.dll");
        wchar_t module_path[32768]{};
        const auto path_size = GetModuleFileNameW(module, module_path, 32768);
        check(module != nullptr && path_size > 0 && path_size < 32768, "runtime DLL path unavailable");
        std::cout << "runtime DLL: " << std::filesystem::path{module_path}.u8string() << '\n';
#endif
        empty_errors();
        rt::runtime_keep_alive_token cpu_fixture; // CPU-only runtime kept alive across all cases.
        const bool broken = argc == 2;
        single(broken);
        if (!broken) { batch(); copied(); throwing(); }
        empty_errors();
        std::cout << (broken ? "PASS: expected-broken negative control (not fixed behavior)\n"
                             : "PASS: adaptivecpp_event_async (synthetic errors; no GPU work)\n");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: adaptivecpp_event_async: " << e.what() << '\n';
        return 1;
    } catch (...) {
        std::cerr << "FAIL: adaptivecpp_event_async: non-standard exception\n";
        return 1;
    }
}
