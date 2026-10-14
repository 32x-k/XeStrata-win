// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "strata/platform/direct_file.hpp"
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
using strata::platform::DirectFile;
using strata::platform::Completion;
static void check(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
int main() {
    namespace fs = std::filesystem;
    const fs::path path = fs::temp_directory_path() /
        ("strata-direct-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    const std::string name_s = path.string();
    const char* name = name_s.c_str();
    std::error_code ec;
    void* memory = DirectFile::alloc_aligned(size_t(4096) * 18);
    try {
        check(memory != nullptr, "buffer allocation");
        { std::ofstream f(name, std::ios::binary); std::string bytes(4096 * 16 + 17, 'x'); f.write(bytes.data(), (std::streamsize) bytes.size()); }
        DirectFile f;
        std::string error;
#if defined(_WIN32)
        _putenv_s("STRATA_IO_THREADS", "4");   // the pool's size (out-of-range values are clamped to 1..64)
#else
        setenv("STRATA_IO_THREADS", "4", 1);   // the pool's size (out-of-range values are clamped to 1..64)
#endif
        check(f.open(name, error), error.c_str());
        check(!f.submit(1, memory, 4096, 0, error), "unaligned request accepted");
        for (unsigned i = 0; i < 17; ++i)
            check(f.submit(uint64_t(i) * 4096, static_cast<char*>(memory) + uint64_t(i) * 4096, 4096, i, error), error.c_str());
        std::vector<bool> seen(17);
        unsigned count = 0;
        while (count < 17) {
            Completion c[17]; int n = f.wait(c, 17, 5000);
            check(n > 0, "completion timeout");
            for (int i = 0; i < n; ++i) {
                check(c[i].tag < 17 && !seen[c[i].tag], "wrong or repeated completion");
                seen[c[i].tag] = true; ++count;
                check(c[i].ok && c[i].bytes == (c[i].tag == 16 ? 17u : 4096u), "incorrect full/EOF completion");
                check(static_cast<char*>(memory)[c[i].tag * 4096] == 'x', "read data mismatch");
            }
        }
        std::jthread waker([&] { f.wake(); });
        Completion c; check(f.wait(&c, 1, 5000) == 1 && c.tag == DirectFile::WAKE_TAG, "wake lost");
        waker.join();
        check(f.submit(0, memory, 4096, 99, error), error.c_str());
        f.close(); // Must drain outstanding buffer writes before returning.
        std::memset(memory, 0, 4096);
        check(f.open(name, error), error.c_str());
        check(f.wait(&c, 1, 0) == 0, "stale completion on reopen");
        check(f.submit(0, memory, 4096, 100, error), error.c_str());
        check(f.wait(&c, 1, 5000) == 1 && c.ok && c.tag == 100, "reopened read failed");
        f.close();
        DirectFile::free_aligned(memory); fs::remove(path, ec);
        std::cout << "direct_file_async: passed queued reads, short EOF, wake, close drain, reopen\n";
        return 0;
    } catch (const std::exception& e) {
        DirectFile::free_aligned(memory); fs::remove(path, ec);
        std::cerr << e.what() << '\n'; return 1;
    }
}
