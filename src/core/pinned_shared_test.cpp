// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// Selftest for the optional shared file backing of PinnedArena (MAP_SHARED; a file view on Windows).
#include "strata/core/pinned.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <unistd.h>
#endif

int main() {
#if defined(_WIN32)
    char dir[MAX_PATH], path[MAX_PATH];
    if (GetTempPathA(MAX_PATH, dir) == 0 || GetTempFileNameA(dir, "sa", 0, path) == 0) {
        std::fprintf(stderr, "GetTempFileName failed: %lu\n", (unsigned long) GetLastError());
        return 2;
    }
#else
    char path[] = "/tmp/strata-shared-arena-XXXXXX";
    const int fd = mkstemp(path);
    if (fd < 0) {
        std::perror("mkstemp");
        return 2;
    }
    close(fd);
#endif

    int fail = 0;
    const uint64_t bytes = 64u << 10;
    const uint64_t pack_hash = 0x0123456789abcdefull;
    const std::vector<uint64_t> no_bounds;
    {
        strata::core::PinnedArena first(bytes, no_bounds, 0, path, pack_hash);
        if (!first.valid()) {
            std::fprintf(stderr, "first shared arena failed: %s\n", first.note.c_str());
            fail = 1;
        } else {
            first.data()[123] = 0x5a;

            strata::core::PinnedArena second(bytes, no_bounds, 0, path, pack_hash);
            if (!second.valid()) {
                std::fprintf(stderr, "second shared arena failed: %s\n", second.note.c_str());
                fail = 1;
            } else if (second.data()[123] != 0x5a) {
                std::fprintf(stderr, "shared mapping did not expose the same bytes\n");
                fail = 1;
            }

            strata::core::PinnedArena wrong_pack(bytes, no_bounds, 0, path, pack_hash + 1);
            if (wrong_pack.valid() || wrong_pack.note.find("pack hash") == std::string::npos) {
                std::fprintf(stderr, "pack mismatch was not refused: %s\n", wrong_pack.note.c_str());
                fail = 1;
            }

            strata::core::PinnedArena wrong_size(bytes * 2, no_bounds, 0, path, pack_hash);
            if (wrong_size.valid() || wrong_size.note.find("expected") == std::string::npos) {
                std::fprintf(stderr, "size mismatch was not refused: %s\n", wrong_size.note.c_str());
                fail = 1;
            }
        }
    }

    if (std::remove(path) != 0) {
        std::perror("remove");
        fail = 1;
    }
    std::printf("pinned_shared_test: %s\n", fail ? "FAILED" : "OK");
    return fail;
}
