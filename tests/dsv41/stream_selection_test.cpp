// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "ds4_stream_selection.hpp"
#include <array>
#include <cstdio>
#include <exception>
#include <random>

#define CHECK(condition) do { if (!(condition)) { std::fprintf(stderr, "selection check failed at %d\n", __LINE__); return 1; } } while (0)

int main() try {
    using strata::ds4::stream_selection_limit;
    CHECK(stream_selection_limit(nullptr, nullptr, true) == INT_MAX);
    CHECK(stream_selection_limit(nullptr, nullptr, false) == 0);
    CHECK(stream_selection_limit("0", "128", true) == 0);
    CHECK(stream_selection_limit("1", "0", false) == INT_MAX);
    CHECK(stream_selection_limit(nullptr, "128", false) == 128);
    CHECK(stream_selection_limit(nullptr, "-1", true) == 0);
    CHECK(stream_selection_limit(nullptr, "999999999999999999999", true) == INT_MAX);
    std::minstd_rand random(1729);  // NOLINT(bugprone-random-generator-seed): reproducible regression inputs.
    for (int trial = 0; trial < 100; ++trial) {
        const size_t resident = 1 + random() % 13, host = 1 + random() % 127;
        const size_t batch = 1 + random() % 17;
        std::vector<int> bounds(resident + host + 1, 0);
        for (size_t i = 0; i < resident + host; ++i)
            bounds[i + 1] = bounds[i] + (trial == 0 ? 0 : int(random() % 4));
        strata::ds4::StreamSelection select;
        select.chunks = {1, 0, 1};
        CHECK(!select.ready(3, 0));
        CHECK(select.ready(3, 1));
        select.routed(3, 0, bounds);
        CHECK(select.ready(3, 0));
        CHECK(!select.ready(4, 0) && !select.ready(3, 2));
        std::vector<std::array<uint8_t, 31>> payload(host);
        for (auto& bytes : payload) for (auto& value : bytes) value = uint8_t(random());
        for (size_t start = 0; start < host; start += batch) {
            const size_t count = std::min(batch, host - start);
            std::vector<uint8_t> ring(2 + count * 31, 0xa5);
            // Replaying the original batch offsets must neither compact selected
            // experts nor read/write the holes an unpicked expert leaves behind.
            for (size_t i = 0; i < count; ++i)
                if (select.selected(resident + start + i))
                    std::copy(payload[start + i].begin(), payload[start + i].end(), ring.data() + 1 + i * 31);
            CHECK(ring.front() == 0xa5 && ring.back() == 0xa5);
            for (size_t i = 0; i < count; ++i) {
                const bool expected = bounds[resident + start + i + 1] != bounds[resident + start + i];
                for (size_t j = 0; j < 31; ++j)
                    CHECK(ring[1 + i * 31 + j] == (expected ? payload[start + i][j] : 0xa5));
            }
        }
        select.routed(4, 0, std::vector<int>(resident + host + 1, 0));
        CHECK(select.ready(4, 0) && !select.ready(3, 0));
        for (size_t i = 0; i < resident + host; ++i) CHECK(!select.selected(i));
        CHECK(!select.selected(resident + host));
    }
    std::puts("stream selection: policy, routing fences, batch offsets and guards PASS");
} catch (const std::exception& error) {
    std::fprintf(stderr, "stream selection: %s\n", error.what());
    return 1;
}
