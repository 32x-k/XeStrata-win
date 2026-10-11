// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Model-free checks of production scalar decoding; no ggml or device runtime is linked.
#include "strata/artifact/dequant.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>

namespace {
float half_reference(uint16_t h) {
    const int sign = (h & 0x8000) ? -1 : 1;
    const int exponent = (h >> 10) & 31;
    const int fraction = h & 1023;
    if (exponent == 31)
        return fraction ? std::numeric_limits<float>::quiet_NaN() :
                          sign * std::numeric_limits<float>::infinity();
    if (exponent == 0) return sign * std::ldexp((float) fraction, -24);
    return sign * std::ldexp(1.0f + (float) fraction / 1024.0f, exponent - 15);
}

bool half_check() {
    for (uint32_t bits = 0; bits < 65536; ++bits) {
        const float want = half_reference((uint16_t) bits);
        const float got = strata::fp16_to_fp32((uint16_t) bits);
        if (std::isnan(want)) {
            if (!std::isnan(got)) return false;
        } else if (got != want || std::signbit(got) != std::signbit(want)) {
            std::fprintf(stderr, "half mismatch: 0x%04x\n", (unsigned) bits);
            return false;
        }
    }
    return true;
}

bool q2_check() {
    const uint16_t scales[] = {0, 1, 0x0400, 0x3800, 0x3c00, 0xc000};
    for (uint16_t scale : scales) {
        const float d = half_reference(scale);
        for (int seed = 0; seed < 64; ++seed) {
            std::array<uint8_t, 18> block{};
            block[0] = (uint8_t) scale;
            block[1] = (uint8_t) (scale >> 8);
            for (int j = 0; j < 64; ++j) {
                const int code = ((j * 7 + seed) ^ (j / 4)) & 3;
                block[2 + j / 4] |= (uint8_t) (code << (2 * (j % 4)));
            }
            std::array<float, 66> out;
            out.fill(12345.0f);
            strata::dequantize_q2_0(block.data(), out.data() + 1);
            if (out.front() != 12345.0f || out.back() != 12345.0f) return false;
            for (int j = 0; j < 64; ++j) {
                const int code = ((j * 7 + seed) ^ (j / 4)) & 3;
                if (out[1 + j] != (code - 1) * d) return false;
            }
        }
    }
    return true;
}

bool iq4_check() {
    const int table[] = {-127, -104, -83, -65, -49, -35, -22, -10,
                        1, 13, 25, 38, 53, 69, 89, 113};
    for (uint16_t scale : {uint16_t(0), uint16_t(1), uint16_t(0x3800), uint16_t(0xc000)}) {
        std::array<uint8_t, 18> block{};
        block[0] = (uint8_t) scale;
        block[1] = (uint8_t) (scale >> 8);
        for (int j = 0; j < 16; ++j) block[2 + j] = (uint8_t) (j | ((15 - j) << 4));
        std::array<float, 34> out;
        out.fill(12345.0f);
        strata::dequantize_iq4_nl(block.data(), out.data() + 1);
        if (out.front() != 12345.0f || out.back() != 12345.0f) return false;
        const float d = half_reference(scale);
        for (int j = 0; j < 16; ++j)
            if (out[1 + j] != d * table[j] || out[17 + j] != d * table[15 - j]) return false;
    }
    return true;
}
}  // namespace

int main() {
    if (!half_check() || !q2_check() || !iq4_check()) {
        std::fprintf(stderr, "scalar decode contract FAIL\n");
        return 1;
    }
    std::puts("PASS: 65536 half patterns, 384 Q2_0 blocks, IQ4_NL codebook/layout, output canaries");
    return 0;
}
