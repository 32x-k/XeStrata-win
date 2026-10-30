// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "ds4_sycl.hpp"
#include "ds4_numerics.cuh"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>

int main() {
    using namespace strata::ds4;
    using namespace strata::ds4::sycl_detail;
    int failed = 0;
    auto check = [&](bool ok, const char* what) {
        if (!ok) { std::fprintf(stderr, "%s\n", what); ++failed; }
    };
    for (uint32_t bits = 0; bits < 65536; ++bits) {
        const auto h = uint16_t(bits);
        const bool nan = (h & 0x7c00u) == 0x7c00u && (h & 0x03ffu);
        const float f = f16_value(h);
        const uint16_t back = f16_bits(f);
        if (nan) check(std::isnan(f) && (back & 0x7fffu) == 0x7e00u, "half NaN must remain a quiet NaN");
        else if (back != h) { check(false, "half round trip changed a value"); break; }
    }
    check(f16_value(0x0001) == std::ldexp(1.f, -24), "smallest half subnormal");
    check(f16_value(0x0400) == std::ldexp(1.f, -14), "smallest half normal");
    check(f16_value(0x7bff) == 65504.f, "largest finite half");
    check(f16_bits(std::ldexp(1.f, -25)) == 0, "half subnormal tie rounds to even zero");
    check(f16_bits(std::nextafter(std::ldexp(1.f, -25), 1.f)) == 1, "value above half subnormal tie");
    check(f16_bits(1.f + std::ldexp(1.f, -11)) == 0x3c00, "half even tie");
    check(f16_bits(1.f + 3.f * std::ldexp(1.f, -11)) == 0x3c02, "half odd tie");
    check(f16_bits(65520.f) == 0x7c00, "half overflow tie");
    check(f16_bits(-0.f) == 0x8000, "half negative zero");
    check(bf16_rn(1.f + std::ldexp(1.f, -8)) == 0x3f80, "BF16 even tie");
    check(bf16_rn(1.f + 3.f * std::ldexp(1.f, -8)) == 0x3f82, "BF16 odd tie");
    check(bf16_rn(-0.f) == 0x8000, "BF16 negative zero");
    check((bf16_rn(std::numeric_limits<float>::quiet_NaN()) & 0x7fc0) == 0x7fc0, "BF16 quiet NaN");
    if (!failed) std::puts("dsv41 numeric conversion tests passed");
    return failed ? 1 : 0;
}
