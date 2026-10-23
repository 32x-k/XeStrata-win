// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// src/kernels/cpu/iq256_portable.cpp - the AVX2 multi-token i-quant rows, on a host that is not x86.
//
// strata::ds4::gguf::dots() asks iq256_supported() before it uses iq256_rows(); on arm64 (an Apple Silicon Mac) the answer
// is simply no, and the CPU lane keeps ggml-cpu's rows, which have their own NEON kernels.  Without these two symbols the
// CPU-only build links nothing on the Mac (found on the M5 Max, 2026-10-23).
#include "strata/kernels/cpu/iq_avx2.hpp"

#include <cstdlib>

namespace strata::kernels::cpu {

bool iq256_supported(int) noexcept { return false; }

void iq256_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    std::abort();   // unreachable: iq256_supported() is false here, and callers check it
}

}   // namespace strata::kernels::cpu
