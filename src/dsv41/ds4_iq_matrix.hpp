// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include "ds4_iq.cuh"

namespace strata::ds4::gguf::iq::matrix_detail {
// Enqueue only the busy experts. False leaves all work to the caller's FMA path.
// The matrix kernels live in a separate device image from the portable kernels.
bool gate_up(const SetArgs& a, sycl::queue& s);
bool down(const SetArgs& a, sycl::queue& s);
}  // namespace strata::ds4::gguf::iq::matrix_detail

namespace strata::ds4::gguf::iq::native_f16_matrix_detail {
bool gate_up(const SetArgs& a, sycl::queue& s);
}
