// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/prefill/gdn_tc.hpp - the prompt path's DeltaNet recurrence as a chunked scan on the matrix engines
// (STRATA_GDN_TC=1, gdn_tc.cpp), for kernels.cpp's Gdn::recurrence.
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>

namespace strata::prefill {

/// The recurrence of a prompt chunk of T tokens for hk key and hv value heads of 128 (Gdn<hk, hv> in kernels.cpp):
/// the state updated in place and each token's output before the norm written to y [T][hv 128], as the kernels there
/// do, in chunks of 32 tokens whose products run on the matrix engines.  Not those kernels' bits.  False when nothing
/// was launched: the GPU has no matrix engines this scan is written for (or STRATA_NO_XMX=1), too little local memory,
/// no room for its scratch, or the shape is not one it is built for; the caller then runs its own kernels.
bool gdn_rec_tc(sycl::queue& q, int64_t h_k, int64_t h_v, float* state, const float* h, const float* gate,
                const float* beta, float* y, int64_t T);

}  // namespace strata::prefill
