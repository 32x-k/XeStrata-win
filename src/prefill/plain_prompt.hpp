// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/prefill/plain_prompt.hpp - PlainModel::PromptGemm, the state of the prompt's products, shared by plain_prompt.cpp
// and plain_prompt_glm5.cpp.
#pragma once

#include "strata/core/gpu.hpp"
#include "strata/core/plain_model.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/moe_mmq.hpp"

namespace strata::core {

/// The products' state: Gemm over the dequantization scratch carved with the other buffers.
struct PlainModel::PromptGemm {
    prefill::Gemm gemm;
    prefill::mmq::Context mmq;
    bool identity = false;   // pb_.identity written
    // the int8 products' copies from RAM on a stream of their own, into the staging's two halves in turn: ready[h]
    // when half h holds its group, used[h] when that group's products are done with it
    strata::gpu::Stream xs = nullptr;
    strata::gpu::Event *ready[2] = {}, *used[2] = {};
    bool used_live[2] = {};
    bool ready_live[2] = {};   // ready[h] recorded: a pinned bounce half (h_stage) is free once it completes
    ~PromptGemm() {
        if (xs != nullptr) strata::gpu::stream_sync(xs);
        for (int h = 0; h < 2; ++h) {
            if (ready[h] != nullptr) strata::gpu::event_destroy(ready[h]);
            if (used[h] != nullptr) strata::gpu::event_destroy(used[h]);
        }
        if (xs != nullptr) strata::gpu::stream_destroy(xs);
    }
    bool overlap() const { return xs != nullptr && ready[0] && ready[1] && used[0] && used[1]; }
};

}  // namespace strata::core
