// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/qi_text.hpp - the image model's text encoder (Qwen3-VL-8B's language part, its GGUF) on the GPU:
// the token ids of the prompt (the server tokenizes it in the image model's chat form) to the last block's output
// states, before the final norm, the DiT's context (the rows after the system turn).
#pragma once

#include "strata/prefill/gemm.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata {
class GgufFile;
}

namespace strata::image {

class QiText {
public:
    QiText();
    ~QiText();
    QiText(const QiText&) = delete;
    QiText& operator=(const QiText&) = delete;

    bool open(const std::string& path, void* stream, std::string& err);

    /// The last block's output [ids.size()][hidden] (host FP32) for the prompt's token ids.
    std::vector<float> encode(const std::vector<int32_t>& ids);

    int64_t hidden() const { return C_; }
    uint64_t weight_bytes() const { return weight_bytes_; }

private:
    struct W {
        int type = 0;
        int64_t n = 0, k = 0;
        const void* d = nullptr;
    };
    struct Layer {
        W q, k, v, o, gate, up, down;
        const float *attn_norm = nullptr, *ffn_norm = nullptr, *qn = nullptr, *kn = nullptr;
    };
    W upload(const std::string& name);
    const float* upload_f32(const std::string& name);
    void ensure(int64_t T);

    std::unique_ptr<GgufFile> g_;
    void* s_ = nullptr;
    prefill::Gemm gemm_;
    std::vector<void*> allocs_;
    uint64_t weight_bytes_ = 0;
    int64_t C_ = 4096, heads_ = 32, kv_heads_ = 8, head_dim_ = 128, inter_ = 12288;
    float theta_ = 5e6f, eps_ = 1e-6f;
    int embd_type_ = 0;
    uint64_t embd_row_bytes_ = 0;
    const uint8_t* embd_ = nullptr;   // the token embeddings in the mapped file (rows gathered per prompt)
    std::vector<Layer> layers_;
    int64_t cap_T_ = 0;
    float *x_ = nullptr, *q_ = nullptr, *k_ = nullptr, *v_ = nullptr, *a_ = nullptr, *h_ = nullptr, *gu_ = nullptr;
    uint16_t *h16_ = nullptr, *q16_ = nullptr, *k16_ = nullptr, *v16_ = nullptr;
    uint8_t* rows_ = nullptr;
};

}  // namespace strata::image
