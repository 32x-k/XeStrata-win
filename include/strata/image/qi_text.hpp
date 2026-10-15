// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/qi_text.hpp - the image model's text encoder (Qwen3-VL-8B's language part, its GGUF) on the GPU:
// the token ids of the prompt (the server tokenizes it in the image model's chat form) to the last block's output
// states, before the final norm, the DiT's context (the rows after the system turn).
#pragma once

#include "strata/image/qi_stream.hpp"
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
    /// The layers' bytes (in the file) and the two slots a streamed walk takes at least.
    uint64_t layer_bytes() const { return layers_stream_.total(); }
    uint64_t layer_min() const { return 2 * (layers_stream_.count() ? layers_stream_.bytes(0) : 0); }
    /// The VRAM the layers may take (LayerStream: those that fit stay, the rest are streamed).
    void set_layer_budget(uint64_t bytes) { layers_stream_.configure(bytes, s_); }
    int resident_layers() const { return layers_stream_.resident(); }
    void drop_layers() { layers_stream_.drop(); }

private:
    struct W {
        int type = 0;
        int64_t n = 0, k = 0;
        const void* d = nullptr;
    };
    struct Layer {
        W q, k, v, o, gate, up, down;   // d: the offset within the layer's region (LayerStream)
        uint64_t attn_norm = 0, ffn_norm = 0, qn = 0, kn = 0;
    };
    void ensure(int64_t T);

    std::unique_ptr<GgufFile> g_;
    void* s_ = nullptr;
    prefill::Gemm gemm_;
    std::vector<void*> allocs_;
    int64_t C_ = 4096, heads_ = 32, kv_heads_ = 8, head_dim_ = 128, inter_ = 12288;
    float theta_ = 5e6f, eps_ = 1e-6f;
    int embd_type_ = 0;
    uint64_t embd_row_bytes_ = 0;
    const uint8_t* embd_ = nullptr;   // the token embeddings in the mapped file (rows gathered per prompt)
    std::vector<Layer> layers_;
    LayerStream layers_stream_;
    int64_t cap_T_ = 0;
    float *x_ = nullptr, *q_ = nullptr, *k_ = nullptr, *v_ = nullptr, *a_ = nullptr, *h_ = nullptr, *gu_ = nullptr;
    uint16_t *h16_ = nullptr, *q16_ = nullptr, *k16_ = nullptr, *v16_ = nullptr;
    uint8_t* rows_ = nullptr;
};

}  // namespace strata::image
