// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/image_lora.hpp - the LoRA adapters the image engine applies at run time: an adapter file read
// into the host tensors of the forms ComfyUI's core reads (comfy/lora.py, comfy/weight_adapter/), and the delta a
// DiT's products add.
//
// A file's keys are brought to the DiT's own names (the prefix dropped, the suffix that says which tensor of the pair
// it is), then grouped: one entry a weight, with the form it holds.  The DiT says which of its weights an entry
// changes and over which rows (a fused weight such as Qwen-Image-2.1's img_mlp.gate_up takes a row range, or the
// separate gate_layer and proj of diffusers' form).  The delta, for the plain form: y[t] += scale * (x[t] A^T) B^T,
// A [rank][k] and B [n][rank], in FP16 with FP32 sums (LoraDeviceEntry::add).
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace strata::prefill {
class Gemm;
}

namespace strata::image {

/// The delta forms a file may hold, as ComfyUI's adapters name them.
enum class LoraKind : int {
    lora = 0,   // lora_A/B, lora.down/up, lora_down/up: y += scale * (x A^T) B^T
    lokr = 1,   // lokr_w1 (x) lokr_w2, or their a/b factors (Tucker's lokr_t2 too)
    loha = 2,   // (hada_w1_a hada_w1_b) * (hada_w2_a hada_w2_b)
    glora = 3,  // a1/a2/b1/b2
    oft = 4,    // oft_blocks: the output's rows turned by a block-diagonal orthogonal matrix (BOFT: 4 dimensions)
    diff = 5,   // a full delta (.diff, .set_weight): y += scale * x Delta^T
};

/// The kind's name, as the reports and the errors write it.
const char* lora_kind_name(LoraKind k);

/// One weight's delta, on the host: the tensors the file gave, A and B in FP16 (half the RAM of FP32, which is what
/// the GPU takes), the scales in FP32.  The other forms' tensors are kept in `extra` by their role (`lokr_w1`,
/// `hada_w1_a`, `oft_blocks`, `diff`, ...) and FP32, for the forms the engine learns to apply.
struct LoraEntry {
    std::string key;                  // the DiT's name, e.g. "transformer_blocks.0.attn.to_q"
    LoraKind kind = LoraKind::lora;
    int64_t rows = 0, cols = 0, rank = 0;
    bool alpha_set = false;
    float alpha = 0.0f;               // from ".alpha" (ComfyUI's rule: none, 1.0)
    /// alpha / rank, the file's own scale (the caller's strength multiplies it).
    float scale() const { return alpha_set && rank > 0 ? alpha / static_cast<float>(rank) : 1.0f; }
    std::vector<uint16_t> a, b;       // lora: a [rank][cols], b [rows][rank], row-major
    std::vector<float> dora;          // dora_scale (DoRA): over the rows, else over the columns
    bool dora_over_rows = true;
    std::map<std::string, std::vector<float>> extra;   // the other forms' tensors, FP32, by role
    std::map<std::string, std::vector<int64_t>> shapes;   // their shapes, the file's order
};

/// An adapter file, read: its entries, its metadata.
class LoraFile {
public:
    /// Reads `path` (a safetensors or a GGUF file, by its first bytes).  Throws std::runtime_error saying what is
    /// wrong (the file, the key), as the image engine's other readers do.
    static std::unique_ptr<LoraFile> open(const std::string& path);

    const std::vector<LoraEntry>& entries() const { return entries_; }
    const std::string& path() const { return path_; }
    /// The container's string metadata (safetensors' __metadata__; a GGUF's string fields), for the listing.
    const std::map<std::string, std::string>& metadata() const { return meta_; }
    std::string meta(const std::string& key) const;
    /// What the entries hold on the host.
    uint64_t host_bytes() const;

private:
    std::string path_;
    std::vector<LoraEntry> entries_;
    std::map<std::string, std::string> meta_;
};

/// One entry's A and B on the GPU (FP16), and the entry's delta applied to a product's output.
class LoraDeviceEntry {
public:
    LoraDeviceEntry() = default;
    ~LoraDeviceEntry();
    LoraDeviceEntry(const LoraDeviceEntry&) = delete;
    LoraDeviceEntry& operator=(const LoraDeviceEntry&) = delete;

    /// Puts the entry's A and B on the stream's GPU.  False with err on a failure (the memory, a form the GPU cannot
    /// take: the shapes are the file's own and were checked when it was read).
    bool upload(const LoraEntry& e, void* stream, std::string& err);
    void release();
    const uint16_t* a() const { return a_; }
    const uint16_t* b() const { return b_; }
    int64_t rank() const { return rank_; }
    int64_t cols() const { return cols_; }
    int64_t rows() const { return rows_; }
    uint64_t bytes() const { return bytes_; }

    /// y[t][ldy] (from column `row0`) += scale * (x16[t] A^T) B^T, t < T, for the entry's first `n` rows.  A W8A8
    /// product's output is its weights' and activations' scales applied (no row scale of the input): `row_scale`
    /// (null: none) is the input rows' scale, which the base product's sum does not carry and this delta must.
    /// `tmp_f32` and `tmp16` hold T * rank values each (grown by the caller).
    void add(const uint16_t* x16, int64_t T, float* y, int64_t ldy, int64_t row0, int64_t n, float scale,
             const float* row_scale, prefill::Gemm& gemm, float* tmp_f32, uint16_t* tmp16, void* stream) const;

private:
    uint16_t* a_ = nullptr;
    uint16_t* b_ = nullptr;
    int64_t rank_ = 0, cols_ = 0, rows_ = 0;
    uint64_t bytes_ = 0;
};

/// The host tensors of `e` as FP16 (a file's BF16, F16 or F32), for a test or a form built by hand.
void lora_to_f16(const std::vector<float>& f32, std::vector<uint16_t>& out);

}  // namespace strata::image