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
// A [rank][k] and B [n][rank], in FP16 with FP32 sums (LoraDeviceEntry::add).  LoKr's delta is its two factors'
// Kronecker product, w1 [out_l][in_m] and w2 [out_k][in_n]: the input's rows read in in_m groups of in_n, w2's product
// of each group, then w1's over the groups (the same two steps ComfyUI's adapter takes).
//
// A DoRA (dora_scale, ComfyUI's weight_decompose) keeps the weight it changes: c = m / (|W| a row + eps) over the
// weight's output rows, and W' = W + s (c (W + alpha/rank * BA) - W), so a product's output is
// y = [1 + s (c - 1)] (W x) + s c (alpha/rank BA x): the base product's rows scaled by 1 + s (c - 1) (LoraPart::base,
// the DiT's product applies it where its output is in the input's units) and the delta's rows by c (folded into B when
// the entry goes to the GPU).  eps is FP32's, whatever the file's tensors are (LP64_IEC_559's 1.1920929e-07).
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

/// DoRA's eps: the weight dtype's rounding ComfyUI adds to a row's norm (torch.finfo(weight.dtype).eps); the engine's
/// weights are FP32 on the host whatever their file holds, so FP32's, the same for every file and GPU.
constexpr float kLoraDoraEps = 1.1920929e-07f;

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

/// LoKr's geometry: w1 [out_l][in_m] and w2 [out_k][in_n], the file's matrices or its two factors' products (in
/// `extra`), the delta their Kronecker product is at the weight's rows out_l * out_k and input columns in_m * in_n.
struct LokrShape {
    int64_t out_l = 0;   // w1's rows
    int64_t in_m = 0;    // w1's columns, the input's groups
    int64_t out_k = 0;   // w2's rows
    int64_t in_n = 0;    // w2's columns
    bool empty() const { return out_l <= 0 || in_m <= 0 || out_k <= 0 || in_n <= 0; }
};

/// One weight's delta, on the host: the tensors the file gave, A and B in FP16 (half the RAM of FP32, which is what
/// the GPU takes), the scales in FP32.  The other forms' tensors are kept in `extra` by their role (`hada_w1_a`,
/// `oft_blocks`, `diff`, ...) and FP32, for the forms the engine learns to apply; a LoKr's materialized w1 and w2 are
/// in `extra` under "lokr_w1" and "lokr_w2".
struct LoraEntry {
    std::string key;                  // the DiT's name, e.g. "transformer_blocks.0.attn.to_q"
    LoraKind kind = LoraKind::lora;
    int64_t rows = 0, cols = 0, rank = 0;
    bool alpha_set = false;
    float alpha = 0.0f;               // from ".alpha" (ComfyUI's rule: none, 1.0)
    LokrShape lokr;                   // LoraKind::lokr: the factors' shape, `rows` and `cols` their product's
    /// alpha / rank, the file's own scale (the caller's strength multiplies it).  LoKr's rank is the factorization's
    /// (ComfyUI divides the file's alpha by it, and by nothing where both of its matrices are whole).
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

/// One adapter a request applies: its file, read once and shared while its request runs, and the strength its delta is
/// scaled by (the file's own alpha / rank is the entry's).
struct LoraUse {
    std::shared_ptr<const LoraFile> file;
    float scale = 1.0f;
};

/// The bytes the adapters take on the GPU: their entries' tensors (FP16), and the products' scratch of T rows (T times
/// the most values an entry's delta holds a row, FP32 and FP16 each): what a request's budget puts beside the model.
uint64_t lora_gpu_bytes(const std::vector<LoraUse>& loras, int64_t T);

/// The values an entry's delta holds for each row of the product's scratch (FP32): a plain LoRA's rank, LoKr's
/// in_m * out_k (its product's intermediate [T][in_m][out_k]).  A DiT sizes its deltas' scratch by the largest.
int64_t lora_scratch_cols(const LoraEntry& e);

/// One entry's tensors on the GPU (FP16: a plain LoRA's A and B, a LoKr's w1 and w2), and the entry's delta applied
/// to a product's output.
class LoraDeviceEntry {
public:
    LoraDeviceEntry() = default;
    ~LoraDeviceEntry();
    LoraDeviceEntry(const LoraDeviceEntry&) = delete;
    LoraDeviceEntry& operator=(const LoraDeviceEntry&) = delete;

    /// Puts the entry's A and B on the stream's GPU.  `row_coef` (null: 1), one a row of the entry's, folds into B
    /// (the plain form's DoRA: the rows' c, a product's output scaled there; a LoKr's rows are the Kronecker product's
    /// and the caller refuses its DoRA).  False with err on a failure (the memory, a form the GPU cannot take: the
    /// shapes are the file's own and were checked when it was read).
    bool upload(const LoraEntry& e, void* stream, std::string& err, const float* row_coef = nullptr);
    void release();
    const uint16_t* a() const { return a_; }
    const uint16_t* b() const { return b_; }
    int64_t rank() const { return rank_; }
    int64_t cols() const { return cols_; }
    int64_t rows() const { return rows_; }
    uint64_t bytes() const { return bytes_; }

    /// y[t][row0 + c] (row stride ldy) += scale * (x16[t] A^T) B^T, t < T, for the entry's rows
    /// [src_row, src_row + n) into the weight's rows [row0, row0 + n) (a fused weight's half, or the whole).  An
    /// LoKr's rows are its Kronecker product's; the delta is the same product's, its own way.  A W8A8 product's output
    /// is its weights' and activations' scales applied (no row scale of the input): `row_scale` (null: none) is the
    /// input rows' scale, which the base product's sum does not carry and this delta must.  `tmp_f32` and `tmp16` hold
    /// T * lora_scratch_cols(e) values each (grown by the caller).
    void add(const uint16_t* x16, int64_t T, float* y, int64_t ldy, int64_t src_row, int64_t row0, int64_t n,
             float scale, const float* row_scale, prefill::Gemm& gemm, float* tmp_f32, uint16_t* tmp16,
             void* stream) const;

private:
    void add_lokr(const uint16_t* x16, int64_t T, float* y, int64_t ldy, int64_t src_row, int64_t row0, int64_t n,
                  float scale, const float* row_scale, prefill::Gemm& gemm, float* tmp_f32, void* stream) const;

    LoraKind kind_ = LoraKind::lora;
    uint16_t* a_ = nullptr;   // lora: A [rank][cols]
    uint16_t* b_ = nullptr;   // lora: B [rows][rank]
    uint16_t* w1_ = nullptr;  // lokr: w1 [out_l][in_m]
    uint16_t* w2_ = nullptr;  // lokr: w2 [out_k][in_n]
    LokrShape lokr_;
    int64_t rank_ = 0, cols_ = 0, rows_ = 0;
    uint64_t bytes_ = 0;
};

/// One share of a weight's delta: the entry's rows [src_row, src_row + n) at the weight's rows [row0, row0 + n), times
/// `scale` (the file's and the request's).  Two files may change one weight: the weight takes a part each.
struct LoraPart {
    std::shared_ptr<const LoraDeviceEntry> entry;
    int64_t row0 = 0, n = 0, src_row = 0;
    float scale = 1.0f;
    /// DoRA: the base product's factor of each of the part's rows, 1 + strength (c - 1) (empty: not a DoRA, the rows
    /// as the weight has them).
    std::vector<float> base;
};

/// The deltas one weight adds to its products, as the DiT bound them.  A weight without one has no layer.
class LoraLayer {
public:
    LoraLayer() = default;
    LoraLayer(const LoraLayer&) = delete;
    LoraLayer& operator=(const LoraLayer&) = delete;

    std::vector<LoraPart> parts;
    bool empty() const { return parts.empty(); }
    /// The GPU memory the parts' entries hold (and the base product's factors).
    uint64_t bytes() const;

    /// The base product's factors over all of the weight's rows, 1 where no DoRA part changes the row: made from the
    /// parts (`rows`: the weight's rows), on the GPU once the DiT has bound them all.  null: no part is a DoRA's.
    bool build_base(int64_t rows, void* stream, std::string& err);
    bool has_base() const { return base_ != nullptr; }
    const float* base() const { return base_; }

    /// y[t][ldy] += the parts that cover the weight's rows [w_row0, w_row0 + w_rows), each into its own rows there.
    /// `tmp_f32` and `tmp16` hold T * rank values each (grown by the caller).
    void add(const uint16_t* x16, int64_t T, float* y, int64_t ldy, int64_t w_row0, int64_t w_rows,
             const float* row_scale, prefill::Gemm& gemm, float* tmp_f32, uint16_t* tmp16, void* stream) const;

    ~LoraLayer();

private:
    float* base_ = nullptr;
    int64_t base_rows_ = 0;
    bool base_built_ = false;
};

/// One DoRA part's rows: `norms` (the weight's rows' norms, FP32) -> the delta's factor of each of the part's rows
/// [src_row, src_row + n) of the entry at the weight's rows [row0, row0 + n), c = dora_scale / (norm + eps) in `coef`,
/// and the base product's 1 + strength (c - 1) in `base`.  False with err for the DoRA the engine does not take: a
/// scale over the columns (ComfyUI's other axis: its norm is of the weight with the delta in it, and the columns are
/// an input's), or a form whose rows a product's output does not have (a LoKr's are its Kronecker product's).
bool lora_dora_rows(const LoraEntry& e, int64_t src_row, int64_t row0, int64_t n, const std::vector<float>& norms,
                    float strength, std::vector<float>& coef, std::vector<float>& base, std::string& err);

/// The bound part whose rows overlap [row0, row0 + n) of a weight's, where one of the two is a DoRA's: the two cannot
/// be composed (ComfyUI patches the weights in the file's order, and a DoRA's c is the weight's as it was before the
/// other's patch), so the caller refuses the set.  null: no such part.
const LoraPart* lora_dora_overlap(const LoraLayer* layer, int64_t row0, int64_t n, bool dora);

/// The host tensors of `e` as FP16 (a file's BF16, F16 or F32), for a test or a form built by hand.
void lora_to_f16(const std::vector<float>& f32, std::vector<uint16_t>& out);

}  // namespace strata::image