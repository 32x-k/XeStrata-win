// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/image_lora.cpp - an adapter file read into the host tensors of ComfyUI's LoRA forms, its keys brought to
// the DiT's own names; and one entry's A and B on the GPU, its delta added to a product's output.
//
// The keys: ComfyUI's adapters take a weight's name `x` and the tensors `x.lora_up.weight` and `x.lora_down.weight`
// (or `x.lora_B.weight` and `x.lora_A.weight`, `x.lora.up.weight` and `x.lora.down.weight`, or the bare
// `x.lora_B`/`x.lora_A` of PEFT's own saves), with `x.alpha` and `x.dora_scale` beside them.  A file's prefix says
// whose names they are: `transformer.` for diffusers' saves, `diffusion_model.` and `model.diffusion_model.` for
// ComfyUI's and a GGUF's.  What is left is the DiT's own name, which the DiT's target table matches.
#include "strata/image/image_lora.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/image/safetensors.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/prefill/gemm.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace strata::image {

namespace {

bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

void strip_suffix(std::string& s, const std::string& suffix) { s.erase(s.size() - suffix.size()); }

// The prefixes a file's keys may carry; the first that matches is dropped (`model.diffusion_model.` before
// `diffusion_model.`, so the longest is listed first).
const char* const kPrefixes[] = {"model.diffusion_model.", "diffusion_model.", "transformer.", "lora_unet_"};

// The suffixes that say which tensor of a weight a key is; the role is the name here, the key is what is left.  The
// first that matches wins, so a longer suffix is listed before its prefix (`lora_A.default` before `lora_A`).
struct RoleSuffix {
    const char* suffix;
    const char* role;
};
const RoleSuffix kRoles[] = {
    {".lora_A.default", "lora_A"},   {".lora_B.default", "lora_B"},
    {".lora.down", "lora_A"},        {".lora.up", "lora_B"},
    {".lora_A", "lora_A"},           {".lora_B", "lora_B"},
    {".lora_down", "lora_A"},        {".lora_up", "lora_B"},
    {".lora_mid", "lora_mid"},       {".lora_linear_layer.down", "lora_A"},
    {".lora_linear_layer.up", "lora_B"},
    {".alpha", "alpha"},             {".dora_scale", "dora_scale"},
    {".lokr_w1_a", "lokr_w1_a"},     {".lokr_w1_b", "lokr_w1_b"},
    {".lokr_w2_a", "lokr_w2_a"},     {".lokr_w2_b", "lokr_w2_b"},
    {".lokr_w1", "lokr_w1"},         {".lokr_w2", "lokr_w2"},
    {".lokr_t2", "lokr_t2"},
    {".hada_w1_a", "hada_w1_a"},     {".hada_w1_b", "hada_w1_b"},
    {".hada_w2_a", "hada_w2_a"},     {".hada_w2_b", "hada_w2_b"},
    {".hada_t1", "hada_t1"},         {".hada_t2", "hada_t2"},
    {".oft_blocks", "oft_blocks"},   {".rescale", "rescale"},
    {".a1", "a1"},                   {".a2", "a2"},   {".b1", "b1"},   {".b2", "b2"},
    {".diff_b", "diff_b"},           {".diff", "diff"},   {".set_weight", "set_weight"},
    {".w_norm", "w_norm"},           {".b_norm", "b_norm"},
    {".reshape_weight", "reshape_weight"},
};

// A key as the file writes it -> the DiT's name and the role (empty: not one of the forms' tensors).
void split_key(const std::string& raw, std::string& key, std::string& role) {
    std::string s = raw;
    bool kohya = false;
    for (const char* p : kPrefixes) {
        if (s.rfind(p, 0) == 0) {
            kohya = std::strcmp(p, "lora_unet_") == 0;
            s.erase(0, std::strlen(p));
            break;
        }
    }
    if (ends_with(s, ".weight")) strip_suffix(s, ".weight");   // every form but PEFT's bare lora_A and lora_B
    role.clear();
    for (const RoleSuffix& r : kRoles) {
        if (ends_with(s, r.suffix)) {
            strip_suffix(s, r.suffix);
            role = r.role;
            break;
        }
    }
    if (kohya)   // sd-scripts' keys: the module's dots written as underscores (these models name none this way)
        std::replace(s.begin(), s.end(), '_', '.');
    key = s;
}

// A file's tensor, as it was read: its shape (a matrix's rows then columns) and its values in FP32.
struct RawTensor {
    std::vector<int64_t> shape;
    std::vector<float> v;
};

// The tensors of one weight, by role.
struct Group {
    std::string key;
    std::map<std::string, RawTensor> by_role;
};

std::string path_and(const std::string& path, const std::string& key) {
    std::string s = path;
    s += ": ";
    s += key;
    return s;
}

// "<path>: <key>: <what>", the errors' shape (a chain of + would copy the string at each step)
std::string path_key_what(const std::string& path, const std::string& key, const std::string& what) {
    std::string s = path_and(path, key);
    s += ": ";
    s += what;
    return s;
}

// A safetensors' tensors, by name (the bytes of a tensor are read only when its name is asked for).
class SafeReader {
public:
    explicit SafeReader(const std::string& path) : path_(path), st_(path) {}
    std::vector<std::string> names() const {
        std::vector<std::string> out;
        for (const auto& [name, t] : st_.tensors()) out.push_back(name);
        return out;
    }
    const std::map<std::string, std::string>& metadata() const { return st_.metadata(); }
    RawTensor get(const std::string& name) {
        const SafeTensor* t = st_.find(name);
        if (t == nullptr) throw std::runtime_error(path_ + ": no tensor " + name);
        if (t->dtype != "F32" && t->dtype != "F16" && t->dtype != "BF16")
            throw std::runtime_error(path_ + ": " + name + ": a tensor of " + t->dtype +
                                     " (adapters are F16, BF16 or F32)");
        if (t->shape.size() > 2) throw std::runtime_error(path_ + ": " + name + ": more than two dimensions");
        return {t->shape, st_.f32(name)};
    }

private:
    std::string path_;
    SafeTensors st_;
};

// A GGUF's tensors, by name (F32, F16 and BF16; a quantized adapter is not one).
class GgufReader {
public:
    explicit GgufReader(const std::string& path) : g_(path) {}
    std::vector<std::string> names() const {
        std::vector<std::string> out;
        for (const TensorInfo& t : g_.tensors()) out.push_back(t.name);
        return out;
    }
    std::map<std::string, std::string> metadata() const {
        std::map<std::string, std::string> out;
        for (const auto& [k, v] : g_.metadata())
            if (v.type == MetaType::STRING) out[k] = v.s;
        return out;
    }
    RawTensor get(const std::string& name) {
        const TensorInfo* t = g_.find(name);
        if (t == nullptr) throw std::runtime_error(g_.path() + ": no tensor " + name);
        // GGUF's shape runs from the innermost dimension (dimension 0 varies fastest): a matrix of rows x columns is
        // {columns, rows} there, and its data is row-major in the engine's order.
        std::vector<int64_t> shape(t->shape.rbegin(), t->shape.rend());
        if (shape.size() > 2) throw std::runtime_error(g_.path() + ": " + name + ": more than two dimensions");
        const int64_t n = (int64_t) t->elements();
        const auto* p = g_.tensor_data(*t);
        RawTensor out;
        out.shape = shape;
        out.v.resize((size_t) n);
        if (t->type == 0) {   // F32
            std::memcpy(out.v.data(), p, out.v.size() * 4);
        } else if (t->type == 1) {   // F16
            const auto* h = reinterpret_cast<const uint16_t*>(p);
            for (size_t i = 0; i < out.v.size(); ++i) out.v[i] = strata::kernels::f32_from_f16(h[i]);
        } else if (t->type == 30) {   // BF16
            const auto* h = reinterpret_cast<const uint16_t*>(p);
            for (size_t i = 0; i < out.v.size(); ++i) out.v[i] = strata::kernels::f32_from_bf16(h[i]);
        } else {
            throw std::runtime_error(g_.path() + ": " + name + ": a tensor of " + t->type_name() +
                                     " (adapters are F16, BF16 or F32)");
        }
        return out;
    }

private:
    GgufFile g_;
};

// The metadata's first number after "key" (PEFT writes alpha and rank there as JSON, its keys prefixed with the
// model's, e.g. "transformer.lora_alpha": 64).
bool json_number(const std::string& s, const std::string& key, double& out) {
    const std::string pat = "\"" + key + "\"";
    size_t at = s.find(pat);
    while (at != std::string::npos) {
        size_t i = at + pat.size();
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
        if (i < s.size() && s[i] == ':') {
            ++i;
            while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
            const char* b = s.c_str() + i;
            char* e = nullptr;
            const double v = std::strtod(b, &e);
            if (e != b) {
                out = v;
                return true;
            }
        }
        at = s.find(pat, at + 1);
    }
    return false;
}

bool json_true(const std::string& s, const std::string& key) {
    const std::string pat = "\"" + key + "\"";
    const size_t at = s.find(pat);
    if (at == std::string::npos) return false;
    size_t i = s.find(':', at + pat.size());
    if (i == std::string::npos) return false;
    ++i;
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) ++i;
    return s.compare(i, 4, "true") == 0;
}

bool is_training_aid(const std::string& key) {
    // SimpleTuner's auxiliary projector: trained beside the adapter, not applied to the model.
    return key.find("crepa_projector") != std::string::npos;
}

LoraKind kind_of(const Group& g, const std::string& path) {
    const auto has = [&g](const char* r) { return g.by_role.count(r) != 0; };
    if (has("lora_A") || has("lora_B")) return LoraKind::lora;
    if (has("lokr_w1") || has("lokr_w2") || has("lokr_w1_a") || has("lokr_w2_a")) return LoraKind::lokr;
    if (has("hada_w1_a") || has("hada_w2_a")) return LoraKind::loha;
    if (has("a1") || has("b1")) return LoraKind::glora;
    if (has("oft_blocks")) return LoraKind::oft;
    if (has("diff") || has("set_weight")) return LoraKind::diff;
    throw std::runtime_error(path_and(path, g.key) +
                             ": not a LoRA tensor (no lora_A/B, lokr_w1/w2, hada_w1_a, a1/b1, oft_blocks, diff or "
                             "set_weight beside it)");
}

// One weight's role tensors -> its entry.
LoraEntry make_entry(Group& g, const std::string& path) {
    LoraEntry e;
    e.key = g.key;
    e.kind = kind_of(g, path);
    const auto role = [&g](const char* r) -> RawTensor* {
        auto it = g.by_role.find(r);
        return it == g.by_role.end() ? nullptr : &it->second;
    };
    if (e.kind == LoraKind::lora) {
        RawTensor* a = role("lora_A");
        RawTensor* b = role("lora_B");
        if (a == nullptr || b == nullptr)
            throw std::runtime_error(path_and(path, e.key) + ": a LoRA weight needs both lora_A (down) and lora_B (up)");
        if (a->shape.size() != 2 || b->shape.size() != 2)
            throw std::runtime_error(path_and(path, e.key) + ": lora_A and lora_B must be matrices");
        e.rank = a->shape[0];
        e.cols = a->shape[1];
        e.rows = b->shape[0];
        if (b->shape[1] != e.rank)
            throw std::runtime_error(path_and(path, e.key) + ": lora_B is " + std::to_string(b->shape[0]) + " x " +
                                     std::to_string(b->shape[1]) + ", not " + std::to_string(e.rows) + " x " +
                                     std::to_string(e.rank) + " (lora_A's rank)");
        lora_to_f16(a->v, e.a);
        lora_to_f16(b->v, e.b);
    }
    if (RawTensor* al = role("alpha"); al != nullptr) {
        if (al->v.size() != 1) throw std::runtime_error(path_and(path, e.key) + ": alpha is not one value");
        e.alpha = al->v[0];
        e.alpha_set = true;
    }
    if (RawTensor* d = role("dora_scale"); d != nullptr) {
        if (d->v.size() == (size_t) e.rows) {
            e.dora = d->v;
            e.dora_over_rows = true;
        } else if (d->v.size() == (size_t) e.cols) {
            e.dora = d->v;
            e.dora_over_rows = false;
        } else {
            throw std::runtime_error(path_and(path, e.key) + ": dora_scale is neither the rows nor the columns");
        }
    }
    if (role("lora_mid") != nullptr)
        throw std::runtime_error(path_and(path, e.key) +
                                 ": a LoRA with a middle layer (a convolution's) is not one the image engine's linear "
                                 "products take");
    // the other forms' tensors, as they are (the forms the engine learns to apply later read them)
    for (const auto& [r, t] : g.by_role) {
        if (r == "lora_A" || r == "lora_B" || r == "alpha" || r == "dora_scale") continue;
        e.shapes[r] = t.shape;
        e.extra[r] = t.v;
    }
    return e;
}

// Every key of the file, grouped by the weight it belongs to, then one entry a weight.
template <class Reader>
std::vector<LoraEntry> read_entries(Reader& rd, const std::string& path) {
    std::vector<std::string> names = rd.names();
    std::sort(names.begin(), names.end());
    std::vector<Group> groups;
    std::map<std::string, size_t> index;
    for (const std::string& raw : names) {
        std::string key, role;
        split_key(raw, key, role);
        if (role.empty()) continue;   // a key of the file's own (a quantization's, a config's)
        if (key.empty()) throw std::runtime_error(path_key_what(path, raw, "a tensor with no weight's name"));
        if (is_training_aid(key)) {
            std::fprintf(stderr, "image engine: %s: %s: a training aid, not a LoRA tensor: left out\n", path.c_str(),
                         raw.c_str());
            continue;
        }
        auto it = index.find(key);
        if (it == index.end()) {
            it = index.emplace(key, groups.size()).first;
            groups.push_back(Group{key, {}});
        }
        auto& roles = groups[it->second].by_role;
        if (roles.count(role) != 0) {
            std::string what = "two tensors of ";
            what += role;
            throw std::runtime_error(path_key_what(path, key, what));
        }
        roles.emplace(role, rd.get(raw));
    }
    std::vector<LoraEntry> entries;
    entries.reserve(groups.size());
    for (Group& g : groups) entries.push_back(make_entry(g, path));
    return entries;
}

// The scale a PEFT-form file keeps in its metadata ("lora_adapter_metadata": a JSON string), where no ".alpha"
// tensor says it: alpha / rank.  rsLoRA's scale is another one (alpha / sqrt(rank)) and is refused.
void apply_peft_metadata(std::vector<LoraEntry>& entries, const std::map<std::string, std::string>& meta) {
    std::string json;
    for (const auto& [k, v] : meta)
        if (k == "lora_adapter_metadata") json = v;
    if (json.empty()) return;
    if (json_true(json, "use_rslora"))
        throw std::runtime_error("the adapter is rsLoRA (use_rslora), whose scale is not the plain alpha / rank: not "
                                 "supported yet");
    double alpha = 0;
    if (!json_number(json, "lora_alpha", alpha)) return;
    for (LoraEntry& e : entries) {
        if (e.alpha_set) continue;   // the file's own ".alpha" wins
        e.alpha = (float) alpha;
        e.alpha_set = true;
    }
}

}  // namespace

const char* lora_kind_name(LoraKind k) {
    switch (k) {
    case LoraKind::lora: return "lora";
    case LoraKind::lokr: return "lokr";
    case LoraKind::loha: return "loha";
    case LoraKind::glora: return "glora";
    case LoraKind::oft: return "oft";
    case LoraKind::diff: return "diff";
    }
    return "?";
}

void lora_to_f16(const std::vector<float>& f32, std::vector<uint16_t>& out) {
    out.resize(f32.size());
    for (size_t i = 0; i < f32.size(); ++i) out[i] = strata::kernels::f16_from_f32(f32[i]);
}

std::string LoraFile::meta(const std::string& key) const {
    auto it = meta_.find(key);
    return it == meta_.end() ? std::string() : it->second;
}

uint64_t LoraFile::host_bytes() const {
    uint64_t n = 0;
    for (const LoraEntry& e : entries_) {
        n += e.a.size() * 2 + e.b.size() * 2 + e.dora.size() * 4;
        for (const auto& [role, v] : e.extra) n += v.size() * 4;
    }
    return n;
}

std::unique_ptr<LoraFile> LoraFile::open(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    char magic[4] = {};
    f.read(magic, 4);
    if (!f) throw std::runtime_error(path + ": shorter than a file's header");
    f.close();
    auto out = std::unique_ptr<LoraFile>(new LoraFile());
    out->path_ = path;
    if (std::memcmp(magic, "GGUF", 4) == 0) {
        GgufReader rd(path);
        out->entries_ = read_entries(rd, path);
        out->meta_ = rd.metadata();
    } else {
        SafeReader rd(path);
        out->entries_ = read_entries(rd, path);
        out->meta_ = rd.metadata();
    }
    apply_peft_metadata(out->entries_, out->meta_);
    if (out->entries_.empty()) throw std::runtime_error(path + ": no LoRA tensors");
    return out;
}

// ---------------------------------------------------------------------------------------------- the GPU side
namespace lora_kernels {

// tmp16[i] = f16(tmp[i] * s * row_scale[i / N]): the first product's output, scaled into the second's FP16 input
struct ScaleRows {
    const float* in;
    float s;
    const float* row_scale;
    int64_t N;
    uint16_t* out;
    void operator()(sycl::id<1> id) const {
        const size_t i = id[0];
        float v = in[i] * s;
        if (row_scale != nullptr) v *= row_scale[i / (size_t) N];
        out[i] = strata::kernels::f16_from_f32(v);
    }
};

}  // namespace lora_kernels

LoraDeviceEntry::~LoraDeviceEntry() { release(); }

void LoraDeviceEntry::release() {
    if (a_ != nullptr) strata::gpu::free(a_);
    if (b_ != nullptr) strata::gpu::free(b_);
    a_ = b_ = nullptr;
    rank_ = cols_ = rows_ = 0;
    bytes_ = 0;
}

bool LoraDeviceEntry::upload(const LoraEntry& e, void* stream, std::string& err) {
    release();
    if (e.kind != LoraKind::lora) {
        err = "the image engine: ";
        err += lora_kind_name(e.kind);
        err += " adapters are not applied yet (";
        err += e.key;
        err += ")";
        return false;
    }
    rank_ = e.rank;
    cols_ = e.cols;
    rows_ = e.rows;
    if (rank_ <= 0 || e.a.size() != (size_t) (e.rank * e.cols) || e.b.size() != (size_t) (e.rows * e.rank)) {
        err = "the image engine: " + e.key + ": the adapter's tensors do not match its rank and shape";
        return false;
    }
    if (!strata::gpu::alloc_device(&a_, e.a.size() * 2) || !strata::gpu::alloc_device(&b_, e.b.size() * 2)) {
        err = "the image engine: the adapter " + e.key + " on the GPU: " + strata::gpu::last_error();
        release();
        return false;
    }
    strata::gpu::copy_async(a_, e.a.data(), e.a.size() * 2, stream);
    strata::gpu::copy_async(b_, e.b.data(), e.b.size() * 2, stream);
    if (!strata::gpu::stream_sync(stream)) {
        err = "the image engine: the adapter " + e.key + " to the GPU: " + strata::gpu::last_error();
        release();
        return false;
    }
    bytes_ = (uint64_t) (e.a.size() + e.b.size()) * 2;
    return true;
}

void LoraDeviceEntry::add(const uint16_t* x16, int64_t T, float* y, int64_t ldy, int64_t row0, int64_t n, float scale,
                          const float* row_scale, prefill::Gemm& gemm, float* tmp_f32, uint16_t* tmp16,
                          void* stream) const {
    if (T <= 0 || n <= 0 || rank_ <= 0 || scale == 0.0f) return;
    if (ldy <= 0) ldy = n;
    gemm.f16(x16, a_, tmp_f32, T, rank_, cols_);   // t = x . A^T, [T][rank]
    sycl::queue& q = strata::core::Runtime::get().stream(stream);
    q.parallel_for(sycl::range<1>((size_t) (T * rank_)),
                   lora_kernels::ScaleRows{tmp_f32, scale, row_scale, rank_, tmp16});
    gemm.f16(tmp16, b_, y + row0, T, n, rank_, ldy, false, true);   // y[:, row0 ..] += t16 . B^T
}

}  // namespace strata::image