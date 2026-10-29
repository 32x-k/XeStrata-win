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
//
// A LoKr (LyCORIS) writes one weight's two matrices whole (`x.lokr_w1`, `x.lokr_w2`) or as two factors each
// (`x.lokr_w1_a` [rows][rank] and `x.lokr_w1_b` [rank][cols]), w2's under a Tucker core (`x.lokr_t2`); the delta is
// their Kronecker product, and the file's alpha is divided by the factorization's rank (comfy/weight_adapter/lokr.py).
//
// A whole delta is the weight's own matrix: `x.diff` (LyCORIS's full form, a fine tune's difference from its base),
// `x.w_norm` (the same role in ComfyUI's patch table) and `x.set_weight` (the weight itself).  ComfyUI patches them
// as W += strength * diff and W = set_weight; the engine keeps the first and interpolates the second
// (W + s (S - W)), so that a strength scales both.  `x.diff_b` and `x.b_norm` are a bias's delta, which the engine
// does not apply yet: a file holding one is refused when a set is bound, never silently sent with it.
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
        if (t->shape.size() > 4)   // a matrix; a LoKr's Tucker core has four dimensions
            throw std::runtime_error(path_ + ": " + name + ": more than four dimensions");
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
        if (shape.size() > 4)   // a matrix; a LoKr's Tucker core has four dimensions
            throw std::runtime_error(g_.path() + ": " + name + ": more than four dimensions");
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

// A row-major matrix product in FP32: out [m][n] = a [m][k] . b [k][n], as a LoKr's factors make its matrices.
std::vector<float> mat_mul(const float* a, int64_t m, int64_t k, const float* b, int64_t n) {
    std::vector<float> out((size_t) (m * n), 0.0f);
    for (int64_t i = 0; i < m; ++i) {
        float* o = out.data() + i * n;
        for (int64_t t = 0; t < k; ++t) {
            const float v = a[i * k + t];
            if (v == 0.0f) continue;
            const float* br = b + t * n;
            for (int64_t j = 0; j < n; ++j) o[j] += v * br[j];
        }
    }
    return out;
}

bool is_training_aid(const std::string& key) {
    // SimpleTuner's auxiliary projector: trained beside the adapter, not applied to the model.
    return key.find("crepa_projector") != std::string::npos;
}

LoraKind kind_of(const Group& g, const std::string& path) {
    const auto has = [&g](const char* r) { return g.by_role.count(r) != 0; };
    const bool lora = has("lora_A") || has("lora_B");
    const bool lokr = has("lokr_w1") || has("lokr_w2") || has("lokr_w1_a") || has("lokr_w2_a");
    const bool loha = has("hada_w1_a") || has("hada_w2_a");
    const bool glora = has("a1") || has("b1");
    const bool oft = has("oft_blocks");
    const bool diff = has("diff") || has("set_weight") || has("w_norm");
    // one weight, one form: a file that holds two would have one of them silently dropped (ComfyUI keeps the last of
    // its patches' types), so it is refused here, saying which
    const int forms = (int) lora + (int) lokr + (int) loha + (int) glora + (int) oft + (int) diff;
    if (forms > 1) {
        std::vector<const char*> which;
        if (lora) which.push_back("lora");
        if (lokr) which.push_back("lokr");
        if (loha) which.push_back("loha");
        if (glora) which.push_back("glora");
        if (oft) which.push_back("oft");
        if (diff) which.push_back("a whole delta");
        std::string what = "two forms of one weight: ";
        for (size_t i = 0; i < which.size(); ++i) {
            if (i != 0) what += i + 1 == which.size() ? " and " : ", ";
            what += which[i];
        }
        what += " (the engine applies one form a weight)";
        throw std::runtime_error(path_key_what(path, g.key, what));
    }
    if (lora) return LoraKind::lora;
    if (lokr) return LoraKind::lokr;
    if (loha) return LoraKind::loha;
    if (glora) return LoraKind::glora;
    if (oft) return LoraKind::oft;
    if (diff) return LoraKind::diff;
    if (has("diff_b") || has("b_norm"))
        throw std::runtime_error(path_key_what(path, g.key,
                                               "a bias diff (diff_b/b_norm) with no weight's delta: the image "
                                               "engine applies no bias's diff yet"));
    throw std::runtime_error(path_and(path, g.key) +
                             ": not a LoRA tensor (no lora_A/B, lokr_w1/w2, hada_w1_a, a1/b1, oft_blocks, diff or "
                             "set_weight beside it)");
}

// A LoKr's two matrices -> its entry: w1 [out_l][in_m] and w2 [out_k][in_n], the file's whole matrices or its two
// factors' products (w2's under a Tucker core), the delta their Kronecker product is at the weight's rows
// out_l * out_k and columns in_m * in_n.  `rank` is the factorization's, where a matrix has one: ComfyUI divides the
// file's alpha by w2's factors' rank, else w1's, and by nothing where both matrices are whole
// (comfy/weight_adapter/lokr.py's calculate_weight).
void make_lokr(LoraEntry& e, Group& g, const std::string& path) {
    const auto get = [&g](const char* r) -> const RawTensor* {
        auto it = g.by_role.find(r);
        return it == g.by_role.end() ? nullptr : &it->second;
    };
    // One of the two: the whole matrix, or its two factors' product.
    int64_t rank = 0;
    const auto matrix_of = [&](const char* full, const char* fa, const char* fb) -> RawTensor {
        const RawTensor* whole = get(full);
        const RawTensor* a = get(fa);
        const RawTensor* b = get(fb);
        if (whole != nullptr) {
            if (a != nullptr || b != nullptr)
                throw std::runtime_error(path_key_what(path, e.key, std::string("both ") + full + " and its factors " +
                                                                       fa + "/" + fb));
            if (whole->shape.size() != 2)
                throw std::runtime_error(path_key_what(path, e.key, std::string(full) + " must be a matrix"));
            return *whole;
        }
        if (a == nullptr || b == nullptr)
            throw std::runtime_error(path_key_what(path, e.key, std::string("a LoKr weight needs ") + full +
                                                                   " or both " + fa + " and " + fb));
        if (a->shape.size() != 2 || b->shape.size() != 2)
            throw std::runtime_error(path_key_what(path, e.key, std::string(fa) + " and " + fb + " must be matrices"));
        if (a->shape[1] != b->shape[0])
            throw std::runtime_error(path_key_what(path, e.key, std::string(fa) + " is " +
                                                                   std::to_string(a->shape[0]) + " x " +
                                                                   std::to_string(a->shape[1]) + ", " + fb + " " +
                                                                   std::to_string(b->shape[0]) + " x " +
                                                                   std::to_string(b->shape[1]) +
                                                                   ": not the rank each other has"));
        rank = b->shape[0];
        RawTensor out;
        out.shape = {a->shape[0], b->shape[1]};
        out.v = mat_mul(a->v.data(), a->shape[0], a->shape[1], b->v.data(), b->shape[1]);
        return out;
    };
    RawTensor w1 = matrix_of("lokr_w1", "lokr_w1_a", "lokr_w1_b");
    RawTensor w2;
    const RawTensor* t2 = get("lokr_t2");
    if (t2 == nullptr || get("lokr_w2") != nullptr) {   // a core beside the whole matrix stays unread, as ComfyUI
        w2 = matrix_of("lokr_w2", "lokr_w2_a", "lokr_w2_b");
    } else {
        // Tucker: w2 [out_k][in_n] = w2_a [core_out][out_k]^T . T . w2_b [core_in][in_n], T the core's kernel, 1 x 1
        const RawTensor* a = get("lokr_w2_a");
        const RawTensor* b = get("lokr_w2_b");
        if (a == nullptr || b == nullptr)
            throw std::runtime_error(path_key_what(path, e.key,
                                                   "a Tucker core (lokr_t2) needs w2's two factors beside it"));
        if (a->shape.size() != 2 || b->shape.size() != 2)
            throw std::runtime_error(path_key_what(path, e.key,
                                                   "lokr_w2_a and lokr_w2_b must be matrices"));
        if (!(t2->shape.size() == 2 || (t2->shape.size() == 4 && t2->shape[2] == 1 && t2->shape[3] == 1)))
            throw std::runtime_error(path_key_what(path, e.key,
                                                   "a LoKr Tucker core with a convolution's kernel dimensions: the "
                                                   "image engine's products are linear"));
        if (a->shape[0] != t2->shape[0] || b->shape[0] != t2->shape[1])
            throw std::runtime_error(path_key_what(path, e.key, "the Tucker core is " +
                                                                   std::to_string(t2->shape[0]) + " x " +
                                                                   std::to_string(t2->shape[1]) + "; w2_a's rows " +
                                                                   std::to_string(a->shape[0]) + ", w2_b's " +
                                                                   std::to_string(b->shape[0])));
        const int64_t core_out = t2->shape[0], core_in = t2->shape[1];
        const int64_t out_k = a->shape[1], in_n = b->shape[1];
        rank = core_in;   // ComfyUI's dim: w2_b's rows
        const std::vector<float> tb = mat_mul(t2->v.data(), core_out, core_in, b->v.data(), in_n);
        w2.shape = {out_k, in_n};
        w2.v.assign((size_t) (out_k * in_n), 0.0f);
        for (int64_t i = 0; i < core_out; ++i)
            for (int64_t p = 0; p < out_k; ++p) {
                const float va = a->v[(size_t) (i * out_k + p)];
                if (va == 0.0f) continue;
                for (int64_t r = 0; r < in_n; ++r) w2.v[(size_t) (p * in_n + r)] += va * tb[(size_t) (i * in_n + r)];
            }
    }
    e.lokr = {w1.shape[0], w1.shape[1], w2.shape[0], w2.shape[1]};
    e.rows = e.lokr.out_l * e.lokr.out_k;
    e.cols = e.lokr.in_m * e.lokr.in_n;
    e.rank = rank;
    e.shapes["lokr_w1"] = w1.shape;
    e.shapes["lokr_w2"] = w2.shape;
    e.extra["lokr_w1"] = std::move(w1.v);
    e.extra["lokr_w2"] = std::move(w2.v);
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
    if (e.kind == LoraKind::lokr) {
        make_lokr(e, g, path);
    }
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
    if (e.kind == LoraKind::diff) {
        // the weight's own delta: set_weight (the weight itself) before diff, and w_norm, ComfyUI's patch table's
        // order (comfy/lora.py writes each into the same key)
        const char* const which =
            role("set_weight") != nullptr ? "set_weight" : (role("diff") != nullptr ? "diff" : "w_norm");
        const RawTensor* t = role(which);
        if (t == nullptr)
            throw std::runtime_error(path_key_what(path, e.key,
                                                   "a bias diff (diff_b/b_norm) with no weight's delta: the image "
                                                   "engine applies no bias's diff yet"));
        if (t->shape.size() != 2)
            throw std::runtime_error(path_key_what(path, e.key,
                                                   std::string("a whole delta (") + which +
                                                       ") must be the weight's matrix"));
        e.rows = t->shape[0];
        e.cols = t->shape[1];
        e.replaces = std::strcmp(which, "set_weight") == 0;
        lora_to_f16(t->v, e.delta);
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
    // the other forms' tensors, as they are (the forms the engine learns to apply later read them); a LoKr's are
    // its two materialized matrices (above), not the file's factors, and a whole delta's is above
    for (const auto& [r, t] : g.by_role) {
        if (r == "lora_A" || r == "lora_B" || r == "alpha" || r == "dora_scale") continue;
        if (r.rfind("lokr_", 0) == 0) continue;
        if (r == "diff" || r == "set_weight" || r == "w_norm") continue;
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
        n += e.a.size() * 2 + e.b.size() * 2 + e.dora.size() * 4 + e.delta.size() * 2;
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

// tmp16[i] = f16(f32(x16[i]) * s * row_scale[i / N]): a whole delta's input, scaled into its product's FP16 rows
struct ScaleRows16 {
    const uint16_t* in;
    float s;
    const float* row_scale;
    int64_t N;
    uint16_t* out;
    void operator()(sycl::id<1> id) const {
        const size_t i = id[0];
        float v = strata::kernels::f32_from_f16(in[i]) * s;
        if (row_scale != nullptr) v *= row_scale[i / (size_t) N];
        out[i] = strata::kernels::f16_from_f32(v);
    }
};

// A LoKr's second step: y[t][row0 + c] += scale * row_scale[t] * sum_j w1[i][j] u[t][j][k], the weight's row
// src_row + c = i * out_k + k (the Kronecker product's rows are w1's, each w2's rows).
struct KronRows {
    const uint16_t* w1;   // [out_l][in_m]
    const float* u;       // [T][in_m][out_k], the first product's
    float* y;
    const float* row_scale;
    float scale;
    int64_t in_m, out_k, ldy, src_row, row0, n;
    void operator()(sycl::id<1> id) const {
        const int64_t at = (int64_t) id[0];
        const int64_t t = at / n, c = at - t * n;
        const int64_t row = src_row + c;
        const int64_t i = row / out_k, k = row - i * out_k;
        const uint16_t* w = w1 + i * in_m;
        const float* up = u + (t * in_m) * out_k + k;
        float acc = 0.0f;
        for (int64_t j = 0; j < in_m; ++j) acc += strata::kernels::f32_from_f16(w[j]) * up[j * out_k];
        float v = acc * scale;
        if (row_scale != nullptr) v *= row_scale[t];
        y[t * ldy + row0 + c] += v;
    }
};

}  // namespace lora_kernels

LoraDeviceEntry::~LoraDeviceEntry() { release(); }

void LoraDeviceEntry::release() {
    for (uint16_t** p : {&a_, &b_, &w1_, &w2_, &diff_}) {
        if (*p != nullptr) strata::gpu::free(*p);
        *p = nullptr;
    }
    lokr_ = {};
    rank_ = cols_ = rows_ = 0;
    bytes_ = 0;
}

bool LoraDeviceEntry::upload(const LoraEntry& e, void* stream, std::string& err, const float* row_coef) {
    release();
    kind_ = e.kind;
    if (e.extra.count("diff_b") != 0 || e.extra.count("b_norm") != 0) {
        // a bias's delta is another place than the weight's product (the DiTs add their biases there): not applied
        err = "the image engine: " + e.key +
              ": a bias diff (diff_b/b_norm): the engine applies a weight's delta, not a bias's yet";
        return false;
    }
    if (e.kind == LoraKind::diff) {
        if (row_coef != nullptr) {
            err = "the image engine: " + e.key + ": a DoRA (dora_scale) on a whole delta (diff/set_weight)";
            return false;
        }
        if (e.rows <= 0 || e.cols <= 0 || (int64_t) e.delta.size() != e.rows * e.cols) {
            err = "the image engine: " + e.key + ": the whole delta's tensors do not match its shape";
            return false;
        }
        rank_ = 0;
        cols_ = e.cols;
        rows_ = e.rows;
        if (!strata::gpu::alloc_device(&diff_, e.delta.size() * 2)) {
            err = "the image engine: the adapter " + e.key + " on the GPU: " + strata::gpu::last_error();
            release();
            return false;
        }
        strata::gpu::copy_async(diff_, e.delta.data(), e.delta.size() * 2, stream);
        if (!strata::gpu::stream_sync(stream)) {
            err = "the image engine: the adapter " + e.key + " to the GPU: " + strata::gpu::last_error();
            release();
            return false;
        }
        bytes_ = (uint64_t) e.delta.size() * 2;
        return true;
    }
    if (e.kind == LoraKind::lokr) {
        if (row_coef != nullptr) {   // a LoKr's rows are its Kronecker product's: its DoRA is not folded into them
            err = "the image engine: " + e.key + ": a LoKr's rows are its factors' Kronecker product's";
            return false;
        }
        const auto w1 = e.extra.find("lokr_w1");
        const auto w2 = e.extra.find("lokr_w2");
        if (e.lokr.empty() || w1 == e.extra.end() || w2 == e.extra.end() ||
            (int64_t) w1->second.size() != e.lokr.out_l * e.lokr.in_m ||
            (int64_t) w2->second.size() != e.lokr.out_k * e.lokr.in_n) {
            err = "the image engine: " + e.key + ": the LoKr's factors do not match its shape";
            return false;
        }
        std::vector<uint16_t> h1, h2;
        lora_to_f16(w1->second, h1);
        lora_to_f16(w2->second, h2);
        lokr_ = e.lokr;
        rank_ = e.rank;
        cols_ = e.cols;
        rows_ = e.rows;
        if (!strata::gpu::alloc_device(&w1_, h1.size() * 2) || !strata::gpu::alloc_device(&w2_, h2.size() * 2)) {
            err = "the image engine: the adapter " + e.key + " on the GPU: " + strata::gpu::last_error();
            release();
            return false;
        }
        strata::gpu::copy_async(w1_, h1.data(), h1.size() * 2, stream);
        strata::gpu::copy_async(w2_, h2.data(), h2.size() * 2, stream);
        if (!strata::gpu::stream_sync(stream)) {
            err = "the image engine: the adapter " + e.key + " to the GPU: " + strata::gpu::last_error();
            release();
            return false;
        }
        bytes_ = (uint64_t) (h1.size() + h2.size()) * 2;
        return true;
    }
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
    // a DoRA's c folded into B's rows (no copy where there is none, a rank-64 adapter's B being hundreds of MiB)
    std::vector<uint16_t> scaled;
    const uint16_t* b = e.b.data();
    if (row_coef != nullptr) {
        scaled = e.b;
        for (int64_t r = 0; r < e.rows; ++r)
            for (int64_t j = 0; j < e.rank; ++j) {
                uint16_t& h = scaled[(size_t) (r * e.rank + j)];
                h = strata::kernels::f16_from_f32(strata::kernels::f32_from_f16(h) * row_coef[r]);
            }
        b = scaled.data();
    }
    strata::gpu::copy_async(a_, e.a.data(), e.a.size() * 2, stream);
    strata::gpu::copy_async(b_, b, e.b.size() * 2, stream);
    if (!strata::gpu::stream_sync(stream)) {
        err = "the image engine: the adapter " + e.key + " to the GPU: " + strata::gpu::last_error();
        release();
        return false;
    }
    bytes_ = (uint64_t) (e.a.size() + e.b.size()) * 2;
    return true;
}

void LoraDeviceEntry::add(const uint16_t* x16, int64_t T, float* y, int64_t ldy, int64_t src_row, int64_t row0,
                          int64_t n, float scale, const float* row_scale, prefill::Gemm& gemm, float* tmp_f32,
                          uint16_t* tmp16, void* stream) const {
    if (T <= 0 || n <= 0 || scale == 0.0f) return;
    if (src_row < 0 || src_row + n > rows_) return;   // the caller keeps the rows within the entry
    if (kind_ == LoraKind::lokr) {
        add_lokr(x16, T, y, ldy, src_row, row0, n, scale, row_scale, gemm, tmp_f32, stream);
        return;
    }
    if (kind_ == LoraKind::diff) {
        add_diff(x16, T, y, ldy, src_row, row0, n, scale, row_scale, gemm, tmp16, stream);
        return;
    }
    if (rank_ <= 0) return;
    if (ldy <= 0) ldy = n;
    gemm.f16(x16, a_, tmp_f32, T, rank_, cols_);   // t = x . A^T, [T][rank]
    sycl::queue& q = strata::core::Runtime::get().stream(stream);
    q.parallel_for(sycl::range<1>((size_t) (T * rank_)),
                   lora_kernels::ScaleRows{tmp_f32, scale, row_scale, rank_, tmp16});
    gemm.f16(tmp16, b_ + src_row * rank_, y + row0, T, n, rank_, ldy, false, true);   // y[:, row0 ..] += t16 . B^T
}

void LoraDeviceEntry::add_diff(const uint16_t* x16, int64_t T, float* y, int64_t ldy, int64_t src_row, int64_t row0,
                               int64_t n, float scale, const float* row_scale, prefill::Gemm& gemm,
                               uint16_t* tmp16, void* stream) const {
    if (diff_ == nullptr || cols_ <= 0) return;
    if (ldy <= 0) ldy = n;
    // y[t][row0 + c] += scale * row_scale[t] * sum_k x[t][k] Delta[src_row + c][k]: the weight's own product
    const uint16_t* x = x16;
    if (scale != 1.0f || row_scale != nullptr) {   // the input scaled, as the plain form's first product does
        sycl::queue& q = strata::core::Runtime::get().stream(stream);
        q.parallel_for(sycl::range<1>((size_t) (T * cols_)),
                       lora_kernels::ScaleRows16{x16, scale, row_scale, cols_, tmp16});
        x = tmp16;
    }
    gemm.f16(x, diff_ + src_row * cols_, y + row0, T, n, cols_, ldy, false, true);
}

void LoraDeviceEntry::add_lokr(const uint16_t* x16, int64_t T, float* y, int64_t ldy, int64_t src_row, int64_t row0,
                               int64_t n, float scale, const float* row_scale, prefill::Gemm& gemm, float* tmp_f32,
                               void* stream) const {
    if (lokr_.empty()) return;
    if (ldy <= 0) ldy = n;
    // u[t][j][k] = sum_l x[t][j][l] w2[k][l]: the input's rows in in_m groups of in_n, w2's product of each
    gemm.f16(x16, w2_, tmp_f32, T * lokr_.in_m, lokr_.out_k, lokr_.in_n);
    sycl::queue& q = strata::core::Runtime::get().stream(stream);
    q.parallel_for(sycl::range<1>((size_t) (T * n)),
                   lora_kernels::KronRows{w1_, tmp_f32, y, row_scale, scale, lokr_.in_m, lokr_.out_k, ldy, src_row,
                                          row0, n});
}

uint64_t LoraLayer::bytes() const {
    uint64_t n = base_rows_ > 0 ? (uint64_t) base_rows_ * 4 : 0;
    for (const LoraPart& p : parts)
        if (p.entry) n += p.entry->bytes();
    return n;
}

LoraLayer::~LoraLayer() {
    if (base_ != nullptr) strata::gpu::free(base_);
}

bool LoraLayer::build_base(int64_t rows, void* stream, std::string& err) {
    if (base_built_) return true;
    base_built_ = true;
    bool any = false;
    for (const LoraPart& p : parts)
        if (!p.base.empty()) any = true;
    if (!any || rows <= 0) return true;
    std::vector<float> host((size_t) rows, 1.0f);
    std::vector<char> taken((size_t) rows, 0);
    for (const LoraPart& p : parts) {
        if (p.base.empty()) continue;
        if ((int64_t) p.base.size() != p.n || p.row0 < 0 || p.row0 + p.n > rows) {
            err = "the image engine: a DoRA's rows are not the weight's";
            return false;
        }
        for (int64_t j = 0; j < p.n; ++j)
            if (taken[(size_t) (p.row0 + j)] != 0) {
                err = "the image engine: two adapters' base factors on the weight's rows " + std::to_string(p.row0) +
                      " .. " + std::to_string(p.row0 + p.n) + " (each row takes one)";
                return false;
            }
        std::copy(p.base.begin(), p.base.end(), host.begin() + p.row0);
        std::fill(taken.begin() + p.row0, taken.begin() + p.row0 + p.n, 1);
    }
    if (!strata::gpu::alloc_device(&base_, (size_t) rows * 4)) {
        err = std::string("the image engine: a DoRA's row factors on the GPU: ") + strata::gpu::last_error();
        return false;
    }
    strata::gpu::copy_async(base_, host.data(), host.size() * 4, stream);
    if (!strata::gpu::stream_sync(stream)) {
        err = std::string("the image engine: a DoRA's row factors to the GPU: ") + strata::gpu::last_error();
        return false;
    }
    base_rows_ = rows;
    return true;
}

void LoraLayer::add(const uint16_t* x16, int64_t T, float* y, int64_t ldy, int64_t w_row0, int64_t w_rows,
                    const float* row_scale, prefill::Gemm& gemm, float* tmp_f32, uint16_t* tmp16,
                    void* stream) const {
    for (const LoraPart& p : parts) {
        // the part's rows the product wrote: [a, b) of the weight's rows
        const int64_t a = std::max(p.row0, w_row0), b = std::min(p.row0 + p.n, w_row0 + w_rows);
        if (b <= a) continue;
        p.entry->add(x16, T, y, ldy, p.src_row + (a - p.row0), a - w_row0, b - a, p.scale, row_scale, gemm, tmp_f32,
                     tmp16, stream);
    }
}

int64_t lora_scratch_cols(const LoraEntry& e) {
    if (e.kind == LoraKind::lokr) return e.lokr.empty() ? 0 : e.lokr.in_m * e.lokr.out_k;
    if (e.kind == LoraKind::diff) return e.cols;   // a scaled input's rows: the weight's columns
    return e.rank;
}

uint64_t lora_gpu_bytes(const std::vector<LoraUse>& loras, int64_t T) {
    uint64_t bytes = 0;
    int64_t cols = 0;
    for (const LoraUse& u : loras) {
        if (!u.file) continue;
        for (const LoraEntry& e : u.file->entries()) {
            if (e.kind == LoraKind::lora)
                bytes += 2 * (uint64_t) e.rank * (uint64_t) (e.rows + e.cols);   // A and B, FP16
            else if (e.kind == LoraKind::lokr && !e.lokr.empty())
                bytes += 2 * (uint64_t) (e.lokr.out_l * e.lokr.in_m + e.lokr.out_k * e.lokr.in_n);   // w1 and w2
            else if (e.kind == LoraKind::diff && e.rows > 0 && e.cols > 0)
                bytes += 2 * (uint64_t) e.rows * (uint64_t) e.cols;   // the whole delta
            else
                continue;
            if (!e.dora.empty()) bytes += (uint64_t) e.rows * 4;   // the base product's rows' factors
            cols = std::max(cols, lora_scratch_cols(e));
        }
    }
    return bytes + (uint64_t) std::max<int64_t>(T, 0) * (uint64_t) cols * 6;
}

bool lora_dora_rows(const LoraEntry& e, int64_t src_row, int64_t row0, int64_t n, const std::vector<float>& norms,
                    float strength, std::vector<float>& coef, std::vector<float>& base, std::string& err) {
    if (e.kind != LoraKind::lora) {
        err = "a DoRA (dora_scale) on a ";
        err += lora_kind_name(e.kind);
        err += " adapter: the engine's DoRA is the plain form's (its rows are the weight's); this form's are not";
        return false;
    }
    if (!e.dora_over_rows) {
        err = "a DoRA (dora_scale) over the weight's columns: the engine's DoRA scales its output's rows, as ComfyUI's "
              "weight_decompose does where the scale is the output axis's";
        return false;
    }
    if (src_row < 0 || row0 < 0 || n < 0 || (int64_t) e.dora.size() < src_row + n ||
        (int64_t) norms.size() < row0 + n) {
        err = "a DoRA (dora_scale) whose rows are not the weight's";
        return false;
    }
    coef.resize((size_t) n);
    base.resize((size_t) n);
    for (int64_t j = 0; j < n; ++j) {
        const float c = e.dora[(size_t) (src_row + j)] / (norms[(size_t) (row0 + j)] + kLoraDoraEps);
        coef[(size_t) j] = c;
        base[(size_t) j] = 1.0f + strength * (c - 1.0f);
    }
    return true;
}

const LoraPart* lora_dora_overlap(const LoraLayer* layer, int64_t row0, int64_t n, bool dora) {
    if (layer == nullptr) return nullptr;
    for (const LoraPart& p : layer->parts) {
        if (!dora && !p.dora) continue;   // neither is a DoRA's: the two add up, whatever their rows
        if (row0 < p.row0 + p.n && p.row0 < row0 + n) return &p;
    }
    return nullptr;
}

}  // namespace strata::image