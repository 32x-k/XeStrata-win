// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/moss_dit.cpp - see include/strata/image/moss_dit.hpp.

#include "strata/image/moss_dit.hpp"

#include "dit_rows.hpp"

#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/image/mapped_file.hpp"
#include "strata/image/pipeline.hpp"
#include "strata/image/qi_kernels.hpp"
#include "strata/image/safetensors.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <stdexcept>

namespace strata::image {

namespace gpu = strata::gpu;

namespace moss_dit_detail {

using namespace dit_rows;

// kTypeW8: int8 rows rotated as qi_rows_i8 rotates them, then a scale a row (FP32), for prefill::Gemm::i8
constexpr int kTypeF16 = 1, kTypeQ8_0 = 8, kTypeW8 = -8;
constexpr float kEps = 1e-6f;

template<class T> T* alloc(std::vector<void*>& owned, int64_t n, bool zero = false) {
    T* p = nullptr;
    if (!gpu::alloc_device(&p, (size_t) std::max<int64_t>(n, 1) * sizeof(T)))
        throw std::runtime_error(std::string("image engine (MOSS-Image DiT): GPU memory: ") + gpu::last_error());
    if (zero && !gpu::memset(p, 0, (size_t) std::max<int64_t>(n, 1) * sizeof(T)))
        throw std::runtime_error(std::string("image engine (MOSS-Image DiT): memset: ") + gpu::last_error());
    owned.push_back(p);
    return p;
}

void release(std::vector<void*>& owned, void* p) {
    if (p == nullptr) return;
    owned.erase(std::remove(owned.begin(), owned.end(), p), owned.end());
    gpu::free(p);
}

// RMSNorm(x[t]) * w (eps 1e-6), then * (1 + scale) + shift where the time gives them (null: not)
struct RmsMod {
    const float *x, *w, *shift, *scale;
    int64_t C;
    template<class R> Stats stats(int64_t t, int64_t lane, int64_t step, R reduce) const {
        const float* r = x + t * C;
        float s = 0.f;
        for (int64_t c = lane; c < C; c += step) s += r[c] * r[c];
        return {0.f, 1.f / sycl::sqrt(reduce(s) / (float) C + kEps)};
    }
    float value(int64_t t, int64_t c, const Stats& st) const {
        const float y = x[t * C + c] * st.inv * w[c];
        return scale != nullptr ? y * (1.f + scale[c]) + shift[c] : y;
    }
};

// x as it is (the text encoder's states, for their FP16 rows' scales)
struct Plain {
    const float* x;
    int64_t C;
    template<class R> Stats stats(int64_t, int64_t, int64_t, R) const { return {}; }
    float value(int64_t t, int64_t c, const Stats&) const { return x[t * C + c]; }
};

// SiTU of the up_gate product's sums g [T][2F] (the hidden values, then the gates): (4 tanh(g / 4) sigmoid(g)) times
// (25 tanh(h / 25)), F values a row
struct Situ {
    MfOut g;
    int64_t F;
    template<class R> Stats stats(int64_t, int64_t, int64_t, R) const { return {}; }
    float value(int64_t t, int64_t c, const Stats&) const {
        const float h = g.at(t, c), gt = g.at(t, c + F);
        return 4.f * sycl::tanh(gt * 0.25f) / (1.f + sycl::exp(-gt)) * (25.f * sycl::tanh(h * 0.04f));
    }
    // below 4 x 25 in magnitude: FP16 takes the rows as they are (their scales 1)
    float bound(int64_t, int64_t) const { return 0.f; }
};

// RowsI8Of's ops (a W8A8 product's input without the FP16 rows): RmsMod's rows, after x += gate * add where add is a
// deferred product (gate_add, its sums written back to x); Situ's; the attention's output times its heads' scales
struct RmsI8 {
    static constexpr bool kRms = true, kKeep = true;
    float* x;
    MfOut add;
    const float *gate, *w, *shift, *scale;
    int64_t C;
    float raw(int64_t t, int64_t c) const {
        const float v = x[t * C + c];
        return add.C > 0 ? v + gate[c] * add.at(t, c) : v;
    }
    void keep(int64_t t, int64_t c, float v) const {
        if (add.C > 0) x[t * C + c] = v;
    }
    float inv(float ss) const { return 1.f / sycl::sqrt(ss / (float) C + kEps); }
    float out(int64_t, int64_t c, float v, float inv) const {
        const float y = v * inv * w[c];
        return scale != nullptr ? y * (1.f + scale[c]) + shift[c] : y;
    }
};

struct SituI8 {
    static constexpr bool kRms = false, kKeep = false;
    Situ s;
    float raw(int64_t t, int64_t c) const { return s.value(t, c, Stats{}); }
    void keep(int64_t, int64_t, float) const {}
    float inv(float) const { return 1.f; }
    float out(int64_t, int64_t, float v, float) const { return v; }
};

struct HeadsI8 {
    static constexpr bool kRms = false, kKeep = false;
    Heads h;
    float raw(int64_t t, int64_t c) const { return h.value(t, c, Stats{}); }
    void keep(int64_t, int64_t, float) const {}
    float inv(float) const { return 1.f; }
    float out(int64_t, int64_t, float v, float) const { return v; }
};

// For each row and head: q and k RMS-normed over the head (weights wq, wk; eps 1e-6) and rotated in pairs (j,
// j + head_dim / 2) by cos[t][j], sin[t][j] (tables [T][head_dim / 2]), v divided by its head's scale; FP16 out
struct QkRope {
    MfOut q, k, v;
    const float *wq, *wk, *cos, *sin, *hs;
    int64_t T, heads, hd;
    uint16_t *q16, *k16, *v16;
    void operator()(sycl::nd_item<1> it) const {
        const auto sg = it.get_sub_group();
        const int64_t i = static_cast<int64_t>(it.get_group_linear_id()) * kRows + sg.get_group_linear_id();
        if (i >= T * heads) return;
        const int64_t t = i / heads, h = i - t * heads, c0 = h * hd, o = t * heads * hd + c0, half = hd / 2;
        const auto lane = static_cast<int64_t>(sg.get_local_linear_id());
        for (int which = 0; which < 2; ++which) {
            const MfOut& src = which == 0 ? q : k;
            const float* w = which == 0 ? wq : wk;
            uint16_t* dst = (which == 0 ? q16 : k16) + o;
            float ss = 0.f;
            for (int64_t j = lane; j < half; j += kSG) {
                const float a = src.at(t, c0 + j), b = src.at(t, c0 + half + j);
                ss += a * a + b * b;
            }
            const float r = 1.f / sycl::sqrt(sycl::reduce_over_group(sg, ss, sycl::plus<float>()) / (float) hd + kEps);
            for (int64_t j = lane; j < half; j += kSG) {
                const float a = src.at(t, c0 + j) * r * w[j], b = src.at(t, c0 + half + j) * r * w[half + j];
                const float c = cos[t * half + j], sn = sin[t * half + j];
                dst[j] = f16_from_f32(a * c - b * sn);
                dst[half + j] = f16_from_f32(b * c + a * sn);
            }
        }
        const float inv = 1.f / hs[h];
        for (int64_t d = lane; d < hd; d += kSG) v16[o + d] = f16_from_f32(v.at(t, c0 + d) * inv);
    }
    auto get(syclex::properties_tag) const { return syclex::properties{syclex::sub_group_size<kSG>}; }
};

void qk_rope16(const MfOut& q, const MfOut& k, const MfOut& v, const float* wq, const float* wk, const float* cos,
               const float* sin, const float* hs, int64_t T, int64_t heads, int64_t hd, uint16_t* q16, uint16_t* k16,
               uint16_t* v16, void* s) {
    if (T <= 0) return;
    sycl::queue& qu = q_of(s);
    if (has_sg32(qu)) {
        qu.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) ((T * heads + kRows - 1) / kRows * kRows * kSG)),
                                          sycl::range<1>((size_t) (kRows * kSG))),
                        QkRope{q, k, v, wq, wk, cos, sin, hs, T, heads, hd, q16, k16, v16});
        return;
    }
    const int64_t C = heads * hd, half = hd / 2;
    qu.parallel_for(sycl::range<1>((size_t) (T * heads)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t t = i / heads, h = i - t * heads, c0 = h * hd, o = t * C + c0;
        for (int which = 0; which < 2; ++which) {
            const MfOut& src = which == 0 ? q : k;
            const float* w = which == 0 ? wq : wk;
            uint16_t* dst = (which == 0 ? q16 : k16) + o;
            float ss = 0.f;
            for (int64_t d = 0; d < hd; ++d) ss += src.at(t, c0 + d) * src.at(t, c0 + d);
            const float r = 1.f / sycl::sqrt(ss / (float) hd + kEps);
            for (int64_t j = 0; j < half; ++j) {
                const float a = src.at(t, c0 + j) * r * w[j], b = src.at(t, c0 + half + j) * r * w[half + j];
                const float c = cos[t * half + j], sn = sin[t * half + j];
                dst[j] = f16_from_f32(a * c - b * sn);
                dst[half + j] = f16_from_f32(b * c + a * sn);
            }
        }
        const float inv = 1.f / hs[h];
        for (int64_t d = 0; d < hd; ++d) v16[o + d] = f16_from_f32(v.at(t, c0 + d) * inv);
    });
}

// p's n bf16 values rewritten in place as FP16, saturated at FP16's largest
void bf16_to_f16_inplace(uint16_t* p, int64_t n, sycl::queue& q) {
    q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const float v = kernels::f32_from_bf16(p[id[0]]);
        p[id[0]] = f16_from_f32(sycl::clamp(v, -65504.f, 65504.f));
    });
}

}  // namespace moss_dit_detail

using namespace moss_dit_detail;

// the transformer's shards (a folder's *.safetensors, or one file), each mapped
struct MossDit::Shards {
    struct Shard {
        std::unique_ptr<SafeTensors> st;
        std::unique_ptr<MappedFile> map;
    };
    std::vector<Shard> s;
    explicit Shards(const std::string& path) {
        std::vector<std::string> files;
        if (std::filesystem::is_directory(path)) {
            for (const auto& e : std::filesystem::directory_iterator(path))
                if (e.path().extension() == ".safetensors") files.push_back(e.path().string());
            std::sort(files.begin(), files.end());
        } else {
            files.push_back(path);
        }
        if (files.empty()) throw std::runtime_error("image engine (MOSS-Image DiT): no .safetensors in " + path);
        for (const auto& f : files) s.push_back({std::make_unique<SafeTensors>(f), std::make_unique<MappedFile>(f)});
    }
    // the tensor and its shard, or null
    const SafeTensor* find(const std::string& name, const Shard** at = nullptr) const {
        for (const Shard& x : s)
            if (const SafeTensor* t = x.st->find(name); t != nullptr) {
                if (at != nullptr) *at = &x;
                return t;
            }
        return nullptr;
    }
    // a bf16 tensor: its bytes in the mapping
    const SafeTensor& bf16(const std::string& name, const uint8_t** data) const {
        const Shard* at = nullptr;
        const SafeTensor* t = find(name, &at);
        if (t == nullptr) throw std::runtime_error("image engine (MOSS-Image DiT): no " + name);
        if (t->dtype != "BF16") throw std::runtime_error("image engine (MOSS-Image DiT): " + name + " is not bf16");
        if (data != nullptr) *data = at->map->data() + at->st->data_base() + t->begin;
        return *t;
    }
    std::vector<float> f32(const std::string& name) const {
        const Shard* at = nullptr;
        if (find(name, &at) == nullptr) throw std::runtime_error("image engine (MOSS-Image DiT): no " + name);
        return at->st->f32(name);
    }
    void drop_cache(const std::string& name) const {
        const Shard* at = nullptr;
        if (const SafeTensor* t = find(name, &at); t != nullptr)
            at->map->drop_cache(at->st->data_base() + t->begin, t->end - t->begin);
    }
};

MossDit::MossDit() = default;

MossDit::~MossDit() {
    blocks_stream_.drop();
    for (void* p : allocs_) gpu::free(p);
    for (void* p : text_allocs_) gpu::free(p);
}

bool MossDit::open(const std::string& path, void* stream, std::string& err, bool dry, bool q8,
                   std::vector<std::vector<uint8_t>>* reuse) {
    try {
        s_ = stream;
        dry_ = dry;
        q8_ = q8;
        w8_ = q8 && std::getenv("STRATA_NO_W8A8") == nullptr && prefill::Gemm::i8_usable(stream);
        files_ = std::make_unique<Shards>(path);
        const Shards& f = *files_;
        if (std::filesystem::is_directory(path)) {   // config.json's rope_base, where it differs from 10000
            std::ifstream cf(std::filesystem::path(path) / "config.json");
            const std::string cfg((std::istreambuf_iterator<char>(cf)), std::istreambuf_iterator<char>());
            if (const size_t at = cfg.find("\"rope_base\""); at != std::string::npos)
                rope_base_ = std::stod(cfg.substr(cfg.find(':', at) + 1));
        }
        vin_ = dense({"video_embedder"}, true, &b_vin_, allocs_, weight_bytes_);
        in_ch_ = vin_.k;
        C_ = vin_.n;
        t1_ = dense({"time_embedding.0"}, true, &b_t1_, allocs_, weight_bytes_);
        freq_ = t1_.k;
        t2_ = dense({"time_embedding.2"}, true, &b_t2_, allocs_, weight_bytes_);
        smod_ = dense({"shared_block_modulation"}, true, &b_smod_, allocs_, weight_bytes_);
        hmod_ = dense({"video_head_modulation"}, true, &b_hmod_, allocs_, weight_bytes_);
        fout_ = dense({"final_linear_video"}, false, nullptr, allocs_, weight_bytes_);
        fnorm_ = vec("final_norm_video.weight", allocs_, weight_bytes_);
        hd_ = f.bf16("layers.0.q_norm.weight", nullptr).shape[0];
        heads_ = C_ / hd_;
        if (C_ % hd_ != 0 || hd_ % 8 != 0 || smod_.n != 6 * C_ || hmod_.n != 2 * C_)
            throw std::runtime_error("image engine (MOSS-Image DiT): an unexpected shape");
        load_text();
        int64_t& scratch = scratch_;
        for (int i = 0;; ++i) {
            const std::string p = "layers." + std::to_string(i) + ".";
            if (f.find(p + "proj.q.weight") == nullptr) break;
            Block b;
            const std::vector<std::pair<W*, std::string>> items = {
                {&b.q, "proj.q"}, {&b.k, "proj.k"}, {&b.v, "proj.v"}, {&b.o, "proj.o"},
                {&b.up_gate, "proj.ffn_up_gate"}, {&b.down, "proj.ffn_down"}};
            const std::vector<std::pair<uint64_t*, std::string>> norms = {
                {&b.n1, "norm1.weight"}, {&b.n2, "norm2.weight"}, {&b.nq, "q_norm.weight"}, {&b.nk, "k_norm.weight"}};
            // the block's host bytes: the quantized weights (q8), then every FP32 vector
            uint64_t qbytes = 0, vbytes = 0;
            for (const auto& [w, name] : items) {
                const SafeTensor& t = f.bf16(p + name + ".weight", nullptr);
                w->n = t.shape[0];
                w->k = t.shape[1];
                if (w->k % 32 != 0) throw std::runtime_error("image engine (MOSS-Image DiT): columns not 32s");
                const bool w8 = w8_ && w->k % kQiRotate == 0;
                w->type = w8 ? kTypeW8 : q8_ ? kTypeQ8_0 : kTypeF16;   // FP16: the file's bf16, rewritten on the GPU
                w->row_bytes = w8 ? (uint64_t) w->k : q8_ ? (uint64_t) (w->k / 32 * 34) : (uint64_t) w->k * 2;
                if (q8_) qbytes += w->row_bytes * (uint64_t) w->n + (w8 ? (uint64_t) w->n * 4 : 0);
                if (w->type == kTypeQ8_0) scratch = std::max(scratch, w->n * w->k);
            }
            for (const auto& [off, name] : norms) vbytes += (uint64_t) f.bf16(p + name, nullptr).elements() * 4;
            std::vector<LayerStream::Span> spans;
            std::vector<uint8_t> host;
            // an earlier open's bytes of this block, in the same form (their size says it: W8A8's rows and Q8_0's
            // differ), else the file's quantized again
            const bool reused = !dry_ && q8_ && reuse != nullptr && (size_t) i < reuse->size() &&
                                (*reuse)[(size_t) i].size() == qbytes + vbytes;
            if (reused) host = std::move((*reuse)[(size_t) i]);
            else if (!dry_) host.resize((size_t) (qbytes + vbytes));
            uint64_t at = 0;
            std::vector<size_t> wspan, nspan;
            for (const auto& [w, name] : items) {
                const uint8_t* d = nullptr;
                f.bf16(p + name + ".weight", &d);
                const uint64_t bytes = w->row_bytes * (uint64_t) w->n + (w->type == kTypeW8 ? (uint64_t) w->n * 4 : 0);
                if (q8_) {
                    if (!dry_ && !reused) {
                        quantize_matrix(reinterpret_cast<const uint16_t*>(d), w->n, w->k, host.data() + at,
                                        w->type == kTypeW8);
                        f.drop_cache(p + name + ".weight");   // read once, not kept in the page cache
                    }
                    spans.push_back({host.data() + at, bytes, 0});
                    at += bytes;
                } else {
                    spans.push_back({d, bytes, 0});
                }
                wspan.push_back(spans.size() - 1);
            }
            for (const auto& [off, name] : norms) {
                const int64_t n = f.bf16(p + name, nullptr).elements();
                if (!dry_ && !reused) {
                    const auto v = f.f32(p + name);
                    std::memcpy(host.data() + at, v.data(), v.size() * 4);
                }
                spans.push_back({host.data() + at, (uint64_t) n * 4, 0});
                at += (uint64_t) n * 4;
                nspan.push_back(spans.size() - 1);
            }
            const int layer = blocks_stream_.add(spans);
            for (size_t j = 0; j < items.size(); ++j)
                items[j].first->d = reinterpret_cast<const void*>(blocks_stream_.offset(layer, (int) wspan[j]));
            for (size_t j = 0; j < norms.size(); ++j) *norms[j].first = blocks_stream_.offset(layer, (int) nspan[j]);
            host_.push_back(std::move(host));
            blocks_.push_back(b);
        }
        if (blocks_.empty()) throw std::runtime_error("image engine (MOSS-Image DiT): no blocks");
        F_ = blocks_[0].down.k;
        if (blocks_[0].up_gate.n != 2 * F_)
            throw std::runtime_error("image engine (MOSS-Image DiT): ffn_up_gate is not twice ffn_down's width");
        if (!q8_)   // each block's weights, as they reach the GPU (resident, or a streamed slot): bf16 to FP16
            blocks_stream_.set_prepare([this](int i, uint8_t* d) {
                sycl::queue& q = q_of(s_);
                const Block& b = blocks_[(size_t) i];
                for (const W* w : {&b.q, &b.k, &b.v, &b.o, &b.up_gate, &b.down})
                    bf16_to_f16_inplace(reinterpret_cast<uint16_t*>(d + reinterpret_cast<uintptr_t>(w->d)),
                                        w->n * w->k, q);
            });
        if (dry_) return true;
        if (!gemm_.init(s_, scratch, err)) return false;
        blocks_stream_.configure(blocks_stream_.total(), s_);   // all resident until a budget is set
        t16_ = alloc<uint16_t>(allocs_, freq_ + 2 * C_);
        temb_ = alloc<float>(allocs_, C_);
        mod_ = alloc<float>(allocs_, 6 * C_);
        head_ = alloc<float>(allocs_, 2 * C_);
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

// a part outside the blocks: the rows of the named weights one after the other as FP16, their biases FP32 (in `owned`,
// their bytes added to `bytes`)
MossDit::W MossDit::dense(const std::vector<std::string>& names, bool biased, const float** bias,
                          std::vector<void*>& owned, uint64_t& bytes) {
    const Shards& f = *files_;
    W w;
    w.type = kTypeF16;
    std::vector<const uint8_t*> src;
    for (const auto& name : names) {
        const uint8_t* d = nullptr;
        const SafeTensor& t = f.bf16(name + ".weight", &d);
        if (w.k != 0 && t.shape[1] != w.k)
            throw std::runtime_error("image engine (MOSS-Image DiT): " + name + ": columns differ");
        w.k = t.shape[1];
        w.n += t.shape[0];
        src.push_back(d);
    }
    w.row_bytes = (uint64_t) w.k * 2;
    bytes += (uint64_t) (w.n * w.k) * 2 + (biased ? (uint64_t) w.n * 4 : 0);
    if (dry_) return w;
    std::vector<uint16_t> h((size_t) (w.n * w.k));
    std::vector<float> b;
    size_t at = 0;
    for (size_t j = 0; j < names.size(); ++j) {
        const int64_t n = f.bf16(names[j] + ".weight", nullptr).shape[0];
        const auto* s16 = reinterpret_cast<const uint16_t*>(src[j]);
        for (int64_t i = 0; i < n * w.k; ++i) h[at++] = f16_from_f32(bf16_value(s16[i]));
        if (biased) {
            const auto v = f.f32(names[j] + ".bias");
            b.insert(b.end(), v.begin(), v.end());
        }
    }
    uint16_t* d = alloc<uint16_t>(owned, (int64_t) h.size());
    gpu::copy(d, h.data(), h.size() * 2);
    w.d = d;
    if (biased) {
        float* db = alloc<float>(owned, (int64_t) b.size());
        gpu::copy(db, b.data(), b.size() * 4);
        *bias = db;
    }
    return w;
}

const float* MossDit::vec(const std::string& name, std::vector<void*>& owned, uint64_t& bytes) {
    const int64_t n = files_->bf16(name, nullptr).elements();
    bytes += (uint64_t) n * 4;
    if (dry_) return nullptr;
    const auto v = files_->f32(name);
    float* d = alloc<float>(owned, n);
    gpu::copy(d, v.data(), v.size() * 4);
    return d;
}

// the text embedder: its input projection and its blocks (their q, k and v rows of one qkv), FP16 on the GPU
void MossDit::load_text() {
    text_bytes_ = 0;
    auto rows = [](const W& w, int64_t r0, int64_t n) {
        W r = w;
        r.n = n;
        if (r.d != nullptr) r.d = static_cast<const uint16_t*>(w.d) + r0 * w.k;
        return r;
    };
    tproj_ = dense({"text_embedder.proj"}, true, &b_tproj_, text_allocs_, text_bytes_);
    txt_dim_ = tproj_.k;
    tblocks_.clear();
    for (int i = 0;; ++i) {
        const std::string p = "text_embedder.blocks." + std::to_string(i) + ".";
        if (files_->find(p + "qkv.weight") == nullptr) break;
        TextBlock b;
        const float* bqkv = nullptr;
        const W qkv = dense({p + "qkv"}, true, &bqkv, text_allocs_, text_bytes_);
        b.q = rows(qkv, 0, C_);
        b.k = rows(qkv, C_, C_);
        b.v = rows(qkv, 2 * C_, C_);
        if (!dry_) {
            b.bq = bqkv;
            b.bk = bqkv + C_;
            b.bv = bqkv + 2 * C_;
        }
        b.o = dense({p + "proj"}, true, &b.bo, text_allocs_, text_bytes_);
        b.up_gate = dense({p + "ffn_up", p + "ffn_gate"}, true, &b.bug, text_allocs_, text_bytes_);
        b.down = dense({p + "ffn_down"}, true, &b.bdown, text_allocs_, text_bytes_);
        b.n1 = vec(p + "norm1.weight", text_allocs_, text_bytes_);
        b.n2 = vec(p + "norm2.weight", text_allocs_, text_bytes_);
        b.nq = vec(p + "q_norm.weight", text_allocs_, text_bytes_);
        b.nk = vec(p + "k_norm.weight", text_allocs_, text_bytes_);
        tblocks_.push_back(b);
    }
    text_loaded_ = !dry_;
}

void MossDit::drop_text() {
    if (!text_loaded_) return;
    gpu::stream_sync(s_);
    for (void* p : text_allocs_) gpu::free(p);
    text_allocs_.clear();
    text_loaded_ = false;
}

std::pair<double, double> MossDit::time_forms(int64_t rows) const {
    const int64_t T = rows, K = C_, N = 2 * F_;
    if (T <= 0 || K <= 0 || N <= 0) return {0.0, 0.0};
    std::vector<void*> held;
    auto get = [&](size_t bytes) -> void* {
        void* p = nullptr;
        if (!gpu::alloc_device(&p, bytes)) return nullptr;
        held.push_back(p);
        gpu::memset_async(p, 0, bytes, s_);
        return p;
    };
    auto best = [&](const std::function<void()>& run) {
        double t = 1e30;
        for (int i = 0; i < 4; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            run();
            if (!gpu::stream_sync(s_)) return 0.0;
            if (i > 0) t = std::min(t, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        }
        return t;
    };
    double f16 = 0.0, w8 = 0.0;
    try {
        auto* x16 = static_cast<uint16_t*>(get((size_t) (T * K) * 2));
        auto* w16 = static_cast<uint16_t*>(get((size_t) (N * K) * 2));
        auto* y = static_cast<float*>(get((size_t) (T * N) * 4));
        prefill::Gemm g;
        std::string err;
        if (x16 != nullptr && w16 != nullptr && y != nullptr && g.init(s_, 0, err))
            f16 = best([&] { g.f16(x16, w16, y, T, N, K); });
        if (x16 != nullptr && y != nullptr && prefill::Gemm::i8_usable(s_)) {
            auto* x8 = static_cast<int8_t*>(get((size_t) (T * K)));
            auto* w8p = static_cast<int8_t*>(get((size_t) (N * K)));
            auto* sx = static_cast<float*>(get((size_t) T * 4));
            auto* sw = static_cast<float*>(get((size_t) N * 4));
            if (x8 != nullptr && w8p != nullptr && sx != nullptr && sw != nullptr)
                w8 = best([&] {
                    qi_rows_i8(x16, false, T, K, x8, sx, nullptr, s_);
                    prefill::Gemm::i8(x8, w8p, reinterpret_cast<int32_t*>(y), T, N, K, N, s_);
                    qi_i8_scale(y, T, N, N, sx, sw, nullptr, s_);
                });
        }
    } catch (const std::exception& e) {   // a form the library could not run: the 8-bit one is not chosen
        std::fprintf(stderr, "image engine: timing MOSS-Image's DiT in 16 and 8 bits: %s (16 bits kept)\n", e.what());
        w8 = 0.0;
    }
    gpu::stream_sync(s_);
    for (void* p : held) gpu::free(p);
    return {f16, w8};
}

MfOut MossDit::product(const uint16_t* x16, const float* rs, const W& w, const uint8_t* base, float* y, int64_t T,
                       const float* bias, bool same_input, bool defer) {
    MfOut out;
    out.f = y;
    out.C = w.n;
    if (T <= 0) return out;
    const auto* d = base != nullptr ? base + reinterpret_cast<uintptr_t>(w.d) : static_cast<const uint8_t*>(w.d);
    if (w.type == kTypeW8) {
        // the activations rotated and quantized as the weights (once for the products of one input), their scales
        // times the rows' (rs), the int32 sums into y and scaled there, or by the reader (defer)
        float* scales = reinterpret_cast<float*>(xq8_ + xq8_rows_ * xq8_cols_);
        if (!same_input) xs_next_ ^= 1;
        float* sx = scales + xs_next_ * xq8_rows_;
        if (!same_input) qi_rows_i8(x16, false, T, w.k, xq8_, sx, rs, s_);
        prefill::Gemm::i8(xq8_, reinterpret_cast<const int8_t*>(d), reinterpret_cast<int32_t*>(y), T, w.n, w.k, w.n,
                          s_);
        const auto* sw = reinterpret_cast<const float*>(d + w.n * w.k);
        if (defer) {
            out.f = nullptr;
            out.i = reinterpret_cast<const int32_t*>(y);
            out.sx = sx;
            out.sw = sw;
            out.b = bias;
            return out;
        }
        qi_i8_scale(y, T, w.n, w.n, sx, sw, bias, s_);
        return out;
    }
    if (w.type == kTypeF16) gemm_.f16(x16, reinterpret_cast<const uint16_t*>(d), y, T, w.n, w.k);
    else gemm_.native(x16, w.type, d, y, T, w.n, w.k);
    if (defer) {
        out.rs = rs;
        out.b = bias;
        return out;
    }
    if (rs != nullptr || bias != nullptr) rows_back(y, rs, bias, T, w.n, s_);
    return out;
}

uint64_t MossDit::activation_bytes(int64_t hw, int n, int64_t max_text, int64_t mlp_rows) const {
    const int64_t R = n * (hw + max_text), TP = qi_attn_rows(hw + max_text);
    const int64_t band = mlp_rows > 0 ? std::min(R, mlp_rows) : R;
    return (uint64_t) R * C_ * 4 * 5 +                          // x, q, k, v, a
           (uint64_t) band * 2 * F_ * 4 +                       // the MLP's up_gate sums
           (uint64_t) std::max(R * C_, band * F_) * 2 +         // the products' FP16 input
           (uint64_t) n * TP * C_ * 2 * 3 + (uint64_t) TP * C_ * 2 +   // q, k, v of each context, the output
           (uint64_t) R * C_ * 2 +                              // the attention's output, a row's scale
           (uint64_t) n * (hw + max_text) * hd_ * 4 +           // the rotary tables
           (uint64_t) TP * C_ * 4 +                             // the plain attention's scratch
           (w8_ ? (uint64_t) std::max(R, band) * std::max(C_, F_) + (uint64_t) std::max(R, band) * 8 + C_ * 4 : 0);
}

void MossDit::ensure(int64_t hw, int n, int64_t max_text) {
    const int64_t cap_rows = cap_n_ * (cap_hw_ + cap_text_);
    if (hw <= cap_hw_ && n <= cap_n_ && max_text <= cap_text_ &&
        mlp_rows_ == (mlp_cap_ > 0 ? std::min(cap_rows, mlp_cap_) : cap_rows))
        return;
    hw = std::max(hw, cap_hw_);
    n = std::max<int>(n, (int) cap_n_);
    max_text = std::max(max_text, cap_text_);
    drop_activations();
    const int64_t R = n * (hw + max_text), TP = qi_attn_rows(hw + max_text);
    mlp_rows_ = mlp_cap_ > 0 ? std::min(R, mlp_cap_) : R;
    x_ = alloc<float>(allocs_, R * C_);
    q_ = alloc<float>(allocs_, R * C_);
    k_ = alloc<float>(allocs_, R * C_);
    v_ = alloc<float>(allocs_, R * C_);
    a_ = alloc<float>(allocs_, R * C_);
    gu_ = alloc<float>(allocs_, mlp_rows_ * 2 * F_);
    h16_ = alloc<uint16_t>(allocs_, std::max(R * C_, mlp_rows_ * F_));
    rs_ = alloc<float>(allocs_, std::max(R, mlp_rows_));
    rs2_ = alloc<float>(allocs_, std::max(R, mlp_rows_));
    rs_o_ = alloc<float>(allocs_, R);
    hs_ = alloc<float>(allocs_, heads_);
    vpart_ = alloc<float>(allocs_, (hw + max_text) * heads_);
    if (w8_) {   // W8A8's int8 input: the most rows of the widest input, then two arrays of a scale a row
        xq8_rows_ = std::max(R, mlp_rows_);
        xq8_cols_ = std::max(C_, F_);
        xq8_ = alloc<int8_t>(allocs_, xq8_rows_ * xq8_cols_ + 2 * xq8_rows_ * 4);
        pend_sw_ = alloc<float>(allocs_, C_);
    }
    jrows_ = TP;
    jq16_ = alloc<uint16_t>(allocs_, n * TP * C_, true);
    jk16_ = alloc<uint16_t>(allocs_, n * TP * C_, true);
    jv16_ = alloc<uint16_t>(allocs_, n * TP * C_, true);
    jo16_ = alloc<uint16_t>(allocs_, TP * C_);
    o16_ = alloc<uint16_t>(allocs_, R * C_);
    if (std::strcmp(qi_attention_path(), "plain (FP32)") == 0) attn_scratch_ = alloc<float>(allocs_, TP * C_);
    if (keep0_) b0_ = alloc<float>(allocs_, (hw + max_text) * C_);
    cap_hw_ = hw;
    cap_n_ = n;
    cap_text_ = max_text;
    dirty_[0] = dirty_[1] = 0;
}

void MossDit::drop_activations() {
    gpu::stream_sync(s_);
    for (void* p : {(void*) x_, (void*) q_, (void*) k_, (void*) v_, (void*) a_, (void*) gu_, (void*) h16_, (void*) rs_,
                    (void*) rs2_, (void*) rs_o_, (void*) hs_, (void*) vpart_, (void*) xq8_, (void*) pend_sw_,
                    (void*) jq16_, (void*) jk16_, (void*) jv16_, (void*) jo16_, (void*) o16_, (void*) attn_scratch_,
                    (void*) b0_})
        release(allocs_, p);
    x_ = q_ = k_ = v_ = a_ = gu_ = rs_ = rs2_ = rs_o_ = hs_ = vpart_ = attn_scratch_ = b0_ = pend_sw_ = nullptr;
    h16_ = jq16_ = jk16_ = jv16_ = jo16_ = o16_ = nullptr;
    xq8_ = nullptr;
    cap_hw_ = cap_n_ = cap_text_ = 0;
}

namespace {

// The RoPE's angles as the reference makes them: a coordinate (FP32) times each axis's hd / 8 frequencies
// 1 / base^(k / (hd / 8)) (FP32), the product in FP32; its cosine and sine rounded to bf16, as the reference casts
// them to its activations' type (and was trained with).  Row r's four coordinates from coord(r).
void rope_rows(int64_t rows, int64_t hd, double base, const std::function<void(int64_t, float*)>& coord,
               std::vector<float>& c, std::vector<float>& s) {
    const int64_t bands = hd / 8, half = hd / 2;
    std::vector<float> f((size_t) bands);
    for (int64_t k = 0; k < bands; ++k) {
        const auto e = (float) k / (float) bands;
        f[(size_t) k] = 1.f / (float) std::pow(base, (double) e);
    }
    c.assign((size_t) (rows * half), 0.f);
    s.assign(c.size(), 0.f);
    for (int64_t r = 0; r < rows; ++r) {
        float xyz[4];
        coord(r, xyz);
        for (int64_t j = 0; j < half; ++j) {
            const float ph = xyz[j / bands] * f[(size_t) (j % bands)];
            c[(size_t) (r * half + j)] = dit_rows::round_bf16((float) std::cos((double) ph));
            s[(size_t) (r * half + j)] = dit_rows::round_bf16((float) std::sin((double) ph));
        }
    }
}

}  // namespace

// a context's tables: the image's rows (0, the text's length, row and column centred), then the text's (0, i, 0, 0)
void MossDit::rope_tables(Ctx& c, int64_t h, int64_t w) {
    if (c.rope_h == h && c.rope_w == w && c.rope_rows == c.rows) return;
    const int64_t HW = h * w, L = c.rows;
    std::vector<float> cv, sv;
    rope_rows(HW + L, hd_, rope_base_, [&](int64_t r, float* p) {
        if (r < HW) {
            const int64_t y = r / w, x = r - y * w;
            p[0] = 0.f;
            p[1] = (float) L;
            p[2] = (float) y - (float) (h - 1) / 2.f;
            p[3] = (float) x - (float) (w - 1) / 2.f;
        } else {
            p[0] = 0.f;
            p[1] = (float) (r - HW);
            p[2] = p[3] = 0.f;
        }
    }, cv, sv);
    gpu::stream_sync(s_);
    release(allocs_, c.cos);
    release(allocs_, c.sin);
    c.cos = alloc<float>(allocs_, (int64_t) cv.size());
    c.sin = alloc<float>(allocs_, (int64_t) sv.size());
    gpu::copy(c.cos, cv.data(), cv.size() * 4);
    gpu::copy(c.sin, sv.data(), sv.size() * 4);
    c.rope_h = h;
    c.rope_w = w;
    c.rope_rows = L;
}

// the time's embedding as the reference makes it: the sinusoid [cos, sin] of the timestep by freq / 2 frequencies
// 10000^(-i / (freq / 2)) in FP64, rounded to bf16 (its first layer's type), through the two layers; then the shared
// modulation's six rows (mod_) and the head's two (head_)
void MossDit::time_embedding(float timestep) {
    const int64_t half = freq_ / 2;
    std::vector<uint16_t> e((size_t) freq_);
    for (int64_t i = 0; i < half; ++i) {
        const double ph = (double) timestep * std::pow(10000.0, -(double) i / (double) half);
        e[(size_t) i] = f16_from_f32(round_bf16((float) std::cos(ph)));
        e[(size_t) (half + i)] = f16_from_f32(round_bf16((float) std::sin(ph)));
    }
    gpu::copy_async(t16_, e.data(), e.size() * 2, s_);
    gpu::stream_sync(s_);   // e leaves scope
    uint16_t* h16 = t16_ + freq_;
    product(t16_, nullptr, t1_, nullptr, temb_, 1, b_t1_);
    qi_silu(temb_, C_, h16, s_);
    product(h16, nullptr, t2_, nullptr, temb_, 1, b_t2_);
    qi_to_f16(temb_, C_, h16 + C_, s_);
    product(h16 + C_, nullptr, smod_, nullptr, mod_, 1, b_smod_);
    product(h16 + C_, nullptr, hmod_, nullptr, head_, 1, b_hmod_);
}

void MossDit::run_block(const W* const* w, const float* const* bias, const float* n1, const float* n2,
                        const float* nq, const float* nk, const uint8_t* base, const float* mod, float* x,
                        const std::vector<int64_t>& seg, const std::vector<const float*>& cos,
                        const std::vector<const float*>& sin) {
    const int64_t C = C_;
    int64_t R = 0;
    for (int64_t r : seg) R += r;
    auto b = [&](int j) { return bias != nullptr ? bias[j] : nullptr; };
    const float* shift_a = mod;
    const float* scale_a = mod != nullptr ? mod + C : nullptr;
    const float* gate_a = mod != nullptr ? mod + 2 * C : nullptr;
    const float* shift_f = mod != nullptr ? mod + 3 * C : nullptr;
    const float* scale_f = mod != nullptr ? mod + 4 * C : nullptr;
    const float* gate_f = mod != nullptr ? mod + 5 * C : nullptr;
    // W8A8 blocks take their products' int8 inputs from the row-wise kernels themselves (rows_i8), each into the
    // scales' slot the deferred sums before it do not read (i8_slot)
    const bool i8 = std::all_of(w, w + 6, [](const W* p) { return p->type == kTypeW8; });
    // the block before's MLP output, added to x by this block's first kernel where it ran over the same rows
    MfOut pre;
    const float* pre_gate = nullptr;
    if (pend_.x != nullptr) {
        if (i8 && pend_.x == x && pend_.R == R) {
            pre = pend_.add;
            pre_gate = pend_.gate;
            pend_ = {};
        } else {
            flush_pending();
        }
    }
    // q, k, v of every row (the products' sums scaled by their readers)
    if (i8) rows_i8(RmsI8{x, pre, pre_gate, n1, shift_a, scale_a, C}, R, C, xq8_, i8_slot(), s_);
    else rows16(RmsMod{x, n1, shift_a, scale_a, C}, R, C, h16_, rs_, s_);
    const MfOut q = product(h16_, rs_, *w[0], base, q_, R, b(0), i8, true);
    const MfOut k = product(h16_, rs_, *w[1], base, k_, R, b(1), w[0]->type == kTypeW8, true);
    const MfOut v = product(h16_, rs_, *w[2], base, v_, R, b(2), w[1]->type == kTypeW8, true);
    // each segment: its v heads' scales, its q and k normed and rotated, the attention among its rows, the output
    // back as FP16 rows scaled by row
    int64_t r0 = 0;
    float* so = i8 ? i8_slot() : nullptr;   // the attention's output's scales (q, k and v's sums read the other slot)
    for (size_t i = 0; i < seg.size(); ++i) {
        const int64_t T = seg[i], o = (int64_t) i * jrows_ * C;
        if (dirty_[i] > T)
            for (uint16_t* jb : {jq16_, jk16_, jv16_})
                gpu::memset_async(jb + o + T * C, 0, (size_t) ((dirty_[i] - T) * C) * 2, s_);
        dirty_[i] = T;
        head_max(v.from(r0), T, MfOut{}, 0, heads_, hd_, vpart_, s_);
        head_scale(vpart_, T, heads_, hs_, s_);
        qk_rope16(q.from(r0), k.from(r0), v.from(r0), nq, nk, cos[i], sin[i], hs_, T, heads_, hd_, jq16_ + o,
                  jk16_ + o, jv16_ + o, s_);
        qi_attention(jq16_ + o, jk16_ + o, jv16_ + o, T, heads_, hd_, 0, jo16_, attn_scratch_, s_);
        if (i8) rows_i8(HeadsI8{Heads{jo16_, hs_, C, hd_}}, T, C, xq8_ + r0 * C, so + r0, s_);
        else rows16(Heads{jo16_, hs_, C, hd_}, T, C, o16_ + r0 * C, rs_o_ + r0, s_);
        r0 += T;
    }
    const MfOut ao = product(o16_, rs_o_, *w[3], base, a_, R, b(3), i8, true);
    // the MLP, a band of rows at a time; in one band, the attention's gate_add in the kernel that makes its input
    const bool one_band = i8 && mlp_rows_ >= R;
    if (!one_band) gate_add(x, ao, gate_a, R, C, s_);
    for (int64_t m0 = 0; m0 < R; m0 += mlp_rows_) {
        const int64_t mr = std::min(mlp_rows_, R - m0);
        if (i8) {
            rows_i8(RmsI8{x + m0 * C, one_band ? ao : MfOut{}, gate_a, n2, shift_f, scale_f, C}, mr, C, xq8_,
                    i8_slot(), s_);
            const MfOut g = product(h16_, rs_, *w[4], base, gu_, mr, b(4), true, true);
            rows_i8(SituI8{Situ{g, F_}}, mr, F_, xq8_, i8_slot(), s_);
            MfOut d = product(h16_, rs2_, *w[5], base, a_, mr, b(5), true, true);
            if (!one_band || mod == nullptr) {
                gate_add(x + m0 * C, d, gate_f, mr, C, s_);
                continue;
            }
            // left to the next block's first kernel, the weights' scales copied out of the block's region (the
            // stream may refill it before then)
            gpu::copy_async(pend_sw_, d.sw, (size_t) C * 4, s_);
            d.sw = pend_sw_;
            pend_ = {d, gate_f, x, R};
            continue;
        }
        rows16(RmsMod{x + m0 * C, n2, shift_f, scale_f, C}, mr, C, h16_, rs_, s_);
        // SiTU reads the first product's sums with rs_, so its own rows' scales go to rs2_
        rows16(Situ{product(h16_, rs_, *w[4], base, gu_, mr, b(4), false, true), F_}, mr, F_, h16_, rs2_, s_);
        gate_add(x + m0 * C, product(h16_, rs2_, *w[5], base, a_, mr, b(5), false, true), gate_f, mr, C, s_);
    }
}

void MossDit::flush_pending() {
    if (pend_.x != nullptr) gate_add(pend_.x, pend_.add, pend_.gate, pend_.R, C_, s_);
    pend_ = {};
}

void MossDit::set_context(int i, const float* states, int64_t rows) {
    Ctx& c = ctx_[i];
    gpu::stream_sync(s_);
    release(allocs_, c.txt0);
    c.txt0 = nullptr;
    c.rows = rows;
    c.rope_rows = -1;
    if (rows <= 0) return;
    if (!text_loaded_) load_text();
    ensure(cap_hw_, std::max<int>(1, (int) cap_n_), std::max(cap_text_, rows));
    // the input projection (the states' rows scaled into FP16's range), then the text blocks over the text's rows
    float* xs = alloc<float>(allocs_, rows * txt_dim_);
    uint16_t* x16 = alloc<uint16_t>(allocs_, rows * txt_dim_);
    gpu::copy(xs, states, (size_t) (rows * txt_dim_) * 4);
    rows16(Plain{xs, txt_dim_}, rows, txt_dim_, x16, rs_, s_);
    c.txt0 = alloc<float>(allocs_, rows * C_);
    product(x16, rs_, tproj_, nullptr, c.txt0, rows, b_tproj_);
    std::vector<float> cv, sv;
    rope_rows(rows, hd_, rope_base_, [](int64_t r, float* p) {
        p[0] = p[2] = p[3] = 0.f;
        p[1] = (float) r;
    }, cv, sv);
    float* dc = alloc<float>(allocs_, (int64_t) cv.size());
    float* ds = alloc<float>(allocs_, (int64_t) sv.size());
    gpu::copy(dc, cv.data(), cv.size() * 4);
    gpu::copy(ds, sv.data(), sv.size() * 4);
    for (const TextBlock& tb : tblocks_) {
        const W* w[6] = {&tb.q, &tb.k, &tb.v, &tb.o, &tb.up_gate, &tb.down};
        const float* bias[6] = {tb.bq, tb.bk, tb.bv, tb.bo, tb.bug, tb.bdown};
        run_block(w, bias, tb.n1, tb.n2, tb.nq, tb.nk, nullptr, nullptr, c.txt0, {rows}, {dc}, {ds});
    }
    gpu::stream_sync(s_);
    for (void* p : {(void*) xs, (void*) x16, (void*) dc, (void*) ds}) release(allocs_, p);
}

void MossDit::forward(const float* x, float timestep, int64_t h, int64_t w, int n, float* out) {
    const int64_t HW = h * w, C = C_;
    int64_t maxL = 0;
    std::vector<int64_t> off((size_t) n + 1, 0);
    for (int i = 0; i < n; ++i) {
        maxL = std::max(maxL, ctx_[i].rows);
        off[(size_t) i + 1] = off[(size_t) i] + HW + ctx_[i].rows;
    }
    ensure(HW, n, maxL);
    for (int i = 0; i < n; ++i) rope_tables(ctx_[i], h, w);
    // each context's rows: the image's input projection (the same rows in each), then its text embedder's rows
    qi_to_f16(x, HW * in_ch_, h16_, s_);
    product(h16_, nullptr, vin_, nullptr, x_, HW, b_vin_);
    for (int i = 0; i < n; ++i) {
        if (i > 0) gpu::copy_async(x_ + off[(size_t) i] * C, x_, (size_t) (HW * C) * 4, s_);
        gpu::copy_async(x_ + (off[(size_t) i] + HW) * C, ctx_[i].txt0, (size_t) (ctx_[i].rows * C) * 4, s_);
    }
    time_embedding(timestep);
    for (int bi = 0; bi < (int) blocks_.size(); ++bi) {
        const Block& b = blocks_[(size_t) bi];
        // the contexts that take this block: a run of them (two at most), their rows one after the other
        int c0 = -1, c1 = -1;
        for (int i = 0; i < n; ++i)
            if (std::find(ctx_[i].skip.begin(), ctx_[i].skip.end(), bi) == ctx_[i].skip.end()) {
                if (c0 < 0) c0 = i;
                c1 = i + 1;
            }
        const uint8_t* base = blocks_stream_.begin(bi);
        if (c0 >= 0) {
            auto f = [base](uint64_t o) { return reinterpret_cast<const float*>(base + o); };
            std::vector<int64_t> seg;
            std::vector<const float*> cs, ss;
            for (int i = c0; i < c1; ++i) {
                seg.push_back(HW + ctx_[i].rows);
                cs.push_back(ctx_[i].cos);
                ss.push_back(ctx_[i].sin);
            }
            const W* ws[6] = {&b.q, &b.k, &b.v, &b.o, &b.up_gate, &b.down};
            run_block(ws, nullptr, f(b.n1), f(b.n2), f(b.nq), f(b.nk), base, mod_, x_ + off[(size_t) c0] * C, seg,
                      cs, ss);
        }
        if (bi == 0 && keep0_) flush_pending();
        blocks_stream_.end(bi);
        if (bi == 0 && keep0_) gpu::copy_async(b0_, x_, (size_t) ((HW + ctx_[0].rows) * C) * 4, s_);
    }
    flush_pending();
    // the head: each context's image rows normed, scaled and shifted by the time, projected to the latent's channels
    for (int i = 0; i < n; ++i) {
        rows16(RmsMod{x_ + off[(size_t) i] * C, fnorm_, head_, head_ + C, C}, HW, C, h16_, rs_, s_);
        product(h16_, rs_, fout_, nullptr, out + i * HW * in_ch_, HW, nullptr);
    }
}

namespace {

// an Euler step x += ds p in FP32, the prediction p as the reference pipeline makes it: the prompt's velocity c, with
// guidance (u: the negative pass's) u + cfg (c - u), with skip-layer guidance (k: the skipping pass's) that (or k
// without guidance) plus slg (c - k)
void moss_euler(float* x, const float* c, const float* u, const float* k, int64_t n, float cfg, float slg, float ds,
                void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const size_t i = id[0];
        const float ci = c[i];
        float p = u != nullptr ? u[i] + cfg * (ci - u[i]) : ci;
        if (k != nullptr) p = (u != nullptr ? p : k[i]) + slg * (ci - k[i]);
        x[i] += ds * p;
    });
}

// x0 = x - sigma * v, the step's prediction of the finished latent (a partial image's), v as moss_euler's
void moss_predicted(float* x0, const float* x, const float* c, const float* u, const float* k, int64_t n, float cfg,
                    float slg, float sigma, void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const size_t i = id[0];
        const float ci = c[i];
        float p = u != nullptr ? u[i] + cfg * (ci - u[i]) : ci;
        if (k != nullptr) p = (u != nullptr ? p : k[i]) + slg * (ci - k[i]);
        x0[i] = x[i] - sigma * p;
    });
}

}  // namespace

void MossDit::sample(float* x, int64_t h, int64_t w, const std::vector<float>& sigmas, const Guide& g, int first,
                     int total, const std::function<bool(int, int)>& step,
                     const std::function<void(float*, float)>& after,
                     const std::function<void(int, const std::function<void(float*)>&)>& peek) {
    const int n = g.cfg != 1.f && ctx_[1].rows > 0 ? 2 : 1;
    const bool slg = !g.slg.empty() && g.slg_scale > 0.f;
    const int64_t HW = h * w, N = HW * in_ch_;
    float* out = alloc<float>(allocs_, (n + (slg ? 1 : 0)) * N);
    const std::vector<int> before = ctx_[1].skip;
    const int steps = (int) sigmas.size() - 1;
    try {
        for (int i = 0; i < steps; ++i) {
            // both skips act in the steps whose share of the whole schedule is in [start, stop)
            const double at = (double) (first + i) / (double) total;
            const bool active = g.start <= at && at < g.stop;
            ctx_[1].skip = active ? g.uncond_skip : std::vector<int>{};
            const float t = sigmas[(size_t) i] * 1000.f;
            forward(x, t, h, w, n, out);
            const float* k = nullptr;
            if (slg && active) {
                ctx_[0].skip = g.slg;
                forward(x, t, h, w, 1, out + n * N);
                ctx_[0].skip.clear();
                k = out + n * N;
            }
            moss_euler(x, out, n == 2 ? out + N : nullptr, k, N, g.cfg, g.slg_scale,
                       sigmas[(size_t) i + 1] - sigmas[(size_t) i], s_);
            if (after) after(x, sigmas[(size_t) i + 1]);
            if (peek) {
                const float s_next = sigmas[(size_t) i + 1];
                peek(i + 1, [&](float* x0) {
                    moss_predicted(x0, x, out, n == 2 ? out + N : nullptr, k, N, g.cfg, g.slg_scale, s_next, s_);
                });
            }
            if (step && !step(i + 1, steps)) throw QiStopped();
        }
    } catch (...) {
        gpu::stream_sync(s_);
        ctx_[0].skip.clear();
        ctx_[1].skip = before;
        release(allocs_, out);
        throw;
    }
    gpu::stream_sync(s_);
    ctx_[1].skip = before;
    release(allocs_, out);
}

}  // namespace strata::image
