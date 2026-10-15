// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/qi_text.cpp - see include/strata/image/qi_text.hpp.

#include "strata/image/qi_text.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/gpu.hpp"
#include "strata/image/qi_kernels.hpp"
#include "strata/kernels/dequant_bf16.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace strata::image {

namespace gpu = strata::gpu;

namespace {

template<class T> T* alloc(std::vector<void*>& owned, int64_t n) {
    T* p = nullptr;
    if (!gpu::alloc_device(&p, (size_t) n * sizeof(T)))
        throw std::runtime_error(std::string("image engine (text): GPU memory: ") + gpu::last_error());
    owned.push_back(p);
    return p;
}

void release(std::vector<void*>& owned, void* p) {
    if (p == nullptr) return;
    owned.erase(std::remove(owned.begin(), owned.end(), p), owned.end());
    gpu::free(p);
}

}  // namespace

QiText::QiText() = default;

QiText::~QiText() {
    for (void* p : allocs_) gpu::free(p);
}

bool QiText::open(const std::string& path, void* stream, std::string& err, bool dry) {
    try {
        s_ = stream;
        g_ = std::make_unique<GgufFile>(path);
        const MetaValue* arch = g_->get("general.architecture");
        const std::string a = arch ? arch->s : "";
        auto num = [&](const std::string& key, double def) {
            const MetaValue* v = g_->get(a + "." + key);
            return v ? v->num() : def;
        };
        const auto layers = (int64_t) num("block_count", 0);
        C_ = (int64_t) num("embedding_length", 4096);
        heads_ = (int64_t) num("attention.head_count", 32);
        kv_heads_ = (int64_t) num("attention.head_count_kv", 8);
        head_dim_ = (int64_t) num("attention.key_length", 128);
        inter_ = (int64_t) num("feed_forward_length", 12288);
        theta_ = (float) num("rope.freq_base", 5e6);
        eps_ = (float) num("attention.layer_norm_rms_epsilon", 1e-6);
        if (layers <= 0) throw std::runtime_error("image engine (text): " + path + " has no blocks");
        const TensorInfo* e = g_->find("token_embd.weight");
        if (e == nullptr) throw std::runtime_error("image engine (text): no token_embd.weight");
        embd_type_ = (int) e->type;
        embd_row_bytes_ = tensor_payload_bytes(*e) / e->shape[1];
        embd_ = g_->tensor_data(*e);
        int64_t scratch = 0;
        for (int64_t i = 0; i < layers; ++i) {   // the layers: spans of the mapped file, through layers_stream_
            const std::string p = "blk." + std::to_string(i) + ".";
            Layer l;
            std::vector<LayerStream::Span> spans;
            auto span = [&](const std::string& name, W* w) {
                const std::string full = std::string(p).append(name);
                const TensorInfo* t = g_->find(full);
                if (t == nullptr) throw std::runtime_error(std::string("image engine (text): the encoder has no ").append(full));
                if (w != nullptr) {
                    w->type = (int) t->type;
                    w->k = (int64_t) t->shape[0];
                    w->n = t->shape.size() > 1 ? (int64_t) t->shape[1] : 1;
                } else if (t->type != 0) {
                    throw std::runtime_error(std::string("image engine (text): ").append(full).append(" is not FP32"));
                }
                spans.push_back({g_->tensor_data(*t), tensor_payload_bytes(*t), 0});
                return (int) spans.size() - 1;
            };
            const int iq = span("attn_q.weight", &l.q), ik = span("attn_k.weight", &l.k),
                      iv = span("attn_v.weight", &l.v), io = span("attn_output.weight", &l.o),
                      ig = span("ffn_gate.weight", &l.gate), iu = span("ffn_up.weight", &l.up),
                      id = span("ffn_down.weight", &l.down), ian = span("attn_norm.weight", nullptr),
                      ifn = span("ffn_norm.weight", nullptr), iqn = span("attn_q_norm.weight", nullptr),
                      ikn = span("attn_k_norm.weight", nullptr);
            const int layer = layers_stream_.add(spans);
            auto off = [&](int k) { return layers_stream_.offset(layer, k); };
            l.q.d = reinterpret_cast<const void*>(off(iq));
            l.k.d = reinterpret_cast<const void*>(off(ik));
            l.v.d = reinterpret_cast<const void*>(off(iv));
            l.o.d = reinterpret_cast<const void*>(off(io));
            l.gate.d = reinterpret_cast<const void*>(off(ig));
            l.up.d = reinterpret_cast<const void*>(off(iu));
            l.down.d = reinterpret_cast<const void*>(off(id));
            l.attn_norm = off(ian);
            l.ffn_norm = off(ifn);
            l.qn = off(iqn);
            l.kn = off(ikn);
            for (const W* w : {&l.q, &l.k, &l.v, &l.o, &l.gate, &l.up, &l.down}) scratch = std::max(scratch, w->n * w->k);
            layers_.push_back(l);
        }
        if (dry) return true;
        if (!gemm_.init(s_, scratch, err)) return false;
        layers_stream_.configure(layers_stream_.total(), s_);   // all resident until a budget is set
        return true;
    } catch (const std::exception& ex) {
        err = ex.what();
        return false;
    }
}

void QiText::ensure(int64_t T) {
    if (T <= cap_T_) return;
    for (void* p : {(void*) x_, (void*) q_, (void*) k_, (void*) v_, (void*) a_, (void*) h_, (void*) gu_, (void*) h16_,
                    (void*) q16_, (void*) k16_, (void*) v16_, (void*) rows_})
        release(allocs_, p);
    const int64_t kv = kv_heads_ * head_dim_;
    x_ = alloc<float>(allocs_, T * C_);
    q_ = alloc<float>(allocs_, T * C_);
    k_ = alloc<float>(allocs_, T * kv);
    v_ = alloc<float>(allocs_, T * kv);
    a_ = alloc<float>(allocs_, T * C_);
    h_ = alloc<float>(allocs_, T * C_);
    gu_ = alloc<float>(allocs_, T * 2 * inter_);
    h16_ = alloc<uint16_t>(allocs_, T * std::max(C_, inter_));
    q16_ = alloc<uint16_t>(allocs_, T * C_);
    k16_ = alloc<uint16_t>(allocs_, T * kv);
    v16_ = alloc<uint16_t>(allocs_, T * kv);
    rows_ = alloc<uint8_t>(allocs_, (int64_t) (T * (int64_t) embd_row_bytes_));
    cap_T_ = T;
}

std::vector<float> QiText::encode(const std::vector<int32_t>& ids) {
    const auto T = (int64_t) ids.size();
    if (T == 0) return {};
    ensure(T);
    // the prompt's embedding rows, gathered from the mapped file, expanded on the GPU
    std::vector<uint8_t> host((size_t) (T * (int64_t) embd_row_bytes_));
    for (int64_t t = 0; t < T; ++t)
        std::memcpy(&host[(size_t) t * embd_row_bytes_], embd_ + (uint64_t) ids[(size_t) t] * embd_row_bytes_,
                    embd_row_bytes_);
    gpu::copy(rows_, host.data(), host.size());
    kernels::dequant_f32(embd_type_, rows_, 0, T, C_, x_, s_);
    for (int li = 0; li < (int) layers_.size(); ++li) {
        const Layer& l = layers_[(size_t) li];
        const uint8_t* base = layers_stream_.begin(li);
        auto w = [base](const W& x) { return base + reinterpret_cast<uintptr_t>(x.d); };
        auto f = [base](uint64_t off) { return reinterpret_cast<const float*>(base + off); };
        qi_rms_norm16(x_, f(l.attn_norm), T, C_, h16_, s_);
        gemm_.native(h16_, l.q.type, w(l.q), q_, T, l.q.n, l.q.k);
        gemm_.native(h16_, l.k.type, w(l.k), k_, T, l.k.n, l.k.k);
        gemm_.native(h16_, l.v.type, w(l.v), v_, T, l.v.n, l.v.k);
        qi_text_qk_rope16(q_, k_, v_, f(l.qn), f(l.kn), T, heads_, kv_heads_, head_dim_, theta_, q16_, k16_, v16_, s_);
        qi_attention_causal(q16_, k16_, v16_, T, heads_, kv_heads_, head_dim_, a_, s_);
        qi_to_f16(a_, T * C_, h16_, s_);
        gemm_.native(h16_, l.o.type, w(l.o), h_, T, l.o.n, l.o.k);
        qi_axpy(x_, h_, 1.0f, T * C_, s_);
        qi_rms_norm16(x_, f(l.ffn_norm), T, C_, h16_, s_);
        gemm_.native(h16_, l.gate.type, w(l.gate), gu_, T, l.gate.n, l.gate.k, 2 * inter_);
        gemm_.native(h16_, l.up.type, w(l.up), gu_ + inter_, T, l.up.n, l.up.k, 2 * inter_);
        qi_swiglu16(gu_, T, inter_, h16_, s_);
        gemm_.native(h16_, l.down.type, w(l.down), h_, T, l.down.n, l.down.k);
        qi_axpy(x_, h_, 1.0f, T * C_, s_);
        layers_stream_.end(li);
    }
    std::vector<float> out((size_t) (T * C_));
    gpu::stream_sync(s_);
    gpu::copy(out.data(), x_, out.size() * 4);
    return out;
}

}  // namespace strata::image
