// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/qi_dit.cpp - see include/strata/image/qi_dit.hpp.

#include "strata/image/qi_dit.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/gpu.hpp"
#include "strata/image/qi_kernels.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace strata::image {

namespace gpu = strata::gpu;

namespace {

constexpr int kTypeF16 = 1, kTypeF32 = 0, kTypeBF16 = 30;
constexpr int64_t kHead = 128;
const int kAxes[3] = {16, 56, 56};   // the rotary embedding's dims per axis: text position, height, width

template<class T> T* alloc(std::vector<void*>& owned, int64_t n, bool zero = false) {
    T* p = nullptr;
    if (!gpu::alloc_device(&p, (size_t) n * sizeof(T)))
        throw std::runtime_error(std::string("image engine: GPU memory: ") + gpu::last_error());
    if (zero && !gpu::memset(p, 0, (size_t) n * sizeof(T)))
        throw std::runtime_error(std::string("image engine: memset: ") + gpu::last_error());
    owned.push_back(p);
    return p;
}

void put(void* d, const void* h, size_t bytes) {
    if (!gpu::copy(d, h, bytes)) throw std::runtime_error(std::string("image engine: copy: ") + gpu::last_error());
}

void put_async(void* d, const void* h, size_t bytes, void* s) {
    if (!gpu::copy_async(d, h, bytes, s))
        throw std::runtime_error(std::string("image engine: copy: ") + gpu::last_error());
}

// the sinusoidal embedding of time t: [cos(t f_i) for i < 128, sin(t f_i)], f_i = 10000^(-i / 128)
void timestep_embedding(float t, float* out) {
    constexpr int half = 128;
    for (int i = 0; i < half; ++i) {
        const float f = std::exp(-std::log(10000.0f) * (float) i / (float) half);
        out[i] = std::cos(t * f);
        out[half + i] = std::sin(t * f);
    }
}

}  // namespace

std::vector<float> philox_randn(uint64_t seed, uint64_t n, uint32_t offset) {
    std::vector<float> out(n);
    const uint32_t M0 = 0xD2511F53u, M1 = 0xCD9E8D57u, W0 = 0x9E3779B9u, W1 = 0xBB67AE85u;
    const float inv32 = 2.3283064e-10f, inv32_2pi = 2.3283064e-10f * 6.2831855f;
    for (uint64_t i = 0; i < n; ++i) {
        uint32_t c0 = offset, c1 = 0, c2 = (uint32_t) i, c3 = 0;
        uint32_t k0 = (uint32_t) seed, k1 = (uint32_t) (seed >> 32);
        for (int r = 0; r < 10; ++r) {
            const uint64_t v1 = (uint64_t) c0 * M0, v2 = (uint64_t) c2 * M1;
            const uint32_t n0 = (uint32_t) (v2 >> 32) ^ c1 ^ k0, n2 = (uint32_t) (v1 >> 32) ^ c3 ^ k1;
            c1 = (uint32_t) v2;
            c3 = (uint32_t) v1;
            c0 = n0;
            c2 = n2;
            k0 += W0;
            k1 += W1;
        }
        const float u = (float) c0 * inv32 + inv32 / 2, v = (float) c1 * inv32_2pi + inv32_2pi / 2;
        out[i] = std::sqrt(-2.0f * std::log(u)) * std::sin(v);
    }
    return out;
}

const std::vector<float>& turbo_sigmas() {
    static const std::vector<float> s = {1.0f, 0.978453f, 0.95418f, 0.926626f, 0.89508f, 0.845148f, 0.704534f,
                                         0.414568f, 0.0f};
    return s;
}

QiDit::QiDit() = default;

QiDit::~QiDit() {
    for (void* p : allocs_) gpu::free(p);
}

QiDit::W QiDit::upload(const std::string& name) {
    const TensorInfo* t = g_->find("model.diffusion_model." + name);
    if (t == nullptr) throw std::runtime_error("image engine: the DiT has no " + name);
    W w;
    w.k = (int64_t) t->shape[0];
    w.n = t->shape.size() > 1 ? (int64_t) t->shape[1] : 1;
    const uint64_t bytes = tensor_payload_bytes(*t);
    if (bytes == 0) throw std::runtime_error("image engine: " + name + ": unknown type " + std::to_string(t->type));
    if ((int) t->type == kTypeF32 && w.n > 1 && dry_) {
        w.type = kTypeF16;
        w.row_bytes = (uint64_t) w.k * 2;
        weight_bytes_ += (uint64_t) (w.n * w.k) * 2;
        return w;
    }
    if ((int) t->type == kTypeF32 && w.n > 1) {   // a matrix in FP32 goes to the GPU as FP16 (the products take it)
        std::vector<uint16_t> h((size_t) (w.n * w.k));
        const auto* f = reinterpret_cast<const float*>(g_->tensor_data(*t));
        for (size_t i = 0; i < h.size(); ++i) h[i] = kernels::f16_from_f32(f[i]);
        uint16_t* d = alloc<uint16_t>(allocs_, (int64_t) h.size());
        put(d, h.data(), h.size() * 2);
        w.type = kTypeF16;
        w.d = d;
        w.row_bytes = (uint64_t) w.k * 2;
        weight_bytes_ += h.size() * 2;
        return w;
    }
    w.type = (int) t->type;
    w.row_bytes = bytes / (uint64_t) w.n;
    weight_bytes_ += bytes;
    if (dry_) return w;
    uint8_t* d = alloc<uint8_t>(allocs_, (int64_t) bytes);
    put(d, g_->tensor_data(*t), bytes);
    w.d = d;
    return w;
}

const float* QiDit::upload_f32(const std::string& name, float add) {
    const TensorInfo* t = g_->find("model.diffusion_model." + name);
    if (t == nullptr) throw std::runtime_error("image engine: the DiT has no " + name);
    const auto n = (int64_t) t->elements();
    if (dry_) {
        weight_bytes_ += (uint64_t) n * 4;
        return nullptr;
    }
    std::vector<float> h((size_t) n);
    const uint8_t* src = g_->tensor_data(*t);
    if ((int) t->type == kTypeF32) std::memcpy(h.data(), src, (size_t) n * 4);
    else if ((int) t->type == kTypeBF16)
        for (int64_t i = 0; i < n; ++i) h[(size_t) i] = kernels::f32_from_bf16(reinterpret_cast<const uint16_t*>(src)[i]);
    else throw std::runtime_error("image engine: " + name + ": a vector of type " + std::to_string(t->type));
    for (float& v : h) v += add;
    float* d = alloc<float>(allocs_, n);
    put(d, h.data(), (size_t) n * 4);
    weight_bytes_ += (uint64_t) n * 4;
    return d;
}

bool QiDit::open(const std::string& path, void* stream, std::string& err, bool dry) {
    try {
        s_ = stream;
        dry_ = dry;
        g_ = std::make_unique<GgufFile>(path);
        iin_ = upload("img_in.weight");
        in_ch_ = iin_.k;
        C_ = iin_.n;
        t1_ = upload("time_text_embed.timestep_embedder.linear_1.weight");
        t2_ = upload("time_text_embed.timestep_embedder.linear_2.weight");
        mod_ = upload("modulation.1.weight");
        nout_ = upload("norm_out.linear.weight");
        pout_ = upload("proj_out.weight");
        tin_ = upload("txt_in.in_layer.weight");
        tout_ = upload("txt_in.out_layer.weight");
        tnorm_ = upload_f32("txt_in.text_norm.weight", 1.0f);   // zero-centred: the weight is w + 1
        int64_t scratch = 0;
        for (int i = 0;; ++i) {   // the blocks: spans of the mapped file, on the GPU through blocks_stream_
            const std::string p = "transformer_blocks." + std::to_string(i) + ".";
            if (g_->find("model.diffusion_model." + p + "attn.to_q.weight") == nullptr) break;
            Block b;
            std::vector<LayerStream::Span> spans;
            auto span = [&](const std::string& name, W& w) {
                const std::string full = std::string(p).append(name);
                const TensorInfo* t = g_->find(std::string("model.diffusion_model.").append(full));
                if (t == nullptr) throw std::runtime_error(std::string("image engine: the DiT has no ").append(full));
                w.type = (int) t->type;
                w.k = (int64_t) t->shape[0];
                w.n = t->shape.size() > 1 ? (int64_t) t->shape[1] : 1;
                const uint64_t bytes = tensor_payload_bytes(*t);
                w.row_bytes = bytes / (uint64_t) w.n;
                spans.push_back({g_->tensor_data(*t), bytes, 0});
                return spans.size() - 1;
            };
            W nq, nk;
            const size_t iq = span("attn.to_q.weight", b.q), ik = span("attn.to_k.weight", b.k),
                         iv = span("attn.to_v.weight", b.v), io = span("attn.to_out.0.weight", b.o),
                         ig = span("img_mlp.gate_up.weight", b.gu), id = span("img_mlp.out.weight", b.down),
                         inq = span("attn.norm_q.weight", nq), ink = span("attn.norm_k.weight", nk);
            if (nq.type != 0 || nk.type != 0) throw std::runtime_error("image engine: the DiT's q/k norms are not FP32");
            const int layer = blocks_stream_.add(spans);
            auto off = [&](size_t k) { return blocks_stream_.offset(layer, (int) k); };
            b.q.d = reinterpret_cast<const void*>(off(iq));
            b.k.d = reinterpret_cast<const void*>(off(ik));
            b.v.d = reinterpret_cast<const void*>(off(iv));
            b.o.d = reinterpret_cast<const void*>(off(io));
            b.gu.d = reinterpret_cast<const void*>(off(ig));
            b.down.d = reinterpret_cast<const void*>(off(id));
            b.nq = off(inq);
            b.nk = off(ink);
            for (const W* w : {&b.q, &b.k, &b.v, &b.o, &b.gu, &b.down}) scratch = std::max(scratch, w->n * w->k);
            blocks_.push_back(b);
        }
        if (blocks_.empty()) throw std::runtime_error("image engine: the DiT has no transformer blocks");
        inter_ = blocks_[0].gu.n / 2;
        heads_ = C_ / kHead;
        for (const W* w : {&t1_, &t2_, &mod_, &nout_, &pout_, &tin_, &tout_, &iin_})
            scratch = std::max(scratch, w->n * w->k);
        if (dry_) return true;
        if (!gemm_.init(s_, scratch, err)) return false;
        blocks_stream_.configure(blocks_stream_.total(), s_);   // all resident until a budget is set
        temb_ = alloc<float>(allocs_, 2 * C_);
        mods_ = alloc<float>(allocs_, 8 * C_);
        scale_ = alloc<float>(allocs_, 2 * C_);
        small16_ = alloc<uint16_t>(allocs_, 2 * C_);
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

void QiDit::product(const uint16_t* x16, const W& w, float* y, int64_t T, int64_t ldy, int64_t row0, int64_t rows) {
    if (rows < 0) rows = w.n;
    const auto* base = static_cast<const uint8_t*>(w.d) + (uint64_t) row0 * w.row_bytes;
    if (w.type == kTypeF16) gemm_.f16(x16, reinterpret_cast<const uint16_t*>(base), y, T, rows, w.k, ldy);
    else gemm_.native(x16, w.type, base, y, T, rows, w.k, ldy);
}

void QiDit::ensure(int64_t T, int64_t HW) {
    if (T > cap_T_) {
        for (void* p : {(void*) x_, (void*) q_, (void*) k_, (void*) v_, (void*) h_, (void*) gu_, (void*) h16_,
                        (void*) q16_, (void*) k16_, (void*) v16_, (void*) o16_, (void*) cos_, (void*) sin_,
                        (void*) attn_scratch_})
            if (p != nullptr) {
                allocs_.erase(std::remove(allocs_.begin(), allocs_.end(), p), allocs_.end());
                gpu::free(p);
            }
        const int64_t TP = qi_attn_rows(T);
        x_ = alloc<float>(allocs_, T * C_);
        q_ = alloc<float>(allocs_, T * C_);
        k_ = alloc<float>(allocs_, T * C_);
        v_ = alloc<float>(allocs_, T * C_);
        h_ = alloc<float>(allocs_, T * std::max(C_, in_ch_));
        mlp_rows_ = std::min(T, mlp_cap_ > 0 ? mlp_cap_ : T);
        gu_ = alloc<float>(allocs_, mlp_rows_ * 2 * inter_);
        h16_ = alloc<uint16_t>(allocs_, T * std::max(C_, inter_));
        q16_ = alloc<uint16_t>(allocs_, TP * C_, true);
        k16_ = alloc<uint16_t>(allocs_, TP * C_, true);
        v16_ = alloc<uint16_t>(allocs_, TP * C_, true);
        o16_ = alloc<uint16_t>(allocs_, T * C_);
        cos_ = alloc<float>(allocs_, T * kHead / 2);
        sin_ = alloc<float>(allocs_, T * kHead / 2);
        // the plain attention's FP32 output: only where the matrix engines' path is not taken
        if (std::strcmp(qi_attention_path(), "plain (FP32)") == 0) attn_scratch_ = alloc<float>(allocs_, T * C_);
        cap_T_ = T;
        rope_H_ = rope_W_ = rope_text_ = -1;
    }
    (void) HW;
}

void QiDit::set_context(const float* ctx, int64_t n_text) {
    ensure(std::max(n_text, cap_T_), 0);
    n_text_ = n_text;
    float* raw = q_;   // the encoder's states, then the projection's first product
    put_async(raw, ctx, (size_t) (n_text * C_) * 4, s_);
    qi_rms_norm16(raw, tnorm_, n_text, C_, h16_, s_);
    product(h16_, tin_, k_, n_text);
    qi_gelu16(k_, n_text * C_, h16_, s_);
    if (text_ != nullptr) {
        allocs_.erase(std::remove(allocs_.begin(), allocs_.end(), (void*) text_), allocs_.end());
        gpu::free(text_);
    }
    text_ = alloc<float>(allocs_, n_text * C_);
    product(h16_, tout_, text_, n_text);
    for (Block& b : blocks_) {
        for (uint16_t** p : {&b.pk, &b.pv}) {
            if (*p != nullptr) {
                allocs_.erase(std::remove(allocs_.begin(), allocs_.end(), (void*) *p), allocs_.end());
                gpu::free(*p);
            }
            *p = alloc<uint16_t>(allocs_, n_text * C_);
        }
    }
    gpu::stream_sync(s_);
}

void QiDit::step_modulation(float t) {
    std::vector<float> e((size_t) 2 * 256);
    timestep_embedding(t, e.data());
    timestep_embedding(0.0f, e.data() + 256);
    std::vector<uint16_t> e16(e.size());
    for (size_t i = 0; i < e.size(); ++i) e16[i] = kernels::f16_from_f32(e[i]);
    put_async(small16_, e16.data(), e16.size() * 2, s_);
    gpu::stream_sync(s_);   // e16 is the host's: the copy is done before it goes
    product(small16_, t1_, temb_, 2);
    qi_silu(temb_, 2 * C_, small16_, s_);
    product(small16_, t2_, temb_, 2);
    qi_silu(temb_, 2 * C_, small16_, s_);   // the time after its SiLU: the modulation's and the final norm's input
    for (int k = 0; k < 4; ++k) product(small16_, mod_, mods_ + (int64_t) k * 2 * C_, 2, C_, (int64_t) k * C_, C_);
    product(small16_, nout_, scale_, 1);
}

void QiDit::forward(const float* x, float t, int64_t H, int64_t W, bool prefix, float* out) {
    const int64_t HW = H * W, T = n_text_ + HW, C = C_;
    ensure(T, HW);
    if (rope_H_ != H || rope_W_ != W || rope_text_ != n_text_) {   // the 3-axis rotary tables [T][64]
        std::vector<float> pos((size_t) T * 3);
        for (int64_t i = 0; i < n_text_; ++i) pos[i * 3] = pos[i * 3 + 1] = pos[i * 3 + 2] = (float) i;
        for (int64_t y = 0; y < H; ++y)
            for (int64_t xx = 0; xx < W; ++xx) {
                float* p = &pos[(size_t) (n_text_ + y * W + xx) * 3];
                p[0] = (float) n_text_;
                const int64_t hh = H - H / 2, wh = W - W / 2;   // the grid's centre, as the model was trained
                p[1] = (float) (y - hh);
                p[2] = (float) (xx - wh);
            }
        std::vector<float> c((size_t) T * kHead / 2), s(c.size());
        for (int64_t i = 0; i < T; ++i) {
            int j = 0;
            for (int a = 0; a < 3; ++a)
                for (int f = 0; f < kAxes[a] / 2; ++f, ++j) {
                    const float omega = 1.0f / std::pow(10000.0f, (float) (2 * f) / (float) kAxes[a]);
                    const float ang = pos[(size_t) i * 3 + a] * omega;
                    c[(size_t) (i * kHead / 2 + j)] = std::cos(ang);
                    s[(size_t) (i * kHead / 2 + j)] = std::sin(ang);
                }
        }
        put(cos_, c.data(), c.size() * 4);
        put(sin_, s.data(), s.size() * 4);
        rope_H_ = H;
        rope_W_ = W;
        rope_text_ = n_text_;
    }
    step_modulation(t);
    const int64_t r0 = prefix ? n_text_ : 0, Tr = T - r0, ntr = n_text_ - r0;
    if (!prefix) gpu::copy_async(x_, text_, (size_t) (n_text_ * C) * 4, s_);
    qi_latent_tokens16(x, in_ch_, HW, h16_, s_);
    product(h16_, iin_, x_ + n_text_ * C, HW);
    auto mod = [&](int i) { return mods_ + (int64_t) i * 2 * C; };
    float* xr = x_ + r0 * C;
    const size_t text_bytes = (size_t) (n_text_ * C) * 2;
    for (int bi = 0; bi < (int) blocks_.size(); ++bi) {
        Block& b = blocks_[(size_t) bi];
        const uint8_t* base = blocks_stream_.begin(bi);   // the block's weights on the GPU (resident or a slot)
        auto at = [base](const QiDit::W& w) {
            QiDit::W r = w;
            r.d = base + reinterpret_cast<uintptr_t>(w.d);
            return r;
        };
        const QiDit::W wq = at(b.q), wk = at(b.k), wv = at(b.v), wo = at(b.o), wgu = at(b.gu), wdown = at(b.down);
        const auto* nq = reinterpret_cast<const float*>(base + b.nq);
        const auto* nk = reinterpret_cast<const float*>(base + b.nk);
        qi_ln_mod16(xr, mod(0), ntr, Tr, C, h16_, s_);
        product(h16_, wq, q_ + r0 * C, Tr);
        product(h16_, wk, k_ + r0 * C, Tr);
        product(h16_, wv, v_ + r0 * C, Tr);
        qi_qk_norm_rope16(q_ + r0 * C, k_ + r0 * C, v_ + r0 * C, nq, nk, cos_ + r0 * kHead / 2,
                          sin_ + r0 * kHead / 2, Tr, heads_, kHead, q16_ + r0 * C, k16_ + r0 * C, v16_ + r0 * C, s_);
        if (n_text_ > 0) {
            if (prefix) {
                gpu::copy_async(k16_, b.pk, text_bytes, s_);
                gpu::copy_async(v16_, b.pv, text_bytes, s_);
            } else {
                gpu::copy_async(b.pk, k16_, text_bytes, s_);
                gpu::copy_async(b.pv, v16_, text_bytes, s_);
            }
        }
        qi_attention(q16_, k16_, v16_, T, heads_, kHead, n_text_, o16_, attn_scratch_, s_);
        product(o16_ + r0 * C, wo, h_, Tr);
        qi_gate_add(xr, h_, mod(1), ntr, Tr, C, s_);
        for (int64_t m0 = 0; m0 < Tr; m0 += mlp_rows_) {   // the MLP a band of rows at a time (its 2 x inter wide product)
            const int64_t mr = std::min(mlp_rows_, Tr - m0), mt = std::max<int64_t>(0, std::min(ntr - m0, mr));
            qi_ln_mod16(xr + m0 * C, mod(2), mt, mr, C, h16_, s_);
            product(h16_, wgu, gu_, mr);
            qi_swiglu16(gu_, mr, inter_, h16_, s_);
            product(h16_, wdown, h_, mr);
            qi_gate_add(xr + m0 * C, h_, mod(3), mt, mr, C, s_);
        }
        blocks_stream_.end(bi);
    }
    qi_ln_mod16(x_ + n_text_ * C, scale_, 0, HW, C, h16_, s_);   // the final norm: row 0 of scale_ for every token
    product(h16_, pout_, h_, HW);
    qi_tokens_latent(h_, in_ch_, HW, out, s_);
}

uint64_t QiDit::activation_bytes(int64_t T, int64_t mlp_rows) const {
    const int64_t TP = qi_attn_rows(T), mr = std::min(T, mlp_rows > 0 ? mlp_rows : T);
    const bool plain = std::strcmp(qi_attention_path(), "plain (FP32)") == 0;
    return (uint64_t) T * (uint64_t) C_ * 4 * 4 + (uint64_t) T * (uint64_t) std::max(C_, in_ch_) * 4 +
           (uint64_t) mr * 2 * (uint64_t) inter_ * 4 + (uint64_t) T * (uint64_t) std::max(C_, inter_) * 2 +
           (uint64_t) TP * (uint64_t) C_ * 2 * 3 + (uint64_t) T * (uint64_t) C_ * 2 + (uint64_t) T * kHead * 4 +
           (plain ? (uint64_t) T * (uint64_t) C_ * 4 : 0);
}

void QiDit::set_mlp_rows(int64_t rows) {
    if (rows != mlp_cap_) {
        mlp_cap_ = rows;
        drop_activations();
    }
}

void QiDit::drop_activations() {
    for (void* p : {(void*) x_, (void*) q_, (void*) k_, (void*) v_, (void*) h_, (void*) gu_, (void*) h16_, (void*) q16_,
                    (void*) k16_, (void*) v16_, (void*) o16_, (void*) cos_, (void*) sin_, (void*) attn_scratch_})
        if (p != nullptr) {
            allocs_.erase(std::remove(allocs_.begin(), allocs_.end(), p), allocs_.end());
            gpu::free(p);
        }
    x_ = q_ = k_ = v_ = h_ = gu_ = cos_ = sin_ = attn_scratch_ = nullptr;
    h16_ = q16_ = k16_ = v16_ = o16_ = nullptr;
    cap_T_ = 0;
    rope_H_ = rope_W_ = rope_text_ = -1;
}

void QiDit::sample(float* x, int64_t H, int64_t W, const std::vector<float>& sigmas,
                   const std::function<bool(int, int)>& step) {
    const int64_t n = in_ch_ * H * W;
    float* v = alloc<float>(allocs_, n);
    const int steps = (int) sigmas.size() - 1;
    for (int i = 0; i < steps; ++i) {
        forward(x, sigmas[(size_t) i] * 1000.0f, H, W, i > 0, v);
        qi_axpy(x, v, sigmas[(size_t) i + 1] - sigmas[(size_t) i], n, s_);
        if (step) {
            gpu::stream_sync(s_);
            if (!step(i + 1, steps)) {
                allocs_.erase(std::remove(allocs_.begin(), allocs_.end(), (void*) v), allocs_.end());
                gpu::free(v);
                throw QiStopped();
            }
        }
    }
    gpu::stream_sync(s_);
    allocs_.erase(std::remove(allocs_.begin(), allocs_.end(), (void*) v), allocs_.end());
    gpu::free(v);
}

}  // namespace strata::image
