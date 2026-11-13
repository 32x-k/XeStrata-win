// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/hunyuanimage3_gpu.cpp - see include/strata/image/hunyuanimage3_gpu.hpp.
//
// The DIRECT path: the layer's weights are read from the file, dequantized on the host and uploaded each time the
// layer runs.  It is right, and it is slow - one bank is 1.36 GB over PCIe a layer - so it is the number the host
// FP32 reference is compared against, not the engine's route; that is hunyuanimage3_dit's, and it runs the same
// hunyuanimage3_layer_run with the weights already on the device.

#include "strata/image/hunyuanimage3_gpu.hpp"

#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/image/hunyuanimage3_dit.hpp"
#include "strata/image/hunyuanimage3_file.hpp"
#include "strata/image/hunyuanimage3_kernels.hpp"
#include "strata/image/qi_kernels.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/prefill/gemm.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace strata::image {

namespace {

namespace gpu = strata::gpu;
using strata::kernels::f16_from_f32;

/// Device memory for one layer's work, freed when its guard goes away.
class Arena {
public:
    /// The stream is the caller's: the arena only frees what it took, once the layer's work is queued and synced.
    explicit Arena(void* /* stream */) {}
    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;
    ~Arena() {
        for (void* p : owned_) gpu::free(p);
    }

    template <class T> T* take(int64_t n, bool zero = false) {
        T* p = nullptr;
        const size_t bytes = (size_t) std::max<int64_t>(n, 1) * sizeof(T);
        if (!gpu::alloc_device(&p, bytes))
            throw std::runtime_error(std::string("HunyuanImage 3.0 (GPU layer): GPU memory: ") + gpu::last_error());
        if (zero && !gpu::memset(p, 0, bytes))
            throw std::runtime_error(std::string("HunyuanImage 3.0 (GPU layer): memset: ") + gpu::last_error());
        owned_.push_back(p);
        return p;
    }

    /// host -> device, with the host array's element count checked by the caller.
    template <class T> void up(T* dst, const T* src, int64_t n) {
        if (!gpu::copy(dst, src, (size_t) n * sizeof(T)))
            throw std::runtime_error(std::string("HunyuanImage 3.0 (GPU layer): copy H2D: ") + gpu::last_error());
    }

    void sync() { gpu::device_sync(); }

private:
    std::vector<void*> owned_;
};

/// A quantized matrix's rows [0, n) as FP16 on the device: the file's dequantization on the host (the reader's, the
/// one the calibration held to the BF16 weights), then one upload.  `stage` and `scratch16` are the caller's.
void upload_matrix(const HunyuanImage3File& f, const HunyuanImage3Matrix& m, int64_t expert, uint16_t* dst,
                   float* stage, void* scratch16, Arena& ar) {
    const int64_t n = m.n * m.k;
    f.rows(m, expert, 0, m.n, stage, HunyuanImage3File::low_first());
    uint16_t* h16 = (uint16_t*) scratch16;
    for (int64_t i = 0; i < n; ++i) h16[i] = f16_from_f32(stage[i]);
    ar.up(dst, h16, n);
}

void upload_bf16(const HunyuanImage3File& f, const std::string& name, float* dst, int64_t n, Arena& ar) {
    std::vector<float> h((size_t) n);
    if (!f.bf16(name, h.data())) throw std::runtime_error("HunyuanImage 3.0 (GPU layer): " + f.path() + ": " + name);
    ar.up(dst, h.data(), n);
}

/// The direct path's experts: one host dequantization and one upload each time the walk reaches one.
struct DirectExperts : HunyuanImage3Experts {
    const HunyuanImage3File* f = nullptr;
    const HunyuanImage3Matrix* egu = nullptr;
    const HunyuanImage3Matrix* edn = nullptr;
    float* stage = nullptr;
    void* scratch16 = nullptr;
    Arena* ar = nullptr;

    bool fill(int64_t expert, uint16_t* gate_up16, uint16_t* down16, std::string& err) override {
        (void) err;
        upload_matrix(*f, *egu, expert, gate_up16, stage, scratch16, *ar);
        upload_matrix(*f, *edn, expert, down16, stage, scratch16, *ar);
        return true;
    }
};

}  // namespace

bool hunyuanimage3_layer_gpu(HunyuanImage3File& f, int64_t layer, const float* x, float* out, int64_t T,
                   const int32_t* pos_y, const int32_t* pos_x, int64_t n_img, float* stage, uint16_t* scratch16,
                   void* stream, std::string& err) {
    HunyuanImage3Matrix qkv, o, sgu, sdn, egu, edn;
    const std::string p = "model.layers." + std::to_string(layer) + ".";
    if (!f.matrix(p + "self_attn.qkv_proj.weight", qkv, err) || !f.matrix(p + "self_attn.o_proj.weight", o, err) ||
        !f.matrix(p + "mlp.shared_mlp.gate_and_up_proj.weight", sgu, err) ||
        !f.matrix(p + "mlp.shared_mlp.down_proj.weight", sdn, err) ||
        !f.matrix(p + "mlp.experts_gate_up_proj.weight", egu, err) ||
        !f.matrix(p + "mlp.experts_down_proj.weight", edn, err))
        return false;

    HunyuanImage3Dims d;   // the model's shapes, and the file's in place of the ones it states
    d.hidden = qkv.k;
    d.inter = sgu.n / 2;
    d.experts = egu.experts;
    d.rotate = qkv.rotate;

    Arena ar(stream);
    HunyuanImage3Scratch sc{};
    if (!hunyuanimage3_scratch_alloc(T, d, sc, stream, err)) {
        err = "HunyuanImage 3.0 (GPU layer): " + err;
        return false;
    }
    struct Guard {
        HunyuanImage3Scratch* s;
        ~Guard() { hunyuanimage3_scratch_free(*s); }
    } guard{&sc};

    try {
        HunyuanImage3LayerWeights w;
        uint16_t *qkv16 = ar.take<uint16_t>(qkv.n * qkv.k), *o16 = ar.take<uint16_t>(o.n * o.k);
        uint16_t *sgu16 = ar.take<uint16_t>(sgu.n * sgu.k), *sdn16 = ar.take<uint16_t>(sdn.n * sdn.k);
        float *n_in = ar.take<float>(d.hidden), *n_post = ar.take<float>(d.hidden);
        float *n_q = ar.take<float>(d.head_dim), *n_k = ar.take<float>(d.head_dim);
        float *wg = ar.take<float>(d.experts * d.hidden);
        w.qkv = qkv16, w.o = o16, w.sgu = sgu16, w.sdn = sdn16;
        w.n_in = n_in, w.n_post = n_post, w.n_q = n_q, w.n_k = n_k, w.wg = wg;
        w.qkv_n = qkv.n, w.qkv_k = qkv.k, w.o_n = o.n, w.o_k = o.k;
        w.sgu_n = sgu.n, w.sgu_k = sgu.k, w.sdn_n = sdn.n, w.sdn_k = sdn.k;
        w.egu_n = egu.n, w.egu_k = egu.k, w.edn_n = edn.n, w.edn_k = edn.k;
        upload_matrix(f, qkv, 0, qkv16, stage, scratch16, ar);
        upload_matrix(f, o, 0, o16, stage, scratch16, ar);
        upload_matrix(f, sgu, 0, sgu16, stage, scratch16, ar);
        upload_matrix(f, sdn, 0, sdn16, stage, scratch16, ar);
        upload_bf16(f, p + "input_layernorm.weight", n_in, d.hidden, ar);
        upload_bf16(f, p + "post_attention_layernorm.weight", n_post, d.hidden, ar);
        upload_bf16(f, p + "self_attn.query_layernorm.weight", n_q, d.head_dim, ar);
        upload_bf16(f, p + "self_attn.key_layernorm.weight", n_k, d.head_dim, ar);
        upload_bf16(f, p + "mlp.gate.wg.weight", wg, d.experts * d.hidden, ar);

        prefill::Gemm gemm;
        if (!gemm.init(stream, egu.n * egu.k, err)) return false;
        DirectExperts experts;
        experts.f = &f;
        experts.egu = &egu;
        experts.edn = &edn;
        experts.stage = stage;
        experts.scratch16 = scratch16;
        experts.ar = &ar;
        return hunyuanimage3_layer_run(w, experts, sc, x, out, T, pos_y, pos_x, n_img, d, gemm, stream, err);
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

}  // namespace strata::image
