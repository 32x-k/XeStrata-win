// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/core/plain_model.cpp - an MoE model with one residual stream, run from its GGUF (plain_model.hpp).
//
// The block follows llama.cpp's qwen35moe graph (src/models/qwen35moe.cpp at de7fa0a3) step for step:
//
//   GDN:        qkv = attn_qkv(h), z = attn_gate(h), beta = sigmoid(ssm_beta(h)), alpha = ssm_alpha(h)
//               conv over [history | qkv], SiLU on all of it; q, k l2-normalized (v head j reads q/k head j % h_k)
//               gate = softplus(alpha + dt) * ssm_a; the delta rule; y = rms_norm(o) * ssm_norm * silu(z)
//               out = ssm_out(y)
//   attention:  [q | gate] per head = attn_q(h), k = attn_k(h), v = attn_v(h); q, k rms-normed per head, then
//               rotated in their first 64 dims (text positions: every IMRoPE section at the same position)
//               softmax(q k / sqrt(d)) v over every cell, times sigmoid(gate); out = attn_output(.)
//   MoE:        softmax over the experts, top-k, the k weights renormalized; sum of w_j expert_j(h)
//               + sigmoid(ffn_gate_inp_shexp . h) * shared(h)
//
// and qwen3moe's (llm-jp-4.1, src/models/qwen3moe.cpp): every block a full attention without the gate, rotated over the
// whole head; the MoE without the shared expert; the second norm named ffn_norm.
#include "strata/core/plain_model.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/device.hpp"
#include "strata/core/runtime.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/gdn.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/iq_mmq.hpp"
#include "strata/kernels/native_gdn.hpp"
#include "strata/kernels/native_gdn_preprocess.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/rope_scaling.hpp"
#include "strata/kernels/router_top10.hpp"
#include "strata/kernels/sampler.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <type_traits>
#include <utility>

namespace strata::core {
namespace {

namespace k = strata::kernels;

constexpr int kQ6K = 14;
constexpr int kBF16 = 30;   // a BF16 file's projections: bf16_gemv_fp32_mmvf_multi on the FP32 rows
constexpr int kF32 = 0;

bool mapped(size_t bytes, void** h, void** d) {
    if (!strata::gpu::alloc_host(h, bytes)) return false;
    std::memset(*h, 0, bytes);
    return strata::gpu::device_pointer(d, *h);
}

k::QsaShapes attn_shapes(const ModelSpec& s, int64_t n_head_kv) {
    k::QsaShapes q = k::qsa_real_shapes();
    q.n_head = s.n_head;
    q.n_head_kv = n_head_kv;
    q.head_dim = s.key_length;
    q.n_rot = s.rope_dims;
    return q;
}

}  // namespace

PlainModel::PlainModel() = default;

PlainModel::~PlainModel() {
    if (swap_cs_ != nullptr) strata::gpu::stream_sync(swap_cs_);
    if (cs_ != nullptr) strata::gpu::stream_sync(cs_);
    prompt_gemm_.reset();   // the products' state before the stream
    strata::gpu::free(prompt_host_);
    for (auto* g : exec_) strata::gpu::graph_destroy(g);
    for (auto& v : segs_)
        for (auto* g : v) strata::gpu::graph_destroy(g);
    for (void* p : allocs_) strata::gpu::free(p);
    for (void* p : {(void*) h_emb_, (void*) h_step_, (void*) h_pos_, (void*) h_out_, (void*) fill_tmp_})
        strata::gpu::free(p);
    if (swap_ev_ != nullptr) strata::gpu::event_destroy(swap_ev_);
    if (swap_cs_ != nullptr) strata::gpu::stream_destroy(swap_cs_);
    if (cs_ != nullptr) strata::gpu::stream_destroy(cs_);
}

bool PlainModel::upload(const GgufModel& m, const std::string& name, W& w, std::string& err) {
    size_t at = 0;
    const TensorInfo* t = m.find(name, &at);
    if (t == nullptr || t->shape.size() != 2 || !m.in_bounds(*t, at)) {
        err = "plain model: " + name + " is missing, not 2-D, or truncated";
        return false;
    }
    w.type = (int) t->type;
    w.n_in = (int) t->shape[0];
    w.n_out = (int) t->shape[1];
    if (!k::native_mmvq_supported(w.type) && !(w.type == kBF16 && w.n_in % 2 == 0)) {
        err = "plain model: " + name + " is " + t->type_name() + ", which the native matrix-vector kernels do not take";
        return false;
    }
    const uint64_t bytes = tensor_payload_bytes(*t);
    void* d = nullptr;
    if (!strata::gpu::alloc_device(&d, bytes)) {
        err = "plain model: no VRAM for " + name;
        return false;
    }
    allocs_.push_back(d);
    if (!strata::gpu::copy(d, m.shard(at).tensor_data(*t), bytes)) {
        err = "plain model: uploading " + name + " failed";
        return false;
    }
    if (w.type == kQ6K && k::native_q6_k_rows_ok(w.n_in)) {   // the same results, each row's loads aligned
        k::native_q6_k_to_rows(d, w.n_in, w.n_out, cs_);
        w.type = k::kNativeQ6KRows;
    }
    w.data = d;
    return true;
}

bool PlainModel::upload_f32(const GgufModel& m, const std::string& name, uint64_t elements, const float*& out,
                            std::string& err) {
    size_t at = 0;
    const TensorInfo* t = m.find(name, &at);
    if (t == nullptr || t->type != kF32 || t->elements() != elements || !m.in_bounds(*t, at)) {
        err = "plain model: " + name + " is missing, not F32 or not " + std::to_string(elements) + " values";
        return false;
    }
    void* d = nullptr;
    if (!strata::gpu::alloc_device(&d, elements * 4)) {
        err = "plain model: no VRAM for " + name;
        return false;
    }
    allocs_.push_back(d);
    if (!strata::gpu::copy(d, m.shard(at).tensor_data(*t), elements * 4)) {
        err = "plain model: uploading " + name + " failed";
        return false;
    }
    out = (const float*) d;
    return true;
}

bool PlainModel::load(const PlainOptions& opt, std::string& err) {
    opt_ = opt;
    if (opt_.max_t < 1 || opt_.max_t > k::cpu::MAXT || opt_.max_cells < 16) {
        err = "plain model: the window or the context is out of range";
        return false;
    }
    // not on the processor's own graphics through Level Zero yet: on the UHD 770 the driver aborted (its command stream
    // full) in the batched prompt and after some hundreds of segmented windows
    if (const sycl::device& dev = Runtime::get().device();
        integrated_gpu(dev) && dev.get_backend() == sycl::backend::ext_oneapi_level_zero) {
        err = "plain model: this model does not run on the processor's own graphics yet; choose another GPU (--gpu)";
        return false;
    }
    try {
        model_ = std::make_unique<GgufModel>(GgufModel::open(opt_.gguf));
    } catch (const std::exception& e) {
        err = std::string("plain model: ") + e.what();
        return false;
    }
    if (!read_model_spec(model_->meta(), spec_, err)) return false;
    const ModelSpec& s = spec_;
    if (s.arch != ModelArch::Qwen35Moe && s.arch != ModelArch::Qwen3Moe) {
        err = "plain model: " + s.arch_name + " is not run by this engine yet";
        return false;
    }
    const bool q35 = s.arch == ModelArch::Qwen35Moe;
    has_gate_ = q35;
    has_shared_ = q35;
    bool shape_ok = s.key_length == s.value_length && (s.key_length == 128 || s.key_length == 256) &&
                    (s.rope_dims == 64 || s.rope_dims == s.key_length) && s.n_expert_used <= 15 && s.n_dense_lead == 0;
    if (q35)
        shape_ok = shape_ok && s.ssm_d_state == 128 && s.ssm_d_conv == 4 && s.ssm_n_group > 0 &&
                   s.ssm_dt_rank % s.ssm_n_group == 0 && s.ssm_d_inner == s.ssm_dt_rank * s.ssm_d_state &&
                   s.n_ff_shexp > 0;
    if (!shape_ok) {
        err = "plain model: a shape this engine's kernels are not built for";
        return false;
    }
    cs_ = strata::gpu::stream_create();
    if (cs_ == nullptr) { err = "plain model: stream create failed"; return false; }
    // process constants, before any graph: the rotation (its base, and the scaling the options or the file ask for, as
    // generate.cpp resolves it), and the GDN output gate (SiLU in qwen35moe)
    {
        using RST = k::RopeScalingType;
        k::RopeScaling rs;
        rs.freq_base = s.rope_base;
        const std::string type = !opt_.rope_scaling.empty() ? opt_.rope_scaling : s.rope_scaling_type;
        rs.type = type == "yarn" ? RST::YaRN : type == "linear" ? RST::Linear : RST::None;
        if (rs.type != RST::None) {
            rs.factor = opt_.rope_scale > 0 ? opt_.rope_scale : s.rope_scaling_factor > 1.0 ? s.rope_scaling_factor : 1.0;
            rs.orig_ctx = opt_.yarn_orig_ctx > 0 ? opt_.yarn_orig_ctx
                          : s.rope_scaling_orig_ctx >= 1 ? s.rope_scaling_orig_ctx : (double) s.context_length;
            rs.ext_factor = rs.type == RST::YaRN ? 1.0 : 0.0;
        }
        if (const char* why = k::rope_scaling_invalid(rs)) {
            err = std::string("plain model: the rope scaling: ") + why;
            return false;
        }
        k::rope_scaling_set(rs);
    }
    k::gdn_set_silu_gate(q35);

    const GgufModel& m = *model_;
    // the embedding stays in the mapping: a token's row is dequantized on the host
    {
        size_t at = 0;
        const TensorInfo* t = m.find("token_embd.weight", &at);
        if (t == nullptr || t->shape.size() != 2 || t->shape[0] != (uint64_t) s.n_embd || !m.in_bounds(*t, at)) {
            err = "plain model: token_embd.weight is missing or of another width";
            return false;
        }
        emb_type_ = (int) t->type;
        n_vocab_ = (int64_t) t->shape[1];
        emb_row_ = k::cpu::native_row_bytes(emb_type_, s.n_embd);
        if (emb_row_ == 0) { err = "plain model: token_embd.weight's type has no host dequantizer"; return false; }
        emb_ = m.shard(at).tensor_data(*t);
    }
    if (!upload(m, "output.weight", head_, err) ||
        !upload_f32(m, "output_norm.weight", (uint64_t) s.n_embd, out_norm_, err))
        return false;
    if (head_.n_in != s.n_embd || head_.n_out != n_vocab_) { err = "plain model: output.weight's shape"; return false; }

    const uint64_t N = (uint64_t) s.n_embd, E = (uint64_t) s.n_expert;
    blocks_.resize((size_t) s.n_layer);
    int64_t n_gdn = 0, n_attn = 0;
    for (int64_t l = 0; l < s.n_layer; ++l) {
        Block& b = blocks_[(size_t) l];
        const std::string p = "blk." + std::to_string(l) + ".";
        b.recurrent = s.recurrent[(size_t) l];
        if (!upload_f32(m, p + "attn_norm.weight", N, b.attn_norm, err) ||
            !upload_f32(m, p + (q35 ? "post_attention_norm.weight" : "ffn_norm.weight"), N, b.post_norm, err))
            return false;
        if (b.recurrent) {
            b.gdn = n_gdn++;
            const uint64_t C = (uint64_t) (2 * s.ssm_n_group * s.ssm_d_state + s.ssm_d_inner);
            if (!upload(m, p + "attn_qkv.weight", b.qkv, err) || !upload(m, p + "attn_gate.weight", b.z, err) ||
                !upload(m, p + "ssm_alpha.weight", b.alpha, err) || !upload(m, p + "ssm_beta.weight", b.beta, err) ||
                !upload(m, p + "ssm_out.weight", b.ssm_out, err) ||
                !upload_f32(m, p + "ssm_conv1d.weight", C * (uint64_t) s.ssm_d_conv, b.conv, err) ||
                !upload_f32(m, p + "ssm_dt.bias", (uint64_t) s.ssm_dt_rank, b.dt, err) ||
                !upload_f32(m, p + "ssm_a", (uint64_t) s.ssm_dt_rank, b.a, err) ||
                !upload_f32(m, p + "ssm_norm.weight", (uint64_t) s.ssm_d_state, b.ssm_norm, err))
                return false;
            if ((uint64_t) b.qkv.n_out != C || b.z.n_out != s.ssm_d_inner || b.alpha.n_out != s.ssm_dt_rank ||
                b.beta.n_out != s.ssm_dt_rank || b.ssm_out.n_in != s.ssm_d_inner) {
                err = "plain model: block " + std::to_string(l) + "'s GDN projections have other shapes";
                return false;
            }
        } else {
            b.attn = n_attn++;
            const int64_t nkv = s.n_head_kv[(size_t) l], hd = s.key_length;
            if (!upload(m, p + "attn_q.weight", b.q, err) || !upload(m, p + "attn_k.weight", b.k, err) ||
                !upload(m, p + "attn_v.weight", b.v, err) || !upload(m, p + "attn_output.weight", b.o, err) ||
                !upload_f32(m, p + "attn_q_norm.weight", (uint64_t) hd, b.q_norm, err) ||
                !upload_f32(m, p + "attn_k_norm.weight", (uint64_t) hd, b.k_norm, err))
                return false;
            if (b.q.n_out != (has_gate_ ? 2 : 1) * s.n_head * hd || b.k.n_out != nkv * hd || b.v.n_out != nkv * hd ||
                b.o.n_in != s.n_head * hd || nkv <= 0 || s.n_head % nkv != 0 || (nkv_ != 0 && nkv != nkv_)) {
                err = "plain model: block " + std::to_string(l) + "'s attention projections have other shapes";
                return false;
            }
            nkv_ = nkv;
        }
        if (!upload_f32(m, p + "ffn_gate_inp.weight", N * E, b.router, err)) return false;
        if (!has_shared_) continue;
        if (!upload_f32(m, p + "ffn_gate_inp_shexp.weight", N, b.sh_gate_inp, err) ||
            !upload(m, p + "ffn_gate_shexp.weight", b.sh_gate, err) ||
            !upload(m, p + "ffn_up_shexp.weight", b.sh_up, err) ||
            !upload(m, p + "ffn_down_shexp.weight", b.sh_down, err))
            return false;
        if (b.sh_gate.type != b.sh_up.type || b.sh_gate.n_out != s.n_ff_shexp || b.sh_down.n_in != s.n_ff_shexp) {
            err = "plain model: block " + std::to_string(l) + "'s shared expert has other shapes";
            return false;
        }
    }

    if (nkv_ == 0) { err = "plain model: the model has no attention block"; return false; }
    // the BF16 projections' two GEMV forms timed for each shape a window takes, before any graph is recorded; the
    // head is left to the default form (its timing would allocate a second copy of it)
    try {
        for (const Block& b : blocks_)
            for (const W* w : {&b.qkv, &b.z, &b.alpha, &b.beta, &b.ssm_out, &b.q, &b.k, &b.v, &b.o, &b.sh_gate, &b.sh_up,
                               &b.sh_down})
                if (w->type == kBF16) k::bf16_gemv_fp32_mmvf_multi_tune(w->n_in, w->n_out, cs_);
    } catch (const std::exception& e) {
        err = std::string("plain model: timing the BF16 products: ") + e.what();
        return false;
    }
    // per-sequence state
    const int64_t C = 2 * s.ssm_n_group * s.ssm_d_state + s.ssm_d_inner;
    const k::QsaShapes qs = attn_shapes(s, nkv_);
    for (const Block& b : blocks_) {
        if (b.recurrent) {
            float *st = nullptr, *cv = nullptr;
            if (!strata::gpu::alloc_device(&st, (size_t) (s.ssm_d_state * s.ssm_dt_rank * s.ssm_d_state) * 4) ||
                !strata::gpu::alloc_device(&cv, (size_t) (C * (s.ssm_d_conv - 1)) * 4)) {
                err = "plain model: no VRAM for the recurrent states";
                return false;
            }
            allocs_.push_back(st);
            allocs_.push_back(cv);
            gdn_state_.push_back(st);
            gdn_conv_.push_back(cv);
        }
    }
    n_pages_ = (opt_.max_cells + qs.page_size - 1) / qs.page_size;
    opt_.max_cells = n_pages_ * qs.page_size;
    {
        const size_t pool = (size_t) (n_pages_ * qs.n_head_kv * qs.page_size * qs.head_dim) * 2;
        for (const Block& b : blocks_) {
            if (b.recurrent) continue;
            uint16_t *kp = nullptr, *vp = nullptr;
            if (!strata::gpu::alloc_device(&kp, pool) || !strata::gpu::alloc_device(&vp, pool)) {
                err = "plain model: no VRAM for the K/V cache (" + std::to_string(opt_.max_cells) + " cells)";
                return false;
            }
            allocs_.push_back(kp);
            allocs_.push_back(vp);
            k_pool_.push_back(kp);
            v_pool_.push_back(vp);
        }
        std::vector<int32_t> table((size_t) n_pages_);
        for (int64_t i = 0; i < n_pages_; ++i) table[(size_t) i] = (int32_t) i;
        if (!strata::gpu::alloc_device(&page_table_, table.size() * 4) ||
            !strata::gpu::copy(page_table_, table.data(), table.size() * 4)) {
            err = "plain model: the page table";
            return false;
        }
        allocs_.push_back(page_table_);
    }

    // the window's buffers
    const uint64_t T = (uint64_t) opt_.max_t, NH = (uint64_t) s.n_head, HD = (uint64_t) s.key_length;
    const uint64_t NKV = (uint64_t) qs.n_head_kv, HV = (uint64_t) s.ssm_dt_rank, ZV = (uint64_t) s.ssm_d_inner;
    const uint64_t K = (uint64_t) s.n_expert_used, FS = (uint64_t) s.n_ff_shexp;
    if (!mapped(T * N * 4, (void**) &h_emb_, (void**) &m_emb_) ||
        !mapped(T * k::kStepCount * 4, (void**) &h_step_, (void**) &m_step_) ||
        !mapped(T * (NH + NKV) * 4, (void**) &h_pos_, (void**) &m_pos_) ||
        !mapped(T * 4 + 16, (void**) &h_out_, (void**) &m_out_)) {
        err = "plain model: mapped staging allocation failed";
        return false;
    }
    attn_scratch_floats_ = (int64_t) k::qsa_decode_attn_scratch_floats(opt_.max_cells, qs);
    const uint64_t max_in = std::max<uint64_t>({N, ZV, NH * HD, FS});
    auto dev = [&](auto** p, uint64_t bytes) {
        void* d = nullptr;
        if (!strata::gpu::alloc_device(&d, (size_t) std::max<uint64_t>(bytes, 256))) return false;
        allocs_.push_back(d);
        *p = (std::remove_reference_t<decltype(*p)>) d;
        return true;
    };
    if (!dev(&step_, T * k::kStepCount * 4) || !dev(&pos_, T * (NH + NKV) * 4) || !dev(&x_, T * N * 4) ||
        !dev(&hn_, T * N * 4) || !dev(&mix_, T * N * 4) || !dev(&xq_, k::native_q8_1_bytes((int) max_in, (int) T)) ||
        !dev(&qkv_, T * (uint64_t) C * 4) || !dev(&craw_, (uint64_t) C * 4) || !dev(&hc_, T * (uint64_t) C * 4) ||
        !dev(&z_, T * ZV * 4) || !dev(&alpha_, T * HV * 4) || !dev(&beta_, T * HV * 4) || !dev(&gate_, T * HV * 4) ||
        !dev(&o_, T * ZV * 4) || !dev(&y_, T * ZV * 4) || !dev(&qfull_, T * NH * 2 * HD * 4) ||
        !dev(&qcur_, T * NH * HD * 4) || !dev(&kraw_, T * NKV * HD * 4) || !dev(&kcur_, T * NKV * HD * 4) ||
        !dev(&vcur_, T * NKV * HD * 4) || !dev(&attn_, T * NH * HD * 4) || !dev(&attn32_, T * NH * HD * 4) ||
        !dev(&attn_scratch_, T * (uint64_t) attn_scratch_floats_ * 4) || !dev(&rlog_, T * E * 4) ||
        !dev(&ids_, T * K * 4) || !dev(&w_, T * K * 4) || !dev(&shared_, T * N * 4) || !dev(&sh_g_, (T + 4) * 4) ||
        !dev(&sh_gate_, T * FS * 4) || !dev(&sh_up_, T * FS * 4) || !dev(&moe_out_, T * N * 4) ||
        !dev(&logits_, T * (uint64_t) n_vocab_ * 4)) {
        err = "plain model: no VRAM for the window's buffers";
        return false;
    }
    if (!strata::gpu::stream_sync(cs_)) { err = "plain model: the uploads failed"; return false; }
    if (!open_experts(err)) return false;
    exec_.assign((size_t) opt_.max_t + 1, nullptr);
    segs_.assign((size_t) opt_.max_t + 1, {});
    return reset(err);
}

bool PlainModel::prompt_int8() const {
    const char* v = std::getenv("STRATA_PLAIN_MMQ");
    if (v != nullptr && std::strtol(v, nullptr, 10) == 0) return false;
    sycl::queue& q = Runtime::get().compute();
    if (!k::iq_mmq_usable(q) && !k::iq_mmq_dp4a_usable(q)) return false;
    // the products' outputs in 128s (2 n_ff and n_embd), their K in blocks of 32; gate's rows right before up's
    const int64_t N = spec_.n_embd, FF = spec_.n_ff_exp;
    if (N % 128 != 0 || (2 * FF) % 128 != 0 || FF % 32 != 0) return false;
    for (const auto& f : k::cpu::expert_layout().fmt)
        if (k::cpu::native_mixed_gu(f) ||   // a layer whose gate and up differ in type: the FP16 path's split dequant
            !k::iq_mmq_type_ok(f.gu_type) || !k::iq_mmq_type_ok(f.d_type) ||
            f.up_off != (size_t) FF * k::iq_mmq_row_bytes(f.gu_type, N))
            return false;
    return true;
}

void PlainModel::prompt_carve(int64_t T, PromptBufs& b, uint8_t* dev, uint8_t* host, uint64_t& dev_bytes,
                              uint64_t& host_bytes) const {
    const ModelSpec& s = spec_;
    const auto& lay = k::cpu::expert_layout();
    const uint64_t N = (uint64_t) s.n_embd, NH = (uint64_t) s.n_head, HD = (uint64_t) s.key_length, NKV = (uint64_t) nkv_;
    const uint64_t HV = (uint64_t) s.ssm_dt_rank, ZV = (uint64_t) s.ssm_d_inner;
    const uint64_t C = (uint64_t) (2 * s.ssm_n_group * s.ssm_d_state) + ZV, E = (uint64_t) s.n_expert;
    const uint64_t K = (uint64_t) s.n_expert_used, FS = (uint64_t) s.n_ff_shexp, FF = (uint64_t) s.n_ff_exp;
    const uint64_t U = (uint64_t) T, TK = U * K, QW = (has_gate_ ? 2u : 1u) * NH * HD;
    const uint64_t max_in = std::max<uint64_t>({N, ZV, NH * HD, FS, FF});
    b.T = T;
    b.gu_elems = (int64_t) (2 * FF * N);
    b.d_elems = (int64_t) (N * FF);
    b.int8 = prompt_int8();
    // the largest dense weight, dequantized at once (Gemm::native slices past 2^25 values)
    b.gemm_elems = 1;
    for (const Block& bl : blocks_)
        for (const W* w : {&bl.qkv, &bl.z, &bl.alpha, &bl.beta, &bl.ssm_out, &bl.q, &bl.k, &bl.v, &bl.o, &bl.sh_gate,
                           &bl.sh_up, &bl.sh_down})
            b.gemm_elems = std::max<int64_t>(b.gemm_elems, (int64_t) w->n_in * w->n_out);
    b.gemm_elems = std::min<int64_t>(b.gemm_elems, (int64_t) 1 << 25);
    uint64_t at = 0;
    auto take = [&](auto*& ptr, uint64_t bytes) {
        ptr = dev ? (std::remove_reference_t<decltype(ptr)>) (dev + at) : nullptr;
        at += (std::max<uint64_t>(bytes, 1) + 255) & ~255ull;
    };
    take(b.x, U * N * 4); take(b.hn, U * N * 4); take(b.mix, U * N * 4); take(b.x16, U * max_in * 2);
    take(b.qkv, U * C * 4); take(b.z, U * ZV * 4); take(b.ab, U * 2 * HV * 4); take(b.gate, U * HV * 4);
    take(b.beta, U * HV * 4); take(b.h, U * C * 4); take(b.y, U * ZV * 4); take(b.y16, U * ZV * 2);
    take(b.qfull, U * QW * 4); take(b.qcur, U * NH * HD * 4); take(b.kraw, U * NKV * HD * 4);
    take(b.kcur, U * NKV * HD * 4); take(b.vcur, U * NKV * HD * 4); take(b.attn, U * NH * HD * 4);
    take(b.attn32, U * NH * HD * 4);
    take(b.steps, U * k::kStepCount * 4); take(b.pos_q, U * NH * 4); take(b.pos_k, U * NKV * 4);
    take(b.rlog, U * E * 4); take(b.ids, TK * 4); take(b.w, TK * 4); take(b.shared, U * N * 4);
    take(b.sh_g, (U + 4) * 4); take(b.sh_gate, U * FS * 4); take(b.sh_up, U * FS * 4);
    // the FP16 path's: the routed rows and the SwiGLU in FP16, kGroup experts' FP16 weights, the down product's rows
    // in grouped order (the int8 one writes them in routing order, straight into parts)
    const uint64_t f16 = b.int8 ? 0 : 1;
    take(b.xs16, f16 * TK * N * 2); take(b.gu, TK * 2 * FF * 4); take(b.hh, TK * FF * 4); take(b.h16, f16 * TK * FF * 2);
    take(b.dm, f16 * TK * N * 4); take(b.parts, TK * N * 4); take(b.moe_out, U * N * 4);
    take(b.src, TK * 4); take(b.slot, TK * 4); take(b.bounds, (E + 1) * 4); take(b.entry, TK * 4);
    take(b.dq_gu, f16 * (uint64_t) PromptBufs::kGroup * (uint64_t) b.gu_elems * 2);
    take(b.dq_d, f16 * (uint64_t) PromptBufs::kGroup * (uint64_t) b.d_elems * 2);
    take(b.stage, 2 * (uint64_t) PromptBufs::kGroup * lay.max_blob);   // two halves: one copied while one is read
    // the int8 activations: values, then a scale and a sum per 32 (iq_mmq_act_bytes)
    const uint64_t xq_cols = std::max<uint64_t>(TK * N, U * max_in);   // the routed rows', or a dense input's
    take(b.xq, xq_cols + 2 * (xq_cols / 32) * 4); take(b.hq, TK * FF + 2 * TK * (FF / 32) * 4); take(b.identity, TK * 4);
    take(b.wtab, E * sizeof(void*)); take(b.rbounds, (2 * E + 2) * 4); take(b.dense_bounds, 2 * sizeof(int32_t));
    // the CPU's share of the missed experts (plain_prompt.cpp): at most a window's tokens an expert, so 8 E rows
    b.cpu_cap = std::min<int64_t>((int64_t) TK, (int64_t) k::cpu::MAXT * (int64_t) E);
    take(b.cpu, (uint64_t) b.cpu_cap * N * 4); take(b.cpu_row, (uint64_t) b.cpu_cap * 4);
    take(b.gemm_scratch, (uint64_t) b.gemm_elems * 2);
    dev_bytes = at;
    at = 0;
    auto hold = [&](auto*& ptr, uint64_t bytes) {
        ptr = host ? (std::remove_reference_t<decltype(ptr)>) (host + at) : nullptr;
        at += (std::max<uint64_t>(bytes, 1) + 255) & ~255ull;
    };
    hold(b.emb, U * N * 4); hold(b.h_steps, U * k::kStepCount * 4); hold(b.h_pos, U * (NH + NKV) * 4);
    hold(b.h_ids, TK * 4); hold(b.h_src, TK * 4); hold(b.h_slot, TK * 4); hold(b.h_bounds, (E + 1) * 4);
    hold(b.h_entry, TK * 4);
    hold(b.h_wtab, E * sizeof(void*)); hold(b.h_rbounds, (2 * E + 2) * 4);
    hold(b.h_hn, U * N * 4); hold(b.h_cpu, (uint64_t) b.cpu_cap * N * 4); hold(b.h_cpu_row, (uint64_t) b.cpu_cap * 4);
    hold(b.h_nact, U * k::cpu::kNativeActBytes); hold(b.h_dense_bounds, 2 * sizeof(int32_t));
    host_bytes = at;
}

bool PlainModel::open_experts(std::string& err) {
    const ModelSpec& s = spec_;
    const int64_t L = s.n_moe_layers(), E = s.n_expert;
    std::vector<int64_t> blocks;
    for (int64_t l = 0; l < s.n_layer; ++l)
        if (s.layer_routes(l)) blocks.push_back(l);
    if (!expert_layout_from_gguf(opt_.gguf, blocks, E, s.n_embd, s.n_ff_exp, err)) return false;
    const auto& lay = k::cpu::expert_layout();
    for (int64_t l = 0; l < L; ++l)
        if (!k::native_expert_supported(lay.fmt[(size_t) l].gu_type, lay.fmt[(size_t) l].d_type, s.n_embd, s.n_ff_exp,
                                        lay.fmt[(size_t) l].up_type)) {
            err = "plain model: the GPU has no expert kernel for MoE layer " + std::to_string(l) + "'s types";
            return false;
        }
    // the RAM tier: every expert loaded into pinned RAM when the host has the room (the GPU reads them over PCIe and
    // the pool from RAM), else the GGUF read in place through the page cache
    bool in_ram = !opt_.mmap_experts;
    if (in_ram) {
        detail::HostMemory hm;
        const uint64_t need = lay.total + lay.max_blob + (4ull << 30);   // and room for everything else
        if (detail::host_available_memory(hm) && std::min(hm.available, hm.cgroup_limit) < need) {
            std::fprintf(stderr, "strata plain: %.1f GiB of RAM available, the experts need %.1f GiB: read in place\n",
                         (double) std::min(hm.available, hm.cgroup_limit) / 1073741824.0, (double) need / 1073741824.0);
            in_ram = false;
        }
    }
    if (in_ram) {
        auto arena = std::make_unique<ArenaExpertSource>();
        arena->set_gguf(opt_.gguf);
        if (!arena->open(std::string(), L, E, /*threads=*/6, err)) return false;
        if (!arena->ram_warning().empty()) std::fprintf(stderr, "strata plain: WARNING: %s\n", arena->ram_warning().c_str());
        std::fprintf(stderr, "strata plain: experts in RAM, %.2f GiB at %.2f GiB/s\n", (double) lay.total / 1073741824.0,
                     arena->load_gib_per_second());
        src_ = std::move(arena);
    } else {
        auto file = std::make_unique<FileExpertSource>();
        file->set_gguf(opt_.gguf);
        if (!file->open(std::string(), L, E, err)) return false;
        src_ = std::move(file);
    }
    pool_ = std::make_unique<k::cpu::ExpertPool>(opt_.pool_workers);

    // the VRAM tier: what the device has left, less the window's arena and a reserve, in slots of each layer's blob,
    // the same number for every layer
    uint64_t free_b = 0, total_b = 0;
    {
        size_t f = 0, t = 0;
        if (!strata::gpu::mem_info(&f, &t)) { err = "plain model: the device's memory is unknown"; return false; }
        free_b = f;
        total_b = t;
        apply_vram_limit(free_b, total_b);
    }
    // the prompt path's host buffers, for its longest chunk; the device ones are borrowed from the tier a prompt at a
    // time (prompt_borrow), so the tier is sized from all the free memory
    if (opt_.prompt_chunk > 0) {
        uint64_t db = 0, hb = 0;
        prompt_carve(opt_.prompt_chunk, pb_, nullptr, nullptr, db, hb);
        if (!strata::gpu::alloc_host(&prompt_host_, hb)) {
            err = "plain model: no pinned memory for the prompt path's chunks of " + std::to_string(opt_.prompt_chunk) +
                  " tokens (" + std::to_string(hb >> 20) + " MiB; --prompt-chunk sets a smaller one, 0 none)";
            return false;
        }
        prompt_carve(opt_.prompt_chunk, pb_, nullptr, (uint8_t*) prompt_host_, db, hb);
    }
    const uint64_t window_b = (uint64_t) opt_.max_t * (uint64_t) s.n_expert_used * (uint64_t) s.n_embd * 4 * 3 +
                              16 * lay.max_blob + (64ull << 20);
    const uint64_t reserve = (uint64_t) opt_.reserve_mib << 20;
    const uint64_t room = free_b > window_b + reserve ? free_b - window_b - reserve : 0;
    uint64_t per_layer_b = 0;
    for (int64_t l = 0; l < L; ++l) per_layer_b += lay.bytes[(size_t) l];
    int64_t q = per_layer_b > 0 ? (int64_t) (room / per_layer_b) : 0;
    q = std::min<int64_t>(q, E);
    if (q < 1) {
        err = "plain model: no VRAM is left for the expert tier (" + std::to_string(free_b >> 20) + " MiB free)";
        return false;
    }
    std::vector<int64_t> slot_bytes;
    free_.assign((size_t) L, {});
    for (int64_t i = 0; i < q; ++i)   // the layers in turn: slot s is layer s % L's (prompt_borrow takes the last ones)
        for (int64_t l = 0; l < L; ++l) {
            free_[(size_t) l].push_back((int32_t) slot_bytes.size());
            slot_bytes.push_back((int64_t) lay.bytes[(size_t) l]);
        }
    for (auto& f : free_) std::reverse(f.begin(), f.end());   // taken from the back: the lowest slot first
    cache_.set_max_piece(max_alloc_bytes());   // in pieces where the tier passes the device's largest allocation
    if (!cache_.open_sized(slot_bytes, L, E, err)) return false;
    n_slots_ = (int64_t) slot_bytes.size();
    res_.assign((size_t) (L * E), -1);
    heat_.assign((size_t) (L * E), 0.0);
    if (!strata::gpu::alloc_host(&fill_tmp_, (size_t) lay.max_blob)) {
        err = "plain model: pinned staging for the slot fills";
        return false;
    }

    MoeWindowShape ws;
    ws.n_embd = s.n_embd;
    ws.n_expert = E;
    ws.k = s.n_expert_used;
    ws.max_t = opt_.max_t;
    MoeWindowTier tier;
    tier.cache_base = cache_.device_slot(0);
    tier.blob = (int64_t) lay.max_blob;
    if (!win_.init(ws, tier, err)) return false;
    win_.sink()->pcie_mode = 2;   // the PCIe share staged by the copy kernel inside the window (the split's timing)

    disp_.pool = pool_.get();
    disp_.src = src_.get();
    disp_.n_expert = E;
    disp_.cache = &cache_;
    disp_.cache_base = cache_.device_slot(0);
    disp_.cache_blob = (int64_t) lay.max_blob;
    disp_.cache_slot_off = cache_.slot_offsets();
    disp_.host_res = res_.data();
    disp_.plan = win_.sink();
    disp_.pcie_num = 0;
    if (opt_.adapt_every > 0 && opt_.adapt_swaps > 0) disp_.usage.assign((size_t) (L * E), 0.0f);
    // the split of the missed experts between the CPU and PCIe (after Project Maya's CPU lane): the times measured now,
    // while no expert is in VRAM; only from pinned RAM
    if (opt_.pcie_split && in_ram) {
        std::string why;
        if (measure_pcie_split(disp_, L, split_, why)) {
            disp_.pcie_split = &split_;
            disp_.pcie_overlap = false;   // the window reads the PCIe share after its VRAM hits
            bool any = false;
            for (int f = 1; f <= 16 && !any; ++f) any = pcie_split_m(split_, false, f, f) > 0;
            if (any && src_->device_alias(split_.layer, 0) == nullptr && !src_->make_kernel_readable(why)) {
                std::fprintf(stderr, "strata plain: the experts stay registered for copies (%s): the CPU takes every "
                                     "missed expert\n", why.c_str());
                disp_.pcie_split = nullptr;
            }
            std::fprintf(stderr, "strata plain: PCIe split: an expert %.3f ms on the CPU, %.3f ms over PCIe, %.3f ms on "
                                 "the GPU%s\n", split_.c, split_.p, split_.q, any ? "" : " (the CPU takes them all)");
        } else {
            std::fprintf(stderr, "strata plain: no PCIe split (%s): the CPU takes every missed expert\n", why.c_str());
        }
    }
    // the adaptive tier's copy stream: its rounds bring experts in beside the windows (adapt)
    swap_cs_ = strata::gpu::stream_create();
    if (swap_cs_ == nullptr || !strata::gpu::event_create(&swap_ev_)) {
        err = "plain model: the adaptive tier's copy stream";
        return false;
    }
    // the tier from a profile, after the split's timing (which needs experts in RAM outside the VRAM): its ranking,
    // each layer up to its slots
    if (!opt_.profile.empty()) {
        int64_t ranked_slots = 0;
        if (!read_expert_profile(opt_.profile, L, E, prior_, ranked_slots, err)) return false;
        int64_t filled = 0;
        for (const auto& [l, e] : prior_)
            if (!free_[(size_t) l].empty()) {
                if (!fill_slot(l, e, err)) return false;
                ++filled;
            }
        std::fprintf(stderr, "strata plain: the VRAM tier filled from %s: %lld experts\n", opt_.profile.c_str(),
                     (long long) filled);
    }

    std::fprintf(stderr, "strata plain: %lld MoE layers x %lld experts, %lld VRAM slots (%lld a layer, %.2f GiB), "
                         "%d CPU workers, %lld MiB free before the tier\n",
                 (long long) L, (long long) E, (long long) n_slots_, (long long) q, (double) cache_.bytes() / 1073741824.0,
                 pool_->workers(), (long long) (free_b >> 20));
    return true;
}

int64_t PlainModel::resident() const {
    int64_t n = 0;
    for (int32_t r : res_) n += r >= 0;
    return n;
}

bool PlainModel::reset(std::string& err) {
    const ModelSpec& s = spec_;
    const int64_t C = 2 * s.ssm_n_group * s.ssm_d_state + s.ssm_d_inner;
    bool ok = true;
    for (float* p : gdn_state_)
        ok = ok && strata::gpu::memset(p, 0, (size_t) (s.ssm_d_state * s.ssm_dt_rank * s.ssm_d_state) * 4);
    for (float* p : gdn_conv_) ok = ok && strata::gpu::memset(p, 0, (size_t) (C * (s.ssm_d_conv - 1)) * 4);
    const k::QsaShapes qs = attn_shapes(s, nkv_);
    const size_t pool = (size_t) (n_pages_ * qs.n_head_kv * qs.page_size * qs.head_dim) * 2;
    for (size_t i = 0; i < k_pool_.size(); ++i)
        ok = ok && strata::gpu::memset(k_pool_[i], 0, pool) && strata::gpu::memset(v_pool_[i], 0, pool);
    if (!ok) err = "plain model: resetting the state failed";
    return ok;
}

void PlainModel::pool_fn(void* user, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k, float* out,
                         int64_t layer) {
    PlainModel* m = (PlainModel*) user;
    const int64_t E = m->spec_.n_expert;
    for (int64_t i = 0; i < n_tok * k; ++i)
        if (ids[i] >= 0 && ids[i] < E) m->heat_[(size_t) (layer * E + ids[i])] += 1.0;
    m->disp_.layers = layer;
    expert_pool_dispatch_multi(m->disp_, x_f, ids, n_tok, k, out);
}

bool PlainModel::record_window(int T, strata::gpu::Stream cs, std::string& err) {
    const ModelSpec& s = spec_;
    const int64_t N = s.n_embd, NH = s.n_head, HD = s.key_length, HV = s.ssm_dt_rank, HK = s.ssm_n_group;
    const int64_t ZV = s.ssm_d_inner, S = s.ssm_d_state, C = 2 * HK * S + ZV, E = s.n_expert, K = s.n_expert_used;
    const int64_t FS = s.n_ff_shexp;
    const float eps = (float) s.rms_eps;
    const k::QsaShapes qs = attn_shapes(s, nkv_);
    const int64_t NKV = qs.n_head_kv;
    const k::GdnShapes gs{S, HK, HV};
    const k::RopeScaling& rs = k::rope_scaling();
    const int32_t* pos_q = pos_;
    const int32_t* pos_k = pos_ + (size_t) opt_.max_t * NH;
    auto proj = [&](const W& w, const float* x, float* y, int ncols, bool quantize) {
        if (w.type == kBF16) {   // from the FP32 rows x (row stride n_in), as llama.cpp's float products
            k::bf16_gemv_fp32_mmvf_multi(x, w.n_in, static_cast<const uint16_t*>(w.data), y, w.n_out, w.n_in, w.n_out,
                                         ncols, cs);
            return;
        }
        if (quantize) k::native_quantize_q8_1(x, xq_, w.n_in, ncols, cs);
        k::native_mmvq(w.type, w.data, xq_, y, w.n_in, w.n_out, ncols, cs);
    };
    k::copy_from_mapped(x_, m_emb_, (int64_t) T * N, cs);
    k::copy_i32_from_mapped(step_, m_step_, (int64_t) T * k::kStepCount, cs);
    k::copy_i32_from_mapped(pos_, m_pos_, (int64_t) opt_.max_t * (NH + NKV), cs);
    uint32_t ring = 0;
    for (int64_t l = 0; l < s.n_layer; ++l) {
        const Block& b = blocks_[(size_t) l];
        // the norm writes the q8_1 rows the projections read as well (one launch; after nextsycl's Qwen3.5 norm)
        k::native_qsa_rms_norm_weighted_q8_1(x_, b.attn_norm, hn_, xq_, (int) N, T, eps, cs);
        if (b.recurrent) {
            float* state = gdn_state_[(size_t) b.gdn];
            float* hist = gdn_conv_[(size_t) b.gdn];
            proj(b.qkv, hn_, qkv_, T, false);
            proj(b.z, hn_, z_, T, false);
            // alpha and beta, one shape and type: one launch where the pair kernel takes the type (nextsycl's
            // Qwen3.5 decode fuses the two products)
            if (b.alpha.type == kBF16 || b.alpha.type != b.beta.type || b.alpha.n_in != b.beta.n_in ||
                b.alpha.n_out != b.beta.n_out ||
                !k::native_mmvq_pair(b.alpha.type, b.alpha.data, b.beta.data, xq_, alpha_, beta_, b.alpha.n_in,
                                     b.alpha.n_out, T, cs)) {
                proj(b.alpha, hn_, alpha_, T, false);
                proj(b.beta, hn_, beta_, T, false);
            }
            for (int t = 0; t < T; ++t) {   // the recurrence, token by token
                float* h = hc_ + (size_t) t * C;
                k::native_gdn_conv_silu(hist, qkv_ + (size_t) t * C, b.conv, craw_, h, C, s.ssm_d_conv, cs);
                k::native_gdn_l2_norm(h, HK, S, eps, cs);
                k::native_gdn_l2_norm(h + HK * S, HK, S, eps, cs);
                k::native_gdn_beta_gate(beta_ + (size_t) t * HV, HV, cs);
                k::native_gdn_gate(alpha_ + (size_t) t * HV, b.dt, b.a, gate_ + (size_t) t * HV, HV, cs);
                k::native_gdn_step(state, h, h + HK * S, h + 2 * HK * S, gate_ + (size_t) t * HV, beta_ + (size_t) t * HV,
                                   o_ + (size_t) t * ZV, gs, cs);
                k::native_gdn_out_norm(o_ + (size_t) t * ZV, z_ + (size_t) t * ZV, b.ssm_norm, y_ + (size_t) t * ZV, HV, S,
                                       eps, cs);
            }
            proj(b.ssm_out, y_, mix_, T, true);
        } else {
            const int64_t a = b.attn;
            proj(b.q, hn_, qfull_, T, false);
            proj(b.k, hn_, kraw_, T, false);
            proj(b.v, hn_, vcur_, T, false);
            k::native_qsa_rms_norm_rope(qfull_, (int) ((has_gate_ ? 2 : 1) * HD), b.q_norm, qcur_, (int) (T * NH), (int) HD,
                                        (int) s.rope_dims, eps, rs, pos_q, cs);
            k::native_qsa_rms_norm_rope(kraw_, (int) HD, b.k_norm, kcur_, (int) (T * NKV), (int) HD,
                                        (int) s.rope_dims, eps, rs, pos_k, cs);
            for (int t = 0; t < T; ++t)
                k::kv_append_step(k_pool_[(size_t) a], v_pool_[(size_t) a], page_table_,
                                  step_ + (size_t) t * k::kStepCount, kcur_ + (size_t) t * NKV * HD,
                                  vcur_ + (size_t) t * NKV * HD, qs, cs);
            k::QsaAttnPools pools;
            pools.k_pool = k_pool_[(size_t) a];
            pools.v_pool = v_pool_[(size_t) a];
            pools.page_table = page_table_;
            k::qsa_dense_attn_batch(qcur_, pools, step_, opt_.max_cells, qs, attn_scratch_, attn_, T, cs);
            if (has_gate_) {
                k::native_qsa_gate_apply(attn_, qfull_, attn32_, (int) (T * NH), (int) HD, cs);
                proj(b.o, attn32_, mix_, T, true);
            } else {
                proj(b.o, attn_, mix_, T, true);
            }
        }
        k::add_inplace(x_, mix_, (int64_t) T * N, cs);

        // the MoE: route, publish, the shared expert while the host serves, then the experts and the combine
        const bool sh_q8 = has_shared_ && b.sh_gate.type != kBF16 && b.sh_down.type != kBF16;
        if (sh_q8) k::native_qsa_rms_norm_weighted_q8_1(x_, b.post_norm, hn_, xq_, (int) N, T, eps, cs);
        else k::native_qsa_rms_norm_weighted(x_, b.post_norm, hn_, (int) N, T, eps, cs);
        k::f32_gemv_fp32_mmvf_multi(hn_, N, b.router, rlog_, E, N, E, T, cs);
        k::router_top10(rlog_, T, (int) E, (int) K, ids_, w_, cs);
        ++ring;
        win_.publish(ring, 0, 0, T, hn_, ids_, w_, cs);
        if (has_shared_ && !sh_q8) {   // in FP32 between the products
            proj(b.sh_gate, hn_, sh_gate_, T, true);
            proj(b.sh_up, hn_, sh_up_, T, false);
            k::native_swiglu_f32(sh_gate_, sh_up_, sh_gate_, (int) FS, T, cs);
            proj(b.sh_down, sh_gate_, shared_, T, true);
            k::f32_gemv_fp32_mmvf_multi(hn_, N, b.sh_gate_inp, sh_g_, 1, N, 1, T, cs);
        } else if (has_shared_) {
            if (!k::native_mmvq_pair(b.sh_gate.type, b.sh_gate.data, b.sh_up.data, xq_, sh_gate_, sh_up_, (int) N,
                                     (int) FS, T, cs)) {
                k::native_mmvq(b.sh_gate.type, b.sh_gate.data, xq_, sh_gate_, (int) N, (int) FS, T, cs);
                k::native_mmvq(b.sh_up.type, b.sh_up.data, xq_, sh_up_, (int) N, (int) FS, T, cs);
            }
            k::native_swiglu_q8_1(sh_gate_, sh_up_, xq_, (int) FS, T, cs);
            k::native_mmvq(b.sh_down.type, b.sh_down.data, xq_, shared_, (int) FS, (int) N, T, cs);
            k::f32_gemv_fp32_mmvf_multi(hn_, N, b.sh_gate_inp, sh_g_, 1, N, 1, T, cs);
        }
        if (win_.segmented()) {   // a graph ends where the host must answer the ring
            strata::gpu::Graph* g = nullptr;
            if (!strata::gpu::end_capture(cs, &g)) {
                err = std::string("plain model: segment capture: ") + strata::gpu::last_error();
                return false;
            }
            segs_[(size_t) T].push_back(g);
            if (!strata::gpu::begin_capture(cs)) { err = "plain model: begin capture failed"; return false; }
        }
        if (!win_.experts(s.moe_index(l), ring, 0, 0, T, cs, err)) return false;
        k::native_moe_combine_multi(win_.parts(), w_, has_shared_ ? shared_ : nullptr, moe_out_, N, K, T, cs,
                                    has_shared_ ? sh_g_ : nullptr);
        k::add_inplace(x_, moe_out_, (int64_t) T * N, cs);
    }
    k::native_qsa_rms_norm_weighted(x_, out_norm_, hn_, (int) N, T, eps, cs);
    proj(head_, hn_, logits_, T, true);   // the tokens are picked after the graph (forward), with the run's sampler
    return true;
}

bool PlainModel::record(int T, std::string& err) {
    if (exec_[(size_t) T] != nullptr || !segs_[(size_t) T].empty()) return true;
    if (!strata::gpu::begin_capture(cs_)) {
        err = std::string("plain model: begin capture: ") + strata::gpu::last_error();
        return false;
    }
    bool ok = false;
    try {
        ok = record_window(T, cs_, err);
    } catch (const std::exception& e) {
        err = std::string("plain model: recording the window: ") + e.what();
    }
    if (!ok) {
        strata::gpu::abandon_capture(cs_);
        for (auto* g : segs_[(size_t) T]) strata::gpu::graph_destroy(g);
        segs_[(size_t) T].clear();
        return false;
    }
    strata::gpu::Graph* g = nullptr;
    if (!strata::gpu::end_capture(cs_, &g)) {
        err = std::string("plain model: end capture: ") + strata::gpu::last_error();
        return false;
    }
    if (win_.segmented()) segs_[(size_t) T].push_back(g);
    else exec_[(size_t) T] = g;
    return true;
}

bool PlainModel::forward(const int32_t* tokens, int T, int64_t pos, int32_t* out, float* logits, std::string& err) {
    const ModelSpec& s = spec_;
    if (T < 1 || T > opt_.max_t || pos < 0 || pos + T > opt_.max_cells) {
        err = "plain model: the window is out of range";
        return false;
    }
    if (!record(T, err) || !settle_swaps(false, err)) return false;
    const int64_t N = s.n_embd, NH = s.n_head;
    const k::QsaShapes qs = attn_shapes(s, nkv_);
    for (int t = 0; t < T; ++t) {
        if (tokens[t] < 0 || tokens[t] >= n_vocab_) { err = "plain model: a token is out of the vocabulary"; return false; }
        if (!k::cpu::native_row_to_float(emb_type_, emb_ + (size_t) tokens[t] * emb_row_, h_emb_ + (size_t) t * N, N)) {
            err = "plain model: the embedding row could not be dequantized";
            return false;
        }
        k::qsa_step_fill(h_step_ + (size_t) t * k::kStepCount, pos + t, qs);
        for (int64_t h = 0; h < NH; ++h) h_pos_[(size_t) (t * NH + h)] = (int32_t) (pos + t);
        int32_t* pk = h_pos_ + (size_t) opt_.max_t * NH;
        for (int64_t h = 0; h < qs.n_head_kv; ++h) pk[(size_t) (t * qs.n_head_kv + h)] = (int32_t) (pos + t);
    }
    win_.reset();
    const bool seg = win_.segmented();
    if (!strata::gpu::graph_launch(seg ? segs_[(size_t) T][0] : exec_[(size_t) T], cs_)) {
        err = std::string("plain model: launch: ") + strata::gpu::last_error();
        return false;
    }
    (void) strata::gpu::stream_idle(cs_);
    const int64_t L = s.n_moe_layers();
    std::vector<int64_t> layers((size_t) L);
    std::vector<int> grps((size_t) L, 0), tbs((size_t) L, 0), ns((size_t) L, T);
    for (int64_t l = 0; l < L; ++l) layers[(size_t) l] = l;
    if (seg) win_.set_segments(&segs_[(size_t) T]);
    if (!win_.serve(L, layers.data(), grps.data(), tbs.data(), ns.data(), &PlainModel::pool_fn, this, cs_, err))
        return false;
    k::SamplerParams sp = sampler_;
    sp.counter = (uint64_t) pos;   // row t's draw: position pos + t
    k::sample_tokens(logits_, T, (int) n_vocab_, nullptr, 0, sp, m_out_, cs_);
    if (!strata::gpu::stream_sync(cs_)) {
        err = std::string("plain model: the window: ") + strata::gpu::last_error();
        return false;
    }
    if (disp_.failed) {
        err = std::string("plain model: the CPU experts: ") + (disp_.fail ? disp_.fail : "failed");
        return false;
    }
    for (int t = 0; t < T; ++t) out[t] = ((volatile int32_t*) h_out_)[t];
    if (logits != nullptr && !strata::gpu::copy(logits, logits_, (size_t) T * (size_t) n_vocab_ * 4)) {
        err = "plain model: reading the logits failed";
        return false;
    }
    ++windows_;
    if (!disp_.usage.empty() && windows_ % opt_.adapt_every == 0 && !adapt(err)) return false;
    return true;
}

// One of `layer`'s free slots takes `expert`, read from the expert source (blocking: between windows, or at load).
bool PlainModel::fill_slot(int64_t layer, int32_t expert, std::string& err) {
    auto& f = free_[(size_t) layer];
    if (f.empty()) { err = "plain model: no free slot in the layer"; return false; }
    const int32_t slot = f.back();
    if (!src_->copy_blob(layer, expert, fill_tmp_) ||
        !cache_.fill_slot_blocking(slot, fill_tmp_, err, (int64_t) k::cpu::expert_layout().blob_bytes(layer))) {
        if (err.empty()) err = "plain model: an expert could not be read for the VRAM tier";
        return false;
    }
    f.pop_back();
    res_[(size_t) (layer * spec_.n_expert + expert)] = slot;
    return true;
}

bool PlainModel::save_profile(const std::string& path, std::string& err) const {
    const int64_t L = spec_.n_moe_layers(), E = spec_.n_expert;
    std::vector<uint8_t> resident((size_t) (L * E), 0);
    for (size_t i = 0; i < res_.size(); ++i) resident[i] = res_[i] >= 0;
    return write_expert_profile(path, L, E, rank_learned_profile(L, E, resident, heat_, prior_), err);
}

// XeStrata's adaptive tier (generate.cpp's `adapt`): per MoE layer, a missing expert routed at least twice (decayed
// counts) replaces the least-routed resident one when it leads by 1.5, or takes a free slot; at most adapt_swaps a
// round, the largest gains first; then the counts decay.  Runs between windows, so no window reads a slot it changes.
bool PlainModel::adapt(std::string& err) {
    const int64_t L = spec_.n_moe_layers(), E = spec_.n_expert;
    if (!settle_swaps(false, err)) return false;
    if (!pending_.empty()) return true;   // the last round's copies are still landing: no new round yet
    struct Swap { float gain; int32_t layer, in, out; };   // out < 0: a free slot
    std::vector<Swap> swaps;
    std::vector<std::pair<float, int32_t>> cand, vict;
    for (int64_t l = 0; l < L; ++l) {
        cand.clear();
        vict.clear();
        const float* u = disp_.usage.data() + l * E;
        const int32_t* r = res_.data() + l * E;
        for (int32_t e = 0; e < (int32_t) E; ++e) {
            if (r[e] < 0) { if (u[e] >= 2.0f) cand.emplace_back(u[e], e); }
            else vict.emplace_back(u[e], e);
        }
        for (size_t i = 0; i < free_[(size_t) l].size(); ++i) vict.emplace_back(-1.0f, -1);
        if (cand.empty() || vict.empty()) continue;
        std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.first > b.first; });
        const size_t nc = std::min(cand.size(), vict.size());
        std::partial_sort(vict.begin(), vict.begin() + (ptrdiff_t) nc, vict.end(),
                          [](auto& a, auto& b) { return a.first < b.first; });
        for (size_t i = 0; i < nc; ++i) {
            if (cand[i].first < vict[i].first + 1.5f) break;
            swaps.push_back({cand[i].first - vict[i].first, (int32_t) l, cand[i].second, vict[i].second});
        }
    }
    std::sort(swaps.begin(), swaps.end(), [](const Swap& a, const Swap& b) { return a.gain > b.gain; });
    if ((int) swaps.size() > opt_.adapt_swaps) swaps.resize((size_t) opt_.adapt_swaps);
    for (const Swap& w : swaps) {
        if (w.out < 0) {
            if (free_[(size_t) w.layer].empty()) continue;
        } else {   // the evicted expert's slot becomes the layer's free one, which the fill takes
            const int32_t slot = res_[(size_t) (w.layer * E + w.out)];
            res_[(size_t) (w.layer * E + w.out)] = -1;
            free_[(size_t) w.layer].push_back(slot);
        }
        // from the pinned RAM copy on the copy stream, beside the next windows: the evicted expert is out of the plan
        // already, and the new one goes in once the copy is done (settle_swaps); else read into a slot and wait
        const uint8_t* hb = src_->transient(w.layer, w.in) ? nullptr : src_->blob(w.layer, w.in);
        if (hb != nullptr && strata::gpu::is_host_usm(hb)) {
            auto& f = free_[(size_t) w.layer];
            const int32_t slot = f.back();
            if (!strata::gpu::copy_async(cache_.device_slot(slot), hb, k::cpu::expert_layout().blob_bytes(w.layer),
                                         swap_cs_)) {
                err = std::string("plain model: the adaptive tier's copy: ") + strata::gpu::last_error();
                return false;
            }
            f.pop_back();
            pending_.push_back({w.layer, w.in, slot});
        } else if (!fill_slot(w.layer, w.in, err)) {
            return false;
        }
        ++swaps_total_;
    }
    if (!pending_.empty() && !strata::gpu::event_record(swap_ev_, swap_cs_)) {
        err = std::string("plain model: the adaptive tier's copies: ") + strata::gpu::last_error();
        return false;
    }
    for (float& v : disp_.usage) v *= opt_.adapt_decay;
    return true;
}

bool PlainModel::settle_swaps(bool wait, std::string& err) {
    if (pending_.empty()) return true;
    if (wait ? !strata::gpu::event_sync(swap_ev_) : !strata::gpu::event_query(swap_ev_)) {
        if (!wait) return true;
        err = std::string("plain model: the adaptive tier's copies: ") + strata::gpu::last_error();
        return false;
    }
    for (const PendingSwap& p : pending_) res_[(size_t) (p.layer * spec_.n_expert + p.expert)] = p.slot;
    pending_.clear();
    return true;
}

}  // namespace strata::core
