// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/prefill/plain_prompt.cpp - PlainModel::prompt: a plain model's prompt in chunks of batched products.
//
// The block of plain_model.cpp, T tokens at a time: every projection a matrix product of the chunk's rows by the weight
// dequantized to FP16 (Gemm::native), the GDN conv and recurrence walking the chunk inside their kernels (the prompt
// path's, kernels.hpp), and the routed experts grouped by expert - each expert's weights read once for all the tokens
// routed to it, from its VRAM slot or copied from RAM, dequantized to FP16 and multiplied in one grouped product per
// group of experts.  The norms with the rotation, the gate, the K/V append, the attention over every cell, the router
// and the combine are the decode's kernels, called over the chunk's rows.
//
// The state is left as the windows leave it: the GDN recurrent state and conv history (the same layouts), the K/V
// cells of the chunk's positions.  The decode then continues at the next position.
#include "strata/core/plain_model.hpp"

#include "strata/core/gpu.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/rope_scaling.hpp"
#include "strata/kernels/router_top10.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/xmx_gemm.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/kernels.hpp"
#include "strata/prefill/moe_mmq.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <exception>
#include <type_traits>
#include <vector>

namespace strata::core {

namespace k = strata::kernels;

/// The prompt path's buffers, sized for chunks of T tokens.
struct PlainModel::Prompt {
    static constexpr int kGroup = 8;   ///< experts dequantized and multiplied in one grouped product
    int64_t T = 0;
    prefill::Gemm gemm;
    std::vector<void*> dev;            ///< every device allocation
    void* host_block = nullptr;        ///< the pinned host staging, one allocation
    // device
    float *x = nullptr, *hn = nullptr, *mix = nullptr;
    uint16_t* x16 = nullptr;           ///< T rows of the widest projection input, FP16
    float *qkv = nullptr, *z = nullptr, *ab = nullptr, *gate = nullptr, *beta = nullptr, *h = nullptr, *y = nullptr;
    uint16_t* y16 = nullptr;
    float *qfull = nullptr, *qcur = nullptr, *kraw = nullptr, *kcur = nullptr, *vcur = nullptr, *attn = nullptr,
          *attn32 = nullptr;
    int32_t *steps = nullptr, *pos_q = nullptr, *pos_k = nullptr;
    float *rlog = nullptr, *w = nullptr, *shared = nullptr, *sh_g = nullptr, *sh_gate = nullptr, *sh_up = nullptr;
    int32_t* ids = nullptr;
    uint16_t* xs16 = nullptr;          ///< the routed rows grouped by expert, FP16 [T * K, n_embd]
    float *gu = nullptr, *hh = nullptr, *dm = nullptr, *parts = nullptr, *moe_out = nullptr;
    uint16_t* h16 = nullptr;
    int32_t *src = nullptr, *slot = nullptr, *bounds = nullptr;
    uint16_t *dq_gu = nullptr, *dq_d = nullptr;   ///< kGroup experts' FP16 weights
    uint8_t* stage = nullptr;          ///< kGroup expert blobs copied from RAM
    // pinned host
    float* emb = nullptr;              ///< T rows of n_embd
    int32_t *h_steps = nullptr, *h_pos = nullptr, *h_ids = nullptr, *h_src = nullptr, *h_slot = nullptr,
            *h_bounds = nullptr;
    int64_t gu_elems = 0, d_elems = 0;
};

void PlainModel::prompt_free(Prompt* p) {
    if (p == nullptr) return;
    for (void* d : p->dev) strata::gpu::free(d);
    strata::gpu::free(p->host_block);
    delete p;
}

bool PlainModel::prompt_init(std::string& err) {
    if (prompt_) return true;
    const ModelSpec& s = spec_;
    const auto& lay = k::cpu::expert_layout();
    const int64_t T = opt_.prompt_chunk, N = s.n_embd, NH = s.n_head, HD = s.key_length, NKV = nkv_;
    const int64_t HV = s.ssm_dt_rank, ZV = s.ssm_d_inner, C = 2 * s.ssm_n_group * s.ssm_d_state + ZV;
    const int64_t E = s.n_expert, K = s.n_expert_used, FS = s.n_ff_shexp, FF = s.n_ff_exp;
    if (T < 1 || T > 65535 / std::max<int64_t>(NH, 1)) { err = "plain prompt: the chunk is out of range"; return false; }
    std::unique_ptr<Prompt, void (*)(Prompt*)> p(new Prompt, &PlainModel::prompt_free);
    p->T = T;
    bool ok = true;
    auto dev = [&](auto** ptr, uint64_t bytes) {
        void* d = nullptr;
        if (ok && strata::gpu::alloc_device(&d, (size_t) std::max<uint64_t>(bytes, 256))) {
            p->dev.push_back(d);
            *ptr = (std::remove_reference_t<decltype(*ptr)>) d;
        } else {
            ok = false;
        }
    };
    const uint64_t TK = (uint64_t) (T * K), max_in = (uint64_t) std::max<int64_t>({N, ZV, NH * HD, FS, FF});
    const int64_t qw = (has_gate_ ? 2 : 1) * NH * HD;
    dev(&p->x, T * N * 4); dev(&p->hn, T * N * 4); dev(&p->mix, T * N * 4); dev(&p->x16, T * max_in * 2);
    dev(&p->qkv, T * C * 4); dev(&p->z, T * ZV * 4); dev(&p->ab, T * 2 * HV * 4); dev(&p->gate, T * HV * 4);
    dev(&p->beta, T * HV * 4); dev(&p->h, T * C * 4); dev(&p->y, T * ZV * 4); dev(&p->y16, T * ZV * 2);
    dev(&p->qfull, T * qw * 4); dev(&p->qcur, T * NH * HD * 4); dev(&p->kraw, T * NKV * HD * 4);
    dev(&p->kcur, T * NKV * HD * 4); dev(&p->vcur, T * NKV * HD * 4); dev(&p->attn, T * NH * HD * 4);
    dev(&p->attn32, T * NH * HD * 4);
    dev(&p->steps, T * k::kStepCount * 4); dev(&p->pos_q, T * NH * 4); dev(&p->pos_k, T * NKV * 4);
    dev(&p->rlog, T * E * 4); dev(&p->ids, TK * 4); dev(&p->w, TK * 4); dev(&p->shared, T * N * 4);
    dev(&p->sh_g, (T + 4) * 4); dev(&p->sh_gate, T * FS * 4); dev(&p->sh_up, T * FS * 4);
    dev(&p->xs16, TK * N * 2); dev(&p->gu, TK * 2 * FF * 4); dev(&p->hh, TK * FF * 4); dev(&p->h16, TK * FF * 2);
    dev(&p->dm, TK * N * 4); dev(&p->parts, TK * N * 4); dev(&p->moe_out, T * N * 4);
    dev(&p->src, TK * 4); dev(&p->slot, TK * 4); dev(&p->bounds, (E + 1) * 4);
    p->gu_elems = 2 * FF * N;
    p->d_elems = N * FF;
    dev(&p->dq_gu, (uint64_t) Prompt::kGroup * p->gu_elems * 2);
    dev(&p->dq_d, (uint64_t) Prompt::kGroup * p->d_elems * 2);
    dev(&p->stage, (uint64_t) Prompt::kGroup * lay.max_blob);
    if (!ok) { err = "plain prompt: no VRAM for chunks of " + std::to_string(T) + " tokens"; return false; }
    {   // the pinned host staging, carved from one block
        const uint64_t parts[] = {(uint64_t) T * N * 4, (uint64_t) T * k::kStepCount * 4, (uint64_t) T * (NH + NKV) * 4,
                                  TK * 4, TK * 4, TK * 4, (uint64_t) (E + 1) * 4};
        uint64_t total = 0;
        for (uint64_t b : parts) total += (b + 255) & ~255ull;
        if (!strata::gpu::alloc_host(&p->host_block, total)) { err = "plain prompt: pinned staging"; return false; }
        uint8_t* q = (uint8_t*) p->host_block;
        void* out[7];
        for (int i = 0; i < 7; ++i) { out[i] = q; q += (parts[i] + 255) & ~255ull; }
        p->emb = (float*) out[0];
        p->h_steps = (int32_t*) out[1];
        p->h_pos = (int32_t*) out[2];
        p->h_ids = (int32_t*) out[3];
        p->h_src = (int32_t*) out[4];
        p->h_slot = (int32_t*) out[5];
        p->h_bounds = (int32_t*) out[6];
    }
    // the dequantization scratch of Gemm::native: the largest dense weight, in row slices past 32M values
    if (!p->gemm.init(cs_, (int64_t) 1 << 25, err)) return false;
    prompt_ = std::move(p);
    return true;
}

bool PlainModel::prompt(const int32_t* tokens, int64_t n, int64_t pos, int32_t* out, float* logits, std::string& err) {
    if (opt_.prompt_chunk <= 0 || n <= 0) { err.clear(); return false; }
    if (pos < 0 || pos + n > opt_.max_cells) { err = "plain prompt: past the context"; return false; }
    if (!prompt_init(err)) return false;
    Prompt& P = *prompt_;
    const ModelSpec& s = spec_;
    const auto& lay = k::cpu::expert_layout();
    const int64_t N = s.n_embd, NH = s.n_head, HD = s.key_length, NKV = nkv_, HV = s.ssm_dt_rank, HK = s.ssm_n_group;
    const int64_t ZV = s.ssm_d_inner, SS = s.ssm_d_state, C = 2 * HK * SS + ZV, E = s.n_expert, K = s.n_expert_used;
    const int64_t FS = s.n_ff_shexp, FF = s.n_ff_exp, qw = (has_gate_ ? 2 : 1) * NH * HD;
    const float eps = (float) s.rms_eps;
    const k::QsaShapes qs = [&] {
        k::QsaShapes q = k::qsa_real_shapes();
        q.n_head = NH; q.n_head_kv = NKV; q.head_dim = HD; q.n_rot = s.rope_dims;
        return q;
    }();
    const k::RopeScaling& rs = k::rope_scaling();
    strata::gpu::Stream cs = cs_;
    prefill::Gemm& gm = P.gemm;
    auto f16 = [&](const float* x, int64_t rows, int64_t cols) { prefill::to_f16(x, P.x16, rows * cols, cs); };
    auto proj = [&](const W& w, float* y, int64_t T, int64_t ldy = 0) {
        gm.native(P.x16, w.type, w.data, y, T, w.n_out, w.n_in, ldy);
    };
    const int attn_batch = std::max(1, std::min(opt_.max_t, 8));   // queries an attention launch takes (its scratch)
    try {
        for (int64_t c0 = 0; c0 < n; c0 += P.T) {
            const int64_t T = std::min<int64_t>(P.T, n - c0), p0 = pos + c0;
            // the chunk's inputs: embedding rows, per-token step counts and rotary positions
            for (int64_t t = 0; t < T; ++t) {
                const int32_t tok = tokens[c0 + t];
                if (tok < 0 || tok >= n_vocab_ ||
                    !k::cpu::native_row_to_float(emb_type_, emb_ + (size_t) tok * emb_row_, P.emb + t * N, N)) {
                    err = "plain prompt: a token is out of the vocabulary";
                    return false;
                }
                k::qsa_step_fill(P.h_steps + t * k::kStepCount, p0 + t, qs);
                for (int64_t hd = 0; hd < NH; ++hd) P.h_pos[t * NH + hd] = (int32_t) (p0 + t);
                for (int64_t hd = 0; hd < NKV; ++hd) P.h_pos[T * NH + t * NKV + hd] = (int32_t) (p0 + t);
            }
            if (!strata::gpu::copy_async(P.x, P.emb, (size_t) (T * N) * 4, cs) ||
                !strata::gpu::copy_async(P.steps, P.h_steps, (size_t) (T * k::kStepCount) * 4, cs) ||
                !strata::gpu::copy_async(P.pos_q, P.h_pos, (size_t) (T * NH) * 4, cs) ||
                !strata::gpu::copy_async(P.pos_k, P.h_pos + T * NH, (size_t) (T * NKV) * 4, cs)) {
                err = std::string("plain prompt: staging: ") + strata::gpu::last_error();
                return false;
            }
            for (int64_t l = 0; l < s.n_layer; ++l) {
                const Block& b = blocks_[(size_t) l];
                k::native_qsa_rms_norm_weighted(P.x, b.attn_norm, P.hn, (int) N, (int) T, eps, cs);
                f16(P.hn, T, N);
                if (b.recurrent) {
                    proj(b.qkv, P.qkv, T);
                    proj(b.z, P.z, T);
                    proj(b.alpha, P.ab, T, 2 * HV);
                    proj(b.beta, P.ab + HV, T, 2 * HV);
                    prefill::gdn_gates(P.ab, b.dt, b.a, P.gate, P.beta, T, cs, HK, HV);
                    prefill::gdn_conv(gdn_conv_[(size_t) b.gdn], P.qkv, b.conv, P.h, T, eps, cs, HK, HV);
                    prefill::gdn_recurrence(gdn_state_[(size_t) b.gdn], P.h, P.gate, P.beta, P.z, b.ssm_norm, eps, P.y,
                                            P.y16, T, cs, HK, HV, /*silu=*/true);
                    gm.native(P.y16, b.ssm_out.type, b.ssm_out.data, P.mix, T, b.ssm_out.n_out, b.ssm_out.n_in);
                } else {
                    const int64_t a = b.attn;
                    proj(b.q, P.qfull, T);
                    proj(b.k, P.kraw, T);
                    proj(b.v, P.vcur, T);
                    k::native_qsa_rms_norm_rope(P.qfull, (int) (qw / NH), b.q_norm, P.qcur, (int) (T * NH), (int) HD,
                                                (int) s.rope_dims, eps, rs, P.pos_q, cs);
                    k::native_qsa_rms_norm_rope(P.kraw, (int) HD, b.k_norm, P.kcur, (int) (T * NKV), (int) HD,
                                                (int) s.rope_dims, eps, rs, P.pos_k, cs);
                    for (int64_t t = 0; t < T; ++t)
                        k::kv_append_step(k_pool_[(size_t) a], v_pool_[(size_t) a], page_table_,
                                          P.steps + t * k::kStepCount, P.kcur + t * NKV * HD, P.vcur + t * NKV * HD, qs, cs);
                    k::QsaAttnPools pools;
                    pools.k_pool = k_pool_[(size_t) a];
                    pools.v_pool = v_pool_[(size_t) a];
                    pools.page_table = page_table_;
                    for (int64_t t = 0; t < T; t += attn_batch) {   // every query reads the cells up to its own
                        const int nq = (int) std::min<int64_t>(attn_batch, T - t);
                        k::qsa_dense_attn_batch(P.qcur + t * NH * HD, pools, P.steps + t * k::kStepCount,
                                                opt_.max_cells, qs, attn_scratch_, P.attn + t * NH * HD, nq, cs);
                    }
                    const float* ao = P.attn;
                    if (has_gate_) {
                        k::native_qsa_gate_apply(P.attn, P.qfull, P.attn32, (int) (T * NH), (int) HD, cs);
                        ao = P.attn32;
                    }
                    f16(ao, T, NH * HD);
                    proj(b.o, P.mix, T);
                }
                k::add_inplace(P.x, P.mix, T * N, cs);

                // the MoE
                k::native_qsa_rms_norm_weighted(P.x, b.post_norm, P.hn, (int) N, (int) T, eps, cs);
                for (int64_t t = 0; t < T; ++t)
                    k::f32_gemv_fp32_mmvf(P.hn + t * N, b.router, P.rlog + t * E, N, E, cs);
                k::router_top10(P.rlog, (int) T, (int) E, (int) K, P.ids, P.w, cs);
                f16(P.hn, T, N);
                if (has_shared_) {
                    proj(b.sh_gate, P.sh_gate, T);
                    proj(b.sh_up, P.sh_up, T);
                    for (int64_t t = 0; t < T; ++t)
                        k::f32_gemv_fp32_mmvf(P.hn + t * N, b.sh_gate_inp, P.sh_g + t, N, 1, cs);
                }
                // the routed experts grouped by expert, on the host from this layer's ids
                if (!strata::gpu::copy_async(P.h_ids, P.ids, (size_t) (T * K) * 4, cs) || !strata::gpu::stream_sync(cs)) {
                    err = std::string("plain prompt: the routes: ") + strata::gpu::last_error();
                    return false;
                }
                const int64_t ml = s.moe_index(l);
                std::vector<int32_t> count((size_t) E, 0);
                for (int64_t i = 0; i < T * K; ++i) {
                    const int32_t e = P.h_ids[i];
                    if (e < 0 || e >= E) { err = "plain prompt: a routed expert is out of range"; return false; }
                    ++count[(size_t) e];
                    heat_[(size_t) (ml * E + e)] += 1.0;
                    if (!disp_.usage.empty()) disp_.usage[(size_t) (ml * E + e)] += 1.0f;
                }
                std::vector<int32_t> experts, start((size_t) E + 1, 0);
                for (int64_t e = 0; e < E; ++e) {
                    start[(size_t) e + 1] = start[(size_t) e] + count[(size_t) e];
                    if (count[(size_t) e] > 0) experts.push_back((int32_t) e);
                }
                {
                    std::vector<int32_t> fill(start.begin(), start.end() - 1);
                    for (int64_t i = 0; i < T * K; ++i) {
                        const int32_t e = P.h_ids[i], row = fill[(size_t) e]++;
                        P.h_src[row] = (int32_t) (i / K);   // the token whose row it reads
                        P.h_slot[i] = row;                  // where routing entry i finds its expert's output
                    }
                }
                // the group bounds: the routed rows of the experts in `experts` order, absolute
                int64_t nb = 0;
                for (int32_t e : experts) P.h_bounds[nb++] = start[(size_t) e];
                P.h_bounds[nb] = (int32_t) (T * K);
                if (!strata::gpu::copy_async(P.src, P.h_src, (size_t) (T * K) * 4, cs) ||
                    !strata::gpu::copy_async(P.slot, P.h_slot, (size_t) (T * K) * 4, cs) ||
                    !strata::gpu::copy_async(P.bounds, P.h_bounds, (size_t) (nb + 1) * 4, cs)) {
                    err = std::string("plain prompt: the groups: ") + strata::gpu::last_error();
                    return false;
                }
                prefill::gather_rows16(P.x16, P.src, P.xs16, T * K, N, cs);
                const auto& f = lay.fmt[(size_t) ml];
                const bool transient = src_->transient(ml, experts.empty() ? 0 : experts[0]);
                int64_t max_rows = 0;
                for (int32_t e : experts) max_rows = std::max<int64_t>(max_rows, count[(size_t) e]);
                for (size_t g0 = 0; g0 < experts.size(); g0 += Prompt::kGroup) {
                    const int G = (int) std::min<size_t>(Prompt::kGroup, experts.size() - g0);
                    if (transient && !strata::gpu::stream_sync(cs)) {   // the staging's last readers are done
                        err = std::string("plain prompt: ") + strata::gpu::last_error();
                        return false;
                    }
                    for (int q = 0; q < G; ++q) {   // each expert's weights, then FP16
                        const int32_t e = experts[g0 + (size_t) q];
                        const int32_t sl = res_[(size_t) (ml * E + e)];
                        const uint8_t* blob = nullptr;
                        if (sl >= 0) {
                            blob = cache_.device_slot(sl);
                        } else {
                            uint8_t* dst = P.stage + (size_t) q * lay.max_blob;
                            const uint8_t* hb = src_->blob(ml, e);
                            const bool copied = hb != nullptr &&
                                (transient ? strata::gpu::copy(dst, hb, f.bytes) : strata::gpu::copy_async(dst, hb, f.bytes, cs));
                            if (!copied) { err = "plain prompt: an expert could not be read"; return false; }
                            blob = dst;
                        }
                        k::dequant_f16(f.gu_type, blob, 0, 2 * FF, N, P.dq_gu + (size_t) q * P.gu_elems, cs);
                        k::dequant_f16(f.d_type, blob + f.down_off, 0, N, FF, P.dq_d + (size_t) q * P.d_elems, cs);
                    }
                    const int32_t* hbnd = P.h_bounds + g0;
                    const int32_t* dbnd = P.bounds + g0;
                    const int64_t r0 = hbnd[0], r1 = hbnd[G];
                    if (!prefill::Gemm::f16_groups(P.xs16, P.dq_gu, P.gu_elems, P.gu, hbnd, G, 2 * FF, N, cs))
                        k::xmx_gemm_grouped(P.xs16, P.dq_gu, P.gu_elems, P.gu, dbnd, G, max_rows, 2 * FF, N, cs);
                    // gate rows then up rows (the GGUF order): h = silu(gate) * up, then FP16 for the down product
                    prefill::mmq::swiglu(P.gu + r0 * 2 * FF, P.hh + r0 * FF, r1 - r0, FF, /*interleaved=*/false, cs);
                    prefill::to_f16(P.hh + r0 * FF, P.h16 + r0 * FF, (r1 - r0) * FF, cs);
                    if (!prefill::Gemm::f16_groups(P.h16, P.dq_d, P.d_elems, P.dm, hbnd, G, N, FF, cs))
                        k::xmx_gemm_grouped(P.h16, P.dq_d, P.d_elems, P.dm, dbnd, G, max_rows, N, FF, cs);
                }
                // routing entry i's output into row i, then the weighted sum (and the gated shared expert)
                k::gather_rows(reinterpret_cast<const uint8_t*>(P.dm), N * 4, P.slot, T * K,
                               reinterpret_cast<uint8_t*>(P.parts), cs);
                if (has_shared_) {   // shared = down(silu(gate) * up): the SwiGLU into FP16 (x16 is free again)
                    prefill::swiglu_pair(P.sh_gate, P.sh_up, P.x16, T, cs, FS);
                    proj(b.sh_down, P.shared, T);
                }
                k::native_moe_combine_multi(P.parts, P.w, has_shared_ ? P.shared : nullptr, P.moe_out, N, K, (int) T,
                                            cs, has_shared_ ? P.sh_g : nullptr);
                k::add_inplace(P.x, P.moe_out, T * N, cs);
            }
            if (c0 + T >= n) {   // the last token's logits, by the decode's head
                k::native_qsa_rms_norm_weighted(P.x + (T - 1) * N, out_norm_, P.hn, (int) N, 1, eps, cs);
                k::native_quantize_q8_1(P.hn, xq_, head_.n_in, 1, cs);
                k::native_mmvq(head_.type, head_.data, xq_, logits_, head_.n_in, head_.n_out, 1, cs);
                k::SamplerParams sp;
                sp.greedy = true;
                sp.temperature = 0.0f;
                k::sample_tokens(logits_, 1, (int) n_vocab_, nullptr, 0, sp, m_out_, cs);
            }
        }
    } catch (const std::exception& e) {
        err = std::string("plain prompt: ") + e.what();
        return false;
    }
    if (!strata::gpu::stream_sync(cs)) {
        err = std::string("plain prompt: ") + strata::gpu::last_error();
        return false;
    }
    *out = ((volatile int32_t*) h_out_)[0];
    if (logits != nullptr && !strata::gpu::copy(logits, logits_, (size_t) n_vocab_ * 4)) {
        err = "plain prompt: reading the logits failed";
        return false;
    }
    return true;
}

}  // namespace strata::core
