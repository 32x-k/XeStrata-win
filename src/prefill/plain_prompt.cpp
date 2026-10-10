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
#include "strata/kernels/prompt_attn.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/rope_scaling.hpp"
#include "strata/kernels/router_top10.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/xmx_gemm.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "strata/prefill/kernels.hpp"
#include "strata/prefill/moe_mmq.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <set>
#include <tuple>
#include <vector>

namespace strata::core {

namespace k = strata::kernels;

/// The products' state: Gemm over the dequantization scratch carved with the other buffers.
struct PlainModel::PromptGemm {
    prefill::Gemm gemm;
    prefill::mmq::Context mmq;
    bool identity = false;   // pb_.identity written
    // the int8 products' copies from RAM on a stream of their own, into the staging's two halves in turn: ready[h]
    // when half h holds its group, used[h] when that group's products are done with it
    strata::gpu::Stream xs = nullptr;
    strata::gpu::Event *ready[2] = {}, *used[2] = {};
    bool used_live[2] = {};
    ~PromptGemm() {
        if (xs != nullptr) strata::gpu::stream_sync(xs);
        for (int h = 0; h < 2; ++h) {
            if (ready[h] != nullptr) strata::gpu::event_destroy(ready[h]);
            if (used[h] != nullptr) strata::gpu::event_destroy(used[h]);
        }
        if (xs != nullptr) strata::gpu::stream_destroy(xs);
    }
    bool overlap() const { return xs != nullptr && ready[0] && ready[1] && used[0] && used[1]; }
};

void PlainModel::prompt_gemm_free(PromptGemm* g) { delete g; }

PlainModel::PromptGemm& PlainModel::prompt_gemm() {
    if (!prompt_gemm_) {
        prompt_gemm_ = std::unique_ptr<PromptGemm, void (*)(PromptGemm*)>(new PromptGemm, &PlainModel::prompt_gemm_free);
        prompt_gemm_->gemm.init_external(cs_, pb_.gemm_scratch, pb_.gemm_elems);
        PromptGemm& g = *prompt_gemm_;
        g.xs = strata::gpu::stream_create();
        for (int h = 0; h < 2; ++h) {
            if (!strata::gpu::event_create(&g.ready[h])) g.ready[h] = nullptr;
            if (!strata::gpu::event_create(&g.used[h])) g.used[h] = nullptr;
        }
        // without them the copies stay on the compute stream, in order with the products
    }
    return *prompt_gemm_;
}

bool PlainModel::prompt_warm(std::string& err) {
    if (opt_.prompt_chunk <= 0 || prompt_dev_ == nullptr) return true;
    // A library's first product of a kind loads its kernels for it, inside whichever prompt comes first (rocBLAS on
    // the RX 9060 XT: 0.5 s for the first product, 0.9 s more for the first of 32 rows): each dense shape the prompt
    // takes, once, on a chunk of zeros into the buffers the prompt overwrites.
    PromptBufs& P = pb_;
    prefill::Gemm& gm = prompt_gemm().gemm;
    const int64_t T = P.T, HV = spec_.ssm_dt_rank;
    std::set<std::tuple<int, int, int, int64_t>> seen;
    auto warm = [&](const W& w, float* y, int64_t ldy = 0) {
        if (w.data != nullptr && seen.insert({w.type, w.n_out, w.n_in, ldy}).second)
            gm.native(P.x16, w.type, w.data, y, T, w.n_out, w.n_in, ldy);
    };
    try {
        if (!strata::gpu::memset_async(P.x16, 0, (size_t) (T * spec_.n_embd) * 2, cs_)) {
            err = std::string("plain prompt: warming: ") + strata::gpu::last_error();
            return false;
        }
        for (const Block& b : blocks_) {
            if (b.recurrent) {
                warm(b.qkv, P.qkv);
                warm(b.z, P.z);
                warm(b.alpha, P.ab, 2 * HV);
                warm(b.beta, P.ab + HV, 2 * HV);
                warm(b.ssm_out, P.mix);
            } else {
                warm(b.q, P.qfull);
                warm(b.k, P.kraw);
                warm(b.v, P.vcur);
                warm(b.o, P.mix);
            }
            if (has_shared_) {
                warm(b.sh_gate, P.sh_gate);
                warm(b.sh_up, P.sh_up);
                warm(b.sh_down, P.shared);
            }
        }
    } catch (const std::exception& e) {
        err = std::string("plain prompt: warming: ") + e.what();
        return false;
    }
    if (!strata::gpu::stream_sync(cs_)) {
        err = std::string("plain prompt: warming: ") + strata::gpu::last_error();
        return false;
    }
    return true;
}

bool PlainModel::prompt(const int32_t* tokens, int64_t n, int64_t pos, int32_t* out, float* logits, std::string& err) {
    if (opt_.prompt_chunk <= 0 || n <= 0 || prompt_dev_ == nullptr) { err.clear(); return false; }
    if (pos < 0 || pos + n > opt_.max_cells) { err = "plain prompt: past the context"; return false; }
    PromptBufs& P = pb_;
    const ModelSpec& s = spec_;
    const auto& lay = k::cpu::expert_layout();
    const int64_t N = s.n_embd, NH = s.n_head, HD = s.key_length, NKV = nkv_, HV = s.ssm_dt_rank, HK = s.ssm_n_group;
    const int64_t E = s.n_expert, K = s.n_expert_used;
    const int64_t FS = s.n_ff_shexp, FF = s.n_ff_exp, qw = (has_gate_ ? 2 : 1) * NH * HD;
    const float eps = (float) s.rms_eps;
    const k::QsaShapes qs = [&] {
        k::QsaShapes q = k::qsa_real_shapes();
        q.n_head = NH; q.n_head_kv = NKV; q.head_dim = HD; q.n_rot = s.rope_dims;
        return q;
    }();
    const k::RopeScaling& rs = k::rope_scaling();
    strata::gpu::Stream cs = cs_;
    prefill::Gemm& gm = prompt_gemm().gemm;
    auto f16 = [&](const float* x, int64_t rows, int64_t cols) { prefill::to_f16(x, P.x16, rows * cols, cs); };
    auto proj = [&](const W& w, float* y, int64_t T, int64_t ldy = 0) {
        gm.native(P.x16, w.type, w.data, y, T, w.n_out, w.n_in, ldy);
    };
    // the routed experts in their quantized blocks (int8 products, iq_mmq: on the matrix engines, else by dp4a) where
    // every layer's types are theirs (prompt_int8, decided with the buffers), else dequantized to FP16
    const bool use_mmq = P.int8;
    static const bool prompt_attn_on = [] {   // STRATA_PLAIN_PROMPT_ATTN=0: the decode's attention kernel
        const char* v = std::getenv("STRATA_PLAIN_PROMPT_ATTN");
        return v == nullptr || std::strtol(v, nullptr, 10) != 0;
    }();
    PromptGemm& pg = prompt_gemm();
    if (use_mmq && !pg.identity) {
        prefill::mmq::iota(P.identity, P.T * K, cs);
        pg.identity = true;
    }
    // STRATA_PROMPT_PROFILE=1: the time of each stage, the stream synchronized after each (slower; for finding where it
    // goes): 0 the GDN layers, 4 the attention layers, 1 the router and the host's grouping, 2 the experts, 3 the shared
    // expert and the combine
    static const bool profile = std::getenv("STRATA_PROMPT_PROFILE") != nullptr;
    double stage_ms[5] = {};
    auto mark = [&](int stage, std::chrono::steady_clock::time_point& t) {
        if (!profile) return;
        strata::gpu::stream_sync(cs);
        const auto now = std::chrono::steady_clock::now();
        stage_ms[stage] += std::chrono::duration<double, std::milli>(now - t).count();
        t = now;
    };
    auto t_mark = std::chrono::steady_clock::now();
    try {
        // the chunk's routes, every MoE layer's, for the adaptive tier's counts (below, after the chunk)
        const int64_t L = s.n_moe_layers();
        std::vector<int32_t> routes(disp_.usage.empty() ? 0 : (size_t) (L * P.T * K));
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
                    // the norm and RoPE take at most 65535 rows a launch: the chunk's heads in token slices
                    auto norm_rope = [&](const float* in, int64_t in_row, const float* gamma, float* out, int64_t heads,
                                         const int32_t* pos) {
                        const int64_t per = 65535 / heads;
                        for (int64_t t0 = 0; t0 < T; t0 += per)
                            k::native_qsa_rms_norm_rope(in + t0 * heads * in_row, (int) in_row, gamma, out + t0 * heads * HD,
                                                        (int) (std::min(per, T - t0) * heads), (int) HD,
                                                        (int) s.rope_dims, eps, rs, pos + t0 * heads, cs);
                    };
                    norm_rope(P.qfull, qw / NH, b.q_norm, P.qcur, NH, P.pos_q);
                    norm_rope(P.kraw, HD, b.k_norm, P.kcur, NKV, P.pos_k);
                    k::kv_append_steps(k_pool_[(size_t) a], v_pool_[(size_t) a], page_table_, P.steps, k::kStepCount,
                                       P.kcur, P.vcur, (int) (NKV * HD), (int) T, qs, cs);
                    k::QsaAttnPools pools;
                    pools.k_pool = k_pool_[(size_t) a];
                    pools.v_pool = v_pool_[(size_t) a];
                    pools.page_table = page_table_;
                    // the prompt attention (prompt_attn: K and V read once a block of queries), else the decode's
                    // kernel: every query reads the cells up to its own; the grid as wide as the chunk's last context (not
                    // the cache's capacity: no graph holds it), as many queries a launch as the scratch takes
                    const int64_t cap = std::min<int64_t>(opt_.max_cells, p0 + T);
                    const int64_t per_q = (int64_t) k::qsa_decode_attn_scratch_floats(cap, qs);
                    const int64_t nq_max = std::max<int64_t>(1, std::min<int64_t>((int64_t) opt_.max_t * attn_scratch_floats_ / per_q, 4096));
                    const bool batched = prompt_attn_on &&
                                         k::prompt_attn(P.qcur, k_pool_[(size_t) a], v_pool_[(size_t) a], page_table_, p0,
                                                        T, qs, P.attn, cs);
                    for (int64_t t = 0; !batched && t < T; t += nq_max) {
                        const int nq = (int) std::min<int64_t>(nq_max, T - t);
                        k::qsa_dense_attn_batch(P.qcur + t * NH * HD, pools, P.steps + t * k::kStepCount, cap, qs,
                                                attn_scratch_, P.attn + t * NH * HD, nq, cs);
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
                mark(b.recurrent ? 0 : 4, t_mark);

                // the MoE
                k::native_qsa_rms_norm_weighted(P.x, b.post_norm, P.hn, (int) N, (int) T, eps, cs);
                k::f32_gemm_rows(P.hn, N, b.router, P.rlog, E, N, E, T, cs);
                k::router_top10(P.rlog, (int) T, (int) E, (int) K, P.ids, P.w, cs);
                f16(P.hn, T, N);
                if (has_shared_) {
                    proj(b.sh_gate, P.sh_gate, T);
                    proj(b.sh_up, P.sh_up, T);
                    k::f32_gemm_rows(P.hn, N, b.sh_gate_inp, P.sh_g, 1, N, 1, T, cs);
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
                }
                if (!routes.empty()) std::copy(P.h_ids, P.h_ids + T * K, routes.begin() + ml * P.T * K);
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
                        P.h_entry[row] = (int32_t) i;       // the int8 down product's: the row it writes
                    }
                }
                // the group bounds: the routed rows of the experts in `experts` order, absolute
                int64_t nb = 0;
                for (int32_t e : experts) P.h_bounds[nb++] = start[(size_t) e];
                P.h_bounds[nb] = (int32_t) (T * K);
                if (!strata::gpu::copy_async(P.src, P.h_src, (size_t) (T * K) * 4, cs) ||
                    !(use_mmq ? strata::gpu::copy_async(P.entry, P.h_entry, (size_t) (T * K) * 4, cs)
                              : strata::gpu::copy_async(P.slot, P.h_slot, (size_t) (T * K) * 4, cs)) ||
                    !strata::gpu::copy_async(P.bounds, P.h_bounds, (size_t) (nb + 1) * 4, cs)) {
                    err = std::string("plain prompt: the groups: ") + strata::gpu::last_error();
                    return false;
                }
                const auto& f = lay.fmt[(size_t) ml];
                if (use_mmq) prefill::mmq::quantize(P.hn, P.src, P.xq, f.gu_type, N, N, T * K, cs);
                else prefill::gather_rows16(P.x16, P.src, P.xs16, T * K, N, cs);
                mark(1, t_mark);
                const bool transient = src_->transient(ml, experts.empty() ? 0 : experts[0]);
                int64_t max_rows = 0;
                for (int32_t e : experts) max_rows = std::max<int64_t>(max_rows, count[(size_t) e]);
                // int8: as many experts a launch as the staging lets (those in VRAM read where they lie, kGroup from
                // RAM at most), gate/up, the SwiGLU, its rows to int8, down
                for (size_t g0 = 0, gi = 0; use_mmq && g0 < experts.size(); ++gi) {
                    if (transient && !strata::gpu::stream_sync(cs)) {   // the staging's last readers are done
                        err = std::string("plain prompt: ") + strata::gpu::last_error();
                        return false;
                    }
                    size_t g1 = g0;
                    int staged = 0;
                    int64_t mr = 0;
                    // the half of the staging this group takes; its copies wait for the products two groups back
                    const bool ovl = !transient && pg.overlap();
                    const int h = (int) (gi % 2);
                    strata::gpu::Stream ccs = ovl ? pg.xs : cs;
                    if (ovl && pg.used_live[h] && !strata::gpu::stream_wait_event(ccs, pg.used[h])) {
                        err = std::string("plain prompt: ") + strata::gpu::last_error();
                        return false;
                    }
                    for (; g1 < experts.size(); ++g1) {
                        const int32_t e = experts[g1];
                        const int32_t sl = res_[(size_t) (ml * E + e)];
                        const uint8_t* blob = nullptr;
                        if (sl >= 0) {
                            blob = cache_.device_slot(sl);
                        } else {
                            if (staged == PromptBufs::kGroup) break;
                            uint8_t* dst = P.stage + (size_t) ((ovl ? h * PromptBufs::kGroup : 0) + staged++) * lay.max_blob;
                            const uint8_t* hb = src_->blob(ml, e);
                            const bool copied = hb != nullptr &&
                                (transient ? strata::gpu::copy(dst, hb, f.bytes) : strata::gpu::copy_async(dst, hb, f.bytes, ccs));
                            if (!copied) { err = "plain prompt: an expert could not be read"; return false; }
                            blob = dst;
                        }
                        P.h_wtab[g1] = blob;
                        mr = std::max<int64_t>(mr, count[(size_t) e]);
                    }
                    if (ovl && (!strata::gpu::event_record(pg.ready[h], ccs) || !strata::gpu::stream_wait_event(cs, pg.ready[h]))) {
                        err = std::string("plain prompt: ") + strata::gpu::last_error();
                        return false;
                    }
                    const int G = (int) (g1 - g0);
                    const int32_t r0 = P.h_bounds[g0], nr = P.h_bounds[g1] - r0;
                    int32_t* hrb = P.h_rbounds + g0 + gi;
                    for (int q = 0; q <= G; ++q) hrb[q] = P.h_bounds[g0 + (size_t) q] - r0;
                    if (!strata::gpu::copy_async(static_cast<void*>(P.wtab + g0), static_cast<const void*>(P.h_wtab + g0),
                                                 (size_t) G * sizeof(void*), cs) ||
                        !strata::gpu::copy_async(P.rbounds + g0 + gi, hrb, (size_t) (G + 1) * 4, cs)) {
                        err = std::string("plain prompt: the groups: ") + strata::gpu::last_error();
                        return false;
                    }
                    prefill::mmq::Product gu;
                    gu.type = f.gu_type; gu.w_rows = 2 * FF; gu.w_cols = N; gu.n = G; gu.xq = P.xq;
                    gu.bounds = P.bounds + g0; gu.ids = P.identity; gu.total_rows = T * K; gu.max_rows = mr;
                    gu.dst = P.gu; gu.ld_dst = 2 * FF; gu.wtab = P.wtab + g0; gu.woff = 0;
                    pg.mmq.run(gu, cs);
                    prefill::mmq::swiglu(P.gu + (int64_t) r0 * 2 * FF, P.hh + (int64_t) r0 * FF, nr, FF,
                                         /*interleaved=*/false, cs);
                    prefill::mmq::quantize(P.hh + (int64_t) r0 * FF, nullptr, P.hq, f.d_type, FF, FF, nr, cs);
                    prefill::mmq::Product dn;
                    dn.type = f.d_type; dn.w_rows = N; dn.w_cols = FF; dn.n = G; dn.xq = P.hq;
                    dn.bounds = P.rbounds + g0 + gi; dn.ids = P.entry + r0; dn.total_rows = nr; dn.max_rows = mr;
                    dn.dst = P.parts; dn.ld_dst = N; dn.wtab = P.wtab + g0; dn.woff = f.down_off;   // routing order
                    pg.mmq.run(dn, cs);
                    if (ovl) {
                        if (!strata::gpu::event_record(pg.used[h], cs)) {
                            err = std::string("plain prompt: ") + strata::gpu::last_error();
                            return false;
                        }
                        pg.used_live[h] = true;
                    }
                    g0 = g1;
                }
                for (size_t g0 = 0; !use_mmq && g0 < experts.size(); g0 += PromptBufs::kGroup) {
                    const int G = (int) std::min<size_t>(PromptBufs::kGroup, experts.size() - g0);
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
                mark(2, t_mark);
                // routing entry i's output into row i, then the weighted sum (and the gated shared expert)
                if (!use_mmq)
                    k::gather_rows(reinterpret_cast<const uint8_t*>(P.dm), N * 4, P.slot, T * K,
                                   reinterpret_cast<uint8_t*>(P.parts), cs);
                if (has_shared_) {   // shared = down(silu(gate) * up): the SwiGLU into FP16 (x16 is free again)
                    prefill::swiglu_pair(P.sh_gate, P.sh_up, P.x16, T, cs, FS);
                    proj(b.sh_down, P.shared, T);
                }
                k::native_moe_combine_multi(P.parts, P.w, has_shared_ ? P.shared : nullptr, P.moe_out, N, K, (int) T,
                                            cs, has_shared_ ? P.sh_g : nullptr);
                k::add_inplace(P.x, P.moe_out, T * N, cs);
                mark(3, t_mark);
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
            if (!routes.empty()) {
                // each window's counts as the windows would leave them at the prompt's end: times the decay of every
                // round that would come after it; the rounds themselves are left to the decode (their swaps would
                // cost the prompt more than they save)
                const int64_t W = opt_.max_t, last = (n - 1) / W;
                for (int64_t t0 = 0; t0 < T; t0 += W) {
                    const int64_t win = (c0 + t0) / W;
                    const float wgt = std::pow(opt_.adapt_decay, (float) ((windows_ + last + 1) / opt_.adapt_every -
                                                                          (windows_ + win) / opt_.adapt_every));
                    for (int64_t ml = 0; ml < L; ++ml)
                        for (int64_t i = t0 * K; i < std::min(T, t0 + W) * K; ++i)
                            disp_.usage[(size_t) (ml * E + routes[(size_t) (ml * P.T * K + i)])] += wgt;
                }
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
    windows_ += (n + opt_.max_t - 1) / opt_.max_t;
    if (profile)
        std::fprintf(stderr, "strata plain: prompt of %lld tokens: GDN %.1f ms, attention %.1f, router and grouping %.1f, "
                             "experts %.1f, shared and combine %.1f\n", (long long) n, stage_ms[0], stage_ms[4], stage_ms[1],
                     stage_ms[2], stage_ms[3]);
    *out = ((volatile int32_t*) h_out_)[0];
    if (logits != nullptr && !strata::gpu::copy(logits, logits_, (size_t) n_vocab_ * 4)) {
        err = "plain prompt: reading the logits failed";
        return false;
    }
    return true;
}

}  // namespace strata::core
