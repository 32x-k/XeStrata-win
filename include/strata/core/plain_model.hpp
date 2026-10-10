// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/core/plain_model.hpp - an MoE model with one residual stream, run from its GGUF.
//
// qwen35moe (Ornith-1.5: Gated DeltaNet layers and full-attention layers, a shared expert) - the models whose block is
//
//     x += mixer(rms_norm(x))            the mixer a GDN layer or a gated, partly rotated full attention
//     x += moe(rms_norm(x))              routed experts (softmax top-k, renormalized) + the gated shared expert
//
// and whose head is rms_norm, then the output projection.  Qwen3.8 (four residual streams, the QSA indexer, PLE) keeps
// its own engine; GLM-5.3 will have its own.
//
// The dense weights are uploaded once in their GGUF form and read by the native kernels.  The routed experts go
// through the engine's expert tiers, as Qwen3.8's do: a VRAM cache of slots (expert_cache) refilled between windows by
// the adaptive rule, the GGUF read in place (FileExpertSource: the page cache, the drive), the CPU pool for what the
// VRAM does not hold, all in one window protocol (MoeWindow).  A window is 1..max_t tokens at consecutive positions,
// recorded once per size as a graph.
#pragma once

#include "strata/artifact/model_spec.hpp"
#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/gpu.hpp"
#include "strata/core/moe_window.hpp"
#include "strata/kernels/cpu/pool.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata {
class GgufModel;
}

namespace strata::core {

struct PlainOptions {
    std::string gguf;               ///< any shard of the model
    int64_t max_cells = 4096;       ///< the context the K/V cache holds
    int max_t = 8;                  ///< the largest window
    int threads = 0;                ///< CPU pool workers (0: every physical core but the first)
    int64_t reserve_mib = 768;      ///< VRAM left free after the expert slots
    int adapt_every = 4;            ///< windows between adaptive rounds (0: off)
    int adapt_swaps = 96;           ///< the most experts one round moves into the VRAM tier
    float adapt_decay = 0.7f;       ///< the routing counts are multiplied by this after each round
    /// The experts read from the GGUF in place (the page cache, the drive) instead of loaded into pinned RAM: the
    /// low-RAM mode, taken by itself when the RAM available is less than the experts.  It has no PCIe share.
    bool mmap_experts = false;
    bool pcie_split = true;         ///< split the missed experts between the CPU and PCIe by measured times
};

class PlainModel {
public:
    PlainModel();
    ~PlainModel();
    PlainModel(const PlainModel&) = delete;
    PlainModel& operator=(const PlainModel&) = delete;

    /// Reads the model's shape, uploads its dense weights, opens its experts and sizes the VRAM tier from what the
    /// device has left.  False, with the reason, for a model this engine does not run.
    bool load(const PlainOptions& opt, std::string& err);
    const ModelSpec& spec() const { return spec_; }
    int64_t vocab() const { return n_vocab_; }

    /// Zeroes the recurrent states and the K/V cache: the next window starts a sequence at position 0.
    bool reset(std::string& err);
    /// Tokens [0, T) at positions pos .. pos + T - 1 (T <= max_t): out[t] is the greedy token after token t, and with
    /// `logits` the T rows of n_vocab logits.  Positions must follow the previous window's.
    bool forward(const int32_t* tokens, int T, int64_t pos, int32_t* out, float* logits, std::string& err);

    /// The VRAM tier, for the report: slots, slots holding an expert, and the routed entries it served so far.
    int64_t slots() const { return n_slots_; }
    int64_t resident() const;
    int64_t hits() const { return disp_.cache_hits; }
    int64_t misses() const { return disp_.cache_refused; }
    int64_t swaps() const { return swaps_total_; }

private:
    struct W {   // one 2-D projection in VRAM, in its GGUF blocks (or the engine's row order for Q6_K)
        const void* data = nullptr;
        int type = -1;
        int n_in = 0, n_out = 0;
    };
    struct Block {
        bool recurrent = false;
        int64_t gdn = -1, attn = -1;    ///< index among the GDN / attention blocks
        const float *attn_norm = nullptr, *post_norm = nullptr;
        W qkv, z, alpha, beta, ssm_out;                       // GDN
        const float *conv = nullptr, *dt = nullptr, *a = nullptr, *ssm_norm = nullptr;
        W q, k, v, o;                                         // attention
        const float *q_norm = nullptr, *k_norm = nullptr;
        const float* router = nullptr;                        // F32 [n_embd, n_expert]
        const float* sh_gate_inp = nullptr;                   // F32 [n_embd]
        W sh_gate, sh_up, sh_down;
    };

    bool upload(const GgufModel& m, const std::string& name, W& w, std::string& err);
    bool upload_f32(const GgufModel& m, const std::string& name, uint64_t elements, const float*& out, std::string& err);
    bool open_experts(std::string& err);
    bool record(int T, std::string& err);
    bool record_window(int T, strata::gpu::Stream cs, std::string& err);
    bool adapt(std::string& err);
    static void pool_fn(void* user, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k, float* out,
                        int64_t layer);

    PlainOptions opt_;
    ModelSpec spec_;
    std::unique_ptr<GgufModel> model_;
    int64_t n_vocab_ = 0;
    int64_t nkv_ = 0;                ///< the attention blocks' K/V heads (the same in every one)
    int emb_type_ = -1;
    const uint8_t* emb_ = nullptr;   ///< token_embd in the GGUF mapping (rows dequantized on the host)
    size_t emb_row_ = 0;
    W head_;
    const float* out_norm_ = nullptr;
    std::vector<Block> blocks_;
    std::vector<void*> allocs_;      ///< every device allocation, freed in the destructor

    // the expert tiers
    std::unique_ptr<ExpertSource> src_;
    PcieSplitTimes split_;
    std::unique_ptr<strata::kernels::cpu::ExpertPool> pool_;
    ExpertCache cache_;
    ExpertDispatch disp_;
    MoeWindow win_;
    std::vector<int32_t> res_;                    ///< [moe layer][expert] slot or -1
    std::vector<std::vector<int32_t>> free_;      ///< per moe layer, its free slots
    int64_t n_slots_ = 0;
    int64_t windows_ = 0, swaps_total_ = 0;
    uint8_t* fill_tmp_ = nullptr;                 ///< pinned staging for a slot fill

    // per-sequence device state
    std::vector<float*> gdn_state_, gdn_conv_;   ///< per GDN block
    std::vector<uint16_t*> k_pool_, v_pool_;      ///< per attention block
    int32_t* page_table_ = nullptr;
    int64_t n_pages_ = 0;

    // the window's buffers (device unless noted)
    strata::gpu::Stream cs_ = nullptr;
    std::vector<strata::gpu::Graph*> exec_;                      ///< per T: the whole window
    std::vector<std::vector<strata::gpu::Graph*>> segs_;         ///< per T: the segments (MoeWindow::segmented)
    float *h_emb_ = nullptr, *m_emb_ = nullptr;                  ///< mapped: T rows of n_embd
    int32_t *h_step_ = nullptr, *m_step_ = nullptr;              ///< mapped: T x kStepCount
    int32_t *h_pos_ = nullptr, *m_pos_ = nullptr;                ///< mapped: T x n_head, then T x n_head_kv
    int32_t *h_out_ = nullptr, *m_out_ = nullptr;                ///< mapped: T greedy tokens
    int32_t *step_ = nullptr, *pos_ = nullptr;
    float *x_ = nullptr, *hn_ = nullptr, *mix_ = nullptr;
    uint8_t* xq_ = nullptr;
    float *qkv_ = nullptr, *craw_ = nullptr, *hc_ = nullptr, *z_ = nullptr, *alpha_ = nullptr, *beta_ = nullptr,
          *gate_ = nullptr, *o_ = nullptr, *y_ = nullptr;
    float *qfull_ = nullptr, *qcur_ = nullptr, *kraw_ = nullptr, *kcur_ = nullptr, *vcur_ = nullptr, *attn_ = nullptr,
          *attn32_ = nullptr, *attn_scratch_ = nullptr;
    float *rlog_ = nullptr, *w_ = nullptr, *shared_ = nullptr, *sh_g_ = nullptr, *sh_gate_ = nullptr, *sh_up_ = nullptr,
          *moe_out_ = nullptr, *logits_ = nullptr;
    int32_t* ids_ = nullptr;
    int64_t attn_scratch_floats_ = 0;
};

}  // namespace strata::core
