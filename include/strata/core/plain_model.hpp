// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/core/plain_model.hpp - an MoE model with one residual stream, run from its GGUF.
//
// qwen35moe (Ornith-1.5: Gated DeltaNet layers and full-attention layers, a shared expert) and qwen3moe (llm-jp-4.1:
// full attention in every block, no shared expert) - the models whose block is
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
    /// A ranking of (MoE layer, expert) pairs (expert_cache.hpp's profile format) the VRAM tier is filled from at load,
    /// each layer up to its slots; empty: the tier starts empty and the adaptive rule fills it.
    std::string profile;
    /// The rope scaling, as llama.cpp's options: "yarn" / "linear" / "none" (empty: the file's), the factor (0: the
    /// file's), the trained context (0: the file's, else its context length).  llm-jp-4.1 at 256K: yarn, 4, 65536.
    std::string rope_scaling;
    double rope_scale = 0;
    double yarn_orig_ctx = 0;
    /// The most tokens a prompt chunk takes through the batched products (`prompt`; a shorter prompt, one chunk of its
    /// length); 0: the prompt runs in windows.
    int64_t prompt_chunk = 4096;
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
    /// The prompt: tokens [0, n) at positions pos .. pos + n - 1 in chunks of `prompt_chunk` tokens, every projection a
    /// matrix product and the routed experts grouped by expert, all on the GPU (src/prefill/plain_prompt.cpp, in the
    /// prompt path's library).  `out`: the greedy token after the last; `logits`: its row (or null).  The state is left
    /// where the windows would have left it.
    bool prompt(const int32_t* tokens, int64_t n, int64_t pos, int32_t* out, float* logits, std::string& err);
    /// Runs each dense product the prompt takes once, so the products' library has loaded its kernels before the
    /// first prompt (plain_prompt.cpp); nothing when the prompt runs in windows.
    bool prompt_warm(std::string& err);

    /// The VRAM tier, for the report: slots, slots holding an expert, and the routed entries it served so far.
    int64_t slots() const { return n_slots_; }
    int64_t resident() const;
    int64_t hits() const { return disp_.cache_hits; }
    int64_t misses() const { return disp_.cache_refused; }
    int64_t swaps() const { return swaps_total_; }
    /// Writes what this run learned as a profile (rank_learned_profile: the experts in VRAM now, then by the routing
    /// counted since the load, then by the profile it started from), for the next load's `profile`.
    bool save_profile(const std::string& path, std::string& err) const;

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
    bool fill_slot(int64_t layer, int32_t expert, std::string& err);
    /// The prompt path's buffers for chunks of `T` tokens, carved by prompt_carve: the pinned host block allocated at
    /// load for the longest chunk, the device block borrowed from the tier for each prompt (plain_prompt.cpp uses them).
    struct PromptBufs {
        static constexpr int kGroup = 8;   ///< experts dequantized and multiplied in one grouped product
        int64_t T = 0, gu_elems = 0, d_elems = 0, gemm_elems = 0;
        bool int8 = false;                 ///< every MoE layer's experts through the int8 products (no FP16 buffers)
        float *x = nullptr, *hn = nullptr, *mix = nullptr;
        uint16_t* x16 = nullptr;           ///< T rows of the widest projection input, FP16
        float *qkv = nullptr, *z = nullptr, *ab = nullptr, *gate = nullptr, *beta = nullptr, *h = nullptr, *y = nullptr;
        uint16_t* y16 = nullptr;
        float *qfull = nullptr, *qcur = nullptr, *kraw = nullptr, *kcur = nullptr, *vcur = nullptr, *attn = nullptr,
              *attn32 = nullptr;
        int32_t *steps = nullptr, *pos_q = nullptr, *pos_k = nullptr;
        float *rlog = nullptr, *w = nullptr, *shared = nullptr, *sh_g = nullptr, *sh_gate = nullptr, *sh_up = nullptr;
        int32_t* ids = nullptr;
        uint16_t* xs16 = nullptr;          ///< the routed rows grouped by expert, FP16 [T * k, n_embd]
        float *gu = nullptr, *hh = nullptr, *dm = nullptr, *parts = nullptr, *moe_out = nullptr;
        uint16_t* h16 = nullptr;
        int32_t *src = nullptr, *slot = nullptr, *bounds = nullptr;
        int32_t* entry = nullptr;          ///< grouped row -> routing entry: where the int8 down product writes
        uint16_t *dq_gu = nullptr, *dq_d = nullptr;   ///< kGroup experts' FP16 weights
        uint8_t* stage = nullptr;          ///< 2 kGroup expert blobs copied from RAM (two halves)
        uint8_t *xq = nullptr, *hq = nullptr;   ///< the int8 products' activations: the routed rows, the SwiGLU's
        int32_t* identity = nullptr;       ///< row i is row i (the int8 products' output map)
        const uint8_t** wtab = nullptr;    ///< a layer's experts' blobs, for the int8 products
        int32_t* rbounds = nullptr;        ///< each group's bounds from its first row (the down product's)
        uint16_t* gemm_scratch = nullptr;  ///< Gemm::native's dequantized weight
        // pinned host
        float* emb = nullptr;              ///< T rows of n_embd
        int32_t *h_steps = nullptr, *h_pos = nullptr, *h_ids = nullptr, *h_src = nullptr, *h_slot = nullptr,
                *h_bounds = nullptr, *h_entry = nullptr;
        const uint8_t** h_wtab = nullptr;
        int32_t* h_rbounds = nullptr;
    };
    /// Lays the buffers for chunks of `T` tokens out from `dev` / `host` (null: only counts), returning the device and
    /// host bytes.
    /// Whether every MoE layer's experts take the int8 products on this GPU (iq_mmq; STRATA_PLAIN_MMQ=0: no).
    bool prompt_int8() const;
    void prompt_carve(int64_t T, PromptBufs& b, uint8_t* dev, uint8_t* host, uint64_t& dev_bytes,
                      uint64_t& host_bytes) const;
    PromptBufs pb_;
    void* prompt_host_ = nullptr;
    /// The prompt's device buffers, carved from the tier's last slots for one prompt (prompt_borrow) and given back
    /// after it (prompt_return), with the experts that were in them put back: the tier keeps that memory between
    /// prompts.  The slots go to the layers in turn (slot s is layer s % L's), so the last ones are every layer's.
    bool prompt_borrow(int64_t T, std::string& err);
    bool prompt_chunks(const int32_t* tokens, int64_t n, int64_t pos, int32_t* out, float* logits, std::string& err);
    bool prompt_warm_products(std::string& err);
    bool prompt_return(std::string& err);
    int64_t borrow_first_ = -1;
    std::vector<std::pair<int32_t, int32_t>> borrowed_res_;   // (layer, expert) the borrowed slots held
    struct PromptGemm;   // plain_prompt.cpp: the products' state over pb_.gemm_scratch
    static void prompt_gemm_free(PromptGemm* g);
    PromptGemm& prompt_gemm();
    std::unique_ptr<PromptGemm, void (*)(PromptGemm*)> prompt_gemm_{nullptr, nullptr};
    static void pool_fn(void* user, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k, float* out,
                        int64_t layer);

    PlainOptions opt_;
    ModelSpec spec_;
    std::unique_ptr<GgufModel> model_;
    int64_t n_vocab_ = 0;
    int64_t nkv_ = 0;                ///< the attention blocks' K/V heads (the same in every one)
    bool has_gate_ = false;          ///< the attention's output gate: [q | gate] per head (qwen35moe)
    bool has_shared_ = false;        ///< the gated shared expert (qwen35moe)
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
    std::vector<double> heat_;                    ///< [moe layer][expert] routed entries since the load
    std::vector<std::pair<int32_t, int32_t>> prior_;   ///< the profile the tier was filled from
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
