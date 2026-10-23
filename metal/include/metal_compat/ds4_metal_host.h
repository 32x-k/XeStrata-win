// metal/include/metal_compat/ds4_metal_host.h - the dsv41 kernels that have a Metal dispatch.
//
// GENERATED from metal/src/engine/ds4_host.cpp and ds4_finish.cpp, which are the source of truth: do not hand-edit the
// declaration block, regenerate it (see RIGS/m1002.md).
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::gpu {
struct StreamImpl;
using Stream = StreamImpl*;
}   // namespace strata::gpu

namespace strata::dsv41::metal_host {

// ds4_kernels.cu:1894 finish_kernel<false> - the engine's Finish, as ds4_kernels.cuh declares it
struct FinishIn {
    const float* w;
    float eps;
    int rd, pos, inverse, vpp, pos_step;
    const float* freqs;
    int quant;
    const uint16_t* dst;
    long long dst_stride;
    uint8_t* pdst;
    long long pdst_stride;
};

bool embed(const uint16_t* rows, int T, float* h, int dim, int hc, void*);
bool engram_dequant_tok(const uint8_t* src, size_t per, int cols, int T, int hd, float* out, int fmt);
bool ext_to_ring(const void* ext, void* ring, long long p0, int n, int T, int d, size_t row_bytes, void* backup);
bool finish(float* x, int n_vec, int len, long long stride, const FinishIn& f);
bool gemv_f32(const float* W, const float* x, int n, int k, int T, float* y, int ldy);
bool hc_mix(const float* h, const float* fn, int hcdim, int n_mix, int T, float eps, float* mixes);
bool hc_mix_scale(const float* h, int hcdim, int n_mix, int T, float eps, float* mixes);
bool hc_post(const float* y, float* h, const float* post, const float* comb, int dim, int hc, int T);
bool hc_post_v(const float* y, const float* res, const float* post, const float* comb, int dim, int hc, int T, float* out);
bool hc_pre_norm(const float* h, const float* mix, const float* w, int dim, int hc, int T, float eps, float* x);
bool hc_split(const float* mixes, const float* scale, const float* base, int hc, int iters, float hc_eps, int T, float* pre, float* post, float* comb);
bool init_pre(float* pre, int T, void*);
bool iq_mv(const void* jobs, int nj);
bool moe_combine(void* acc, const void* y, const int* pos, const float* w_, int T, int k, int r0, int r1, int dim);
bool moe_sort(const int* idx, int T, int k, const int* vs, int n_vs, int* bounds, int* row_tok, int* pos);
bool moe_win_group_cpu(const void* plan_ptr, void* grp_ptr, void* grp_start, void* grp_ent, void* n_groups, int T, int k);
bool mx_mv(const void* jobs, int nj, int max_n, size_t job_bytes, int fmt);
bool q8_quant_rows(const float* x, size_t ldx, int rows, int k, int8_t* q, float* d);
bool ring_to_ext(const void* ring, void* ext, long long p0, int n, int d, size_t row_bytes);
bool route_publish(const float* logits, const float* bias, int n, int k, int T, float route_scale, const void* vram, int* idx, float* w, void* plan_ptr, const float* bias_vl, unsigned vl);
bool sgemm_tn(const float* A, const float* B, float* C, int m, int nt, int k, int lda, int ldb, int ldc);
bool swiglu(const float* g, const float* u, int n, float lim, float* h);
bool unpack_sel(const void* ckv, const int* sel, int sel_stride, int gen, long long p0, int t0, int R, int kcap, const int* nsel, int cap, void* out, int T, int nsel_count, size_t row_bytes);

// the entry points, as metal/src/engine/ds4_finish.cpp defines them
bool block_max(const float* score, long long stride, int T, const void* rows, int blk, float* bm, long long bm_stride, int nb_max);
bool compress_win(const float* kv, const float* score, float* kv_state, float* score_state, int ratio, int p0, int T, int d, const float* w, float eps, float* latent);
bool embed(const uint16_t* rows, int T, float* h, int dim, int hc, void*);
bool engram_dequant(const uint8_t* rows, const uint8_t* scales, int n, int hd, float* out, int fmt);
bool engram_dequant_tok(const uint8_t* src, size_t per, int cols, int T, int hd, float* out, int fmt);
bool engram_gate(float* h, const float* kv, const float* w, int hc, int dim, int T, float eps);
bool ext_to_ring(const void* ext, void* ring, long long p0, int n, int T, int d, size_t row_bytes, void* backup);
bool finish(float* x, int n_vec, int len, long long stride, const FinishIn& f);
bool gemv_f16_rows(const uint16_t* W, const float* x, size_t ldx, int n, int k, int T, float alpha, float* y, size_t ldy);
bool gemv_f32(const float* W, const float* x, int n, int k, int T, float* y, int ldy);
bool hc_mean(const float* h, int T, int hc, int dim, float* out, long long out_stride);
bool hc_mix(const float* h, const float* fn, int hcdim, int n_mix, int T, float eps, float* mixes);
bool hc_mix_scale(const float* h, int hcdim, int n_mix, int T, float eps, float* mixes);
bool hc_post(const float* y, float* h, const float* post, const float* comb, int dim, int hc, int T);
bool hc_post_v(const float* y, const float* res, const float* post, const float* comb, int dim, int hc, int T, float* out);
bool hc_pre_norm(const float* h, const float* mix, const float* w, int dim, int hc, int T, float eps, float* x);
bool hc_split(const float* mixes, const float* scale, const float* base, int hc, int iters, float hc_eps, int T, float* pre, float* post, float* comb);
bool index_score(const float* q, const float* wq, const void* kc, int heads, int hd, int T, int n, const int* cand, int cand_stride, int blk, const void* rows, float* out, long long out_stride, int kfmt);
bool init_pre(float* pre, int T, void*);
bool iota_rows(int* out, int T, int n, int stride);
bool iq_dn(const unsigned long long* plan_ptr, const unsigned long long* grp_ptr, const int* grp_start, const int* grp_ent, const int* n_groups, const size_t* tr, int n_embd, int n_ff, const int8_t* hq, const float* hd, float* part, int type, int groups);
bool iq_gu(const unsigned long long* plan_ptr, const unsigned long long* grp_ptr, const int* grp_start, const int* grp_ent, const int* n_groups, const size_t* tr, int n_embd, int n_ff, float limit, const int8_t* xq, const float* xd, float* h, int type, int groups);
bool iq_mv(const void* jobs, int nj);
bool moe_combine(void* acc, const void* y, const int* pos, const float* w_, int T, int k, int r0, int r1, int dim);
bool moe_sort(const int* idx, int T, int k, const int* vs, int n_vs, int* bounds, int* row_tok, int* pos);
bool moe_win_group_cpu(const void* plan_ptr, void* grp_ptr, void* grp_start, void* grp_ent, void* n_groups, int T, int k);
bool mx_combine(const void* plan_ptr, const void* plan_w, const void* part, const void* cpu_part, size_t cpu_stride, const int* cpu_flag, const void* sh_out, void* out, size_t out_stride, int T, int k, int n_embd);
bool mx_mv(const void* jobs, int nj, int max_n, size_t job_bytes, int fmt);
bool q8_quant_rows(const float* x, size_t ldx, int rows, int k, int8_t* q, float* d);
bool ring_to_ext(const void* ring, void* ext, long long p0, int n, int d, size_t row_bytes);
bool route_publish(const float* logits, const float* bias, int n, int k, int T, float route_scale, const void* vram, int* idx, float* w, void* plan_ptr, const float* bias_vl, unsigned vl);
bool sample_greedy(const float* logits, int n_tokens, int n_vocab, const int* history, int history_len, int penalty_last_n, float penalty_repeat, float penalty_freq, float penalty_present, const float* logit_bias, int* out);
bool sgemm_tn(const float* A, const float* B, float* C, int m, int nt, int k, int lda, int ldb, int ldc);
bool swiglu(const float* g, const float* u, int n, float lim, float* h);
bool topk(const float* vals, long long stride, int T, const void* rows, const int* cand, int cand_stride, int blk, int* out, int out_stride);
bool unpack_sel(const void* ckv, const int* sel, int sel_stride, int gen, long long p0, int t0, int R, int kcap, const int* nsel, int cap, void* out, int T, int nsel_count, size_t row_bytes);

}  // namespace strata::dsv41::metal_host
