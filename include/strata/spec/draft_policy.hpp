// include/strata/spec/draft_policy.hpp - per verify round: the MTP's window, or a lookup (suffix) window?
//
// The suffix drafter (prompt lookup) proposes the tokens that followed an earlier repeat of the context. Taken
// whenever it proposes more than the MTP, it lost 2-8% on ordinary text: a long lookup window costs much more to
// verify than the MTP's usual 3-4 tokens, and it is only worth that when enough of it is accepted. llama.cpp's
// lookup decoding answers the same problem with confidence thresholds on its n-gram statistics; here the policy
// learns both sides online and compares expected committed tokens per millisecond:
//
//   MTP     E = the mean tokens a window of that size has committed (EMA), at the measured cost of that size
//   lookup  E(k) = 1 + q + q^2 + ... + q^k for k <= the proposal, q = the acceptance rate of lookup drafts whose
//           match was about as long (4 buckets of match length, decayed counts), at the measured cost of k + 1
//
// and takes the lookup window only when its best E/cost beats the MTP's by `margin`. Costs are the measured round
// times per window size (EMA; sizes not seen yet are scaled from seen ones by a prior shape), so the policy adapts
// to the machine and the context length. It only chooses which drafts to verify: the output is unchanged.
#pragma once

#include <array>

namespace strata::spec {

class DraftPolicy {
public:
    static constexpr int kMaxT = 8;
    static constexpr int kBuckets = 4;

    explicit DraftPolicy(int max_t, double margin = 0.03);

    struct Pick {
        bool lookup = false;
        int t = 1;                      // window size (1 + drafts)
    };
    /// `t_mtp`: the MTP's window; `lookup_k`: the lookup proposal's length (0 = none); `match`: its match length.
    Pick choose(int t_mtp, int lookup_k, int match) const;
    /// After the round: the window it used, the drafts accepted, and the round's time (verify + commit + draft).
    void observe(bool lookup, int t, int accepted, int match, double round_ms);

    /// --lookup-chain: how many of `k_avail` lookup tokens to append after an MTP window of `t_mtp` tokens whose
    /// drafts all hold with probability `p_mtp` (estimated from the draft layer's own probabilities): the k with the
    /// best expected tokens per ms, E(t_mtp) + p_mtp (c + c^2 + .. + c^k) at the cost of t_mtp + k, if it beats the MTP
    /// window alone by `margin`; c = the acceptance of chained drafts by match length (their own counts). 0 = none.
    int chain(int t_mtp, double p_mtp, int k_avail, int match) const;
    /// After a chained round: the MTP window's size, the chained tokens, the window's accepted drafts, its time.
    void observe_chain(int t_mtp, int k, int accepted, int match, double round_ms);
    double chain_rate(int match) const;
    double lookup_rate(int match) const;   // current q for a match length
    double cost_ms(int t) const;           // measured or scaled round time of a window of t tokens

    /// COST-AWARE MTP WINDOW (STRATA_SPEC_COST=1 / --spec-cost): the window of 1..cap rows with the best expected committed
    /// tokens per ms, from the MTP chain's draft probabilities p[0..n) (n = drafts computed; the chain is drafted in full):
    ///   E(T) = 1 + q_1 + .. + q_{T-1},  q_j = prod_{i<=j} a(p_i),  a() = the acceptance measured for drafts of that probability
    ///   cost(T) = the measured round time of that window size (cost_ms, kept non-decreasing)
    /// A wider window is taken only if its E/cost beats the narrower best by `margin`.  Only chooses which rows are verified.
    int best_rows(const float* p, int n, int cap) const;
    /// After a round of the MTP window T whose drafts were p[0..T-1): `accepted` of them held.  Teaches a().
    void observe_probs(const float* p, int n_window_drafts, int accepted);
    double draft_rate(double p) const;     // acceptance of an MTP draft that came with probability p

private:
    static int bucket(int match);
    double mtp_tokens(int t) const;
    bool stale(int t) const;               // a size not measured for a long stretch of rounds

    int max_t_;
    double margin_;
    std::array<double, kMaxT + 1> cost_{}, cost_n_{};      // round ms by window size
    std::array<double, kMaxT + 1> mtp_tok_{}, mtp_n_{};    // tokens committed by MTP windows of that size
    std::array<double, kBuckets> ok_{}, bad_{};            // lookup drafts accepted / windows cut short, decayed
    std::array<double, kBuckets> cok_{}, cbad_{};          // chained lookup drafts, the same (reached rounds only)
    double rounds_ = 0;                                    // rounds observed so far
    static constexpr int kPBuckets = 8;
    std::array<double, kPBuckets> pa_ok_{}, pa_n_{};       // MTP drafts reached / accepted by probability bucket (decayed)
    std::array<double, kMaxT + 1> last_{};                 // the round at which each size was last measured
};

}  // namespace strata::spec
