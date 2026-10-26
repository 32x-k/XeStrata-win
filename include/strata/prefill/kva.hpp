// include/strata/prefill/kva.hpp - STRATA_KVA: the optional approximate-prefill plan (research, opt-in).
//
// After layer `split` the prompt's late layers need only their recurrent state (GDN recurrence + conv history)
// and their KV/indexer caches; the attention output, the output projection, the MoE half and its expert
// transfers have no remaining consumer for the approximate positions.  The plan is:
//
//   positions [0, n - suffix)          approximated: late layers run state-only on a predicted block input
//   positions [n - suffix, n)          fully computed (the suffix), request-global
//
// v1 keeps the suffix on a CHUNK boundary (a chunk is approximated only when its last position is before the
// frontier), so no layer ever has to split one chunk into two regimes.  The late layers' block input is the
// hyper-connection residual R (10,240 floats = hc * n_embd) that the chunk's layer loop carries; it comes from
//   identity  the residual entering layer `split` (no projector at all, the zero-cost baseline),
//   oracle    a captured real input per late layer (STRATA_KVA_ORACLE=<dir>; the plumbing/parity mode),
//   linear    a fitted low-rank projector (STRATA_KVA_PROJECTOR=<file>; not implemented in this step).
//
// Everything here is read from the environment once per process and is off unless STRATA_KVA is set, so a
// default build behaves exactly as before.
#pragma once

#include <cstdint>
#include <cstdlib>
#include <string>

namespace strata::prefill {

struct KvaConfig {
    enum class Mode { Identity, Oracle, Linear };

    bool on = false;             ///< STRATA_KVA=1
    bool capture = false;        ///< STRATA_KVA_CAPTURE=<dir>: dump every late layer's real input rows
    bool capture_fp32 = false;   ///< STRATA_KVA_CAPTURE_FP32=1: fp32 records (else bf16; fp32 is for parity runs)
    bool oracle_fp32 = false;    ///< STRATA_KVA_ORACLE_FP32=1: the oracle files are fp32 (else bf16)
    bool parity = false;         ///< STRATA_KVA_PARITY=1: report the state-only path against the full path
    int64_t split = 24;          ///< STRATA_KVA_SPLIT: first approximated layer (zero-based)
    int64_t suffix = 2048;       ///< STRATA_KVA_SUFFIX: fully computed tail positions (rounded up to a chunk)
    int64_t min_prompt = 4096;   ///< STRATA_KVA_MIN_PROMPT: shorter prompts stay on the full path
    Mode mode = Mode::Identity;  ///< STRATA_KVA_MODE=identity|oracle|linear
    std::string oracle_dir;      ///< STRATA_KVA_ORACLE
    std::string capture_dir;     ///< STRATA_KVA_CAPTURE
    std::string projector;       ///< STRATA_KVA_PROJECTOR

    /// Does this run qualify at all?  A cold read of one prompt, on one stage, long enough for a suffix to
    /// leave approximated positions behind it.
    bool applies(int64_t pos0, int64_t n, int64_t lb, int64_t le, int64_t n_layers) const {
        if (!on || capture) return false;   // capture runs are ordinary full reads
        if (pos0 != 0 || lb != 0 || (le >= 0 && le != n_layers)) return false;
        return n >= min_prompt && n > suffix + 64;
    }
};

inline int64_t kva_env_i64(const char* name, int64_t dflt) {
    const char* v = std::getenv(name);
    if (v == nullptr || *v == '\0') return dflt;
    return (int64_t) std::atoll(v);
}

inline bool kva_env_flag(const char* name, bool dflt) {
    const char* v = std::getenv(name);
    if (v == nullptr || *v == '\0') return dflt;
    return v[0] != '0';
}

/// The process's KVA settings, parsed once (the engine's own options are parsed once too).
inline const KvaConfig& kva_config() {
    static const KvaConfig c = [] {
        KvaConfig k;
        k.on = kva_env_flag("STRATA_KVA", false);
        k.split = kva_env_i64("STRATA_KVA_SPLIT", 24);
        k.suffix = kva_env_i64("STRATA_KVA_SUFFIX", 2048);
        k.min_prompt = kva_env_i64("STRATA_KVA_MIN_PROMPT", 4096);
        k.parity = kva_env_flag("STRATA_KVA_PARITY", false);
        if (const char* d = std::getenv("STRATA_KVA_CAPTURE"); d != nullptr && *d != '\0') {
            k.capture = true;
            k.capture_dir = d;
            k.capture_fp32 = kva_env_flag("STRATA_KVA_CAPTURE_FP32", false);
        }
        if (const char* d = std::getenv("STRATA_KVA_ORACLE"); d != nullptr && *d != '\0') {
            k.oracle_dir = d;
            k.mode = KvaConfig::Mode::Oracle;
            k.oracle_fp32 = kva_env_flag("STRATA_KVA_ORACLE_FP32", false);
        }
        if (const char* d = std::getenv("STRATA_KVA_PROJECTOR"); d != nullptr && *d != '\0') {
            k.projector = d;
            k.mode = KvaConfig::Mode::Linear;
        }
        if (const char* m = std::getenv("STRATA_KVA_MODE"); m != nullptr && *m != '\0') {
            if (std::string(m) == "oracle") k.mode = KvaConfig::Mode::Oracle;
            else if (std::string(m) == "linear") k.mode = KvaConfig::Mode::Linear;
            else k.mode = KvaConfig::Mode::Identity;
        }
        return k;
    }();
    return c;
}

}  // namespace strata::prefill
