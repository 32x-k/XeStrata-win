// include/strata/prefill/kva.hpp - STRATA_KVA: the approximate late-layer prompt read (on by default since 0.2.7).
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
// 0.2.7: the read is ON by default for a cold prompt of STRATA_KVA_MIN_PROMPT tokens or more on one CUDA/HIP
// device.  STRATA_KVA=0 brings back the exact 0.2.6 prompt path and the 0.2.6 draft window.  Everything is read from
// the environment once per process.
//
// The two facts the engine tells this header at start-up (kva_engine()): whether a draft layer (MTP) is loaded and
// how many cells it attends to, and whether the engine configuration rules the read out for the whole process
// (batch slots, a layer split).  The fully computed suffix must cover the drafter's window plus the 64 cells its
// catch-up reaches back (MtpDrafter::first_needed), or the drafter would build its K/V from approximated cells
// (that costs acceptance, never correctness: the target model verifies every token).
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

namespace strata::prefill {

/// Does this build have the state-only late-layer path?  CUDA and HIP: yes.  The Metal engine compiles prefill.cpp
/// against its own kernel library, which has none.
#if defined(STRATA_GPU_METAL)
inline constexpr bool kKvaBackend = false;
#else
inline constexpr bool kKvaBackend = true;
#endif

/// The draft layer's window (--mtp-window) the engine picks when KVA is on, the draft layer is loaded and the user
/// gave no --mtp-window: 2048 cells (suffix 2112).  The 0.2.6 default was 32768, i.e. every cell of a chat up to
/// that length, which no suffix could cover.
inline constexpr int64_t kKvaDrafterWindow = 2048;
/// The fully computed tail when no draft layer needs more.
inline constexpr int64_t kKvaSuffixDefault = 2048;

/// What the running engine knows and the environment cannot say.
struct KvaEngine {
    bool has_drafter = false;      ///< an MTP draft layer is loaded (--mtp with --spec 2 or more)
    int64_t drafter_window = 0;    ///< the cells it attends to; 0 = every cell
    std::string blocked;           ///< non-empty: this process never reads approximately, and why
};

inline KvaEngine& kva_engine() {
    static KvaEngine e;
    return e;
}

struct KvaConfig {
    enum class Mode { Identity, Oracle, Linear };
    enum class Only { All, Gdn, Qsa };

    bool on = true;              ///< on unless STRATA_KVA=0 (or this build has no path)
    bool explicit_on = false;    ///< STRATA_KVA was set to something other than 0: every declined read says why
    bool capture = false;        ///< STRATA_KVA_CAPTURE=<dir>: dump every late layer's real input rows
    bool capture_fp32 = false;   ///< STRATA_KVA_CAPTURE_FP32=1: fp32 records (else bf16; fp32 is for parity runs)
    bool oracle_fp32 = false;    ///< STRATA_KVA_ORACLE_FP32=1: the oracle files are fp32 (else bf16)
    bool parity = false;         ///< STRATA_KVA_PARITY=1: report the state-only path against the full path
    int64_t split = 24;          ///< STRATA_KVA_SPLIT: first approximated layer (zero-based)
    int64_t suffix_env = 0;      ///< STRATA_KVA_SUFFIX: fully computed tail positions (0 = not set: effective_suffix())
    int64_t min_prompt = 4096;   ///< STRATA_KVA_MIN_PROMPT: shorter prompts stay on the full path
    Mode mode = Mode::Identity;  ///< STRATA_KVA_MODE=identity|oracle|linear
    Only only = Only::All;       ///< STRATA_KVA_ONLY=all|gdn|qsa: which late layers take the approximate path
    std::string oracle_dir;      ///< STRATA_KVA_ORACLE
    std::string capture_dir;     ///< STRATA_KVA_CAPTURE
    std::string projector;       ///< STRATA_KVA_PROJECTOR

    /// The fully computed tail of a read.  STRATA_KVA_SUFFIX when set; otherwise 2048, or the draft layer's window +
    /// 64 when that is larger.  (A draft layer that attends to every cell has no suffix that covers it:
    /// decline() says so.)
    int64_t effective_suffix(const KvaEngine& e) const {
        if (suffix_env > 0) return suffix_env;
        int64_t s = kKvaSuffixDefault;
        if (e.has_drafter && e.drafter_window > 0) s = std::max<int64_t>(s, e.drafter_window + 64);
        return s;
    }

    /// Why this read cannot take the approximate path, or nullptr when it can: a cold read of one prompt (not a
    /// continuation, no layer split, no pictures, no DFlash2 taps), on an engine that allows it, long enough for a
    /// suffix to leave approximated positions behind it.  `routine` (optional) is set when the reason is an
    /// everyday one (a continuation, a short prompt) that an engine running KVA by default does not log.
    const char* decline(const KvaEngine& e, int64_t pos0, int64_t n, int64_t lb, int64_t le, int64_t n_layers,
                        bool pictures = false, bool taps = false, bool* routine = nullptr,
                        bool unvalidated_model = false) const {
        if (routine) *routine = false;
        if (!on) return "off (STRATA_KVA=0)";
        if (capture) return "capture mode";
        if (!e.blocked.empty()) return e.blocked.c_str();
        // The split layer, the state-only layers and the suffix rule were measured on one geometry (Qwen3.8-Flash-Next,
        // 48 layers, 512 experts).  Another model reads exactly unless STRATA_KVA=1 asks for it.
        if (unvalidated_model && !explicit_on) return "KVA was not validated for this model (STRATA_KVA=1 forces it)";
        if (pos0 != 0) {
            if (routine) *routine = true;
            return "a continuation read (pos0 != 0)";
        }
        if (lb != 0 || (le >= 0 && le != n_layers)) return "a layer split (not the whole trunk)";
        if (pictures) return "a prompt with pictures";
        if (taps) return "the DFlash2 drafter reads the late layers' hidden states";
        if (suffix_env <= 0 && e.has_drafter && e.drafter_window <= 0)
            return "the draft layer attends to every cell (set --mtp-window)";
        if (n < min_prompt) {
            if (routine) *routine = true;
            return "the prompt is shorter than STRATA_KVA_MIN_PROMPT";
        }
        if (n <= effective_suffix(e) + 64) {
            if (routine) *routine = true;
            return "the suffix leaves too few positions to approximate";
        }
        return nullptr;
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

/// The settings as the environment says now (uncached: the tests call it after changing the environment).
inline KvaConfig kva_config_from_env() {
    KvaConfig k;
    const char* sw = std::getenv("STRATA_KVA");
    const bool set = sw != nullptr && *sw != '\0';
    k.on = !set || sw[0] != '0';
    k.explicit_on = set && sw[0] != '0';
    if (!kKvaBackend) k.on = false;
    k.split = kva_env_i64("STRATA_KVA_SPLIT", 24);
    k.suffix_env = std::max<int64_t>(0, kva_env_i64("STRATA_KVA_SUFFIX", 0));
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
    if (const char* o = std::getenv("STRATA_KVA_ONLY"); o != nullptr && *o != '\0') {
        if (std::string(o) == "gdn") k.only = KvaConfig::Only::Gdn;
        else if (std::string(o) == "qsa") k.only = KvaConfig::Only::Qsa;
        else k.only = KvaConfig::Only::All;
    }
    return k;
}

/// The process's KVA settings, parsed once (the engine's own options are parsed once too).
inline const KvaConfig& kva_config() {
    static const KvaConfig c = kva_config_from_env();
    return c;
}

}  // namespace strata::prefill
