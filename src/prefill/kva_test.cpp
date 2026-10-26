// src/prefill/kva_test.cpp - the KVA switch, the automatic tail and the reasons a read is turned away (no GPU needed).
#include "strata/prefill/kva.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

using namespace strata::prefill;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

static void unset(const char* k) {
#if defined(_WIN32)
    _putenv_s(k, "");
#else
    unsetenv(k);
#endif
}
static void put(const char* k, const char* v) {
#if defined(_WIN32)
    _putenv_s(k, v);
#else
    setenv(k, v, 1);
#endif
}
static void clear_all() {
    for (const char* k : {"STRATA_KVA", "STRATA_KVA_SPLIT", "STRATA_KVA_SUFFIX", "STRATA_KVA_MIN_PROMPT", "STRATA_KVA_PARITY",
                          "STRATA_KVA_CAPTURE", "STRATA_KVA_CAPTURE_FP32", "STRATA_KVA_ORACLE", "STRATA_KVA_ORACLE_FP32",
                          "STRATA_KVA_PROJECTOR", "STRATA_KVA_MODE", "STRATA_KVA_ONLY"})
        unset(k);
}
static bool is(const char* got, const char* want) { return got != nullptr && std::strcmp(got, want) == 0; }

int main() {
    // ---- the switch: on by default, STRATA_KVA=0 off, anything else on and chatty
    clear_all();
    {
        const KvaConfig c = kva_config_from_env();
        CHECK(c.on && !c.explicit_on);
        CHECK(c.split == 24 && c.min_prompt == 4096 && c.suffix_env == 0);
        CHECK(c.mode == KvaConfig::Mode::Identity && c.only == KvaConfig::Only::All && !c.capture);
    }
    put("STRATA_KVA", "0");
    { const KvaConfig c = kva_config_from_env(); CHECK(!c.on && !c.explicit_on); }
    put("STRATA_KVA", "1");
    { const KvaConfig c = kva_config_from_env(); CHECK(c.on && c.explicit_on); }
    put("STRATA_KVA", "");   // an empty value is "not set"
    { const KvaConfig c = kva_config_from_env(); CHECK(c.on && !c.explicit_on); }
    clear_all();

    // ---- the automatic tail: 2048, or the draft layer's window + 64 when that is larger; STRATA_KVA_SUFFIX wins
    {
        const KvaConfig c = kva_config_from_env();
        KvaEngine none;                                   // no draft layer
        CHECK(c.effective_suffix(none) == kKvaSuffixDefault);
        KvaEngine d;                                      // the window the engine picks (no --mtp-window given)
        d.has_drafter = true;
        d.drafter_window = kKvaDrafterWindow;
        CHECK(c.effective_suffix(d) == 2112);
        d.drafter_window = 8192;                          // --mtp-window 8192
        CHECK(c.effective_suffix(d) == 8256);
        d.drafter_window = 512;                           // a small window: the default tail still holds
        CHECK(c.effective_suffix(d) == 2048);
        d.drafter_window = 0;                             // every cell: no tail can cover it, the plain tail is returned
        CHECK(c.effective_suffix(d) == 2048);
        put("STRATA_KVA_SUFFIX", "3000");
        const KvaConfig e = kva_config_from_env();
        d.drafter_window = 8192;
        CHECK(e.effective_suffix(d) == 3000);
        clear_all();
    }

    // ---- who is turned away, and why (the first reason that holds is the one named)
    {
        const KvaConfig c = kva_config_from_env();
        KvaEngine eng;
        eng.has_drafter = true;
        eng.drafter_window = kKvaDrafterWindow;
        bool routine = false;
        // a cold 14.6K prompt on one device: applies
        CHECK(c.decline(eng, 0, 14600, 0, -1, 48) == nullptr);
        CHECK(c.decline(eng, 0, 14600, 0, 48, 48) == nullptr);   // le == n_layers is the whole trunk too
        // a continuation (turn 2 of a chat, a resumed prompt): routine
        CHECK(is(c.decline(eng, 9000, 14600, 0, -1, 48, false, false, &routine), "a continuation read (pos0 != 0)") && routine);
        // short prompts and a prompt the tail would swallow: routine
        CHECK(is(c.decline(eng, 0, 3000, 0, -1, 48, false, false, &routine), "the prompt is shorter than STRATA_KVA_MIN_PROMPT") && routine);
        CHECK(c.decline(eng, 0, 4096, 0, -1, 48) == nullptr);   // 4096 > 2112 + 64: it applies
        put("STRATA_KVA_SUFFIX", "5000");
        { const KvaConfig s = kva_config_from_env();
          CHECK(is(s.decline(eng, 0, 5000, 0, -1, 48, false, false, &routine), "the suffix leaves too few positions to approximate") && routine); }
        clear_all();
        // not routine: the user may want to know
        CHECK(is(c.decline(eng, 0, 14600, 24, -1, 48, false, false, &routine), "a layer split (not the whole trunk)") && !routine);
        CHECK(is(c.decline(eng, 0, 14600, 0, 24, 48), "a layer split (not the whole trunk)"));
        CHECK(is(c.decline(eng, 0, 14600, 0, -1, 48, true), "a prompt with pictures"));
        CHECK(is(c.decline(eng, 0, 14600, 0, -1, 48, false, true), "the DFlash2 drafter reads the late layers' hidden states"));
        // a model the split and the tail were not measured on reads exactly unless STRATA_KVA=1 asks
        CHECK(is(c.decline(eng, 0, 14600, 0, -1, 40, false, false, &routine, true),
                 "KVA was not validated for this model (STRATA_KVA=1 forces it)") && !routine);
        put("STRATA_KVA", "1");
        { const KvaConfig f = kva_config_from_env(); CHECK(f.decline(eng, 0, 14600, 0, -1, 40, false, false, nullptr, true) == nullptr); }
        clear_all();
        // a draft layer that attends to every cell has no tail that covers it
        KvaEngine all = eng;
        all.drafter_window = 0;
        CHECK(is(c.decline(all, 0, 14600, 0, -1, 48), "the draft layer attends to every cell (set --mtp-window)"));
        put("STRATA_KVA_SUFFIX", "2112");   // ... unless the user fixes the tail by hand
        { const KvaConfig s = kva_config_from_env(); CHECK(s.decline(all, 0, 14600, 0, -1, 48) == nullptr); }
        clear_all();
        // no draft layer at all: nothing to cover
        KvaEngine nod;
        CHECK(c.decline(nod, 0, 14600, 0, -1, 48) == nullptr);
        // the engine rules the whole process out (batch slots, a layer split, a peer device)
        KvaEngine blocked = eng;
        blocked.blocked = "batch slots (--batch)";
        CHECK(is(c.decline(blocked, 0, 14600, 0, -1, 48, false, false, &routine), "batch slots (--batch)") && !routine);
        // off wins over everything, capture over the rest
        put("STRATA_KVA", "0");
        { const KvaConfig o = kva_config_from_env(); CHECK(is(o.decline(eng, 0, 14600, 0, -1, 48), "off (STRATA_KVA=0)")); }
        put("STRATA_KVA", "1");
        put("STRATA_KVA_CAPTURE", "/tmp/kva-cap");
        { const KvaConfig o = kva_config_from_env(); CHECK(o.capture && is(o.decline(eng, 0, 14600, 0, -1, 48), "capture mode")); }
        clear_all();
    }

    // ---- the research modes still parse
    put("STRATA_KVA_ORACLE", "/tmp/o");
    put("STRATA_KVA_ORACLE_FP32", "1");
    put("STRATA_KVA_ONLY", "qsa");
    put("STRATA_KVA_SPLIT", "30");
    put("STRATA_KVA_MIN_PROMPT", "1024");
    {
        const KvaConfig c = kva_config_from_env();
        CHECK(c.mode == KvaConfig::Mode::Oracle && c.oracle_fp32 && c.oracle_dir == "/tmp/o");
        CHECK(c.only == KvaConfig::Only::Qsa && c.split == 30 && c.min_prompt == 1024);
    }
    clear_all();

    if (fails == 0) std::printf("kva_test: ok\n");
    return fails == 0 ? 0 : 1;
}
