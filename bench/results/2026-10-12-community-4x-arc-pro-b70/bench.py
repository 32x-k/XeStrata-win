#!/usr/bin/env python3
"""Measure the running XeStrata engine through its OpenAI-compatible API.

Usage: measure.py [WORDS] [MAXTOK] [RUNS] [LABEL]
Prints prefill and decode tok/s per run (from the response's own `timings`) and the median.
A warm-up request is discarded, so the first-token driver work does not skew the numbers.
"""
import json, statistics, sys, time, urllib.request

WORDS = int(sys.argv[1]) if len(sys.argv) > 1 else 750     # ~1K tokens at 6 chars/word
MAXTOK = int(sys.argv[2]) if len(sys.argv) > 2 else 128
RUNS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
LABEL = sys.argv[4] if len(sys.argv) > 4 else "run"
URL = "http://127.0.0.1:8095/v1/chat/completions"

BODY = ("The quick brown fox jumps over the lazy dog while counting prime numbers. "
        * (WORDS // 12 + 2))[: WORDS * 6]
PROMPT = ("Summarise the following text in one sentence, then list three facts it states.\n\n"
          + BODY + "\n\nAnswer:")


def one(max_tokens, nonce=0):
    # a varying marker at the START: the server reuses a cached conversation prefix, and a suffix-only
    # change would be read as ~5 new tokens (prompt_n 5).  Leading difference forces a full prefill.
    prompt = f"[run {nonce}] " + PROMPT
    req = urllib.request.Request(URL, data=json.dumps({
        "model": "x", "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0}).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    return r.get("timings") or {}, time.time() - t0


try:
    one(8, -1)                                # warm-up, discarded
except Exception as e:
    print(f"  {LABEL}: warm-up failed: {e}"); sys.exit(1)

pre, dec, acc, walls = [], [], [], []
for i in range(RUNS):
    t, w = one(MAXTOK, i)
    if not t:
        print("  no timings in the response"); continue
    pre.append(t["prompt_per_second"]); dec.append(t["predicted_per_second"])
    acc.append((t.get("draft_n_accepted", 0), t.get("draft_n", 0)))
    walls.append(w)
    print(f"  {LABEL} run{i+1}: prefill {t['prompt_per_second']:7.1f} tok/s "
          f"({t['prompt_n']} tok, {t['prompt_ms']:.0f} ms)  decode {t['predicted_per_second']:6.1f} tok/s "
          f"({t['predicted_n']} tok, {t['predicted_ms']:.0f} ms)  drafts {t.get('draft_n_accepted')}/{t.get('draft_n')}")

if pre:
    print(f"  {LABEL} MEDIAN: prefill {statistics.median(pre):7.1f} tok/s   decode {statistics.median(dec):6.1f} tok/s"
          f"   drafts {statistics.median([a for a, _ in acc])}/{statistics.median([b for _, b in acc])}")
