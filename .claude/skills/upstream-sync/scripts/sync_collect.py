# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Reads bench.sh's OUT/runs/*.txt and writes OUT/matrix.json (every run) and, per GPU, the prompt and output tables
(the median of the rounds; a cell no run finished is "-", one bench.sh stopped as too slow "too slow") as Markdown to
stdout.  An upstream run bench.sh stopped as too slow has "stopped" (prompt or output); one stopped in its output
takes its prompt rate from the last PP line (or, without PP lines, from its "strata generate: prefill" line).
When bench_serve.py ran too, OUT/serve-runs/*.json go to OUT/matrix-serve.json and three more tables per GPU and mode:
the tier's prompt, the follow-ups (the tokens they read over their prompt time) and the output of all four turns.
When bench_usage.py ran too, OUT/usage-runs/*.json go to OUT/matrix-usage.json and a table per GPU: per conversation,
the time to the first token (the median of its turns), the prompt and the output speed (the server's timings, summed).

    python .claude/skills/upstream-sync/scripts/sync_collect.py OUT XE_NAME UP_NAME   # e.g. OUT xe0.1.41 v0.1.41
"""
import json
import pathlib
import re
import statistics
import sys

out = pathlib.Path(sys.argv[1])
NAMES = {"xe": sys.argv[2], "up": sys.argv[3]}
TIERS = ["1k", "4k", "32k", "64k", "128k", "192k", "262k"]


def grab(pat, text, cast=float):
    m = re.search(pat, text, re.M)
    return cast(m.group(1)) if m else None


rows = []
for f in sorted((out / "runs").glob("*.txt")):
    m = re.fullmatch(r"(.+)-(xe|up)-(\w+)-(\d+)", f.stem)
    if not m:                                 # the warm-ups
        continue
    s = f.read_text(errors="replace")
    stopped = grab(r"^bench: stopped, upstream's (\w+)", s, str)
    pp = re.findall(r"^PP (\d+) (\d+) \d+ ([\d.]+)$", s, re.M)
    rows.append({
        "gpu": m.group(1), "arm": m.group(2), "engine": NAMES[m.group(2)], "tier": m.group(3), "round": int(m.group(4)),
        "prompt_tokens": grab(r"^prefill\s+(\d+) tokens", s, int),
        "prefill_tok_s": grab(r"^prefill\s+.*->\s+([\d.]+) tok/s", s),
        "decoded": grab(r"^decode\s+(\d+) tokens", s, int),
        "decode_tok_s": grab(r"^decode\s+.*->\s+([\d.]+) tok/s", s),
        "spec_accept": grab(r"^speculation\s+.*\(([\d.]+)\)", s),
        "stopped": stopped,
    })
    if stopped == "output" and pp and pp[-1][0] == pp[-1][1]:   # stopped after the prompt: its rate from PP
        rows[-1]["prefill_tok_s"] = float(pp[-1][2])
    elif stopped == "output" and not pp:   # no PP lines (upstream's sycl/ port): from its prefill line
        rows[-1]["prefill_tok_s"] = grab(r"^strata generate: prefill \d+ tokens in .*?\(([\d.]+) tok/s\)", s)
(out / "matrix.json").write_text(json.dumps(rows, indent=1) + "\n")

for gpu in sorted({r["gpu"] for r in rows}):
    for key, title, fmt in (("prefill_tok_s", "Prompt (tokens/s)", "{:,.0f}"),
                            ("decode_tok_s", "Output (tokens/s)", "{:.1f}")):
        print(f"\n### {gpu}: {title}\n\n| Engine | " + " | ".join(t.upper() for t in TIERS) + " |")
        print("| --- |" + " ---: |" * len(TIERS))
        for name in NAMES.values():
            cells = []
            for t in TIERS:
                v = [r[key] for r in rows if (r["gpu"], r["engine"], r["tier"]) == (gpu, name, t) and r[key]]
                slow = [r for r in rows if (r["gpu"], r["engine"], r["tier"]) == (gpu, name, t) and r["stopped"]]
                cells.append(fmt.format(statistics.median(v)) if v else "too slow" if slow else "-")
            print(f"| {name} | " + " | ".join(cells) + " |")

serve = []
for f in sorted((out / "serve-runs").glob("*.json")):
    r = json.loads(f.read_text())
    if r["round"] == 0:                       # the warm-ups
        continue
    t = r["turns"]
    whole = len(t) > 1 and not r["why"]
    rate = lambda n, ms: n / ms * 1e3 if ms > 0 else None   # noqa: E731
    serve.append({
        "gpu": r["gpu"], "arm": r["arm"], "engine": NAMES[r["arm"]], "tier": r["tier"], "mode": r["mode"],
        "round": r["round"],
        "prefill_tok_s": rate(t[0]["read"], t[0]["prompt_ms"]) if t else None,
        "cont_tok_s": rate(sum(x["read"] for x in t[1:]), sum(x["prompt_ms"] for x in t[1:])) if whole else None,
        "decode_tok_s": rate(sum(x["generated"] for x in t), sum(x["decode_ms"] for x in t)) if whole else None,
        "spec_accept": (sum(x["accepted"] for x in t) / max(sum(x["offered"] for x in t), 1)) if whole else None,
        "stopped": (r["why"] or "").removeprefix("stopped: ") if (r["why"] or "").startswith("stopped") else None,
    })
if serve:
    (out / "matrix-serve.json").write_text(json.dumps(serve, indent=1) + "\n")
    tiers = [t for t in TIERS if any(r["tier"] == t for r in serve)]
    for gpu in sorted({r["gpu"] for r in serve}):
        for mode in sorted({r["mode"] for r in serve}):
            for key, title, fmt in (("prefill_tok_s", "Prompt (tokens/s)", "{:,.0f}"),
                                    ("cont_tok_s", "Follow-ups (tokens/s)", "{:,.0f}"),
                                    ("decode_tok_s", "Output (tokens/s)", "{:.1f}")):
                print(f"\n### {gpu} serve {mode}: {title}\n\n| Engine | " + " | ".join(t.upper() for t in tiers) + " |")
                print("| --- |" + " ---: |" * len(tiers))
                for name in NAMES.values():
                    cells = []
                    for t in tiers:
                        c = [r for r in serve if (r["gpu"], r["engine"], r["tier"], r["mode"]) == (gpu, name, t, mode)]
                        v = [r[key] for r in c if r[key]]
                        cells.append(fmt.format(statistics.median(v)) if v else
                                     "too slow" if any(r["stopped"] for r in c) else "-")
                    print(f"| {name} | " + " | ".join(cells) + " |")

usage = []
for f in sorted((out / "usage-runs").glob("*.json")):
    if f.name.endswith(".config.json"):
        continue
    r = json.loads(f.read_text())
    if r["round"] == 0 or (r["why"] and not r["why"].startswith("stopped")):   # warm-ups, runs that failed
        continue
    for conv in dict.fromkeys(t["conv"] for t in r["turns"]):
        ct = [x for x in r["turns"] if x["conv"] == conv]
        tot = lambda k: sum(x[k] or 0 for x in ct)   # noqa: E731
        usage.append({
            "gpu": r["gpu"], "arm": r["arm"], "engine": NAMES[r["arm"]], "conv": conv, "round": r["round"],
            "turns": len(ct), "context": r["context"],
            "ttft_s": statistics.median([x["ttft_s"] for x in ct if x["ttft_s"] is not None] or [0]) or None,
            "prefill_tok_s": tot("prompt_n") / tot("prompt_ms") * 1e3 if tot("prompt_ms") else None,
            "decode_tok_s": tot("predicted_n") / tot("predicted_ms") * 1e3 if tot("predicted_ms") else None,
            "read": tot("prompt_n"), "reused": tot("cache_n"), "generated": tot("predicted_n"),
            "stopped": (r["why"] or "").removeprefix("stopped: ") or None,
        })
        if r["why"]:                          # stopped as too slow: its one answer is no conversation's speed
            usage[-1].update(ttft_s=None, prefill_tok_s=None, decode_tok_s=None)
if usage:
    (out / "matrix-usage.json").write_text(json.dumps(usage, indent=1) + "\n")
    convs = list(dict.fromkeys(r["conv"] for r in usage))
    for gpu in sorted({r["gpu"] for r in usage}):
        print(f"\n### {gpu} everyday use\n\n| Engine | " + " | ".join(
            f"{c} {m}" for c in convs for m in ("first token (s)", "prompt (tok/s)", "output (tok/s)")) + " |")
        print("| --- |" + " ---: |" * (3 * len(convs)))
        for name in NAMES.values():
            cells = []
            for c in convs:
                for key, fmt in (("ttft_s", "{:.2f}"), ("prefill_tok_s", "{:,.0f}"), ("decode_tok_s", "{:.1f}")):
                    v = [r[key] for r in usage if (r["gpu"], r["engine"], r["conv"]) == (gpu, name, c) and r[key]]
                    cells.append(fmt.format(statistics.median(v)) if v else "-")
            print(f"| {name} | " + " | ".join(cells) + " |")
