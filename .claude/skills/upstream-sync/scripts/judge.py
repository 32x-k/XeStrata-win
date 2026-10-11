# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Judges XeStrata against upstream from sync_collect.py's matrix.json files: per model file, GPU, tier and
measure (prompt, output), the median of each engine's rounds; a cell passes when XeStrata is at most 1% slower.
One folder per model file.  A folder with a matrix-serve.json (bench_serve.py) is judged on the serve path too: per
GPU, tier and mode, the prompt, the follow-ups' prompt time (lower is better) and the output.  A folder with a
matrix-usage.json (bench_usage.py) is judged on everyday use too: per GPU and conversation, the time to the first
token (lower is better), the prompt and the output.  There each engine writes its own text, so all but the prompt
speed move with it from round to round: those cells pass when XeStrata is at most SERVE_LIMIT worse, the prompt
still at 1%.

    python .claude/skills/upstream-sync/scripts/judge.py [--prev NAME=DIR]... OUT/IQ3_XXS ...

A cell where upstream did not finish, ran at SLOW tok/s or less, or was stopped by bench_generate.py as that slow, is
left out (not measurable); one where only XeStrata did not finish fails.  A GPU measured with UP=none (XeStrata
alone: the B70 in an upstream merge) is judged against --prev only.
--prev NAME=DIR: the previous XeStrata's record of the folder named NAME (the last merge's OUT/NAME, or a release
record, whose serve and everyday tables are in DIR/serve): each cell both records have is judged against it too,
XeStrata's median against its median there (the prompt at 1%, the rest at SERVE_LIMIT: the text is upstream's or
each engine's own, and changes with the version).
Prints a line per cell and, last, "ALL PASS" (exit 0) or "FAIL <n>" (exit 1; also when no cell was compared).
"""
import json
import pathlib
import statistics
import sys

LIMIT = -0.01
# serve and everyday use, the measures that move with the text (each engine writes its own): on the B70 and the
# RX 9060 XT (2026-10-10) a round sat up to 10% off its cell's median at temperature 0.7, 1.5% greedy.  Their
# prompt speed does not move with the text (under 1%) and keeps LIMIT.
SERVE_LIMIT = -0.05
SLOW = 10.0                                   # tok/s: upstream this slow is not a speed to compare with
TIERS = ["1k", "4k", "32k", "64k", "128k", "192k", "262k"]
MEASURES = (("prefill_tok_s", "prompt"), ("decode_tok_s", "output"))
SERVE_MEASURES = (("prefill_tok_s", "prompt"), ("cont_ms", "follow"), ("decode_tok_s", "output"))
USAGE_MEASURES = (("ttft_s", "first"), ("prefill_tok_s", "prompt"), ("decode_tok_s", "output"))
LOWER_BETTER = {"ttft_s", "cont_ms"}


def median(rows, key):
    v = [r[key] for r in rows if r[key]]
    return statistics.median(v) if v else None


args = sys.argv[1:]
prev_dirs = {}
while len(args) >= 2 and args[0] == "--prev":
    name, _, d = args[1].partition("=")
    if not d:
        sys.exit(f"--prev {args[1]}: NAME=DIR")
    prev_dirs[name] = pathlib.Path(d)
    args = args[2:]


def load(path):
    """A record's rows: generate's (matrix.json), serve's and everyday use's (in the folder, or in its serve/ as a
    release record keeps them).  A release record names its engines (xe..., the upstream tag): they become arms."""
    def rows(name):
        for d in (path, path / "serve"):
            if (d / name).exists():
                r = json.loads((d / name).read_text())
                for x in r:
                    x.setdefault("arm", "xe" if x.get("engine", "").startswith("xe") else "up")
                return r
        return []
    return rows("matrix.json"), rows("matrix-serve.json"), rows("matrix-usage.json")


records = [(pathlib.Path(a), load(pathlib.Path(a))) for a in args]
unread = set(prev_dirs) - {folder.name for folder, _ in records}
if unread:
    sys.exit(f"--prev: no folder named {', '.join(sorted(unread))}")
fails = judged = 0


def cells(name, rec):
    """(GPU, key, where, rows, measures, limit) for each cell of a record: generate's, serve's, then everyday use's;
    `key` finds the same cell in another record."""
    gen, serve, usage = rec
    for rows, measures, limit, modes in ((gen, MEASURES, LIMIT, [None]), (serve, SERVE_MEASURES, SERVE_LIMIT, None)):
        for gpu in sorted({r["gpu"] for r in rows}):
            for tier in TIERS:
                # serve's modes as measured: greedy and the temperatures (t07, ...)
                for mode in modes or sorted({r["mode"] for r in rows}, key=lambda m: (m != "greedy", m)):
                    cell = [r for r in rows if (r["gpu"], r["tier"], r.get("mode")) == (gpu, tier, mode)]
                    if cell:
                        path = "generate" if mode is None else f"serve-{mode}"
                        yield (gpu, (path, gpu, tier), f"{name:10s} {path:12s} {gpu:8s} {tier:5s}", cell, measures,
                               limit)
    for gpu in sorted({r["gpu"] for r in usage}):
        for conv in dict.fromkeys(r["conv"] for r in usage):
            cell = [r for r in usage if (r["gpu"], r["conv"]) == (gpu, conv)]
            yield gpu, ("usage", gpu, conv), f"{name:10s} {'usage':12s} {gpu:8s} {conv:7s}", cell, USAGE_MEASURES, \
                SERVE_LIMIT


def against(where, key, xe, other, limit, what):
    """One comparison of XeStrata's median with another's; True when it passes."""
    d = other / xe - 1 if key in LOWER_BETTER else xe / other - 1
    ok = d >= limit
    n = {"ttft_s": 2, "cont_ms": 0}.get(key, 1)
    print(f"{where} {what}{'PASS' if ok else 'FAIL'}  {xe:.{n}f} vs {other:.{n}f} ({100 * d:+.1f}%)")
    return ok


for folder, rec in records:
    prev = ({k: cell for _, k, _, cell, _, _ in cells(folder.name, load(prev_dirs[folder.name]))}
            if folder.name in prev_dirs else {})
    for gpu, ckey, head, cell, measures, limit in cells(folder.name, rec):
        for key, name in measures:
            xe = median([r for r in cell if r["arm"] == "xe"], key)
            up = median([r for r in cell if r["arm"] == "up"], key)
            where = f"{head} {name:6s}"
            stopped = {r["stopped"] for r in cell if r["arm"] == "up" and r.get("stopped")}
            if not any(r["arm"] == "up" for r in cell):
                print(f"{where} SKIP  upstream not run (UP=none)")
            elif up is None and stopped:
                print(f"{where} SKIP  upstream stopped at {SLOW:g} tok/s or less ({', '.join(sorted(stopped))})")
            elif up is None:
                print(f"{where} SKIP  upstream did not finish")
            elif key not in LOWER_BETTER and up <= SLOW:
                print(f"{where} SKIP  upstream at {up:.1f} tok/s")
            elif xe is None:
                fails += 1
                print(f"{where} FAIL  XeStrata did not finish (upstream {up:.1f})")
            else:
                judged += 1
                fails += not against(where, key, xe, up, LIMIT if key == "prefill_tok_s" else limit, "")
            before = median([r for r in prev.get(ckey, []) if r["arm"] == "xe"], key)
            if before is not None:
                if xe is None:
                    fails += 1
                    print(f"{where} prev FAIL  XeStrata did not finish (before {before:.1f})")
                else:
                    judged += 1
                    fails += not against(where, key, xe, before,
                                         LIMIT if key == "prefill_tok_s" else SERVE_LIMIT, "prev ")
fails += judged == 0                          # nothing measured is no win
print("ALL PASS" if fails == 0 else f"FAIL {fails}")
sys.exit(1 if fails else 0)
