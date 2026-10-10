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

    python .claude/skills/upstream-sync/scripts/judge.py [--prompt-only GPU]... OUT/IQ3_XXS OUT/IQ2_XS ...

A cell where upstream did not finish, ran at SLOW tok/s or less, or was stopped by bench.sh as that slow, is left
out (not measurable); one where only XeStrata did not finish fails.  --prompt-only GPU: that GPU's output cells are
left out too (a GPU no matrix.json names is an error: bench.sh's GPU argument, spelled the same).
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
prompt_only = set()
while len(args) >= 2 and args[0] == "--prompt-only":
    prompt_only.add(args[1])
    args = args[2:]
matrices = [(pathlib.Path(a), json.loads((pathlib.Path(a) / "matrix.json").read_text())) for a in args]
gpus = {r["gpu"] for folder, rows in matrices for r in rows + [
    x for name in ("matrix-serve.json", "matrix-usage.json") if (folder / name).exists()
    for x in json.loads((folder / name).read_text())]}
unknown = prompt_only - gpus
if unknown:
    sys.exit(f"--prompt-only: no measurements of {', '.join(sorted(unknown))} (GPUs: {', '.join(sorted(gpus))})")
fails = judged = 0


def cells(folder):
    """(GPU, where, rows, measures, limit) for each cell of a folder: generate's, then serve's."""
    for rows, measures, limit, modes in (
            (dict(matrices)[folder], MEASURES, LIMIT, [None]),
            (json.loads((folder / "matrix-serve.json").read_text()) if (folder / "matrix-serve.json").exists()
             else [], SERVE_MEASURES, SERVE_LIMIT, ["greedy", "t07"])):
        for gpu in sorted({r["gpu"] for r in rows}):
            for tier in TIERS:
                for mode in modes:
                    cell = [r for r in rows if (r["gpu"], r["tier"], r.get("mode")) == (gpu, tier, mode)]
                    if cell:
                        path = "generate" if mode is None else f"serve-{mode}"
                        yield gpu, f"{folder.name:10s} {path:12s} {gpu:8s} {tier:5s}", cell, measures, limit
    usage = json.loads((folder / "matrix-usage.json").read_text()) if (folder / "matrix-usage.json").exists() else []
    for gpu in sorted({r["gpu"] for r in usage}):
        for conv in dict.fromkeys(r["conv"] for r in usage):
            cell = [r for r in usage if (r["gpu"], r["conv"]) == (gpu, conv)]
            yield gpu, f"{folder.name:10s} {'usage':12s} {gpu:8s} {conv:7s}", cell, USAGE_MEASURES, SERVE_LIMIT


for folder, _ in matrices:
    for gpu, head, cell, measures, limit in cells(folder):
        for key, name in measures:
            xe = median([r for r in cell if r["arm"] == "xe"], key)
            up = median([r for r in cell if r["arm"] == "up"], key)
            where = f"{head} {name:6s}"
            stopped = {r["stopped"] for r in cell if r["arm"] == "up" and r.get("stopped")}
            if name == "output" and gpu in prompt_only:
                print(f"{where} SKIP  output not compared on this GPU (--prompt-only)")
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
                d = up / xe - 1 if key in LOWER_BETTER else xe / up - 1
                ok = d >= (LIMIT if key == "prefill_tok_s" else limit)
                fails += not ok
                n = {"ttft_s": 2, "cont_ms": 0}.get(key, 1)
                print(f"{where} {'PASS' if ok else 'FAIL'}  {xe:.{n}f} vs {up:.{n}f} ({100 * d:+.1f}%)")
fails += judged == 0                          # nothing measured is no win
print("ALL PASS" if fails == 0 else f"FAIL {fails}")
sys.exit(1 if fails else 0)
