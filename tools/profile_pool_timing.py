# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Offline host PoolTiming accounting; never executes an engine or GPU tool.

Run: python tools/profile_pool_timing.py LOG [LOG ...] [--output report.json]
Accepts one complete sequential run per UTF-8 log, not planned dump shapes.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

FIELDS = (
    "embed", "pre_submit", "routing_D2H_submit+record", "routing_wait",
    "cpu_pool", "parts_H2D_submit", "parts_wait(post)", "post_submit",
    "dump_D2H_submit", "session_final_drain", "head_submit", "head_wait",
)
CAUTIONS = [
    "metadataNotProvided: command/environment/profiling and cache settings are not "
    "established by log absence; conditionsRaw is evidence only, not inferred settings.",
    "Host elapsed boundaries are not GPU execution timings. GPU work/transfers can "
    "overlap host calls; waits include scheduling and submit calls can include waits.",
    "rest is unpartitioned token-timer work, NOT GPU idle or residual GPU execution.",
    "CPU wp/dr/rp (wait-park/drain/re-park) are nested inside cpu_pool; never add "
    "them to the twelve nonnested spans. Drain is not pure CPU arithmetic.",
    "Zero parts_wait(post) does NOT mean sync-free: subsequent routing waits/final "
    "drain protect staging reuse. run_split's zero wp/rp does not exclude its internal "
    "barriers, which are inside dr; printed timing is rounded.",
    "Payload bytes count successful explicit copies, not physical bus traffic; "
    "API counts count attempted calls, not necessarily successful calls.",
    "Wall is the sum of executed token timers, not process elapsed or TTFT; startup "
    "and timing-line printing outside timers are excluded. Planned rows/EOS cause unknown.",
    "Derived precision is limited by printed rounding: spans 0.001 ms, token totals "
    "0.1 ms; negative rest is retained only within 12*N*0.0005+0.1 ms uncertainty.",
    "Cache temperature/cold/warm and external load are not controlled by this parser.",
    "Independent ideal zero-window bounds are theoretical, not predicted gains; "
    "assume all other work unchanged and cannot be combined across windows.",
]


def nonnegative(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError("expected nonnegative finite milliseconds")
    return number


def total(values):
    try:
        return nonnegative(math.fsum(values))
    except OverflowError as exc:
        raise ValueError("millisecond total overflow") from exc


def scope(rows, wall):
    n = len(rows)
    if not n and wall != 0:
        raise ValueError("empty scope must have a zero token timer")
    uncertainty = 12 * n * .0005 + .1
    def window(ms, count=None, payload=None):
        fraction = ms / wall if wall else None
        if fraction is not None and not math.isfinite(100 * fraction):
            raise ValueError("nonfinite derived share")
        result = {"wall_ms": ms, "mean_ms": ms / n if n else None,
                  "share_percent": 100 * fraction if fraction is not None else None}
        if count is not None:
            result.update(api_counts=count, payload_bytes=payload)
        if fraction is not None and 0 <= fraction < 1:
            bound = 1 / (1 - fraction)
            if math.isfinite(bound):
                result["independent_ideal_zero_window_upper_bound"] = {
                    "speedup": bound, "theoretical_not_predicted": True}
        return result
    spans = {key: window(total(row[key][0] for row in rows),
                         sum(row[key][1] for row in rows),
                         sum(row[key][2] for row in rows)) for key in FIELDS}
    span_sum = total(item["wall_ms"] for item in spans.values())
    rest = wall - span_sum
    if rest < -uncertainty:
        raise ValueError("span sum exceeds token timer beyond rounding uncertainty")
    result = window(wall)
    result.update(positions=n, spans=spans, span_sum_ms=span_sum,
                  api_counts=sum(item["api_counts"] for item in spans.values()),
                  payload_bytes=sum(item["payload_bytes"] for item in spans.values()),
                  rest=window(rest), rounding_uncertainty_ms=uncertainty)
    # A whole scope is not an independently removable window.
    result.pop("independent_ideal_zero_window_upper_bound", None)
    return result


def parse_bytes(raw, source_path="<memory>"):
    text = raw.decode("utf-8-sig")
    ids, timers, rows, conditions = {}, {}, {}, []
    for line in text.splitlines():
        stripped = line.strip()
        if re.match(r"^(prompt|output)\b", stripped):
            match = re.fullmatch(r"(prompt|output)\s*:\s*(\d+(?:\s+\d+)*)", stripped)
            if not match or match[1] in ids:
                raise ValueError("invalid or duplicate prompt/output IDs")
            ids[match[1]] = [int(x) for x in match[2].split()]
        elif re.match(r"^(prefill|decode)\b", stripped):
            match = re.fullmatch(r"(prefill|decode)\s+(\d+) tokens in (\S+) ms(?:\s+.*)?", stripped)
            if not match or match[1] in timers:
                raise ValueError("invalid or duplicate token timer")
            timers[match[1]] = (int(match[2]), nonnegative(match[3]))
        elif stripped.startswith("pool timing position"):
            match = re.fullmatch(r"pool timing position (\d+):\s*(.*)", stripped)
            if not match or int(match[1]) in rows:
                raise ValueError("invalid or duplicate position")
            fields = {}
            for part in match[2].split():
                field = re.fullmatch(r"([^=\s]+)=([^/\s]+)/(\d+)/(\d+)", part)
                if not field or field[1] not in FIELDS or field[1] in fields:
                    raise ValueError("invalid, unknown or duplicate span field")
                fields[field[1]] = (nonnegative(field[2]), int(field[3]), int(field[4]))
            if set(fields) != set(FIELDS):
                raise ValueError("missing span fields")
            rows[int(match[1])] = fields
        elif stripped.startswith(("strata generate:", "pool timing:", "pool phases", "the CPU expert pool")):
            conditions.append(line)
    if set(ids) != {"prompt", "output"} or set(timers) != {"prefill", "decode"}:
        raise ValueError("missing IDs or token timers")
    pre, dec = len(ids["prompt"]) - 1, len(ids["output"])
    actual = pre + dec
    if timers["prefill"][0] != pre or timers["decode"][0] != dec:
        raise ValueError("token timer counts disagree with actual IDs")
    if sorted(rows) != list(range(actual)):
        raise ValueError("positions must be contiguous 0..actual-1 (truncated/extra rows)")
    ordered = [rows[i] for i in range(actual)]
    return {"source_path": str(source_path), "sha256": hashlib.sha256(raw).hexdigest(),
            "prompt_ids": ids["prompt"], "output_ids": ids["output"], "actual_rows": actual,
            "conditionsRaw": conditions, "metadataNotProvided": True,
            "cautions": CAUTIONS, "scopes": {
                "prefill": scope(ordered[:pre], timers["prefill"][1]),
                "decode": scope(ordered[pre:], timers["decode"][1]),
                "all": scope(ordered, total(timer[1] for timer in timers.values()))}}


def parse_log(path):
    return parse_bytes(Path(path).read_bytes(), str(path))


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logsPaths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, help="UTF-8 JSON file (default: stdout)")
    args = parser.parse_args(argv)
    try:
        if args.output and args.output.resolve() in {p.resolve() for p in args.logsPaths}:
            raise ValueError("output must not overwrite a source log")
        runs = []
        for path in args.logsPaths:
            try:
                runs.append(parse_log(path))
            except (OSError, ValueError) as exc:
                raise ValueError(f"{path}: {exc}") from exc
        report = json.dumps({"runs": runs}, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
        if args.output:
            args.output.write_text(report, encoding="utf-8")
        else:
            sys.stdout.write(report)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
