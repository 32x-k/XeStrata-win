"""``strata drafting plan|fetch``: the DFlash2 drafter file (strata_control/drafting_fetch.py). The app calls it with ``--output jsonl``."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from strata_control import acquisition, drafting, drafting_fetch


def add_parser(sp) -> None:
    drafting_p = sp.add_parser("drafting", help="the DFlash2 drafter file: what it would download, and fetch it")
    d_sp = drafting_p.add_subparsers(dest="drafting_command", required=True)
    for name, text in (("plan", "say what fetching would download; nothing is changed"),
                       ("fetch", "download the drafter (pinned, checked) and pack it for the engine")):
        one = d_sp.add_parser(name, help=text)
        one.add_argument("--data-dir", metavar="DIR", help="the model data folder (default: the one Strata uses)")
        one.add_argument("--output", choices=("human", "json", "jsonl"), default="human")


def _data_dir(a) -> Path:
    if a.data_dir:
        return Path(a.data_dir)
    from strata_cli.launch.catalog import Catalog
    return Catalog().data_dir()


def run(a) -> int:
    machine = a.output in ("json", "jsonl")
    folder = _data_dir(a)
    if a.drafting_command == "plan":
        result = drafting_fetch.plan(folder)
        if machine:
            print(json.dumps({"schema_version": 1, "ok": True, **result}, ensure_ascii=False))
        else:
            print(result["text"])
        return 0

    def show(step: str, done: int, total: int) -> None:
        if a.output == "jsonl":
            print(json.dumps({"event": "progress", "step": step, "done": done, "total": total}, ensure_ascii=False), flush=True)
        elif not machine and total:
            print(f"\r{step} {done / 1e9:.2f} of {total / 1e9:.2f} GB", end="", flush=True)

    try:
        out = drafting_fetch.ensure(folder, progress=show)
    except (drafting_fetch.DraftingError, acquisition.AcquisitionRefused) as exc:
        if machine:
            print(json.dumps({"schema_version": 1, "ok": False, "error": {"code": "drafting_failed", "message": str(exc)}}, ensure_ascii=False))
        else:
            print(f"\nstrata: {exc}", file=sys.stderr)
        return 5
    if machine:
        print(json.dumps({"schema_version": 1, "ok": True, "event": "result", "pack": str(out)}, ensure_ascii=False))
    else:
        print(f"\nThe DFlash2 drafter is ready: {out}")
    return 0
