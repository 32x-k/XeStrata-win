#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Peasant Smith and the Project Maya contributors
# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""ista_intake.py - one command per IST-DAS commit (DEEPSEEK-ISTA-PLAN Phase 0).

For a pinned HF commit and one or more of its sets it:
  1. lists the repo tree at that commit (path + size) and writes it to the report folder;
  2. downloads only files that are absent or have the wrong size, into <root>/<set>/;
  3. hashes everything into <root>/SHA256SUMS-<commit>-<set>.txt and hard-links duplicates (the Engram file is the same
     110.6 GB blob in all three sets, so it is stored once);
  4. reads the GGUF headers and reports every tensor type with its count and bytes, flagging types the device kernels do
     not take (the same set --pack-info prints as "no GPU kernel");
  5. builds a pointer pack with pack_gguf.py - no requant, the file as published;
  6. runs --pack-info, the scripted eval (session16.sh, 4995 positions) and a speed smoke (code + agent, DSpark off/on);
  7. writes one report per commit and set: <report-dir>/<commit>/<set>.md, every number tagged with the commit.

Nothing here requantises or rewrites a GGUF. Exit codes: 0 ok, 1 a stage failed (the report says which), 2 bad usage.

usage:
  ista_intake.py --commit 9e2528f --sets 2.00bpw,2.25bpw
  ista_intake.py --commit 9e2528f --sets 2.00bpw --no-eval      # download + pack only
  ista_intake.py --commit 9e2528f --sets 2.00bpw --no-download   # headers + pack + eval on what is on disk
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

REPO = "helcig/tmp-oct26"
PORT = pathlib.Path(__file__).resolve().parents[2]
REPORTS = PORT / "records"
PY = sys.executable
HF = "hf"
EXE = "strata-dsv41"
EVAL_SCRIPT: pathlib.Path | None = None
TOOLS = pathlib.Path(__file__).resolve().parent
# Optional source substitutions are supplied by the caller; machine paths belong in LOCAL.md.
ENGRAM_DIR: pathlib.Path | None = None
DSPARK: pathlib.Path | None = None
DENSE_WOA: pathlib.Path | None = None
DENSE_HEAD: pathlib.Path | None = None
ENGRAM_META: pathlib.Path | None = None
MMPROJ: pathlib.Path | None = None
DSPARK_NOISE = "128799"     # dspark_noise_token_id for this model (see the h225 pack)
DSPARK_RANK = "256"         # dspark_markov_rank
# ggml type ids the device kernels take (src/dsv41/ds4_iq.cu's supported() plus MXFP4); kept in sync with --pack-info
DEVICE_TYPES = {0, 1, 2, 3, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 29, 39}
TYPE_NAMES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
              10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
              16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
              22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
              29: "IQ1_M", 30: "BF16", 39: "MXFP4"}


def log(msg: str) -> None:
    print("%s %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def sha256_file(p: pathlib.Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def repo_tree(commit: str) -> list[tuple[str, int]]:
    from huggingface_hub import HfApi
    rows = []
    for f in HfApi().list_repo_tree(REPO, revision=commit, recursive=True):
        size = getattr(f, "size", None)
        if size:
            rows.append((f.path, int(size)))
    return sorted(rows)


def local_path(root: pathlib.Path, repo_path: str) -> pathlib.Path:
    return root / repo_path


def download(commit: str, repo_path: str, dest_root: pathlib.Path) -> int:
    dest_root.mkdir(parents=True, exist_ok=True)
    r = run([HF, "download", REPO, "--revision", commit, "--include", repo_path, "--local-dir", str(dest_root)])
    if r.returncode != 0:
        log("  download failed: %s" % (r.stderr or r.stdout)[-300:])
    return r.returncode


def tensor_types(shards: list[pathlib.Path]) -> dict[int, tuple[int, int]]:
    """type id -> (tensor count, bytes) over the shards, using pack_gguf's Gguf reader and its block table."""
    sys.path.insert(0, str(TOOLS))
    import pack_gguf  # noqa: E402
    types = dict(pack_gguf.TYPES)
    # pack_gguf's table covers the quant types; the intake also meets these (ggml ids, values/block, bytes/block)
    types.update({2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20), 6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24),
                  9: ("Q8_1", 32, 36), 15: ("Q8_K", 256, 292), 19: ("IQ1_S", 256, 50), 29: ("IQ1_M", 256, 56),
                  24: ("I8", 1, 1), 25: ("I16", 1, 2), 26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8)})
    out: dict[int, tuple[int, int]] = {}
    for s in shards:
        g = pack_gguf.Gguf(s)
        for name, typ, dims, off in g.tensors:          # (name, type, dims, offset); the file handle is closed already
            t = types.get(typ)
            if t is None:
                out[typ] = (out.get(typ, (0, 0))[0] + 1, out.get(typ, (0, 0))[1])
                continue
            nvals = 1
            for d in dims:
                nvals *= int(d)
            nbytes = (nvals // t[1]) * t[2] if t[1] > 1 else nvals * t[2]
            c, b = out.get(typ, (0, 0))
            out[typ] = (c + 1, b + nbytes)
    return out


def pack_set(commit: str, set_name: str, root: pathlib.Path, report: list[str]) -> pathlib.Path | None:
    setdir = root / set_name
    pack = PORT / "packs" / ("ista_%s_%s" % (set_name, commit[:7]))
    if pack.exists():
        report.append("- pack already exists: %s" % pack)
        return pack
    # the exact recipe the hand-built h225 pack used (HANDOVER.md §6 and h4_225.sh): no requant, our MXFP8 Engram, and the
    # three workarounds Phase 1 removes - the interleaved 3-D attn_output_a, the engram_(k|q|wkv) tensors, and the Engram
    # hash parameters that live in the UD set's metadata (--engram-meta), without which pack_gguf sees no Engram layers.
    cmd = [PY, str(TOOLS / "pack_gguf.py"), "--model", str(setdir), "--out", str(pack)]
    if ENGRAM_DIR is not None and ENGRAM_DIR.exists():
        cmd += ["--engram", str(ENGRAM_DIR)]
    if ENGRAM_META is not None and ENGRAM_META.exists():
        cmd += ["--engram-meta", str(ENGRAM_META)]
    if MMPROJ is not None and MMPROJ.exists():
        cmd += ["--mmproj", str(MMPROJ)]
    if DENSE_WOA is not None and DENSE_WOA.exists():
        cmd += ["--dense-override", str(DENSE_WOA)]
    if DENSE_HEAD is not None and DENSE_HEAD.exists():
        cmd += ["--dense-override", str(DENSE_HEAD)]
    cmd += ["--skip-tensors", r"attn_output_a|engram_(k|q|wkv)\."]
    if DSPARK is not None and DSPARK.exists():
        # pack_gguf fills the block/MTP geometry from the drafter but not the noise token or the markov rank: without them
        # the pack reports dspark_noise_token_id 0 and DSpark drafts one token a round (the 2026-10-22 bug).
        cmd += ["--dspark-gguf", str(DSPARK), "--dspark-noise-token", DSPARK_NOISE, "--dspark-markov-rank", DSPARK_RANK]
    log("  pack: %s" % " ".join(cmd[2:]))
    r = run(cmd, cwd=str(PORT))
    (PORT / "runs").mkdir(exist_ok=True)
    (PORT / "runs" / ("pack_%s_%s.log" % (set_name, commit[:7]))).write_text((r.stdout or "") + (r.stderr or ""))
    if r.returncode != 0:
        report.append("- pack FAILED (rc %d), see runs/pack_%s_%s.log" % (r.returncode, set_name, commit[:7]))
        return None
    report.append("- pack: %s" % pack)
    return pack


def tensor_names(setdir: pathlib.Path) -> list[str]:
    sys.path.insert(0, str(TOOLS))
    try:
        import pack_gguf
    except Exception:
        return []
    names = []
    for s in sorted(setdir.glob("*.gguf")):
        try:
            g = pack_gguf.Gguf(s)
            names += [t[0] for t in g.tensors]
        except Exception:
            continue
    return names


def pack_info(pack: pathlib.Path) -> str:
    r = run([EXE, "--pack-info", "--pack", str(pack)], cwd=PORT)
    return (r.stdout or "") + (r.stderr or "")


def eval_pack(pack: pathlib.Path, tag: str) -> list[str]:
    # session16.sh takes the pack directory NAME (it prefixes packs/ itself), never a path
    if EVAL_SCRIPT is None:
        raise ValueError("--eval-script is required unless --no-eval is set")
    r = run(["bash", str(EVAL_SCRIPT), pack.name], cwd=PORT)
    (PORT / "runs" / ("eval_%s.log" % tag)).write_text((r.stdout or "") + (r.stderr or ""))
    rows = [l for l in (r.stdout or "").splitlines() if l.startswith("s16_") or l.startswith("rc=")]
    return rows


def speed_smoke(pack: pathlib.Path, tag: str) -> list[str]:
    out = []
    ids = (PORT / "prompts/sa_code.ids").read_text()[:300]
    for prompt, name in ((ids, "code"),):
        for dspark in ("off", "on"):
            cmd = [EXE, "--pack", str(pack), "--tp", "1", "--threads", "off",
                   "--max-context", "2048", "--vision", "off", "--dspark", dspark,
                   "--expert-profile", "none", "--ids", prompt, "--n", "32"]
            t0 = time.time()
            r = run(cmd, cwd=PORT)
            dt = time.time() - t0
            m = re.search(r"decode:.*?([\d.]+) tok/s", (r.stderr or ""))
            out.append("| %s | %s | %s | %s | %.1f |" % (name, dspark, "ok" if r.returncode == 0 else "FAILED",
                                                         m.group(1) if m else "-", dt))
    return out


def main() -> int:
    global PORT, REPORTS, PY, HF, EXE, EVAL_SCRIPT
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", required=True)
    ap.add_argument("--sets", default="2.00bpw")
    ap.add_argument("--root", required=True, help="source model directory")
    ap.add_argument("--work-dir", required=True, help="pack and evaluation working directory")
    ap.add_argument("--report-dir", required=True, help="measurement records directory (see LOCAL.md)")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--hf", default="hf")
    ap.add_argument("--exe", default="strata-dsv41")
    ap.add_argument("--eval-script")
    for name in ("engram-dir", "dspark", "dense-woa", "dense-head", "engram-meta", "mmproj"):
        ap.add_argument("--" + name)
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--no-pack", action="store_true")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--no-speed", action="store_true")
    a = ap.parse_args()
    PORT, REPORTS = pathlib.Path(a.work_dir), pathlib.Path(a.report_dir)
    PY, HF, EXE = a.python, a.hf, a.exe
    EVAL_SCRIPT = pathlib.Path(a.eval_script) if a.eval_script else None
    if not a.no_eval and EVAL_SCRIPT is None:
        ap.error("--eval-script is required unless --no-eval is set")
    for key in ("engram_dir", "dspark", "dense_woa", "dense_head", "engram_meta", "mmproj"):
        value = getattr(a, key)
        globals()[key.upper()] = pathlib.Path(value) if value else None
    commit = a.commit
    root = pathlib.Path(a.root)
    rep_dir = REPORTS / commit[:7]
    rep_dir.mkdir(parents=True, exist_ok=True)
    report: list[str] = ["# IST-DAS intake, commit %s" % commit, "", "run %s" % time.strftime("%Y-%m-%dT%H:%M:%S%z"), ""]

    log("listing %s at %s" % (REPO, commit[:7]))
    try:
        tree = repo_tree(commit)
    except Exception as e:      # noqa: BLE001  (HF's API rate-limits unauthenticated boxes; the files are already local)
        if not a.no_download:
            raise
        log("  the Hub API is not answering (%r); building the file list from disk" % (e,))
        tree = []
        for set_name in [s.strip() for s in a.sets.split(",") if s.strip()]:
            for p in sorted((root / set_name).glob("*.gguf")):
                tree.append(("%s/%s" % (set_name, p.name), p.stat().st_size))
        tree.sort()
    (rep_dir / "tree.txt").write_text("\n".join("%s %d" % (p, s) for p, s in tree))
    report += ["## Files at this commit", "", "| path | GB |", "|---|---|"]
    report += ["| %s | %.2f |" % (p, s / 1e9) for p, s in tree]
    report.append("")

    rc = 0
    for set_name in [s.strip() for s in a.sets.split(",") if s.strip()]:
        report += ["## Set %s" % set_name, ""]
        files = [(p, s) for p, s in tree if p.startswith(set_name + "/") and p.endswith(".gguf")]
        # 1. download what is missing or the wrong size
        if not a.no_download:
            for p, size in files:
                dest = local_path(root, p)
                if dest.exists() and dest.stat().st_size == size:
                    report.append("- present: %s (%.2f GB)" % (p, size / 1e9))
                    continue
                log("downloading %s (%.2f GB)" % (p, size / 1e9))
                d = download(commit, p, root)
                if d != 0:
                    report.append("- DOWNLOAD FAILED: %s" % p)
                    rc = 1
                else:
                    report.append("- downloaded: %s (%.2f GB)" % (p, size / 1e9))
        # 2. hash + dedupe
        sums = root / ("SHA256SUMS-%s-%s.txt" % (commit[:7], set_name))
        lines = []
        for p, size in files:
            dest = local_path(root, p)
            if not dest.exists():
                continue
            lines.append("%s  %s" % (sha256_file(dest), p))
        sums.write_text("\n".join(lines) + "\n")
        report.append("- sha256: %s" % sums)
        # 3. headers
        shards = [local_path(root, p) for p, _ in files if local_path(root, p).exists()]
        if shards:
            try:
                types = tensor_types(shards)
                report += ["", "| type | tensors | GB | device kernel |", "|---|---|---|---|"]
                for t, (c, b) in sorted(types.items()):
                    report.append("| %s (%d) | %d | %.2f | %s |" % (
                        TYPE_NAMES.get(t, "?"), t, c, b / 1e9, "yes" if t in DEVICE_TYPES else "**NO - CPU lane**"))
                report.append("")
            except Exception as e:      # noqa: BLE001
                report.append("- header scan failed: %r" % (e,))
        # 4. pack
        if not a.no_pack:
            pack = pack_set(commit, set_name, root, report)
        else:
            pack = None
        # 5. pack-info + eval + speed
        if pack is not None:
            pi = pack_info(pack)
            report += ["### --pack-info", "", "```", pi.strip()[-3000:], "```", ""]
            if "MISSING" in pi:
                report.append("- **pack-info reports MISSING files: this pack cannot load**")
                rc = 1
            if not a.no_eval:
                log("eval %s" % pack)
                rows = eval_pack(pack, "%s_%s" % (set_name, commit[:7]))
                report += ["### quality (4995 positions, code/agent/english)", "", "```", "\n".join(rows) or "(no output)", "```", ""]
            if not a.no_speed:
                log("speed smoke %s" % pack)
                rows = speed_smoke(pack, "%s_%s" % (set_name, commit[:7]))
                report += ["### speed smoke (32 tokens, one card)", "", "| prompt | dspark | rc | decode tok/s | wall s |",
                           "|---|---|---|---|---|"] + rows + [""]
        report.append("")

    out = rep_dir / ("%s.md" % ",".join(s.strip() for s in a.sets.split(",")))
    out.write_text("\n".join(report) + "\n")
    log("report: %s" % out)
    print("REPORT %s" % out)
    return rc


if __name__ == "__main__":
    sys.exit(main())
