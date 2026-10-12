#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Validate/apply the pinned AdaptiveCpp DAG lane-affinity patch in an isolated checkout.

Run tools/adaptivecpp_build.py --apply-only for the event fix first. This tool
never builds, installs, fetches, resets, or replaces the working runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

COMMIT = "0ea863abe1af4422593583f91e5608ef75044a16"
FILE = "src/runtime/multi_queue_executor.cpp"
ORIGINAL_SHA256 = "44fc5eb3c1629c8b3f3712e02c1855246a7f754c883fc03b4661a63c3742f8a7"
ORIGINAL_CRLF_SHA256 = "7156ded5d10cb6af05f0e97048d14f92ef9044b1dd98d26f5dc5f80d467876ea"
PATCHED_SHA256 = "c22094c874acab2a5abee4d8728a6172df13f28ed200d229f9f61b04408238ad"
PATCHED_CRLF_SHA256 = "5aff0b4d6f4d4538bc5ae782466838faa3102e42f5585444414e54ee76cda779"
PATCH_SHA256 = "176a21c6e3a7d564d395b33cc684ff9cd961055cca7982a29f28ca246cc01148"
PATCH = Path(__file__).resolve().parents[1] / "third_party/main/adaptivecpp/patches/0002-dag-lane-affinity.patch"


class PatchError(RuntimeError):
    """No source was changed on a validation error."""


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(source: Path, *args: str, data: bytes | None = None) -> bytes:
    result = subprocess.run(["git", "-C", str(source), *args], input=data, capture_output=True)
    if result.returncode:
        raise PatchError(f"git {args[0]} failed ({result.returncode}): " +
                         result.stderr.decode(errors="replace"))
    return result.stdout


def check(source: Path) -> tuple[Path, bytes, bytes, bool, str]:
    source = source.resolve()
    top = Path(git(source, "rev-parse", "--show-toplevel").decode().strip()).resolve()
    if source != top or git(source, "rev-parse", "HEAD").decode().strip() != COMMIT:
        raise PatchError("AdaptiveCpp checkout root/HEAD differs from pinned commit")
    original = git(source, "show", f"{COMMIT}:{FILE}")
    if digest(original) != ORIGINAL_SHA256 or digest(original.replace(b"\n", b"\r\n")) != ORIGINAL_CRLF_SHA256:
        raise PatchError("committed source SHA256 differs from pinned original")
    path = source / FILE
    if not path.is_file() or path.is_symlink():
        raise PatchError("source must be a regular, non-symlinked file")
    current = path.read_bytes()
    if digest(current) in (ORIGINAL_SHA256, PATCHED_SHA256):
        ending = "LF"
    elif digest(current) in (ORIGINAL_CRLF_SHA256, PATCHED_CRLF_SHA256):
        ending = "CRLF"
    else:
        raise PatchError("working source is neither the exact original nor the exact patched file")
    patched = digest(current) in (PATCHED_SHA256, PATCHED_CRLF_SHA256)
    patch = PATCH.read_bytes()
    if digest(patch) != PATCH_SHA256 or b"\r" in patch:
        raise PatchError("pinned patch SHA256/line endings differ")
    applied = patch.replace(b"\n", b"\r\n") if ending == "CRLF" else patch
    git(source, "apply", "--unidiff-zero", "--check", *(["--reverse"] if patched else []),
        "-", data=applied)
    return path, current, applied, patched, ending


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="existing pinned AdaptiveCpp checkout")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--apply-only", action="store_true")
    modes.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        path, before, patch, patched, ending = check(args.source)
        if args.verify_only and not patched:
            raise PatchError("lane-affinity patch not applied")
        if args.apply_only and not patched:
            if path.read_bytes() != before or digest(PATCH.read_bytes()) != PATCH_SHA256:
                raise PatchError("source or patch changed after validation")
            git(args.source.resolve(), "apply", "--unidiff-zero", "-", data=patch)
            expected = PATCHED_CRLF_SHA256 if ending == "CRLF" else PATCHED_SHA256
            if digest(path.read_bytes()) != expected:
                raise PatchError("applied source differs from exact pinned postimage; not reverting user changes")
        print(json.dumps({"adaptivecpp_commit": COMMIT, "path": str(path),
                          "source_sha256": digest(path.read_bytes()), "patch_sha256": PATCH_SHA256,
                          "disposition": "verification-only" if args.verify_only else
                                         "already-applied-skip" if patched else "applied"}, indent=2))
    except (PatchError, OSError) as error:
        print(f"adaptivecpp_lane_affinity: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
