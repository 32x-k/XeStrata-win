#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Patch a local, pinned AdaptiveCpp checkout before configuring/building/installing.

No cloning, fetching, resetting, or cleanup is performed. Compiler/backend choices
belong to the caller (--cmake-arg=-D...). --apply-only needs only --source;
--verify-installed-only also needs --prefix and never writes a receipt or patches.
The default-event constructor and its async-handler behavior remain unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

COMMIT = "0ea863abe1af4422593583f91e5608ef75044a16"
# Git's committed blob is LF; Windows checkouts may contain its exact CRLF form.
ORIGINAL_SHA256 = "0576c693880346dcd6d85ec3934be332c8e1678af7f126c267d603916338b2ad"
ORIGINAL_CRLF_SHA256 = "0eb54a547e68eaf0055f75431fbc8af7186e7221988d0829c333acec76797526"
HEADER = "include/hipSYCL/sycl/event.hpp"
INSTALLED_HEADER = "include/AdaptiveCpp/hipSYCL/sycl/event.hpp"
PATCH = Path(__file__).resolve().parents[1] / "third_party/main/adaptivecpp/patches/0001-event-async-handler.patch"
OLD = b"      : _node{evt} {}\n"
NEW = b"      : _node{evt}, _handler{handler} {}\n"


class BuildError(RuntimeError):
    """A validation or command failure; existing user changes are left alone."""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run(args: list[str], *, cwd: Path | None = None, data: bytes | None = None,
        verbose: bool = False) -> bytes:
    result = subprocess.run(args, cwd=cwd, input=data, capture_output=True, check=False)
    stdout = result.stdout.decode(errors="replace")
    stderr = result.stderr.decode(errors="replace")
    if verbose:
        # Keep JSON alone on stdout; stderr can be captured as the local build log.
        print("+ " + " ".join(args), file=sys.stderr, flush=True)
        print(stdout, end="", file=sys.stderr, flush=True)
        print(stderr, end="", file=sys.stderr, flush=True)
    if result.returncode:
        detail = f"stdout:\n{stdout}\nstderr:\n{stderr}".strip()
        raise BuildError(f"{args[0]} failed ({result.returncode}): {detail}")
    return result.stdout


def git(source: Path, *args: str, data: bytes | None = None) -> bytes:
    return run(["git", "-C", str(source), *args], data=data)


def check_patch_body(patch: bytes) -> None:
    """Permit one ordinary unified hunk and exactly the agreed constructor edit."""
    lines = patch.splitlines(keepends=True)
    while lines and (lines[0].startswith(b"#") or not lines[0].strip()):
        lines.pop(0)
    headers = [f"diff --git a/{HEADER} b/{HEADER}\n".encode(),
               f"--- a/{HEADER}\n".encode(), f"+++ b/{HEADER}\n".encode()]
    if lines[:3] != headers:
        raise BuildError("patch must target only event.hpp with ordinary unified headers")
    if len(lines) < 4 or not re.fullmatch(rb"@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@[^\n]*\n", lines[3]):
        raise BuildError("patch must contain exactly one unified hunk")
    body = lines[4:]
    if [line for line in body if line.startswith(b"-")] != [b"-" + OLD] or \
            [line for line in body if line.startswith(b"+")] != [b"+" + NEW] or \
            any(not line.startswith((b" ", b"-", b"+")) for line in body):
        raise BuildError("patch body is not the minimal copy-handler constructor fix")


@dataclass(frozen=True)
class CheckedSource:
    source: Path
    original: bytes
    expected: bytes
    current: bytes
    patch: bytes
    line_ending: str

    @property
    def patched(self) -> bool:
        return self.current == self.expected

    @property
    def application_patch(self) -> bytes:
        return self.patch.replace(b"\n", b"\r\n") if self.line_ending == "CRLF" else self.patch


def check_source(source: Path, patch_path: Path | None = None) -> CheckedSource:
    patch_path = PATCH if patch_path is None else patch_path
    source = source.resolve()
    top = Path(git(source, "rev-parse", "--show-toplevel").decode().strip()).resolve()
    if top != source:
        raise BuildError("--source must be the AdaptiveCpp checkout root")
    commit = git(source, "rev-parse", "--verify", "HEAD^{commit}").decode().strip()
    if commit != COMMIT:
        raise BuildError(f"AdaptiveCpp HEAD must be exactly {COMMIT}, got {commit}")
    original = git(source, "show", f"{COMMIT}:{HEADER}")
    if sha256(original) != ORIGINAL_SHA256 or original.count(OLD) != 1:
        raise BuildError("committed event.hpp does not match the pinned original SHA256/constructor")
    expected = original.replace(OLD, NEW, 1)
    original_crlf = original.replace(b"\n", b"\r\n")
    expected_crlf = expected.replace(b"\n", b"\r\n")
    if sha256(original_crlf) != ORIGINAL_CRLF_SHA256:
        raise BuildError("CRLF representation does not match the pinned original SHA256")
    header = source / HEADER
    if header.resolve() != header or not header.is_file():
        raise BuildError("source event.hpp must be a regular, non-symlinked checkout file")
    current = header.read_bytes()
    if current in (original, expected):
        line_ending = "LF"
    elif current in (original_crlf, expected_crlf):
        line_ending = "CRLF"
        original, expected = original_crlf, expected_crlf
    else:
        raise BuildError("source event.hpp is neither the exact original nor the fully patched header (LF/CRLF)")
    # Hash the canonical LF patch, not its in-memory checkout-specific encoding.
    patch = patch_path.read_bytes().replace(b"\r\n", b"\n")
    check_patch_body(patch)
    # Reverse-check an already patched file, without reverting it. Check/apply the
    # same in-memory bytes so a patch-file change cannot slip between the commands.
    checked = CheckedSource(source, original, expected, current, patch, line_ending)
    options = ["--reverse"] if checked.patched else []
    # The pinned hunk has leading context only, avoiding whitespace-only patch
    # lines. Full-file hashes and the exact postimage gate this, not loose context.
    git(source, "apply", "--unidiff-zero", "--check", *options, "-", data=checked.application_patch)
    return checked


def apply_patch(checked: CheckedSource) -> CheckedSource:
    if (checked.source / HEADER).read_bytes() != checked.current:
        raise BuildError("source header changed after validation")
    if checked.patched:
        return checked
    git(checked.source, "apply", "--unidiff-zero", "-", data=checked.application_patch)
    current = (checked.source / HEADER).read_bytes()
    if current != checked.expected:
        raise BuildError("applied header differs from expected; no rollback performed")
    return CheckedSource(checked.source, checked.original, checked.expected, current,
                         checked.patch, checked.line_ending)


def verify_installed(checked: CheckedSource, prefix: Path) -> bytes:
    header = prefix / INSTALLED_HEADER
    installed = header.read_bytes()
    if installed != checked.expected:
        raise BuildError(f"installed header does not match patched SHA256: {header}")
    return installed


def overlaps(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def check_paths(source: Path, build: Path | None, prefix: Path | None, receipt: Path | None) -> None:
    for label, path in (("build", build), ("prefix", prefix)):
        if path is not None:
            if overlaps(source, path):
                raise BuildError(f"{label} and source must be separate, non-overlapping paths")
            if path.exists() and not path.is_dir():
                raise BuildError(f"{label} is not a directory: {path}")
    if build is not None and prefix is not None and overlaps(build, prefix):
        raise BuildError("build and prefix must be separate, non-overlapping paths")
    if receipt is not None:
        if receipt.resolve() == source or source in receipt.resolve().parents:
            raise BuildError("receipt must be outside source")
        if receipt.exists() or receipt.is_symlink():
            raise BuildError(f"refusing to overwrite existing receipt: {receipt}")
    if build is not None and (build / "CMakeCache.txt").exists():
        values = {}
        for line in (build / "CMakeCache.txt").read_text(encoding="utf-8").splitlines():
            if "=" in line and ":" in line.split("=", 1)[0]:
                key, value = line.split("=", 1)
                values[key.split(":", 1)[0]] = value
        for key, wanted in (("CMAKE_HOME_DIRECTORY", source), ("CMAKE_INSTALL_PREFIX", prefix)):
            if key in values and wanted is not None and Path(values[key]).resolve() != wanted:
                raise BuildError(f"existing CMake cache has a different {key}; use another build directory")


def check_cmake_args(args: list[str]) -> None:
    # Caller chooses compilers/backends, but cannot redirect this wrapper's paths
    # or turn configure into another CMake operation.
    blocked = ("--build", "--install", "--install-prefix", "--preset", "--workflow")
    for arg in args:
        if arg == "--" or arg.startswith(("-S", "-B", "-P", "-E")) or \
                any(arg == flag or arg.startswith(flag + "=") for flag in blocked):
            raise BuildError(f"CMake path/mode argument is owned by the wrapper: {arg}")
        name = arg.removeprefix("-D").split("=", 1)[0].split(":", 1)[0]
        if name in ("CMAKE_INSTALL_PREFIX", "CMAKE_HOME_DIRECTORY", "CMAKE_CACHEFILE_DIR"):
            raise BuildError(f"CMake path argument is owned by the wrapper: {arg}")


def record(checked: CheckedSource, installed: bytes | None, prefix: Path | None,
           disposition: str) -> dict[str, str | None]:
    return {"adaptivecpp_commit": COMMIT, "patch_sha256": sha256(checked.patch),
            "patch_disposition": disposition, "source_line_ending": checked.line_ending,
            "committed_header_sha256": ORIGINAL_SHA256,
            "original_header_sha256": sha256(checked.original),
            "expected_patched_header_sha256": sha256(checked.expected),
            "source_header_sha256": sha256(checked.current),
            "installed_header_sha256": sha256(installed) if installed is not None else None,
            "source": str(checked.source), "prefix": str(prefix) if prefix is not None else None}


def positive_jobs(value: str) -> int:
    try:
        jobs = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("--jobs must be a positive integer") from error
    if jobs < 1:
        raise argparse.ArgumentTypeError("--jobs must be a positive integer")
    return jobs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="existing pinned AdaptiveCpp checkout")
    parser.add_argument("--build", type=Path, help="out-of-source CMake build directory")
    parser.add_argument("--prefix", type=Path, help="installation prefix outside source")
    parser.add_argument("--receipt", type=Path, help="optional new local JSON receipt (never overwritten)")
    parser.add_argument("--jobs", type=positive_jobs, default=2, help="positive build parallelism (default: 2)")
    parser.add_argument("--cmake-arg", action="append", default=[], help="repeat; use --cmake-arg=-DNAME=VALUE")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--apply-only", action="store_true")
    modes.add_argument("--verify-installed-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.apply_only and args.prefix is None:
        parser.error("--prefix is required except with --apply-only")
    if not args.apply_only and not args.verify_installed_only and args.build is None:
        parser.error("normal build mode requires --build")
    if args.verify_installed_only and args.receipt is not None:
        parser.error("verification-only mode cannot write --receipt (JSON is printed instead)")
    source = args.source.resolve()
    build = args.build.resolve() if args.build is not None else None
    prefix = args.prefix.resolve() if args.prefix is not None else None
    receipt = args.receipt.absolute() if args.receipt is not None else None
    try:
        check_paths(source, build, prefix, receipt)
        check_cmake_args(args.cmake_arg)
        checked = check_source(source)
        if not args.apply_only and not args.verify_installed_only and prefix is not None:
            # Do not overwrite an unrelated existing installation. A rerun must
            # already carry the exact fix; new or empty prefixes are safe.
            if prefix.exists() and any(prefix.iterdir()):
                verify_installed(checked, prefix)
        installed = None
        disposition = "already-applied-skip" if checked.patched else "applied"
        if args.verify_installed_only:
            disposition = "verification-only"
            assert prefix is not None
            installed = verify_installed(checked, prefix)
        else:
            checked = apply_patch(checked)
            if not args.apply_only:
                assert build is not None and prefix is not None
                run(["cmake", *args.cmake_arg, "-S", str(source), "-B", str(build),
                     f"-DCMAKE_INSTALL_PREFIX={prefix}"], verbose=True)
                run(["cmake", "--build", str(build), "--parallel", str(args.jobs)], verbose=True)
                run(["cmake", "--install", str(build)], verbose=True)
                checked = check_source(source)
                if not checked.patched:
                    raise BuildError("source header is no longer patched after build/install")
                installed = verify_installed(checked, prefix)
        output = json.dumps(record(checked, installed, prefix, disposition), indent=2) + "\n"
        if receipt is not None:
            receipt.parent.mkdir(parents=True, exist_ok=True)
            with receipt.open("x", encoding="utf-8") as stream:
                stream.write(output)
        print(output, end="")
    except (BuildError, OSError) as error:
        print(f"adaptivecpp_build: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
