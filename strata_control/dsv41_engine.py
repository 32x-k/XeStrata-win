"""DeepSeek-V4.1's engine as its own package (feed member ``dsv41``), downloaded when DeepSeek is chosen.

It goes in ``<data>/engine-dsv41/<version>-<platform>-<backend>[-<variant>]/``, beside the Strata engine and never inside it, so the Qwen
engine's files are not touched.  The package is picked, downloaded (resumable), unpacked and checked with the same code as an engine update
(strata_control/updater.py: the signed feed, the archive's SHA-256, the manifest hash the feed names, the package signature, every file's hash);
there is no test start here (setup's own 16-token question follows the pack).
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Callable

NAME = "strata-dsv41"
DIR = "engine-dsv41"


def exe_name() -> str:
    return NAME + (".exe" if os.name == "nt" else "")


def find(dest: Path) -> Path | None:
    """The newest installed package's strata-dsv41 in `dest` (engine-dsv41/), or None."""
    try:
        folders = sorted((d for d in Path(dest).iterdir() if d.is_dir() and (d / exe_name()).is_file()), key=lambda d: d.name)
    except OSError:
        return None
    return folders[-1] / exe_name() if folders else None


def install(dest: Path, source: str, *, hardware: dict | None = None, emit: Callable[[str], None] = print, cancel=None) -> Path | None:
    """Fetch the DeepSeek engine package for this PC from the signed feed at `source` (a URL or a folder) into `dest`.
    Returns the new strata-dsv41, or None with a plain line through `emit` when there is none (not in the feed, nothing for this PC)."""
    from strata_control import feed as feedmod, signing, updater

    dest = Path(dest)
    work = dest.parent / (DIR + ".work")
    hw = hardware or updater.detect_hardware()
    state_dir = work / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    ctx = updater.Context(components=work, source=source, trusted=signing.load_trusted_keys(), hardware=hw, cancel=cancel,
                          emit=lambda event: emit(event["message"]) if event.get("event") == "phase" and event.get("message") else None)
    state = updater.State(work)
    try:
        feed = updater.open_feed(ctx, state)
        pkg, reasons = feedmod.select_engine(list(feed.dsv41), hw)
        if pkg is None:
            emit("There is no DeepSeek engine package for this computer in the update feed"
                 + (": " + "; ".join(reasons[:2]) if reasons else "") + ".")
            return None
        archive = updater._download(ctx, "dsv41", pkg.url, pkg.size, pkg.sha256, "the DeepSeek engine", feed)
        stage_root, component = updater.stage_engine(ctx, feed, pkg, archive)
        try:
            folder = stage_root / "engine" / updater.folder_name(pkg)
            dest.mkdir(parents=True, exist_ok=True)
            target = dest / folder.name
            shutil.rmtree(target, ignore_errors=True)
            shutil.move(str(folder), str(target))
        finally:
            shutil.rmtree(stage_root, ignore_errors=True)
        if os.name != "nt":
            (target / exe_name()).chmod(0o755)
        for old in dest.iterdir():                              # one generation: the older folders of this engine go
            if old.is_dir() and old.name != target.name:
                shutil.rmtree(old, ignore_errors=True)
        emit(f"DeepSeek engine {pkg.version} installed: {target}")
        shutil.rmtree(work, ignore_errors=True)               # (on a failure the partial download stays in `work`: the next try resumes it)
        return target / exe_name()
    except (updater.UpdateError, feedmod.FeedError, signing.SignatureError) as e:
        emit(f"The DeepSeek engine could not be fetched: {e}")
        return None
