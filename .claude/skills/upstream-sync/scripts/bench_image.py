# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""The image table of an upstream merge or a release on one GPU: XeStrata's and upstream's image generators
(Qwen-Image-2.1-Turbo through stable-diffusion.cpp's sd-cli), each as its own setup sets it up on this GPU, one
1024x1024 image of 8 steps from a fixed prompt and seed a run, through the command its server runs
(serve/images.py).  XeStrata: the Vulkan sd-cli with setup's timed choices (qimage.tune); upstream: the backend its
setup picks for the GPU (CUDA on NVIDIA, HIP on AMD, Vulkan on Intel; UP_IMAGE_BACKEND), built by its setup
(build_image_engine).  The runs are tools/bench_core.py's (ab_image), which setup's --bench makes too.

    XE_ROOT=DIR UP_ROOT=DIR DATA=DIR QUANT=Q8_0 UP_IMAGE_BACKEND=cuda [XE_IMAGE_GPU=N] [UP_IMAGE_GPU=N] \\
      python .claude/skills/upstream-sync/scripts/bench_image.py OUT GPU [--dry-run]     # from the repository root

XE_ROOT and UP_ROOT: each engine's tree, with its sd-cli in engine-image/ and the BUILD.json its setup writes beside
it (its backend, checked against the one asked for, and the "lib_dirs" its config takes); UP_ROOT=none: XeStrata
alone.  XE_SD_CLI and UP_SD_CLI: an sd-cli elsewhere (its BUILD.json beside it), such as the other tree's Vulkan build
on the B70, where both trees build the same one.  DATA: the data folder (the image model's files in its models/).
QUANT: the denoiser's quant setup picks for the GPU's memory (Q8_0 above 12 GB, Q6_K up to 12, Q4_K_M up to 8).
XE_IMAGE_GPU and UP_IMAGE_GPU: the GPU's index among the backend's devices where the PC has more than one (Vulkan
counts the processor's graphics too).  PY (default this Python): the Python that runs each tree's image code.
ROUNDS (default 2): images per engine, alternating, after a discarded warm-up per engine.
OUT/image-runs/GPU-ARM-ROUND.json gets each run's times, .txt sd-cli's log and .png the image; sync_collect.py OUT
reads them.  --dry-run: print the configs instead of running them.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "tools"))
import bench_core as B  # noqa: E402

args = [a for a in sys.argv[1:] if a != "--dry-run"]
if len(args) != 2:
    sys.exit(f"usage: {sys.argv[0]} OUT GPU [--dry-run]")
models = str(Path(B.need("DATA")) / "models")
quant = B.need("QUANT")
py = os.environ.get("PY", sys.executable)


def gpu_of(name: str) -> int | None:
    v = os.environ.get(name, "")
    return int(v) if v not in ("", "-") else None


engines, cfgs = [], {}
for arm, backend, gpu in (("xe", "vulkan", gpu_of("XE_IMAGE_GPU")),
                          ("up", os.environ.get("UP_IMAGE_BACKEND", ""), gpu_of("UP_IMAGE_GPU"))):
    root = B.need(f"{arm.upper()}_ROOT")
    if root == "none":
        continue
    if not backend:
        sys.exit("UP_IMAGE_BACKEND: the backend upstream's setup picks for this GPU (cuda, hip or vulkan)")
    e = B.Engine(arm, arm, [], "", root=Path(root), py=py)
    exe = Path(os.environ.get(f"{arm.upper()}_SD_CLI") or
               Path(root) / "engine-image" / ("sd-cli.exe" if os.name == "nt" else "sd-cli"))
    try:
        stamp = json.loads((exe.parent / "BUILD.json").read_text())
    except (OSError, ValueError) as err:
        sys.exit(f"{arm}: {exe.parent / 'BUILD.json'}: {err} (the sd-cli its setup built)")
    if stamp.get("backend") != backend:
        sys.exit(f"{arm}: {exe} is the {stamp.get('backend')} build, not {backend}")
    lib_dirs = stamp.get("lib_dirs") or []
    cfg = B.image_config(e, backend, quant, str(exe), models, gpu, lib_dirs)
    if "error" in cfg:
        sys.exit(f"{arm}: its image config: {cfg['error']}")
    print(f"{arm}: {backend} {quant}, args {' '.join(cfg.get('args') or [])}, env {cfg.get('env') or {}}", flush=True)
    engines.append(e)
    cfgs[arm] = cfg
B.ab_image(Path(args[0]), args[1], engines, cfgs, int(os.environ.get("ROUNDS", str(B.ROUNDS))),
           dry="--dry-run" in sys.argv)
