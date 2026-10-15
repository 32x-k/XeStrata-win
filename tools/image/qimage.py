# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""tools/image/qimage.py - Qwen-Image-2.1-Turbo for XeStrata's image endpoint: the model's three files at pinned
Hugging Face revisions, and the run config's "images" section for XeStrata's own image engine (strata-image, built
with the text engine).

setup.py (--images qwen-image) and tools/image/install.py use these pieces; serve/images.py runs the result.

  files     the denoiser (DiT) quant: Q8_0 by default (Unsloth's LPIPS vs bf16: Q4_K_M 0.153, Q6_K 0.055, Q8_0 0.036);
            the engine streams what does not fit in VRAM, so the GPU's memory does not decide it, and the smaller
            files only save disk and RAM.  The VAE (bf16); the Qwen3-VL-8B text encoder UD-Q4_K_XL (the model card's
            choice).  Each file is checked against the SHA-256 the Hugging Face API lists for it at the pinned
            revision.
"""
from __future__ import annotations

import os
from pathlib import Path

REVISIONS = {                                       # the `sha` of https://huggingface.co/api/models/<repo>, 2026-10-12
    "unsloth/Qwen-Image-2.1-Turbo-GGUF": "5d1f9f759ea877ae556cda88dbec958b62ad3809",
    "unsloth/Qwen-Image-2.1-FP8": "bb21a8ba7f0371c19ad1f72c82a5e8fc10f39939",
    "unsloth/Qwen3-VL-8B-Instruct-GGUF": "b93a7ee713758252c555be4210c00540df954dc2",
}
_DIT = "unsloth/Qwen-Image-2.1-Turbo-GGUF"
# name -> (repo, path in the repo, bytes, sha256): the Hub's LFS pointers at the pinned revisions
FILES = {
    "Q4_K_M": (_DIT, "qwen-image-2.1-turbo-Q4_K_M.gguf", 4_199_565_120,
               "8940e8b522af886e0c0f5afddf720e82dd223cb9a43580b5f61c4d740452ac20"),
    "Q6_K": (_DIT, "qwen-image-2.1-turbo-Q6_K.gguf", 6_271_551_296,
             "f4560fb47ad863ffed7e32737d3c42988c44bd0c459606777d7ffe2e895d0a70"),
    "Q8_0": (_DIT, "qwen-image-2.1-turbo-Q8_0.gguf", 7_640_860_480,
             "399d2ddc3636f36456e2f5bfa57187da4e38721bc046430317f42a9eab5b1712"),
    "vae": ("unsloth/Qwen-Image-2.1-FP8", "vae/qwen_image_2.1_vae_bf16.safetensors", 675_508_656,
            "71879ffd5321e6d10c3c87513e2b474b1252efa7f3dec2969214a9bf06a6dd5c"),
    "llm": ("unsloth/Qwen3-VL-8B-Instruct-GGUF", "Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf", 5_148_699_488,
            "e3d1a6e87c5cb31e054f2c3bc0dd82ffde052f613de5eef3665b7bd33c9b703e"),
}
QUANTS = ("Q4_K_M", "Q6_K", "Q8_0")
LICENSE_NOTE = ("Qwen-Image-2.1-Turbo is under the Qwen Research License (not Apache 2.0): "
                "https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/blob/main/LICENSE - read it before you use the "
                "images commercially")
MODEL_DIR = "qwen-image-2.1-turbo"
IMAGE_EXE = "strata-image.exe" if os.name == "nt" else "strata-image"
DEFAULT_QUANT = "Q8_0"


def file_url(name: str, endpoint: str = "https://huggingface.co") -> str:
    repo, path, _, _ = FILES[name]
    return f"{endpoint.rstrip('/')}/{repo}/resolve/{REVISIONS[repo]}/{path}"


def local_path(models_dir: Path, name: str) -> Path:
    return Path(models_dir) / MODEL_DIR / Path(FILES[name][1]).name


def image_config(quant: str, exe: Path, models_dir: Path, *, uma: bool = False, policy: str = "auto") -> dict:
    """The run config's "images" section (serve/images.py reads it)."""
    return {"model_name": "qwen-image-2.1-turbo", "exe": str(exe), "quant": quant,
            "diffusion_model": str(local_path(models_dir, quant)), "vae": str(local_path(models_dir, "vae")),
            "llm": str(local_path(models_dir, "llm")), "gpu_policy": policy, "uma": bool(uma)}
