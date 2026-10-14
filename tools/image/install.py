# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""tools/image/install.py - image generation (Qwen-Image-2.1-Turbo) for an existing XeStrata install: compiles sd-cli
(Vulkan, or the CPU), downloads and checks the three model files, and writes the "images" section into a run config
(or prints it), as setup.py --images qwen-image does inside the normal setup (setup.py's setup_images).

    python tools/image/install.py --config ~/.config/xestrata/xestrata-iq3_xxs.json   (size picked for this GPU)
    python tools/image/install.py --quant Q8_0 --print

--gpu-policy auto|keep|shrink|unload: what the text model does during an image job (serve/images.py).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["vulkan", "cpu"], default="vulkan")
    ap.add_argument("--quant", choices=["Q4_K_M", "Q6_K", "Q8_0"])
    ap.add_argument("--models-dir", help="where the model files go (default: the data folder's models)")
    ap.add_argument("--config", help="a run config (xestrata-<model>.json) to write the \"images\" section into")
    ap.add_argument("--gpu-policy", choices=["auto", "keep", "shrink", "unload"], default="auto")
    ap.add_argument("--print", action="store_true", help="print the images section instead of writing it")
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args(argv)
    import setup as S
    gpu = S.gpu_info() or {}
    if a.models_dir is None:
        data, _ = S.data_folder(None)
        a.models_dir = str(data / "models")
    ns = argparse.Namespace(images="qwen-image", image_backend=a.backend, image_quant=a.quant,
                            image_gpu_policy=a.gpu_policy, yes=a.yes)
    section = S.setup_images(ns, gpu, Path(a.models_dir))
    if section is None:
        return 1
    if a.print or not a.config:
        print(json.dumps({"images": section}, indent=1))
        return 0
    path = Path(a.config)
    cfg = json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else {}
    cfg["images"] = section
    S.write_config(path, cfg)
    S.ok(f"{path.name}: image generation added (POST /v1/images/generations once the server starts again)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
