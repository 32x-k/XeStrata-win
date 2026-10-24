# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""serve/images.py - image generation (Qwen-Image-2.1-Turbo, Qwen-Image-2.1, Mage-Flow) through XeStrata's own image
engine, strata-image.

POST /v1/images/generations takes OpenAI's request shape (prompt, size, n, response_format b64_json|url) plus Strata's
own field seed; POST /v1/images/edits the same with reference images (OpenAI's multipart form, image or image[]; or JSON
with images: [{"image_url": ...}]), which the image model edits or takes as its subject (Qwen-Image-2.1's edits: the
vision encoder reads them into the prompt, the VAE's encoder into the DiT's).  The engine (src/image/strata_image.cpp,
`strata-image --serve`) starts with the first image and stays, outside the server (serve/image_daemon.py, the next
server of the same image model and GPU finds it: a model switch does not lose it): its kernels built once, the
prompt's text states and the layers that fit in VRAM kept between images, one image a request.  It runs Turbo's
8 steps (CFG 1, the Euler flow sampler, the model's own sigmas), the image the one stable-diffusion.cpp makes with the
model card's command for the same seed (within LPIPS 0.03); the base model ("schedule": "base") the steps asked for
(default the config's "steps", 40) on its scheduler's sigmas (base_sigmas), CFG 1 as well.  A size whose sides are
not multiples of 16 (1920x1080) is made at the sides rounded up (1920x1088) and its middle cut out.

Mage-Flow ("family": "mage-flow", "schedule": "mage"): its own prompt form (MF_PROMPT_HEAD, the 34 ids of its system
turn dropped), the steps asked for (default the config's: Turbo 4, the RL model 20, the base model 30) on its static
shift (mage_sigmas), and guidance by guidance_scale or true_cfg_scale (the one scale it has; default the config's
"cfg_scale": Turbo 1, none; the others 5.0) with negative_prompt (default " ", its pipeline's).  Its edits take up to
MF_MAX_REFS reference images, in its edit form (MF_EDIT_HEAD, "Image j: <pad>" before the instruction, the 64 ids of
the system turn dropped) and the config's "edit_steps" (its edit models': Turbo 4, the others 30), on its edit DiT
("edit_model"); masks and transparent images it has none of.

The GPU is shared with the text engine one job at a time: an image job takes the service's turn (the FIFO every chat
request takes), so no chat turn runs on the GPU meanwhile.  The image engine works in the VRAM it is given: what it
holds and what is free, less a reserve; layers that do not fit are streamed from RAM, so it runs, more slowly, in a few
GB.  Before each job it says what the image needs (NEED: the least, and the DiT resident with the VAE in one piece),
and the config's "gpu_policy" says what happens to the text engine during the job (gpu_turn):
  keep    - it stays loaded and idle (an APU's shared memory, a card with room for both)
  shrink  - an engine started with --vram-elastic gives the image job VRAM from its expert cache and takes it back
            afterwards (POST /v1/vram's mechanism; the conversation stays in the engine)
  unload  - its conversation is saved to a session file, the engine is stopped, and the next start restores it
  auto    - (default) keep when the image fits beside it at full speed, else shrink when the engine can, else keep
            when the image fits at all (its layers streamed), else unload
After a job the image engine keeps what it holds (the next image starts warm) only under keep with VRAM to spare;
otherwise it lets go of everything but its process (about 200 MiB).  The daemon ends after daemon_idle_s (default
1800) without a server.

The config's "images" section (written by setup --images, tools/image/qimage.py):
  {"exe": "engine/strata-image", "diffusion_model": "...Q8_0.gguf", "schedule": "turbo"|"base", "steps": 8|40,
   "vae": "...", "llm": "...Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf", "quant": "Q8_0", "gpu_policy": "auto",
   "reload": "lazy"|"eager", "max_n": 4, "max_side": 2048, "timeout_s": 3600, "uma": false, "out_dir": null,
   "keep_files": 64,
   "vision": "...mmproj-F16.gguf", "max_refs": 10, "reserve_mib": 512, "spare_mib": 1024, "ssd_offload": false,
   "text_encoder": "stock", "text_quant": "Q4_K",
   "quant_lowered": {"from": "Q8_0", "to": "Q6_K", "text_from": "Q8_0", "text_to": "Q4_K", "need_gb": 20.2,
                     "ram_left_gb": 14.0}}
  schedule: Turbo's own sigmas or the base model's (base_sigmas; no "schedule" is Turbo); steps: the base model's
  default (a request's "steps", 2 to MAX_STEPS, overrides it); vision: Qwen3-VL's vision encoder (mmproj), for the
  edits (without it they are refused); max_refs: the reference images an edit takes at most; reserve_mib: VRAM the
  image engine leaves free when it sizes itself; spare_mib: the free VRAM that must remain after a job under keep for
  the engine to keep its layers; quant_lowered (only when setup lowered them): the sizes setup took instead of the ones
  asked for, because the RAM the text model leaves was short (note(): the server's log and the web page say so).
"""
from __future__ import annotations

import base64
import collections
import contextlib
import json
import math
import os
import queue
import random
import re
import secrets
import struct
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

TURBO_STEPS = 8
MAX_STEPS = 100                    # the base model's
CFG_MAX = 30.0                     # guidance_scale / true_cfg_scale (vLLM-Omni: "typically 0.0-20.0")
MODEL_NAME = "qwen-image-2.1-turbo"
POLICIES = ("auto", "keep", "shrink", "unload")
SIZE_STEP = 16                     # the VAE's 8x and the DiT's 2x2 patches: sides made at a multiple of 16, then cut
MIN_SIDE = 256
FILE_NAME = re.compile(r"^[0-9a-f]{32}\.(png|jpeg|webp)$")
# OpenAI's output_format -> the file's MIME type (Pillow writes jpeg and webp; PNG is png_rgba's)
FORMATS = {"png": "image/png", "jpeg": "image/jpeg", "webp": "image/webp"}
# Qwen's recommended prompt for a transparent image (its README's "Transparent Image Generation (RGBA)")
TRANSPARENT_PROMPT = ("This is an RGBA image with transparency. {prompt}. The image has alpha channel and the "
                      "background is transparent.")
# OpenAI's quality -> the base model's steps (None: the config's); Turbo keeps its 8 whatever it says
QUALITY_STEPS = {"low": 20, "medium": 30, "standard": 30, "high": None, "hd": None, "auto": None}
# the model's prompt (stable-diffusion.cpp's Qwen-Image template): the system turn, whose DROP leading ids' states the
# DiT does not take, then the user's prompt as plain text
PROMPT_HEAD = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n<|im_start|>user\n"
PROMPT_TAIL = "<|im_end|>\n<|im_start|>assistant\n"
DROP = 14
# Mage-Flow's prompt (its pipeline's "mage-flow" template): the system turn's 34 ids are not the DiT's
MF_PROMPT_HEAD = ("<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, "
                  "text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n")
MF_DROP = 34
# its edits' (the "mage-flow-edit" template): the system turn's 64 ids dropped, each reference "Image j: <pad>"
MF_EDIT_HEAD = ("<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, "
                "objects, background), then explain how the user's text instruction should alter or modify the image. "
                "Generate a new image that meets the user's requirements while maintaining consistency with the "
                "original input where appropriate.<|im_end|>\n<|im_start|>user\n")
MF_EDIT_DROP = 64
MF_EDIT_REF = "Image {i}: <|vision_start|><|image_pad|><|vision_end|>"
MF_MAX_REFS = 3                    # the edit models' training takes up to three references
MF_SHIFT = 6.0                     # Mage-Flow's static time shift
MF_NEG = " "                       # its pipeline's negative prompt when none is given
# an edit's reference image in the prompt: one pad, which the engine widens to the image's tokens
REF_PROMPT = "{sep}<image{i}><|vision_start|><|image_pad|><|vision_end|>"
REF_MAX_BYTES = 32 << 20
MAX_REFS = 10                      # Qwen-Image-2.1's own limit (its model card)
RAM_SPARE_MIB = 2048               # RAM an image job leaves free beside its prefix cache (the OS, the server)
START_TIMEOUT_S = 600              # the engine's start: the parts opened and a small image made (kernels built)


class ImageRequestError(ValueError):
    """A request the image endpoint refuses (400)."""


class ImageJobFailed(RuntimeError):
    """The image engine did not make the images (500); the message says why."""


def parse_size(value, max_side: int) -> tuple[int, int]:
    """OpenAI's size, "1024x1024" (or "auto": 1024x1024) -> (width, height); each side from 256 to max_side."""
    if value is None or value == "auto":
        return 1024, 1024
    if not isinstance(value, str):
        raise ImageRequestError("size: a string WIDTHxHEIGHT, e.g. \"1024x1024\"")
    m = re.fullmatch(r"\s*(\d{2,5})\s*[xX×]\s*(\d{2,5})\s*", value)
    if not m:
        raise ImageRequestError(f"size: WIDTHxHEIGHT, e.g. \"1024x1024\", not {value!r}")
    w, h = int(m.group(1)), int(m.group(2))
    for side in (w, h):
        if side < MIN_SIDE or side > max_side:
            raise ImageRequestError(f"size {w}x{h}: each side from {MIN_SIDE} to {max_side}")
    return w, h


def _int_field(req: dict, key: str, lo: int, hi: int, default):
    v = req.get(key)
    if v is None:
        return default
    if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
        raise ImageRequestError(f"{key}: a whole number from {lo} to {hi}")
    return v


def model_defaults(cfg: dict) -> dict:
    """What a request leaves out gets from the image model of the config's "images" section `cfg` (parse_request and
    cfg_request take theirs from here; the web page shows them): steps (Turbo's fixed, steps_fixed; Mage-Flow's edit
    model's for an edit), the least steps,
    the guidance scale (1: none), the negative prompt when none is given (None: none, so no guidance without one),
    whether it makes transparent images, and whether its edits take a mask."""
    mage = cfg.get("family") == "mage-flow"
    if cfg.get("schedule") == "base":
        steps, least, fixed = int(cfg.get("steps") or 40), 2, False
    elif cfg.get("schedule") == "mage":
        steps, least, fixed = int(cfg.get("steps") or 4), 1, False
    else:
        steps, least, fixed = TURBO_STEPS, TURBO_STEPS, True
    edit = int(cfg.get("edit_steps") or steps) if cfg.get("edit_model") else steps
    return {"steps": steps, "edit_steps": edit, "steps_min": least, "steps_max": MAX_STEPS, "steps_fixed": fixed,
            "guidance": float(cfg.get("cfg_scale") or 1.0) if mage else 1.0,
            "negative_prompt": MF_NEG if mage else None, "transparency": cfg.get("transparency") is not False,
            "mask": cfg.get("mask") is not False}


def parse_request(req: dict, cfg: dict) -> dict:
    """The fields of one /v1/images/generations request, checked.  output_format png, jpeg or webp
    (output_compression 0-100, the quality jpeg and webp are written at); quality low, medium or high (the base model's
    steps, QUALITY_STEPS; an explicit steps wins); background transparent (the prompt in Qwen's form for it, an image
    with alpha; not with jpeg), opaque (alpha flattened on white) or auto.  vLLM-Omni's fields too: num_inference_steps
    (steps by its name), negative_prompt, guidance_scale and true_cfg_scale (CFG_FIELDS).  Unknown fields (style,
    user, moderation) are accepted and ignored."""
    prompt = req.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ImageRequestError("prompt: a non-empty string")
    if len(prompt) > 16000:
        raise ImageRequestError("prompt: at most 16000 characters")
    if "\0" in prompt:
        raise ImageRequestError("prompt: no NUL characters")
    fmt = req.get("response_format") or "b64_json"
    if fmt not in ("b64_json", "url"):
        raise ImageRequestError("response_format: b64_json or url")
    out_fmt = req.get("output_format") or "png"
    if out_fmt not in FORMATS:
        raise ImageRequestError("output_format: png, jpeg or webp")
    compression = _int_field(req, "output_compression", 0, 100, 100)
    quality = req.get("quality")
    if quality is not None and quality not in QUALITY_STEPS:
        raise ImageRequestError("quality: low, medium, high or auto")
    background = req.get("background") or "auto"
    if background not in ("auto", "opaque", "transparent"):
        raise ImageRequestError("background: auto, opaque or transparent")
    if background == "transparent" and out_fmt == "jpeg":
        raise ImageRequestError("background transparent: png or webp (jpeg has no alpha channel)")
    if background == "transparent" and cfg.get("transparency") is False:
        raise ImageRequestError("background transparent: this image model does not make transparent images")
    w, h = parse_size(req.get("size"), int(cfg.get("max_side") or 2048))
    n = _int_field(req, "n", 1, int(cfg.get("max_n") or 4), 1)
    if "num_inference_steps" in req:            # vLLM-Omni's name for steps
        if req.get("steps") not in (None, req["num_inference_steps"]):
            raise ImageRequestError("steps and num_inference_steps: give one, or the same number")
        req = {**req, "steps": req["num_inference_steps"]}
    dflt = model_defaults(cfg)
    if cfg.get("schedule") == "base":
        steps = _int_field(req, "steps", 2, MAX_STEPS, QUALITY_STEPS.get(quality or "auto") or dflt["steps"])
    elif cfg.get("schedule") == "mage":   # Mage-Flow: any count on its static shift, the config's by default
        steps = _int_field(req, "steps", 1, MAX_STEPS, dflt["steps"])
    else:   # Turbo's own sigmas: steps asked for are ignored, as its diffusers pipeline (sample_sigmas) does
        _int_field(req, "steps", 1, MAX_STEPS, None)
        steps = dflt["steps"]
    cfg_fields = cfg_request(req, cfg)
    seed = _int_field(req, "seed", 0, 2**62, None)
    if seed is None:
        seed = random.SystemRandom().randrange(0, 2**31)
    return {"prompt": prompt, "width": w, "height": h, "n": n, "steps": steps, "seed": seed,
            "quality": quality or "auto", "response_format": fmt, "output_format": out_fmt,
            "output_compression": compression, "background": background, **cfg_fields}


def cfg_request(req: dict, cfg: dict | None = None) -> dict:
    """vLLM-Omni's guidance fields, checked: negative_prompt, guidance_scale and true_cfg_scale.  Qwen-Image-2.1 takes
    classifier-free guidance from true_cfg_scale above 1 with a negative_prompt, as diffusers' QwenImage21Pipeline
    does (without one, or at 1, none: "cfg" stays out of the job); guidance_scale it has no use for, and ignores.
    Mage-Flow (cfg "family" mage-flow) has one scale: true_cfg_scale, else guidance_scale, else the config's
    "cfg_scale"; above 1 it guides, with negative_prompt or its pipeline's " "."""
    out: dict = {}
    neg = req.get("negative_prompt")
    if neg is not None:
        if not isinstance(neg, str) or len(neg) > 16000 or "\0" in neg:
            raise ImageRequestError("negative_prompt: a string of at most 16000 characters")
        out["negative_prompt"] = neg
    for key in ("guidance_scale", "true_cfg_scale"):
        v = req.get(key)
        if v is None:
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= CFG_MAX:
            raise ImageRequestError(f"{key}: a number from 0 to {CFG_MAX:g}")
        out[key] = float(v)
    if (cfg or {}).get("family") == "mage-flow":
        dflt = model_defaults(cfg or {})
        scale = out.get("true_cfg_scale", out.get("guidance_scale", dflt["guidance"]))
        if scale > 1:
            out["cfg"] = scale
            out.setdefault("negative_prompt", dflt["negative_prompt"])
        return out
    if out.get("true_cfg_scale", 1.0) > 1 and out.get("negative_prompt") is not None:
        out["cfg"] = out["true_cfg_scale"]
    return out


def model_prompt(job: dict) -> str:
    """The prompt as the model reads it: a transparent image's in Qwen's form for one (TRANSPARENT_PROMPT)."""
    if job.get("background") == "transparent":
        return TRANSPARENT_PROMPT.format(prompt=job["prompt"].strip().rstrip("."))
    return job["prompt"]


def encode_image(png: bytes, fmt: str, compression: int, opaque: bool = False) -> bytes:
    """A made image (png_rgba's PNG) in OpenAI's output_format: PNG as it is, jpeg (no alpha: on white) and webp at
    `compression` as their quality (Pillow); `opaque` (background opaque): its alpha flattened on white."""
    if fmt == "png" and not opaque:
        return png
    import io
    from PIL import Image
    im = Image.open(io.BytesIO(png))
    if fmt == "jpeg" or opaque:
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        im = bg
    out = io.BytesIO()
    im.save(out, format=fmt.upper(), **({} if fmt == "png" else {"quality": compression}))
    return out.getvalue()


def base_sigmas(steps: int, width: int, height: int) -> list[float]:
    """Qwen-Image-2.1's sigmas for `steps` steps of a width x height image (the size the engine makes), and the
    terminal 0: diffusers' QwenImage21Pipeline with the FlowMatchEulerDiscreteScheduler its scheduler_config sets up
    (linspace(1, 1/steps), the exponential time shift by mu, a line in the image's tokens from base_shift 0.5 at 256 to
    max_shift 0.9 at 8192, then stretched to end at shift_terminal 0.02).  An image token is a 16 x 16 pixel square."""
    seq = (width // SIZE_STEP) * (height // SIZE_STEP)
    mu = 0.5 + (0.9 - 0.5) / (8192 - 256) * (seq - 256)
    t = [1.0 - i * (1.0 - 1.0 / steps) / (steps - 1) for i in range(steps)]
    shifted = [math.exp(mu) / (math.exp(mu) + (1.0 / x - 1.0)) for x in t]
    scale = (1.0 - shifted[-1]) / (1.0 - 0.02)
    return [1.0 - (1.0 - x) / scale for x in shifted] + [0.0]


def image_dims(data: bytes) -> tuple[int, int]:
    """(width, height) of a PNG, JPEG, GIF or BMP from its header; ImageRequestError for another format."""
    if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24:
        return struct.unpack(">II", data[16:24])
    if data[:6] in (b"GIF87a", b"GIF89a") and len(data) >= 10:
        return struct.unpack("<HH", data[6:10])
    if data[:2] == b"BM" and len(data) >= 26:
        w, h = struct.unpack("<ii", data[18:26])
        return w, abs(h)
    if data[:2] == b"\xff\xd8":
        i = 2
        while i + 9 < len(data):
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            if marker in (0xD8, 0x01, 0xFF) or 0xD0 <= marker <= 0xD7:
                i += 1 if marker == 0xFF else 2
                continue
            length = struct.unpack(">H", data[i + 2:i + 4])[0]
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):   # a frame header: its size
                h, w = struct.unpack(">HH", data[i + 5:i + 9])
                return w, h
            i += 2 + length
    raise ImageRequestError("a reference image: PNG, JPEG, GIF or BMP")


def _qimage():
    """tools/image/qimage.py, the one table of setup's --images choices (setup.py takes it the same way)."""
    tools = str(Path(__file__).resolve().parent.parent / "tools" / "image")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    import qimage
    return qimage


def setup_choice(cfg: dict) -> str:
    """The setup --images value that installs the image model of the config's "images" section `cfg` (its
    "model_name" is the model's name, which for Qwen-Image-2.1 is not the choice's)."""
    models = _qimage().IMAGE_MODELS
    name = cfg.get("model_name") or MODEL_NAME
    return next((k for k, v in models.items() if v[0] == name), _qimage().DEFAULT_MODEL)


def not_set_up_message() -> str:
    """What a request for an image is told when the server has no image generation (the 404)."""
    choices = [k for k in _qimage().IMAGE_MODELS if not k.endswith("-uncensored")]
    return ("image generation is not set up on this server: run setup with --images and an image model ("
            + ", ".join(choices) + "; -uncensored after one for its uncensored text encoder), see docs/IMAGE.md")


def parse_edit_request(req: dict, images: list[bytes], cfg: dict, mask: bytes | None = None) -> dict:
    """An edit (/v1/images/edits): parse_request's fields with `images` (the reference images' bytes, PNG, JPEG, GIF or
    BMP) as job["refs"].  size "auto" (or none): the first reference's shape at about 1024 x 1024 pixels, its sides
    multiples of SIZE_STEP.  `mask` (OpenAI's): a PNG of the first image's size whose transparent pixels are where it
    changes, the rest kept as it is (the engine's inpainting), as job["mask"]; refused by a model whose config says
    "mask": false."""
    mage = cfg.get("family") == "mage-flow"
    if not cfg.get("vision") or (mage and not cfg.get("edit_model")):
        raise ImageRequestError("edits need the vision encoder" + (" and the edit model" if mage else "") +
                                f": run setup with --images {setup_choice(cfg)} again")
    if not images:
        raise ImageRequestError("image: one reference image or more")
    max_refs = int(cfg.get("max_refs") or (MF_MAX_REFS if mage else MAX_REFS))
    if len(images) > max_refs:
        raise ImageRequestError(f"image: at most {max_refs} reference images")
    dims = []
    for data in images:
        if len(data) > REF_MAX_BYTES:
            raise ImageRequestError(f"image: at most {REF_MAX_BYTES >> 20} MiB each")
        w, h = image_dims(data)
        if w < 16 or h < 16:
            raise ImageRequestError(f"image: {w}x{h} is too small")
        dims.append((w, h))
    if mask is not None:
        mask = check_mask(mask, dims[0], cfg)
    if req.get("size") in (None, "auto"):
        w0, h0 = dims[0]
        side = int(cfg.get("max_side") or 2048)
        tw = (1024 * 1024 * w0 / h0) ** 0.5
        th = tw * h0 / w0
        tw, th = (min(side, max(MIN_SIDE, round(v / SIZE_STEP) * SIZE_STEP)) for v in (tw, th))
        req = {**req, "size": f"{tw}x{th}"}
    # Mage-Flow's edit model has its own steps
    job = parse_request(req, {**cfg, "steps": cfg.get("edit_steps") or cfg.get("steps")} if mage else cfg)
    job["refs"] = list(images)
    if mask is not None:
        job["mask"] = mask
    return job


def check_mask(data: bytes, size: tuple[int, int], cfg: dict) -> bytes:
    """An edit's mask, checked as OpenAI checks it (a PNG with an alpha channel, the first image's size) -> RGBA PNG
    bytes the engine reads."""
    if cfg.get("mask") is False:          # no "mask" (configs before it) is Qwen-Image-2.1, which takes one
        raise ImageRequestError("mask: this image model does not take a mask")
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ImageRequestError("mask: a PNG with an alpha channel")
    import io
    from PIL import Image
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except Exception as e:
        raise ImageRequestError(f"mask: the PNG could not be read ({e})") from None
    if "A" not in im.getbands() and "transparency" not in im.info:
        raise ImageRequestError("mask: a PNG with an alpha channel (its transparent pixels are where the image "
                                "changes)")
    if im.size != size:
        raise ImageRequestError(f"mask: {im.size[0]}x{im.size[1]}, not the image's {size[0]}x{size[1]}")
    out = io.BytesIO()
    im.convert("RGBA").save(out, format="PNG")
    return out.getvalue()


def ref_size(ref_w: int, ref_h: int) -> tuple[int, int]:
    """The size the engine works an edit's reference image at (qi_ref_size in src/image/qi_pipeline.cpp): diffusers'
    calculate_dimensions(1024 * 1024, its width over its height)."""
    ratio = ref_w / ref_h
    vw = math.sqrt(1024 * 1024 * ratio)
    return max(32, round(vw / 32) * 32), max(32, round(vw / ratio / 32) * 32)


def usage(job: dict) -> dict:
    """OpenAI's usage of an image job, in the image engine's tokens: the prompt's ids (the model's chat form) and the
    reference images' DiT tokens in, the made images' DiT tokens out (a token a 16 x 16 pixel square)."""
    mw, mh = made_size(job)
    text = len(job.get("ids") or ())
    sizes = (ref_size(*image_dims(r)) for r in job.get("refs") or ())
    refs = sum((w // SIZE_STEP) * (h // SIZE_STEP) for w, h in sizes)
    out = job["n"] * (mw // SIZE_STEP) * (mh // SIZE_STEP)
    return {"input_tokens": text + refs, "output_tokens": out, "total_tokens": text + refs + out,
            "input_tokens_details": {"text_tokens": text, "image_tokens": refs}}


def made_size(job: dict) -> tuple[int, int]:
    """The size the engine makes for `job`: each side rounded up to a multiple of SIZE_STEP."""
    def up(v) -> int:
        return -(-int(v) // SIZE_STEP) * SIZE_STEP
    return up(job["width"]), up(job["height"])


def png_rgba(rgba: bytes, width: int, height: int, made_w: int, made_h: int) -> bytes:
    """A PNG (8-bit RGBA, as stable-diffusion.cpp writes) of the middle width x height of an RGBA8 image of
    made_w x made_h.  Pillow's adaptive row filters at zlib's level 1: 58 ms and 1.5 MB for a 1024 x 1024 image,
    where filter 0 at level 6 took 100 ms and 2.0 MB (a tenth of a Turbo image's time on a B70)."""
    import io
    from PIL import Image
    x0, y0 = (made_w - width) // 2, (made_h - height) // 2
    im = Image.frombuffer("RGBA", (made_w, made_h), rgba, "raw", "RGBA", 0, 1)
    if (width, height) != (made_w, made_h):
        im = im.crop((x0, y0, x0 + width, y0 + height))
    out = io.BytesIO()
    im.save(out, format="PNG", compress_level=1)
    return out.getvalue()


def prompt_ids(tok, prompt: str, refs: int = 0, family: str = "qwen-image") -> list[int]:
    """The prompt in the model's chat form, tokenized (Qwen3-VL's tokenizer): the user's text plain, so a special
    token spelled in it stays text; an edit's `refs` reference images before it, one pad each (stable-diffusion.cpp's
    prompt for Qwen-Image-2.1's edits, the pads widened by the engine).  Mage-Flow's in its own forms (MF_PROMPT_HEAD;
    an edit's MF_EDIT_HEAD, each reference "Image j: <pad>" before the instruction)."""
    if family == "mage-flow":
        head = MF_EDIT_HEAD + "".join(MF_EDIT_REF.format(i=i + 1) for i in range(refs)) if refs else MF_PROMPT_HEAD
    else:
        head = PROMPT_HEAD + "".join(REF_PROMPT.format(sep=" " if i else "", i=i + 1) for i in range(refs))
    text = head + prompt + PROMPT_TAIL
    return tok.encode(text, parse_special=True, plain=[(len(head), len(head) + len(prompt))])


def mage_sigmas(steps: int, shift: float = MF_SHIFT) -> list[float]:
    """Mage-Flow's sigmas for `steps` and the terminal 0: shift t / (1 + (shift - 1) t), t from 1 down to 1 / steps
    (its FlowMatchEulerDiscreteScheduler with a static shift; the engine's own default for Turbo's 4)."""
    out = []
    for i in range(steps):
        t = 1.0 - (1.0 - 1.0 / steps) * i / (steps - 1) if steps > 1 else 1.0
        out.append(shift * t / (1.0 + (shift - 1.0) * t))
    return out + [0.0]


def load_tokenizer(gguf: str):
    """Qwen3-VL's tokenizer from the text encoder's GGUF (tools/strata_tokenizer.py)."""
    tools = str(Path(__file__).resolve().parent.parent / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    import strata_tokenizer
    return strata_tokenizer.Tokenizer.from_gguf(gguf)


def _kv(line: str) -> dict:
    out: dict = {}
    for part in line.split()[1:]:
        k, _, v = part.partition("=")
        try:
            out[k] = float(v) if "." in v else int(v)
        except ValueError:
            out[k] = v
    return out


# the environment that makes one image engine another: the GPU it takes and the libraries it loads
ENGINE_ENV = ("STRATA_GPU_PCI", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "ONEAPI_DEVICE_SELECTOR",
              "HIP_VISIBLE_DEVICES", "ZE_AFFINITY_MASK", "STRATA_VRAM_LIMIT_MIB", "STRATA_MAX_ALLOC_MIB",
              "STRATA_NO_XMX", "LD_LIBRARY_PATH")


def daemon_address(cmd: list[str], env: dict) -> str:
    """Where the image daemon (serve/image_daemon.py) of this engine command and GPU listens: a Unix socket's path, on
    Windows the file with its loopback port and token; one per user."""
    import hashlib
    import json
    key = hashlib.sha256(json.dumps([cmd, [env.get(k) for k in ENGINE_ENV]]).encode()).hexdigest()[:16]
    user = str(os.getuid()) if hasattr(os, "getuid") else re.sub(r"\W", "", os.environ.get("USERNAME", "user"))
    return str(Path(tempfile.gettempdir()) / f"xestrata-image-{user}-{key}.{'port' if os.name == 'nt' else 'sock'}")


def gpu_key(env: dict) -> str:
    """The GPU an engine environment names (its device selection; none: the system's first), for GpuTurn."""
    import hashlib
    sel = [env.get(k) for k in ("STRATA_GPU_PCI", "CUDA_VISIBLE_DEVICES", "ONEAPI_DEVICE_SELECTOR",
                                "HIP_VISIBLE_DEVICES", "ZE_AFFINITY_MASK")]
    return hashlib.sha256(json.dumps(sel).encode()).hexdigest()[:16]


class GpuTurn:
    """Every server's image jobs on one GPU, one at a time, and the image daemons started on it.  A file lock in the
    temp folder (the GPU's, gpu_key) is held for a job's whole turn, so a job of another server waits for it; a
    file beside it lists the daemons of the GPU, and the job that holds the turn ends every one but its own model's
    (another model's, or another server's: they are idle, having no turn), so two engines do not share the VRAM.
    Servers of the same model share its daemon, which is never ended for them."""

    def __init__(self, env: dict):
        user = str(os.getuid()) if hasattr(os, "getuid") else re.sub(r"\W", "", os.environ.get("USERNAME", "user"))
        base = Path(tempfile.gettempdir()) / f"xestrata-image-gpu-{user}-{gpu_key(env)}"
        self.lock_path, self.list_path = Path(str(base) + ".lock"), Path(str(base) + ".daemons")
        self.f: Any = None

    def _try(self, f) -> bool:
        try:
            if sys.platform == "win32":
                import msvcrt
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False

    def acquire(self, cancel: threading.Event | None = None, waiting=None) -> None:
        """Take the GPU's turn, waiting while another server's job holds it (`waiting()` once, when it does);
        ImageJobFailed when `cancel` is set meanwhile."""
        f = open(self.lock_path, "a+b")
        told = False
        while not self._try(f):
            if not told and waiting is not None:
                waiting()
                told = True
            if cancel is not None and cancel.is_set():
                f.close()
                raise ImageJobFailed("cancelled while another server's image job had the GPU")
            time.sleep(0.25)
        self.f = f

    def release(self) -> None:
        f, self.f = self.f, None
        if f is None:
            return
        with contextlib.suppress(OSError):
            if sys.platform == "win32":
                import msvcrt
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        f.close()                                     # flock's lock goes with the file

    def daemons(self) -> list[str]:
        try:
            got = json.loads(self.list_path.read_text(encoding="utf-8"))
            return [a for a in got if isinstance(a, str)] if isinstance(got, list) else []
        except (OSError, ValueError):
            return []

    def keep_only(self, addr: str) -> list[str]:
        """End every listed daemon but `addr` (the caller holds the turn) and list `addr` -> those that were ended."""
        ended = [a for a in self.daemons() if a != addr and stop_daemon_at(a)]
        tmp = self.list_path.with_suffix(".tmp")
        with contextlib.suppress(OSError):
            tmp.write_text(json.dumps([addr]), encoding="utf-8")
            os.replace(tmp, self.list_path)
        return ended


def stop_daemon_at(addr: str) -> bool:
    """End the image daemon at `addr` and its engine (SHUTDOWN: the engine frees its memory itself) -> whether one
    was there.  Its READY is read first (a connection closed before it ends the daemon's handling unread) and its
    STOPPED, the engine gone, waited for (a daemon without it closes the connection instead)."""
    e = ImageEngine({}, {}, Path(os.devnull), lambda text: None)
    if not e._connect(addr):
        return False
    try:
        e._next(0, 30)
        e.send("SHUTDOWN")
        while e._next(0, 60) != "STOPPED":
            pass
    except ImageJobFailed:
        pass
    e.close()
    return True


class ImageEngine:
    """The image engine, `strata-image --serve`, through the image daemon (serve/image_daemon.py): started outside the
    server the first time and found again by the next server of the same image model and GPU, so a model switch does
    not lose it; one request line at a time (the caller serializes, the daemon between servers)."""

    def __init__(self, cfg: dict, env: dict, log_path: Path, say):
        self.cfg, self.env, self.log_path, self.say = cfg, env, log_path, say
        self.sock: Any = None
        self.f: Any = None
        self.lines: queue.Queue = queue.Queue()
        self.ended = threading.Event()              # the daemon closed this connection

    def command(self) -> list[str]:
        cmd = [str(self.cfg["exe"]), "--dit", str(self.cfg["diffusion_model"]), "--vae", str(self.cfg["vae"]),
               "--text", str(self.cfg["llm"])]
        if self.cfg.get("vision"):
            cmd += ["--vision", str(self.cfg["vision"])]
        if self.cfg.get("dit_q8") is not None:   # Mage-Flow's DiT: 8 bits or FP16 as the config says, else by the VRAM
            cmd += ["--dit-q8", "1" if self.cfg["dit_q8"] else "0"]
        if self.cfg.get("edit_model"):
            cmd += ["--edit-dit", str(self.cfg["edit_model"])]
        if self.cfg.get("dit_keep") is False:
            cmd += ["--dit-keep", "0"]
        return cmd + ["--serve"]

    def alive(self) -> bool:
        """Connected, and the daemon has not closed the connection (another server's job may have ended it, GpuTurn):
        a closed one is let go, so the next job starts the engine again."""
        if self.f is not None and self.ended.is_set():
            self.close()
        return self.f is not None

    def _connect(self, addr: str) -> bool:
        import socket
        try:
            if os.name != "nt":
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.connect(addr)
                token = None
            else:
                port, token = Path(addr).read_text(encoding="utf-8").split()
                s = socket.create_connection(("127.0.0.1", int(port)), timeout=10)
                s.settimeout(None)
        except (OSError, ValueError):
            return False
        self.sock = s
        self.f = s.makefile("rw", encoding="utf-8", errors="replace", newline="\n")
        if token is not None:
            self.f.write(token + "\n")
            self.f.flush()
        self.lines = queue.Queue()
        self.ended = threading.Event()

        def pump(f, q, ended):
            try:
                for line in f:
                    q.put(line.rstrip("\r\n"))
            except (OSError, ValueError):
                pass
            ended.set()
            q.put(None)                               # the daemon closed the connection
        threading.Thread(target=pump, args=(self.f, self.lines, self.ended), daemon=True).start()
        return True

    def _spawn(self, addr: str):
        """Start the daemon (its environment the server's engine's), detached from this server."""
        cmd = [sys.executable, str(Path(__file__).resolve().parent / "image_daemon.py"), "--socket", addr,
               "--idle", str(float(self.cfg.get("daemon_idle_s") or 1800)), "--log", str(self.log_path), "--",
               *self.command()]
        kw: dict = {"start_new_session": True} if os.name != "nt" else \
            {"creationflags": 0x00000200 | 0x00000008 | 0x01000000}   # its own group, no console, out of the job
        try:
            return subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL, env=self.env, **kw)
        except OSError:
            if os.name != "nt":
                raise
            kw["creationflags"] = 0x00000200 | 0x00000008                # a job that allows no breakaway
            return subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL, env=self.env, **kw)

    def start(self) -> dict:
        """Connect to the image daemon, starting it when there is none, and wait for READY -> its fields.
        ImageJobFailed when it does not start."""
        addr = daemon_address(self.command(), self.env)
        t0 = time.monotonic()
        proc = None
        lock = addr + ".lock"
        while not self._connect(addr):
            if proc is None:
                try:   # one server starts it; another waits for its socket
                    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                    os.close(fd)
                    proc = self._spawn(addr)
                except FileExistsError:
                    with contextlib.suppress(OSError):
                        if time.time() - os.path.getmtime(lock) > START_TIMEOUT_S:
                            os.unlink(lock)               # a start that never finished
                except OSError as e:
                    raise ImageJobFailed(f"the image engine did not start: {e}") from e
            elif proc.poll() is not None:
                with contextlib.suppress(OSError):
                    os.unlink(lock)
                raise ImageJobFailed(f"the image engine did not start; its log: {self.log_path}")
            if time.monotonic() - t0 > START_TIMEOUT_S:
                raise ImageJobFailed(f"the image engine did not start in {START_TIMEOUT_S} s; its log: {self.log_path}")
            time.sleep(0.25)
        if proc is not None:
            with contextlib.suppress(OSError):
                os.unlink(lock)
        line = self._next(0, START_TIMEOUT_S)
        if not line.startswith("READY"):
            self.close()
            raise ImageJobFailed(line[6:] if line.startswith("ERROR") else f"the image engine said: {line}")
        self.say(f"image engine {'started' if proc else 'found'} in {time.monotonic() - t0:.1f} s ({addr})")
        return _kv(line)

    def _next(self, waited: float, limit: float, cancel: threading.Event | None = None) -> str:
        """The engine's next line; ImageJobFailed when the connection ended or `limit` passed."""
        while True:
            try:
                line = self.lines.get(timeout=0.25)
            except queue.Empty:
                if cancel is not None and cancel.is_set():
                    self.send("STOP")
                    cancel = None                     # the engine answers ERROR stopped after its step
                    continue
                if waited > limit:
                    self.close()
                    raise ImageJobFailed(f"the image engine did not answer in {limit:g} s; its log: "
                                         f"{self.log_path}")
                waited += 0.25
                continue
            if line is None:
                self.close()
                raise ImageJobFailed(f"the image engine ended; its log: {self.log_path}")
            return line

    def send(self, line: str) -> None:
        f = self.f
        try:
            if f is None:
                raise OSError("not connected")
            f.write(line + "\n")
            f.flush()
        except (OSError, ValueError) as e:
            self.close()
            raise ImageJobFailed(f"the image engine is gone ({e}); its log: {self.log_path}") from e

    def ask(self, line: str, want: str, timeout: float, cancel: threading.Event | None = None, step=None,
            stage=None) -> dict:
        """Send a request line and wait for its answer (the line starting with `want`) -> its fields; STEP lines go to
        `step`, STAGE lines (refs, text, sampling, decode) to `stage`.  ImageJobFailed on ERROR."""
        self.send(line)
        t0 = time.monotonic()
        while True:
            got = self._next(time.monotonic() - t0, timeout, cancel)
            if got.startswith("STEP "):
                if step is not None:
                    done, steps = got.split()[1:3]
                    step(int(done), int(steps))
            elif got.startswith("STAGE "):
                if stage is not None:
                    stage(got[6:].strip())
            elif got.startswith(want):
                return _kv(got)
            elif got.startswith("ERROR"):
                raise ImageJobFailed(got[6:] or "the image engine failed")

    def close(self, shutdown: bool = False) -> None:
        """Leave the daemon (it stays for the next server); `shutdown`: end it and its engine."""
        f, sock, self.f, self.sock = self.f, self.sock, None, None
        if f is not None:
            with contextlib.suppress(OSError, ValueError):
                f.write(("SHUTDOWN" if shutdown else "QUIT") + "\n")
                f.flush()
        if sock is not None:   # the reader blocked in the file first: closing the file under it would wait for it
            import socket
            with contextlib.suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)
        if f is not None:
            with contextlib.suppress(OSError, ValueError):
                f.close()
        if sock is not None:
            with contextlib.suppress(OSError):
                sock.close()


class ImageGen:
    """The resident image engine, one request at a time (the service's turn, gpu_turn)."""

    def __init__(self, cfg: dict, log=None, env: dict | None = None):
        self.cfg = dict(cfg)
        policy = self.cfg.get("gpu_policy") or "auto"
        if policy not in POLICIES:
            raise ValueError(f'images "gpu_policy" must be one of {", ".join(POLICIES)}, not {policy!r}')
        reload = self.cfg.get("reload") or "lazy"
        if reload not in ("lazy", "eager"):
            raise ValueError(f'images "reload" must be lazy or eager, not {reload!r}')
        for k in ("exe", "diffusion_model", "vae", "llm"):
            if not isinstance(self.cfg.get(k), str) or not self.cfg[k]:
                raise ValueError(f'the config\'s "images" section needs "{k}" (run setup with --images again)')
        if Path(self.cfg["exe"]).name.startswith("sd-cli"):
            raise ValueError('the config\'s "images" section is for stable-diffusion.cpp, which XeStrata no longer '
                             "uses (run setup with --images again)")
        self.policy, self.reload = policy, reload
        self.drop = MF_DROP if self.cfg.get("family") == "mage-flow" else DROP   # the system turn's ids
        self.model = self.cfg.get("model_name") or MODEL_NAME
        self.log = log
        out = self.cfg.get("out_dir")
        self.dir = Path(out) if out else Path(tempfile.mkdtemp(prefix="strata-images-"))
        self.dir.mkdir(parents=True, exist_ok=True)
        self.keep_files = int(self.cfg.get("keep_files") or 64)
        self.reserve = int(self.cfg.get("reserve_mib") or 512)
        self.spare = int(self.cfg.get("spare_mib") or 1024)
        self.lock = threading.Lock()              # one request line at a time even when the service's turn is not taken
        self.engine = ImageEngine(self.cfg, dict(os.environ) if env is None else env, self.dir / "strata-image.log",
                                  self._say)
        self.tok = None
        self.totals = {"images": 0, "jobs": 0, "seconds": 0.0, "failed": 0}
        self.last: dict | None = None              # the last job's figures (/v1/status)
        self.progress: tuple | None = None         # (step, steps) of the running job
        # the job in the GPU's turn as the Monitor shows it (live()), and the jobs that ended, newest last (recent()):
        # the one record of an image job's progress, which /metrics reads
        self.job: dict | None = None
        self.ended: collections.deque[dict] = collections.deque(maxlen=100)

    def note(self) -> str | None:
        """What the user is told about this setup at start and on the web page: the sizes of the image model and its
        text encoder lowered for the RAM (setup's "quant_lowered"), or None."""
        q = self.cfg.get("quant_lowered")
        if not isinstance(q, dict) or not q.get("from") or not q.get("to"):
            return None
        parts = []
        if q["to"] != q["from"]:
            parts.append(f"the image model at {q['to']} instead of {q['from']}")
        if q.get("text_from") and q.get("text_to") and q["text_to"] != q["text_from"]:
            parts.append(f"its text encoder at {q['text_to']} instead of {q['text_from']}")
        if not parts:
            return None
        return (f"Image generation runs {' and '.join(parts)}: the larger sizes need about "
                f"{q.get('need_gb', '?')} GB of RAM beside the text model, which leaves about "
                f"{q.get('ram_left_gb', '?')} GB. Run setup with --image-ssd-offload on to keep them "
                "(what does not fit is read from the SSD, slower).")

    def defaults(self) -> dict:
        """What a request leaves out gets from this image model (model_defaults)."""
        return model_defaults(self.cfg)

    def missing(self) -> list[str]:
        keys = ("exe", "diffusion_model", "vae", "llm") + (("vision",) if self.cfg.get("vision") else ())
        return [self.cfg[k] for k in keys if not Path(self.cfg[k]).exists()]

    def _say(self, text: str) -> None:
        print(f"[strata] {text}", flush=True)
        if self.log is not None:
            try:
                self.log.write(f"[images] {text}\n")
                self.log.flush()
            except (OSError, ValueError):
                pass

    def _prune(self) -> None:
        """Keep the newest keep_files images of response_format url (the oldest go)."""
        files = sorted((p for p in self.dir.iterdir() if FILE_NAME.match(p.name)), key=lambda p: p.stat().st_mtime)
        for p in files[:max(0, len(files) - self.keep_files)]:
            with contextlib.suppress(OSError):
                p.unlink()

    def file(self, name: str) -> Path | None:
        """A stored image (response_format url) by its name, or None."""
        if not FILE_NAME.match(name or ""):
            return None
        p = self.dir / name
        return p if p.is_file() else None

    def _ready(self) -> None:
        if self.tok is None:
            self.tok = load_tokenizer(self.cfg["llm"])
        if not self.engine.alive():
            self.engine.start()

    def ids(self, job: dict) -> list[int]:
        if self.tok is None:
            self.tok = load_tokenizer(self.cfg["llm"])
        if "ids" not in job:
            fam = self.cfg.get("family") or "qwen-image"
            job["ids"] = prompt_ids(self.tok, model_prompt(job), len(job.get("refs") or ()), fam)
            if job.get("cfg"):   # guidance's negative prompt, in the same form (its pads as the prompt's)
                job["neg_ids"] = prompt_ids(self.tok, job["negative_prompt"], len(job.get("refs") or ()), fam)
        return job["ids"]

    def drop_of(self, job: dict) -> int:
        """The system turn's ids the DiT does not take: Mage-Flow's edit form has its own."""
        return MF_EDIT_DROP if self.cfg.get("family") == "mage-flow" and job.get("refs") else self.drop

    @staticmethod
    def _cfg_words(job: dict) -> str:
        """IMG's guidance words: the negative prompt's ids and the scale (none without guidance)."""
        if not job.get("cfg"):
            return ""
        return f" NEG:{','.join(str(i) for i in job['neg_ids'])} CFG:{job['cfg']:g}"

    def stage(self, job: dict) -> None:
        """An edit's reference images and mask into files the engine reads (job["ref_paths"], job["mask_path"]);
        unstage removes them."""
        if job.get("refs") and not job.get("ref_paths"):
            work = Path(tempfile.mkdtemp(prefix="refs-", dir=self.dir))
            job["ref_dir"] = str(work)
            job["ref_paths"] = []
            for i, data in enumerate(job["refs"]):
                p = work / f"ref_{i}"
                p.write_bytes(data)
                job["ref_paths"].append(str(p))
            if job.get("mask"):
                p = work / "mask"
                p.write_bytes(job["mask"])
                job["mask_path"] = str(p)

    @staticmethod
    def unstage(job: dict) -> None:
        for p in (job.pop("ref_paths", []) or []) + ([job.pop("mask_path")] if job.get("mask_path") else []):
            with contextlib.suppress(OSError):
                os.unlink(p)
        if job.get("ref_dir"):
            with contextlib.suppress(OSError):
                os.rmdir(job.pop("ref_dir"))

    @staticmethod
    def _refs_words(job: dict) -> str:
        return ("".join(f" REF:{p}" for p in job.get("ref_paths") or ())
                + (f" MASK:{job['mask_path']}" if job.get("mask_path") else ""))

    def need(self, job: dict) -> dict:
        """What the image needs, from the engine (started when it is not running): least_mib, full_mib (every layer
        resident), held_mib (what it holds now), free_mib (the device's free VRAM), cap_mib (the most the engine could
        have) and host_mib (the prefix cache in pinned RAM at cap_mib)."""
        with self.lock:
            self._ready()
            w, h = made_size(job)
            self.stage(job)
            n = len(self.ids(job)) - self.drop_of(job)
            neg = f" NEG:{len(job['neg_ids']) - self.drop_of(job)}" if job.get("cfg") else ""
            return self.engine.ask(f"NEED {w} {h} {n}{self._refs_words(job)}{neg}", "NEED", 600)

    def free_mib(self) -> int:
        """The device's free VRAM, as the engine sees it (its parts left as they are)."""
        with self.lock:
            self._ready()
            return int(self.engine.ask("FREE", "FREE", 60)["free_mib"])

    def fit(self) -> int:
        """The engine's budget set to what it holds and what is free now, less the reserve -> the budget (MiB)."""
        with self.lock:
            self._ready()
            return int(self.engine.ask(f"FIT {self.reserve}", "BUDGET", 600).get("budget_mib", 0))

    def release(self, everything: bool) -> None:
        """The engine lets go of its cache (layers, activations, buffers); `everything`: its fixed weights too."""
        with self.lock:
            if self.engine.alive():
                try:
                    self.engine.ask("RELEASE ALL" if everything else "RELEASE", "RELEASE", 120)
                except ImageJobFailed as e:
                    self._say(f"image engine: release failed ({e})")

    def close(self, shutdown: bool = False) -> None:
        """Leave the image daemon (it stays, warm, for the next server: a model switch); `shutdown`: end it."""
        with self.lock:
            self.engine.close(shutdown)

    def run(self, job: dict, cancel: threading.Event | None = None) -> tuple[list[bytes], dict]:
        """Make `job`'s images (parse_request's dict) -> (the PNG files' bytes in order, figures).  The caller holds
        the GPU turn.  ImageJobFailed when the engine fails, times out or the job is cancelled."""
        timeout = float(self.cfg.get("timeout_s") or 3600)
        mw, mh = made_size(job)
        t0 = time.monotonic()
        images, parts = [], []
        work = Path(tempfile.mkdtemp(prefix="job-", dir=self.dir))
        try:
            with self.lock:
                self._ready()
                self.stage(job)
                ids = ",".join(str(i) for i in self.ids(job))
                sched = self.cfg.get("schedule")
                sig = (base_sigmas(job["steps"], mw, mh) if sched == "base" else
                       mage_sigmas(job["steps"]) if sched == "mage" else [])
                sigmas = " SIGMAS:" + ",".join(f"{x:.7g}" for x in sig) if sig else ""
                for i in range(job["n"]):
                    if cancel is not None and cancel.is_set():
                        self.totals["failed"] += 1
                        raise ImageJobFailed("cancelled (the client left)")
                    self.progress = (i * job["steps"], job["n"] * job["steps"])
                    self._live(image=i + 1, step=0, steps=job["steps"], stage="refs" if job.get("refs") else "text",
                               sampling_at=None)
                    out = work / f"img_{i}.rgba"

                    def step(done, steps, i=i):
                        self.progress = (i * steps + done, job["n"] * steps)
                        self._live(step=done, steps=steps)

                    def stage(name):
                        self._live(stage=name, **({"sampling_at": time.monotonic()} if name == "sampling" else {}))
                    try:
                        r = self.engine.ask(f"IMG {mw} {mh} {job['seed'] + i} {self.drop_of(job)} {out} {ids}{sigmas}"
                                            f"{self._refs_words(job)}{self._cfg_words(job)}", "IMAGE", timeout,
                                            cancel, step, stage)
                    except ImageJobFailed as e:
                        self.totals["failed"] += 1
                        why = "cancelled (the client left)" if str(e) == "stopped" else str(e)
                        self._say(f"image job failed: {why}")
                        raise ImageJobFailed(why) from e
                    images.append(png_rgba(out.read_bytes(), job["width"], job["height"], mw, mh))
                    parts.append(r)
        finally:
            self.progress = None
            self.unstage(job)
            for p in work.iterdir():
                with contextlib.suppress(OSError):
                    p.unlink()
            with contextlib.suppress(OSError):
                work.rmdir()
        secs = time.monotonic() - t0
        figures = {"seconds": round(secs, 2), "steps": job["steps"], "cfg": job.get("cfg") or 1.0, "n": job["n"],
                   "refs": len(job.get("refs") or ()),
                   "size": f"{job['width']}x{job['height']}", "seed": job["seed"], "engine": "strata-image",
                   "quant": self.cfg.get("quant"),
                   "text_encode_ms": round(sum(r.get("text_s", 0) for r in parts) * 1000, 1),
                   "sampling_s": round(sum(r.get("sampling_s", 0) for r in parts), 2),
                   "vae_decode_s": round(sum(r.get("decode_s", 0) for r in parts), 2),
                   "plan": {k: v for k, v in parts[-1].items() if k not in ("text_s", "sampling_s", "decode_s")}}
        if (mw, mh) != (job["width"], job["height"]):      # made larger and cut: the web page says so
            figures["made_size"] = f"{mw}x{mh}"
        self.totals["jobs"] += 1
        self.totals["images"] += job["n"]
        self.totals["seconds"] += secs
        self.last = figures
        return images, figures

    def store(self, data: bytes, fmt: str = "png") -> str:
        """Keep one image for response_format url -> its file name (32 random hex digits: the name is the access, and
        the output_format)."""
        name = f"{secrets.token_hex(16)}.{fmt}"
        (self.dir / name).write_bytes(data)
        self._prune()
        return name

    def response(self, images: list[bytes], job: dict, figures: dict, base_url: str) -> dict:
        data = []
        fmt = job.get("output_format") or "png"
        for i, png in enumerate(images):
            item = {"revised_prompt": model_prompt(job), "seed": job["seed"] + i}
            img = encode_image(png, fmt, job.get("output_compression", 100), job.get("background") == "opaque")
            if job["response_format"] == "url":
                item["url"] = f"{base_url}/v1/images/files/{self.store(img, fmt)}"
            else:
                item["b64_json"] = base64.b64encode(img).decode("ascii")
            data.append(item)
        return {"created": int(time.time()), "data": data, "model": self.model, "output_format": fmt,
                "size": f"{job['width']}x{job['height']}", "background": job.get("background") or "auto",
                "quality": job.get("quality") or "auto", "usage": usage(job), "strata": figures}

    # ---- the progress the Monitor shows (/metrics' live.image) and the jobs that ended (its Requests table)
    STAGE_LABELS = {"waiting": "Making room on the GPU", "refs": "Reading the reference images",
                    "text": "Reading the prompt", "sampling": "Sampling", "decode": "Decoding the image (VAE)"}

    def begin(self, job: dict) -> None:
        """The job has the GPU's turn (gpu_turn): from here it is the running job."""
        self.job = {"started": time.time(), "t0": time.monotonic(), "stage": "waiting", "image": 1, "images": job["n"],
                    "step": 0, "steps": job["steps"], "size": f"{job['width']}x{job['height']}",
                    "refs": len(job.get("refs") or ()), "edit": bool(job.get("refs")), "sampling_at": None}

    def _live(self, **kw) -> None:
        if self.job is not None:
            self.job.update(kw)

    def live(self) -> dict | None:
        """The running job's stage, step n of the steps, the seconds so far and a step, its size and image i of n, and
        its overall progress (0..1: each image a share; its sampling by the steps, the stages around it at their
        boundaries)."""
        j = dict(self.job) if self.job is not None else None
        if j is None:
            return None
        now = time.monotonic()
        stage, done, steps = j["stage"], j["step"], max(1, j["steps"])
        step_s = round((now - j["sampling_at"]) / done, 2) if j["sampling_at"] is not None and done > 0 else None
        within = {"waiting": 0.0, "refs": 0.02, "text": 0.05, "sampling": 0.1 + 0.8 * done / steps,
                  "decode": 0.9}.get(stage, 0.0)
        frac = (j["image"] - 1 + within) / max(1, j["images"])
        return {"stage": stage, "label": self.STAGE_LABELS.get(stage, stage), "step": done, "steps": j["steps"],
                "image": j["image"], "images": j["images"], "size": j["size"], "refs": j["refs"], "edit": j["edit"],
                "elapsed_s": round(now - j["t0"], 1), "step_s": step_s, "progress": round(min(1.0, frac), 3)}

    def end(self, error: str | None, cancelled: bool = False) -> None:
        """The running job ended: one row of the Monitor's Requests table (the time it got the GPU, its size, the
        references, the seconds and each stage's, done or the error)."""
        j, self.job = self.job, None
        if j is None:
            return
        last = self.last if error is None and self.last is not None else {}
        self.ended.append({"kind": "image", "time": j["started"], "duration_s": round(time.monotonic() - j["t0"], 1),
                           "finish": "stop" if error is None else "cancel" if cancelled else "error",
                           "error": None if cancelled else error,
                           "size": j["size"], "n": j["images"], "refs": j["refs"], "edit": j["edit"],
                           "seconds": last.get("seconds"), "text_encode_ms": last.get("text_encode_ms"),
                           "sampling_s": last.get("sampling_s"), "vae_decode_s": last.get("vae_decode_s")})

    def recent(self) -> list[dict]:
        return list(self.ended)

    def status(self) -> dict:
        out = {"enabled": True, "model": self.model, "engine": "strata-image", "quant": self.cfg.get("quant"),
               "gpu_policy": self.policy, "running": self.engine.alive(), **self.totals}
        if self.note():
            out["note"] = self.note()
        if self.progress:
            out["step"], out["steps"] = self.progress
        if self.last:
            out["last"] = self.last
        return out


class ImageModels:
    """The config's image models: its "images" section (the default) and those of "image_models", one engine on the
    GPU at a time.  A request names its model by "model" (pick: an unknown name or none is the default's, as OpenAI's
    clients send their own); its job takes the GPU's turn (GpuTurn: other servers' jobs on the GPU wait for it, and
    it for theirs) and ends every other daemon of the GPU first, this server's other models' and other servers'.
    Anything else is the model of the last job's (the Monitor's, /v1/status)."""

    def __init__(self, cfgs: list[dict], log=None, env: dict | None = None):
        self.gens: dict[str, ImageGen] = {}
        for c in cfgs:
            g = ImageGen(c, log=log, env=env)
            if g.model in self.gens:
                raise ValueError(f'two image models are named {g.model!r} ("images" and "image_models")')
            self.gens[g.model] = g
        self.default = self.current = next(iter(self.gens.values()))
        self.gpu = GpuTurn(self.default.engine.env)   # the models share the text engine's environment, so its GPU

    def __getattr__(self, name):
        return getattr(self.current, name)

    def pick(self, name) -> ImageGen:
        return self.gens.get(name, self.default) if isinstance(name, str) else self.default

    def activate(self, gen: ImageGen) -> None:
        """Make `gen` the model on the GPU: every other daemon of the GPU ended, this server's other models' and other
        servers' (the caller holds the GPU's turn, GpuTurn)."""
        mine = daemon_address(gen.engine.command(), gen.engine.env)
        names = {daemon_address(g.engine.command(), g.engine.env): g.model for g in self.gens.values()}
        for addr in self.gpu.keep_only(mine):
            gen._say(f"image model: {names.get(addr, 'another server')}'s engine ended for {gen.model}")
        self.current = gen

    def missing(self) -> list[str]:
        return [f for g in self.gens.values() for f in g.missing()]

    def recent(self) -> list[dict]:
        """The jobs that ended, every model's, oldest first."""
        return sorted((r for g in self.gens.values() for r in g.recent()), key=lambda r: r.get("time", 0))

    def status(self) -> dict:
        return {**self.current.status(), "models": list(self.gens), "default": self.default.model}

    def file(self, name: str) -> Path | None:
        return next((f for g in self.gens.values() if (f := g.file(name)) is not None), None)

    def close(self, shutdown: bool = False) -> None:
        for g in self.gens.values():
            g.close(shutdown)

    def models(self) -> list[dict]:
        """Each model's name, defaults and whether it edits (the web page's choice), the default first."""
        return [{"id": g.model, "defaults": g.defaults(), "edits": bool(g.cfg.get("vision"))}
                for g in self.gens.values()]


# ------------------------------------------------------------------------------------------- chat's image tools
OPTION_KEYS = ("size", "quality", "output_format", "output_compression", "background")
FILE_REF = re.compile(r"/v1/images/files/([0-9a-f]{32}\.(?:png|jpeg|webp))")
TOOLS: list[dict[str, Any]] = [
    {"name": "generate_image",
     "description": "Make an image from a description with the image model on this computer. "
                    "Describe it in detail: the subject, the setting, the style, the light, and any text it shows in "
                    "quotes. The image is shown to the user; the result says where it is.",
     "parameters": {"type": "object", "properties": {
         "prompt": {"type": "string", "description": "the image's description"},
         "size": {"type": "string",
                  "description": "WIDTHxHEIGHT, e.g. 1024x1024, 1536x1024 (wide), 1024x1536 (tall)"},
         "seed": {"type": "integer", "description": "the same seed and prompt make the same image"}},
         "required": ["prompt"]}},
    {"name": "edit_image",
     "description": "Change an image, or make a new one from it (its subject, its style): the conversation's last "
                    "image (one the user attached or one made here) unless `image` names another (a "
                    "/v1/images/files/... link from an earlier result). The new image is shown to the user.",
     "parameters": {"type": "object", "properties": {
         "prompt": {"type": "string", "description": "what to change, or what to make from the image"},
         "image": {"type": "string", "description": "the image: a /v1/images/files/... link; default the last one"},
         "size": {"type": "string", "description": "WIDTHxHEIGHT; default the image's shape at about 1024x1024"},
         "seed": {"type": "integer"}},
         "required": ["prompt"]}},
]


class ImageTools:
    """generate_image and edit_image for a chat (or Responses) request that asks for them: the model writes a call,
    the server makes the image in the GPU's turn as /v1/images does, keeps it (/v1/images/files/), and the call's
    result tells the model where it is.  With several image models (ImageModels) the tools take a "model" too,
    each model named and described in it, so the user's words choose one ("draw it with MageTrail", "an anime
    illustration"); none named is the default.  A request that names its image model (`model`) gets that one
    alone; a Responses request's input_image_mask (`mask`) is the mask of its edits.  The hub run_with_mcp runs
    (template_tools, routes, call, settings)."""

    def __init__(self, svc, images, messages: list, load, normalize, options: dict | None = None,
                 model: str | None = None, mask: bytes | None = None):
        self.models = images if isinstance(images, ImageModels) else None
        gen = images.default if isinstance(images, ImageModels) else images
        if model is not None:   # the request's own image model (strata_images {"model": ...}): that one alone
            gen = images.pick(model) if isinstance(images, ImageModels) else gen
            if gen.model != model:
                raise ValueError(f"strata_images: model must be one of "
                                 f"{', '.join(images.gens if isinstance(images, ImageModels) else [gen.model])}")
            self.models = None
        self.svc, self.gen, self.messages, self.load, self.normalize = svc, gen, messages, load, normalize
        # a Responses request's image_generation tool: its size, quality, output_format, output_compression and
        # background, over what the model's call says (size "auto": the call's)
        self.options = {k: v for k, v in (options or {}).items()
                        if k in OPTION_KEYS and v is not None and not (k == "size" and v == "auto")}
        self.settings = {"max_rounds": 4}
        self.made: list[str] = []          # the images this request made (file names)
        self.mask = mask                   # an edit's mask (Responses' input_image_mask), over the image it edits

    def _gens(self, edits: bool = False) -> list[ImageGen]:
        """The image models the tools offer (those that edit, for edit_image), the default first."""
        gens = list(self.models.gens.values()) if self.models is not None else [self.gen]
        return [g for g in gens if g.cfg.get("vision")] if edits else gens

    def template_tools(self, exclude=()) -> list[dict]:
        out = []
        for t in TOOLS:
            gens = self._gens(t["name"] == "edit_image")
            if not gens or t["name"] in exclude:
                continue
            t = dict(t)
            if len(self._gens()) > 1:   # the model's choice: each one's name and what it is good for
                about = _qimage().ABOUT
                choice = {v[0]: k.removesuffix("-uncensored") for k, v in _qimage().IMAGE_MODELS.items()}
                lines = "; ".join(f"{g.model}: {about.get(choice.get(g.model, ''), '')}".rstrip(": ") for g in gens)
                params = dict(t["parameters"])
                params["properties"] = {**params["properties"], "model": {
                    "type": "string", "enum": [g.model for g in gens],
                    "description": f"the image model, when the user asks for one or for what one is made for "
                                   f"({lines}); default {gens[0].model}"}}
                t["parameters"] = params
            out.append(t)
        return out

    def routes(self) -> dict:
        return {t["name"]: (None, t["name"]) for t in self.template_tools()}

    def _file(self, name: str) -> bytes | None:
        p = (self.models or self.gen).file(name)
        return p.read_bytes() if p is not None else None

    def _image(self, which: str) -> bytes | None:
        """The image an edit takes: a /v1/images/files/ link (or its file name), else the conversation's last."""
        m = FILE_REF.search(which or "") or re.fullmatch(r"([0-9a-f]{32}\.(?:png|jpeg|webp))", which or "")
        if m:
            return self._file(m.group(1))
        if self.made:
            return self._file(self.made[-1])
        for msg in reversed(self.messages):
            content = msg.get("content")
            parts = content if isinstance(content, list) else [content]
            for item in reversed(parts):
                if isinstance(item, dict) and item.get("type") == "image" and item.get("source"):
                    return self.load(item["source"])
                text = item if isinstance(item, str) else item.get("text") if isinstance(item, dict) else None
                found = FILE_REF.findall(text or "")
                if found and (data := self._file(found[-1])) is not None:
                    return data
        return None

    def call(self, name: str, arguments, cancel: threading.Event | None = None) -> dict:
        t0 = time.monotonic()
        args = arguments if isinstance(arguments, dict) else {}
        urls: list[str] = []
        try:
            req = {**{k: args[k] for k in ("prompt", "size", "seed") if args.get(k) not in (None, "")}, **self.options}
            gens = self._gens(name == "edit_image")
            asked = args.get("model")
            gen = next((g for g in gens if g.model == asked), gens[0] if gens else self.gen)
            if asked not in (None, "") and gen.model != asked:
                raise ImageRequestError(f"model: one of {', '.join(g.model for g in gens)}, not {asked!r}")
            if name == "generate_image":
                job = parse_request(req, gen.cfg)
            elif name == "edit_image":
                data = self._image(str(args.get("image") or ""))
                if data is None:
                    raise ImageRequestError("there is no image in the conversation to edit")
                job = parse_edit_request(req, [self.normalize(data)], gen.cfg, self.mask)
            else:
                raise ImageRequestError(f"there is no tool named {name!r}")
            with gpu_turn(self.svc, gen, job, cancel):
                images, figures = gen.run(job, cancel)
            fmt = job["output_format"]
            names = [gen.store(encode_image(png, fmt, job["output_compression"], job["background"] == "opaque"),
                               fmt) for png in images]
            self.made += names
            urls = [f"/v1/images/files/{n}" for n in names]
            by = f" with {gen.model}" if len(self._gens()) > 1 else ""
            text = (f"The image is made{by} ({job['width']}x{job['height']}, seed {job['seed']}, "
                    f"{figures['seconds']:.0f} s) and the user sees it: {urls[0]}. To change it, call edit_image "
                    f"(its image: {urls[0]}).")
            ok = True
        except (ValueError, ImageJobFailed) as e:
            text, ok = f"error: {e}", False
        return {"ok": ok, "text": text, "chars": len(text), "truncated": False,
                "ms": int((time.monotonic() - t0) * 1000), "server": None, "tool": name, "images": urls,
                **({"image": {"size": f"{job['width']}x{job['height']}", "quality": job["quality"],
                              "background": job["background"], "output_format": job["output_format"],
                              "revised_prompt": model_prompt(job)}} if ok else {})}


# ------------------------------------------------------------------------------------------- the GPU turn
def resolve_policy(svc, gen: ImageGen, need: dict) -> str:
    """auto -> keep / shrink / unload for this job (see the module's docstring); a fixed policy is kept, except that
    an engine that is not running has nothing to give back (keep).  `need`: ImageGen.need's answer."""
    eng = svc.engine
    running = not hasattr(eng, "alive") or eng.alive()
    if not running or not hasattr(eng, "unload"):
        return "keep"
    if gen.policy != "auto":
        return gen.policy
    if gen.cfg.get("uma"):
        return "keep"                         # an APU: the image job and the engine share the RAM either way
    room = need["free_mib"] + need["held_mib"] - gen.reserve
    if room >= need["full_mib"]:
        return "keep"
    if engine_elastic(eng):
        return "shrink"
    if room >= need["least_mib"]:
        return "keep"                         # the image's layers streamed: slower than unloading only for a big batch
    return "unload"


def engine_elastic(eng) -> bool:
    """An engine that resizes its expert cache on request (--vram-elastic, #533)."""
    if not hasattr(eng, "vram"):
        return False
    if (getattr(eng, "info", None) or {}).get("vram_elastic"):
        return True
    spawn = getattr(eng, "spawn", None)
    args = spawn[1] if isinstance(spawn, (list, tuple)) and len(spawn) > 1 else []
    return "--vram-elastic" in (args or [])


def _client_gone(svc) -> Exception:
    """The server module's ClientGone (it may run as __main__, so not imported by name)."""
    import sys
    return getattr(sys.modules.get(type(svc).__module__), "ClientGone", RuntimeError)()


def check_room(need: dict, ram_mib: int | None) -> None:
    """An image the GPU cannot make even with the text model unloaded (its least above the engine's cap), or whose
    prefix cache would not fit the free RAM (`ram_mib`, None: not known): refused before the job, not shrunk (the
    user's decision, 2026-10-21)."""
    least, cap, host = (int(need.get(k) or 0) for k in ("least_mib", "cap_mib", "host_mib"))
    hint = "use fewer or smaller reference images, or a smaller size"
    if cap and least > cap:
        raise ImageRequestError(f"this image needs at least {least / 1024:.1f} GiB of VRAM and the GPU has "
                                f"{cap / 1024:.1f} GiB for images: {hint}")
    if ram_mib is not None and host > ram_mib - RAM_SPARE_MIB:
        raise ImageRequestError(f"this image keeps {host / 1024:.1f} GiB of its prompt's cache in RAM and "
                                f"{max(ram_mib, 0) / 1024:.1f} GiB is free: {hint}")


def ram_available_mib() -> int | None:
    try:
        import psutil
        return int(psutil.virtual_memory().available >> 20)
    except (ImportError, OSError):
        return None


@contextlib.contextmanager
def gpu_turn(svc, gen: ImageGen, job: dict, cancel: threading.Event | None = None):
    """The image job's turn on the GPU: the service's FIFO (and, with parallel requests, the engine's control lines
    once no slot is generating), and the text engine made room for it per the policy.  Yields the policy used."""
    with svc.status_lock:
        svc.status["queued"] += 1
    queued = True
    error: str | None = None
    held_gpu: GpuTurn | None = None
    try:
        with svc._turn(cancel):
            with svc.status_lock:
                svc.status["queued"] -= 1
            queued = False
            models = getattr(svc, "images", None)
            if isinstance(models, ImageModels):   # every server's image jobs on this GPU in turn
                def waiting():
                    gen._say("image job: waiting for another server's image job on this GPU")
                    with svc.status_lock:
                        svc.status["phase"] = "waiting for another server's image job"
                models.gpu.acquire(cancel, waiting)
                held_gpu = models.gpu
                models.activate(gen)
            gen.begin(job)
            ctl = getattr(svc.engine, "ctl", None) if getattr(svc.engine, "batch", 0) else None
            if ctl is not None:
                while not ctl.acquire(timeout=0.25):
                    if cancel is not None and cancel.is_set():
                        raise _client_gone(svc)
            try:
                if ctl is not None:                   # running parallel requests finish first
                    while getattr(svc, "live_reqs", None):
                        if cancel is not None and cancel.is_set():
                            raise _client_gone(svc)
                        time.sleep(0.25)
                try:
                    need = gen.need(job)
                except ImageJobFailed:
                    gen.totals["failed"] += 1
                    raise
                check_room(need, ram_available_mib())
                policy = resolve_policy(svc, gen, need)
                with svc.status_lock:
                    svc.status.update(busy=True, phase="generating an image", started=time.time(),
                                      first_token=None, prompt_tokens=None, generated=None, max_tokens=None)
                restore = _make_room(svc, gen, policy, need)
                try:
                    budget = gen.fit()
                    gen._say(f"image job: {policy}, the image engine's budget {budget} MiB (the DiT resident from "
                             f"{need['full_mib']} MiB, at least {need['least_mib']} MiB)")
                    yield policy
                finally:
                    _give_back(svc, gen, policy, restore)
                    with svc.status_lock:
                        svc.status["busy"] = False
                    svc.last_request_at = time.time()
            finally:
                if ctl is not None:
                    ctl.release()
    except BaseException as e:
        error = str(e) or type(e).__name__
        raise
    finally:
        if held_gpu is not None:
            held_gpu.release()
        gen.end(error, cancelled=cancel is not None and cancel.is_set())
        if queued:
            with svc.status_lock:
                svc.status["queued"] -= 1


def _make_room(svc, gen: ImageGen, policy: str, need: dict):
    """Do what `policy` says before the job; -> what _give_back needs."""
    eng = svc.engine
    if policy == "shrink":
        want = max(0, need["full_mib"] - need["held_mib"]) + gen.reserve
        try:
            out = eng.vram(want)
            gen._say(f"image job: the engine's expert cache gave VRAM back ({out.get('vram_free_mib')} MiB free)")
            return "shrunk"
        except Exception as e:                       # noqa: BLE001 - the job still runs, with what is free
            gen._say(f"image job: the engine could not give VRAM back ({e}); the job runs with what is free")
            return None
    if policy == "unload":
        saved = None
        if hasattr(eng, "session_file") and not getattr(eng, "batch", 0):
            path = str(gen.dir / f"llm-session-{secrets.token_hex(6)}.bin")
            try:
                r = eng.session_file("save", path)
                saved = path
                gen._say(f"image job: the conversation ({r.get('tokens')} tokens) saved; unloading the model")
            except Exception as e:                   # noqa: BLE001 - unloading still frees the GPU
                gen._say(f"image job: the conversation could not be saved ({e}); unloading the model")
                with contextlib.suppress(OSError):
                    os.unlink(path)
        eng.unload()
        if svc.vision is not None and hasattr(svc.vision, "unload"):
            svc.vision.unload()
        return saved or "unloaded"
    return None


def _give_back(svc, gen: ImageGen, policy: str, restore) -> None:
    """After the job: the image engine keeps its layers (the next image starts warm) only under keep with spare_mib
    still free; else it lets them go, and the text engine gets back what it gave."""
    keep = policy == "keep"
    if keep:
        try:
            keep = gen.engine.alive() and gen.free_mib() >= gen.spare
        except ImageJobFailed:
            keep = False
    if not keep:
        gen.release(True)
    eng = svc.engine
    if restore == "shrunk":
        try:
            eng.vram(svc.vram_reserve)               # the reserve asked for last (None: the start's)
        except Exception as e:                       # noqa: BLE001
            gen._say(f"image job: the expert cache did not grow back ({e}); POST /v1/vram can retry")
    elif policy == "unload" and restore:
        if restore != "unloaded":
            svc.restore_after_load = restore         # Service.ensure_loaded restores it after the next start
        if gen.reload == "eager":
            try:
                svc.ensure_loaded()                  # the caller still holds the FIFO
            except Exception as e:                   # noqa: BLE001 - the next request tries again
                gen._say(f"image job: the model did not load again yet ({e}); the next request loads it")


def restore_session(svc) -> None:
    """After an engine start: restore the conversation an image job saved before it unloaded the engine."""
    path = getattr(svc, "restore_after_load", None)
    if not path:
        return
    svc.restore_after_load = None
    try:
        r = svc.engine.session_file("restore", path)
        print(f"[strata] the conversation from before the image job is back ({r.get('tokens')} tokens)", flush=True)
    except Exception as e:                           # noqa: BLE001 - the engine runs, without the old cache
        print(f"[strata] the conversation from before the image job was not restored ({e}); the next request "
              "reads its prompt again", flush=True)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(path)
