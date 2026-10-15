# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""serve/images.py - image generation (Qwen-Image-2.1-Turbo) through XeStrata's own image engine, strata-image.

POST /v1/images/generations takes OpenAI's request shape (prompt, size, n, response_format b64_json|url) plus Strata's
own field seed; POST /v1/images/edits the same with reference images (OpenAI's multipart form, image or image[]; or JSON
with images: [{"image_url": ...}]), which the image model edits or takes as its subject (Qwen-Image-2.1's edits: the
vision encoder reads them into the prompt, the VAE's encoder into the DiT's).  The engine (src/image/strata_image.cpp,
`strata-image --serve`) starts with the first image and stays, outside the server (serve/image_daemon.py, the next
server of the same image model and GPU finds it: a model switch does not lose it): its kernels built once, the
prompt's text states and the layers that fit in VRAM kept between images, one image a request.  It runs Turbo's
8 steps (CFG 1, the Euler flow sampler, the model's own sigmas), the image the one stable-diffusion.cpp makes with the
model card's command for the same seed (within LPIPS 0.03).  A size whose sides are not multiples of 16 (1920x1080)
is made at the sides rounded up (1920x1088) and its middle cut out.

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
  {"exe": "engine/strata-image", "diffusion_model": "...Q8_0.gguf", "vae": "...",
   "llm": "...Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf", "quant": "Q8_0", "gpu_policy": "auto", "reload": "lazy"|"eager",
   "max_n": 4, "max_side": 2048, "timeout_s": 3600, "uma": false, "out_dir": null, "keep_files": 64,
   "vision": "...mmproj-F16.gguf", "max_refs": 3, "reserve_mib": 512, "spare_mib": 1024, "ssd_offload": false,
   "quant_lowered": {"from": "Q8_0", "to": "Q6_K", "need_gb": 15.5, "ram_left_gb": 14.0}}
  vision: Qwen3-VL's vision encoder (mmproj), for the edits (without it they are refused); max_refs: the reference
  images an edit takes at most; reserve_mib: VRAM the image engine leaves free when it sizes itself; spare_mib: the
  free VRAM that must remain after a job under keep for the engine to keep its layers; quant_lowered (only when setup
  lowered it): the size setup took instead of the one asked for, because the RAM the text model leaves was short
  (note(): the server's log and the web page say so).
"""
from __future__ import annotations

import base64
import contextlib
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
import zlib
from pathlib import Path
from typing import Any

TURBO_STEPS = 8
MODEL_NAME = "qwen-image-2.1-turbo"
POLICIES = ("auto", "keep", "shrink", "unload")
SIZE_STEP = 16                     # the VAE's 8x and the DiT's 2x2 patches: sides made at a multiple of 16, then cut
MIN_SIDE = 256
FILE_NAME = re.compile(r"^[0-9a-f]{32}\.png$")
# the model's prompt (stable-diffusion.cpp's Qwen-Image template): the system turn, whose DROP leading ids' states the
# DiT does not take, then the user's prompt as plain text
PROMPT_HEAD = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n<|im_start|>user\n"
PROMPT_TAIL = "<|im_end|>\n<|im_start|>assistant\n"
DROP = 14
# an edit's reference image in the prompt: one pad, which the engine widens to the image's tokens
REF_PROMPT = "{sep}<image{i}><|vision_start|><|image_pad|><|vision_end|>"
REF_MAX_BYTES = 32 << 20
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


def parse_request(req: dict, cfg: dict) -> dict:
    """The fields of one /v1/images/generations request, checked.  Unknown OpenAI fields (quality, style, user,
    background, moderation) are accepted and ignored; an output format other than PNG is refused."""
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
    out_fmt = req.get("output_format")
    if out_fmt not in (None, "png"):
        raise ImageRequestError("output_format: only png")
    w, h = parse_size(req.get("size"), int(cfg.get("max_side") or 2048))
    n = _int_field(req, "n", 1, int(cfg.get("max_n") or 4), 1)
    if req.get("steps") not in (None, TURBO_STEPS):
        raise ImageRequestError(f"steps: Qwen-Image-2.1-Turbo runs its own {TURBO_STEPS}-step schedule only")
    seed = _int_field(req, "seed", 0, 2**62, None)
    if seed is None:
        seed = random.SystemRandom().randrange(0, 2**31)
    return {"prompt": prompt, "width": w, "height": h, "n": n, "steps": TURBO_STEPS, "seed": seed,
            "response_format": fmt}


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


def parse_edit_request(req: dict, images: list[bytes], cfg: dict) -> dict:
    """An edit (/v1/images/edits): parse_request's fields with `images` (the reference images' bytes, PNG, JPEG, GIF or
    BMP) as job["refs"].  size "auto" (or none): the first reference's shape at about 1024 x 1024 pixels, its sides
    multiples of SIZE_STEP."""
    if not cfg.get("vision"):
        raise ImageRequestError("edits need the vision encoder: run setup with --images qwen-image again")
    if not images:
        raise ImageRequestError("image: one reference image or more")
    max_refs = int(cfg.get("max_refs") or 3)
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
    if req.get("size") in (None, "auto"):
        w0, h0 = dims[0]
        side = int(cfg.get("max_side") or 2048)
        tw = (1024 * 1024 * w0 / h0) ** 0.5
        th = tw * h0 / w0
        tw, th = (min(side, max(MIN_SIDE, round(v / SIZE_STEP) * SIZE_STEP)) for v in (tw, th))
        req = {**req, "size": f"{tw}x{th}"}
    job = parse_request(req, cfg)
    job["refs"] = list(images)
    return job


def made_size(job: dict) -> tuple[int, int]:
    """The size the engine makes for `job`: each side rounded up to a multiple of SIZE_STEP."""
    def up(v) -> int:
        return -(-int(v) // SIZE_STEP) * SIZE_STEP
    return up(job["width"]), up(job["height"])


def png_rgba(rgba: bytes, width: int, height: int, made_w: int, made_h: int) -> bytes:
    """A PNG (8-bit RGBA, as stable-diffusion.cpp writes) of the middle width x height of an RGBA8 image of
    made_w x made_h."""
    x0, y0 = (made_w - width) // 2, (made_h - height) // 2
    rows = bytearray()
    for y in range(y0, y0 + height):
        o = (y * made_w + x0) * 4
        rows += b"\0" + rgba[o:o + width * 4]           # filter type 0 a row

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes(rows), 6)) + chunk(b"IEND", b""))


def prompt_ids(tok, prompt: str, refs: int = 0) -> list[int]:
    """The prompt in the model's chat form, tokenized (Qwen3-VL's tokenizer): the user's text plain, so a special
    token spelled in it stays text; an edit's `refs` reference images before it, one pad each (stable-diffusion.cpp's
    prompt for Qwen-Image-2.1's edits, the pads widened by the engine)."""
    head = PROMPT_HEAD + "".join(REF_PROMPT.format(sep=" " if i else "", i=i + 1) for i in range(refs))
    text = head + prompt + PROMPT_TAIL
    return tok.encode(text, parse_special=True, plain=[(len(head), len(head) + len(prompt))])


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


class ImageEngine:
    """The image engine, `strata-image --serve`, through the image daemon (serve/image_daemon.py): started outside the
    server the first time and found again by the next server of the same image model and GPU, so a model switch does
    not lose it; one request line at a time (the caller serializes, the daemon between servers)."""

    def __init__(self, cfg: dict, env: dict, log_path: Path, say):
        self.cfg, self.env, self.log_path, self.say = cfg, env, log_path, say
        self.sock: Any = None
        self.f: Any = None
        self.lines: queue.Queue = queue.Queue()

    def command(self) -> list[str]:
        cmd = [str(self.cfg["exe"]), "--dit", str(self.cfg["diffusion_model"]), "--vae", str(self.cfg["vae"]),
               "--text", str(self.cfg["llm"])]
        if self.cfg.get("vision"):
            cmd += ["--vision", str(self.cfg["vision"])]
        return cmd + ["--serve"]

    def alive(self) -> bool:
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

        def pump(f, q):
            try:
                for line in f:
                    q.put(line.rstrip("\r\n"))
            except (OSError, ValueError):
                pass
            q.put(None)                               # the daemon closed the connection
        threading.Thread(target=pump, args=(self.f, self.lines), daemon=True).start()
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

    def ask(self, line: str, want: str, timeout: float, cancel: threading.Event | None = None, step=None) -> dict:
        """Send a request line and wait for its answer (the line starting with `want`) -> its fields; STEP lines go to
        `step`.  ImageJobFailed on ERROR."""
        self.send(line)
        t0 = time.monotonic()
        while True:
            got = self._next(time.monotonic() - t0, timeout, cancel)
            if got.startswith("STEP "):
                if step is not None:
                    done, steps = got.split()[1:3]
                    step(int(done), int(steps))
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

    def note(self) -> str | None:
        """What the user is told about this setup at start and on the web page: the image model's size lowered for
        the RAM (setup's "quant_lowered"), or None."""
        q = self.cfg.get("quant_lowered")
        if not isinstance(q, dict) or not q.get("from") or not q.get("to"):
            return None
        return (f"The image model runs at {q['to']} instead of {q['from']}: {q['from']} needs about "
                f"{q.get('need_gb', '?')} GB of RAM beside the text model, which leaves about "
                f"{q.get('ram_left_gb', '?')} GB. Run setup with --image-ssd-offload on to keep {q['from']} "
                "(what does not fit is read from the SSD, slower).")

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
        """Keep the newest keep_files PNGs of response_format url (the oldest go)."""
        files = sorted((p for p in self.dir.glob("*.png") if FILE_NAME.match(p.name)), key=lambda p: p.stat().st_mtime)
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
            job["ids"] = prompt_ids(self.tok, job["prompt"], len(job.get("refs") or ()))
        return job["ids"]

    def stage(self, job: dict) -> None:
        """An edit's reference images into files the engine reads (job["ref_paths"]); unstage removes them."""
        if job.get("refs") and not job.get("ref_paths"):
            work = Path(tempfile.mkdtemp(prefix="refs-", dir=self.dir))
            job["ref_dir"] = str(work)
            job["ref_paths"] = []
            for i, data in enumerate(job["refs"]):
                p = work / f"ref_{i}"
                p.write_bytes(data)
                job["ref_paths"].append(str(p))

    @staticmethod
    def unstage(job: dict) -> None:
        for p in job.pop("ref_paths", []) or []:
            with contextlib.suppress(OSError):
                os.unlink(p)
        if job.get("ref_dir"):
            with contextlib.suppress(OSError):
                os.rmdir(job.pop("ref_dir"))

    @staticmethod
    def _refs_words(job: dict) -> str:
        return "".join(f" REF:{p}" for p in job.get("ref_paths") or ())

    def need(self, job: dict) -> dict:
        """What the image needs, from the engine (started when it is not running): least_mib, full_mib (every layer
        resident), held_mib (what it holds now) and free_mib (the device's free VRAM)."""
        with self.lock:
            self._ready()
            w, h = made_size(job)
            self.stage(job)
            return self.engine.ask(f"NEED {w} {h} {len(self.ids(job)) - DROP}{self._refs_words(job)}", "NEED", 600)

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
                for i in range(job["n"]):
                    if cancel is not None and cancel.is_set():
                        self.totals["failed"] += 1
                        raise ImageJobFailed("cancelled (the client left)")
                    self.progress = (i * TURBO_STEPS, job["n"] * TURBO_STEPS)
                    out = work / f"img_{i}.rgba"

                    def step(done, steps, i=i):
                        self.progress = (i * steps + done, job["n"] * steps)
                    try:
                        r = self.engine.ask(f"IMG {mw} {mh} {job['seed'] + i} {DROP} {out} {ids}"
                                            f"{self._refs_words(job)}", "IMAGE", timeout, cancel, step)
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
        figures = {"seconds": round(secs, 2), "steps": job["steps"], "n": job["n"], "refs": len(job.get("refs") or ()),
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

    def store(self, data: bytes) -> str:
        """Keep one PNG for response_format url -> its file name (32 random hex digits: the name is the access)."""
        name = secrets.token_hex(16) + ".png"
        (self.dir / name).write_bytes(data)
        self._prune()
        return name

    def response(self, images: list[bytes], job: dict, figures: dict, base_url: str) -> dict:
        data = []
        for i, png in enumerate(images):
            item = {"revised_prompt": job["prompt"], "seed": job["seed"] + i}
            if job["response_format"] == "url":
                item["url"] = f"{base_url}/v1/images/files/{self.store(png)}"
            else:
                item["b64_json"] = base64.b64encode(png).decode("ascii")
            data.append(item)
        return {"created": int(time.time()), "data": data, "model": self.model,
                "output_format": "png", "size": f"{job['width']}x{job['height']}", "strata": figures}

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


# ------------------------------------------------------------------------------------------- chat's image tools
FILE_REF = re.compile(r"/v1/images/files/([0-9a-f]{32}\.png)")
TOOLS = [
    {"name": "generate_image",
     "description": "Make an image from a description with the image model on this computer (Qwen-Image-2.1-Turbo). "
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
    result tells the model where it is.  The hub run_with_mcp runs (template_tools, routes, call, settings)."""

    def __init__(self, svc, gen: ImageGen, messages: list, load, normalize):
        self.svc, self.gen, self.messages, self.load, self.normalize = svc, gen, messages, load, normalize
        self.settings = {"max_rounds": 4}
        self.made: list[str] = []          # the images this request made (file names)

    def template_tools(self, exclude=()) -> list[dict]:
        names = ("generate_image", "edit_image") if self.gen.cfg.get("vision") else ("generate_image",)
        return [dict(t) for t in TOOLS if t["name"] in names and t["name"] not in exclude]

    def routes(self) -> dict:
        return {t["name"]: (None, t["name"]) for t in self.template_tools()}

    def _file(self, name: str) -> bytes | None:
        p = self.gen.file(name)
        return p.read_bytes() if p is not None else None

    def _image(self, which: str) -> bytes | None:
        """The image an edit takes: a /v1/images/files/ link (or its file name), else the conversation's last."""
        m = FILE_REF.search(which or "") or re.fullmatch(r"([0-9a-f]{32}\.png)", which or "")
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
            req = {k: args[k] for k in ("prompt", "size", "seed") if args.get(k) not in (None, "")}
            if name == "generate_image":
                job = parse_request(req, self.gen.cfg)
            elif name == "edit_image":
                data = self._image(str(args.get("image") or ""))
                if data is None:
                    raise ImageRequestError("there is no image in the conversation to edit")
                job = parse_edit_request(req, [self.normalize(data)], self.gen.cfg)
            else:
                raise ImageRequestError(f"there is no tool named {name!r}")
            with gpu_turn(self.svc, self.gen, job, cancel):
                images, figures = self.gen.run(job, cancel)
            names = [self.gen.store(png) for png in images]
            self.made += names
            urls = [f"/v1/images/files/{n}" for n in names]
            text = (f"The image is made ({job['width']}x{job['height']}, seed {job['seed']}, "
                    f"{figures['seconds']:.0f} s) and the user sees it: {urls[0]}. To change it, call edit_image "
                    f"(its image: {urls[0]}).")
            ok = True
        except (ValueError, ImageJobFailed) as e:
            text, ok = f"error: {e}", False
        return {"ok": ok, "text": text, "chars": len(text), "truncated": False,
                "ms": int((time.monotonic() - t0) * 1000), "server": None, "tool": name, "images": urls}


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


@contextlib.contextmanager
def gpu_turn(svc, gen: ImageGen, job: dict, cancel: threading.Event | None = None):
    """The image job's turn on the GPU: the service's FIFO (and, with parallel requests, the engine's control lines
    once no slot is generating), and the text engine made room for it per the policy.  Yields the policy used."""
    with svc.status_lock:
        svc.status["queued"] += 1
    queued = True
    try:
        with svc._turn(cancel):
            with svc.status_lock:
                svc.status["queued"] -= 1
            queued = False
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
    finally:
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
