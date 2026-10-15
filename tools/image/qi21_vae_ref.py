# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""tools/image/qi21_vae_ref.py - Qwen-Image-2.1's VAE decoder for one image in NumPy (FP32), the reference the image
engine's VAE kernels are checked against.  Written from the model's structure (the diffusers names of its
safetensors): the latent un-normalised by the config's mean and standard deviation, a 1x1 convolution, conv_in, a
middle of two residual blocks around a one-head attention, five up blocks (three residual blocks each; the first four
double the size by nearest neighbours and a 3x3 convolution, plus a shortcut that spreads the block's input channels
over the new pixels), an RMS norm, SiLU and conv_out to RGBA in -1 .. 1.

    PY=... python tools/image/qi21_vae_ref.py VAE.safetensors DUMP_DIR

DUMP_DIR holds a run's latent_final and image (.bin + .json); prints how far this decoder's image is from the run's.
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

# Qwen/Qwen-Image-2.1's vae/config.json: latents_mean, latents_std
MEAN = [0.5126, 0.7721, -0.0631, 1.3506, -0.7855, -2.1025, -0.3458, 1.3722, 1.8873, -1.7177, -0.651, 0.2732, 0.7562,
        -0.6163, -1.0277, 3.8363, 2.021, 0.0472, 0.932, 2.0087, 2.4954, -0.1391, -1.4249, 1.8464, -0.5236, 1.2826,
        3.7046, -1.3035, 2.7286, -1.4518, -1.9036, -1.9955, -0.0342, -1.0265, -0.7636, 3.0555, 0.0746, -3.0751,
        -0.1076, 1.7376, -1.0914, -1.9435, -0.2784, -1.368, 0.4809, -0.4433, 0.3764, 0.5729, -2.0595, 1.096, -1.326,
        -2.0211, -5.0179, 0.5275, 4.0162, 1.8505, 0.3026, 1.9373, 1.4937, 0.2632, 0.5547, -1.7121, -0.1562, 0.0304]
STD = [3.2001, 3.2936, 3.4321, 3.0091, 3.1061, 4.0379, 4.0705, 3.791, 3.0785, 3.65, 3.9308, 3.0904, 2.8778, 3.7675,
       3.732, 5.0756, 3.2864, 4.0397, 3.1317, 4.0443, 2.9249, 3.9454, 3.0988, 4.2489, 3.4896, 3.8513, 3.9323, 3.4719,
       3.7498, 4.283, 3.5694, 4.2467, 3.9037, 3.2947, 5.077, 3.5075, 3.27, 3.4767, 2.8063, 5.1125, 3.5327, 4.7833,
       3.1286, 4.1819, 3.8527, 3.8312, 3.5605, 4.3875, 3.9624, 4.0168, 3.5643, 4.055, 5.5614, 4.2963, 4.408, 3.4959,
       3.8747, 3.7608, 3.5735, 3.149, 3.7662, 3.6746, 3.4563, 3.8161]
TEMPORAL_UP = (True, True, True, False)   # the up blocks whose input shortcut also spans time (factor_t 2)


class SafeTensors:
    def __init__(self, path: str):
        self.f = open(path, "rb")
        n = struct.unpack("<Q", self.f.read(8))[0]
        self.h = json.loads(self.f.read(n))
        self.base = 8 + n

    def __call__(self, name: str) -> np.ndarray:
        m = self.h[name]
        self.f.seek(self.base + m["data_offsets"][0])
        raw = self.f.read(m["data_offsets"][1] - m["data_offsets"][0])
        if m["dtype"] == "BF16":
            a = (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif m["dtype"] == "F32":
            a = np.frombuffer(raw, np.float32)
        else:
            raise ValueError(f"{name}: {m['dtype']}")
        return a.reshape(m["shape"]).astype(np.float32)

    def has(self, name: str) -> bool:
        return name in self.h


def conv(x, w, b, pad):
    """x [C, H, W], w [O, C, k, k] -> [O, H, W] (stride 1, `pad` zeros around)."""
    n_out, C, k, _ = w.shape
    _, H, W = x.shape
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad))) if pad else x
    out = np.zeros((n_out, H * W), np.float32)
    for dy in range(k):
        for dx in range(k):
            out += w[:, :, dy, dx] @ xp[:, dy:dy + H, dx:dx + W].reshape(C, H * W)
    return (out + b[:, None]).reshape(n_out, H, W)


def rms(x, gamma):
    return x / np.sqrt((x * x).mean(0, keepdims=True) + 1e-12) * gamma.reshape(-1, 1, 1)


def silu(x):
    return x / (1 + np.exp(-x))


def resblock(S, p, x):
    h = conv(x, S(p + "conv_shortcut.weight"), S(p + "conv_shortcut.bias"), 0) \
        if S.has(p + "conv_shortcut.weight") else x
    r = conv(silu(rms(x, S(p + "norm1.gamma"))), S(p + "conv1.weight"), S(p + "conv1.bias"), 1)
    r = conv(silu(rms(r, S(p + "norm2.gamma"))), S(p + "conv2.weight"), S(p + "conv2.bias"), 1)
    return r + h


def attention(S, p, x):
    C, H, W = x.shape
    n = rms(x, S(p + "norm.gamma"))
    qkv = conv(n, S(p + "to_qkv.weight"), S(p + "to_qkv.bias"), 0).reshape(3, C, H * W)
    q, k, v = qkv[0].T, qkv[1].T, qkv[2].T
    s = q @ k.T / np.sqrt(C)
    s -= s.max(1, keepdims=True)
    a = np.exp(s)
    a /= a.sum(1, keepdims=True)
    o = (a @ v).T.reshape(C, H, W)
    return x + conv(o, S(p + "proj.weight"), S(p + "proj.bias"), 0)


def dup_up(x, cout, ft, fs=2):
    """The up block's shortcut for one frame: the input's channels repeated, then spread over fs x fs pixels (and ft
    frames, of which the first chunk keeps the last)."""
    cin, H, W = x.shape
    rep = cout * ft * fs * fs // cin
    x = np.repeat(x, rep, axis=0)                                 # repeat_interleave along the channels
    x = x.reshape(cout, ft, fs, fs, H, W)
    x = x.transpose(0, 1, 4, 2, 5, 3).reshape(cout, ft, H * fs, W * fs)
    return x[:, ft - 1]


def decode(S, z):
    """z [64, h, w] (the DiT's latent) -> RGBA [4, 16 h, 16 w] in -1 .. 1."""
    z = z * np.array(STD, np.float32)[:, None, None] + np.array(MEAN, np.float32)[:, None, None]
    x = conv(z, S("post_quant_conv.weight"), S("post_quant_conv.bias"), 0)
    x = conv(x, S("decoder.conv_in.weight"), S("decoder.conv_in.bias"), 1)
    x = resblock(S, "decoder.mid_block.resnets.0.", x)
    x = attention(S, "decoder.mid_block.attentions.0.", x)
    x = resblock(S, "decoder.mid_block.resnets.1.", x)
    for i in range(5):
        p = f"decoder.up_blocks.{i}."
        x_in = x
        for j in range(3):
            x = resblock(S, p + f"resnets.{j}.", x)
        if S.has(p + "upsampler.resample.1.weight"):
            up = np.repeat(np.repeat(x, 2, axis=1), 2, axis=2)
            x = conv(up, S(p + "upsampler.resample.1.weight"), S(p + "upsampler.resample.1.bias"), 1)
            x = x + dup_up(x_in, x.shape[0], 2 if TEMPORAL_UP[i] else 1)
        print(f"  up block {i}: {x.shape}, |x| {np.abs(x).mean():.4f}", flush=True)
    x = silu(rms(x, S("decoder.norm_out.gamma")))
    return conv(x, S("decoder.conv_out.weight"), S("decoder.conv_out.bias"), 1)


def load(d: Path, name: str) -> np.ndarray:
    meta = json.loads((d / f"{name}.json").read_text())
    return np.fromfile(d / f"{name}.bin", np.float32).reshape(list(reversed(meta["shape"])))


def main() -> None:
    S, d = SafeTensors(sys.argv[1]), Path(sys.argv[2])
    z = load(d, "latent_final")[0]
    out = decode(S, z)
    ref = load(d, "image")[0]                     # [4, H, W] in 0 .. 1
    img = np.clip((out + 1) / 2, 0, 1)
    diff = np.abs(img - ref)
    print(f"image: max {diff.max():.4g} mean {diff.mean():.4g} (x 255: {diff.mean() * 255:.3f}); "
          f"raw range {out.min():.3f} .. {out.max():.3f}")


if __name__ == "__main__":
    main()
