#!/usr/bin/env python3
"""engram_fold.py - a pack of their data with none of ours: fold the Engram gate and rename the kv straight from the
quant's own GGUF shards, so `pack_gguf --engram <their engram-IQ4_NL.gguf>` needs no Engram set for those two tensors.

The IST-DAS sets store the Engram as three per-layer tensors (`blk.N.engram_k`, `engram_q`, `engram_wkv`); the engine's
layout wants `engram.gate` (k*q folded) and `engram.wkv` (their wkv, renamed).  Their `engram-IQ4_NL.gguf` carries only the
ple table, so without this fold a pack built around their Engram file stops at "blk.1.engram_kv.weight: missing".

The output is a `dense_requant.py`-style folder (dense.json + dense.bin) for `pack_gguf.py --dense-override`, which supplies
tensors a pack would otherwise take from an Engram set.  Nothing is requantised: the gate is computed in float and written
F32, the kv is copied byte for byte.

usage:
  engram_fold.py --model /data/helcig/set200/2.00bpw --out /data/helcig/engram_fold_200 [--layers 1,14]
  pack_gguf.py --model <set> --out <pack> --engram <their engram dir> \
      --skip-tensors 'attn_output_a|engram_(k|q|wkv)\\.' --dense-override <fold dir> ...
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pack_gguf  # noqa: E402  (its GGUF reader and its type table; one source of truth for both)

# the types the fold can read as float; the kv is copied, so any type is fine for it
FLOATY = {"F32", "F16", "BF16"}


def shards(model: pathlib.Path) -> list[pack_gguf.Gguf]:
    return [pack_gguf.Gguf(p) for p in sorted(model.glob("*.gguf"))]


def find(gs, name):
    for g in gs:
        for t in g.tensors:
            if t[0] == name:
                return g, t
    return None, None


def nbytes(typ: int, dims) -> int:
    tname, per, bpb = pack_gguf.TYPES[typ]
    nel = 1
    for d in dims:
        nel *= int(d)
    return nel // per * bpb if per > 1 else nel * bpb


def to_float(g, t) -> np.ndarray:
    name, typ, dims, off = t
    tname = pack_gguf.TYPES[typ][0]
    if tname not in FLOATY:
        raise SystemExit("%s: type %s is not one the fold reads (%s)" % (name, tname, ", ".join(sorted(FLOATY))))
    nb = nbytes(typ, dims)
    raw = np.fromfile(g.path, dtype=np.uint8, count=nb, offset=off)
    if tname == "F32":
        return raw.view(np.float32).astype(np.float64)
    if tname == "BF16":
        return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    return raw.view(np.float16).astype(np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="the folder of the quant's GGUF shards")
    ap.add_argument("--out", required=True, help="the override folder to write (dense.json + dense.bin)")
    ap.add_argument("--layers", default="", help="comma list of Engram layers; default: every layer with an engram_k")
    a = ap.parse_args()
    model = pathlib.Path(a.model).expanduser().resolve()
    out = pathlib.Path(a.out).expanduser().resolve()
    gs = shards(model)
    if not gs:
        raise SystemExit("%s: no .gguf files" % model)
    names = {t[0] for g in gs for t in g.tensors}
    if a.layers:
        layers = [int(x) for x in a.layers.split(",") if x.strip()]
    else:
        layers = sorted({int(n.split(".")[1]) for n in names if n.startswith("blk.") and ".engram_k." in n})
    out.mkdir(parents=True, exist_ok=True)
    idx: dict[str, list] = {}
    blobs: list[bytes] = []
    off = 0

    def add(name: str, tname: str, data: bytes, shape: list[int]) -> None:
        nonlocal off
        pad = (-off) % 64
        if pad:
            blobs.append(b"\0" * pad)
            off += pad
        blobs.append(data)
        idx[name] = [tname, "dense.bin", off, len(data), shape]
        off += len(data)

    for il in layers:
        kg, kt = find(gs, "blk.%d.engram_k.weight" % il)
        qg, qt = find(gs, "blk.%d.engram_q.weight" % il)
        wg, wt = find(gs, "blk.%d.engram_wkv.weight" % il)
        if kt is None or qt is None:
            print("layer %d: no engram_k/q in these shards, skipped" % il)
            continue
        k, q = to_float(kg, kt), to_float(qg, qt)
        if k.shape != q.shape:
            raise SystemExit("layer %d: k %s and q %s differ in shape" % (il, k.shape, q.shape))
        gate = (k * q).astype(np.float32)
        add("layers.%d.engram.gate.weight" % il, "F32", gate.tobytes(), list(kt[2]))
        kv_note = "MISSING"
        if wt is not None:
            nb = nbytes(wt[1], wt[2])
            add("layers.%d.engram.wkv.weight" % il, pack_gguf.TYPES[wt[1]][0],
                np.fromfile(wg.path, dtype=np.uint8, count=nb, offset=wt[3]).tobytes(), list(wt[2]))
            kv_note = "%s %s" % (pack_gguf.TYPES[wt[1]][0], list(wt[2]))
        print("layer %d: gate %s folded from k*q (F32), kv %s" % (il, list(kt[2]), kv_note))

    (out / "dense.bin").write_bytes(b"".join(blobs))
    json.dump(idx, open(out / "dense.json", "w"), indent=1)
    print("wrote %s (%.2f MB), dense.json with %d tensors" % (out / "dense.bin", os.path.getsize(out / "dense.bin") / 1e6, len(idx)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
