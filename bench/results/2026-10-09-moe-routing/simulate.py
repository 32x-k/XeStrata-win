# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Scores VRAM expert caches against routing traces written by dump_routing: the share of the generated tokens'
routed experts that a cache of S slots holds.

    python simulate.py TRACE.bin ... [--slots 512,1024,...] [--expert-mib 5.8]

Policies (each scored on one conversation's generated tokens; "LOO" ranks experts on the other conversations):
  layers     whole layers' experts on the GPU, the rest on the CPU (llama.cpp's --n-cpu-moe): S // n_expert layers
  profile    the S experts the other conversations routed most (XeStrata's --expert-profile, static)
  adaptive   profile, then XeStrata's adaptive tier (src/program/generate.cpp `adapt`): every 4 tokens, per layer,
             a missing expert routed >= 2 (decayed) replaces the least-routed resident one when it leads by >= 1.5;
             at most 96 swaps a round; counts x 0.7 after each round.  Swaps are taken to land before the next token.
  lru        S / n_layer slots per layer, least recently used out, every miss copied in
  oracle     the S experts this conversation's own generated tokens routed most (static; an upper bound)
"""
import argparse
from pathlib import Path

import numpy as np


def read_trace(path):
    blob = Path(path).read_bytes()
    if blob[:4] != b"XRT1":
        raise SystemExit(f"{path}: not a routing trace")
    n_layer, k = np.frombuffer(blob, "<u4", 2, 4)
    dt = np.dtype([("phase", "u1"), ("token", "<i4"), ("ids", "<u2", (int(n_layer), int(k)))])
    rec = np.frombuffer(blob, dt, offset=12)
    return int(n_layer), int(k), rec["phase"], rec["ids"]


def counts(ids, n_layer, n_expert):
    c = np.zeros((n_layer, n_expert), np.int64)
    for layer in range(n_layer):
        c[layer] = np.bincount(ids[:, layer, :].ravel(), minlength=n_expert)
    return c


def top_s(c, s):
    flat = np.argsort(-c.ravel(), kind="stable")[:s]
    res = np.zeros(c.size, bool)
    res[flat] = True
    return res.reshape(c.shape)


def hit_static(ids, res):
    n, n_layer, k = ids.shape
    layer = np.broadcast_to(np.arange(n_layer)[None, :, None], ids.shape)
    return res[layer, ids].mean()


def hit_adaptive(ids, res, every=4, decay=0.7, max_swaps=96):
    res = res.copy()
    n_layer, n_expert = res.shape
    usage = np.zeros(res.shape, np.float32)
    hits = swaps_total = 0
    for t in range(ids.shape[0]):
        for layer in range(n_layer):
            e = ids[t, layer]
            hits += int(res[layer, e].sum())
            usage[layer, e] += 1.0
        if (t + 1) % every:
            continue
        swaps = []
        for layer in range(n_layer):
            u, r = usage[layer], res[layer]
            cand = np.flatnonzero(~r & (u >= 2.0))
            vict = np.flatnonzero(r)
            if cand.size == 0 or vict.size == 0:
                continue
            cand = cand[np.argsort(-u[cand], kind="stable")]
            vict = vict[np.argsort(u[vict], kind="stable")]
            for ci, vi in zip(cand, vict):
                if u[ci] < u[vi] + 1.5:
                    break
                swaps.append((u[ci] - u[vi], layer, ci, vi))
        swaps.sort(key=lambda s: -s[0])
        for _, layer, ci, vi in swaps[:max_swaps]:
            res[layer, ci], res[layer, vi] = True, False
        swaps_total += min(len(swaps), max_swaps)
        usage *= decay
    return hits / ids[..., 0].size / ids.shape[2], swaps_total / ids.shape[0]


def hit_lru(ids, per_layer):
    n_layer = ids.shape[1]
    caches = [dict() for _ in range(n_layer)]
    hits = misses = 0
    for t in range(ids.shape[0]):
        for layer in range(n_layer):
            c = caches[layer]
            for e in ids[t, layer]:
                e = int(e)
                if e in c:
                    hits += 1
                else:
                    misses += 1
                    if len(c) >= per_layer:
                        del c[min(c, key=c.get)]
                c[e] = t
    return hits / (hits + misses), misses / ids.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--slots", default="512,1024,1600,2048,2560,3072")
    ap.add_argument("--n-expert", type=int, default=128)
    # this model's "Q6_K": gate and up in Q6_K, down in Q8_0 (960 columns do not divide into 256-wide K blocks)
    ap.add_argument("--expert-mib", type=float, default=(2 * 2560 * 960 * 6.5625 + 960 * 2560 * 8.5) / 8 / 2**20)
    args = ap.parse_args()

    runs = []
    for p in args.traces:
        n_layer, k, phase, ids = read_trace(p)
        runs.append((Path(p).stem, ids[phase == 1].astype(np.int64), counts(ids, n_layer, args.n_expert)))
    n_expert = args.n_expert
    print(f"{len(runs)} conversations, {sum(r[1].shape[0] for r in runs)} generated tokens, "
          f"{n_layer} layers x {n_expert} experts, k = {k}, expert {args.expert_mib:.2f} MiB")

    every = np.concatenate([r[1] for r in runs])
    total = counts(every, n_layer, n_expert).ravel()
    share = np.sort(total)[::-1].cumsum() / total.sum()
    print("routing concentration (all generated tokens): top 25% of experts take "
          f"{share[total.size // 4 - 1]:.3f}, top 50% take {share[total.size // 2 - 1]:.3f}")

    print(f"{'slots':>6} {'GiB':>5} {'layers':>7} {'profile':>8} {'adaptive':>9} {'swapMiB/t':>9} "
          f"{'lru':>6} {'lruMiB/t':>8} {'oracle':>7}")
    for s in (int(x) for x in args.slots.split(",")):
        row = {"layers": [], "profile": [], "adaptive": [], "swap": [], "lru": [], "lrumiss": [], "oracle": []}
        for i, (name, gen, _) in enumerate(runs):
            if gen.shape[0] == 0:
                continue
            others = sum(r[2] for j, r in enumerate(runs) if j != i)
            lay = np.zeros((n_layer, n_expert), bool)
            lay[: s // n_expert] = True
            prof = top_s(others, s)
            row["layers"].append(hit_static(gen, lay))
            row["profile"].append(hit_static(gen, prof))
            h, sw = hit_adaptive(gen, prof)
            row["adaptive"].append(h)
            row["swap"].append(sw * args.expert_mib)
            h, m = hit_lru(gen, s // n_layer)
            row["lru"].append(h)
            row["lrumiss"].append(m * args.expert_mib)
            row["oracle"].append(hit_static(gen, top_s(counts(gen, n_layer, n_expert), s)))
        m = {key: float(np.mean(v)) for key, v in row.items()}
        print(f"{s:>6} {s * args.expert_mib / 1024:>5.1f} {m['layers']:>7.3f} {m['profile']:>8.3f} "
              f"{m['adaptive']:>9.3f} {m['swap']:>9.1f} {m['lru']:>6.3f} {m['lrumiss']:>8.1f} {m['oracle']:>7.3f}")


if __name__ == "__main__":
    main()
