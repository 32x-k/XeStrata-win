# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compares XeStrata's greedy decode of a prompt with llama.cpp's (oracle_greedy).

    python compare.py ORACLE_PREFIX XESTRATA_TOKENS XESTRATA_LOGITS

ORACLE_PREFIX.tokens / .logits are oracle_greedy's; XESTRATA_TOKENS holds `strata`'s `tokens: ...` line and
XESTRATA_LOGITS its --dump-logits rows.  Prints how many leading tokens agree and, per position up to the first
difference, the largest logit difference relative to the row's largest |logit|, the KL divergence of XeStrata's
distribution from llama.cpp's, and the margin between llama.cpp's top two logits.
"""
import math
import sys
from array import array


def ids(path):
    text = open(path).read().strip()
    if text.startswith("tokens:"):
        text = text.split(":", 1)[1]
    return [int(x) for x in text.replace("\n", "").split(",") if x.strip()]


def rows(path, n_vocab):
    a = array("f")
    with open(path, "rb") as f:
        a.frombytes(f.read())
    return [a[i * n_vocab:(i + 1) * n_vocab] for i in range(len(a) // n_vocab)]


def softmax(r):
    m = max(r)
    e = [math.exp(v - m) for v in r]
    s = sum(e)
    return [v / s for v in e]


def main():
    ref_ids, xs_ids = ids(sys.argv[1] + ".tokens"), ids(sys.argv[2])
    same = 0
    while same < min(len(ref_ids), len(xs_ids)) and ref_ids[same] == xs_ids[same]:
        same += 1
    n_ref = len(ref_ids)
    with open(sys.argv[1] + ".logits", "rb") as f:
        n_vocab = len(f.read()) // 4 // n_ref
    ref, xs = rows(sys.argv[1] + ".logits", n_vocab), rows(sys.argv[3], n_vocab)
    print(f"{same} of {n_ref} leading tokens agree (vocabulary {n_vocab})")
    worst_rel = worst_kl = 0.0
    for i in range(min(same + 1, len(ref), len(xs))):
        r, x = ref[i], xs[i]
        scale = max(abs(v) for v in r)
        rel = max(abs(a - b) for a, b in zip(r, x)) / scale
        p, q = softmax(r), softmax(x)
        kl = sum(pi * (math.log(pi) - math.log(max(qi, 1e-30))) for pi, qi in zip(p, q) if pi > 0)
        top = sorted(r, reverse=True)[:2]
        worst_rel, worst_kl = max(worst_rel, rel), max(worst_kl, kl)
        print(f"  position {i:2d}: token {ref_ids[i]:6d} / {xs_ids[i] if i < len(xs_ids) else -1:6d}  "
              f"max |dlogit| / max |logit| {rel:.2e}  KL {kl:.2e}  top-2 margin {top[0] - top[1]:.3f}")
    print(f"worst relative logit difference {worst_rel:.2e}, worst KL {worst_kl:.2e}")


if __name__ == "__main__":
    main()
