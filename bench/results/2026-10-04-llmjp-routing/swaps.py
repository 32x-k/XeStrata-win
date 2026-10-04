# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""The adaptive tier at one slot count with fewer swaps a round, and its spread over the conversations.

    python swaps.py TRACE_DIR SLOTS
"""
import sys
from pathlib import Path

import numpy as np

import simulate as s

MIB = (2 * 2560 * 960 * 6.5625 + 960 * 2560 * 8.5) / 8 / 2**20


def main():
    runs = []
    for p in sorted(Path(sys.argv[1]).glob("*.bin")):
        n_layer, _, phase, ids = s.read_trace(p)
        runs.append((p.stem, ids[phase == 1].astype(np.int64), s.counts(ids, n_layer, 128)))
    slots = int(sys.argv[2])
    for swaps in (8, 24, 48, 96):
        hits, moved = [], []
        for i, (_, gen, _) in enumerate(runs):
            others = sum(r[2] for j, r in enumerate(runs) if j != i)
            h, w = s.hit_adaptive(gen, s.top_s(others, slots), max_swaps=swaps)
            hits.append(h)
            moved.append(w * MIB)
        print(f"swaps {swaps:>3}: hit {np.mean(hits):.3f}  swap {np.mean(moved):6.1f} MiB/token  "
              f"per conversation {' '.join(f'{h:.2f}' for h in hits)}")
    print("conversations:", " ".join(r[0] for r in runs))


if __name__ == "__main__":
    main()
