# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""The generate table of an upstream merge on one GPU: XeStrata and upstream's release, IQ3_XXS by default, the
tiers of the 2026-10-04-speed-matrix record and 192K (1K to 262K) with what setup writes for each context (images
off), 256 greedy tokens, each run a new process.  The runs are tools/bench_core.py's (ab_generate), which setup's
--bench makes too.

    XE="CMD..." UP="CMD..." DATA=DIR IDS=DIR XE_PROFILE=... UP_PROFILE=... \\
      python .claude/skills/upstream-sync/scripts/bench_generate.py OUT GPU [--dry-run]

XE and UP: the command that starts each engine (its strata binary, or a wrapper that runs it in its container);
the generate options are appended; UP=none measures XeStrata alone (the B70 in a merge, judged against the previous
record).  XE_PROFILE and UP_PROFILE: each engine's shipped expert profile (its
data/expert-profile.bin).  DATA: the data folder (packs/qwen-iq3_xxs, models/...IQ3_XXS-0000N-of-00002.gguf, mtp/rt).
IDS: the prompts of sync_prompts.py beside this script.  ROUNDS (default 2): runs per arm and tier.  Another model
than IQ3_XXS (a smaller one on a PC with less RAM): CONFIG, a model config setup wrote (xestrata-<model>.json: its
files, its draft layer, its RAM budget and calibrated settings), or PACK, NATIVE (its first shard) and PLE (the shard
with per_layer_token_embd); EXTRA: options both engines take besides (e.g. UD-Q4_K_XL's --resident-budget-gib).
TIERS (default all seven, the longest first): a subset.
OUT/runs/GPU-ARM-TIER-ROUND.txt gets each log; one discarded warm-up per arm first (JIT); the arms alternate.
sync_collect.py OUT reads them.  Both engines write one text a tier: upstream runs first in the tier and writes its
own answer (OUT/runs/GPU-TIER-follow.csv), and every later run of the tier, both engines', follows it
(--spec-follow, upstream's own tool).  FOLLOW=0: each engine writes its own text.  upstream plainly at SLOW tok/s or
less (default 10) is stopped mid-run and not run again in that tier; the log ends with "bench: stopped, upstream's
prompt|output ...".  SLOW=0: never.  --dry-run: print the engine commands instead of running them.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "tools"))
import bench_core as B  # noqa: E402

args = [a for a in sys.argv[1:] if a != "--dry-run"]
if len(args) != 2:
    sys.exit(f"usage: {sys.argv[0]} OUT GPU [--dry-run]")
m = B.Merge()
B.ab_generate(Path(args[0]), args[1], m.engines, m.args, Path(B.need("IDS")),
              os.environ.get("TIERS", "262k 192k 128k 64k 32k 4k 1k").split(), m.rounds,
              os.environ.get("FOLLOW", "1") != "0", m.slow, dry="--dry-run" in sys.argv)
