# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""The everyday table of an upstream merge on one GPU: XeStrata and upstream's release behind their own
serve/server.py, as a user runs them, with the options setup writes; requests through the OpenAI API, streamed.
Three conversations per run, each engine's server started anew for the run:

- chat-ja, chat-en: a short system prompt and five short questions in a row, each sent with the whole chat before it;
- agent: a coding agent's task with two tools; the tool results are fixed source files of XeStrata at commit 58dbbe5
  (one per step, whatever the call asked for), so the context grows by a few thousand tokens a step; a tool call is
  required for the first STEPS - 1 steps, then the model answers.

Each turn is at most 1024 tokens (the thinking included), sampled as Qwen recommends (temperature 0.7, top_p 0.8,
top_k 20; TEMPERATURE sets another temperature, 0 greedy) with a seed per round.  CONVS (default "chat-ja chat-en
agent"): the conversations to hold.  Measured per conversation: the time to the first streamed token (median of its
turns), the prompt and the output speed (the server's `timings`, summed over its turns).  The runs are
tools/bench_core.py's (ab_usage), which setup's --bench makes too.

    XE="CMD..." UP="CMD..." XE_ROOT=DIR UP_ROOT=DIR DATA=DIR XE_PROFILE=... UP_PROFILE=... \\
      python .claude/skills/upstream-sync/scripts/bench_usage.py OUT GPU [--dry-run]     # from the repository root

XE and UP: each engine's binary or wrapper, optionally after `env K=V ...` (those go to its server's environment);
UP=none: XeStrata alone, as for bench_generate.py.
XE_ROOT and UP_ROOT: each engine's tree (its serve/server.py runs).  PY (default this Python): the Python that runs
them.  CONTEXT (default 262144): the context, with what setup writes for it.  CONFIG, PACK, NATIVE, PLE and EXTRA as for
bench_generate.py.  ROUNDS (default 3): runs per arm, alternating, after a discarded warm-up per arm.
An upstream run whose first answer comes at SLOW tok/s or less (default 10; upstream's sycl/ port on a card its
experts do not fit) is stopped there and not run again (marked "stopped").  PORT (default 18095).
OUT/usage-runs/GPU-ARM-ROUND.json gets each run's turns and .txt its server's log; sync_collect.py OUT reads them.
--dry-run: print the server commands and configs instead of running them.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "tools"))
import bench_core as B  # noqa: E402

args = [a for a in sys.argv[1:] if a != "--dry-run"]
if len(args) != 2:
    sys.exit(f"usage: {sys.argv[0]} OUT GPU [--dry-run]")
m = B.Merge(usage=True, rounds=3)   # as bench_serve.py
ctx = int(os.environ.get("CONTEXT", "262144"))


def cfg(e: B.Engine) -> dict:
    return {"exe": e.cmd[0], "cwd": str(e.root), "tokenizer": f"{m.pack}/tokenizer", "model_name": "bench",
            "args": B.engine_args(m.model, e.profile, m.mtp, B.usage_context(m.model, ctx))}


convs = tuple(os.environ.get("CONVS", " ".join(B.CONVERSATIONS)).split())
if set(convs) - set(B.CONVERSATIONS):
    sys.exit(f"CONVS: some of {' '.join(B.CONVERSATIONS)}")
B.ab_usage(Path(args[0]), args[1], m.engines, cfg, ctx, B.load_corpus()["agent"], m.rounds, m.slow,
           port=int(os.environ.get("PORT", "18095")), dry="--dry-run" in sys.argv,
           temperature=float(os.environ.get("TEMPERATURE", str(B.TEMPERATURE))), convs=convs)
