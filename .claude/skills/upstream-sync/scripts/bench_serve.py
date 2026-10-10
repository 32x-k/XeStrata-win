# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""The serve table of an upstream merge on one GPU: XeStrata and upstream's release run as the server runs them
(`strata --serve`, the options setup writes), one chat of four turns per run: the tier's prompt, then the three
follow-ups of sync_prompts.py, each with the answer before it, 256 tokens a turn.  bench.sh measures the generate
path; this one the serve loop users meet (the conversation checkpoints, the per-request sampling, its own decode
loop).  The engines are driven over stdin (GEN, DONE), not through serve/server.py.

    XE="CMD..." UP="CMD..." DATA=DIR IDS=DIR XE_PROFILE=... UP_PROFILE=... \\
      python .claude/skills/upstream-sync/scripts/bench_serve.py OUT GPU

XE, UP, XE_PROFILE, UP_PROFILE, DATA, IDS, PACK, NATIVE, PLE, EXTRA and TIERS (default "64k 32k 1k") as for bench.sh;
a wrapper command must pass stdin through (docker run -i).  MODES (default "greedy t07"): greedy, and t07 =
temperature 0.7, top_p 0.95, top_k 20, a seed per round.  ROUNDS (default 3): runs per arm, tier and mode, each a new
process, the arms alternating; one discarded warm-up per arm first.  No --spec-follow: serve has none, so each
engine writes its own text and the drafts it accepts count in its output speed, as they do for a user.
OUT/serve-runs/GPU-ARM-TIER-MODE-ROUND.txt gets each run's log, and .json the four turns' DONE fields;
sync_collect.py OUT reads them.  An upstream run whose first turn reads its prompt or writes its output at SLOW tok/s
or less (default 10) is not run again in that tier and mode (marked "stopped"); a run past 2400 s is killed.
"""
import json
import os
import pathlib
import queue
import shlex
import subprocess
import sys
import threading
import time

TURNS = 3                   # follow-ups after the tier's prompt
MAX_NEW = 256
DEADLINE = 2400.0           # seconds a run may take
CONTEXT = {"1k": ["--max-context", "8192"], "4k": ["--max-context", "8192"],
           "32k": ["--max-context", "32768", "--kv", "int8"]}
for t, c in (("64k", "65536"), ("128k", "131072"), ("192k", "204800"), ("262k", "262144")):
    CONTEXT[t] = ["--max-context", c, "--kv", "int8", "--kv-resident", "32768"]


def env(name, default=None):
    v = os.environ.get(name, default)
    if v is None:
        sys.exit(f"bench_serve: {name} is not set")
    return v


if len(sys.argv) != 3:
    sys.exit(f"usage: {sys.argv[0]} OUT GPU")
out, gpu = pathlib.Path(sys.argv[1]), sys.argv[2]
data, ids_dir = pathlib.Path(env("DATA")), pathlib.Path(env("IDS"))
g = data / "models" / "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS"
model = ["--pack", env("PACK", str(data / "packs" / "qwen-iq3_xxs")),
         "--native", env("NATIVE", f"{g}-00001-of-00002.gguf"), "--ple-gguf", env("PLE", f"{g}-00002-of-00002.gguf"),
         *shlex.split(env("EXTRA", ""))]
arms = {"xe": (shlex.split(env("XE")), env("XE_PROFILE")), "up": (shlex.split(env("UP")), env("UP_PROFILE"))}
rounds = int(env("ROUNDS", "3"))
slow = float(env("SLOW", "10"))
modes = env("MODES", "greedy t07").split()
followups = [[int(x) for x in (ids_dir / f"turn{i}.ids").read_text().split(",")] for i in range(2, TURNS + 2)]
im_end = followups[0][0]
runs = out / "serve-runs"
runs.mkdir(parents=True, exist_ok=True)


def sampling(mode, rnd):
    if mode == "greedy":
        return ""
    if mode == "t07":
        return f"temperature=0.7 top_p=0.95 top_k=20 seed={1000 + rnd} "
    sys.exit(f"bench_serve: unknown mode {mode!r}")


def run(arm, tier, mode, rnd, stem):
    """One process, one chat; returns its turns (the DONE fields) or None when it did not finish."""
    cmd, profile = arms[arm]
    args = [*cmd, "--serve", *model, "--expert-profile", profile, "--expert-cache", "auto", "--prefill", "auto",
            "--spec", "4", "--spec-min-p", "0.5", "--mtp", str(data / "mtp" / "rt"), *CONTEXT[tier]]
    log = open(runs / f"{stem}.txt", "w")
    log.write("# " + shlex.join(args) + "\n")
    p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         bufsize=1)
    lines = queue.Queue()
    threading.Thread(target=lambda: [lines.put(x) for x in p.stdout] + [lines.put(None)], daemon=True).start()
    end = time.monotonic() + DEADLINE

    def line():
        try:
            x = lines.get(timeout=max(end - time.monotonic(), 0.1))
        except queue.Empty:
            raise TimeoutError
        if x is not None and not x.startswith("T "):
            log.write(x)
        return x

    turns, why = [], None
    try:
        while (x := line()) is not None and not x.startswith("READY"):
            pass
        if x is None:
            raise EOFError
        prompt = [int(v) for v in (ids_dir / f"{tier}.ids").read_text().split(",")]
        for turn in range(TURNS + 1):
            p.stdin.write(f"GEN {MAX_NEW} {sampling(mode, rnd)}{','.join(map(str, prompt))}\n")
            p.stdin.flush()
            gen = []
            while (x := line()) is not None:
                if x.startswith("T "):
                    gen.append(int(x.split()[1]))
                elif x.startswith(("DONE ", "ERR")):
                    break
            if x is None or x.startswith("ERR"):
                why = (x or "the engine ended").strip()
                break
            f = x.split()
            # DONE <generated> <prompt> <prompt ms> <decode ms> <finish> <accepted> <offered> <reused> ... [read]
            done = {"generated": int(f[1]), "prompt": int(f[2]), "prompt_ms": float(f[3]), "decode_ms": float(f[4]),
                    "finish": f[5], "accepted": int(f[6]), "offered": int(f[7]), "reused": int(f[8]),
                    "read": int(f[14]) if len(f) > 14 else int(f[2]) - int(f[8])}
            turns.append(done)
            if turn == 0 and arm == "up" and slow > 0:
                if done["read"] / max(done["prompt_ms"], 1e-3) * 1e3 <= slow:
                    why = "stopped: prompt"
                elif done["generated"] / max(done["decode_ms"], 1e-3) * 1e3 <= slow:
                    why = "stopped: output"
                if why:
                    break
            if turn < TURNS:
                while gen and gen[-1] == im_end:          # the follow-up closes the answer itself
                    gen.pop()
                prompt = prompt + gen + followups[turn]
        if not why:
            p.stdin.write("QUIT\n")
            p.stdin.flush()
    except TimeoutError:
        why = f"killed after {DEADLINE:.0f} s"
    except EOFError:
        why = "the engine ended before READY"
    finally:
        try:
            p.stdin.close()
        except OSError:
            pass
        try:
            p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
    if why:
        log.write(f"bench: {why}\n")
    log.close()
    rec = {"gpu": gpu, "arm": arm, "tier": tier, "mode": mode, "round": rnd, "turns": turns, "why": why}
    (runs / f"{stem}.json").write_text(json.dumps(rec, indent=1) + "\n")
    rate = lambda n, ms: f"{n / ms * 1e3:.1f}" if ms > 0 else "-"   # noqa: E731
    print(f"{gpu} {arm} {tier} {mode} {rnd}: " + " | ".join(
        f"read {t['read']} at {rate(t['read'], t['prompt_ms'])}, out {t['generated']} at "
        f"{rate(t['generated'], t['decode_ms'])}" for t in turns) + (f" ({why})" if why else ""), flush=True)
    return rec


for a in ("xe", "up"):
    run(a, "1k", "greedy", 0, f"{gpu}-{a}-warmup")
for tier in env("TIERS", "64k 32k 1k").split():
    for mode in modes:
        up_stopped = False
        for r in range(1, rounds + 1):
            for a in (("xe", "up") if r % 2 else ("up", "xe")):
                if a == "up" and up_stopped:
                    continue
                rec = run(a, tier, mode, r, f"{gpu}-{a}-{tier}-{mode}-{r}")
                up_stopped |= a == "up" and (rec["why"] or "").startswith("stopped")
