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
top_k 20) with a seed per round.  Measured per conversation: the time to the first streamed token (median of its
turns), the prompt and the output speed (the server's `timings`, summed over its turns).

    XE="CMD..." UP="CMD..." XE_ROOT=DIR UP_ROOT=DIR DATA=DIR XE_PROFILE=... UP_PROFILE=... \\
      python .claude/skills/upstream-sync/scripts/bench_usage.py OUT GPU      # from the repository root

XE and UP: each engine's binary or wrapper, optionally after `env K=V ...` (those go to its server's environment).
XE_ROOT and UP_ROOT: each engine's tree (its serve/server.py runs).  PY (default this Python): the Python that runs
them.  CONTEXT (default 262144): the context, with what setup writes for it.  PACK, NATIVE, PLE and EXTRA as for
bench.sh.  ROUNDS (default 3): runs per arm, alternating, after a discarded warm-up per arm.
An upstream run whose first answer comes at SLOW tok/s or less (default 10; upstream's sycl/ port on a card its
experts do not fit) is stopped there and not run again (marked "stopped").  PORT (default 18095).
OUT/usage-runs/GPU-ARM-ROUND.json gets each run's turns and .txt its server's log; sync_collect.py OUT reads them.
"""
import json
import os
import pathlib
import shlex
import signal
import subprocess
import sys
import time
import urllib.request

MAX_TOKENS = 1024
STEPS = 7
COMMIT = "58dbbe5"
SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20}
CHATS = {
    "chat-ja": ("あなたは親切なアシスタントです。簡潔に日本語で答えてください。",
                ["週末に東京で雨の日でも楽しめる過ごし方を3つ教えて。",
                 "その中で一番お金がかからないのはどれ？理由も。",
                 "子ども連れの場合に気をつけることは？",
                 "当日の持ち物リストを作って。",
                 "ここまでの内容を3行でまとめて。"]),
    "chat-en": ("You are a helpful assistant. Answer concisely.",
                ["What is the difference between a process and a thread?",
                 "Give a short Python example that starts two threads.",
                 "How would that change with multiprocessing?",
                 "When is the GIL a problem, and when is it not?",
                 "Summarize our conversation in three bullet points."]),
}
AGENT_SYSTEM = ("You are a coding agent working in the user's repository, a C++/SYCL inference engine. Use the tools "
                "to read what you need before answering. Keep changes small and explain them.")
AGENT_TASK = ("Find where the prompt (prefill) path splits a long prompt into chunks, explain how the chunk size is "
              "chosen, and propose a patch that logs the chosen size once per prompt.")
TOOLS = [{"type": "function", "function": {
    "name": "read_file", "description": "Read a file of the repository.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "list_dir", "description": "List a directory of the repository.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]


def env(name, default=None):
    v = os.environ.get(name, default)
    if v is None:
        sys.exit(f"bench_usage: {name} is not set")
    return v


def git(*a):
    return subprocess.run(["git", *a], check=True, capture_output=True, text=True).stdout


def engine(spec):
    """`env K=V ... EXE` or `EXE` -> (EXE, {K: V})"""
    w = shlex.split(spec)
    extra = {}
    if w and w[0] == "env":
        w = w[1:]
        while w and "=" in w[0]:
            k, v = w.pop(0).split("=", 1)
            extra[k] = v
    if len(w) != 1:
        sys.exit(f"bench_usage: {spec!r}: one binary or wrapper (with arguments, write a wrapper script)")
    return w[0], extra


if len(sys.argv) != 3:
    sys.exit(f"usage: {sys.argv[0]} OUT GPU")
out, gpu = pathlib.Path(sys.argv[1]), sys.argv[2]
data = pathlib.Path(env("DATA"))
g = data / "models" / "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS"
pack = env("PACK", str(data / "packs" / "qwen-iq3_xxs"))
model = ["--pack", pack, "--native", env("NATIVE", f"{g}-00001-of-00002.gguf"),
         "--ple-gguf", env("PLE", f"{g}-00002-of-00002.gguf"), *shlex.split(env("EXTRA", ""))]
ctx = int(env("CONTEXT", "262144"))
context = ["--max-context", str(ctx)] + (["--kv", "int8"] if ctx > 8192 else []) + \
    (["--kv-resident", "32768"] if ctx >= 65536 else [])
arms = {a: (*engine(env(a.upper())), pathlib.Path(env(f"{a.upper()}_ROOT")), env(f"{a.upper()}_PROFILE"))
        for a in ("xe", "up")}
py, port, rounds = env("PY", sys.executable), int(env("PORT", "18095")), int(env("ROUNDS", "3"))
slow = float(env("SLOW", "10"))
runs = out / "usage-runs"
runs.mkdir(parents=True, exist_ok=True)
files = []                                   # the agent's tool results: the larger sources, cut to ~4K tokens each
for f in git("ls-tree", "-r", "--name-only", COMMIT, "src").split("\n"):
    if f.endswith(".cpp"):
        text = git("show", f"{COMMIT}:{f}")
        if len(text) > 12000:
            files.append(f"// {f}\n{text[:14000]}")
    if len(files) == STEPS - 1:
        break


def post(body, timeout=1200):
    """One streamed chat completion: (the assistant message, time to the first token, the server's timings)."""
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", method="POST",
                                 data=json.dumps({"model": "bench", "stream": True, "max_tokens": MAX_TOKENS,
                                                  **SAMPLING, **body}).encode(),
                                 headers={"Content-Type": "application/json"})
    t0, first, text, calls, timings = time.monotonic(), None, "", {}, None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            x = json.loads(line[6:])
            timings = x.get("timings") or timings
            for c in x.get("choices") or []:
                d = c.get("delta") or {}
                if first is None and (d.get("content") or d.get("reasoning_content") or d.get("tool_calls")):
                    first = time.monotonic() - t0
                text += d.get("content") or ""
                for tc in d.get("tool_calls") or []:
                    k = calls.setdefault(tc.get("index", 0), {"id": tc.get("id"), "type": "function",
                                                              "function": {"name": "", "arguments": ""}})
                    k["id"] = tc.get("id") or k["id"]
                    k["function"]["name"] += (tc.get("function") or {}).get("name") or ""
                    k["function"]["arguments"] += (tc.get("function") or {}).get("arguments") or ""
    msg = {"role": "assistant", "content": text}
    if calls:
        msg["tool_calls"] = [calls[i] for i in sorted(calls)]
    return msg, first, timings


def conversations(seed):
    """Every turn of the three conversations: (conversation, turn, ttft, timings)."""
    for name, (system, questions) in CHATS.items():
        msgs = [{"role": "system", "content": system}]
        for i, q in enumerate(questions):
            msgs.append({"role": "user", "content": q})
            m, ttft, tm = post({"messages": msgs, "seed": seed})
            msgs.append(m)
            yield name, i, ttft, tm
    msgs = [{"role": "system", "content": AGENT_SYSTEM}, {"role": "user", "content": AGENT_TASK}]
    for step in range(STEPS):
        last = step == STEPS - 1
        m, ttft, tm = post({"messages": msgs, "seed": seed, "tools": TOOLS,
                            "tool_choice": "auto" if last else "required"})
        yield "agent", step, ttft, tm
        if last or not m.get("tool_calls"):
            break
        msgs.append(m)
        for c in m["tool_calls"]:              # the step's file answers the first call; any others get a note
            msgs.append({"role": "tool", "tool_call_id": c["id"],
                         "content": files[step] if c is m["tool_calls"][0] else "(not available)"})


def run(arm, rnd, stem, warmup=False):
    exe, extra, root, profile = arms[arm]
    cfg = {"exe": exe, "cwd": str(root), "tokenizer": f"{pack}/tokenizer", "model_name": "bench", "port": port,
           "args": [*model, "--expert-profile", profile, "--expert-cache", "auto", "--prefill", "auto", "--spec", "4",
                    "--spec-min-p", "0.5", "--mtp", str(data / "mtp" / "rt"), *context]}
    (runs / f"{stem}.config.json").write_text(json.dumps(cfg, indent=1) + "\n")
    log = open(runs / f"{stem}.txt", "w")
    srv = subprocess.Popen([py, str(root / "serve" / "server.py"), "--engine", "strata", "--config",
                            str(runs / f"{stem}.config.json"), "--port", str(port), "--host", "127.0.0.1"],
                           cwd=root, env={**os.environ, **extra}, stdout=log, stderr=subprocess.STDOUT,
                           stdin=subprocess.DEVNULL, start_new_session=True)
    turns, why = [], None
    try:
        end = time.monotonic() + 900
        while True:                            # the server answers /health at once; the engine loads after
            if srv.poll() is not None:
                raise RuntimeError("the server ended while loading")
            if time.monotonic() > end:
                raise RuntimeError("not loaded after 900 s")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                    if json.loads(r.read()).get("loaded"):
                        break
            except OSError:
                pass
            time.sleep(2)
        for name, i, ttft, tm in conversations(1000 + rnd):
            turns.append({"conv": name, "turn": i, "ttft_s": ttft, **{k: (tm or {}).get(k) for k in (
                "prompt_n", "cache_n", "prompt_ms", "predicted_n", "predicted_ms", "draft_n", "draft_n_accepted")}})
            t = turns[-1]
            if warmup:                         # the JIT and the first loads: one answer is enough
                break
            if arm == "up" and slow > 0 and len(turns) == 1 and t["predicted_ms"] and \
                    t["predicted_n"] / t["predicted_ms"] * 1e3 <= slow:
                why = "stopped: output"
                break
            print(f"{gpu} {arm} {rnd} {name} {i}: ttft {ttft or 0:.2f} s, read {t['prompt_n']} in {t['prompt_ms']} ms,"
                  f" out {t['predicted_n']} in {t['predicted_ms']} ms", flush=True)
    except (OSError, RuntimeError, ValueError) as e:
        why = str(e)
    finally:
        try:
            os.killpg(srv.pid, signal.SIGTERM)
            srv.wait(timeout=120)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(srv.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            srv.wait()
        log.close()
    rec = {"gpu": gpu, "arm": arm, "round": rnd, "context": ctx, "turns": turns, "why": why}
    (runs / f"{stem}.json").write_text(json.dumps(rec, indent=1) + "\n")
    if why:
        print(f"{gpu} {arm} {rnd}: {why}", flush=True)
    return why


for a in ("xe", "up"):
    run(a, 0, f"{gpu}-{a}-warmup", warmup=True)
up_stopped = False
for r in range(1, rounds + 1):
    for a in (("xe", "up") if r % 2 else ("up", "xe")):
        if a == "up" and up_stopped:
            continue
        why = run(a, r, f"{gpu}-{a}-{r}")
        up_stopped |= a == "up" and (why or "").startswith("stopped")
