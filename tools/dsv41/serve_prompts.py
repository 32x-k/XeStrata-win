#!/usr/bin/env python3
"""serve_prompts.py PROMPTS.json OUT.json -- ENGINE CMD...
Starts the engine in --serve mode (appended), waits for READY, sends GEN for every prompt (greedy), compares the
generated ids with the llama.cpp reference's greedy ids (expect_ids), and prints one line each:
  id mode  prompt_tokens  n_gen  match_prefix/n_expect  first_divergence  prefill ms  decode tok/s
"""
import json, subprocess, sys, time

prompts_path, out_path = sys.argv[1], sys.argv[2]
cmd = sys.argv[sys.argv.index("--") + 1:] + ["--serve"]
prompts = json.load(open(prompts_path))
only = [a for a in sys.argv[3:sys.argv.index("--")]]
if only:
    prompts = [p for p in prompts if p["id"] in only or (p["mode"] + ":" + p["id"]) in only]
p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr, text=True, bufsize=1)
t0 = time.time()
while True:
    line = p.stdout.readline()
    if not line:
        print("engine exited before READY", file=sys.stderr)
        sys.exit(2)
    if line.startswith("READY"):
        print("READY after %.0f s: %s" % (time.time() - t0, line.strip()), flush=True)
        break
    print("engine>", line.rstrip()[:200], flush=True)
res = []
for pr in prompts:
    ids = pr["ids"]
    mx = min(pr["max_tokens"], max(len(pr["expect_ids"]) + 4, 8))
    p.stdin.write("GEN %d %s\n" % (mx, ",".join(map(str, ids))))
    p.stdin.flush()
    gen, done = [], None
    while True:
        line = p.stdout.readline()
        if not line:
            break
        if line.startswith("T "):
            gen.append(int(line.split()[1]))
        elif line.startswith("DONE"):
            done = line.split()
            break
    exp = pr["expect_ids"]
    n = 0
    while n < min(len(gen), len(exp)) and gen[n] == exp[n]:
        n += 1
    full = n == len(exp) or (n == len(gen) and n >= len(exp) - 1)
    # DONE <generated> <prompt> <prompt ms> <decode ms> <stop|length|cancel> ...
    pms = float(done[3]) if done and len(done) > 4 else 0.0
    dms = float(done[4]) if done and len(done) > 4 else 0.0
    row = {"id": pr["id"], "mode": pr["mode"], "prompt_tokens": len(ids), "n_gen": len(gen), "n_expect": len(exp),
           "match_prefix": n, "first_div": None if n >= min(len(gen), len(exp)) else n, "prompt_ms": pms,
           "decode_ms": dms, "decode_tps": (len(gen) * 1000.0 / dms) if dms > 0 else 0.0,
           "prompt_tps": (len(ids) * 1000.0 / pms) if pms > 0 else 0.0, "gen": gen}
    res.append(row)
    print("%-26s %-8s prompt %4d (%.1f tok/s)  gen %3d  match %3d/%3d  div %s  decode %.2f tok/s" % (
        pr["id"], pr["mode"], len(ids), row["prompt_tps"], len(gen), n, len(exp), row["first_div"], row["decode_tps"]), flush=True)
    json.dump(res, open(out_path, "w"))
p.stdin.write("QUIT\n")
p.stdin.flush()
try:
    p.wait(timeout=60)
except Exception:
    p.terminate()
