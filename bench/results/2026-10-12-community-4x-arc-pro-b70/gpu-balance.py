#!/usr/bin/env python3
"""Per-card work balance during one decode, with power draw rather than utilisation %.

nvtop's percentage is compute-engine busy time, which differs per *layer type*: stage 0 runs the
embedding gather, PLE and the hyper-connection broadcast (memory-heavy, few FLOPs), the middle stages
run dense projections and MoE experts (compute-heavy), the last runs the head and the MTP drafter.
So unequal percentages are expected even when the planner balanced the stages by *time*.  Power and
memory traffic are the honest measures of how much work each card actually did.

Usage: gpu-balance.py [TOKENS]
"""
import json, statistics, subprocess, sys, threading, time, urllib.request

TOKENS = int(sys.argv[1]) if len(sys.argv) > 1 else 400
N = 4
stop = threading.Event()
power = {i: [] for i in range(N)}
util = {i: [] for i in range(N)}
mem = {i: [] for i in range(N)}


def num(v):
    """xpu-smi nests metrics as {"tile_0": {"avg":..,"current":..}}; dig to a number."""
    if isinstance(v, dict):
        for k in ("current", "avg", "max"):
            if k in v: return float(v[k])
        for sub in v.values():
            r = num(sub)
            if r is not None: return r
        return 0.0
    try: return float(v)
    except (TypeError, ValueError): return 0.0


def sampler():
    while not stop.is_set():
        for d in range(N):
            try:
                j = json.loads(subprocess.run(["sudo", "-n", "xpu-smi", "stats", "-d", str(d), "-j"],
                                              capture_output=True, text=True, timeout=20).stdout)
                power[d].append(num(j["power"]["gpu_power_w"]))
                ce = j["utilization"]["compute_engines"]
                vals = [v.get("current", 0.0) for v in ce.values() if isinstance(v, dict)]
                util[d].append(max(vals) if vals else 0.0)
                mem[d].append(num(j["memory"].get("util_percent", 0.0)))
            except Exception:
                pass


body = "The quick brown fox jumps over the lazy dog. " * 60
req = urllib.request.Request("http://127.0.0.1:8095/v1/chat/completions",
    data=json.dumps({"model": "x", "messages": [{"role": "user",
        "content": body + " Write a detailed essay about numbers."}],
        "max_tokens": TOKENS, "temperature": 0}).encode(),
    headers={"Content-Type": "application/json"})

t = threading.Thread(target=sampler, daemon=True)
t.start()
t0 = time.time()
r = json.load(urllib.request.urlopen(req, timeout=600))
wall = time.time() - t0
stop.set(); t.join(timeout=6)
tm = r.get("timings") or {}

print(f"  {tm.get('predicted_n')} tokens in {tm.get('predicted_ms',0)/1000:.1f} s "
      f"({tm.get('predicted_per_second',0):.1f} tok/s), {len(power[0])} samples/card")
print(f"  {'card':6} {'power W (mean/peak)':22} {'compute % (mean)':18} {'mem % (mean)':14}")
tot = 0.0
for d in range(N):
    if not power[d]:
        print(f"  card{d}  no samples"); continue
    p = statistics.mean(power[d]); tot += p
    print(f"  card{d}   {p:6.1f} / {max(power[d]):6.1f}       {statistics.mean(util[d]):6.1f}            {statistics.mean(mem[d]):6.1f}")
if tot:
    print(f"  total {tot:.0f} W   (each card's share of the work, by power: "
          + ", ".join(f"card{d} {100*statistics.mean(power[d])/tot:.0f}%" for d in range(N)) + ")")
