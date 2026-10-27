#!/usr/bin/env python3
"""Aggregate output: N concurrent clients against one server, measuring tokens per second summed over the clients. Run the
same N against a second configuration to compare.

    python tools/dgx_concurrent.py CONFIG OUT.json --clients 8 [--clients 16] [--prompt 2000] [--max-new 128]
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORDS = ("river stone market lantern copper winter garden signal harbor needle orbit velvet canyon ember ledger quartz "
         "meadow anchor violet thunder cedar marble piston ribbon cobalt falcon glacier hollow ivory juniper kernel").split()


def make_prompt(tokens: int) -> str:
    body = []
    n = int(tokens / 1.35)
    for i in range(n):
        body.append(WORDS[i % len(WORDS)])
    return "Summarise the following log in two sentences.\n\n" + " ".join(body)


def serve(cfg_path: str, port: int):
    cfg = json.loads(Path(cfg_path).read_text())
    cfg["port"] = port
    cfg["log"] = f"/tmp/dgx-conc-{port}.log"
    tmp = Path(f"/tmp/dgx-conc-{port}.json")
    tmp.write_text(json.dumps(cfg, indent=1))
    proc = subprocess.Popen([sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata", "--config",
                             str(tmp), "--port", str(port)], cwd=str(ROOT),
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.time()
    while time.time() - t0 < 1800:
        if proc.poll() is not None:
            raise SystemExit(f"server exited {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as r:
                if r.status == 200:
                    return proc
        except Exception:
            pass
        time.sleep(3)
    raise SystemExit("server did not come up")


def one(port: int, prompt: str, max_new: int, out: list, idx: int):
    body = json.dumps({"messages": [{"role": "user", "content": prompt}], "max_tokens": max_new, "temperature": 0,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        res = json.loads(r.read())
    out[idx] = (time.time() - t0, res.get("timings", {}).get("predicted_n", 0))


def run(cfg_path: str, port: int, clients: int, prompt: str, max_new: int) -> dict:
    proc = serve(cfg_path, port)
    try:
        with urllib.request.urlopen(urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                data=json.dumps({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}).encode(),
                headers={"Content-Type": "application/json"}), timeout=1800) as r:
            r.read()
        out: list = [None] * clients
        t0 = time.time()
        threads = [threading.Thread(target=one, args=(port, prompt, max_new, out, i)) for i in range(clients)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.time() - t0
        tokens = sum(n for _, n in out if n)
        times = [d for d, _ in out if d]
        return {"clients": clients, "wall_s": round(wall, 2), "tokens": tokens,
                "aggregate_tok_s": round(tokens / wall, 2) if wall > 0 else None,
                "median_latency_s": round(statistics.median(times), 2)}
    finally:
        proc.terminate()
        try:
            proc.wait(180)
        except subprocess.TimeoutExpired:
            proc.kill()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("out")
    ap.add_argument("--clients", type=int, nargs="+", default=[8, 16])
    ap.add_argument("--prompt", type=int, default=2000)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--port", type=int, default=8091)
    a = ap.parse_args()
    prompt = make_prompt(a.prompt)
    rows = [run(a.config, a.port, c, prompt, a.max_new) for c in a.clients]
    Path(a.out).write_text(json.dumps(rows, indent=1))
    for r in rows:
        print(json.dumps(r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
