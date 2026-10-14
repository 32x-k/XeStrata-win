#!/usr/bin/env python3
"""Needle-in-a-haystack recall check for the record in this folder.

A unique code is buried at a chosen depth in a long filler prompt; the question then asks for it.
One request per cell, fresh context (the leading tokens vary so the conversation cache cannot serve
it), temperature 0, so each cell is a full cold prefill.

usage: needles.py [sizes-tokens ...]      e.g. needles.py 4096 32768 131072
writes needles.json in this directory.

Checks recall only — it says nothing about answer quality in general.
"""
import json, os, random, re, sys, time, urllib.request
from pathlib import Path

URL = "http://127.0.0.1:8095/v1/chat/completions"
OUT = Path(__file__).with_name("needles.json")
SIZES = [int(a) for a in sys.argv[1:]] or [4096, 32768, 131072]
DEPTHS = [0.1, 0.5, 0.9]
FILLER = ("The archive records a routine quarterly ledger entry for the regional office. ")
TOKENS_PER_WORD = 1.14          # measured on this harness; the true count comes back in `timings`

rows = []
for size in SIZES:
    for depth in DEPTHS:
        rng = random.Random(f"{size}-{depth}")           # same cells every run, reproducible
        code = str(rng.randrange(10000, 99999))
        needle = f" Remember this fact: the access code for vault seven is {code}. "
        words = max(64, int(size / TOKENS_PER_WORD) - 32)
        body = FILLER * (words // 13 + 1)
        cut = int(len(body) * depth)
        # cut on a word boundary so the needle is not split
        cut = body.find(" ", cut)
        haystack = body[:cut] + needle + body[cut:]
        prompt = (f"[probe {size}/{depth}] Read the text and answer.\n\n{haystack}\n\n"
                  f"Question: what is the access code for vault seven? Answer with the digits only.")
        req = urllib.request.Request(URL, data=json.dumps({
            "model": "x", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 32, "temperature": 0, "reasoning_effort": "none"}).encode(),
            headers={"Content-Type": "application/json"})
        t0 = time.time()
        try:
            r = json.load(urllib.request.urlopen(req, timeout=900))
            tim = r.get("timings") or {}
            text = (r.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
            found = bool(re.search(rf"\b{code}\b", text))
            row = {"size_tokens_target": size, "depth": depth, "code": code, "recalled": found,
                   "prompt_tokens": tim.get("prompt_n"), "prompt_ms": round(tim.get("prompt_ms", 0)),
                   "pp_tok_s": round(tim.get("prompt_per_second", 0), 1),
                   "reply": text.strip()[:80], "wall_s": round(time.time() - t0, 1)}
        except Exception as e:
            row = {"size_tokens_target": size, "depth": depth, "code": code, "recalled": False,
                   "error": str(e)[:120]}
        rows.append(row)
        print(f"  {size:>7} target / depth {depth:.1f}: prompt {row.get('prompt_tokens')} tok, "
              f"recalled={row['recalled']}  reply={row.get('reply','')[:40]!r}")

json.dump({"method": "needle-in-a-haystack: a unique 5-digit code buried at 10/50/90% of a filler "
                     "prompt, asked back with temperature 0 and a 24-token cap; one fresh-context "
                     "request per cell (a full cold prefill)",
           "cells": rows,
           "recalled": sum(1 for r in rows if r["recalled"]), "total": len(rows)},
          open(OUT, "w"), indent=1)
print(f"  wrote {OUT.name}: {sum(1 for r in rows if r['recalled'])}/{len(rows)} recalled")
