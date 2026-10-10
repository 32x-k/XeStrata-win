# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Task check of STRATA_ROUTE_TAIL_SKIP (upstream b4e3407b's, redone on XeStrata): greedy, thinking off, through
`strata --serve`; HumanEval every second problem (82) and 30 generated arithmetic reasoning tasks.

    task_tail.py GPU SELECTOR SKIP OUT

Writes OUT/<GPU>-skip<SKIP>.json: per task, passed or not, and the answer.  Environment as kl_tail.py; HumanEval.jsonl.gz
(openai/human-eval's data file) next to this script.  The answers' code is run in a subprocess with a 15 s limit.
"""
import gzip
import json
import os
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(os.environ["XE_REPO"])
sys.path.insert(0, str(REPO / "tools"))
import bench_core as B  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ["DATA"])
G = DATA / "models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS"
ENGINE = [os.environ["XE_BIN"], "--serve", "--pack", str(DATA / "packs/qwen-iq3_xxs"),
          "--native", f"{G}-00001-of-00002.gguf", "--ple-gguf", f"{G}-00002-of-00002.gguf",
          "--expert-profile", str(REPO / "data/expert-profile.bin"), "--expert-cache", "auto", "--prefill", "auto",
          "--spec", "4", "--spec-min-p", "0.5", "--mtp", str(DATA / "mtp/rt"), "--max-context", "8192"]


def wrap(q: str) -> str:
    return f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def reasoning_tasks() -> list[tuple[str, int]]:
    r = random.Random(1421)
    out = []
    for i in range(30):
        k = i % 5
        if k == 0:
            a, b, c = r.randint(12, 40), r.randint(3, 9), r.randint(20, 90)
            q = (f"A shop has {c} boxes with {a} pencils each. It sells {b} boxes and then opens one more box and "
                 f"gives away half of its pencils. How many pencils are left in total?")
            ans = (c - b) * a - a // 2
        elif k == 1:
            s, t1, t2 = r.randint(40, 90), r.randint(2, 5), r.randint(1, 4)
            q = (f"A car drives {t1} hours at {s} km/h and then {t2} hours at {s + 20} km/h. "
                 f"How many kilometres does it drive?")
            ans = t1 * s + t2 * (s + 20)
        elif k == 2:
            x, y = r.randint(3, 20), r.randint(3, 20)
            q = (f"Two numbers add up to {x + y} and their difference is {abs(x - y)}. "
                 f"What is their product?")
            ans = x * y
        elif k == 3:
            n, p = r.randint(5, 12), r.randint(2, 4)
            q = (f"A tournament has {n * p} players split into {p} equal groups. In each group every player plays "
                 f"every other player once. How many games are played in total?")
            ans = p * n * (n - 1) // 2
        else:
            a, b, c = r.randint(100, 900), r.randint(10, 60), r.randint(2, 7)
            q = (f"Mina has {a} yen. She buys {c} notebooks at {b} yen each and then spends a third of what is "
                 f"left (rounded down) on lunch. How many yen does she have now?")
            left = a - b * c
            ans = left - left // 3
        out.append((q + " Think step by step, and end with a line 'Answer: <integer>'.", ans))
    return out


def run_code(program: str) -> bool:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.py"
        p.write_text(program)
        try:
            r = subprocess.run([sys.executable, str(p)], cwd=d, capture_output=True, timeout=15,
                               env={"PATH": "/usr/bin:/bin"})
        except subprocess.TimeoutExpired:
            return False
        return r.returncode == 0


def code_of(answer: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\n(.*?)```", answer, re.S)
    return blocks[0] if blocks else answer


def main() -> None:
    gpu, sel, skip, out = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4])
    out.mkdir(parents=True, exist_ok=True)
    tok = B.load_tokenizer(DATA / "packs/qwen-iq3_xxs/tokenizer")
    he = [json.loads(x) for x in gzip.open(HERE / "HumanEval.jsonl.gz", "rt")][::2]
    tasks = [("he", p["task_id"], "Complete the following Python function. Reply with the whole function in one "
              "```python code block.\n\n```python\n" + p["prompt"] + "```", p, 768) for p in he]
    tasks += [("rs", f"rs/{i}", q, a, 1024) for i, (q, a) in enumerate(reasoning_tasks())]
    env = {**os.environ, "ONEAPI_DEVICE_SELECTOR": sel, "STRATA_ROUTE_TAIL_SKIP": skip,
           "LD_LIBRARY_PATH": os.environ["XE_LD"]}
    log = open(out / f"{gpu}-skip{skip}.log", "w")
    p = subprocess.Popen(ENGINE, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True,
                         bufsize=1)
    assert p.stdin and p.stdout
    while not (x := p.stdout.readline()).startswith("READY"):
        if not x:
            sys.exit("engine ended before READY")
    results = []
    for kind, tid, q, ref, max_new in tasks:
        p.stdin.write(f"GEN {max_new} {','.join(map(str, tok.encode(wrap(q), parse_special=True)))}\n")
        p.stdin.flush()
        gen = []
        while (x := p.stdout.readline()):
            if x.startswith("T "):
                gen.append(int(x.split()[1]))
            elif x.startswith(("DONE", "ERR")):
                break
        text = tok.decode(gen).replace("<|im_end|>", "")
        if kind == "he":
            code = code_of(text)
            prog = code if f"def {ref['entry_point']}" in code else ref["prompt"] + code
            ok = run_code(prog + "\n\n" + ref["test"] + f"\n\ncheck({ref['entry_point']})\n")
        else:
            m = re.findall(r"Answer:\s*\**\s*(-?[\d,]+)", text)
            ok = bool(m) and int(m[-1].replace(",", "")) == ref
        results.append({"id": tid, "ok": ok, "answer": text})
        print(f"{tid} {'ok' if ok else 'NG'}", flush=True)
    p.stdin.write("QUIT\n")
    p.stdin.flush()
    p.wait(timeout=120)
    (out / f"{gpu}-skip{skip}.json").write_text(json.dumps(results, ensure_ascii=False))
    he_ok = sum(r["ok"] for r in results if r["id"].startswith("HumanEval"))
    rs_ok = sum(r["ok"] for r in results if r["id"].startswith("rs/"))
    print(f"{gpu} skip={skip}: HumanEval {he_ok}/{len(he)}, reasoning {rs_ok}/30")


if __name__ == "__main__":
    main()
