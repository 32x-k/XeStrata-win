# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Teacher-forced KL of STRATA_ROUTE_TAIL_SKIP (upstream b4e3407b's check, redone on XeStrata).

    kl_tail.py GPU SELECTOR OUT [N_PROMPTS]

Per prompt: the tail-skip default (=7) writes 220 greedy tokens; that text is then followed (--spec-follow) by =7,
=0 and =0 again, each dumping the logits of the emitted positions.  Prints per prompt and in all: KL(off||on),
p99, max, top-1 agreement, the forced text's perplexity ratio, the first token's KL, and the floor KL(off||off).
Environment: XE_REPO (the tree, for tools/bench_core.py and data/expert-profile.bin), XE_BIN (the engine), XE_LD (its
LD_LIBRARY_PATH), DATA (the data folder with packs/qwen-iq3_xxs, the IQ3_XXS model files and mtp/rt).
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(os.environ["XE_REPO"])
sys.path.insert(0, str(REPO / "tools"))
import bench_core as B  # noqa: E402

DATA = Path(os.environ["DATA"])
G = DATA / "models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS"
ENGINE = [os.environ["XE_BIN"], "--pack", str(DATA / "packs/qwen-iq3_xxs"),
          "--native", f"{G}-00001-of-00002.gguf", "--ple-gguf", f"{G}-00002-of-00002.gguf",
          "--expert-profile", str(REPO / "data/expert-profile.bin"), "--expert-cache", "auto", "--prefill", "auto",
          "--spec", "4", "--spec-min-p", "0.5", "--mtp", str(DATA / "mtp/rt"), "--max-context", "16384"]
MAX_NEW = 220

PROMPTS = [
    ("code", "Write a Python function that merges overlapping intervals in a list and explain its complexity."),
    ("code", "In C++, implement a thread-safe bounded queue with a condition variable. Keep it short."),
    ("code", "Explain what this does and fix the bug:\n\nint sum(int* a, int n) { int s; for (int i = 0; i <= n; i++) s += a[i]; return s; }"),
    ("code", "Write a SQL query that returns, for each customer, their three most recent orders."),
    ("reason", "A train leaves at 9:40 and arrives at 13:15 after two stops of 12 minutes each. How long was it actually moving? Think step by step."),
    ("reason", "If 3 painters paint 3 walls in 3 hours, how many painters are needed to paint 12 walls in 6 hours? Explain."),
    ("reason", "Prove that the square root of 2 is irrational."),
    ("reason", "A bag has 5 red and 7 blue balls. Two are drawn without replacement. What is the probability both are the same colour?"),
    ("chat", "I have a job interview tomorrow and I'm nervous. Any advice?"),
    ("chat", "Suggest a three-day itinerary for Kyoto in autumn."),
    ("chat", "What are the pros and cons of electric cars compared with hybrids?"),
    ("ja", "日本の四季それぞれの魅力を、短い段落で説明してください。"),
    ("ja", "Python のリスト内包表記とジェネレーター式の違いを初心者向けに説明して。"),
    ("zh", "请简要介绍一下中国古代四大发明及其影响。"),
    ("ru", "Объясни простыми словами, как работает квантовый компьютер."),
    ("de", "Erkläre den Unterschied zwischen Wetter und Klima in einfachen Worten."),
    ("fr", "Donne-moi une recette simple de ratatouille et quelques conseils."),
    ("es", "¿Cuáles son las causas principales del cambio climático? Responde de forma breve."),
    ("ko", "한국의 전통 음식 세 가지를 소개해 주세요."),
    ("doc", None),
    ("doc", None),
]


def wrap(q: str) -> str:
    return f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def doc_prompts() -> list[str]:
    corpus = B.load_corpus()
    files = corpus["files"] if isinstance(corpus, dict) and "files" in corpus else None
    texts = []
    if files:
        for f in files:
            path, body = f[0], f[1]
            if 12000 < len(body) < 40000:
                texts.append((path, body[:18000]))
            if len(texts) == 2:
                break
    return [f"Summarise this source file in a few paragraphs: what it is for and how it works.\n\n"
            f"=== {p} ===\n{t}" for p, t in texts]


def run(env_skip: int, sel: str, ids: Path, log: Path, follow: Path | None, dump: Path | None) -> str:
    env = {**os.environ, "ONEAPI_DEVICE_SELECTOR": sel, "STRATA_ROUTE_TAIL_SKIP": str(env_skip),
           "LD_LIBRARY_PATH": os.environ["XE_LD"]}
    cmd = [*ENGINE, "--tokens-file", str(ids), "--max-new", str(MAX_NEW)]
    if follow:
        cmd += ["--spec-follow", str(follow)]
    if dump:
        cmd += ["--dump-logits", str(dump)]
    with open(log, "w") as f:
        subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=1800, check=False)
    return log.read_text(errors="replace")


def logits(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32, count=2)
    v, n = int(raw[0]), int(raw[1])
    return np.fromfile(path, dtype=np.float32, offset=8).reshape(n, v)


def logsm(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=1, keepdims=True))


def kl(a: np.ndarray, b: np.ndarray) -> np.ndarray:   # KL(a || b), per row
    la, lb = logsm(a), logsm(b)
    return (np.exp(la) * (la - lb)).sum(axis=1)


def main() -> None:
    gpu, sel, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    count = int(sys.argv[4]) if len(sys.argv) > 4 else len(PROMPTS)
    out.mkdir(parents=True, exist_ok=True)
    tok = B.load_tokenizer(DATA / "packs/qwen-iq3_xxs/tokenizer")
    docs = doc_prompts()
    prompts, di = [], 0
    for kind, q in PROMPTS:
        if q is None:
            if di >= len(docs):
                continue
            q, di = docs[di], di + 1
        prompts.append((kind, q))
    results = []
    for i, (kind, q) in enumerate(prompts[:count]):
        ids = out / f"p{i}.ids"
        ids.write_text(",".join(str(t) for t in tok.encode(wrap(q), parse_special=True)))
        text = run(7, sel, ids, out / f"p{i}-gen.log", None, None)
        follow_ids = B.output_ids(text)
        if not follow_ids:
            print(f"p{i} {kind}: no output\n{B.engine_error(text)}", flush=True)
            continue
        follow = out / f"p{i}.follow"
        follow.write_text(follow_ids)
        dumps = {}
        for name, skip in (("on", 7), ("off", 0), ("off2", 0)):
            d = out / f"p{i}-{name}.bin"
            t = run(skip, sel, ids, out / f"p{i}-{name}.log", follow, d)
            if not d.exists():
                print(f"p{i} {name}: no dump\n{B.engine_error(t)}", flush=True)
                break
            dumps[name] = d
        if len(dumps) < 3:
            continue
        on, off, off2 = (logits(dumps[n]) for n in ("on", "off", "off2"))
        n = min(len(on), len(off), len(off2))
        on, off, off2 = on[:n], off[:n], off2[:n]
        forced = np.array([int(t) for t in follow_ids.split(",")][:n])
        k, floor = kl(off, on), kl(off, off2)
        nll_on = -logsm(on)[np.arange(n), forced]
        nll_off = -logsm(off)[np.arange(n), forced]
        r = {"i": i, "kind": kind, "n": n, "kl": k.tolist(), "floor": floor.tolist(),
             "top1": float((on.argmax(1) == off.argmax(1)).mean()),
             "ppl_on": float(np.exp(nll_on.mean())), "ppl_off": float(np.exp(nll_off.mean())),
             "skipped": [ln for ln in (out / f"p{i}-on.log").read_text(errors="replace").splitlines()
                         if "tail" in ln.lower()][-3:]}
        results.append(r)
        print(f"p{i:2d} {kind:6s} n={n:3d} KL {k.mean():.5f} p99 {np.percentile(k, 99):.4f} max {k.max():.3f} "
              f"first {k[0]:.4f} top1 {r['top1']:.3f} ppl {r['ppl_on'] / r['ppl_off']:.4f} "
              f"floor {floor.mean():.5f} (first {floor[0]:.4f})", flush=True)
        for d in dumps.values():
            d.unlink()
    (out / "kl.json").write_text(json.dumps(results))
    if results:
        k = np.concatenate([np.array(r["kl"]) for r in results])
        fl = np.concatenate([np.array(r["floor"]) for r in results])
        ppl_on = np.exp(np.mean([np.log(r["ppl_on"]) for r in results]))
        ppl_off = np.exp(np.mean([np.log(r["ppl_off"]) for r in results]))
        top1 = np.average([r["top1"] for r in results], weights=[r["n"] for r in results])
        first = np.array([r["kl"][0] for r in results])
        print(f"ALL {gpu}: tokens {len(k)} KL mean {k.mean():.5f} p99 {np.percentile(k, 99):.4f} max {k.max():.3f} "
              f"top1 {top1:.4f} ppl on/off {ppl_on / ppl_off:.4f} first-token KL {first.mean():.4f} "
              f"floor {fl.mean():.5f} (p99 {np.percentile(fl, 99):.4f}, first token "
              f"{np.mean([r['floor'][0] for r in results]):.4f})")


if __name__ == "__main__":
    main()
