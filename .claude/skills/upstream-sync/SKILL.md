---
name: upstream-sync
description: Bring upstream Strata's newest release tag into XeStrata on dev - sort its commits, merge, port the CUDA and HIP changes to the SYCL kernels, build the three modes - and make XeStrata at least as fast as that tag (at most 1% slower) on the Intel, NVIDIA and AMD GPUs: IQ3_XXS at 1K, 32K and 64K, and the other weight types at 32K, on the generate path, IQ3_XXS at 1K, 32K and 64K on the serve path, and everyday chats and a coding agent through the server. Only when the user asks (/upstream-sync).
disable-model-invocation: true
argument-hint: "[upstream TAG, e.g. v0.1.41]"
---

# Following upstream

upstream is Niko1221/Strata (the remote `upstream`).
A run brings one of its release tags into `dev` and ends when XeStrata is at least as fast as that tag on every GPU it can reach: the check of step 7 prints `ALL PASS`.

Rules for the whole run:

- The tag is `$ARGUMENTS` when given, else upstream's newest: `git ls-remote --tags --refs upstream 'v*' | sed 's#.*refs/tags/##' | sort -V | tail -n 1`; not `git tag -l 'v*'`, which also lists other remotes' tags (Project Maya's `upstream-maya` has v1.0.x). Then `git fetch upstream --tags`.
- Work and commits are on `dev` (Japanese commit messages, one logical change per commit, `git add` and `git commit` joined with `&&`). Nothing is pushed; a release is /release, not this.
- Only merged commits: upstream's open pull requests wait until upstream merges them.
- The GPUs: the Intel Arc Pro B70 and the RTX 4070 in the development machine, the RX 9060 XT in the AMD container. Where they are and how to run there: `LOCAL.md` in the repository root (kept out of git; when it is missing, ask the user and write it there).
  A GPU that cannot be reached (the container's host down, the card missing) is skipped: go on with the others, and report it and its cells as `unverified`.
- AMD work (builds, tests, runs) happens in the AMD container only, never on the development machine; the source goes in as a `git bundle` or a diff.
- A build that fails (any mode, any machine, XeStrata's or upstream's), and a fix that is not plain: stop and report it with the end of the log.
- Speed first: do not stop making a losing cell faster on your own judgement (a small expected gain, hard work, a compiler in the way). When every idea has been tried, report what was tried and the next ideas, and ask.
- Correctness is not traded: the FP64 references and the comparison with llama.cpp still hold, and no GPU gets slower to make another faster.
- Choose a path from what the device reports (AGENTS.md, Hardware independence), never from a GPU's name.

## 1. The starting point

```sh
git status --short                        # empty, on dev
git fetch upstream --tags
grep -n STRATA_UPSTREAM_VERSION CMakeLists.txt
```

The previous integration is the merge `upstream の Strata <version> を Xe に統合する`; its second parent is the last upstream commit `dev` has (`git log --merges --first-parent --format='%h %p %s' dev`).
upstream rewrote its history on 2026-10-06, so `git merge-base dev upstream/main` can be empty: the base is that parent (or, for history before it, its rewritten counterpart).
When the tag is already integrated, go to step 6.

## 2. Sort the commits

Write `.upstream-<version>.md` (add it to `.git/info/exclude`, as `.upstream-0140.md` is) with every commit of `<base>..<tag>` (`git log --no-merges --reverse`), one checkbox line each, in the groups of `.upstream-0140.md`:

| Group | What | Taken |
| --- | --- | --- |
| A | AMD, HIP | ported to the SYCL paths and measured on the RX 9060 XT; one only for a GPU there is no machine for (gfx11, gfx103x): step 4's last part |
| B | bench records | the records as they are |
| D | documents | what is true of XeStrata, into its documents (Japanese first) |
| E | engine features and fixes | all |
| M | several GPUs | all |
| N | NVIDIA, every generation (Pascal, Volta, Turing included) | ported to the SYCL kernels; one only for a generation there is no machine for: step 4's last part |
| P | speed paths | step 4 |
| S | upstream's `sycl/` | not taken: it stays deleted in the merge and is measured as a competitor (step 6) |
| T | serve, setup, tools | the merge |
| W | Windows | not taken |

Each line ends with what became of it (a commit, `N/A` and why, or the measurement that left it out).

## 3. The merge

```sh
tree=$(git merge-tree --write-tree --merge-base=<base> dev <tag>)   # resolve the conflicts it reports
git commit-tree -p dev -p <tag> -m "upstream の Strata <version> を Xe に統合する" <tree>
```

`sycl/` and the CUDA and HIP sources stay deleted (modify/delete conflicts are taken as deletes); their changes are ported by hand in steps 2 and 4.
Set `STRATA_UPSTREAM_VERSION` in `CMakeLists.txt` and `XESTRATA_REVISION` back to 1 (AGENTS.md, "Versions"), and replace any old upstream hash a document names with the rewritten one.

## 4. Port and measure

Build folders are configured with Ninja and ccache (docs/DEVTOOLS.md, "Faster builds"); check the hits with `ccache -s`.
While porting, build and measure with the contrib-llvm build (`build/contrib-llvm`) only; the free mode comes once at the end (step 5).

- Port related items as a group, each item behind an environment switch so the group can be split.
- A change that can be checked bitwise is checked by a probe alone; an opt-in feature is run once to see it works.
- A speed path is measured on all three GPUs and on each with its matrix units off (`STRATA_NO_XMX`, `STRATA_NO_BF16_MMA`, `STRATA_NO_INT8_MMA`): probes on the B70 and the RTX 4070 may run together, whole-engine A/B runs one card at a time.
- Kept when it is faster on one GPU and slower on none (chosen at run time from what the device reports where it helps only some); removed when it is faster nowhere. The numbers go into the commit and the record.
- Only when a group is not faster, split it with its switches.

### GPUs there is no machine for

A speed path only for GPUs there is no machine for (NVIDIA Pascal, Volta and Turing; AMD gfx103x and gfx11) is ported too (the user's decision of 2026-10-09), on upstream's measurements:

- one upstream turns on by default there, with a measurement on such a GPU (its commit or its records): on by default on the same GPUs;
- one upstream keeps opt-in: opt-in, behind the same kind of switch.

It is chosen from what the device reports or from the compiled target (`__CUDA_ARCH__`, `__GFX11__`), never from a GPU's name, and it must not change the path of a GPU there is a machine for.
Its correctness is checked on what there is:

| GPUs | Check |
| --- | --- |
| NVIDIA Pascal, Volta, Turing | a contrib-llvm build with only that generation's code (`STRATA_CUDA_ARCHS=sm_60`, `sm_70`, `sm_75`) run on the RTX 4070 (its driver compiles the PTX; docs/BUILD.md did so for Pascal): CTest and the bitwise probes |
| AMD gfx103x, gfx11 | a build for them (`STRATA_HIP_ARCHS=gfx1030;gfx1100`) in the AMD container; a path without their own instructions forced on the RX 9060 XT by its switch; code with their own instructions (gfx11's WMMA) is checked by the build alone |
| GPUs without matrix units (Pascal, RDNA2, Intel without XMX) | the fallback paths with `STRATA_NO_XMX`, `STRATA_NO_BF16_MMA`, `STRATA_NO_INT8_MMA` on the three GPUs |

What could not run is `unverified`. These GPUs are not part of step 7's check: the record and the report list each such path with upstream's measurement and `unverified` on XeStrata's speed.
Paths an earlier merge left out for want of a machine (the `移さない` lines of `.upstream-0140.md`'s group A) are taken up the same way.

## 5. The three modes

```sh
tools/lint/run.sh                          # the files that differ from HEAD
```

Then build and run CTest in `build/contrib-llvm` and `build/free` (intel/llvm, without oneAPI's environment; the contrib-icpx mode is not built or tested), and in the AMD container its contrib-llvm build with HIP (`STRATA_HIP_ARCHS`).
A warning or a mismatch only one mode shows is traced back to the commit that caused it and fixed there.
AVX-512 code touched: AGENTS.md's Intel SDE runs.

## 6. Against the tag

The engines, each built in a worktree of its own under `.claude/worktrees/` (or in the AMD container):

- XeStrata: `dev`'s HEAD, the contrib-llvm build; one GPU at a time with `ONEAPI_DEVICE_SELECTOR` (`level_zero:*`, `cuda:*`; `hip:*` in the container).
- upstream on the B70: the tag's `sycl/` port, built and run in its Docker image as its docs/INTEL.md says.
- upstream on the RTX 4070: the tag's CUDA build; on the RX 9060 XT: its HIP build; both as its documents say.

Each engine's own shipped expert profile. No other work on a machine while it measures.

Every GPU runs three measurements, each with both engines, and step 7 judges all three:

| Measurement | Script | What it covers |
| --- | --- | --- |
| generate | `bench_generate.py` | the kernels and the speculative loop, both engines writing one text (`--spec-follow`); every model file below |
| serve | `bench_serve.py` | `strata --serve`'s own decode loop and the conversation checkpoints, a chat of four turns; IQ3_XXS |
| everyday use | `bench_usage.py` | the whole path a user meets: `serve/server.py`, the chat template, streaming, chats and a coding agent; IQ3_XXS |

Each GPU is one matrix path (the B70 XMX, the RTX 4070 tensor cores, the RX 9060 XT WMMA), and every GPU runs the same model files (their paths on each machine: `LOCAL.md`):

| File | Tiers (`TIERS`) | What it covers |
| --- | --- | --- |
| IQ3_XXS | `64k 32k 1k`; the final run `262k 64k 32k 1k` | the experts' Q2_0, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S and IQ4_NL; the three kinds of prompt |
| Q2_0 | `32k` | the dense projections' Q3_K, Q4_0 and Q5_0 (no other file has them) |
| IQ2_XS | `32k` | the experts' IQ1_M |
| IQ3_S | `32k` | the experts' IQ4_XS |
| UD-Q4_K_XL | `32k` | the experts' K-quants (Q4_K, Q5_K), Q5_1 and Q8_0 |

Each tier stands for a kind of prompt: 1K the chunks below 1,024 tokens (the CPU takes a share of the experts), 32K a long context whose K/V is all in VRAM, 64K the K/V streamed from RAM (the path of every tier from 64K to 262K).
The attention over the earlier tokens grows with the context (B70, IQ3_XXS: the prompt at 1,488 tok/s at 64K, 1,099 at 262K), so a loss in the attention or the K/V reads shows smaller at 64K: the run that decides (step 7) measures IQ3_XXS at 262K too.
The other files run at 32K: the products' shapes follow the chunk (from 32K a prompt is read in chunks of thousands of tokens, so each expert gets many rows), and the long contexts are what the engine is used at most.

```sh
o=bench/results/<date>-upstream-<version>
python .claude/skills/upstream-sync/scripts/sync_prompts.py <pack> <ids>               # the prompts, once
XE="..." UP="..." XE_PROFILE=... UP_PROFILE=... DATA=<data> IDS=<ids> \
  [CONFIG=xestrata-<model>.json | PACK=... NATIVE=... PLE=...] [EXTRA="..."] [TIERS="..."] \
  python .claude/skills/upstream-sync/scripts/bench_generate.py $o/<file> <GPU>    # per file and GPU: B70, RTX4070, RX9060XT
python .claude/skills/upstream-sync/scripts/sync_collect.py $o/<file> xe-<commit> <tag>
```

`bench_generate.py` alternates the engines, two rounds per tier, after a warm-up each.
A model other than IQ3_XXS comes from a model config setup wrote (`CONFIG`: its files, its draft layer and the settings setup and the calibration chose, such as the RAM budget) or from its files (`PACK`, `NATIVE`, `PLE`); the same variables work for the serve and everyday scripts.
The three scripts are entry points to `tools/bench_core.py`, which setup's `--bench` runs too: the prompts, the conversations and how the runs are made and read are defined there once.
Both engines write one text a tier (the user's decision of 2026-10-10): upstream runs first in the tier and writes its own answer, and every later run of both engines follows it with `--spec-follow` (upstream's own tool, ac131a3a).
The rounds and the drafts accepted are then the same, so the output speeds compare the engines and not which text each happened to write: the two engines round differently, their answers part after a few dozen tokens, and on UD-Q4_K_XL that alone moved the output by 5%.
It stops upstream mid-run once its progress shows it at 10 tok/s or less (the prompt, from its `PP` lines after 30 s; the output, when 256 tokens at 10 tok/s and a minute have passed since the prompt) and does not run it again in that tier.

### The serve path

`strata --serve`, what users run, has a decode loop of its own (`src/program/generate.cpp`: the `--serve` block, apart from generate's speculative loop) and no `--spec-follow`, so bench_generate.py does not cover it.
`bench_serve.py` runs both engines with `--serve` and the options setup writes, over stdin (`GEN`, `DONE`), for IQ3_XXS at 64K, 32K and 1K (the three kinds of prompt above), per GPU, in the same folder as the IQ3_XXS table:

```sh
XE="..." UP="..." XE_PROFILE=... UP_PROFILE=... DATA=<data> IDS=<ids> \
  python .claude/skills/upstream-sync/scripts/bench_serve.py $o/IQ3_XXS <GPU>
python .claude/skills/upstream-sync/scripts/sync_collect.py $o/IQ3_XXS xe-<commit> <tag>
```

Each run is one chat of four turns: the tier's prompt, then the three follow-ups of `sync_prompts.py`, each sent with the answer before it, as a chat client sends them; 256 tokens a turn.
It measures the prompt (the first turn), the follow-ups (the median time a follow-up's prompt takes: the wait before its answer, which the conversation checkpoints decide) and the output (all four turns), greedy and at temperature 0.7, three rounds each, the engines alternating.
Each engine writes its own text, so the drafts it accepts count in its output speed, as they do for a user.
Another temperature: `MODES` (`t10` is temperature 1.0; everyday use's is `TEMPERATURE`); the merge's check keeps 0.7, the temperature the limits were set at.
The wrapper commands in `XE` and `UP` must pass stdin through (`docker run -i` for upstream's `sycl/` image).

### Everyday use

`bench_usage.py` runs each engine behind its own `serve/server.py` (the engine's tree in `XE_ROOT` and `UP_ROOT`), with the options setup writes for a 262K context (the longest; `CONTEXT` sets another), and sends what a user's client sends: streamed requests to `/v1/chat/completions`.
Each run starts the server anew and holds three conversations, at most 1,024 tokens a turn, sampled as Qwen recommends:

| Conversation | What |
| --- | --- |
| `chat-ja`, `chat-en` | a short system prompt and five short questions in a row, Japanese and English |
| `agent` | a coding agent's task with two tools; each step's tool result is a fixed source file (about 4K tokens), so the context grows step by step; seven steps |

It measures per conversation the time to the first streamed token, the prompt and the output speed (the server's `timings`), three rounds per engine, alternating:

```sh
XE="..." UP="..." XE_ROOT=<xe tree> UP_ROOT=<up tree> XE_PROFILE=... UP_PROFILE=... DATA=<data> PY=<python> \
  python .claude/skills/upstream-sync/scripts/bench_usage.py $o/IQ3_XXS <GPU>
```

`XE` and `UP` are a binary or a wrapper, after `env K=V ...` if the engine needs variables (they go to its server's environment).
`PY` is a Python with the server's packages (the data folder's `venv`).

## 7. The check

```sh
python .claude/skills/upstream-sync/scripts/judge.py --prompt-only B70 $o/IQ3_XXS $o/Q2_0 $o/IQ2_XS $o/IQ3_S $o/UD-Q4_K_XL
```

A cell (file, GPU, tier, prompt or output) passes when XeStrata's median is at most 1% below upstream's.
The serve cells (GPU, tier, greedy or t07; prompt, follow-ups, output), read from `matrix-serve.json`, and the everyday cells (GPU, conversation; first token, prompt, output), read from `matrix-usage.json`, pass at most `SERVE_LIMIT` (`judge.py`, 5%) below upstream's, the prompt speed at 1%: each engine writes its own text there, so the other measures vary with the text from round to round (the rounds' seeds differ), up to 10% off a cell's median at temperature 0.7 on 2026-10-10. Running XeStrata against itself shows nothing: with the same seed both write the same text at the same speed.
A cell upstream does not finish, finishes at 10 tok/s or less, or is stopped as that slow, is left out; one only XeStrata does not finish fails.
A file whose experts do not fit in a machine's RAM budget, so that they are read from the SSD while it answers (SSD offload: UD-Q4_K_XL on the AMD container's 62 GB), is not run on that machine (the user's decision of 2026-10-09): as for GPUs there is no machine for (step 4), its paths are checked by the tests and the builds only, and the record lists its cells `unverified`.
The B70's output cells are left out (`--prompt-only B70`, the user's decision of 2026-10-09): upstream's `sycl/` port reads the experts that do not fit in VRAM over PCIe from its kernels, a few tok/s, which says nothing about XeStrata's speed.

For each failing cell: find where the time goes (the GPU profilers of docs/DEVTOOLS.md, the engine's timing lines), read how upstream does that part, make XeStrata's faster (step 4's rules), and measure that cell again.
The run is done only when a full run of step 6 on the final HEAD, every file and tier of the table (IQ3_XXS with 262K), the serve path, everyday use and every reachable GPU, gives `ALL PASS`: a cell fixed earlier can lose again to a later change.

## 8. The record and the report

On `dev`:

- `$o/README.md`: the machines, both engines' commits and builds, the method, the tables (`sync_collect.py`), the check's output, what the numbers say, and the GPUs skipped; `matrix.json` and the logs in `runs/` with this PC's paths written as `<repo>` and `<data>`. Step 4's measurements of kept and removed paths go into the same record.
- docs/XE.ja.md, then docs/XE.md: the upstream version and range brought in, what was not taken and why, linking the record.

Lint and commit. Report: the tag, what was taken and left out by group, the check's table, the GPUs skipped (`unverified`), the paths ported for GPUs there is no machine for, and anything that needs the user.
