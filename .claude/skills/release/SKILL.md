---
name: release
description: Make a GitHub release of XeStrata in stages - the version, main fast-forwarded to dev, the tag (xe and the version, e.g. xe0.1.39); first the three contrib-llvm packages published with the notes (English then Japanese) and the older releases marked as old, then a speed table against upstream on Intel, NVIDIA and AMD GPUs. Only when the user asks for a release (/release).
disable-model-invocation: true
argument-hint: "[VERSION, e.g. 0.1.39 or 0.1.39.1]"
---

# Releasing XeStrata

A release is the annotated tag `xe<VERSION>` on `main`, pushed to origin, and a published GitHub release on it, made in two stages:

- the preliminary release (先行報): the three contrib-llvm packages, `SHA256SUMS` and the notes;
- the final release (確定報): the speed table against upstream on an Intel, an NVIDIA and an AMD GPU added to the notes.

Rules for the whole run:

- Invoking /release is the user's go-ahead for pushing `main` and the tag and for publishing and updating the release at each stage (the user's decision, 2026-10-09): do not stop to ask.
  Stop and report only when the scan of step 7 finds something not clearly public (then push nothing), a package build fails, or an engine of the speed table does not run.
- `gh` only as `gh release ...` (the one exception to "git commands only" in the user's global instructions); everything else with git.
  Every `gh release` names `--repo MistVVK/XeStrata`: without it gh takes the `upstream` remote (Niko1221/Strata).
  Never print or read gh's token (`gh auth status` shows the account; leave its token lines out).
- A release ships the contrib-llvm packages only (the user's decision, 2026-10-09). The free packages are treated like contrib-icpx: built only when the user asks (`build-packages.sh xe<VERSION> free`), and not attached otherwise.
- The check of the packages is their build alone (the user's decision, 2026-10-07): no CTest, no incus run, unless the user asks for them.
- Work and commits are on `dev` (Japanese commit messages, one logical change per commit, `git add` and `git commit` joined with `&&`). `dev` itself is never pushed; `main` is.
- A tag that has been pushed is never moved or deleted. A mistake after the push is fixed by a new version (the fourth number).

## 1. Check the starting point

```sh
git status --short --untracked-files=no   # must be empty, on dev
gh auth status 2>&1 | grep -v -i token    # logged in to github.com as MistVVK
docker info --format '{{.ServerVersion}}' # the package builds run in Docker
git fetch origin main --tags
git merge-base --is-ancestor main dev     # main must fast-forward to dev
git tag -l 'xe*' --sort=-v:refname | head -n 1; git ls-remote --tags origin 'xe*'
```

Stop and report if a tracked file has changes, gh is not logged in, Docker does not answer, or `main` has commits `dev` lacks.
Untracked files (a record not yet committed, for example) may stay: the packages are built from the tag, and the push takes only commits.
The newest `xe*` tag is the previous release (none before the first one).

## 2. The version

The version is `$ARGUMENTS` when given, else the one `CMakeLists.txt` makes from `STRATA_UPSTREAM_VERSION` and `XESTRATA_REVISION` (AGENTS.md, "Versions"); the tag is `xe<VERSION>` (the name the program shows, `STRATA_VERSION_NAME`).
It must not exist yet, locally or on origin.

- A release after integrating a new upstream version takes that version: the integration changes `STRATA_UPSTREAM_VERSION` and sets `XESTRATA_REVISION` back to 1 (`xe0.1.41`, the packages `0.1.41-1`).
- Another release on the same upstream version raises `XESTRATA_REVISION` by one, and the version name gets a number: `xe0.1.40.2` (revision 1) then `xe0.1.40.2.1` (packages `0.1.40.2-2`), `xe0.1.40.2.2`.

When the version changes, change `XESTRATA_REVISION` and every place that shows the version name (`git grep -n 'xe<OLD>'`: docs/XE), Japanese and English files in the same commit, and commit on `dev` (`版を xe<VERSION> にする`).
A new version makes conversations saved by `--conversation-save` under the old one unreadable (the engine version is part of their identity): the notes say so.

## 3. main and the tag

```sh
git fetch . dev:main                      # fast-forwards main to dev without checking it out
git tag -a xe<VERSION> -m "XeStrata xe<VERSION>" main
```

Both stay local until step 8.

## 4. The contrib-llvm packages

```sh
setsid nohup .claude/skills/release/scripts/build-packages.sh xe<VERSION> contrib >/dev/null 2>&1 </dev/null &
```

It builds the three contrib-llvm variants from the tag one after another (ubuntu26.04 cuda13.1 and cuda12.4, fedora44 cuda13.4), each in a clean container (about an hour and a half), and gathers them with `SHA256SUMS` in `build/release/xe<VERSION>/`.
Wait on the PID in `build/release/xe<VERSION>/build-contrib.pid` (`kill -0`) and follow `summary-contrib.txt` (a Monitor that prints its new lines, re-armed every 30 minutes); never wait on `pgrep -f`, which matches the waiting shell.
On `FAILED`, report the end of the variant's log in `logs/`; the fix is an ordinary change on `dev`, after which delete the local tag (`git tag -d xe<VERSION>`) and go back to step 3 (only while the tag has not been pushed).

Steps 5 and 6 are done while it builds.

## 5. The release notes

Write `build/release/xe<VERSION>/NOTES.md`: the English section first, then the same content in Japanese (the user's decision, 2026-10-12), under a first line linking the two: `[English](#english) | [日本語](#日本語)`.

- What changed for users since the previous release tag (`git log --no-merges <previous>..xe<VERSION>`; for the first release, since `origin/main`), grouped by what a user notices, not a list of commits.
- How to install: README's install section, as it is there (the user's decision, 2026-10-11): the four numbered steps (download from the table with `SHA256SUMS`, `sha256sum -c SHA256SUMS --ignore-missing`, `sudo apt install ./…deb` or `sudo dnf install ./…rpm` with the release's file names, `xestrata --setup`, which shows the GPU makers' packages that are missing and the command that installs them: AGENTS.md, "Packages", the notes do not list them), then the `> [!NOTE]` that without an NVIDIA GPU either CUDA version works the same, then the package table (Intel, AMD (RDNA2 or later), NVIDIA (Turing or later): `xestrata-contrib-cuda13.1`, `cuda13.4` on Fedora; Intel, AMD (RDNA2 or later), NVIDIA Pascal and Volta: `xestrata-contrib-cuda12.4`). A change to one goes into the other.
- When the version changed: saved conversations from another version are not read.
- That the release has no free packages: a free build is made from the source (docs/BUILD.md).

Shape the notes with the user's `i-have-adhd` skill (the user's decision, 2026-10-09): read its `SKILL.md` in the user's skills folder (`~/.claude/skills/i-have-adhd/` or `~/.codex/skills/i-have-adhd/`; it cannot be invoked as a tool) and follow its rules for the reader of the notes.
In short, for when it is not there:

- The first lines are what to do: which package to download for which GPU, and the install commands, as a numbered list.
- What changed, as what now works or is faster, in concrete terms (numbers where measured), ranked, at most five items a group.
- No preamble, recap or closing, no tangents; facts only, no plans or alternatives (the stage line excepted).

Each section starts with a line of the stage: 先行報: upstream との速さの比較は後で足す / Preliminary: the speed comparison with upstream follows.
Show the notes to the user.

## 6. The AMD engines

The AMD GPU is in another machine, so its part of the speed table (step 9) is prepared now and can run while the packages build.
Prepare the two engines in the AMD development container (where it is and how to run in it: `LOCAL.md` in the repository root, kept out of git; when it is missing, ask the user and write the answer there):

- XeStrata: bring the tag into the container's clone (it has no access to origin: a `git bundle` of `xe<VERSION>` copied in) and build the contrib-llvm mode with HIP for its GPU (`STRATA_HIP_ARCHS`, docs/BUILD.md) in a folder of the tag's own.
- upstream: its newest release tag (`git ls-remote --tags --refs upstream 'v*' | sed 's#.*refs/tags/##' | sort -V | tail -n 1`; not `git tag -l 'v*'`, which also lists other remotes' tags (Project Maya's `upstream-maya` has v1.0.x)), built with HIP as its own documents say.

Run step 9's script there as soon as both are built; the development machine's GPUs wait until the packages are built (step 8).

## 7. Scan what the push would publish

```sh
git fetch origin main
.claude/skills/release/scripts/prepush-scan.sh origin/main..main
```

It runs gitleaks over each commit's changes and looks in every added line for this PC's paths, user and host names, private IP addresses, e-mail addresses other than the public author address, and for recordings and files over 5 MB.
Judge each finding: third-party license texts, upstream's code and version numbers are public; a real secret or anything of this PC is not.
For anything not clearly public, stop and report it with the commit and file; do not push.

## 8. The preliminary release (先行報)

When `summary-contrib.txt` ends with `ALLDONE`: `build/release/xe<VERSION>/` holds the three contrib-llvm packages (two `.deb`, one `.rpm`) and `SHA256SUMS`, and `(cd build/release/xe<VERSION> && sha256sum -c SHA256SUMS)` passes.

```sh
git push origin main
git push origin xe<VERSION>
d=build/release/xe<VERSION>
gh release create xe<VERSION> --repo MistVVK/XeStrata --verify-tag --title "XeStrata xe<VERSION>" --notes-file "$d/NOTES.md" \
  "$d"/*.deb "$d"/*.rpm "$d/SHA256SUMS"
gh release view xe<VERSION> --repo MistVVK/XeStrata --json url,isDraft
```

Retry a push only for a network failure (up to four times, waiting 2, 4, 8, 16 s).
`--verify-tag` makes gh refuse rather than create a tag of its own.
Report the release's URL, then prepare the development machine's engines for step 9.
A wrong asset or note is fixed with `gh release upload --clobber`, `gh release delete-asset` or `gh release edit --notes-file`.

### The older releases

Once the new release is published (it is then the latest), mark every older one:

```sh
gh release list --repo MistVVK/XeStrata --limit 1000 --json tagName,isLatest -q '.[] | select(.isLatest | not) | .tagName' |
  while read -r t; do .claude/skills/release/scripts/mark-old.sh "$t"; done
```

`mark-old.sh` puts a warning at the top of a release's notes, in English and Japanese, that sends the reader to the latest release (`releases/latest`, so a note put in before stays right); a release that has it already is left as it is, so in practice only the previous release changes.

## 9. The speed table

The release against upstream's newest release tag, IQ3_XXS, on each maker's GPU: the Intel Arc Pro B70 and the RTX 4070 in the development machine, the RX 9060 XT in the AMD container (step 6).
The six tiers and the settings of the 2026-10-04-speed-matrix record (1K to 262K, what setup writes for each context, 256 greedy tokens), with one change (the user's decision of 2026-10-12): KV streaming (`--kv-resident`) only where setup writes it on the machine, from 64K when the RAM holds the KV cache beside the model, so a machine short of that RAM (the AMD container: `LOCAL.md`) does not compare at those lengths with the releases up to xe0.1.42; its `prompts.py`, now `.claude/skills/release/scripts/prompts.py`, writes the prompts (`python .claude/skills/release/scripts/prompts.py <pack> <ids folder>`).
The table's method stays fixed from release to release so the releases compare: the upstream merge's measurements (/upstream-sync) have scripts of their own, and a change to them does not change this table.

The engines, built from the tags (no package installed), each in a worktree of its own under `.claude/worktrees/`:

- XeStrata: the contrib-llvm build of `xe<VERSION>` (docs/BUILD.md, with `LD_LIBRARY_PATH` to intel/llvm's `lib`); one GPU at a time with `ONEAPI_DEVICE_SELECTOR` (`level_zero:*` for the B70, `cuda:*` for the RTX 4070).
- upstream on the B70: its `sycl/` port, built in its Docker image as its docs/INTEL.md says (`docker build -t strata-sycl-dev sycl/tools`, then its `build.sh` with `AOT=bmg-g31`), and run through `scripts/up-sycl.sh` (`UP=".claude/skills/release/scripts/up-sycl.sh"` with `UP_TREE`, `DATA` and `IDS` exported), which runs it as its setup configures an Intel card: the arguments get what its `sycl/setup_intel.py` writes (the xe cards' expert profile, the streamed experts, KV streaming from 64K, the VRAM reserve, the prompt chunks), and the engine the run-time switches of its `sycl/serve/strata-sycl.sh`.
- upstream on the RTX 4070: its CUDA build, as its documents say.

Each engine's own shipped expert profile (`data/expert-profile.bin` of its tree; upstream's `sycl/` port takes the one its `sycl/setup_intel.py` writes in its place, `data/expert-profile-sycl.bin` from v0.1.43).
No other work on a machine while it measures (no build, no other engine).

```sh
o=<data>/records/bench-results/<date>-release-xe<VERSION>   # outside git (the user's decision of 2026-10-11)
XE="..." UP="..." XE_PROFILE=... UP_PROFILE=... DATA=<data> IDS=<ids> .claude/skills/release/scripts/bench.sh $o B70   # <data>: LOCAL.md
python .claude/skills/release/scripts/bench-collect.py $o xe<VERSION> <upstream tag>
```

`bench.sh` runs a warm-up per engine, then each tier twice per engine with the engines alternating; `bench-collect.py` writes `matrix.json` and the tables (the medians), one pair per GPU.
A cell that does not fit the GPU's or the machine's memory stays `-` with the reason in the record; an engine that does not start at all stops the run (report it).
`bench.sh` stops upstream once it is plainly at 10 tok/s or less; such a cell is `too slow`: not measurable, written 計測不能（低速）/ not measurable (too slow) in the notes, with a line saying upstream was stopped at 10 tok/s or less.

### serve and everyday use

Beside that table, two smaller ones on the same GPUs and engines, closer to what a user meets, by the upstream merge's scripts (`tools/bench_core.py`), cut down to about 40 minutes a GPU (the AMD container measures meanwhile): `strata --serve` at 32K, sampled at temperature 0.7, and the engines' own `serve/server.py` with the Japanese chat and the coding agent; two rounds each, the engines alternating, each engine writing its own text.
They go into `$o/serve/`, apart from the table's `matrix.json`; the prompts come from the merge's `sync_prompts.py` (its 32K is the table's 32K, and it writes the follow-up questions).

```sh
u=.claude/skills/upstream-sync/scripts
python $u/sync_prompts.py <pack> <ids-serve>                                   # once
XE="..." UP="..." XE_PROFILE=... UP_PROFILE=... DATA=<data> IDS=<ids-serve> TIERS=32k MODES=t07 ROUNDS=2 \
  python $u/bench_serve.py $o/serve <GPU>
XE="..." UP="..." XE_ROOT=<xe tree> UP_ROOT=<up tree> XE_PROFILE=... UP_PROFILE=... DATA=<data> PY=<python> \
  ROUNDS=2 CONVS="chat-ja agent" python $u/bench_usage.py $o/serve <GPU>
python $u/sync_collect.py $o/serve xe<VERSION> <upstream tag>                 # matrix-serve.json, matrix-usage.json
```

`XE` and `UP` are as above; serve and everyday use pass stdin to them, so upstream's `sycl/` wrapper runs `docker run -i`.
`XE_ROOT` and `UP_ROOT` are the engines' worktrees (their `serve/server.py`), `PY` a Python with the server's packages (the data folder's `venv`); everyday use runs at a 262144 context with what setup writes for it.

The record goes into `$o/`, in the data folder and never into git (XeStrata's own measurements are not shared, the user's decision of 2026-10-11): a README (the machines, the two engines' commits and builds, the method, the tables and what the numbers say), `matrix.json`, `serve/` (its `matrix-serve.json`, `matrix-usage.json` and runs), and the logs in `runs/` with this PC's paths written as `<repo>` and `<data>` (as the speed-matrix record does).
It is not committed; the release notes carry the tables.

## 10. The final release (確定報)

The final release is reached a GPU at a time (the user's decision, 2026-10-11): as soon as one GPU's tables are complete (its generate table and its serve and everyday table both), add them to the notes (English and Japanese sections) and update the release (`gh release edit`, below), without waiting for the other GPUs.

- The comparison section comes after the notes' other sections: its opening lines (the engines' versions, the model, the method), then the note on temperatures and what counts as equal (below) right after them, once, then the GPUs in a fixed order, the B70, the RTX 4070, the RX 9060 XT, each as its tables are done.
- While some are missing, one line under the done ones names them: B70・RTX 4070: 計測中（後で足す） / B70, RTX 4070: being measured (added later).
- The first update replaces the preliminary stage line with that section; when the last GPU is in, the line of the missing GPUs goes too.
- Report the release's URL and the GPU's tables at each update.

The tables:

The notes' table is one per GPU, from `bench-collect.py`'s medians (the record's README keeps `bench-collect.py`'s own tables): a row per tier, named by the model's weight type and the length (`IQ3_XXS 1K`), and for prefill and then decode three columns (the LLM terms, in both languages; the user's decision, 2026-10-11): XeStrata's value, upstream's, and the difference (the value columns named with the engine and its version, `XeStrata xe0.1.40.2.1`, `upstream v0.1.41`), marked ✅ where XeStrata is faster and ❌ where it is slower: prefill as a ratio (`✅ 2.1 倍`, `✅ ×2.1`), decode as a percentage (`✅ +14%`; slower: `❌ −3%`).
Within ±1% (either way) the two count as equal: `≒ 同等（−0.1%）` / `≈ equal (−0.1%)`, for prefill as well.
A value upstream was stopped in is 計測不能（低速） / not measurable (too slow), its difference `✅`, with the line saying upstream was stopped at 10 tok/s or less; a value that does not fit stays `-`.
Above each GPU's tables: the GPU (and its memory) and which build of upstream it ran (`sycl/`, CUDA, HIP).
Each table has a heading line with its temperature: **generate**（温度 0、トークン/秒） / **generate** (temperature 0, tokens/s), and **serve と通常使用**（温度 0.7） / **serve and everyday use** (temperature 0.7).

```markdown
**AMD Radeon RX 9060 XT (16 GB)**: upstream v0.1.41 は HIP 版

**generate**（温度 0、トークン/秒）

| 形式・長さ | prefill XeStrata xe0.1.40.2.1 | prefill upstream v0.1.41 | 差 | decode XeStrata xe0.1.40.2.1 | decode upstream v0.1.41 | 差 |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| IQ3_XXS 1K | 509 | 295 | ✅ 1.7 倍 | 48.8 | 46.8 | ✅ +4% |
```

Below each GPU's table, its serve and everyday table from `serve/`'s medians, with the same columns (XeStrata, upstream, difference): a row per measure, prefill as a ratio, the rest as percentages, the wait and the first token better when shorter.
Equal there is the upstream merge's check (`judge.py`): prefill within ±1%, the rest within ±5%, both engines writing their own texts. A wait or a first token is judged as a speed, upstream's time over XeStrata's (372 ms against 354 ms: −4.8%, equal), and its column shows the time's own difference: `≒ 同等（+5%）` / `≈ equal (+5%)`.

```markdown
**serve と通常使用**（温度 0.7）

| 計測 | XeStrata xe0.1.40.2.1 | upstream v0.1.41 | 差 |
| --- | ---: | ---: | --- |
| serve 32K 最初の prefill (tok/s) | 839 | 380 | ✅ 2.2 倍 |
| serve 32K 会話の続きの待ち (ms) | 372 | 354 | ≒ 同等（+5%） |
| serve 32K decode (tok/s) | 45.3 | 40.3 | ✅ +12% |
| chat-ja 最初のトークン (s) | 0.68 | 1.05 | ✅ −35% |
| chat-ja decode (tok/s) | 42.8 | 41.0 | ≒ 同等（+4%） |
| agent 最初のトークン (s) | 5.74 | 11.97 | ✅ −52% |
| agent prefill (tok/s) | 691 | 334 | ✅ 2.1 倍 |
| agent decode (tok/s) | 46.2 | 45.0 | ≒ 同等（+3%） |
```

Right after the section's opening lines, once (not repeated for each GPU), one note in each language section, with the temperatures and what counts as equal:

```markdown
> [!NOTE]
> - **generate**：温度 0（greedy）。差が ±1% 以内なら ≒ 同等です。
> - **serve と通常使用**：温度 0.7 で、それぞれのエンジンが自分の文を書くので、1 回ごとに中央値から 10% 前後ずれることがあります。差が prefill は ±1% 以内、待ち・decode・最初のトークンは ±5% 以内なら ≒ 同等です。
> - どちらも、交互に 2 回ずつ測った中央値です。
```

```markdown
> [!NOTE]
> - **generate**: temperature 0 (greedy). Within ±1% counts as ≈ equal.
> - **serve and everyday use**: temperature 0.7, each engine writing its own text, so a run can land about 10% off the median. Within ±1% for prefill and ±5% for the wait, decode and first token counts as ≈ equal.
> - Both are the median of two alternating runs.
```

```sh
gh release edit xe<VERSION> --repo MistVVK/XeStrata --notes-file build/release/xe<VERSION>/NOTES.md
```
