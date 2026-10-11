# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Which of upstream's changes to the shared, non-kernel files XeStrata does not have: for each upstream commit up to
TAG, the lines it added that upstream still has at TAG, and how many of them XeStrata's REF has in the same file.
A commit with few of its lines in XeStrata is a candidate for a change that was never ported (or that XeStrata does
another way); each is read before it is taken.

    python .claude/skills/upstream-sync/scripts/carryover.py TAG [--ref dev] [--since REV] [--paths P ...]
        [--max 0.5] [--min-lines 3] [--ledger .upstream-<version>.md ...]

Lines compared: stripped, comments and lines under 12 characters left out (XeStrata rewrites comments and names its
own upstream issue numbers in them).  --since: only commits after REV (default: all of upstream's history up to TAG,
so changes an earlier run missed are found too).  --ledger: commits whose hash a ledger already names are left out.
A commit with fewer than --min-lines surviving lines is not judged: a one- or two-line fix is not found this way.
"""
import argparse
import re
import subprocess
import sys

PATHS = ["setup.py", "chat.py", "update.sh", "serve", "tools"]
COMMENT = re.compile(r"^(#|//|/\*|\*|<!--)")


def git(*a: str) -> str:
    return subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout


def key(line: str) -> str | None:
    """The line as compared: stripped, without a trailing Python comment; None for a line not worth comparing."""
    s = line.strip()
    if COMMENT.match(s):
        return None
    s = re.sub(r"\s+#\s.*$", "", s).strip()
    return s if len(s) >= 12 else None


def lines_at(rev: str, path: str, cache: dict) -> set:
    if (rev, path) not in cache:
        try:
            text = git("show", f"{rev}:{path}")
        except subprocess.CalledProcessError:
            text = ""
        cache[rev, path] = {k for k in map(key, text.splitlines()) if k}
    return cache[rev, path]


def added(commit: str, paths: list) -> dict:
    """path -> the keys of the lines `commit` added there."""
    out: dict = {}
    path = None
    for line in git("show", "--format=", "--unified=0", "--no-renames", commit, "--", *paths).splitlines():
        if line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") else None
        elif line.startswith("+") and path and (k := key(line[1:])):
            out.setdefault(path, set()).add(k)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tag")
    ap.add_argument("--ref", default="HEAD", help="XeStrata's tree to look in (default HEAD)")
    ap.add_argument("--since", help="only upstream's commits after this one")
    ap.add_argument("--paths", nargs="+", default=PATHS)
    ap.add_argument("--max", type=float, default=0.5, help="list commits with at most this share in XeStrata")
    ap.add_argument("--min-lines", type=int, default=3)
    ap.add_argument("--ledger", nargs="*", default=[])
    a = ap.parse_args()
    known = set()
    for f in a.ledger:
        known |= set(re.findall(r"\b[0-9a-f]{8,9}\b", open(f, encoding="utf-8").read()))
    rng = f"{a.since}..{a.tag}" if a.since else a.tag
    commits = git("log", "--no-merges", "--reverse", "--format=%H %ad %s", "--date=short", rng, "--",
                  *a.paths).splitlines()
    cache: dict = {}
    found = 0
    for row in commits:
        h, date, subject = row.split(" ", 2)
        if h[:8] in known or h[:9] in known:
            continue
        have = alive = 0
        files = []
        for path, keys in added(h, a.paths).items():
            live = keys & lines_at(a.tag, path, cache)
            if not live:
                continue
            ours = live & lines_at(a.ref, path, cache)
            alive += len(live)
            have += len(ours)
            files.append(f"{path} {len(ours)}/{len(live)}")
        if alive < a.min_lines or have / alive > a.max:
            continue
        found += 1
        print(f"{have / alive:4.0%} {alive:4d}  {h[:9]} {date} {subject[:100]}")
        print(f"            {'; '.join(files)}")
    print(f"{found} of {len(commits)} commits with at most {a.max:.0%} of their surviving lines in {a.ref}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
