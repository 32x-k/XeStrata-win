#!/bin/bash
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
# .claude/skills/release/scripts/up-sycl.sh - upstream's sycl/ port as an engine command (UP for bench.sh and the
# serve and everyday scripts): its strata, built in its strata-sycl-dev image (its docs/INTEL.md), run in that image
# with the arguments as they are.  stdin passes through (serve).
#
#   UP_TREE=DIR DATA=DIR [IDS=DIR] [UP_SELECTOR=level_zero:N] up-sycl.sh ARGS...
#
# UP_TREE: upstream's worktree (its build-sycl-aot/strata, its data/).  DATA and IDS are mounted read-only at the same
# paths, so the arguments need no rewriting; so is the folder of any other absolute path among the arguments (writable).
# UP_SELECTOR (default level_zero:0): the Level Zero device to take.
set -u
: "${UP_TREE:?}" "${DATA:?}"
mounts=(-v "$UP_TREE:$UP_TREE:ro" -v "$DATA:$DATA:ro")
[ -n "${IDS:-}" ] && mounts+=(-v "$IDS:$IDS:ro")
# the folder of a file argument outside those (--spec-follow's text, a --dump-logits output), at the same path
for a in "$@"; do
  case $a in /*) d=$(dirname "$a"); [ -d "$d" ] && case $d in "$UP_TREE"*|"$DATA"*|"${IDS:-/nonexistent}"*) ;; *) mounts+=(-v "$d:$d") ;; esac ;; esac
done
# The container ends with this script however it ends: a run stopped with SIGKILL (a bench stopping a slow upstream,
# a server ending its engine) took only the docker client, and the engine ran on in its container at 100% of a core
# with the card's memory held (2026-10-12, two hours).  A watcher with its own stdin and stdout (the caller reads this
# script's output to its end) kills the container by name once this script's process is gone.
name="up-sycl-$$-$RANDOM"
me=$$
(while kill -0 "$me" 2>/dev/null; do sleep 2; done; docker kill "$name" >/dev/null 2>&1) </dev/null >/dev/null 2>&1 &
docker run --rm -i --name "$name" -u "$(id -u):$(id -g)" --device /dev/dri \
  --group-add "$(getent group render | cut -d: -f3)" \
  -e ONEAPI_DEVICE_SELECTOR="${UP_SELECTOR:-level_zero:0}" "${mounts[@]}" strata-sycl-dev \
  "exec $UP_TREE/build-sycl-aot/strata $(printf '%q ' "$@")"
