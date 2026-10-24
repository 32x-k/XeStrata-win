#!/bin/bash
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
# .claude/skills/upstream-sync/scripts/up-pack.sh - upstream's engine on a pack of its own: the measuring scripts give
# both engines XeStrata's pack (PACK), and where upstream reads another format (Gyro's: upstream's native experts,
# XeStrata's xs2), this replaces that folder in the arguments with UP_PACK.  UP_REAL: the command it wraps (upstream's
# strata, or .claude/skills/release/scripts/up-sycl.sh).
#
#   UP="up-pack.sh" UP_REAL="CMD..." PACK=DIR UP_PACK=DIR bench_generate.py ...
set -u
a=()
for x in "$@"; do a+=("${x//"$PACK"/"$UP_PACK"}"); done
# shellcheck disable=SC2086  # UP_REAL is a command with its arguments
exec $UP_REAL "${a[@]}"
