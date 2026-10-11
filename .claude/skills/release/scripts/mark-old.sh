#!/bin/sh
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
# .claude/skills/release/scripts/mark-old.sh - put a note at the top of an older release's notes that sends the
# reader to the latest release.
#
#   .claude/skills/release/scripts/mark-old.sh xe0.1.40.2.2
#
# The note links to releases/latest, not to a version, so it never needs rewriting; a release that has it already
# is left as it is.
set -eu

[ $# -eq 1 ] || { echo "usage: $0 TAG (e.g. xe0.1.40.2.2)" >&2; exit 2; }
tag=$1
repo=MistVVK/XeStrata
latest=https://github.com/$repo/releases/latest

body=$(mktemp)
trap 'rm -f "$body"' EXIT
gh release view "$tag" --repo "$repo" --json body -q .body >"$body"
if grep -qF "$latest" "$body"; then
    echo "$tag: already marked"
    exit 0
fi
{
    echo '> [!WARNING]'
    echo "> This is an old release. Get the latest from [the latest release]($latest)."
    echo "> 古いリリースです。最新版は[最新のリリース]($latest)から入れてください。"
    echo
    cat "$body"
} >"$body.new"
mv "$body.new" "$body"
gh release edit "$tag" --repo "$repo" --notes-file "$body" >/dev/null
echo "$tag: marked"
