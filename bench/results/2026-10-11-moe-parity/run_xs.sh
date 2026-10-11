#!/bin/bash
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
# XeStrata's greedy decode of each prompt from oracle_greedy's ids: run_xs.sh DIR TAG [strata options]
# MODEL: the GGUF; STRATA: the strata binary; PROMPTS: the prompt names; the device by ONEAPI_DEVICE_SELECTOR
set -u
M=${MODEL:?}; S=${STRATA:?}; dir=$1; tag=$2; shift 2
cd "$(dirname "$0")/$dir" || exit 1
for p in ${PROMPTS:?}; do
    timeout 1800 "$S" --native "$M" --tokens-file "out/oracle-$p.prompt" --max-new 32 --max-context 1024 \
        --dump-logits "out/xs-$tag-$p.logits" "$@" > "out/xs-$tag-$p.tokens" 2> "out/xs-$tag-$p.log" || echo "$p failed rc=$?"
    python3 ../compare.py "out/oracle-$p" "out/xs-$tag-$p.tokens" "out/xs-$tag-$p.logits" > "out/cmp-$tag-$p.txt" 2>&1
    echo "$tag $p: $(grep -m1 'leading tokens agree' "out/cmp-$tag-$p.txt") | $(grep -m1 'strata plain' "out/xs-$tag-$p.log" | cut -c1-200)"
done
