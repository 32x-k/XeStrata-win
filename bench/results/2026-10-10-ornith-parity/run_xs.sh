#!/bin/bash
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
# XeStrata's greedy decode of each prompt, from the token ids oracle_greedy wrote: run_xs.sh TAG [strata options].
# STRATA: the strata binary; MODEL: the Q6_K GGUF; STRATA_VRAM_LIMIT_MIB caps the VRAM the engine sees.
set -u
M=${MODEL:?set MODEL to Ornith-1.5-35B-Q6_K.gguf}
S=${STRATA:?set STRATA to the strata binary}
tag=$1
shift
for p in ${PROMPTS:-en ja code math story}; do
    timeout 1800 "$S" --native "$M" --tokens-file "out/oracle-$p.prompt" --n 32 --max-context 512 \
        --dump-logits "out/xs-$tag-$p.logits" "$@" > "out/xs-$tag-$p.tokens" 2> "out/xs-$tag-$p.log" || echo "$p failed"
done
