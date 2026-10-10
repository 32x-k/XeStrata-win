#!/bin/bash
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
# Ornith-1.5 Q6_K, prompt.txt (make_prompt.py, 515 tokens), 256 tokens generated greedily: llama.cpp (ROCm) in several expert placements,
# then XeStrata.  MODEL: the GGUF; LLAMA: llama-completion; TOKENIZE: llama-tokenize; STRATA: the strata binary;
# PROFILE_PROMPT: the token ids the profile is learned from (another text).
set -u
M=${MODEL:?}
common=(-m "$M" -f prompt.txt -n 256 --temp 0 --ignore-eos -no-cnv -c 2048 -t 8)
run_l() {
    name=$1
    shift
    timeout 1200 "${LLAMA:?}" "${common[@]}" "$@" > "out-$name.txt" 2> "log-$name.txt" < /dev/null
    echo "$name rc=$? $(grep -E 'prompt eval time|       eval time' "log-$name.txt" | sed -E 's/.*= *//' | tr '\n' ' ')"
}
run_l cpumoe -ngl 99 -ncmoe 99
run_l cpumoe-cache12g -ngl 99 -ncmoe 99 --moe-cache-mib 12000
for n in 26 22 18 14; do run_l "ncmoe$n" -ngl 99 -ncmoe "$n"; done
run_l fit -fit on
"${TOKENIZE:?}" -m "$M" -f prompt.txt --ids --log-disable 2> /dev/null | tail -1 | tr -d '[] ' > prompt.ids
S=${STRATA:?}
timeout 1800 "$S" --native "$M" --tokens-file "${PROFILE_PROMPT:?}" --n 512 --max-context 2048 \
    --expert-profile-save profile.bin > out-profile.txt 2> log-profile.txt
for i in 1 2; do
    timeout 1200 "$S" --native "$M" --tokens-file prompt.ids --n 256 --max-context 2048 > "out-xs-$i.txt" 2> "log-xs-$i.txt"
    echo "xs $i rc=$? $(grep -o 'prompt.*' "log-xs-$i.txt")"
    timeout 1200 "$S" --native "$M" --tokens-file prompt.ids --n 256 --max-context 2048 --expert-profile profile.bin \
        > "out-xsp-$i.txt" 2> "log-xsp-$i.txt"
    echo "xs+profile $i rc=$? $(grep -o 'prompt.*' "log-xsp-$i.txt")"
done
