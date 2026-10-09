#!/bin/bash
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
# llama.cpp's greedy decode of each prompt: oracle_greedy on the CPU backend (the reference) and oracle_greedy_hip
# with every layer on the GPU and the routed experts in host memory (the noise floor).  MODEL: the Q6_K GGUF.
set -u
M=${MODEL:?set MODEL to Ornith-1.5-35B-Q6_K.gguf}
mkdir -p out
for p in ${PROMPTS:-en ja code math story}; do
    ./oracle_greedy "$M" "prompts/$p.txt" 32 "out/oracle-$p" 14 > "out/oracle-$p.log" 2>&1 || echo "$p: cpu failed"
    ./oracle_greedy_hip "$M" "prompts/$p.txt" 32 "out/hipall-$p" 14 99 1 > "out/hipall-$p.log" 2>&1 || echo "$p: rocm failed"
done
