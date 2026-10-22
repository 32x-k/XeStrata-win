# 0.2.3

## What is new

- **Strix Halo decodes about 9% faster.** The hyper-connection weights are read as 8-bit (Q8_0) on gfx1151 by default: 5 interleaved pairs on IQ3_S gave +7.9% to +10.7% on JSON, code, tool calls, chat and an agent transcript. Answers are not bit-identical: they can differ in near-tie tokens, about as much as the engine's own CPU-versus-GPU expert rounding does (the numbers are in docs/AMD_HIP.md). `STRATA_HC_Q8=0` restores the old read.
