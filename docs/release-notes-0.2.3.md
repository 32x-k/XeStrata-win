# 0.2.3

## What is new

- **Strix Halo decodes about 9% faster.** The hyper-connection weights are read as 8-bit (Q8_0) on gfx1151 by default: 5 interleaved pairs on IQ3_S gave +7.9% to +10.7% on JSON, code, tool calls, chat and an agent transcript. Answers are not bit-identical: they can differ in near-tie tokens, about as much as the engine's own CPU-versus-GPU expert rounding does (the numbers are in docs/AMD_HIP.md). `STRATA_HC_Q8=0` restores the old read.

## Updating the Terminal from 0.2.0-beta or 0.2.1

- **Run the install command once more.** The Terminal that came with 0.2.0-beta, 0.2.1, 0.2.1.1 and 0.2.1.2 cannot switch to a newer Control by itself, so `strata update` reported "updated" and kept running the old one. From 0.2.3 on, that update says so instead: "This Terminal cannot switch to the new Control by itself. Run the Strata install command once more to update it." The install command keeps your models and settings. A Terminal from 0.2.1.3 or newer updates with `strata update`.
- **Version numbers.** The app is 0.2.3, Control (the Terminal's tools) is 0.1.48 and the engine is 0.1.48 (the engine source changed since 0.2.2, so the engine packages are rebuilt).
