# Strata 0.2.4: DeepSeek V4.1-Flash in the Terminal

Strata now runs DeepSeek V4.1-Flash on a PC with one 12 GB graphics card and 46 to 64 GB of RAM. The engine keeps the experts it uses most in RAM and streams the rest from your SSD.

**DeepSeek is in the Terminal (`strata launch`, `strata setup`) in this release. The desktop app is unchanged (it stays 0.2.3) and gets DeepSeek in a later update.**

## What you get

- **DeepSeek V4.1-Flash** in the model list of Setup, in two sizes from IST-DAS:

  | set | download | quality against the original model |
  |---|---|---|
  | 2.00 bits | 257 GB | perplexity +16%; code and agent text +8.5% and +5.9% |
  | 2.25 bits | 274 GB | perplexity +7.7% to +8.3%; code and agent text +3.5% and +2.5% |

  "Perplexity" is how surprised the model is by a fixed set of 4,995 test texts; lower is closer to the original. Pick 2.25 for coding and agent work when you have the disk. Setup shows both sizes and warns if your PC is short of disk or RAM. It does not stop you.
- Setup downloads the files, builds the model folder and starts the engine. You set no flags.

## Speed on real PCs

Output is the speed of the answer. Prompt is how fast Strata reads what you send it.

| PC | output | prompt |
|---|---|---|
| RTX 5070 12 GB, 64 GB RAM | about 5 tokens/s (code) | 231 tokens/s at 8K |
| RTX 3060 12 GB, 46 GB RAM | about 6 to 9 tokens/s (6.1-6.7 without DSpark, 8.2-8.7 with) | about 103 tokens/s at 8K |
| 4x Radeon AI PRO R9700 (AMD) | about 31 tokens/s | about 360 tokens/s at 8K, 557 at 32K |

These are the 2.00-bit set. The 2.25-bit set reads 13% more data per token, so expect it to be a little slower. Your drive matters: the 12 GB cards spend most of each token waiting on the SSD and on the CPU, and a slow NVMe drive lowers every number above.

## How it works

- **A RAM expert store that sizes itself.** The model is a mixture of experts: each token uses a few of thousands of small blocks. Strata keeps the busiest ones in RAM, takes whatever RAM is free, and reads the rest from the SSD with direct reads. On the RTX 5070 PC with 64 GB this gave 4.3 to 4.9 tokens/s where memory-mapping the files gave 1.8 to 2.2.
- **Hot-expert profiles ship with the engine,** so the first prompts do not start cold.
- **Faster prompts.** The engine now reads long prompts layer by layer in big chunks instead of 16 tokens at a time. A 200-token prompt went from 3.6 to 6.4 tokens/s on the RTX 5070 PC, and 8K prompts reach 231 tokens/s.
- **DSpark** (a small draft model from Unsloth) is on where it measured faster and off on CPU-bound PCs. On the RTX 3060 PC (20 threads) it raised decode from 6.1-6.7 to 8.2-8.7 tokens/s. On the 4x R9700 box it raised 31 tokens/s to 55 to 69. On the RTX 5070 PC with a 6-core Ryzen 5 7600 it gave no gain at any setting (code 4.64 tokens/s with it off against 4.33 at best with it on; agent 3.41 against 2.86 to 3.21), because checking the drafts costs the CPU about what the saved tokens are worth. There `--dspark auto` does not load the drafter.
- **Faster CPU rows.** The engine has new exact rows for the experts the CPU computes: AVX-512 on Zen 4 and Zen 5, AVX2 on other CPUs, chosen automatically. On the Ryzen 5 7600 PC with an RTX 5070 (2.00 set, DSpark on, three alternating pairs, identical text in all six runs), decode went from 4.08 to 5.11 tokens/s (+25%) and prompt reading rose 26%. This was measured on one CPU, a Zen 4. Zen 5 gained more in the kernel tests.
- **Cards and engines:** tested on an RTX 3060 (Linux, full model), an RTX 5070 (Windows) and the Radeon AI PRO R9700 (gfx1201, Linux). The RTX 30 kernels (sm_86) were also checked on an RTX A4000. Windows on RTX 30 is built but was not run here. RX 9070 (gfx1200) is built but untested. The engine packages carry `strata-dsv41` next to `strata`.

## Needs

- Disk: 257 GB (2.00) or 274 GB (2.25). The download includes the Engram table and the DSpark drafter.
- RAM: 46 GB worked on the RTX 3060; 64 GB is the tested comfortable size. More RAM means a bigger store and faster answers.
- Graphics card: 12 GB or more.

## Known limits

- Prompt speed on 12 GB cards depends on the drive. A slow NVMe drive holds the prompt below the numbers above.
- DSpark is off on PCs where the CPU is the limit.
- Not in this release: Mac and Strix Halo. DeepSeek V4.1-Flash does not run there yet.
- Output on 12 GB cards is 4 to 8 tokens/s. That is usable for chat and slow for long agent runs.
- The answers can differ from run to run with the same prompt when experts move between the card and the CPU. The cause is which experts sit on the card at that moment.

## Updating

- **From 0.2.3 or 0.2.3.1:** run `strata update` in the Terminal. It fetches Control 0.1.49 and, on Windows NVIDIA, Linux NVIDIA and Linux AMD (Radeon AI PRO R9700 and RX 9070) PCs, the engine package that carries DeepSeek. Other engine packages stay at 0.1.48. The app is not touched.
- **Mac:** the Mac Terminal and engine stay on 0.1.48. DeepSeek is not on the Mac yet.
- **New installs:** the install command (`install.ps1` on Windows, `install.sh` on Linux) installs this version.

## Downloads

Everything from 0.2.3 is attached again, so the release page still holds every file. New in this release: `strata-control-0.1.49-windows-x86_64.zip`, `strata-control-0.1.49-linux-x86_64.tar.gz` and the three engine packages `strata-engine-0.1.49-windows-x86_64-cuda-cuda13`, `strata-engine-0.1.49-linux-x86_64-cuda-cuda13` and `strata-engine-0.1.49-linux-x86_64-hip-rx9070`. The installers, AppImages, dmg and the other engine packages are the 0.2.3 files.

## Thanks

- **IST-DAS** for the 2.00 and 2.25-bit sets of DeepSeek V4.1-Flash.
- **Unsloth** for the DSpark drafter GGUF.

---

Strata is free and open source. If it runs well on your PC, a coffee keeps the work on it going:

<a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a>
