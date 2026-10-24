# Strata 0.2.4.2: DeepSeek launch and install fixes

Fixes for `strata launch` with DeepSeek and for the one-line installer. The Terminal's Control (0.1.49.2), `install.sh` and the three DeepSeek engine packages (0.1.49.1) changed. The app stays 0.2.3 and every Qwen engine package stays as it was, so there is nothing new to download for them.

## What is fixed

- **Faster DeepSeek on AMD Ryzen 3000 / Threadripper 3000 / EPYC 7002 CPUs.** The 2.00 and IQ1_M sets answered up to 3x slower there; the answers are unchanged. The DeepSeek engine is now 0.1.49.1 (`strata-dsv41-0.1.49.1-windows-x86_64-cuda-cuda13`, `-linux-x86_64-cuda-cuda13` and `-linux-x86_64-hip-rx9070`).
- **The right DeepSeek engine runs when you have more than one Strata install.** Launch now picks the newest DeepSeek engine package of the install that is running, and of the install your model's settings came from, instead of an old path saved in the model's settings.
- **A plain message when GitHub cannot be reached.** DeepSeek setup checks for 10 seconds before it downloads anything and says in one line that GitHub is not reachable, instead of retrying for minutes.
- **DeepSeek no longer downloads the Qwen engine.** A DeepSeek launch does not fetch or test-start the Strata (Qwen) engine, the llama.cpp source or CUDA libraries it never uses. `strata launch --plan` no longer promises that download.
- **The disk check counts only what is still to download.** If most of the model files are already in your models folder (also under the first release's file names), the model list no longer says "tight: needs more disk space". The room for the prepared model is still counted.
- **Smaller wording fixes.** A setup message that ended with a full stop no longer gets a second one ("engine.."), and the second sentence of "The model could not be installed. The model does not answer sensibly after the preparation: ..." starts with a capital.
- **The install one-liners keep your terminal open on an error.** `install.sh` is now a function that returns an error instead of closing the shell it runs in, and the commands in the README and docs download it first: `curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh`.

## Updating

- **DeepSeek engine:** `strata launch` fetches the DeepSeek engine package when none is installed. If you already have 0.1.49 installed and use one of the CPUs above, remove the `engine-dsv41` folder (or download the 0.1.49.1 package from this page and give it to `strata launch --dsv41-package PATH`) to get the faster one.

- **From 0.2.4 or 0.2.4.1:** run `strata update` in the Terminal. It fetches Control 0.1.49.2 and the next `strata` command runs it (`strata --version` says 0.2.4.2).
- **New installs:** use the command in the README (download `install.sh`, then run it).
- **Mac:** the Mac Terminal gets Control 0.1.49.2 too. DeepSeek is not on the Mac yet.

---

Strata is free and open source. If it runs well on your PC, a coffee keeps the work on it going:

<a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a>
