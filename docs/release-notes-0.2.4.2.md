# Strata 0.2.4.2: DeepSeek launch and install fixes

Fixes for `strata launch` with DeepSeek and for the one-line installer. Only the Terminal's Control (0.1.49.2) and `install.sh` changed. The app stays 0.2.3 and every engine package stays as it was, so there is nothing new to download for them.

## What is fixed

- **The right DeepSeek engine runs when you have more than one Strata install.** Launch now picks the newest DeepSeek engine package of the install that is running, and of the install your model's settings came from, instead of an old path saved in the model's settings.
- **A plain message when GitHub cannot be reached.** DeepSeek setup checks for 10 seconds before it downloads anything and says in one line that GitHub is not reachable, instead of retrying for minutes.
- **DeepSeek no longer downloads the Qwen engine.** A DeepSeek launch does not fetch or test-start the Strata (Qwen) engine, the llama.cpp source or CUDA libraries it never uses. `strata launch --plan` no longer promises that download.
- **The disk check counts only what is still to download.** If most of the model files are already in your models folder (also under the first release's file names), the model list no longer says "tight: needs more disk space". The room for the prepared model is still counted.
- **Smaller wording fixes.** A setup message that ended with a full stop no longer gets a second one ("engine.."), and the second sentence of "The model could not be installed. The model does not answer sensibly after the preparation: ..." starts with a capital.
- **The install one-liners keep your terminal open on an error.** `install.sh` is now a function that returns an error instead of closing the shell it runs in, and the commands in the README and docs download it first: `curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh`.

## Updating

- **From 0.2.4 or 0.2.4.1:** run `strata update` in the Terminal. It fetches Control 0.1.49.2 and the next `strata` command runs it (`strata --version` says 0.2.4.2).
- **New installs:** use the command in the README (download `install.sh`, then run it).
- **Mac:** the Mac Terminal gets Control 0.1.49.2 too. DeepSeek is not on the Mac yet.

---

Strata is free and open source. If it runs well on your PC, a coffee keeps the work on it going:

<a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a>
