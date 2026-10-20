<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->

# Using XeStrata on Windows

How to use the zip for AMD and Intel GPUs on Windows (`XeStrata-<version>-windows-x64-amd-intel.zip`).
Installing, starting and changing the settings all go through `xestrata-cli.bat` in the zip.
`xestrata-cli.bat` passes its options to setup as they are, so they are the same as Linux's `xestrata` (`xestrata-cli.bat --help`).

## Installing

1. Unpack the zip into a short folder (`C:\XeStrata`, for example). The paths inside are long, and unpacking into a deep folder can fail.
1. Run `xestrata-cli.bat` (double-clicking works too). It goes through choosing the GPU and the model, the download and preparing the model on the console, then starts it.

You need the GPU driver (AMD: AMD Software; Intel: Intel's graphics driver) and the [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe).
Without the Visual C++ Redistributable, `xestrata-cli.bat` says so and stops. .NET is not needed.
The disk needs room for the model files and the pack. If there is not enough, setup says so and stops.

## Starting and stopping

- Later, run the same `xestrata-cli.bat`. The installed model starts right away.
- Once it starts, the browser opens `http://127.0.0.1:8095`. Apps connect to `http://127.0.0.1:8095/v1` as an OpenAI-compatible endpoint.
- To stop it, close the console window the model runs in.
- `xestrata-cli.bat --gpu 1` runs this start on that GPU only. `xestrata-cli.bat --check` lists the GPUs' numbers.

## Changing the settings

To change the settings, run `xestrata-cli.bat --setup`. It goes through the questions of the install again.
The installed model's size is chosen already, so pressing Enter does not download another model. The model files and the pack are reused.
You can also pass only what you want to change as options.

| To change | Options |
| --- | --- |
| The context length | `--setup --context 131072` |
| The KV cache's precision | `--setup --kv q4_0` (half the memory, a little less precise) |
| Another size or another model | `--setup --family gyro --model Gyro-M`, for example |
| Connecting from other PCs or a phone | `--setup --host 0.0.0.0 --api-key <a secret string>` |
| The port | `--port 8096` |
| Reading pictures | `--setup --vision yes` (`gpu` or `cpu` to choose the GPU or the CPU build, `no` to turn it off) |
| VRAM left free for other programs | `--setup --vram-reserve-mib 3072` |
| Where the models go | `--setup --data-dir D:\XeStrata-data` |

`xestrata-cli.bat --calibrate` measures a few of the engine's settings on this PC (15-30 minutes or so). The next start uses them.

The settings file is `%APPDATA%\xestrata\xestrata-<model>.json`.
Keys setup does not write (environment variables added to `env`, for example) stay after `--setup`, but engine options added to `args` by hand are dropped by `--setup` (setup names them).
Change the engine's options through setup's options where you can.

## Using model files you already have

Model files you have downloaded can be used in either of two ways.

- Where they are: pass their folder to `--gguf-dir`, as in `xestrata-cli.bat --family gyro --model Gyro-M --gguf-dir D:\models\Gyro`. Nothing is copied or moved.
- In the data folder: put them, with their original names, in the folder setup names ("The model files go in ...").

Either way, keep the original file names (`Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf`; a split one `...-00001-of-00002.gguf` and so on).

Files checked against a pinned version, such as Gyro, get their SHA-256 checked on the first run. A big file takes a few minutes; later runs skip it.
When the SHA-256 does not match the pinned version, setup stops without deleting the file. If it is another version, move it out of the folder and run setup again: it downloads the pinned one.

Put Gyro on an SSD. About half of the file is the n-gram table, and the engine reads it for every token. It is slow on an HDD.

## Where things go

Nothing is written into the zip's folder.

| What | Where |
| --- | --- |
| The settings | `%APPDATA%\xestrata` |
| The logs and the engine's working folder | `%LOCALAPPDATA%\xestrata\state` |
| The models and packs (the data folder) | `%LOCALAPPDATA%\xestrata\data` by default (`--data-dir` changes it) |

If C: is short of space, point `--data-dir` at another SSD when you install.

## Updating and removing

- To update, unpack the new zip into another folder and run its `xestrata-cli.bat`. The settings and the models are used as they are.
- To remove it, delete the model files, settings and logs XeStrata keeps with `xestrata-cli.bat --remove-data` (it asks first), then delete the folder the zip was unpacked into.
