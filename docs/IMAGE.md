<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# Image generation

English | [日本語](IMAGE.ja.md)

Besides the text model, XeStrata can install a model that makes images (optional).
The images are made by XeStrata's own image engine, `strata-image`. It runs on the same SYCL kernels as the text engine, on Intel, AMD and NVIDIA GPUs alike.
It goes with any version of Qwen3.8-Flash-Next (not yet with the BETA models), and is used from the web page's Images tab, from the chat and through the API (vLLM-Omni-compatible, so OpenAI-compatible).

> **Contents:** [Choosing a model](#choosing-a-model) · [Installing](#installing) · [Sizes and memory](#sizes-and-memory) · [Options](#options) ·
> [The web page and the chat](#the-web-page-and-the-chat) · [API](#api) · [Sharing the GPU's memory](#sharing-the-gpus-memory) · [Speed](#speed) ·
> [When something goes wrong](#when-something-goes-wrong) · [Licenses](#licenses)

## Choosing a model

| `--images` | Model | Steps | Good for |
| --- | --- | ---: | --- |
| `qwen-image-turbo` (default) | [Qwen-Image-2.1-Turbo](https://huggingface.co/unsloth/Qwen-Image-2.1-Turbo-GGUF) | 8 | Everyday images and edits (**recommended**) |
| `qwen-image-base` | [Qwen-Image-2.1](https://huggingface.co/unsloth/Qwen-Image-2.1-GGUF) | 40 | Turbo's base model. The steps can be changed; an image takes about 5 times as long as with Turbo |
| `mage-flow-turbo` | [Mage-Flow-Turbo](https://huggingface.co/Comfy-Org/Mage-Flow) | 4 | The fastest, a small model (4B). For PCs with little RAM or VRAM |
| `mage-flow` | [Mage-Flow](https://huggingface.co/Comfy-Org/Mage-Flow) | 20 (CFG) | Tuned by reinforcement learning to follow instructions and draw text better |
| `mage-flow-base` | [Mage-Flow-Base](https://huggingface.co/Comfy-Org/Mage-Flow) | 30 (CFG) | The base model, before that tuning |
| `magetrail` | [MageTrail](https://huggingface.co/RicemanT/MageTrail) | 30 (CFG) | Mage-Flow tuned on Danbooru and e621 images. For illustrations drawn from booru tags. No edits |

In this document "Qwen-Image-2.1" means both Turbo and the base model; what holds for one of them only says "Turbo" or "the base model".

- **The text encoder** reads the prompt: Qwen3-VL-8B for Qwen-Image-2.1, Qwen3-VL-4B for Mage-Flow and MageTrail.
- **`-uncensored`** after the name (`qwen-image-turbo-uncensored`, `mage-flow-turbo-uncensored` and so on) takes a text encoder with the refusal direction removed:
  AtomicChat's Qwen3-VL-8B for Qwen-Image-2.1, huihui-ai's Qwen3-VL-4B (noctrex's GGUF) for Mage-Flow and MageTrail. The image model and the VAE stay the same.
- **CFG:** `mage-flow`, `mage-flow-base` and `magetrail` apply CFG (pulling toward the prompt by its difference from a negative prompt) at a strength of 5 by default, so each step computes twice.
- **Qwen-Image-2.1 and Mage-Flow differ:** Qwen-Image-2.1 takes up to 10 reference images for an edit, and masks and transparent backgrounds.
  Mage-Flow takes up to 3 reference images (as its edit model was trained), and has no masks or transparent backgrounds.
  Mage-Flow has separate image models for generating and for editing; setup installs both, and the engine switches to the edit model for edits only.
- **MageTrail** is Mage-Flow's image model reshaped and tuned; it has a generation model only, and makes no edits.
  Its prompts can be booru tags (`1girl, solo, silver hair, ...`) or sentences; `masterpiece, best quality, very aesthetic, aesthetic` raises the quality, and `Drawn by <artist>` at the start takes that artist's style (the 100 artists of the list MageTrail's author publishes).
  Its VAE is the Flux2 VAE its author uses (as Black Forest Labs publishes it with FLUX.2-klein-4B).

## Installing

```sh
xestrata --setup --images qwen-image-turbo     # installed from the source: ./setup.sh --setup --images qwen-image-turbo
```

- Setup downloads the image model, the VAE, the text encoder and the picture reader for edits (Qwen3-VL's mmproj), and adds `"images"` to the config.
  Every file is pinned to a revision and checked by its SHA-256.
- Setup run interactively asks for the image models too (numbers, comma-separated; with two or more, which is the default, and whether to take the text encoder with the refusal direction removed). Run again for a model, it offers that model's present ones as the default answer.
- It does not ask with `--images` or `--yes`. Run again with `--yes` and without `--images`, setup keeps the earlier image settings as they are.
- The Windows launcher (`xestrata.exe`) has the image models (and the sizes of those with several) on its page for installing a model, and an installed model's "Images ..." chooses its image models again (none takes image generation out).
- `--images no` removes image generation.
- Several models can be installed, comma-separated (`--images qwen-image-turbo,magetrail`). The first is the default; the others are used through the API's `model` or the web page's choice of model.
  A size after a model's name chooses that model's own (`--images qwen-image-turbo:Q8_0,qwen-image-base:Q4_K_M`); a model without one takes `--image-quant`'s, or the RAM decides.
  One runs on the GPU at a time: a request for another model ends the earlier model's engine and starts its own (a few seconds to a dozen or so). The RAM needed is the largest model's.
- With several XeStrata servers on one GPU, image jobs still run one at a time on that GPU: a job waits while another server's runs (the Monitor says so), and when its turn comes it ends the other models' engines first. Servers of the same model use the same engine.
- `tools/image/install.py --config <the config file>` also adds it to an existing config (`--model`, `--quant` and the others mean what setup's options do).

## Sizes and memory

Qwen-Image-2.1's image model comes in three sizes (`--image-quant`).

| Size | Download | RAM, about | VRAM |
| --- | ---: | ---: | ---: |
| Q8_0 (default) | 18.2 GB | 20 GB | 4 GB and up |
| Q6_K | 16.8 GB | 19 GB | 4 GB and up |
| Q4_K_M | 14.7 GB | 17 GB | 4 GB and up |

Mage-Flow comes in one size (bf16): a 21.9 GB download, about 16 GB of RAM (12 GB with `--image-dit-keep off`).
MageTrail comes in one size too: a 10.4 GB download, about 10 GB of RAM.
The Flux2 VAE takes about 4.3 GB of VRAM at 1024x1024; where that does not fit, it reads the image in tiles (slower, and the colors may shift slightly).

- **The download** is the image model (for Mage-Flow, the generation and edit models), the VAE, the text encoder (Q8_0) and the picture reader for edits (none for MageTrail) together.
- **The RAM** is needed besides the text model's: the files' sizes plus about 2 GB for the engine itself.
- **When it does not fit,** setup lowers the text encoder first, to Q6_K, Q5_K and Q4_K, until it fits the RAM the text model leaves, and then the image model, to Q6_K and Q4_K_M (with the image model lowered, the text encoder is the largest that fits beside it).
  For Mage-Flow only the text encoder is lowered.
  Setup's screen, the server's start-up and the web page say so when it lowered them.
  With `--image-ssd-offload on` nothing is lowered: the sizes chosen are installed, and what does not fit the RAM is read from the SSD (slower).
- **VRAM:** the engine streams the layers that do not fit the VRAM from the RAM, so the GPU's memory does not choose the size; more VRAM is faster.

## Options

Setup's options (given to `xestrata --setup`):

| Option | Default | Meaning |
| --- | --- | --- |
| `--images <model>`, `no` | — | Installs the image model; `no` removes image generation |
| `--image-quant Q8_0\|Q6_K\|Q4_K_M` | `Q8_0` | The size of Qwen-Image-2.1's image model (no effect on Mage-Flow) |
| `--image-ssd-offload on\|off` | `off` | `on`: the sizes are not lowered when the RAM is short |
| `--image-dit-q8 auto\|on\|off` | `auto` | Mage-Flow's image model in 8 bits or not. `auto` keeps 16 bits when it fits the VRAM with the text encoder, 8 bits when it does not |
| `--image-dit-keep on\|off` | `on` | Keeps Mage-Flow's image model not in use (the generation or the edit one) in RAM in 8 bits. `off` uses about 4.4 GB less RAM and reads it from the file again at each switch |
| `--image-gpu-policy auto\|keep\|shrink\|unload` | `auto` | What happens to the text model while an image is made ([below](#sharing-the-gpus-memory)); given together with `--images` |

Mage-Flow in 8 bits multiplies both the weights and the activations in int8 where the GPU's math library has int8 products (contrib-llvm on Intel and NVIDIA, and AMD), in the form of ComfyUI's int8_convrot; on other GPUs it is Q8_0.
Qwen-Image-2.1's image model too is multiplied in int8 on a GPU with int8 products.

The config file's (`xestrata-<model>.json`) `"images"` also has entries to set by hand:

| Entry | Default | Meaning |
| --- | --- | --- |
| `max_n` | 4 | The most images one request makes |
| `max_side` | 2048 | The largest side of an image, in pixels |
| `max_refs` | 10 for Qwen-Image-2.1, 3 for Mage-Flow | The most reference images an edit takes |
| `timeout_s` | 3600 | The most seconds a job is waited for |
| `reload` | `lazy` | A text model taken off by `unload` comes back at the next request (`lazy`) or right after the image (`eager`) |
| `daemon_idle_s` | 1800 | The seconds an image engine no server is connected to waits before it ends |
| `keep_files` | 64 | How many `response_format: "url"` images are kept |
| `out_dir` | a temporary folder | The folder for the images and the engine's log (`strata-image.log`) |

## The web page and the chat

**The Images tab:** with image generation set up, the web page (`http://127.0.0.1:8095`) has an Images tab.

- With several models installed, the model can be chosen; the fields' defaults and which fields there are (steps, transparent background, edits, mask) follow it.
- Choose the prompt, the size (1024x1024, 1536x1024, 1024x1536, 768x768, 512x512, 1920x1080, 1080x1920, auto, or any size with sides of 256 to 2048) and the seed, and press Generate.
- A negative prompt (what the image should not have), the steps and the CFG strength can be given too. Left empty, they are the model's defaults (the API's), which the fields show greyed out.
  The steps are not shown for Turbo (fixed at 8). The negative prompt takes effect only with CFG above 1, and a line under it says so when it does not.
- With Qwen-Image-2.1, a transparent background can be chosen as well.
- With Qwen-Image-2.1 and a reference image attached, the part of the first image to draw again can be painted on it (brush size, eraser, clear). Only the painted part is drawn again, and the rest keeps the image's pixels (the API's `mask`). Unpainted, the edit takes the whole image.
- Attaching reference images makes it an edit (PNG, JPEG, GIF, BMP, WebP).
- Images are made one at a time; a chat request waits meanwhile (and the other way round).
- The **Monitor** tab shows the stage, the step and the seconds a step takes.

**The chat:** in the web page's chat the model can make and change images itself.
Ask "draw a cat", and the model calls the image tools; the image appears in the conversation.
Ask "make the background night", and it changes the conversation's last image (one attached or one made in the conversation).

## API

The base URL is `http://127.0.0.1:8095/v1`.

```sh
curl http://127.0.0.1:8095/v1/images/generations -H 'Content-Type: application/json' \
  -d '{"prompt": "a red fox in the snow, watercolor", "size": "1024x1024", "seed": 42}'
```

### Making an image (`POST /v1/images/generations`)

| Field | Default | Meaning |
| --- | --- | --- |
| `prompt` | — | Required; up to 16,000 characters |
| `model` | the default model | The image model's name (as `/v1/models` lists it), to choose one where several are installed. An unknown name (OpenAI's `gpt-image-1`, for example) is the default model |
| `size` | `auto` (1024x1024) | `WIDTHxHEIGHT`, sides of 256 to 2048. A size whose sides are not multiples of 16 (1920x1080 and the like) is made at the multiples of 16 above it and its middle cut out |
| `n` | 1 | 1 to 4; image i uses the seed `seed + i` |
| `seed` | random | 0 to 2^62 |
| `steps` (`num_inference_steps`) | the model's steps | 2 to 100 for Qwen-Image-2.1's base model, 1 to 100 for Mage-Flow; Turbo stays at 8 |
| `quality` | `auto` | The steps of Qwen-Image-2.1's base model: `low` 20, `medium` (`standard`) 30, `high`, `hd` and `auto` the default 40 (`steps`, when given, wins) |
| `negative_prompt` | none (one space for Mage-Flow) | CFG's negative prompt |
| `true_cfg_scale`, `guidance_scale` | the config's for Mage-Flow | CFG's strength (0 to 30; 1 is no CFG). Qwen-Image-2.1 applies it only when `true_cfg_scale` is above 1 and there is a `negative_prompt` (it does not use `guidance_scale`); Mage-Flow takes `true_cfg_scale`, else `guidance_scale` |
| `background` | `auto` | `transparent` makes a transparent background (the prompt wrapped as Qwen recommends; png or webp only); `opaque` flattens the alpha onto white. Mage-Flow refuses `transparent` |
| `output_format` | `png` | `png`, `jpeg`, `webp` |
| `output_compression` | 100 | 0 to 100: jpeg's and webp's quality |
| `response_format` | `b64_json` | `url` returns a `/v1/images/files/<name>` link (the newest 64 are kept) |

Qwen-Image-2.1's CFG is diffusers' Qwen-Image-2.1 formula. It computes the prompt and the negative prompt both, so an image takes about twice as long at 1024x1024, about 1.5 times at 512x512.
Other fields (`style`, `user`, `moderation` and the like) are accepted and not used.

### Changing an image (`POST /v1/images/edits`)

Makes an image from reference images: rewriting a reference image, or making a person or a thing in one the subject.

- **The reference images:** OpenAI's multipart form with `image` or `image[]`, or JSON with `"images": [{"image_url": ...}]` (a data: URL or an http(s) URL).
  Up to 10 for Qwen-Image-2.1, 3 for Mage-Flow; each up to 32 MiB and at least 16x16 pixels.
  PNG, JPEG, GIF and BMP are read as they are, other formats (WebP and others) are turned into PNG first.
- **The size:** `size` `auto`, or none, gives the first reference image's aspect ratio at about 1024x1024 pixels.
- **How the references are read:** Qwen-Image-2.1 reads each at about 1024x1024 pixels with its aspect ratio kept, whatever the output's size; more references take more time and memory.
  Mage-Flow's picture reader reads them with the long side at 384 pixels or less, and its VAE at the output's size.
- **A mask** (Qwen-Image-2.1 only): `mask`, a PNG with an alpha channel the size of the first image, redraws only its transparent part and keeps the rest as the first image's pixels.
- **The steps:** for Qwen-Image-2.1 the same as for generating; for Mage-Flow 4 for Turbo, 30 for the other two.
- A reference image's alpha goes to the image model as it is. The other fields are those of generating.

### The answer

- `data[]`: the images (`b64_json` or `url`), with `revised_prompt` and `seed`.
- `usage`: counted in the image engine's tokens (the prompt's tokens, and one token for each 16x16 pixels of the reference images and the image made).
- `strata`: the seconds taken, the steps, CFG's strength, the text encoder's, the sampling's and the VAE's times, and what happened to the text model (`gpu_policy`).
- The image model is listed in `/v1/models` too.

### From the chat

- `/v1/chat/completions` with `"strata_images": true` (the web page sends it) gives the model the `generate_image` and `edit_image` tools; when the model calls one, the server makes the image and shows it there.
  `edit_image` changes the conversation's last image (one attached or one made in the conversation). One request calls the tools up to 4 times.
- `/v1/responses` does the same with OpenAI's built-in `image_generation` tool, and returns the image in an `image_generation_call` item (`result`, the image in base64, with `size`, `quality`, `background`, `output_format` and `revised_prompt`).
  The tool's `size`, `quality`, `output_format`, `output_compression` and `background` are used for the images it makes (`size` `auto`: the size the model asks for).

## Sharing the GPU's memory

Image jobs run on the GPU one at a time.
Before a job the engine says the VRAM the image needs (the least, and enough to hold the whole image model and run the VAE in one pass), and the server decides what happens to the text model with `--image-gpu-policy`.

| `--image-gpu-policy` | The text model |
| --- | --- |
| `keep` | Stays. The image engine loads the layers the free VRAM holds and streams the rest |
| `shrink` | The text engine frees VRAM (`--vram-elastic`) and takes it back after the image |
| `unload` | The conversation is saved and the text engine taken off; the next start brings the conversation back |
| `auto` (default) | `keep` when everything fits; else `shrink` when VRAM can be freed; `keep` with the layers streamed when the least fits; `unload` otherwise. `keep` on the processor's own graphics |

- After a job the image engine keeps its layers only with `keep` and VRAM to spare; otherwise it lets go of what it holds (waiting, the process alone, about 200 MiB).
- The image engine starts at the first image and stays. Its kernels are prepared once, and the text state of the same prompt and the layers in VRAM stay for the next image.
- The image engine is a process apart from the server, and stays when the text model is changed. It ends 30 minutes after the last server leaves (`daemon_idle_s`).
- The K and V cache of the reference images' tokens in a Qwen-Image-2.1 edit (about 2 GB for one 1024x1024 image) keeps what the VRAM cannot hold in pinned RAM and brings it over at each step.

**The image attention** runs on oneDNN on Intel GPUs and on AOTriton on AMD GPUs (both are in the packages and the 7z), on cuDNN on NVIDIA GPUs where it is installed (contrib-llvm; `xestrata --setup` says how), and on XeStrata's own kernel otherwise.

## Speed

One 1024x1024 image with Turbo (`qwen-image-turbo`), 8 steps, with the engine warm (measured when upstream 0.2.0-beta was integrated):

| GPU | XeStrata | upstream (stable-diffusion.cpp) |
| --- | ---: | ---: |
| Arc Pro B70 | 7.6 s | 52.0 s |
| RTX 4070 | 13.9 s | 26.6 s |
| RX 9060 XT | 22.5 s | 73.7 s |

- Qwen-Image-2.1's base model (40 steps) takes about 5 times as long per image as Turbo.
- A Qwen-Image-2.1 edit at 1024x1024 on the B70 takes about 46 s with 3 reference images and about 4 minutes with 10.
- `xestrata --bench` measures it on your PC last ([BENCH](BENCH.md)).

## When something goes wrong

| Message | Cause and what to do |
| --- | --- |
| `images_not_set_up` (404) | Image generation is not set up. Run `xestrata --setup --images <model>` |
| `image generation: missing ...` (at start) | Image files are missing. Run setup with `--images` again |
| `this image needs at least X GiB of VRAM ...` (400) | Not enough VRAM for this image. Use fewer or smaller reference images, or a smaller image |
| `this image keeps X GiB of its prompt's cache in RAM ...` (400) | Not enough RAM for the reference images' cache. Likewise use fewer or smaller reference images |
| `edits need the vision encoder ...` (400) | The files for edits are missing. Run setup with `--images` again |
| `the image engine did not start ...`, `the image engine ended` | The image engine did not start, or stopped. See the log the message names (`strata-image.log`) |
| `Image generation runs the image model at Q6_K instead of Q8_0 ...` | The RAM is short, so setup lowered the size ([Sizes and memory](#sizes-and-memory)) |
| `faster with NVIDIA's ... cuDNN for the image engine's attention` (setup) | Without cuDNN, the attention on NVIDIA GPUs runs on XeStrata's own kernel. The command shown installs it, which is faster |

## Licenses

- Qwen-Image-2.1-Turbo's and Qwen-Image-2.1's weights are under the Qwen Research License, which is not a free software license. Read it before using them commercially.
- Mage-Flow's weights are under MIT, but Microsoft's README says it is a model for research, not meant for products or services.
  XeStrata adds neither a content filter nor a watermark to Mage-Flow.
- MageTrail's weights are under MIT. MageTrail was trained on Danbooru and e621 images, and XeStrata adds no content filter to it either.
  Its VAE is the Flux2 VAE Black Forest Labs publishes with FLUX.2-klein-4B, under Apache 2.0 (the same VAE in MageTrail's repository is under FLUX.2-dev's non-commercial license, so setup does not take it).
- The text encoders (Qwen3-VL, and AtomicChat's and huihui-ai's versions) are under their own licenses.
