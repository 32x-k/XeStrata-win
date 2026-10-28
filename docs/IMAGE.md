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
| `magetrail` | [MageTrail](https://huggingface.co/RicemanT/MageTrail) | 30 (CFG) | Mage-Flow tuned on Danbooru and e621 images. For illustrations drawn from booru tags. No edits (img2img and masks work) |
| `hunyuan-image-3-distil` | [HunyuanImage 3.0 Instruct-Distil](https://huggingface.co/tencent/HunyuanImage-3.0-Instruct-Distil) | 8 | Tencent's large MoE model (80B): the most detailed and the slowest. It reads its own prompt, so it needs no text encoder, and it uses the most disk and RAM (about 52 GB). Edits with up to 3 reference images (no transparent backgrounds) |

In this document "Qwen-Image-2.1" means both Turbo and the base model; what holds for one of them only says "Turbo" or "the base model".

- **The text encoder** reads the prompt: Qwen3-VL-8B for Qwen-Image-2.1, Qwen3-VL-4B for Mage-Flow and MageTrail.
  HunyuanImage 3.0 has none: its image model reads the prompt itself. XeStrata builds the prompt's word ids from the model's own `tokenizer.json` and the system prompt the model uses by default (the `en_unified.txt` setup wrote out of `system_prompt.py`, 5,674 characters).
- **`-uncensored`** after the name (`qwen-image-turbo-uncensored`, `mage-flow-turbo-uncensored` and so on) takes a text encoder with the refusal direction removed:
  AtomicChat's Qwen3-VL-8B for Qwen-Image-2.1, huihui-ai's Qwen3-VL-4B (noctrex's GGUF) for Mage-Flow and MageTrail. The image model and the VAE stay the same.
- **CFG:** `mage-flow`, `mage-flow-base` and `magetrail` apply CFG (pulling toward the prompt by its difference from a negative prompt) at a strength of 5 by default, so each step computes twice.
- **Guidance:** HunyuanImage 3.0's guidance (2.5 by default) is not a second pass as CFG is: it is one token that carries the number to the model. It has no negative prompt.
- **Qwen-Image-2.1 and Mage-Flow differ:** Qwen-Image-2.1 takes up to 10 reference images for an edit, and transparent backgrounds.
  Mage-Flow takes up to 3 reference images (as its edit model was trained), and has no transparent backgrounds.
  Mage-Flow has separate image models for generating and for editing; setup installs both, and the engine switches to the edit model for edits only.
- **MageTrail** is Mage-Flow's image model reshaped and tuned; it has a generation model only, and makes no edits (img2img from one image and masks it takes).
  Its prompts can be booru tags (`1girl, solo, silver hair, ...`) or sentences; `masterpiece, best quality, very aesthetic, aesthetic` raises the quality, and `Drawn by <artist>` at the start takes that artist's style (the 100 artists of the list MageTrail's author publishes).
  Its VAE is the Flux2 VAE its author uses (as Black Forest Labs publishes it with FLUX.2-klein-4B).
- **HunyuanImage 3.0** edits with up to 3 reference images (the model's own way: a reference goes in as its VAE latent and as SigLIP2's features, before the prompt); it has no transparent backgrounds. The size is made at the one of the model's own 37 sizes (buckets) whose shape is nearest the one asked for (512x512 becomes 1024x1024). With the size `auto`, or none, the model chooses the aspect ratio itself from the prompt (and an edit's reference images), as Tencent's `image_size="auto"` does; the choice takes 4 to 13 seconds on a B70.
  The weights are Tencent's two quantizations (w4a8 and int8). It draws from the prompt at once, not the way Tencent runs it by default (the model reasoning and rewriting the prompt before it draws).
- **Masks and strength:** every model takes a mask (the part to draw again) and a strength (how much to change the image). Neither is the model's own: both bring the latent back to the image's during the sampling, a way that works with any model (as diffusers and ComfyUI do).
  A model with edits takes its first reference image as the image; a model without edits (MageTrail, Mage-Flow without its edit model) starts from one image, img2img.

## Installing

```sh
xestrata --setup --images qwen-image-turbo     # installed from the source: ./setup.sh --setup --images qwen-image-turbo
```

- Setup downloads the image model, the VAE, the text encoder and the picture reader for edits (Qwen3-VL's mmproj), and adds `"images"` to the config.
  Every file is pinned to a revision and checked by its SHA-256.
- HunyuanImage 3.0 downloads five files: its image model (Tencent's quantization), the VAE, the model's own `tokenizer.json`, the `system_prompt.py` that holds its system prompt, and the last of the model's bf16 shards, which holds the SigLIP2 that reads an edit's reference images (3.3 GB, of which the engine reads 0.9 GB). It also takes what the model chooses its size with (the final norm and the output rows of the 37 aspect-ratio tokens) out of the bf16 shards by range requests, into `ratio_head.safetensors` (311 KB).
  Setup reads the system prompt out of `system_prompt.py` into `en_unified.txt` beside them (it never runs the file). It has no text encoder.
  Tencent publishes the weights in bf16 only, so XeStrata takes Tencent's quantized files (single checkpoints made for ComfyUI) pinned where they are published.
- Setup run interactively asks for the image models too (numbers, comma-separated; with two or more, which is the default, and whether to take the text encoder with the refusal direction removed). Run again for a model, it offers that model's present ones as the default answer.
- It does not ask with `--images` or `--yes`. Run again with `--yes` and without `--images`, setup keeps the earlier image settings as they are.
- The Windows launcher (`xestrata.exe`) has the image models (and the sizes of those with several) on its page for installing a model, and an installed model's "Images ..." chooses its image models again (none takes image generation out).
- `--images no` removes image generation.
- Several models can be installed, comma-separated (`--images qwen-image-turbo,magetrail`). The first is the default; the others are used through the API's `model` or the web page's choice of model.
  A size after a model's name chooses that model's own (`--images qwen-image-turbo:Q8_0,qwen-image-base:Q4_K_M`); a model without one takes `--image-quant`'s, or the RAM decides.
  Mage-Flow and MageTrail come in one bf16 file each, so theirs is `:int8` (the engine turns it to 8 bits as it reads it, always) or `:bf16` (kept 16-bit, always; it needs more RAM). Without one, `--image-dit-q8` says, or the VRAM decides.
  HunyuanImage 3.0's is `:w4a8` or `:int8`.
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

HunyuanImage 3.0 comes in Tencent's two quantizations (`--image-quant`; w4a8 by default).

| Size | Download | RAM, about | VRAM |
| --- | ---: | ---: | ---: |
| w4a8 (default) | 52.8 GB | 52 GB | 12 GB and up (edits about 20 GB and up) |
| int8 | 87.7 GB | 87 GB | about 14 GB and up (edits about 22 GB and up) |

w4a8 is 4-bit weights with 8-bit activations (Tencent's quantization for ComfyUI), a step about 2.5 times as fast as int8's; int8 (convrot's int8) comes nearer the bf16 weights' images.
A GPU that does not hold the weights that always stay in VRAM (about 6.7 GB; with edits, SigLIP2's and the VAE encoder's 1.6 GB too, put there with the first reference image) and two slots for the streamed experts (2.7 GB for w4a8, 4.8 GB for int8) cannot run it. More VRAM keeps more experts resident and is faster. The VAE takes about 7.5 GB of working memory at 1024x1024; where that does not fit, it decodes in tiles as the reference implementation does (slower, and the picture differs slightly from the whole decode's). While an edit's reference images are read, the VAE encoder works in buffers of its own (about 7.5 GB for a 1024x1024 reference, given back once they are read); that is not tiled, so an edit that does not fit is refused.

- **The download** is the image model (for Mage-Flow, the generation and edit models), the VAE, the text encoder (Q8_0) and the picture reader for edits (none for MageTrail) together (for HunyuanImage 3.0, its tokenizer and system prompt instead of a text encoder, and the SigLIP2 shard for its edits).
- **The RAM** is needed besides the text model's: the files' sizes plus about 2 GB for the engine itself.
- **When it does not fit,** setup lowers the text encoder first, to Q6_K, Q5_K and Q4_K, until it fits the RAM the text model leaves, and then the image model, to Q6_K and Q4_K_M (with the image model lowered, the text encoder is the largest that fits beside it).
  Mage-Flow lowers only its text encoder, HunyuanImage 3.0 only its image model (int8 to w4a8).
  Setup's screen, the server's start-up and the web page say so when it lowered them.
  With `--image-ssd-offload on` nothing is lowered: the sizes chosen are installed, and what does not fit the RAM is read from the SSD (slower).
- **VRAM:** the engine streams the layers that do not fit the VRAM from the RAM, so the GPU's memory does not choose the size; more VRAM is faster.

## Options

Setup's options (given to `xestrata --setup`):

| Option | Default | Meaning |
| --- | --- | --- |
| `--images <model>`, `no` | — | Installs the image model; `no` removes image generation |
| `--image-quant Q8_0\|Q6_K\|Q4_K_M`, `w4a8\|int8` | `Q8_0` (`w4a8` for HunyuanImage 3.0) | The size of the image model (three for Qwen-Image-2.1, Tencent's two quantizations for HunyuanImage 3.0; no effect on Mage-Flow) |
| `--image-ssd-offload on\|off` | `off` | `on`: the sizes are not lowered when the RAM is short |
| `--image-dit-q8 auto\|on\|off` | `auto` | Mage-Flow's image model in 8 bits or not. `auto` keeps 16 bits when it fits the VRAM with the text encoder, 8 bits when it does not |
| `--image-dit-keep on\|off` | `on` | Keeps Mage-Flow's image model not in use (the generation or the edit one) in RAM in 8 bits. `off` uses about 4.4 GB less RAM and reads it from the file again at each switch |
| `--image-gpu-policy auto\|keep\|shrink\|unload` | `auto` | What happens to the text model while an image is made ([below](#sharing-the-gpus-memory)); given together with `--images` |

Mage-Flow in 8 bits multiplies both the weights and the activations in int8 where the GPU's math library has int8 products (contrib-llvm on Intel and NVIDIA, and AMD), in the form of ComfyUI's int8_convrot; on other GPUs it is Q8_0.
Qwen-Image-2.1's image model too is multiplied in int8 on a GPU with int8 products.

The config file's (`xestrata-<model>.json`) `"images"` also has entries to set by hand:

| Entry | Default | Meaning |
| --- | --- | --- |
| `max_n` | 16 | The most images one request makes |
| `budget_s` | 600 | The most seconds one request's images may take, estimated from the speed the server measured (for each size); more images are refused. Before the server has made an image, 4 at most |
| `max_side` | 2048 | The largest side of an image, in pixels |
| `max_refs` | 10 for Qwen-Image-2.1, 3 for Mage-Flow | The most reference images an edit takes |
| `timeout_s` | 3600 | The most seconds a job is waited for |
| `reload` | `lazy` | A text model taken off by `unload` comes back at the next request (`lazy`) or right after the image (`eager`) |
| `daemon_idle_s` | 1800 | The seconds an image engine no server is connected to waits before it ends |
| `keep_files` | 64 | How many `response_format: "url"` images are kept |
| `out_dir` | a temporary folder | The folder for the images and the engine's log (`strata-image.log`) |

HunyuanImage 3.0's `"images"` also has `steps` (8), `guidance` (2.5), `resolutions` (the resolution buckets' parameters), `vision` (the shard with SigLIP2) and `ratio_head` (what it chooses its size with; without it `auto` is 1024x1024); its `max_refs` defaults to 3.

## The web page and the chat

**The Images tab:** with image generation set up, the web page (`http://127.0.0.1:8095`) has an Images tab.

- With several models installed, the model can be chosen; the fields' defaults and which fields there are (steps, transparent background, an edit or img2img) follow it.
- Choose the prompt, the size (1024x1024, 1536x1024, 1024x1536, 768x768, 512x512, 1920x1080, 1080x1920, auto, or any size with sides of 256 to 2048) and the seed, and press Generate.
  HunyuanImage 3.0 makes the image at the one of its 37 sizes (buckets) whose shape is nearest the one asked for, and says so when the size differs. With `auto` it makes the image at the size the model chooses.
- The number of images and the format (PNG, JPEG, WebP or JPEG XL; all but PNG with a quality) can be chosen; a transparent background takes no JPEG.
  The number goes up to what fits in `budget_s` at the size and steps chosen (`max_n` at most), with the estimated time beside it.
  The format is JPEG XL by default (quality 100, lossless) when the browser shows it and the server writes it, PNG otherwise. The format and quality chosen are kept in the browser.
- A negative prompt (what the image should not have), the steps and the CFG strength can be given too. Left empty, they are the model's defaults (the API's), which the fields show greyed out.
  The steps are not shown for Turbo (fixed at 8). The negative prompt takes effect only with CFG above 1, and a line under it says so when it does not.
  With HunyuanImage 3.0 the guidance is one token, so the field is Guidance and there is no negative prompt field.
- With Qwen-Image-2.1, a transparent background can be chosen as well.
- Attaching an image makes it an edit with a model that edits, and img2img from that one image with a model that does not (MageTrail and the like) (PNG, JPEG, GIF, BMP, WebP). HunyuanImage 3.0 takes up to 3 reference images.
- With an image attached, the part of the first image to draw again can be painted on it (brush size, eraser, clear). Only the painted part is drawn again, and the rest keeps the image's pixels (the API's `mask`). Unpainted, it is the whole image.
- With an image attached, the Strength field appears (the API's `strength`): the lower, the more of the image it keeps. Empty is the model's default: 1 for an edit (no start from the image's latent: the edits as before), 0.87 for img2img (lowered for a model of few steps, so that it skips one step at least).
- Images are made one at a time. A chat request that comes while several are made runs after the image being made, and the rest follow it (under `unload`, which takes the text model off, the images are all made first).
- MOSS-Image has more:
   - "Write the prompt out" has the chat's model write the short prompt out (off, a detailed description, or a layout of boxes: the API's `enhance`). The prompt or layout it wrote is shown under the image, and a layout can be loaded into the boxes with "Edit this layout" and changed.
   - "Place things with boxes" shows a canvas of the image's shape: drag on it to draw a box, drag a box to move it, drag its bottom-right corner to change its size. The box selected takes its kind (object, character, text), a description of how it looks, and a text box's exact words (the API's `layout`). The Delete key or the button deletes a box.
   - "More settings" changes the shift, the blocks the negative prompt's pass skips, and skip-layer guidance's blocks, scale and window. Empty is the model's default.
- The **Monitor** tab shows the stage, the step and the seconds a step takes.

**The chat:** in the web page's chat the model can make and change images itself.
Ask "draw a cat", and the model calls the image tools; the image appears in the conversation.
Ask "make the background night", and it changes the conversation's last image (one attached or one made in the conversation).
With several image models installed, ask in words ("draw it with MageTrail", "like a photograph") and the model picks the image model that fits (none asked for: the default; for a change, those that edit first, and img2img when one without edits is asked for).
With MOSS-Image installed, the model may draft a layout of boxes for an image; "Open the layout in Images" under that tool call loads its prompt and layout into the Images tab for a change.
"Images" under the message box chooses the chat's images: `auto` (as above), `off` (no images), or a model's name (the chat's images always with that model); the browser keeps the choice.

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
| `size` | `auto` (1024x1024) | `WIDTHxHEIGHT`, sides of 256 to 2048. A size whose sides are not multiples of 16 (1920x1080 and the like) is made at the multiples of 16 above it and its middle cut out. HunyuanImage 3.0 makes the image at the bucket (of its 37) whose shape is nearest, and the answer's `size` is that one; with `auto` or none, at the bucket the model chooses |
| `n` | 1 | 1 to `max_n` (16); a number that does not fit in `budget_s` is refused with the estimated time. Image i uses the seed `seed + i` |
| `seed` | random | 0 to 2^62 |
| `steps` (`num_inference_steps`) | the model's steps | 2 to 100 for Qwen-Image-2.1's base model, 1 to 100 for Mage-Flow, HunyuanImage 3.0 and MOSS-Image (8 by default for HunyuanImage 3.0, 50 for MOSS-Image); Turbo stays at 8 |
| `quality` | `auto` | The steps of Qwen-Image-2.1's base model: `low` 20, `medium` (`standard`) 30, `high`, `hd` and `auto` the default 40 (`steps`, when given, wins). Mage-Flow and HunyuanImage 3.0 do not use it |
| `negative_prompt` | none (one space for Mage-Flow, empty for MOSS-Image) | CFG's negative prompt. HunyuanImage 3.0 does not use it (its guidance is one token: there is nothing to steer) |
| `true_cfg_scale`, `guidance_scale` | the config's for Mage-Flow, HunyuanImage 3.0 and MOSS-Image | CFG's strength (0 to 30; 1 is no CFG). Qwen-Image-2.1 applies it only when `true_cfg_scale` is above 1 and there is a `negative_prompt` (it does not use `guidance_scale`); Mage-Flow and MOSS-Image take `true_cfg_scale`, else `guidance_scale` (3 by default for MOSS-Image). HunyuanImage 3.0 takes it as the guidance's strength (`guidance_scale`, else `true_cfg_scale`, else the config's 2.5) and applies no CFG |
| `background` | `auto` | `transparent` makes a transparent background (the prompt wrapped as Qwen recommends; png, webp or jxl only); `opaque` flattens the alpha onto white. Mage-Flow and HunyuanImage 3.0 refuse `transparent` |
| `output_format` | `png` | `png`, `jpeg`, `webp`, `jxl` (JPEG XL; only with libjxl's cjxl) |
| `output_compression` | 100 | 0 to 100: jpeg's, webp's and jxl's quality (jxl's 100 is lossless) |
| `response_format` | `b64_json` | `url` returns a `/v1/images/files/<name>` link (the newest 64 are kept) |

Qwen-Image-2.1's CFG is diffusers' Qwen-Image-2.1 formula. It computes the prompt and the negative prompt both, so an image takes about twice as long at 1024x1024, about 1.5 times at 512x512.
Other fields (`style`, `user`, `moderation` and the like) are accepted and not used.

MOSS-Image also takes these fields, by its official pipeline's names (the other models do not use them).

| Field | Default | Meaning |
| --- | --- | --- |
| `shift` | 5 | The sampler's time shift (above 0, up to 20) |
| `uncond_skip_layers` | `[22]` | The DiT's blocks (0 to 41) CFG's negative side skips; `[]` or `null` skips none |
| `slg_layers`, `slg_scale` | none, 0 | Skip-layer guidance: the prompt computed once more with those blocks skipped, `slg_scale` times the difference added (a step then takes three passes). None when `slg_scale` is 0 or there are no blocks |
| `slg_start`, `slg_stop` | 0, 1 | The steps where the two skips above act (shares of all the steps) |
| `layout` | none | A layout (MOSS-Image's layout, below) |
| `enhance` | none (off) | The prompt written out by the chat's model first: `true` (the model chooses a caption or a layout), `"caption"` or `"layout"`. Not with `layout` |

#### MOSS-Image's layout

With `layout`, a JSON of boxes that place things and text, the image is made from MOSS-Image's layout prompt. A box is `[x_min, y_min, x_max, y_max]` in pixels from the image's top-left corner (the draft of the model's official prompt writer).

```json
{"prompt": "a travel poster", "size": "1024x1024",
 "layout": {"summary": {"caption": "A minimalist travel poster with a cream background, a blue mountain and the title EXPLORE at the top."},
            "items": [{"label": "text", "caption": "Large dark-blue uppercase title.", "text": "EXPLORE", "bbox": [128, 96, 896, 224]},
                      {"label": "object", "caption": "A flat blue mountain illustration.", "bbox": [224, 320, 800, 896]}]}}
```

- `summary`: `caption` (the whole image; `prompt` when none), `meta` (an array of genre strings), `canvas` (`single`, `multi_panel` or `collage`), `palette`.
- `items`: one or more. `label` is `object`, `character` or `text`; `caption` its look; a text item's `text` the exact words (`\n` for a line break); `bbox` its box. `tags` and `palette` are taken too; other fields are refused.
- The boxes stay inside the image. `width` and `height`, when given, must be the `size`'s.
- The server turns each box into `[y_min, x_min, y_max, x_max]` from the image's middle (its edges outward to whole pixels) and gives the model the JSON in its official form; the answer's `revised_prompt` is that JSON.

#### Writing the prompt out (`enhance`)

With `enhance`, the chat's model (the text model loaded) first writes the prompt out into a detailed description or a layout for MOSS-Image, and the image is made from that. It writes it the way MOSS-Image's official prompt writer (skills/moss-image-prompt-enhancer) does.
The answer's `revised_prompt` is the prompt the model read, and `strata.enhance` says what was made (`made`: `caption` or `layout`, with the prompt it wrote (`prompt`) or the layout's draft) and the seconds it took. When it fails (an answer that cannot be read, no chat model), the image is made from the prompt as it was, and `strata.enhance.error` says why.

### Changing an image (`POST /v1/images/edits`)

Makes an image from reference images: rewriting a reference image, or making a person or a thing in one the subject.
A model without edits (MageTrail, Mage-Flow without its edit model, MOSS-Image) makes it from one image, img2img.

- **The reference images:** OpenAI's multipart form with `image` or `image[]`, or JSON with `"images": [{"image_url": ...}]` (a data: URL or an http(s) URL).
  Up to 10 for Qwen-Image-2.1, 3 for Mage-Flow and HunyuanImage 3.0, 1 for a model without edits; each up to 32 MiB and at least 16x16 pixels.
  PNG, JPEG, GIF and BMP are read as they are, other formats (WebP and others) are turned into PNG first.
- **The model:** `model` chooses it. Without one it is the default model; where the default does not edit (MageTrail), the default's img2img when there is a `strength` or a `mask`, else the first model that edits, and the answer's `model` says which.
- **The size:** `size` `auto`, or none, gives the first reference image's aspect ratio at about 1024x1024 pixels. HunyuanImage 3.0 makes the size the model chooses from the reference images and the prompt.
- **How the references are read:** Qwen-Image-2.1 reads each at about 1024x1024 pixels with its aspect ratio kept, whatever the output's size; more references take more time and memory.
  Mage-Flow's picture reader reads them with the long side at 384 pixels or less, and its VAE at the output's size.
  HunyuanImage 3.0 cuts a reference to the bucket whose shape is nearest and reads it with its VAE (4,096 tokens at 1024x1024) and with SigLIP2 (at most 1,024 patches): about 5,000 tokens a reference, which lengthen each step.
- **A mask:** `mask`, a PNG with an alpha channel the size of the first image, redraws only its transparent part and keeps the rest as the first image's pixels. Every model takes one.
- **strength:** above 0 and at most 1: how much of the first image to change; the lower, the more of it is kept. As diffusers' img2img, it skips the first floor(steps × (1 − strength)) of the schedule and starts from the image's latent with that point's share of noise (fewer steps run, so it is faster too).
  The default is 1 for an edit (the edits as before; the default denoise of ComfyUI's KSampler), 0.87 for img2img (the denoise of ComfyUI's own img2img example, but no more than 1 − 1/steps, so that it skips one step at least: 0.75 at 4 steps), and 1 for img2img with a mask alone. A value that runs no step is refused. ComfyUI's `denoise` is not taken (`denoise` 0.5 at 20 steps is `strength` 0.5 at 40).
- **The steps:** for Qwen-Image-2.1, HunyuanImage 3.0 and MOSS-Image the same as for generating; for Mage-Flow's edits 4 for Turbo, 30 for the other two, and for its img2img those of generating.
- A reference image's alpha goes to the image model as it is. The other fields are those of generating.

### The answer

- `data[]`: the images (`b64_json` or `url`), with `revised_prompt` and `seed`.
- `usage`: counted in the image engine's tokens (the prompt's tokens, and one token for each 16x16 pixels of the reference images and the image made). For HunyuanImage 3.0: the prompt's ids, one for each 16x16 pixels of the image made, and a reference image's VAE tokens and 1,024 SigLIP2 slots.
- `strata`: the seconds taken, the steps (those a strength ran), `strength` (when below 1), whether it was img2img (`img2img`) and took a mask (`mask`), whether it took a layout (`layout`), what the prompt's writing out made (`enhance`), CFG's strength, the text encoder's, the sampling's and the VAE's times, and what happened to the text model (`gpu_policy`). For HunyuanImage 3.0 it also has the guidance's strength (`guidance`), the bucket's number (`ratio`), when the size asked for was another, that size (`requested_size`), and when the model chose the size, the seconds the choice took (`size_choice_s`).
- The image model is listed in `/v1/models` too.

### From the chat

- `/v1/chat/completions` with `"strata_images": true` (the web page sends it) gives the model the `generate_image` and `edit_image` tools; when the model calls one, the server makes the image and shows it there.
  `edit_image` changes the conversation's last image (one attached or one made in the conversation); it takes a `strength` too, and is img2img with a model without edits. One request calls the tools up to 4 times.
- With MOSS-Image, `generate_image` takes a layout too (`layout`, as the API's), so the chat's model can draft its boxes.
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

HunyuanImage 3.0 (`hunyuan-image-3-distil`, w4a8, 1024x1024, 8 steps, one image once the engine is warm):

| GPU | An image | Sampling | VAE decode |
| --- | ---: | ---: | ---: |
| Arc Pro B70 (32 GB) | 32 s | 29.4 s | 2.5 s |
| RTX 4070 (12 GB) | 78 s | 74.1 s | 4.1 s (tiled) |
| RX 9060 XT (16 GB, Windows) | 98 s | 85.3 s | 12.6 s (tiled) |

- The first image after the engine starts takes longer by the prompt's reading and the like (about 19 s on the B70).
- int8 takes about 1.6 times as long an image as w4a8 on the B70.
- Most of a step is the streaming of the experts that do not fit the VRAM (43.6 GB over the 32 layers in w4a8) from the RAM and their turning back to 16 bits. More VRAM is faster.

## When something goes wrong

| Message | Cause and what to do |
| --- | --- |
| `images_not_set_up` (404) | Image generation is not set up. Run `xestrata --setup --images <model>` |
| `image generation: missing ...` (at start) | Image files are missing. Run setup with `--images` again |
| `this image needs at least X GiB of VRAM ...` (400) | Not enough VRAM for this image. Use fewer or smaller reference images, or a smaller image |
| `this image keeps X GiB of its prompt's cache in RAM ...` (400) | Not enough RAM for the reference images' cache. Likewise use fewer or smaller reference images |
| `strength: ... runs none of the ... steps` (400) | The strength is too low to run a step. Raise it or the steps |
| `image: one image (this image model has no edits ...` (400) | img2img with a model without edits takes one image |
| `the VAE file has no encoder for a base image` | The VAE file has no encoder, so img2img and masks cannot run. Run setup with `--images` again |
| `the VRAM budget ... is below the weights that always stay` (400) | HunyuanImage 3.0's weights that always stay and its slots (about 10 GB) do not fit the VRAM. It does not run on this GPU |
| `the image engine did not start ...`, `the image engine ended` | The image engine did not start, or stopped. See the log the message names (`strata-image.log`) |
| `Image generation runs the image model at Q6_K instead of Q8_0 ...` | The RAM is short, so setup lowered the size ([Sizes and memory](#sizes-and-memory)) |
| `faster with NVIDIA's ... cuDNN for the image engine's attention` (setup) | Without cuDNN, the attention on NVIDIA GPUs runs on XeStrata's own kernel. The command shown installs it, which is faster |

## Licenses

- Qwen-Image-2.1-Turbo's and Qwen-Image-2.1's weights are under the Qwen Research License, which is not a free software license. Read it before using them commercially.
- Mage-Flow's weights are under MIT, but Microsoft's README says it is a model for research, not meant for products or services.
  XeStrata adds neither a content filter nor a watermark to Mage-Flow.
- MageTrail's weights are under MIT. MageTrail was trained on Danbooru and e621 images, and XeStrata adds no content filter to it either.
  Its VAE is the Flux2 VAE Black Forest Labs publishes with FLUX.2-klein-4B, under Apache 2.0 (the same VAE in MageTrail's repository is under FLUX.2-dev's non-commercial license, so setup does not take it).
- HunyuanImage 3.0's weights are under the Tencent Hunyuan Community License Agreement, which does not apply in the European Union, the United Kingdom or South Korea.
  Read [Tencent's license](https://github.com/Tencent-Hunyuan/HunyuanImage-3.0/blob/main/LICENSE), its Acceptable Use Policy included, before you use the images.
  The quantized files are the ones PedroMarinhoDev publishes for ComfyUI.
- The text encoders (Qwen3-VL, and AtomicChat's and huihui-ai's versions) are under their own licenses.
