<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# 画像の生成

[English](IMAGE.md) | 日本語

XeStrata は、文章のモデルとは別に、画像を作るモデルを入れられます（任意）。
画像は XeStrata 自身の画像のエンジン `strata-image` が作ります。文章のエンジンと同じ SYCL のカーネルで、Intel・AMD・NVIDIA のどの GPU でも動きます。
Qwen3.8-Flash-Next のどの版とも組み合わせられ（BETA のモデルとはまだ組み合わせられません）、Web のページの Images のタブ、チャット、API（vLLM-Omni 互換（OpenAI 互換））から使えます。

> **目次:** [モデルの選び方](#モデルの選び方) · [入れ方](#入れ方) · [大きさとメモリ](#大きさとメモリ) · [オプション](#オプション) ·
> [Web のページとチャット](#web-のページとチャット) · [API](#api) · [GPU のメモリの分け合い](#gpu-のメモリの分け合い) · [速さ](#速さ) ·
> [困ったとき](#困ったとき) · [ライセンス](#ライセンス)

## モデルの選び方

| `--images` | モデル | ステップ | 向いていること |
| --- | --- | ---: | --- |
| `qwen-image-turbo`（既定） | [Qwen-Image-2.1-Turbo](https://huggingface.co/unsloth/Qwen-Image-2.1-Turbo-GGUF) | 8 | ふだんの画像の生成と編集（**おすすめ**） |
| `qwen-image-base` | [Qwen-Image-2.1](https://huggingface.co/unsloth/Qwen-Image-2.1-GGUF) | 40 | Turbo の元のモデル。ステップ数を変えられ、1 枚に Turbo の約 5 倍の時間がかかる |
| `mage-flow-turbo` | [Mage-Flow-Turbo](https://huggingface.co/Comfy-Org/Mage-Flow) | 4 | いちばん速い小さなモデル（4B）。RAM や VRAM の少ない PC 向け |
| `mage-flow` | [Mage-Flow](https://huggingface.co/Comfy-Org/Mage-Flow) | 20（CFG） | 強化学習で、指示への従い方と文字の描き方を調整した版 |
| `mage-flow-base` | [Mage-Flow-Base](https://huggingface.co/Comfy-Org/Mage-Flow) | 30（CFG） | 調整の前の元のモデル |
| `magetrail` | [MageTrail](https://huggingface.co/RicemanT/MageTrail) | 30（CFG） | Mage-Flow を Danbooru と e621 の画像で微調整した版。booru のタグで描くイラスト向け。編集はできない |

この文書で「Qwen-Image-2.1」は Turbo と元のモデルの両方を指し、片方だけのことは「Turbo」「元のモデル」と書きます。

- **文の読み手**: プロンプトを読むモデルです。Qwen-Image-2.1 は Qwen3-VL-8B、Mage-Flow と MageTrail は Qwen3-VL-4B を使います。
- **`-uncensored`**: 名前の後ろに付けると（`qwen-image-turbo-uncensored`、`mage-flow-turbo-uncensored` など）、文の読み手を拒否の方向を取り除いたものに替えます。
  Qwen-Image-2.1 は AtomicChat の Qwen3-VL-8B、Mage-Flow と MageTrail は huihui-ai の Qwen3-VL-4B（noctrex の GGUF）です。画像のモデルと VAE はそのままです。
- **CFG**: `mage-flow`、`mage-flow-base`、`magetrail` は、既定で強さ 5 の CFG（ネガティブのプロンプトとの差でプロンプトに寄せる方法）を掛けるので、1 ステップに 2 回分の計算をします。
- **Qwen-Image-2.1 と Mage-Flow の違い**: Qwen-Image-2.1 は編集の参照画像を 10 枚まで受け、マスクと透明な背景も使えます。
  Mage-Flow は参照画像が 3 枚まで（編集のモデルの学習と同じ）で、マスクと透明な背景はありません。
  Mage-Flow は生成用と編集用で画像のモデルが別で、setup は両方を入れ、編集のときだけエンジンが差し替えます。
- **MageTrail**: Mage-Flow の画像のモデルを作り替えて微調整したもので、生成のモデルだけがあり、編集はできません。
  プロンプトは booru のタグ（`1girl, solo, silver hair, ...`）と文のどちらでも書け、`masterpiece, best quality, very aesthetic, aesthetic` を足すと画質が上がり、`Drawn by <画家名>` を先頭に置くとその画風になります（画家は MageTrail の作者が公開している一覧の 100 人）。
  VAE は作者が使っている Flux2 VAE（Black Forest Labs が FLUX.2-klein-4B で公開しているもの）です。

## 入れ方

```sh
xestrata --setup --images qwen-image-turbo     # ソースから入れたときは ./setup.sh --setup --images qwen-image-turbo
```

- setup は、画像のモデル、VAE、文の読み手、編集のための画像の読み手（Qwen3-VL の mmproj）をダウンロードして、設定に `"images"` を足します。
  どのファイルもリビジョンを固定し、SHA-256 で確かめます。
- `--images` を付けたときだけ画像を設定します。付けずに setup をやり直しても、前の画像の設定はそのまま残ります。
- 画像をやめるときは `--images no` です。
- カンマで区切ると、いくつものモデルを入れられます（`--images qwen-image-turbo,magetrail`）。先頭が既定のモデルで、ほかは API の `model` か Web のページのモデルの選択で使います。
  一度に GPU で動くのは 1 つで、違うモデルの要求が来ると、前のモデルのエンジンを止めてから起動し直します（数秒〜十数秒かかります）。RAM の目安は、いちばん大きいモデルの分です。
- 同じ GPU で XeStrata のサーバーをいくつか動かしているときも、画像の仕事は GPU ごとに 1 つずつ順番に動きます。別のサーバーの仕事が動いている間は待ち（Monitor に出ます）、順番が来ると、ほかのモデルのエンジンを止めてから動きます。同じモデルのサーバーどうしは、同じエンジンを使います。
- 入っている設定にあとから足すときは `tools/image/install.py --config <設定のファイル>` も使えます（`--model`、`--quant` などは setup のオプションと同じ意味です）。

## 大きさとメモリ

Qwen-Image-2.1 の画像のモデルは 3 つの大きさから選べます（`--image-quant`）。

| 大きさ | ダウンロード | RAM の目安 | VRAM |
| --- | ---: | ---: | ---: |
| Q8_0（既定） | 18.2 GB | 20 GB | 4 GB から |
| Q6_K | 16.8 GB | 19 GB | 4 GB から |
| Q4_K_M | 14.7 GB | 17 GB | 4 GB から |

Mage-Flow は大きさが 1 つ（bf16）で、ダウンロードは 21.9 GB、RAM の目安は 16 GB（`--image-dit-keep off` で 12 GB）です。
MageTrail も大きさが 1 つで、ダウンロードは 10.4 GB、RAM の目安は 10 GB です。
Flux2 VAE は 1024×1024 で約 4.3 GB の VRAM を使い、入らないときは画像をタイルに分けて読みます（遅くなり、色味がわずかに変わることがあります）。

- **ダウンロード**: 画像のモデル（Mage-Flow は生成用と編集用の 2 つ）、VAE、文の読み手（Q8_0）、編集のための画像の読み手（MageTrail にはありません）の合計です。
- **RAM の目安**: 文章のモデルとは別に要る量です。ファイルの大きさに、エンジン自身の約 2 GB を足したものです。
- **足りないとき**: setup は、文章のモデルの残す RAM に入るまで、まず文の読み手を Q6_K、Q5_K、Q4_K の順に下げ、それでも入らなければ画像のモデルを Q6_K、Q4_K_M に下げます（画像のモデルを下げたときは、文の読み手をその横に入る中で最も大きいものにします）。
  Mage-Flow では文の読み手だけを下げます。
  下げたときは、setup の画面、サーバーの起動時の表示、Web のページがそのことを知らせます。
  `--image-ssd-offload on` なら下げずに選んだ大きさのまま入れ、RAM に入らない分は SSD から読みます（遅くなります）。
- **VRAM**: VRAM に入らない層は、エンジンが RAM から流し込みます。そのため GPU のメモリでは大きさを選ばず、VRAM が大きいほど速くなります。

## オプション

setup のオプションです（`xestrata --setup` に付けます）。

| オプション | 既定 | 意味 |
| --- | --- | --- |
| `--images <モデル>`、`no` | — | 画像のモデルを入れる。`no` で画像の設定を消す |
| `--image-quant Q8_0\|Q6_K\|Q4_K_M` | `Q8_0` | Qwen-Image-2.1 の画像のモデルの大きさ（Mage-Flow には効きません） |
| `--image-ssd-offload on\|off` | `off` | `on` で、RAM が足りなくても大きさを下げない |
| `--image-dit-q8 auto\|on\|off` | `auto` | Mage-Flow の画像のモデルを 8 ビットにするか。`auto` は、文の読み手と一緒に VRAM に全部入るときは 16 ビットのまま、入らないときは 8 ビット |
| `--image-dit-keep on\|off` | `on` | Mage-Flow の使っていない方（生成用か編集用）の画像のモデルを 8 ビットで RAM に残す。`off` で RAM が約 4.4 GB 減り、差し替えのたびにファイルから読み直す |
| `--image-gpu-policy auto\|keep\|shrink\|unload` | `auto` | 画像を作るときの文章のモデルの扱い（[下](#gpu-のメモリの分け合い)）。`--images` と一緒に指定します |

Mage-Flow の 8 ビットは、GPU の計算ライブラリが int8 の積を持つとき（contrib-llvm の Intel・NVIDIA、AMD）は、重みと活性の両方を int8 にして掛けます（ComfyUI の int8_convrot と同じ形）。ほかの GPU では Q8_0 です。
Qwen-Image-2.1 の画像のモデルも、int8 の積がある GPU では同じく int8 で掛けます。

設定のファイル（`xestrata-<モデル>.json`）の `"images"` には、手で書ける項目もあります。

| 項目 | 既定 | 意味 |
| --- | --- | --- |
| `max_n` | 4 | 1 回の要求で作る画像の最大の数 |
| `max_side` | 2048 | 画像の一辺の最大のピクセル数 |
| `max_refs` | Qwen-Image-2.1 は 10、Mage-Flow は 3 | 編集の参照画像の最大の数 |
| `timeout_s` | 3600 | 1 つの仕事を待つ最大の秒数 |
| `reload` | `lazy` | `unload` で外した文章のモデルを、次の要求で戻す（`lazy`）か、画像のすぐあとに戻す（`eager`） |
| `daemon_idle_s` | 1800 | どのサーバーもつながっていない画像のエンジンが終わるまでの秒数 |
| `keep_files` | 64 | `response_format: "url"` の画像を残す数 |
| `out_dir` | 一時フォルダー | 画像とエンジンのログ（`strata-image.log`）を置くフォルダー |

## Web のページとチャット

**Images のタブ**: 画像を設定すると、Web のページ（`http://127.0.0.1:8095`）に Images のタブが出ます。

- いくつものモデルを入れたときは、モデルを選べます。欄の既定値や、使える欄（ステップ数、透明な背景、編集、マスク）は、選んだモデルに合わせて変わります。
- プロンプト、大きさ（1024x1024、1536x1024、1024x1536、768x768、512x512、1920x1080、1080x1920、auto、一辺 256〜2048 の自由な大きさ）、種を選んで Generate を押します。
- ネガティブのプロンプト（画像に入れたくないもの）、ステップ数、CFG の強さも指定できます。空欄ならモデルの既定（API と同じ）で、欄にその値が薄く出ます。
  ステップ数は Turbo（8 で固定）では出ません。ネガティブのプロンプトは CFG を 1 より大きくしたときだけ効き、効かないときは欄の下にそのことが出ます。
- Qwen-Image-2.1 では、透明な背景も選べます。
- Qwen-Image-2.1 で参照画像を添付すると、1 枚目の画像の上で描き直す部分を塗れます（ブラシの大きさ、消しゴム、全部消す）。塗った部分だけを描き直し、残りは元の画素のまま残します（API の `mask`）。塗らなければ、画像全体の編集です。
- 参照画像を添付すると編集になります（PNG、JPEG、GIF、BMP、WebP）。
- 画像は 1 枚ずつ作り、その間のチャットの要求は待ちます（逆も同じです）。
- **Monitor** のタブに、いまの段階、ステップの進み具合と 1 ステップの秒数が出ます。

**チャット**: Web のページのチャットでは、モデルが自分で画像を作ったり直したりできます。
「猫の絵を描いて」のように頼むと、モデルが画像の道具を呼び、作った画像が会話の中に出ます。
「背景を夜にして」のように頼むと、会話の最後の画像（添付したものか、会話で作ったもの）を直します。

## API

ベース URL は `http://127.0.0.1:8095/v1` です。

```sh
curl http://127.0.0.1:8095/v1/images/generations -H 'Content-Type: application/json' \
  -d '{"prompt": "a red fox in the snow, watercolor", "size": "1024x1024", "seed": 42}'
```

### 画像を作る（`POST /v1/images/generations`）

| 項目 | 既定 | 意味 |
| --- | --- | --- |
| `prompt` | — | 必須。16,000 文字まで |
| `model` | 既定のモデル | 画像のモデルの名前（`/v1/models` にあるもの）。いくつものモデルを入れたときに選びます。知らない名前（OpenAI の `gpt-image-1` など）は既定のモデルです |
| `size` | `auto`（1024x1024） | `幅x高さ`。一辺 256〜2048。16 の倍数でない大きさ（1920x1080 など）は、16 の倍数に切り上げて作り、真ん中を切り出します |
| `n` | 1 | 1〜4。i 枚目は種 `seed + i` で作ります |
| `seed` | ランダム | 0〜2^62 |
| `steps`（`num_inference_steps`） | モデルのステップ数 | Qwen-Image-2.1 の元のモデルは 2〜100、Mage-Flow は 1〜100。Turbo は 8 のままです |
| `quality` | `auto` | Qwen-Image-2.1 の元のモデルのステップ数: `low` 20、`medium`（`standard`）30、`high`・`hd`・`auto` は既定の 40（`steps` を付けたときはそちら） |
| `negative_prompt` | なし（Mage-Flow は空白 1 文字） | CFG のネガティブのプロンプト |
| `true_cfg_scale`、`guidance_scale` | Mage-Flow は設定の値 | CFG の強さ（0〜30、1 なら CFG なし）。Qwen-Image-2.1 は `true_cfg_scale` が 1 より大きく `negative_prompt` があるときだけ掛け（`guidance_scale` は使いません）、Mage-Flow は `true_cfg_scale`、なければ `guidance_scale` |
| `background` | `auto` | `transparent` で透明な背景（Qwen の推奨する形でプロンプトを包みます。png か webp だけ）、`opaque` でアルファを白の上に重ねて消す。Mage-Flow は `transparent` を断ります |
| `output_format` | `png` | `png`、`jpeg`、`webp` |
| `output_compression` | 100 | 0〜100。jpeg と webp の画質 |
| `response_format` | `b64_json` | `url` なら `/v1/images/files/<名前>` のリンクを返します（新しい 64 枚まで残します） |

Qwen-Image-2.1 の CFG は、diffusers の Qwen-Image-2.1 と同じ式です。条件ありとネガティブの 2 つ分を計算するので、時間は 1024x1024 で約 2 倍、512x512 では約 1.5 倍です。
ほかの項目（`style`、`user`、`moderation` など）は受けて、使いません。

### 画像を直す（`POST /v1/images/edits`）

参照画像を元に画像を作ります。参照画像を書き換えたり、その中の人や物を主題にしたりできます。

- **参照画像**: OpenAI の形の multipart の `image` か `image[]`、または JSON の `"images": [{"image_url": ...}]`（data: の URL、http(s) の URL）。
  Qwen-Image-2.1 は 10 枚まで、Mage-Flow は 3 枚までです。1 枚 32 MiB まで、16x16 ピクセル以上です。
  PNG、JPEG、GIF、BMP はそのまま、ほかの形式（WebP など）は PNG に変えて読みます。
- **大きさ**: `size` を `auto` にするか付けないと、1 枚目の参照画像の縦横比で約 1024x1024 ピクセルにします。
- **参照画像の読み方**: Qwen-Image-2.1 は、出力の大きさにかかわらず 1 枚ずつ縦横比を保った約 1024x1024 ピクセルで読みます。参照画像が多いほど時間とメモリがかかります。
  Mage-Flow は、画像の読み手が長い辺 384 ピクセル以下で、VAE が出力の大きさで読みます。
- **マスク**（Qwen-Image-2.1 だけ）: `mask` に、1 枚目の画像と同じ大きさのアルファ付きの PNG を付けると、その透明な部分だけを描き直し、残りは 1 枚目の画像の画素のまま残します。
- **ステップ数**: Qwen-Image-2.1 は生成と同じです。Mage-Flow は Turbo が 4、ほかの 2 つは 30 です。
- 参照画像のアルファも、そのまま画像のモデルに渡します。ほかの項目は生成と同じです。

### 応答

- `data[]`: 画像（`b64_json` か `url`）と、`revised_prompt`、`seed`。
- `usage`: 画像のエンジンのトークンで数えます（プロンプトのトークンと、参照画像と作った画像の 16x16 ピクセルで 1 つのトークン）。
- `strata`: かかった秒数、ステップ数、CFG の強さ、文の読み手・サンプリング・VAE のそれぞれの時間、文章のモデルの扱い（`gpu_policy`）。
- 画像のモデルは `/v1/models` にも出ます。

### チャットから

- `/v1/chat/completions` に `"strata_images": true` を付けると（Web のページは付けます）、モデルに `generate_image` と `edit_image` の道具を渡し、モデルが呼ぶとサーバーが画像を作ってその場に出します。
  `edit_image` は、会話の最後の画像（添付したものか、会話で作ったもの）を直します。1 回の要求で道具を呼べるのは 4 回までです。
- `/v1/responses` では、OpenAI の組み込みの `image_generation` の道具を付けると同じことをし、作った画像を `image_generation_call` の項目（`result` に画像の base64、`size`・`quality`・`background`・`output_format`・`revised_prompt` も）で返します。
  道具の `size`・`quality`・`output_format`・`output_compression`・`background` は、その道具で作る画像に使います（`size` が `auto` ならモデルの指定した大きさ）。

## GPU のメモリの分け合い

画像の仕事は GPU で 1 つずつ動きます。
エンジンは仕事の前に、その画像に要る VRAM（最低限と、画像のモデルを全部載せて VAE を 1 回で回せる量）を答え、サーバーは `--image-gpu-policy` で文章のモデルの扱いを決めます。

| `--image-gpu-policy` | 文章のモデルの扱い |
| --- | --- |
| `keep` | そのまま。画像のエンジンは空いた VRAM に入る分だけ層を載せ、残りを流し込みます |
| `shrink` | 文章のエンジンの VRAM を空けて（`--vram-elastic`）、画像のあとで戻します |
| `unload` | 会話を保存して文章のエンジンを外し、次の起動で会話を戻します |
| `auto`（既定） | 全部が入れば `keep`、入らなければ空けられるときは `shrink`、最低限が入るなら層を流し込んで `keep`、それも入らなければ `unload`。CPU 内蔵の GPU では `keep` |

- 仕事のあと、`keep` で VRAM に余裕があるときだけ、画像のエンジンは層を残します。それ以外は持っているものを手放します（待機中はプロセスだけの約 200 MiB）。
- 画像のエンジンは最初の画像で起動し、そのまま残ります。カーネルの準備は 1 度だけで、同じプロンプトの文の状態と VRAM に載せた層は次の画像まで残ります。
- 画像のエンジンはサーバーとは別のプロセスで、文章のモデルを替えても残ります。どのサーバーもつながらなくなって 30 分たつと終わります（`daemon_idle_s`）。
- Qwen-Image-2.1 の編集の参照画像のトークンの K と V のキャッシュ（1024x1024 の 1 枚で約 2 GB）は、VRAM に入らない分をピン留めした RAM に置いて毎ステップ運びます。

**画像の注意**: Intel の GPU では oneDNN、AMD の GPU では AOTriton（どちらもパッケージと 7z に入っています）、NVIDIA の GPU では cuDNN が入っていれば cuDNN（contrib-llvm。`xestrata --setup` が入れ方を示します）で計算し、ほかは XeStrata のカーネルで計算します。

## 速さ

Turbo（`qwen-image-turbo`）で 1024x1024、8 ステップの画像 1 枚（エンジンが暖まったあと）にかかった時間です（upstream 0.2.0-beta の取り込みのときの計測）。

| GPU | XeStrata | upstream（stable-diffusion.cpp） |
| --- | ---: | ---: |
| Arc Pro B70 | 7.6 秒 | 52.0 秒 |
| RTX 4070 | 13.9 秒 | 26.6 秒 |
| RX 9060 XT | 22.5 秒 | 73.7 秒 |

- Qwen-Image-2.1 の元のモデル（40 ステップ）は、1 枚に Turbo の約 5 倍の時間がかかります。
- Qwen-Image-2.1 の編集は、B70 の 1024x1024 で参照画像 3 枚が約 46 秒、10 枚が約 4 分です。
- 自分の PC での速さは `xestrata --bench` が最後に測ります（[BENCH](BENCH.ja.md)）。

## 困ったとき

| 表示 | 原因と対処 |
| --- | --- |
| `images_not_set_up`（404） | 画像を設定していません。`xestrata --setup --images <モデル>` を実行します |
| `image generation: missing ...`（起動時） | 画像のファイルが足りません。setup を `--images` を付けてもう一度実行します |
| `this image needs at least X GiB of VRAM ...`（400） | この画像に要る VRAM が足りません。参照画像を減らすか小さくするか、画像を小さくします |
| `this image keeps X GiB of its prompt's cache in RAM ...`（400） | 参照画像のキャッシュに RAM が足りません。同じく参照画像を減らすか小さくします |
| `edits need the vision encoder ...`（400） | 編集のためのファイルがありません。setup を `--images` を付けてもう一度実行します |
| `the image engine did not start ...`、`the image engine ended` | 画像のエンジンが起動しないか、止まりました。表示にあるログ（`strata-image.log`）を見てください |
| `Image generation runs the image model at Q6_K instead of Q8_0 ...` | RAM が足りないので、setup が大きさを下げました（[大きさとメモリ](#大きさとメモリ)） |
| `faster with NVIDIA's ... cuDNN for the image engine's attention`（setup） | cuDNN がないので、NVIDIA の GPU の注意を XeStrata のカーネルで計算します。示されたコマンドで入れると速くなります |

## ライセンス

- Qwen-Image-2.1-Turbo と Qwen-Image-2.1 の重みは Qwen Research License で、自由なソフトウェアのライセンスではありません。商用に使う前にそのライセンスを読んでください。
- Mage-Flow の重みは MIT ですが、Microsoft の README は研究のためのモデルで、製品やサービスに使うことは想定していないとしています。
  XeStrata は Mage-Flow の内容のフィルターも透かしも入れていません。
- MageTrail の重みは MIT です。MageTrail は Danbooru と e621 の画像で学習されていて、XeStrata はその内容のフィルターも入れていません。
  VAE は Black Forest Labs が FLUX.2-klein-4B で公開している Flux2 VAE で、Apache 2.0 です（MageTrail のリポジトリにある同じ VAE は FLUX.2-dev の非商用のライセンスなので、setup はそちらを使いません）。
- 文の読み手（Qwen3-VL、AtomicChat と huihui-ai の版）には、それぞれのライセンスが適用されます。
