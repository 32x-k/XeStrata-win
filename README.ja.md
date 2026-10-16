<!--
SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
-->
# XeStrata

[English](README.md) | 日本語

## このブランチについて（Windows と AMD の GPU）

> この節は `feature/windows-r9700-sycl` ブランチで足したものです。区切り線から下は本家 XeStrata の README のままです。

このブランチは、XeStrata を Windows 11 で AMD の GPU を使って動かすための移植です。
intel/llvm（DPC++）を Windows の上で HIP と Level Zero（Intel の GPU）のランタイムを付けてビルドし、XeStrata を free モードでコンパイルします。
本家の更新は取り込みながら進めていて、今は本家の main（上流 Strata 0.1.44.1 を取り込んだもの、2026-10-14）までを含みます。

### していること

- intel/llvm v7.1.1 を Windows でビルドするための修正（`third_party/main/intel-llvm/patches` の 90〜92。本家のパッチと番号が重ならないよう 90 から付けています）
- oneMath の rocBLAS のバックエンドを Windows でビルドし（`third_party/main/oneMath/patches/windows-gnu-driver.patch`）、prefill の密な行列積に使う
- Windows に合わせたエンジンの修正: GPU のメモリの予算（DXGI から LUID で読む）、空き RAM の読み取り、シグナルと異常終了の扱い、ファイルの読み方、rocBLAS に GPU 用のファイルがないときの自前カーネルへの切り替えなど
- 速さの調整: GGUF のエキスパートをそのまま読むネイティブパック、GPU のキャッシュにないエキスパートの RAM 側のコピー（`--resident-experts`）

### 対応している GPU

- Windows では AMD の GPU（HIP）と Intel Arc（Level Zero）で動きます。配布用の zip には、AMD の RDNA2 以降（gfx1030〜gfx1036、gfx1100〜gfx1103、gfx1150、gfx1151、gfx1200、gfx1201）のコードと、Intel の GPU 向けのコード（SPIR-V。起動するときにドライバーがコンパイルします）が入っています。確かめたのは Radeon AI PRO R9700（gfx1201、32 GB）だけです。
- Windows では、NVIDIA の GPU 向けにはビルドしていません（free モードでは使えないため）。
- Linux では本家の XeStrata と同じです（Intel Arc、AMD、NVIDIA）。

### Windows で使える機能

- 導入と起動（配布用の zip の `xestrata-cli.bat`。GPU とモデルの選択、モデルの準備、設定）
- `strata.exe` によるテキストの生成（`--tokens-file`）
- `--serve` と `serve/server.py` による OpenAI 互換の API
- ネイティブパック（`tools/iq_pack.py`）、MTP による推測デコード、`--resident-experts`、rocBLAS による prefill

本家と違うところは次の 3 つです。

- Windows では 7z ではなく zip で配布し、本家の `xestrata.exe`（.NET の画面）の代わりに `xestrata-cli.bat` を使って、端末で導入と起動をします。deb と rpm のパッケージは Linux 用です。
- Windows では、RAM 側のエキスパートのコピーを、HIP から見える GPU が 1 台のときだけ既定でページロックします（RAM に余裕があるとき）。HIP から 2 台以上（内蔵 GPU を含む）見えていると、大きくページロックした後の確保で GPU が落ちることがあるためです。`ONEAPI_DEVICE_SELECTOR=hip:N` で GPU を選ぶと、XeStrata は `HIP_VISIBLE_DEVICES` も同じ GPU に絞ります。
- `--gpu-stages` はネイティブパックでは動きません。

### インストール（Windows、AMD と Intel の GPU）

1. [このフォークの Releases](https://github.com/32x-k/XeStrata-win/releases) から `XeStrata-<版>-windows-x64-amd-intel.zip` をダウンロードして、浅いフォルダー（`C:\XeStrata` など）に展開します。中のパスが長いので、深いフォルダーでは展開に失敗することがあります。
1. `xestrata-cli.bat` を実行します（ダブルクリックでも動きます）。GPU とモデルの選択、ダウンロード、モデルの準備を端末で進め、終わると起動します。次からは実行するとすぐ起動します。オプションは Linux の `xestrata` と同じです（`--help`）。

- **要るもの**: GPU のドライバー（AMD は AMD Software で、HIP のランタイム `amdhip64_7.dll` が一緒に入ります。Intel は Intel のグラフィックスのドライバーで、Level Zero のランタイム `ze_loader.dll` が一緒に入ります）と [Visual C++ の再頒布可能パッケージ](https://aka.ms/vs/17/release/vc_redist.x64.exe)。.NET は要りません。
  Visual C++ の再頒布可能パッケージがなければ、`xestrata-cli.bat` が知らせて止まります。rocBLAS と hipBLASLt は zip に入っています。
- **GPU が 2 台以上のとき**（内蔵 GPU を含む）: setup は選んだ GPU を設定の `ONEAPI_DEVICE_SELECTOR=hip:N` に書き、エンジンが HIP に見せる GPU をそれだけに絞ります。
- **置き場所**: zip のフォルダーには書き込みません。設定は `%APPDATA%\xestrata`、ログなどは `%LOCALAPPDATA%\xestrata`、モデルはデータのフォルダー（既定は `%LOCALAPPDATA%\xestrata\data`）に置きます。
- **更新**: 新しい zip を別のフォルダーに展開して、そちらの `xestrata-cli.bat` を実行します。設定とモデルはそのまま使います。

### ビルド

詳しい手順は [docs/DEVTOOLS.ja.md](docs/DEVTOOLS.ja.md) の「Windows（AMD の GPU、HIP）」にあります。

1. 用意するもの: Visual Studio 2022 Build Tools（C++ のワークロード）、CMake、Ninja、Python、ROCm の Windows 用 SDK（AMD の TheRock の Python の wheel。clang、HIP のランタイム、rocBLAS を含む）、ROCm 7.2 の Linux 用のデバイスライブラリ（`amdgcn/bitcode`）。
2. intel/llvm v7.1.1 を取得し、`third_party/main/intel-llvm/patches` のパッチを番号順に当てます。
3. Visual Studio の `x64 Native Tools Command Prompt` で intel/llvm を構成してビルドします。

   ```bat
   python buildbot\configure.py -o build -t Release --hip --hip-platform AMD ^
     "--cmake-opt=-DCMAKE_INSTALL_PREFIX=<intel-llvm>/install" ^
     "--cmake-opt=-DUR_HIP_ROCM_DIR=<ROCm SDK>" "--cmake-opt=-DUR_HIP_LIB_DIR=<ROCm SDK>/lib" ^
     "--cmake-opt=-DLIBCLC_TARGETS_TO_BUILD=amdgcn--amdhsa" "--cmake-opt=-DSYCL_ENABLE_BACKENDS=hip;level_zero" ^
     "--cmake-opt=-DLLVM_ENABLE_DIA_SDK=OFF" ^
     "--cmake-opt=-DCMAKE_C_FLAGS=/DWIN32 /D_WINDOWS /utf-8" ^
     "--cmake-opt=-DCMAKE_CXX_FLAGS=/DWIN32 /D_WINDOWS /GR /EHsc /utf-8"
   cmake --build build --target deploy-sycl-toolchain
   ```

4. デバイスライブラリの `wchar_size` を Windows に合わせて書き換えます。

   ```bat
   python tools\windows\device_libs_wchar.py <intel-llvm>\build\bin <ROCm 7.2>\amdgcn\bitcode <device-libs>
   ```

5. XeStrata を構成してビルドします（`STRATA_HIP_ARCHS` は GPU のアーキテクチャ。R9700 は gfx1201）。

   ```bat
   cmake -S . -B build\free-win -G Ninja -DCMAKE_BUILD_TYPE=Release ^
     -DCMAKE_C_COMPILER=<intel-llvm>/install/bin/clang.exe ^
     -DCMAKE_CXX_COMPILER=<intel-llvm>/install/bin/clang++.exe ^
     -DSTRATA_LICENSE=free -DSTRATA_HIP_ARCHS=gfx1201 -DSTRATA_CUDA_ARCHS= -DSTRATA_ONEMKL=OFF ^
     -DSTRATA_ROCM_DEVICE_LIBS=<device-libs> -DSTRATA_ROCM_ROOT=<ROCm SDK> ^
     -DSTRATA_ENABLE_ONEMATH=ON -DCMAKE_PREFIX_PATH=<ROCm SDK> -DHIP_PATH=<ROCm SDK> ^
     -DSTRATA_BUILD_TESTS=OFF -DSTRATA_PACKAGE=OFF
   cmake --build build\free-win --target strata onemath_blas_rocblas
   ```

6. ネイティブパックを作ります。

   ```bat
   python tools\iq_pack.py --gguf <model>-00001-of-00002.gguf --out <pack>
   ```

7. 動かすときは、intel/llvm の `install\bin`、ビルドの `bin`（oneMath の DLL）、ROCm の SDK の `bin` を `PATH` に置き、`ONEAPI_DEVICE_SELECTOR=hip:0` で GPU を選びます。

   ```bat
   build\free-win\strata.exe --pack <pack> --native <model>-00001-of-00002.gguf ^
     --ple-gguf <model>-00002-of-00002.gguf --expert-profile data\expert-profile.bin --expert-cache auto ^
     --resident-experts --prefill auto --spec 4 --spec-min-p 0.5 --mtp <MTP の下書き層> ^
     --adapt-every 0 --pcie-frac 0 --tokens-file <prompt.ids> --max-new 256 --stats
   ```

8. テストは、別のフォルダーで `-DSTRATA_BUILD_TESTS=ON` として構成し、全体をビルドして CTest で動かします（5 と同じ引数で、フォルダーだけ変えます）。
   `expert_parity` と `pool_test` はリポジトリ直下の `pack\full` にある AVX-512 向けのパック（`experts.bin`）を読むので、そのパックへのジャンクションを置きます。
   7 と同じ `PATH` と `ONEAPI_DEVICE_SELECTOR=hip:0` で動かします。

   ```bat
   cmake --build build\free-win-test
   mklink /J pack\full <AVX-512 向けのパック>
   cd build\free-win-test && ctest --output-on-failure
   ```

### 今の計測（参考値）

2026-10-14 に、Radeon AI PRO R9700、Ryzen AI 9 HX 370、RAM 28 GB、Windows 11 の PC で測りました。
モデルは Qwen3.8-Flash-Next の Q2_0、引数は上のとおりです（MTP の推測デコードは 4 トークン）。
比べた相手は、XeStrata の上流にあたる Strata v0.1.43.1 の HIP 版です。同じ PC・モデル・プロンプトで、引数は Strata の既定です。
値は tok/s で、2 回測ったものは平均です。

| プロンプト | 出力 | decode（このブランチ） | decode（Strata） | prefill（このブランチ） | prefill（Strata） |
| --- | --- | --- | --- | --- | --- |
| 50 トークン | 300 | 96 | 107 | 125 | 46 |
| 190 トークン | 128 | 90 | 92 | 228 | 101 |
| 2,301 トークン | 200 | 103 | 107 | 522 | 499 |
| 18,801 トークン | 200 | 101 | 107 | 976 | 948 |
| 40 トークン | 2,000 | 84 | 96 | 99 | 37 |

`--serve` に 20 個の質問を続けて送ると、20 個すべてに答え、全体の速さは 93 tok/s でした（Strata は 95 tok/s）。

推測デコードの 1 回の verify にかかる時間は Strata とほぼ同じです（2026-10-16、50 トークンのプロンプトで 22.9 ms と 23.1 ms）。足し算の順序が違うので出力の文章が少し変わり、それによって MTP の下書き（draft）が採用される数も変わるので、decode の差はプロンプトによって変わります。上の表の 50 トークンのプロンプトは、このブランチの下書きの採用が少ない例です。中身の違う 12 個のプロンプトを 1 つのサーバーに続けて送り、300 トークンずつ生成すると、MTP の下書きが採用された数は同じくらいで（2,011 個と 2,016 個。採用率 0.667 と 0.683）、decode はこのブランチが 98.5 tok/s、Strata が 86.4 tok/s でした（12 個すべてでこのブランチが速い）。適応キャッシュの入れ替えを既定（`--adapt-every 4`）にすると、このブランチは 104.5 tok/s でした。

---

1,250 億パラメーターの AI モデルを、Intel Arc、AMD Radeon、NVIDIA のどれかの GPU 1 枚と普通の PC で動かします。

XeStrata は、ふつうはサーバーで動かす大きな AI モデル
**[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** を、自分の PC で動かすためのソフトウェアです。
GPU は Intel Arc、AMD Radeon（RDNA2 以降）、NVIDIA のどれかを 1 枚、OS は Linux を使います。Ubuntu 26.04 と Fedora 44 には deb と rpm のパッケージがあります。
自由ソフトウェアで、Intel と AMD の GPU なら自由ソフトウェアだけでもビルドして動かせます。

> **目次:** [XeStrata とは](#xestrata-とは) · [必要なもの](#必要なもの) · [モデルの選び方](#モデルの選び方) ·
> [インストール](#インストール) · [使い方](#使い方) · [困ったとき](#困ったとき) · [しくみ](#しくみ) ·
> [開発する人へ](#開発する人へ) · [詳しい説明](docs/DETAILS.ja.md)

## XeStrata とは

CUDA で書いたソフトウェアは NVIDIA の GPU でしか動かず、使う人は 1 社の GPU に縛られます。
XeStrata は、オープンな標準の SYCL で書いた 1 つのコードで、Intel・AMD・NVIDIA のどの GPU でも動くことを目指します。

[Strata](https://github.com/Niko1221/Strata)（NVIDIA の GPU 向け）を、Intel の GPU に移したものです。
GPU の計算を CUDA から SYCL と Level Zero に書き直し、Intel の GPU で動くようにしました。
同じコードを、AMD の GPU 向け（free と contrib-llvm のビルド）と NVIDIA の GPU 向け（contrib-llvm のビルド）にもコンパイルします。
モデル、インストールの流れ、画面、API は Strata とほぼ同じです。

Strata との主な違い:

- **Intel Arc 向け**で、AMD と NVIDIA の GPU でも動きます。
- **GPU は 1 枚が基本**です。同じ PC の 2 枚以上の GPU に層を分けて載せることもできます（[MULTIGPU](docs/MULTIGPU.ja.md)）。
  Intel・AMD・NVIDIA のメーカーの違う GPU も組み合わせられます（例: Intel Arc の Battlemage と NVIDIA の Ada。NVIDIA を含むときは contrib-llvm のビルド）。
- **Linux と Windows** で動きます。Windows は Intel と NVIDIA の GPU で、AMD の GPU は Linux だけです（[Windows](#windows)）。WSL 2 は動くかもしれません（プルリクエスト歓迎）。
- **自由ソフトウェアだけでも動きます**（free のビルド、Debian main に入れられる程度に自由な構成。[ソースから作ります](docs/BUILD.ja.md#deb-と-rpm-のパッケージ)）。
  配っている contrib-llvm の版は、自由ソフトウェアでない Intel の oneMKL と NVIDIA の cuBLAS を、入っていれば使います（[インストール](#インストール)）。

## 必要なもの

| | |
| --- | --- |
| GPU | Intel Arc（A シリーズ、B シリーズ）、AMD Radeon（RDNA2 以降、[AMD の GPU](#amd-の-gpu)）か、NVIDIA の GPU（Pascal 以降。Tensor Core のない Pascal は遅い経路で動きます）。VRAM は 12 GB 以上を勧めます（それより少なくても動きますが、遅くなります）。 |
| CPU | x86-64 で AVX2 があるもの。AVX-512（F、BW、VL、VNNI、VBMI）があれば、CPU の計算に AVX-512 を使います。 |
| RAM | モデルの大きさによって 32〜62 GB（[モデルの選び方](#モデルの選び方)）。 |
| ディスク | モデルに 60〜110 GB ほど、ほかに MTP 層に約 6 GB。SSD（NVMe）を強く勧めます。 |
| OS | Linux（WSL 2 は動くかもしれません。プルリクエスト歓迎）。パッケージは Ubuntu 26.04 と Fedora 44 向けです。ほかのディストリビューションでは[ソースから入れます](docs/BUILD.ja.md#ソースから入れる)。Windows 10 と 11（x64）は 7z で配ります（[Windows](#windows)）。 |
| BIOS | 単体の GPU では Above 4G Decoding と Re-Size BAR を有効にし、CSM を無効にします。 |

- **行列エンジン**: Intel の XMX、AMD の WMMA（RDNA3 以降）、NVIDIA の Tensor Core を使い、ない GPU では DP4a などの整数の内積の命令で計算します。
  どれを使うかは、GPU が報告する能力で決めます。行列エンジンがないと prefill が遅くなりますが、動きます。
- **確かめた GPU**: 開発と確認は Arc Pro B70（Xe2、32 GB）で行っています。NVIDIA は RTX 4070 と RTX 3070、AMD は RX 9060 XT（RDNA4）で確かめています。
  Arc A380 では計算の正しさだけを確かめ、モデルは動かしていません。Windows は RTX 3050 Laptop（4 GB）で確かめています。
  小さな GPU は、B70 で使えるメモリと XMX を制限して確かめています。
- **CPU 内蔵のグラフィックス**: 起動と prefill までを確かめています。メモリが RAM と共有で、
  XMX のないものが多いので、遅くなります。

### AMD の GPU

AMD の GPU では、prefill の密な行列積をディストリビューションの ROCm の rocBLAS と hipBLASLt で計算します。
そのため、XeStrata が動くのは、ディストリビューションの rocBLAS がコードを持っている GPU だけです。

| GPU | gfx | Ubuntu 26.04 | Fedora 44 |
| --- | --- | :-: | :-: |
| RX 6800 / 6900 | gfx1030 | ○ | ○ |
| RX 6700 | gfx1031 | × | ○ |
| Ryzen の内蔵 GPU（RDNA2） | gfx1035、gfx1036 | × | ○ |
| RX 7900 | gfx1100 | ○ | ○ |
| RX 7800 / 7700 | gfx1101 | ○ | ○ |
| RX 7600 | gfx1102 | × | ○ |
| Ryzen の内蔵 GPU（780M など） | gfx1103 | × | ○ |
| Ryzen AI の内蔵 GPU | gfx1150 | × | ○ |
| Ryzen AI Max の内蔵 GPU | gfx1151 | ○ | ○ |
| RX 9060 | gfx1200 | ○ | ○ |
| RX 9070 | gfx1201 | ○ | ○ |

- **RX 6700 と RX 7600 では Fedora 44 を使ってください。** Ubuntu 26.04 の rocBLAS には、これらの GPU のコードがありません。
- RX 6600（gfx1032）と RX 6500 / 6400（gfx1034）は、どちらのディストリビューションの rocBLAS にもコードがないので動きません。
- gfx1152 と gfx1153（Ryzen AI の一部の内蔵 GPU）は、Fedora 44 の rocBLAS にはコードがありますが、XeStrata が使う intel/llvm 7.1.1 にそのターゲットがないので動きません。
- 自分の GPU の gfx は、`xestrata --check` が GPU の名前の後ろに出します。
- 表は、どちらのディストリビューションも ROCm 7.1 の rocBLAS のものです。確かめたのは RX 9060 XT（gfx1200）だけです。

## モデルの選び方

| モデル | パラメーター | RAM の目安 | VRAM | 向いていること |
| --- | --- | ---: | ---: | --- |
| **[Qwen3.8-Flash-Next](#qwen38-flash-next)** | 1,250 億 | 32〜62 GB | 12 GB 以上を勧めます | いちばん賢い。RAM の多い PC 向け |
| **[Ornith-1.5](#ornith-15beta)**（BETA） | 350 億（1 トークンで 30 億） | 22〜80 GB | 4 GB から | RAM の少ない PC 向け |
| **[llm-jp-4.1](#llm-jp-41beta)**（BETA） | 320 億（1 トークンで 30 億） | 30〜73 GB | 4 GB から | 日本語と英語の思考するモデル |
| **[Qwen3.6-35B-A3B](#qwen36-35b-a3bbeta)**（BETA） | 350 億（1 トークンで 30 億） | 31〜46 GB | 4 GB から | 思考するモデル。RAM の少ない PC 向け |

BETA の 3 つは、画像、2 枚以上の GPU、Intel の CPU 内蔵の GPU にまだ対応していません（[詳細](docs/DETAILS.ja.md#qwen38-flash-next-以外のモデルornith-15llm-jp-41qwen36-35b-a3b)）。

迷ったら、RAM が 48 GB 以上の PC では Qwen3.8-Flash-Next の **IQ2_XS** を選んでください。
RAM がそれより少ない PC では、Qwen3.8-Flash-Next の Coder（32 GB）か Ornith-1.5 が選べます。
あとから別のモデルを足すときは `xestrata --setup` を実行します。

### Qwen3.8-Flash-Next

**版と大きさ**（大きさは、同じモデルをどれだけ圧縮するかの違い）:

| 版 | 大きさ | ダウンロード | RAM の目安 | 速さ | 品質 |
| --- | --- | ---: | ---: | --- | --- |
| 元のモデル、Swift 1.5 | **Q2_0** | 66 GB | 48 GB | いちばん速い | 良い |
| 元のモデル、Swift 1.5 | **IQ2_XS** | 68 GB | 48 GB | 速い | より良い（**おすすめ**） |
| 元のモデル、Swift 1.5 | **IQ3_XXS** | 76 GB | 60 GB | 遅め | とても良い |
| 元のモデル、Swift 1.5 | **IQ3_S** | 84 GB | 62 GB | 遅い | 元のモデルと同等 |
| Coder | **IQ1_M** | 58 GB | 32 GB | 速い | コードでは良い、ほかは弱い |
| Unsloth | **UD-IQ4_XS** | 94 GB | 48 GB | RAM が約 80 GB より少ないと遅い | IQ3_S と UD-Q4_K_XL の間 |
| Unsloth（実験的） | **UD-Q4_K_XL** | 111 GB | 48 GB | いちばん遅い | いちばん良い |

モデルのエキスパート（[しくみ](#しくみ)）は、すべて RAM に置きます。RAM が目安より少なくても、GPU の VRAM が大きければ
**省 RAM モード**で動きます。このモードでは、エキスパートを RAM にコピーせずにモデルのファイルから読みます
（[詳細](docs/DETAILS.ja.md#ram-がモデルより少ないとき省-ram-モード)）。
Unsloth の 2 つは、RAM に入りきらないエキスパートを答えている間に SSD から読みます。

**版:**

- **Qwen3.8-Flash-Next**: 元のモデル。
- **[Swift 1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)**: UkisAI による調整版です。
  答える前の思考が短く、答えが早く出ます。品質はほぼ同じです。このモデル自身のライセンスに従います。
- **[Coder](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF)**: ISTA-DASLab によるコード向けの版です。
  各層 512 個のエキスパートのうち、コードで使う 256 個だけを残しています。エキスパートが半分なので、コード以外や、英語以外の言語では弱くなります。
- **[Unsloth の UD-IQ4_XS](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)**: 元のモデルの約 4 ビットの i-quant です。
  エキスパートが 59.5 GB あり、RAM が約 80 GB 以上あればすべて RAM に置けます（[詳細](docs/DETAILS.ja.md#unsloth-の-ud-iq4_xs)）。
- **[Unsloth の UD-Q4_K_XL](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)**（実験的）: 元のモデルの 4 ビット版です。
  エキスパートが 77 GB あり、2〜3 ビットの大きさより遅くなります（[詳細](docs/DETAILS.ja.md#unsloth-の-ud-q4_k_xl実験的)）。

OrcaRouter の Flash-Next Uncensored IQ3_XXS は、setup のメニューにはありません。
変換の手順は [docs/ORCA.ja.md](docs/ORCA.ja.md) にあります。

### Ornith-1.5（BETA）

[Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF)（Ornith AI、MIT）: Qwen3.5 系の MoE で、文脈は 256K までです。
公式の版（Q4_K_M〜BF16）は、モデルの MTP の層で下書きを作り、decode を速くします。

| 大きさ | ダウンロード | RAM の目安 | VRAM |
| --- | ---: | ---: | ---: |
| Q4_K_M | 21.7 GB | 30 GB | 4 GB から |
| Q5_K_M | 25.3 GB | 34 GB | 6 GB から |
| Q6_K | 29.2 GB | 38 GB | 6 GB から |
| Q8_0 | 37.8 GB | 46 GB | 6 GB から |
| BF16 | 71.1 GB | 80 GB | 8 GB から |
| GSQ-RCO-3BIT | 13.0 GB | 22 GB | 4 GB から |
| GSQ-RCO-3.5BIT | 15.2 GB | 24 GB | 6 GB から |

GSQ-RCO の 2 つは、pfeifferj による[コミュニティの再現](https://huggingface.co/pfeifferj/Ornith-1.5-35B-A3B-GSQ-RCO-GGUF)で、MTP の層がありません。

### llm-jp-4.1（BETA）

[llm-jp-4.1-32b-a3b-thinking](https://huggingface.co/llm-jp/llm-jp-4.1-32b-a3b-thinking-gguf)（LLM-jp、Apache-2.0）: 日本語と英語の思考するモデルで、文脈は 256K までです。

| 大きさ | ダウンロード | RAM の目安 | VRAM |
| --- | ---: | ---: | ---: |
| Q4_K_M | 21.5 GB | 30 GB | 4 GB から |
| BF16 | 64.3 GB | 73 GB | 8 GB から |

### Qwen3.6-35B-A3B（BETA）

[Qwen3.6-35B-A3B](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF)（Qwen、Apache-2.0。GGUF は Unsloth が MTP の層付きで作ったもの）: Qwen3.5 系の思考する MoE で、文脈は 256K までです。
モデルの MTP の層で下書きを作り、decode を速くします。

| 大きさ | ダウンロード | RAM の目安 | VRAM |
| --- | ---: | ---: | ---: |
| UD-Q4_K_XL | 22.9 GB | 31 GB | 4 GB から |
| UD-Q6_K_XL | 32.6 GB | 41 GB | 6 GB から |
| Q8_0 | 37.8 GB | 46 GB | 6 GB から |

BETA の 3 つの RAM の目安は、ダウンロードの大きさに約 8 GB を足したものです。
RAM がそれより少ないと、エキスパートを RAM に読み込まずにモデルのファイルから読むので、遅くなります。

## インストール

### Podman

ホストに入れるのは Podman と podman-compose だけです。XeStrata は Fedora 44 のコンテナの中に、Releases の rpm から入ります（rootless）。

```bash
git clone https://github.com/MistVVK/XeStrata && cd XeStrata/container
mkdir -p ~/.config/xestrata ~/.local/state/xestrata ~/.local/share/xestrata
podman compose build                         # 最新のリリースと GPU のライブラリを入れたイメージを作ります
podman compose run --rm xestrata --setup     # 質問に答えてモデルを取ってきます
podman compose up -d                         # http://127.0.0.1:8095（止めるときは podman compose down）
```

- **GPU**: Intel はこのままです。AMD は `-f compose.yaml -f compose.amd.yaml`、NVIDIA は `-f compose.yaml -f compose.nvidia.yaml` を
  `podman compose` の後に付けます（`export COMPOSE_FILE=compose.yaml:compose.nvidia.yaml` でも同じです）。
  NVIDIA はホストに NVIDIA Container Toolkit の CDI の設定が要ります（[docs/BUILD.ja.md](docs/BUILD.ja.md#podman)）。
  コンテナの中は cuda13.4 なので、Pascal と Volta（P100、P40、GTX 10 シリーズ、V100 など）は使えません。
- **ホストに要るもの**: GPU のドライバーと、自分が render グループに入っていること（`sudo usermod -aG render $USER`）です。
- **置き場所**: 設定・ログ・モデルのファイルはホストの同じ場所（下のパッケージと同じ）を使うので、パッケージ版とコンテナを入れ替えても、そのまま使えます。
- **ほかのコマンド**: `podman compose run --rm xestrata --bench`、`--calibrate`、`chat` のように、`xestrata` の後に付けるものを並べます。
- **更新**: `podman compose build --pull --no-cache` で最新のリリースを入れ直し、`podman compose up -d` で起動し直します。

### deb と rpm のパッケージ

1. GPU に合うパッケージを下の表から選んで、[GitHub の Releases](https://github.com/MistVVK/XeStrata/releases) から取ってきます（`SHA256SUMS` も一緒に）。
1. `sha256sum -c SHA256SUMS --ignore-missing` で確かめます。
1. 入れます: `sudo apt install ./xestrata-contrib-cuda13.1_*.deb`（Ubuntu）か `sudo dnf install ./xestrata-contrib-cuda13.4-*.rpm`（Fedora）。
1. 端末で `xestrata --setup` を実行します。足りない GPU のライブラリと render グループは、入れるコマンドを表示して止まるので、それを実行してからもう一度 `xestrata --setup` を実行します。

> [!NOTE]
> NVIDIA の GPU を使わない場合は、どちらの CUDA のバージョンを使っても変わりありません。

| GPU | Ubuntu 26.04 | Fedora 44 |
| --- | --- | --- |
| **おすすめ**: Intel、AMD（RDNA2 以降）、NVIDIA（Turing 以降） | `xestrata-contrib-cuda13.1` | `xestrata-contrib-cuda13.4` |
| Intel、AMD（RDNA2 以降）、NVIDIA の Pascal と Volta（P100、P40、GTX 10 シリーズ、V100 など） | `xestrata-contrib-cuda12.4` | なし（[ソースから](docs/BUILD.ja.md#ソースから入れる)） |

- パッケージは一度に 1 つだけ入ります。CUDA 13 の版には Pascal と Volta のコードがなく、これらを扱う NVIDIA のドライバーも 580 の系列（CUDA 12.4 に対応）までです。
- 自由ソフトウェアだけで動かす版（`xestrata-free`、Intel と AMD の GPU）は Releases にはありません。[docs/BUILD.ja.md](docs/BUILD.ja.md#deb-と-rpm-のパッケージ) の手順で作れます（`tools/package/build.sh ubuntu26.04 free` など）。

`xestrata --setup` の質問は、Enter を押せばおすすめの選択になります。

- **モデルと大きさ**: [モデルの選び方](#モデルの選び方)を見てください。
- **文脈の長さ**: 一度に覚えておける文章の量です。GPU に合った長さを勧めます。
- **画像**: 画像も読ませるかどうか。
- **実験的な速度向上用の射影**: 答え方が変わる実験的な機能で、既定ではオフです。使う前に
  [説明](docs/DETAILS.ja.md#実験的な速度向上用の射影実験的既定はオフ)を読んでください。

`xestrata` は、GPU と RAM と CPU を確かめ、モデルをダウンロードして起動します。
ブラウザで `http://127.0.0.1:8095` が開きます。

- **時間**: 初回はダウンロード（60〜110 GB）に時間がかかります。途中で止めても、次は続きから始まります。
  GPU のコードは初めて使うときにドライバーがコンパイルするので、最後にモデルを一度動かしてそれを済ませます。
- **起動中は PC が重くなります**: モデルを起動すると、RAM に 23〜50 GB を読み込みます。初回はとくに時間がかかり、
  1〜3 分ほど PC の反応が遅くなることがあります。ウィンドウを閉じずに待ってください。
- **置き場所**: モデルのファイルは `~/.local/share/xestrata`（`--data-dir` で変えられます）、設定は `~/.config/xestrata`、
  ログは `~/.local/state/xestrata` に置きます。

**次からは** `xestrata` を実行すると、すぐに起動します。止めるときはそのウィンドウを閉じます。

**ログインしている間ずっと動かす**: `systemctl --user enable --now xestrata` で、最後に設定したモデルを起動しておきます
（ブラウザは開きません）。先に `xestrata` で一度設定しておきます。

**更新**: 新しいリリースが出ると、起動したときと、動いている間は 6 時間ごとに調べて、端末とチャットのページの About に表示します（`xestrata --check-update` でも調べられます）。
新しいパッケージを [Releases](https://github.com/MistVVK/XeStrata/releases) から取ってきて、インストールと同じく `sudo apt install ./<ファイル>.deb` か `sudo dnf install ./<ファイル>.rpm` で入れます。
モデルの設定は次の起動で更新します。

**自分の PC に合わせる**: `xestrata --calibrate` は、エンジンの設定のいくつかをこの PC で測り、速いものを残します
（5〜10 分ほど）。

**速さを測る**: `xestrata --bench` は、入れた IQ3_XXS（ほかのモデルは `--model IQ2_XS` などで指定）のこの PC での速さと、どこに時間がかかっているかを測り、
データフォルダの `records/bench-<日時>/` に記録を書きます（40 分ほど。動いている xestrata は先に止めます）。
各計測は 2 回ずつで、`--bench-rounds 3` のように回数を変えられます（1 回あたり 20 分ほど増えます）。
`--bench-verbose` はカーネルごとの時間や GPU の温度も取ります。`--original-strata v0.1.41` を足すと、そのタグの upstream Strata を
git で取ってきて、同じモデルで交互に測ります。記録はどこにも送りません。報告に貼るときは `summary.md` の中身を確かめてから貼ります。
測るものと記録の中身は [docs/BENCH.ja.md](docs/BENCH.ja.md) にあります。

**別のパッケージに替える**: Ubuntu では入れたい方を `sudo apt install` すると入れ替わります。
Fedora では `sudo dnf swap xestrata-contrib-cuda13.4 ./xestrata-free-*.rpm` のように、入っているものと入れたいものを指定します。

**消す**: `xestrata --remove-data` でモデルのファイル・設定・ログを消してから（大きさを見せて確かめます）、パッケージを消します。

```bash
xestrata --remove-data
sudo apt purge xestrata-contrib-cuda13.1     # Ubuntu（入れたパッケージの名前）
sudo dnf remove xestrata-contrib-cuda13.4    # Fedora（同じ）
```

`xestrata --remove-data` をせずにパッケージを消すと、ホームのモデルのファイルと設定は残ります。

ソースから `./setup.sh` で入れる方法は [docs/BUILD.ja.md](docs/BUILD.ja.md#ソースから入れる) にあります。

### Windows

1. [GitHub の Releases](https://github.com/MistVVK/XeStrata/releases) から `XeStrata-<版>-windows-x64.7z` を取ってきて（`SHA256SUMS` も一緒に）、好きなフォルダーに展開します。
1. `xestrata.exe` を起動します。導入（GPU とモデルの選択、ダウンロード）と起動を画面で進めます。端末で使うときは `xestrata-cli.exe`（Linux の `xestrata` と同じ使い方）です。

- **要るもの**: [.NET 10 のランタイム](https://dotnet.microsoft.com/download/dotnet/10.0)と [Visual C++ の再頒布可能パッケージ](https://aka.ms/vs/17/release/vc_redist.x64.exe)、GPU のドライバー。
  Visual C++ の再頒布可能パッケージがなければ、`xestrata.exe` が知らせて止まります。
- **GPU**: Intel と NVIDIA です（NVIDIA は Pascal 以降、cuda12.4 のパッケージと同じコード）。AMD の GPU は Windows ではまだ使えません。
- **置き場所**: 7z のフォルダーには書き込みません。設定は `%APPDATA%\xestrata`、ログなどは `%LOCALAPPDATA%\xestrata`、モデルはデータのフォルダー（既定は `%LOCALAPPDATA%\xestrata\data`）に置きます。
- **更新**: 新しい 7z を別のフォルダーに展開して、そちらの `xestrata.exe` を起動します。設定とモデルはそのまま使います。

## 使い方

- **ブラウザ**: `http://127.0.0.1:8095`（モデルが起動すると自動で開きます）。**Chat**、モデルと GPU・CPU・RAM の様子を
  見る **Monitor**、設定とアドレスを見る **About** があります。
- **端末で会話**: `xestrata chat`
- **アプリやコーディングエージェント**: 「OpenAI 互換」の接続先として、ベース URL **`http://127.0.0.1:8095/v1`** を指定します。
  API キーとモデル名は何でも構いません。Anthropic の API を使うアプリは `http://127.0.0.1:8095/v1/messages` です。
- **思考の深さ**: モデルは答える前に考えます。深さは **Off、Low、Medium、High** から選べます。チャットの画面のメニュー、
  `chat.py` の `/think low`、アプリの「reasoning effort」の設定で変えられます。Off がいちばん速く、難しい質問には High が向きます。
- **画像**: チャットの画面では **Picture** を押します。`chat.py` では `/image <パス>` と入力します。アプリでは添付します。
- **スマートフォンや別の PC から**: `xestrata --host 0.0.0.0 --api-key <秘密の文字列>` を実行し、
  サーバーのウィンドウに出るアドレスを開きます（[詳細](docs/DETAILS.ja.md#ストリーミングと接続)）。

**知っておくと良いこと**: 要求は 1 つずつ処理します（複数を同時に処理するには `"parallel": 2`、[BATCHING](docs/BATCHING.ja.md)）。会話の最初のメッセージはすべて読み込みますが、
そのあとは会話を覚えておき、新しく足された部分だけを読むので、続きの返事はすぐに始まります。

## 困ったとき

**最初の起動で PC が固まった、とても遅くなった**
起動中、とくに初回はよくあります。モデルを RAM に読み込み、GPU に置くエキスパートの量を決めているところです。
ウィンドウを閉じずに待ってください。10 分たっても固まったままなら、PC を再起動し、ほかのプログラム
（とくにブラウザ）を閉じてからやり直してください。何度も起きるなら、小さい大きさ（Q2_0 か IQ2_XS）を選んでください。

**ダウンロードやインストールの途中で止まった**
`xestrata` をもう一度実行してください。止まったところから続けます。

**GPU が見つからない、使えないと言われた**
次の 3 つを確かめてください。

- Intel の GPU のランタイム（Level Zero のドライバー）、AMD の GPU なら ROCm のライブラリ（HIP、rocBLAS、hipBLASLt）、
  NVIDIA の GPU ならそのドライバー（`nvidia-smi` で見えるか）が入っているか確かめてください。
  NVIDIA の GPU は contrib-llvm の版でだけ使えます。AMD の GPU は、[対応する GPU](#amd-の-gpu)かどうかも確かめてください。
- 自分のユーザーで GPU のデバイス（`/dev/dri/renderD*`、AMD の GPU では `/dev/kfd` も）を開けるか確かめてください（`render` グループ）。
- 単体の GPU が OS から見えないときは、BIOS の Above 4G Decoding と Re-Size BAR を確かめてください。

**XMX が使えないと言われた**
今のコンパイラでは、その GPU の行列エンジンが使えません。setup が出す選択肢から選んでください。
XMX なしでも動き、prefill が遅くなります（約 1.6 倍の時間）。

**起動時に VRAM の確保が拒否された（`alloc_device: ... refused`）**
B70 では、まれに起きることがあります。もう一度起動すれば、たいていは動きます。

**ポート 8095 はすでに使われていると言われた**
XeStrata がすでに動いています。そのウィンドウを探してください。ほかのプログラムが使っているなら、
`xestrata --port 8081` のように別のポートを選べます。

**とても遅く、ディスクのランプが点きっぱなし**
RAM が足りていません。ほかのプログラムを閉じるか、小さい大きさ（Q2_0 か IQ2_XS）を選んでください。

**答えの途中で「the engine stopped unexpectedly」と出た**
たいていは RAM 不足です（Linux がエンジンを止めます）。同じメッセージをもう一度送れば、エンジンは自動で起動し直します。
何度も起きるなら、ほかのプログラムを閉じるか、小さい大きさを選んでください。

**プロンプトが文脈の長さを超えると言われた**
会話が、選んだ文脈の長さより長くなっています。新しい会話を始めるか、`xestrata --setup` で長い文脈を選び直してください。

**解決しないとき**
[詳しい対処の表](docs/DETAILS.ja.md#困ったとき)を見てください。
端末を操作できる LLM エージェントには、[復旧と原因調査の指示書](docs/DIAGNOSE.ja.md)と症状を渡せます。
ソース版と deb/rpm 版の両方に対応し、復旧できた場合も、詰まった場合も、原因と根拠を報告します。
開発者の対応が必要なら、秘密や個人情報を除いた報告を添えて [issue](https://github.com/MistVVK/XeStrata/issues) を立ててください。
エンジンのログの場所は、実際に使っているモデルの設定の `log` で確認できます。

## しくみ

このようなモデルは、ふつう数百 GB の GPU のメモリを持つサーバーで動かします。XeStrata は、**PC 全体で仕事を分け合う**ことで、
1 枚の GPU に収めています。

- **モデルは 24,576 個の小さな専門家（エキスパート）の集まり**で、1 語を書くのに使うのはそのうち 10 個だけです。
  そのため、すべてを一度に GPU に置く必要はありません。
- **GPU** は、どの語にも必要な計算と、よく使われる数千個のエキスパートを受け持ちます。どれがよく使われるかは、
  使っている間も学び続けます。
- **RAM** にはすべてのエキスパートを置きます。GPU にないエキスパートが要るときは、**CPU** がそれを計算します。
  GPU と同時に計算するので、互いに待ちません。
- **SSD** には大きな表を置き、1 語ごとにその数行だけを読みます。
- **推測してから確かめる**: モデルに組み込まれた小さな助手が次の数語を推測し、大きなモデルがそれをまとめて確かめます。
  合っているものは残し、次の 1 語は自分で書きます。決めるのはいつも大きなモデルなので、答えの品質は変わりません。
- **長い文章は大きな塊で読みます**（一度に最大 32,768 トークン）。

行列積は、Intel の GPU では XMX、AMD の GPU では WMMA（RDNA3 以降）、NVIDIA の GPU では Tensor Core で計算し、
行列エンジンのない GPU では DP4a などの整数の内積の命令を使います。
各部分の詳しい説明は [docs/DETAILS.ja.md](docs/DETAILS.ja.md#しくみ)、Intel の GPU での実装と確かめたことは
[docs/XE.ja.md](docs/XE.ja.md) にあります。

## 開発する人へ

報告やプルリクエストを送る前に、[CONTRIBUTING.ja.md](CONTRIBUTING.ja.md)（型番に依存しないこと、ライセンス、GPL 第 7 条の追加の許可、取り込む変更の条件、報告の仕方）を読んでください。

- **ビルド**: ソースから入れる方法、ビルドの 3 つの方式（free、contrib-llvm、contrib-icpx）、setup の振る舞いと要るパッケージ、
  deb と rpm のパッケージの作り方は [docs/BUILD.ja.md](docs/BUILD.ja.md) にあります。
  開発に使う道具（リント、Intel SDE、GPU のプロファイラー）は [docs/DEVTOOLS.ja.md](docs/DEVTOOLS.ja.md) にあります。
- **速さの測り方**: `xestrata --bench`、upstream 追従の計測、リリースの表の測り方は [docs/BENCH.ja.md](docs/BENCH.ja.md) にあります。
- **守ること**: 機器に依存しないこと（GPU と CPU が報告する能力で経路を選ぶ）、リント（`tools/lint/run.sh`）、
  テストの実行、ライセンスの表示は [AGENTS.md](AGENTS.md) にまとめてあります。
- **取り込む変更**: 誰が書いたものでも（自分のコミット、プルリクエスト、upstream からの移植）、intel/llvm の free と contrib-llvm の両方式でビルドと試験をしたもので、
  機器への依存ができるだけ少ないもの（CUDA などメーカーの API に頼らない、1 枚のカードで測った値を決め打ちしない）だけを入れます。
  そうでないものはそのままは入れず、参考にして XeStrata のコードとして書き直します（[AGENTS.md](AGENTS.md#what-goes-in)）。
  プルリクエストには、ビルドした方式（free、contrib-llvm、contrib-icpx）と、動かした GPU と CPU を書き、確かめていないことには `unverified` と付けてください。
- **upstream の取り込み**: Strata 0.1.42 までの変更を、Windows の実装を除いて Xe に移しています。
  速くするための経路は、B70・RTX 4070・RX 9060 XT で測って速くなったものを移します（[docs/XE.ja.md](docs/XE.ja.md#strata-0142-の取り込み)）。
- **README と docs**: 日本語版（README.ja.md、docs/*.ja.md）を先に書き、英語版（README.md、docs/*.md）はその訳です。

## クレジット

- **元のソフトウェア**: Niko1221 と Strata の貢献者による [Strata](https://github.com/Niko1221/Strata)。
- **モデル**:
   - Qwen チームの [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)。
   - [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) による圧縮版と、
     [Coder](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF)。
   - UkisAI による [Swift 1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)。
   - [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) による UD-IQ4_XS と UD-Q4_K_XL。

  モデルのファイルには、それぞれのライセンスが適用されます。
- **使っている部品**:
   - [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp)（MIT）の一部。
   - [oneMath](https://github.com/uxlfoundation/oneMath)（Apache-2.0）: 密な行列積（contrib-llvm と contrib-icpx、free では AMD の GPU）。
     CMake が XeStrata の変更の入ったフォーク（[MistVVK/oneMath](https://github.com/MistVVK/oneMath)、同じライセンス）から取得します。
   - [intel/llvm](https://github.com/intel/llvm) の DPC++（Apache-2.0 WITH LLVM-exception）: free と contrib-llvm のコンパイラ。
     XeStrata の修正（`third_party/main/intel-llvm/patches/`、同じライセンス）を当ててビルドし、パッケージにはその SYCL の実行時を入れます。
   - AMD による [ROCm のデバイスライブラリ](https://github.com/ROCm/llvm-project/tree/rocm-7.2.4/amd/device-libs)（University of Illinois/NCSA）:
     intel/llvm と一緒にビルドし、AMD の GPU 向けのコードがリンクします。
     そのビルドに渡す機能の指定は、Debian の [rocm-llvm](https://salsa.debian.org/rocm-team/rocm-llvm) のパッケージ（Expat）の `debian/rules` から写しました。
   - 画面のフォント [Outfit](https://github.com/Outfitio/Outfit-Fonts)（SIL Open Font License 1.1）。
- **参考にした考え方**:
   - Inco AI と Splash の貢献者による [Splash](https://github.com/incoai/splash)。
   - Neroued による [ninfer](https://github.com/Neroued/ninfer)。
   - syv.ai と HyperQwen の貢献者による [HyperQwen](https://github.com/syv-ai/HyperQwen)。
   - mw00 と Project Maya の貢献者による [Project Maya](https://github.com/mw00/project-maya)（VRAM にないエキスパートを CPU と PCIe に分ける数の決め方）。
   - maxfridbe による [nextsycl](https://github.com/maxfridbe/nextsycl)（Qwen3.5 系のモデルの decode でカーネルをまとめるやり方）。
- 詳しくは [docs/DETAILS.ja.md](docs/DETAILS.ja.md#クレジットとライセンス) にあります。

## ライセンス

XeStrata（著作権は MistVVK と XeStrata の貢献者）は自由ソフトウェアで、全体を
[GNU Lesser General Public License の第 3 版以降](COPYING.LESSER)（`COPYING.LESSER` と `COPYING`）で配布します。

- **メーカーの計算ライブラリと OS のランタイムとのリンク**: GPU や CPU のメーカーが自社の製品向けに出している計算ライブラリとその実行時ライブラリ
  （oneMKL や Intel oneAPI のコンパイラの実行時ライブラリ、cuBLAS や CUDA のランタイムとドライバなど）、
  OS とその標準のコンパイラの C・C++ の実行時ライブラリ（Microsoft の Universal C Runtime や Visual C++ の実行時ライブラリなど）とリンクしたものも配布できます。
  GPL 第 3 版の第 7 条による追加の許可で、文面は [NOTICE](NOTICE) にあります。
- **Strata と ggml から来たコード**: MIT ライセンスの Strata と ggml / llama.cpp から来たコード（`src/` に書き写したカーネルを含む）は、
  MIT のままです。XeStrata が変えていない Strata のファイルは MIT だけで、XeStrata が手を入れたファイルや書き写したコードを含むファイルは、
  XeStrata の変更とファイル全体が LGPL です（ヘッダーは `LGPL-3.0-or-later AND MIT`）。著作権表示と許諾表示は [NOTICE](NOTICE) にあります。
  Debian の rocm-llvm のパッケージ（Expat）から `tools/intel_llvm_build.py` に写した部分も同じです。
- **例外**: `third_party/` の次のものは、元のライセンスのままです。プロジェクトごとのフォルダーに、ライセンスの文書と一緒に置いています。
   - intel/llvm への XeStrata の修正のパッチ（`third_party/main/intel-llvm/`）: Apache-2.0 WITH LLVM-exception。
   - oneMath への XeStrata の変更（フォーク、`third_party/main/oneMath/`）: Apache-2.0。
   - ggml の `ggml-common.h`（`third_party/main/ggml/`、変えていない写し）: MIT。
   - 画面のフォント Outfit（`third_party/main/outfit/`）: SIL Open Font License 1.1。
   - `third_party/nonfree/`: 自由ソフトウェアでないもの。元のモデルのチャットテンプレートと、実験的な速度向上用の射影のベクトルで、
     どちらも Qwen Community License 1.0 です。
- **`third_party/nonfree/` がなくても**、XeStrata はビルドでき、動きます。パッケージには入れていません。
- **パッケージに入れているほかのもの**: intel/llvm の SYCL の実行時（Apache-2.0 WITH LLVM-exception）、llama.cpp の gguf-py（MIT）、
  oneMath（Apache-2.0）を、それぞれのライセンスの文書と一緒に入れています。
  AMD の GPU 向けのコードは ROCm のデバイスライブラリ（NCSA）の一部を含むので、そのライセンスの文書も入れています。
  自由ソフトウェアでないもの（oneMKL、cuBLAS、NVIDIA のドライバー）は入れていません。
- **モデル**: このリポジトリには含みません。モデルのファイルには、それぞれのライセンスが適用されます。
