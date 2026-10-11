<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# Windows 移植の進捗

[English](windows-port-progress.md) | 日本語

この文書は、Windows 11 と AMD Radeon AI PRO R9700（gfx1201）で XeStrata の SYCL エンジンを動かす作業を記録します。実機で確認していない項目を完了とはしません。

## 完了条件

- [x] Windows 11 ネイティブ実行ファイルをビルドする
- [x] R9700（gfx1201）を SYCL ランタイムから認識する
- [x] R9700 で SYCL カーネルを実行する
- [x] XeStrata で実モデルを読み込む
- [x] prefill を正常に終える
- [x] decode を正常に終える
- [x] 意味のあるテキストを生成する
- [x] 実行ログで GPU の使用を確認する

## 2026-10-11: ツールチェーンの確認

使用したリポジトリは `e32b8b0594095352365db0e140a68d5bf56d8b0f` です。作業開始時のワークツリーに変更はありませんでした。

実機で次を確認しました。

- Visual Studio Build Tools 2022 17.14、MSVC v143、Windows SDK 10.0.28000.0
- CMake 4.4、Ninja
- Strata の非公開 venv に入っていた TheRock ROCm 7.1 系（AMD clang 24.0.0git）
- `hipInfo.exe` は R9700 を `gfx1201`、31.86 GiB、wave 32 として認識した
- 同じランタイムは内蔵 Radeon 890M（gfx1150）も認識した。初期移植では R9700 を名前ではなく列挙順と能力から明示して使う

AdaptiveCpp の develop nightly `25.10.0+git.0ea863a.20261009`（LLVM 20.1.8）を `.tools/` に展開しました。この配布物は Windows の HIP runtime backend を含んでいませんでした。AdaptiveCpp の同じ revision から `rt-backend-hip.dll` を次の構成でビルドしました。

- ROCm: `<ROCM_SDK_ROOT>`
- `WITH_ROCM_BACKEND=ON`
- `ACPP_COMPILER_FEATURE_PROFILE=none`
- ROCm 7.1 の `hipError_t` と AdaptiveCpp の初期化の型差を許すため、外部ツールのビルドだけに `-Wno-c++11-narrowing` を指定

`acpp-info.exe` にこの backend を読み込ませ、SYCL の HIP backend から次を実機で確認しました。

- device 0: `AMD Radeon AI PRO R9700`、architecture `gfx1201`
- 32 compute units、32,768 MiB 弱の global memory、64 KiB local memory
- subgroup size 32
- device USM、host USM、shared USM
- execution timestamp

これはデバイス列挙の成功であり、カーネル実行の成功ではありません。

## 2026-10-11: R9700 での最小 SYCL カーネル実行

まず nightly と同じ LLVM 20 revision `87f0227cb60147a26a1eeb4fb06e3b505e9c7261` を `X86;AMDGPU` と AdaptiveCpp を組み込んでビルドしました。この clang は AMDGPU target を持ちますが、TheRock ROCm 7.1 の device library は LLVM 24 bitcode でした。個別に指定した `ocml.bc` の読み込みは `Producer: 'LLVM24.0.0git' Reader: 'LLVM 20.1.8'` で停止しました。

そこで TheRock と同じ ROCm LLVM revision `4f43f4746ede4cc49ab4129f473dbd54e0d9db4f`（LLVM 24.0.0git）を取得し、`X86;AMDGPU`、AdaptiveCpp revision `0ea863a`、Windows HIP backend を組み込んでビルドしました。AdaptiveCpp は LLVM 24 を正式対応範囲外としているため、移動した LLVM plugin header と pass API、clang preprocessor include、厳格化された inline attribute の互換修正を `.tools/` 内の外部ソースだけに加えました。

`.tools/sycl-smoke.cpp` を `hip:gfx1201` 向けにコンパイルし、R9700 実機で実行しました。テストは次をすべて確認して終了コード 0 になりました。

- SYCL のデバイス列挙から R9700 を選択し、実行 queue を作成
- device USM を 4,096 float 分確保
- host から device への転送
- 4,096 要素を更新する `parallel_for` の実行
- upload、kernel、download の event 依存と `wait_and_throw()` による同期
- device から host への転送と全要素の結果照合
- 実行デバイス `AMD Radeon AI PRO R9700`、global memory `34208743424` bytes、subgroup size 32

実行出力は `USM transfer/kernel/event verification passed` でした。これで最小 SYCL 実行までの toolchain blocker は解消しました。次は XeStrata 本体の Windows/AdaptiveCpp ビルド対応です。

## 2026-10-11: 本体のビルドと実モデル読み込み

`build/windows-acpp/strata.exe` のコンパイルとリンクが成功し、`--help` も終了コード 0 で実行できました。oneMath、native experts、tests、packaging を無効化した初期構成です。Windows のファイル読み込み、メモリ確保、CPU affinity と、AdaptiveCpp にない oneAPI 拡張の代替経路を追加しました。matrix 系は既存の DP4a または scalar 経路を使います。

最初の実行は GPU 選択前に異常終了しました。例外を表示する経路を追加した結果、AdaptiveCpp が部分対応のため `aspect::fp16` を常に false と返し、XeStrata がデバイスを拒否していたことが分かりました。AdaptiveCpp 経路では、使用する half 変換と積を実カーネルで照合してからデバイスを選びます。R9700 ではこの照合を通りました。

モデルは既存の `<MODEL_ROOT>/Q2_0/` と `packs/q2_0/` を使い、モデルデータのダウンロードは行っていません。実行ログで次を確認しました。

- 実行 GPU は `AMD Radeon AI PRO R9700`、32 compute units、31.9 GiB
- dense weights 4,630 MiB の読み込み完了
- 2 GiB を超える読み込み位置の切り詰めを修正し、全層の weights 読み込みが成功
- PLE の IQ4_NL table（320,001,536 rows）を開いた
- `experts.bin` を Windows の file mapping で開いた
- CPU expert pool の起動

この実行は `--no-capture` が CPU routed experts を処理しないという安全チェックで停止しました。`--no-pool` でチェックを回避するとモデルが変わるため、その方法は使いません。次は CPU expert pool を含む標準 queue/event の実行経路を実装します。prefill、decode、テキスト生成の成功はまだ確認していません。

ビルドには Windows CRT deprecation と subgroup attribute などの警告が残っています。Linux の free と contrib-llvm、NVIDIA、小さい構成での回帰チェックと lint はまだ実施していません。

## 2026-10-11: Q2_0モデルでprefill、decode、生成が完了

command graphを使わない`session_token_pool()`を追加しました。各層のroutingと入力をdeviceからhostへ転送し、eventの完了を待ってCPU expert poolを実行します。その層のexpert出力をGPUへ戻してから残差を更新し、次の層へ進みます。embedding、PLE、各層、headは同じin-order queueで実行します。routed expertsとPLEを省略する設定は使っていません。

既存のQ2_0モデルのtokenizerで、次の会話を13トークンに変換しました。

```text
<|im_start|>user
Say hello in English.<|im_end|>
<|im_start|>assistant
```

R9700実機でprefillとdecodeが完了し、推論文に続いて`Hello!`という応答、`<|im_end|>`、`<|endoftext|>`が生成されました。終了コードは0です。生成した41トークンは、独立した3回の実行ですべて一致しました。最終ビルドの実行ログでは、prefillは12トークンを2,732.6 ms、decodeは41トークンを9,027.6 msで処理しました。最後の入力トークンは最初の生成トークンの計測に含まれます。今回のprefillは1トークンずつの経路であり、native packのbatched prefillを確認したものではありません。

CPU expert poolは3 workersとhost threadで実行し、53 positionsの48層で25,440個のexpert blobを読み込みました。2回目は`STRATA_VRAM_LIMIT_MIB=8192 STRATA_MAX_ALLOC_MIB=4096 STRATA_NO_XMX=1`も指定し、ログのGPUメモリ表示が8.0 GiBになることを確認しました。ただし、現状のdevice allocatorは最大allocationの模擬制限を強制しておらず、dense arenaは4 GiBを超えます。この実行は、最大allocationが4 GiBのカードで動くことの確認にはなりません。

WindowsのGPU Engine counterでも、`strata.exe`のprocess IDに対応するCopyとComputeの使用を記録しました。非ゼロのsnapshotが14回あり、Computeでは58～76%の値も記録されています。SYCLの実行GPU表示と合わせて、CPU-onlyでの生成ではないことを確認しました。

ローカルの確認記録は次のファイルにあります。

- `build/windows-acpp/model-hello-final.txt`: 最終ビルドの生成ログ
- `build/windows-acpp/model-hello-complete-1-decoded.txt`: tokenizerで復号した応答
- `build/windows-acpp/model-hello-small.txt`: メモリ表示を8 GiBに制限した実行
- `build/windows-acpp/model-hello-small-gpu.txt`: process ID別のGPU counter

`session.cpp`のhost構文チェックは`-Wall -Wextra -Werror`で通りました。`git diff --check`も通っています。`tools/lint/run.sh`は実行しましたが、必要なlintツールがこの環境にないため終了コード1になりました。特にgitleaks未導入の失敗も「possible secret」と表示されるため、この表示から秘密情報が見つかったとは判断できません。Linuxのfreeとcontrib-llvm、NVIDIA、内蔵GPU、ASan/UBSan、native/batched経路の回帰チェックはまだ実施していません。

コミット前にREUSEも個別に実行しました。文字コード判定を`REUSE_ENCODING_MODULE=chardet`にすると、日本語文書を含む今回の変更にはライセンス情報の不足はありませんでした。ただし、リポジトリ全体のチェックには、既存の`.agents/skills`が残っています。Gitではsymlinkですが、このWindows checkoutでは通常ファイルとして扱われるためです。全体のREUSEチェックは終了コード1で、合格にはなっていません。

## 未着手または明示的に後回しにする機能

最初の正常な生成までは、次を無効化できる経路を作ります。

- oneAPI experimental command graph と graph replay
- oneAPI virtual memory による elastic cache
- arbitrary host mapping の `prepare_for_device_copy`
- adaptive cache
- shared expert arena
- 複数 GPU の二段 pipeline

command graph を無効にするときも、標準 SYCL の queue と event による依存関係、メモリ寿命、同期順序は維持します。HIP エンジンへの全面置換は行いません。
