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

## 現在の作業方針: リモート接続を守りながら正確性を確立する

ユーザーがPCを直接操作できない間は、リモート接続の維持と復旧可能性を性能より優先します。Windowsの再起動・シャットダウン、GPU driverの更新・再起動・reset、システム全体の環境変数やregistry変更は行いません。Strata本家の実行ファイル、benchmark、GPU負荷テストも起動しません。本家のソースを静的に参照することはできますが、性能比較と出力一致の確認は、ユーザーが直接操作できることを改めて確認してから行います。

作業はPhase 0のベースライン保護、Phase 1の単体の正確性・安定性、Phase 2の時間内訳の分析、Phase 3の測定に基づく小規模な高速化、Phase 4の汎用化・保守性の順に進めます。Phase 3では主な変更を一つずつ行い、同じ条件でprefillとdecode、数値結果、生成品質を比較します。queue/eventの正常経路は残し、同期の削除には依存関係とバッファ寿命の確認が必要です。DPC++/HIPは独立した後続調査とし、当面は公開ソースの静的調査に留めます。

GPU実行は1プロセスずつ、短い入力と既存モデルで行います。初期の生成上限は64、128トークンとし、EOS後に生成を続けて上限まで埋めることはしません。256トークンや長時間stress testは必須ではありません。実行にはtimeoutとWindowsのRAM・commit・GPUメモリの監視を設けます。異常、driverの応答停止、急なメモリ増加が疑われたら追加負荷を止め、復旧リスクのある操作は事前に確認します。

## 2026-10-11: Phase 0のベースライン保護

現在のブランチは`feature/windows-r9700-sycl`、HEADは`e7d67ead89cdc308e737b05d8a6ab09675c7679c`です。この確認を始めた時点の未コミット変更は、将来計画を追加した日英の進捗文書だけでした。これらもpatchとして保存しています。

ローカルの`build/windows-baseline-20261011T074825Z/`に、動作した実行ファイル、AdaptiveCppのruntime/backend DLL、HIP関連DLL、実行・ビルドscript、CMake設定、入力IDs、成功ログ、外部toolchainの互換patchを保存しました。実行ファイルとDLLのコピーはSHA-256で照合し、ソースの`source.bundle`は`git bundle verify`で検証しました。コピーした実行ファイルの`--help`も正常終了しました。この保存先はGitの対象外です。モデルのコピーやダウンロードは行っていません。

復元用の`run-baseline.bat`は保存した実行ファイルとruntimeを選択し、元のWindows実行環境とモデルを使います。ソースはbundleと`worktree.patch`で保存した状態を確認できます。OS、driver、元のtoolchain/ROCm resource treeまで含む完全な環境imageではありません。既存の環境を上書きせずに、この保存した実行経路を維持します。

現在のcompilerはROCm LLVM revision `4f43f4746ede4cc49ab4129f473dbd54e0d9db4f`のclang 24.0.0git、AdaptiveCppは`0ea863abe1af4422593583f91e5608ef75044a16`です。HIP headerの版は7.17.26391で、DLLの`hipRuntimeGetVersion()`も`71726391`を返しました。過去の「ROCm 7.1系」という記録と区別するため、現在のheaderとruntimeの値を保存しています。構成は`STRATA_ADAPTIVECPP=ON`、`hip:gfx1201`、oneMath/native experts/tests/packaging無効で、生成にはCPU experts、PLE、mmapを使います。

保存済みのログは次の二つを分けて扱います。32トークンの出力は、41トークンの完了ログの先頭32トークンと一致しました。

| ログ | 生成数 | prefill | decode | TTFT | CPU poolの時間 |
| --- | --- | --- | --- | --- | --- |
| `model-hello-1.txt` | 32 | 2.00 tok/s | 3.11 tok/s | 6,308.0 ms | 224.633 ms/token |
| `model-hello-final.txt` | 41、EOSまで | 4.39 tok/s | 4.54 tok/s | 2,903.8 ms | 97.727 ms/token |

後者には最終回答までの記録があります。ただし生成上限やcache状態などが同一と確認された変更前後の測定ではないため、この差を最適化の効果とはしません。CPU poolの値も、待機や同期を含み得る経過時間であり、純粋な演算時間ではありません。Strata本家との数値一致は確認していません。

負荷をかける前のWindows計測では、空きRAM約14,660 MiB、空きcommit約18,121 MiB、commit limit約38,984 MiB、全adapterのdedicated使用量約1,502 MiBでした。別processでkernelを実行せずに取得したHIP device 0のメモリ情報は、空き約32,473 MiB、全体32,624 MiBでした。HIPとWindows counterの値は同じ予算を表すとは限らないため、実行中もWindows側の値を監視します。

確認コマンドと記録は次にあります。

- `python .tools/protect-windows-baseline.py`: snapshot作成とhash/bundle検証
- `cmd /d /c "call build\windows-baseline-20261011T074825Z\run-baseline.bat --help"`: 保存した実行ファイルの確認
- `build/windows-acpp/baseline-protection.txt`: 保存結果
- `build/windows-acpp/phase0-hip-memory.txt`: HIPのメモリ照会
- `build/windows-acpp/phase1-preflight.txt`: WindowsのRAM・commit・GPU counter

Phase 0の保存と確認は完了しました。続くPhase 1では、まずhost-onlyテストで非同期I/Oの完了通知件数と終了時の寿命を修正し、その後に小規模な実ファイルとGPUの検証を行いました。

## 2026-10-11: Phase 1の転送順序と数値再現性

### Host-onlyテストとWindows I/O

`tools/test_session_token_pool_host.py`は、実際の`session_token_pool()`と関連する状態更新関数を抽出し、疑似FIFOキューで実行します。48層のQSA/GDN索引、複数tokenにまたがる状態、callbackと転送の順序、stagingの寿命、不正入力、例外時の終了処理を確認しました。追加のlayer dumpも同じテストで確認しています。このテストの疑似演算は、実カーネルの数値結果を保証するものではありません。

`tools/test_dequant_host_contract.py`では、65,536種類のhalf bit pattern、384個のQ2_0 block、IQ4_NLのcodebookと配置を確認しました。これらを合わせた6件のhost-onlyテストは成功しています。

Windowsの`DirectFile`は、wake packetと実際の完了が混在しても呼び出し側の上限を超えて書かないように修正しました。取り出した余剰の完了は次回へ残します。終了時には発行threadを止め、未完了readをcancelしてから、完了通知を回収してrequestとbufferの寿命を終えます。IOCP基盤のエラーが続く場合は、生きたrequestを解放せず待機を続けます。

疑似Win32 APIによるbounds、cancel、dequeue失敗のテストに加え、実Windows SDKで`/W4 /WX`ビルドと実ファイルテストを行いました。自前の32,785-byte file、64-KiB buffer、最大8件のreadだけを使い、発行thread数1/3、完了取得上限1/2、内容と範囲外canary、短いreadとEOF、不正alignmentの拒否、close・破棄・再オープンを確認しました。短いreadの未返却部分はWindowsが変更してよいため、内容を確認するのは返却された範囲だけです。cancel対象が常に未完了だったことや、持続的な基盤エラーからの復帰までは確認していません。

### Native in-order経路の不具合と対処

同じhello入力を2回実行すると、全層の残差と41生成tokenはbit単位で一致しましたが、logitsには455,360個の差があり、最大差は23.862265でした。`atol=rtol=1e-5`の基準を超え、hostへ取り出したlogitsのargmaxとGPUが選んだtokenも一部で食い違いました。生成文の一致だけでは、この問題を検出できませんでした。

host USMへの転送、`queue.wait_and_throw()`によるdrain、D2H event自体の待機を個別に試しましたが、argmaxの食い違いは残りました。改善を確認できなかった変更は取り消しています。

次に、モデルを読み込まない小さなSYCLテストで切り分けました。明示的なGPU allocationは最大993,344 bytesで、20秒のJob Object timeoutとメモリ監視を付けています。

| 検証 | Native dedicated in-order | DAG依存関係を使うin-order |
| --- | --- | --- |
| 実subgroup sizeとXOR reduction | size 32、全1,024要素一致 | 全要素一致 |
| kernel後の248,320要素のcopyとmarker待機 | 全要素が古い0を返す | 全要素が基準値とbit単位で一致 |
| kernel event待機後のcopy event待機 | 全要素一致 | 全要素一致 |
| 実際のQ8 head kernelと独立scalar dot、64行 | 全行一致 | 全行一致 |

現在のWindows HIP環境では、native dedicated in-orderのkernelからcopyへの順序を信用できません。Windows＋AdaptiveCppに限り、`AdaptiveCpp_inorder_executor{nullptr}`をcompute、transfer、追加streamの生成に指定しました。AdaptiveCppは各操作を前のeventへ依存させるDAG経路を使います。これはruntime内部のsubmission DAGであり、oneAPI command graphやcaptureを有効にする変更ではありません。Linux／DPC++のqueue設定、演算カーネル、sampling設定は変えていません。backend内部の原因までは特定していません。

### 実モデルの再確認

修正後、既存Q2_0 packとPLE GGUFを使い、context上限256、最大64生成token、mmap experts、captureなしで確認しました。

- worker数3の独立した2回は、53 positionsの残差26,593,280 floatsとlogits13,160,960 floatsがすべてbit単位で一致しました。最大差は0で、全値がfiniteでした。
- 41生成tokenは従来のhello結果と同じで、各生成位置のhost logitsのargmaxもGPUの選択と一致しました。最後までdecodeすると`Hello!`を含む最終回答とEOSを確認できます。
- worker数1でも、残差、全logits、生成tokenがworker数3とbit単位で一致しました。
- 51入力token、最大128生成tokenの設定でも1回確認しました。55tokenでhello回答とEOSに到達し、105 positionsの全logitsがfinite、全55生成位置のargmaxがGPUの選択と一致しました。128tokenを実際に生成した検証やcontext境界の検証ではありません。
- 修正前後の残差もbit単位で一致しました。変化したのは誤って取り出していたlogitsです。
- 2回のdiagnostic実行はprefill 2.00/2.24 tok/s、decode 2.01/2.37 tok/sでした。layer/logit dumpとcache状態が異なるため、旧ログとの速度比較には使いません。

各実行は独立したJob Objectで監視し、同時にGPUテストを走らせていません。停止条件には150秒の上限、空きRAM 3 GiB、空きcommit 6 GiB、process private memoryとGPU dedicated使用量の8 GiB上限を設けています。今回の実行で停止条件に達したものはなく、終了後のGPU使用量も実行前と同程度に戻りました。新しいモデルの取得、本家Strataの起動、driverやシステム設定の変更は行っていません。

主な記録は`build/windows-acpp/phase1-dag-numerical-dumps.json`、`phase1-dag-worker1-numerical.json`、`phase1-dag-longer128-argmax.json`、`phase1-sycl-minimal.txt`、`phase1-sycl-dag.txt`、`phase1-directfile-real-v2.txt`、各実行のmonitorとJSONです。logit dumpの既存headerは予定行数を表すため、helloでは76行に対して実データ53行、長めの入力では178行に対して105行です。検証ではtokenログから実際の行数を求めています。

修正後の実行ファイル、runtime DLL、変更ソースと追加テスト、検証記録、Git bundleと未コミット差分を`build/windows-correctness-baseline-20261011T094332Z/`に別保存しました。SHA-256とbundleを検証し、保存した実行ファイルの`--help`も成功しました。Phase 0のベースラインは変更していません。モデル本体はコピーせず、復元には既存のdriver、toolchain、モデル環境が必要です。

短い逐次経路の転送順序と数値再現性は確認できました。さらに長い入力、context境界、ほかの量子化、全内部状態の独立した基準値、実GPU障害時のcleanupは引き続き確認が必要です。性能の最適化はまだ採用していません。続くPhase 2で取得した時間内訳は次節に記録します。

## 2026-10-11: Phase 2の時間内訳と停止地点

`STRATA_POOL_TIMING=1`で、逐次CPU pool経路の既存API呼び出しをhostの単調時計で測定しました。標準では無効です。計測用のevent、copy、同期、GPU profilingは追加せず、演算やbuffer寿命も変えていません。`PoolTiming`のcountは呼び出し試行数、bytesは成功した明示的copyのpayloadです。

計測は`.tools/bounded-windows-run.py`の`--pool-timing --diagnostics`で起動し、childだけに`STRATA_POOL_TIMING=1`を渡しました。GPU選択は`HIP_VISIBLE_DEVICES=0`、`STRATA_VRAM_LIMIT_MIB=8192`、`STRATA_NO_XMX=1`、`STRATA_SSD_KEEPALIVE=0`です。4-GiB最大allocationの制限を強制した実行ではありません。既存のローカルtoolchainとpackを使う再現用コマンドは次のとおりです。停止後に追加実行する指示ではありません。

```text
python .tools/bounded-windows-run.py --exe build/windows-acpp/strata.exe --label phase2-new-measurement --max-new 64 --pool-timing --diagnostics
```

保存JSONの旧env一覧には`STRATA_POOL_TIMING`が含まれませんが、起動オプションと実行ログの計測header・53行の内訳で有効化を確認できます。この記録形式の不足は後続課題とし、今回の計測のために追加実行はしていません。

条件はPhase 1のhelloと同じ13入力token、最大64生成token、context上限256、worker数3、mmap experts、captureなしです。layer/logit dumpを付けた独立した2回を、同じ監視と停止条件で実行しました。各53 positionsの全logitsと残差は計測前の保存結果とも、計測後の2回同士でもbit単位で一致しました。41生成tokenと各位置のargmaxも一致し、全値がfiniteでした。追加のhost-only計測テストを含む7件も成功しています。

以下は53 positionsで平均したhost経過時間です。GPU submitはembedding、pre、post、headのAPI呼び出し時間で、GPU実行時間ではありません。待機はruntimeのscheduling、GPU処理、転送を含みます。CPU expert欄もcallback内の計算、入出力、dispatch待ちを含み、これらを個別には分離していません。

| 項目 | 実行A ms/position | 実行B ms/position |
| --- | ---: | ---: |
| routing/inputのevent待機 | 349.611 | 324.439 |
| CPU expert callback | 174.642 | 116.992 |
| GPU submit合計 | 23.269 | 26.295 |
| routing/input D2H submitとevent記録 | 1.430 | 1.760 |
| expert parts H2D submit | 2.590 | 2.296 |
| session末尾のdrain | 1.421 | 1.438 |
| diagnostic残差のD2H submit | 0.625 | 0.660 |

routing/input D2Hは495,360 bytes/position、parts H2Dは4,915,200 bytes/positionです。これはkernel内部やdriverのstaging、PLE、headのlogit転送を含まない明示的payloadであり、物理busの転送量ではありません。通常設定ではpost直後に独立したparts待機を行わないため、その計測欄は0です。完了待ちは次のrouting待機と最終drainに含まれます。診断用残差copyも後続の待機に影響します。

prefillとdecodeは別に集計しています。decodeの平均は、待機352.349/318.846 ms、CPU callback160.867/116.177 ms、GPU submit22.938/25.144 msでした。cacheを消去・制御していないため、実行Aをcold、Bをwarmとは断定しません。計測とdumpの負荷も含むので、以前の通常実行との速度比較には使いません。

観測した上位3項目と、今後検討できる候補は次のとおりです。候補の実装や高速化テストは行っていません。

1. **routing/input待機**。最も長いhost区間です。GPU演算、転送、DAG schedulingをさらに分離し、順序を保つsubmitのまとめ方やkernel融合を検討できます。今回壊れていたnative in-order経路への単純な復帰や同期の削除は候補にしません。
2. **CPU expert callback**。実行間の変動も大きい区間です。ページ入出力と演算を分離したうえで、worker/task粒度やメモリ予算内の固定cacheを検討できます。callback全体の時間を純粋なCPU計算時間とは扱いません。
3. **GPU submitのhost負荷**。pre、postなどの小さな呼び出しが重なる区間です。実際の起動数とruntime処理を調べ、依存関係とstaging寿命を変えないbatchingを検討できます。

主な記録は`build/windows-acpp/phase2-timing-summary.json`、`phase2-timing-summary.txt`、`phase2-timing64-a-numerical.json`、`phase2-repeat-numerical.json`、2回の実行ログとmonitorです。process private memoryの最大値は約5,019 MiB、空きRAMの最小値は約6,046 MiB、空きcommitは約12,665 MiBで、停止条件には達していません。

既存のPhase 0版と、数値再現性を確認したPhase 1版の復元用コピーは維持しています。repository lintは必要ツールが不足しており完了していません。gitleaks未導入時の`possible secret`表示は、秘密情報を検出した証拠ではありません。REUSEで新しいファイルの指摘は解消し、残る指摘は以前からあるWindows上の`.agents/skills`のcheckout状態です。Linuxのfree/contrib-llvm、NVIDIA、内蔵GPU、ASan/UBSan、context境界、ほかの量子化、GPUの実障害cleanup、物理転送時間とhard faultの分離は引き続き確認が必要です。

ユーザーの指定に従い、Phase 2の記録と検証を終えた時点で停止します。Phase 3の高速化、DPC++への移行、他GPUへの対応、大規模なリファクタリングには着手していません。

## Phase 3開始前: AdaptiveCppの暫定async-handlerパッチ

使用中のAdaptiveCpp `0ea863abe1af4422593583f91e5608ef75044a16`では、node付き`sycl::event` constructorが受け取った`async_handler`を保存していませんでした。非空のerror一覧を通知すると、元の例外ではなく`std::bad_function_call`になり得ます。

`third_party/main/adaptivecpp/patches/0001-event-async-handler.patch`で、constructorの1行だけを変更します。元のBSD-2-Clause copyright・license表示は維持しました。`tools/adaptivecpp_build.py`はSHAとheader全文を確認してからビルド前に適用し、適用済みならスキップ、不整合なら停止します。LF/CRLFを区別し、source/build/prefixの重複や正常版installの上書きも拒否します。install後のheader全文とSHA-256をreceiptへ記録します。手順は`third_party/main/adaptivecpp/README.ja.md`にあります。

既存環境を上書きせず、同じcommitの別worktree `.tools/AdaptiveCpp-eventfix-src`からCPU backendのみのruntimeをビルドし、`.tools/adaptivecpp-eventfix`へinstallしました。installed headerが修正済みsourceと一致することを確認しました。最初の単独Windowsビルドは例外を無効にした設定で停止したため、別ビルドのcompiler flagに`/EHsc`を明示しました。XeStrata本体やAdaptiveCppのqueue/DAG/schedulerは変更していません。

隔離テスト`tests/platform/adaptivecpp_event_async.cpp`は、実際のinstalled headerとruntimeの共有error一覧を使います。nodeはnullで、device queue、device allocation、kernel、DMA、HIP直接呼出しはありません。synthetic errorの元diagnostic・SYCL errc、handler呼出し回数、複数error、event copy、再通知なし、throwするhandlerを確認しました。同じCPU runtimeと未修正headerのnegative controlでは、handler未通知と`std::bad_function_call`を再現しました。両方とも20秒timeout内で正常終了し、出力したruntime DLL pathは新prefixでした。Windowsでは再throw後の`exception_ptr`同一性は保証されないため、type/code/diagnosticを検査しています。

wrapperの26件のhostテストは成功しました。主な記録は`build/windows-acpp/async-eventfix-build-v2.log`、`async-eventfix-receipt.json`、最終wrapperで再適用をスキップした`async-eventfix-receipt-v3.json`、`async-eventfix-installed-check.json`、`async-eventfix-wrapper-tests-final-v2.log`、`async-eventfix-installed-final-v2.json`、`async-eventfix-skip-final-v2.json`、`async-eventfix-fixed-test-v2.txt`、`async-eventfix-broken-test-v2.txt`です。

正常版のAdaptiveCpp source/header、HIP install、生成実行ファイルと既存の復元用コピーは維持しています。新prefixは隔離検証用で、正常版のHIP環境へ切り替えたものではありません。GPU実行や意図的なhang、driver更新、Phase 3の性能最適化は行っていません。

この修正が対象とするのは、handlerを渡したnode付きeventの通知です。default生成eventのhandler挙動、process-global error一覧の競合、cancelされたnodeのcomplete扱い、HIP待機失敗後やDMA完了不明時のcleanupは別課題として残します。今回の成功をGPU障害時のbuffer寿命や実モデル生成の検証とは扱わず、ここで停止します。

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

## 今後の計画: AdaptiveCppの安定化とDPC++＋HIPの調査

### 現在の基準と公開実装

現状は、Windows/AMD向けの実験的移植として扱います。実行経路はXeStrata → SYCL → 独自ビルドのAdaptiveCpp → LLVM 24/AMDGPU → HIP/ROCm → R9700です。既存のQ2_0 packとPLE用GGUFで、モデル読み込み、逐次prefill、decode、文章生成まで成功しています。「モデル読み込み・生成は未確認」という調査初期の状態からは進んでいます。ただし、ビルドできたのは選択した構成であり、native/batched経路や全構成のカーネルを確認したものではありません。

DPC++にもAMD向けのHIP実装があります。Unified Runtimeの開発元は現在intel/llvm内の`unified-runtime/`で、oneapi-src/unified-runtimeはmirrorです。HIP adapterのCMakeにはWindows向けのimport library設定もあります。一方、DPC++のGetting Started GuideではWindowsのHIPを非対応としています。公開ソースにWindows用の分岐があることと、R9700で正しく実行できることは別に確認します。

CodeplayのoneAPI for AMD GPUs 2025.2.0の利用手順はLinux向けです。また、intel/llvmのissue #22606には、Ubuntu上のgfx1200で`single_task`は動くものの`nd_range`が起動に失敗する報告があります。後続コメントには修正後の起動とstatic archiveからのdevice linkの成功報告もあります。Windows/gfx1201の成功例とは扱わず、調査時の再現テストとcompiler revision選定の参考にします。

### 第一段階: AdaptiveCpp版を安定化する

動作済みのQ2_0逐次経路を、以後の変更を比較する基準として残します。まず非同期I/Oの終了・キャンセル、queue/eventの依存関係、失敗時のメモリ寿命をテストします。subgroup sizeの指定に関する警告も調べ、実行時のサイズと、reduction・routingなどそのサイズを前提とする演算を照合します。コンパイル成功や短い生成だけで、全カーネルの正確性を判断しません。

次に、toolchainのrevision、互換パッチ、DLL配置、ビルド設定を整理し、空のbuild directoryから同じ生成を再現します。長い入力、長めのdecode、context境界、CPU worker数を変えた実行も追加します。最大allocationの制限を実際に強制し、4 GiBを超えるarenaを分割したうえで、小さい構成の検証を行います。Linuxのfreeとcontrib-llvmで関連テストが通ることも確認します。

この段階の完了条件は、モデルの構成要素を省略せずに生成を再現できること、異常終了時も安全に解放できること、ビルド手順と検証範囲を他の環境でも確認できることです。native expertsとbatched prefillは逐次経路との比較テストを用意してから追加します。command graphの復活を必須条件にはしません。

### 第二段階: WindowsのDPC++＋HIPを別環境で調べる

DPC++の調査は別のブランチまたはworktreeとtoolchain directoryで行い、動作済みのAdaptiveCpp環境を上書きしません。intel/llvmのDPC++とUnified RuntimeのHIP adapterをソースから構築し、使用するROCmのdevice bitcodeとのLLVM互換性を先に確認します。compilerとruntimeのrevisionも揃えます。

検証は、adapterのロードとGPU認識、USMの確保と双方向転送、`single_task`、`nd_range`、複数queueのevent依存の順に進めます。halfとsubgroup演算も、実行時のサイズと計算結果を照合します。XeStrataがstatic libraryを組み合わせているため、複数translation unitとstatic archiveを含むdevice linkも確認します。出力照合まで通った後に、XeStrataのモデル読み込みと逐次生成を試します。

oneAPI拡張のAPIがコンパイルできても、HIP backendで使えるとは判断しません。command graph、virtual memory、matrix機能は個別に確認し、使えない機能はqueue/eventなどの代替経路を維持します。GPUの経路は製品名ではなく、報告された能力と実カーネルの結果から選びます。

この段階の成功条件は、WindowsのR9700で最小テストと実モデル生成を実行し、結果を照合できることです。ビルドや実行に明確な障害がある場合は、最小再現例、revision、エラーを記録して調査を区切り、AdaptiveCpp版の開発を続けます。DPC++への切り替えは調査を始めた時点では決めません。

### 第三段階: 正確性と速度を同じ条件で比較する

ユーザーがPCを直接操作できると確認するまで、Strata本家の実行比較は延期します。その確認後、AdaptiveCpp版とStrata HIP版を、同じR9700、モデルのrevisionと量子化、入力token IDs、context長、KV設定、CPU worker数、メモリ予算、生成設定で比較します。DPC++版が実モデル生成まで通った場合は比較に加えます。既存のモデルとpackを使い、追加データが必要ならユーザーに対応を求めます。モデルを自動でダウンロードしません。

正確性は生成文だけでなく、logitsや各層の状態も数値誤差の基準を決めて照合します。速度はprefill、decode、TTFTを分け、逐次とbatched、coldとwarmの実行を区別して複数回測定します。CPU expert処理、GPU処理、転送、I/O、待機時間と使用メモリも記録し、CPU-onlyやPLE・expertsを省略した実行を成功の根拠にはしません。

計測結果を見て、固定GPU expert cache、転送との重複実行、batched処理、AMD向け行列演算の優先順位を決めます。DPC++がoneAPI拡張に対する互換コードを減らせるか、toolchainを保守しやすいかも比較します。APIの数やcompiler名だけで採用する経路を決めず、同じモデルの正しさ、再現性、速度、保守負担を判断材料にします。非free依存は該当するビルドモードに限定し、XeStrata自体のライセンスとLinuxの経路を維持します。

### 調査資料

- [DPC++ Getting Started GuideのHIP制限](https://intel.github.io/llvm/GetStartedGuide.html#hip-back-end-limitations)
- [Unified Runtimeの現在の開発元](https://github.com/intel/llvm/tree/sycl/unified-runtime)
- [HIP adapterのCMake](https://github.com/intel/llvm/blob/sycl/unified-runtime/source/adapters/hip/CMakeLists.txt)
- [Codeplay oneAPI for AMD GPUs 2025.2.0](https://developer.codeplay.com/products/oneapi/amd/2025.2.0/guides/get-started-guide-amd.html)
- [intel/llvm issue #22606](https://github.com/intel/llvm/issues/22606)

## 未着手または明示的に後回しにする機能

初期生成の経路では次を使っていません。安定化と比較検証までは、対応範囲を広げずに代替経路または明示的な無効化を維持します。

- oneAPI experimental command graph と graph replay
- oneAPI virtual memory による elastic cache
- arbitrary host mapping の `prepare_for_device_copy`
- adaptive cache
- shared expert arena
- 複数 GPU の二段 pipeline

command graph を無効にするときも、標準 SYCL の queue と event による依存関係、メモリ寿命、同期順序は維持します。HIP エンジンへの全面置換は行いません。
