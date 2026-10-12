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

ユーザーの指定に従い、Phase 2の記録と検証を終えた時点で停止しました。その時点ではPhase 3の高速化、DPC++への移行、他GPUへの対応、大規模なリファクタリングには着手していません。

## Phase 3開始前: AdaptiveCppの暫定async-handlerパッチ

使用中のAdaptiveCpp `0ea863abe1af4422593583f91e5608ef75044a16`では、node付き`sycl::event` constructorが受け取った`async_handler`を保存していませんでした。非空のerror一覧を通知すると、元の例外ではなく`std::bad_function_call`になり得ます。

`third_party/main/adaptivecpp/patches/0001-event-async-handler.patch`で、constructorの1行だけを変更します。元のBSD-2-Clause copyright・license表示は維持しました。`tools/adaptivecpp_build.py`はSHAとheader全文を確認してからビルド前に適用し、適用済みならスキップ、不整合なら停止します。LF/CRLFを区別し、source/build/prefixの重複や正常版installの上書きも拒否します。install後のheader全文とSHA-256をreceiptへ記録します。手順は`third_party/main/adaptivecpp/README.ja.md`にあります。

既存環境を上書きせず、同じcommitの別worktree `.tools/AdaptiveCpp-eventfix-src`からCPU backendのみのruntimeをビルドし、`.tools/adaptivecpp-eventfix`へinstallしました。installed headerが修正済みsourceと一致することを確認しました。最初の単独Windowsビルドは例外を無効にした設定で停止したため、別ビルドのcompiler flagに`/EHsc`を明示しました。XeStrata本体やAdaptiveCppのqueue/DAG/schedulerは変更していません。

隔離テスト`tests/platform/adaptivecpp_event_async.cpp`は、実際のinstalled headerとruntimeの共有error一覧を使います。nodeはnullで、device queue、device allocation、kernel、DMA、HIP直接呼出しはありません。synthetic errorの元diagnostic・SYCL errc、handler呼出し回数、複数error、event copy、再通知なし、throwするhandlerを確認しました。同じCPU runtimeと未修正headerのnegative controlでは、handler未通知と`std::bad_function_call`を再現しました。両方とも20秒timeout内で正常終了し、出力したruntime DLL pathは新prefixでした。Windowsでは再throw後の`exception_ptr`同一性は保証されないため、type/code/diagnosticを検査しています。

wrapperの26件のhostテストは成功しました。主な記録は`build/windows-acpp/async-eventfix-build-v2.log`、`async-eventfix-receipt.json`、最終wrapperで再適用をスキップした`async-eventfix-receipt-v3.json`、`async-eventfix-installed-check.json`、`async-eventfix-wrapper-tests-final-v2.log`、`async-eventfix-installed-final-v2.json`、`async-eventfix-skip-final-v2.json`、`async-eventfix-fixed-test-v2.txt`、`async-eventfix-broken-test-v2.txt`です。

正常版のAdaptiveCpp source/header、HIP install、生成実行ファイルと既存の復元用コピーは維持しています。新prefixは隔離検証用で、正常版のHIP環境へ切り替えたものではありません。このCPU隔離検証ではGPU実行や意図的なhang、driver更新、Phase 3の性能最適化は行っていません。

この修正が対象とするのは、handlerを渡したnode付きeventの通知です。default生成eventのhandler挙動、process-global error一覧の競合、cancelされたnodeのcomplete扱い、HIP待機失敗後やDMA完了不明時のcleanupは別課題として残します。今回の成功をGPU障害時のbuffer寿命や実モデル生成の検証とは扱わず、ここで停止します。

## Phase 3B: Routing D2H marker削減

### 基準と変更範囲

開始時のHEADは`ae1d3ac7be3b1e2eba6ccae75f32f8c9ca1c2968`、branchは`feature/windows-r9700-sycl`で、作業ツリーはcleanでした。AdaptiveCppは`0ea863abe1af4422593583f91e5608ef75044a16`、LLVMは24です。変更前のsourceを別worktreeへ固定し、`build/windows-phase3b-baseline`と`build/windows-phase3b-candidate`へ同じfreeモードの設定でビルドしました。両版とも正常版からコピーしたHIP対応header一式に、SHA限定のasync-handler修正headerだけを重ねています。正常版installを置き換える構成ではありません。

モデルprocessでロード済みmoduleを列挙しました。`acpp-rt.dll`とHIP backendは`.tools/adaptivecpp-amd24/bin`から、HIP DLLは共有SDK直下ではなく`C:\Windows\System32\amdhip64_7.dll`からロードされました。パスとSHA-256は各実行のJSONに記録しています。このWindows DLLのFileVersionは`10.0.3683.0`で、HIP APIやROCmのversionを表す値としては扱いません。

`src/core/session.cpp`のweights D2Hを既存の`copy_async_after(..., cs, nullptr, scratch.probe)`へまとめ、直後の空markerとその失敗分岐だけを削除しました。pre → mixed D2H → IDs D2H → weights D2Hの順序、最後のcopy eventを待つ位置、CPU Expert callback、parts H2D、post、最終drain、バッファは変えていません。Windows＋AdaptiveCppのDAG依存方式も維持しています。演算、量子化、sampling、cache、GPUの選択方法は変更していません。

### 正確性・順序・寿命の確認

Phase 3Bに先立つ小規模GPUテストでは、R9700で1 kernelと3回D2Hの6,144値が一致しました。先行kernelやqueue全体を待たず、最後のcopy eventだけで完了を確認しています。GPU完了後に共有error一覧へ合成エラーを登録し、そのcopy eventから元の例外型・code・diagnosticがhandlerへ届くことも確認しました。記録は`async-eventfix-gpu-once.txt`です。実GPU障害を発生させたテストではありません。

実モデルの変更前後は13 input IDs、41生成token、53位置です。全logits 13,160,960値と、input＋48層の残差26,593,280値がbitwise一致し、非有限値は0でした。最大logit差は0、生成tokenも一致し、両版の41 sampled argmaxがGPUの選択IDと一致しました。変更前の新ビルドも保存済みPhase 2の全logits・残差とbitwise一致しています。EOSで打ち切られた実際の行数を使い、既存logit headerの予定行数76を53実行行と混同していません。

FIFO host fixtureは、最後のcopyのevent公開、callback前の3コピー完了、前層のparts DMA後のstaging再利用、event公開失敗時のcallback抑止とdrain試行を確認します。sessionとdequantの既存hostテスト7件が成功しました。mockのdrain成功を、実障害時のDMA完了保証とは扱いません。両ビルドのwarning診断集計は同じで、新しいwarning messageはありませんでした。`git diff --check`と日本語lintは成功しました。全体lintは必要ツール不足で完了せず、REUSEの残る指摘は既存の`.agents/skills`のcheckout状態だけです。

Routing D2Hの成功時API呼び出し数は1位置あたり192から144へ減りました。48層×53位置で2,544個の空markerを削除し、payloadは両版とも495,360 bytes/位置です。parts H2Dは4,915,200 bytes/位置で変わりません。既存の`routing_D2H_submit+record`という表示名は維持しています。bytesは成功を返したAPIのpayload集計で、物理bus転送量ではありません。

### 同条件の反復性能比較

数値確認の診断dump付き2実行は性能集計から除外しました。その後、診断dumpなしで各版3回をA1 → B1 → B2 → A2 → A3 → B3の順に単独実行しました。Aがmarker版、Bが最後のweights copy eventを使う版です。同じQ2_0 pack、13 input IDs、generation cap 64、context 256、greedy/EOS、3 workers＋caller、mmap、固定GPU cache 0、no-capture、no-XMX、8 GiBのVRAM reporting制限を使い、`--check-logits --stats`と既存境界のhost timingを有効にしました。各processはprivate Job内で150秒と従来のmemory guardに制限しています。prefillは12位置、decodeは41位置です。

| 実行順 | Prefill tok/s | Decode tok/s |
| --- | ---: | ---: |
| A1 | 2.083 | 2.073 |
| B1 | 2.265 | 2.282 |
| B2 | 2.275 | 2.525 |
| A2 | 2.471 | 2.636 |
| A3 | 2.546 | 2.684 |
| B3 | 2.517 | 2.689 |

中央値はprefillがA 2.471、B 2.275 tok/s、decodeがA 2.636、B 2.525 tok/sでした。候補の中央値は低いものの範囲は重なり、時間経過に伴う変動もあります。OS/file cache温度や外部負荷は制御していないため、cold/warmを割り当てず、marker削減による高速化とも性能回帰とも断定しません。Routing D2H submitの平均ms/位置を実行ごとに集計した中央値はA 1.603、B 1.094でした。これはhostのAPI時間で、純粋なGPU演算や転送時間ではありません。

8実行すべてでmemory guardに達せず、非同期エラーの診断や生成tokenの差はありませんでした。最大process privateは約5,020 MiB、GPU dedicated集計は約6,627 MiBです。性能6実行の最小空きRAMは約6,002 MiB、空きcommitは約12,456 MiBで、終了後GPU dedicated集計は約1,522 MiBへ戻りました。

### 採用判断と停止点

全体の高速化は確認できませんでしたが、依存関係・バッファ・数値処理を変えずに空markerと失敗分岐を減らせたため、この小さい変更を採用します。上記の中央値低下を含め、性能への効果は保証しません。重大な新規回帰は観測していません。cancel／DMA完了不明時のcleanup、global error一覧の競合、長いcontext、他の量子化、Linuxのfree/contrib-llvm、NVIDIA、内蔵GPU、ASan/UBSanは引き続き別に確認する課題です。上流SDMA修正は取り込まず、DAG方式も外していません。

正常版のAdaptiveCpp source/header、HIP/ROCm環境、`build/windows-acpp/strata.exe`、既存のPhase 0・1復元用コピーは変更していません。基準記録は`build/windows-acpp/phase3b-start-state.json`、数値比較は`phase3b-numerical-comparison.json`と両版のargmax JSON、性能集計は`phase3b-performance.json`、hostテストは`phase3b-combined-host-contracts.log`です。各実行のlog、monitor、module path/hash、最終の正常版integrity記録も同じdirectoryに保存します。これらの生成物はcommit対象から除外します。

Phase 3Bの記録と採用変更だけをcommitして停止しました。その時点ではPhase 3CのGPU Expert cache、Phase 3Dの整理や上流更新取り込みには着手していません。

## Phase 3C: canonical GPU Expert cacheの調査と撤退

### 開始時の基準と既存実装

開始時のHEADは`f4f2350ce9f50d65169644991dfbe898464f8702`、branchは`feature/windows-r9700-sycl`で、作業ツリーはcleanでした。AdaptiveCppのSHAは引き続き`0ea863abe1af4422593583f91e5608ef75044a16`、compilerはLLVM 24です。Phase 3Bで検証した別buildの実行ファイルをそのまま使い、正常版、Phase 0・1の復元用コピー、共有ROCm環境は維持しました。修正済みasync-handler headerと既存DAG方式、3BのD2H event再利用も同じです。モデルprocessがロードした主要runtime DLLのhashは3実行で一致し、HIP DLLは引き続き`C:\Windows\System32\amdhip64_7.dll`でした。

既存の`ExpertCache`と`expert_hit_run`はWindowsでもコンパイルされており、positive cache countとCPU pool、canonical geometry、全hit bufferが揃えば`moe_hit_grouped_s2`へ進みます。graph、VMM、native layoutは必要ありません。「まだconsumer/kernelがない」というcache headerの古い説明は、現在の実行経路を表していません。

`--expert-cache N`はMiBではなくexpertのslot数です。Q2_0 canonical blobはH=2560、FF=640、1 expertあたり1,382,400 bytesで、uniform cacheはN個の個別allocationではなくN×blob bytesの単一device-USM arenaです。今回は4 slots、5,529,600 bytes（約5.27 MiB）から始めました。hit用のscratch、slot/destination list、入力Q8、FP32 scale、outputの6 allocationは合計165,440 bytesです。SYCL queryのglobal memoryと最大単一allocationはともに34,208,743,424 bytes、sub-group sizeは32でした。新しいcache arenaはこの値を下回ります。

AdaptiveCppの`mem_info()`はfree VRAMを返さず、cache内部のfree-memory preflightは動きません。既存allocatorも最大単一allocationの強制制限は行いません。そのため自動サイズを使わず、外部のGPU dedicated集計と従来の8 GiB memory guardで余裕を確認しました。このqueryは既存dense arenaや4 GiB制限の問題を解消するものではありません。

global admissionは最初に出た(layer, expert)を順に入れ、満杯後はevictionしません。新規admissionは同じstreamでblob H2Dをkernelより先に置き、その呼び出しからGPU計算になります。CPUはhit行を0にし、miss行だけを計算します。parts H2D後にGPU hit outputを加え、postでrouter weightsを適用する経路は維持しています。既存のper-layer policyはN/48を各層に割り当て、余りを最後の層へ置きます。N=4でそれを選ぶと前47層のquotaが0になるため、今回はglobal policyだけを使いました。

cache arenaの所有・解放は既存のRAIIとruntime drainに任せます。raw allocationされた6 hit bufferとprobe eventには終了時の個別解放がなく、process終了まで残る既存の所有権上の制限があります。新規admission後のfill失敗も安全な再利用としては確認されていません。cancel／DMA完了不明時のcleanupと合わせ、今回それらを修正したとは扱いません。

### 比較前に固定した数値基準

最初のcache-onより前に、全logits・全層残差の各値について`abs(a-b) <= 0.001 + 0.001 * max(abs(a), abs(b))`、全値finite、sampled argmaxと生成token一致を必須としました。GPU/CPUのreductionとquantization順序が異なるためbitwise一致は必須にしませんが、結果を見て許容値は緩めません。基準は`phase3c-predeclared-contract.json`に保存しました。

まずsourceを変えず、同じQ2_0 packと13 input IDs、CPU pool 3 workers＋caller、mmap、no-capture、context 256、greedy/EOS、cap 64、no-XMX、診断dumpとhost timingでcache-offと4 slotsのdefault cache-onを各1回実行しました。`--adapt-every 0 --pcie-frac 0`を両方へ指定し、auto cache、profile、elastic、VMM、Command Graphは使っていません。

### 数値結果と限定調査

cache-offは保存済み3Bの全logits・全層残差とbitwise一致しました。default cache-onは正常終了し、生成41 tokenと全53位置のargmaxは同じでした。しかし全logits 13,160,960値のうち12,738,073値、残差26,593,280値のうち8,699,164値が事前基準を超えました。最大差はlogits約0.827092、残差約0.391922で、数値ゲートは不合格です。token一致だけを成功と扱いません。

次は4 slotsと許容値を固定し、既存の`--expert-cache-cpu-order`だけを切り替えました。こちらも正常終了しましたが、生成は40 token・52位置となり、tokenとargmaxが変わりました。共通位置の比較でも基準を満たさず、最大差はlogits約34.5925、残差約15.0584でした。EOSによる行数差を記録し、52位置のpayloadを53位置と同じ形状とは扱っていません。

最初の位置のinput checkpointは両cache-onとも完全一致でした。1層後の最大差はどちらも約2.24e-8で、defaultでは4層後に約0.001345へ拡大しています。これは微小差が後続のquantizationやstateで拡大する可能性を示しますが、根本原因や全dispatchの正しさを確定する証拠ではありません。

実モデルのcache-on反復は止め、小規模のraw Expert fixtureへ切り替えました。canonical layer 0/expert 0の1 blobと固定runtime入力を使い、既存のproduction CPU/GPU quantizerとExpert kernelをリンクしました。DAG queueの小さいadapterを使い、engineのdiscovery・probe・teardownは置き換えているため、fixtureを実モデルやcache全体の検証とは扱いません。device allocationは1,413,120 bytes、単一の制限付きprocessです。defaultとCPU-orderを各1回実行し、入力int8 2,560値・FP32 scale 80値・FP16 block header 80値がすべて一致しました。Expert output 2,560値は両方とも基準内で、最大差は約3.35e-8／5.96e-8、非有限値は0でした。

5判断の時点で、cacheの実行経路、CPU-orderだけでは直らないこと、初期checkpointの微小差、raw kernelの局所parityまでは確認できました。実routing/inputでの複数hitや後続stateを含む原因は限定できておらず、安全な小さいWindows互換修正は特定していません。許容値変更、容量拡大、新しいcache algorithmやGPU kernelへの書き換えで回避することはしません。

### hit率、診断時間、メモリ

default 4 slotsでは25,440 lookupsに対し既存resident hit 4、新規admission 4、CPU miss 25,432でした。表示上のhit率は約0.0157%、admissionを含むGPU計算率は約0.0314%、miss率は約99.9686%です。初回routingで4 slotsが埋まり、global first-seenの偏りもあるため、小容量のenablement確認であって実用的なcache配置の評価ではありません。`expert blobs`はhitを含むsource要求数で、物理SSD read数ではありません。

| 診断実行 | Prefill tok/s | Decode tok/s | CPU callback平均 ms/位置 | Routing wait平均 ms/位置 | 推論合計 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| cache-off | 1.71 | 1.87 | 177.260 | 330.578 | 28,952.0 |
| default 4 slots | 2.01 | 2.31 | 107.132 | 303.978 | 23,716.6 |
| CPU-order 4 slots | 1.65 | 1.20 | 203.666 | 529.283 | 40,598.0 |

これは各1回の診断dump付き時間で、反復性能比較ではありません。CPU-orderは生成数も違います。OSページcache温度は制御しておらず、CPU callback短縮やdefaultの速度表示を高速化成功と判定しません。既存timingにはGPU hit launch/combineとcache setupの独立区間がなく、初期fillはhit Launch内に含まれます。その時間を他の区間から推定したり、ロード時間をcache初期化時間と呼んだりしません。数値ゲート不合格のため各3回の性能測定・定常状態評価・容量拡大には進んでいません。

3モデル実行は非同期エラー診断やmemory guard違反なく終了しました。最大process privateは約5,020 MiB、GPU dedicated集計は約6,716 MiBで、終了後は約1,474〜1,546 MiBへ戻りました。これらは新たな実障害時の同期・寿命保証ではありません。

### 撤退判断と再開条件

今回cache-onを採用しません。既存pathはGPUで動作しますが、事前に定義した実モデルの数値条件を満たさず、原因を限定した互換修正がないためです。sourceや正常版に変更はなく、cache-offとPhase 3Bの基準を維持します。再開には、実routingと同じ入力・同じexpertのGPU hit／CPU miss中間出力を照合し、差が拡大する最初のquantization・state・dispatchを限定したうえで、同じ数値基準を満たす小さい修正が必要です。cache-onの性能評価はその後の別指示で行います。

開始・保護対象は`build/windows-acpp/phase3c-start-state.json`、事前基準は`phase3c-predeclared-contract.json`、数値比較は`phase3c-on4-initial-vs-phase3c-off-initial.json`と`phase3c-on4-cpuorder-vs-phase3c-off-initial-v2.json`、raw結果は`phase3c-raw-default.txt`と`phase3c-raw-cpuorder.txt`、診断集計は`phase3c-diagnostic-metrics.json`、integrityは`phase3c-final-integrity.json`です。ログ・モデル・実行ファイル・ローカルfixtureはcommitせず、`git diff --check`と日本語lintは成功しました。全体lintは必要ツール不足で完了せず、REUSEの残る指摘は既存の`.agents/skills`のcheckout状態だけです。日英の撤退記録だけをcommitして停止します。Phase 3D、WMMA、Command Graph、VMM、DPC++移行、他GPU対応には着手していません。

## Phase 3P: 性能内訳と次の調査地点

### 基準、測定方法、終了判断

開始HEADは`570fc69a690d9a9a311fc943a1644a798553c281`、branchは`feature/windows-r9700-sycl`、作業ツリーはcleanでした。Windows 11 Pro build 26200、Ryzen AI 9 HX 370（12 cores／24 logical processors）、R9700 driver `32.0.32015.2008`を記録しました。free modeのLLVM 24は`4f43f4746ede4cc49ab4129f473dbd54e0d9db4f`、AdaptiveCppは`0ea863abe1af4422593583f91e5608ef75044a16`です。SDK headerのHIP versionは7.17.26391です。実際にロードしたHIP DLLはSystem32の`amdhip64_7.dll`で、Windows FileVersion／ProductVersionの`10.0.3683.0`をHIP API versionとは呼びません。新しい2モデルprocessでも主要DLLのhashは同じでした。

Phase 2、3B、3C cache-offの保存済み11実行を再集計しました。現行の速度基準には、dumpなし・3B candidateの3実行だけを使います。共通条件は既存Q2_0 packとIQ4_NL PLE、13 input IDs、41生成token、context 256、cap 64、greedy/EOS、3 workers＋caller、mmap、cache 0、no-capture、DAG方式、no-XMX、VRAM表示上限8 GiBです。旧3B baseはmarker 192、現行candidateは144 calls/位置です。Phase 2の診断dump付き時間を、dumpなしの3Bと高速化の前後比較には使いません。OSページcache温度や外部負荷は制御していません。

追加計測は既存3B実行ファイルでの正常な診断実行1回、model-free SYCL profiling確認1回、ETW記録下の推論1回に限定しました。正常診断は全13,160,960 logits・26,593,280残差値が保存済みcache-offとbitwise一致し、finite、41 sampled argmax／tokenも一致しました。診断時Decodeは1.76 tok/sで、過去のdumpなし中央値2.525 tok/sとの差を回帰とは扱いません。

AMD profilerは確認したPATH・install場所に見つかりませんでした。Windowsの[公式component表](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/conceptual/component-support.html)はRadeon GPU Profilerを挙げますが、この環境に導入済みとは限りません。代替は標準SYCL profilingとETWの2種類に限りました。標準profilingは既存のlast-copy wait後にだけ取得し、6,144 readback値が完全一致しました。HIP backendはstart/endにstream eventを置き、`hipEventElapsedTime`から差を計算します。ただしこれはstream上の操作を囲む時間で、API発行中の空白も含み得ます。hardware kernel dispatch時間やDMA bus占有時間そのものではありません。絶対時刻は`steady_clock`とhost基準eventから構成され、laneをまたぐ比較には基準点誤差があります。

最後のeventからDAGをたどる方法では、weak dependencyの先行nodeがGC済みの場合、全submissionを回収できません。queue observerのローカルprototypeは作りましたが、統一header overlayと全node保持が必要になるため、実モデル用にbuild・実行・採用していません。正常版や推論sourceは変更していません。

ETWはユーザーのUAC承認を受け、WPR `10.0.26100`で専用instanceの`GPU.light`・`CPU.light`・`DiskIO.light`をmemory modeで開始しました。記録中にtraceを書き込んでSSD計測を汚さないための選択でしたが、35.938秒時点で使用可能RAMが3,022.6 MiBとなり、3 GiBのJob guardで推論を停止しました。31位置のtimingが残りましたが、完全なDecode実行ではありません。記録中のメモリ圧迫を確認しただけで、ETWだけが低下の原因だとは断定していません。追加GPU実行は中止し、自分のWPR instanceをstop=0で終了、not recordingを確認しました。新しい正常診断にはdumpがあり、ETW実行はdumpなしで途中停止したため、両者の速度差をprofiling overheadの率として比較できません。完走した同条件の通常／profiling性能比較は未成立です。

WPAExporter `11.8.423.12582`で停止済みtraceをoffline解析しました。JSON configの最初の試行は`Invalid output format`でload前に失敗したため、CLIへ変えて1回だけ再試行しました。PID 10020をcommand lineで確認し、CPU、process、disk、lost-eventのCSVを取得しましたが、GPU presetはno dataでexport失敗しました。終了code 0を全table成功とは扱いません。このpresetではGPU時間を得られず、trace自体にGPU eventがないとは判断していません。lost-event CSVに行がないことも、loss=0の証明ではありません。exporterはprivate 4 GiB・RAM空き3 GiB・120秒・kill-on-close Jobで制限し、23.984秒、peak private約2,468 MiBで終了しました。

### 測定済みの内訳

以下は3B candidate 2、41 Decode位置のhost側内訳です。この実行は3実行の中央値に当たり、16,239.4 ms／41＝396.083 ms/位置です。複数実行の区間ごとの中央値を足して1実行の内訳にはしていません。

| Host区間 | ms/位置 | wall比 | 含まれる処理・限界 |
| --- | ---: | ---: | --- |
| Routing/Input wait | 288.519 | 72.84% | queued pre、前のpost／H2D、転送、DAG/runtimeを含む |
| CPU Expert callback | 72.724 | 18.36% | mmap fault、入力quantization、2段Expert計算、dispatch・barrierを含む |
| GPU操作のsubmit合計 | 25.308 | 6.39% | kernelとcopyのhost呼び出し。GPU実行時間ではない |
| Final drain等の残りの計測区間 | 0.781 | 0.20% | 既存の終了待機。上の区間とhost側では重ならない |
| Token timer内の未分解分 | 8.751 | 2.21% | PLE、head readback／finite scan／sampling、制御処理等。GPU idleではない |

現行3実行のRouting/Input waitは267.042〜330.200 ms/Decode位置、wallの71.81〜75.34%です。CPU callbackは72.676〜75.349 ms、16.58〜20.26%でした。両者のhost区間が大きいことは確実ですが、routing waitの内訳がGPU演算なのかruntimeなのかは分離できていません。pre/post/embedding/headのsubmit API数も、内部のGPU kernel数ではありません。

転送payloadはRouting D2Hが495,360 bytes/位置（48層×3 copies）、parts H2Dが4,915,200 bytes/位置（48 copies）です。代表実行のhost submit平均はD2H 1.152 ms、H2D 2.038 msです。48回のrouting event waitと最後のsession drainを維持しています。`parts_wait(post)=0`でも、次のrouting gateが前のparts H2Dとstaging寿命を保護しているため、転送のcostが0とは言えません。host stagingは既存`malloc_host`経路で、HIP backend source上は`hipHostMalloc`です。Expert mmapは別の通常pageable mappingです。実DMA時間・転送依存待機だけの時間・physical bus bytesは未測定です。

CPU poolの全53位置平均drは3B candidate 2で71.752 ms、外側callbackは71.916 msです。しかしdefaultの`run_split`は中間quantizationと2段のdispatch/barrierをすべてdrへ入れ、wp/rpを別計測しません。表示上のwp/rp=0を「worker待機や同期が軽い」と解釈しません。dequantization、Expert演算、hard fault、SSD service、メモリ帯域、task／workerごとの時間は未分解です。Expert要求量632.8125 MiB/位置は論理blob bytesで、SSD／DRAMの実トラフィックではありません。

新しい正常診断の最終process counterはuser 106.297＋kernel 61.719＝約168.016 CPU秒、sample間の平均は4.79 logical core相当でした。page fault累積は2,233,038ですがsoft/hardを区別しません。`GetProcessIoCounters`のread約6.32 GBもロード等を含み、mmapのphysical SSD read量ではありません。4計算threadのcallback内CPU時間の理論上限は、全callback 9.759秒×4＝39.037 CPU秒です。その計算区間だけではprocess全体の168 CPU秒を説明できませんが、残りをruntimeやworker idleへ割り当てるにはthread別の記録が必要です。

途中ETWのsampled CPUは169.763 CPU秒／process lifetime 36.054秒＝約4.71 logical core相当でした。通常実行と別の途中実行であり、samplingと厳密counterも別の量です。CSVはprocess集計で、Strataのstackが空、thread/module列もありませんでした。diskもdisk単位集計でProcess/Pathが空でした。これらからCPU Expert、driver、DAG scheduler、SSDを個別原因にはできません。

Attention／GDN／QSA、Router、Shared Expert、post、embedding／LM Head、quantization／GEMVのGPU時間・各層kernel数は未測定です。GPUの実idle区間、HIP API時間、submissionから実行開始までの遅延、DAG scheduler時間も未測定です。320 ms前後のwaitや4.79 coreというprocess CPU使用量を、そのままAdaptiveCppのoverheadとは呼びません。

### ボトルネック順位と改善候補

順位は「確認できたhost区間と次に調べる価値」に基づきます。GPU演算対runtimeの原因順位を確定したものではありません。単一区間をcost 0にし他を不変とした理論上限は`1/(1-f)`で、予測改善率ではなく、複数候補の上限も足しません。

| 順位・候補 | 根拠と確信度 | 改善上限・負担・数値リスク・追従性 |
| --- | --- | --- |
| 1. Routing critical windowのGPU対runtime分離 | 71.81〜75.34%という区間は高確信、内部原因は未確定 | 代表区間の理想除去上限3.68倍。安全に消せる時間は不明。kernel融合／WMMAは負担・回帰リスク・本家との差分が大きく、演算占有の証拠を得るまで実装候補を確定しない |
| 2. CPU poolのidle spin設定を独立A/B | process CPU約4.8 core、source default spin 20 ms。routing平均は約5.56〜6.88 ms/層でspin閾値より短い。spin寄与はsource上の仮説で、thread attributionは未取得 | 速度改善上限は未測定。既存`STRATA_POOL_SPIN_US`の測定ならsource差分0、負担小、数値リスク低。ただしwake遅延で悪化もあり得る。旧測定の全envは未保存なので20 ms実適用の断定はしない |
| 3. CPU Expertのwork／fault／barrier分離 | callback 16.58〜20.26%は高確信、各内訳は未確定 | callback全除去でも1.20〜1.25倍という独立理論上限。CPU layout/dequant最適化は中程度の負担と数値リスク。GPU cacheは3Cの数値ゲート解決とGPU追加cost検証が必要で、今回は無効のまま |
| 4. Host submissionの小差分削減 | 代表実行の全submit 6.39%は高確信、runtime全体の割合ではない | 全submit理想除去上限約1.068倍。D2H submitだけは約1.003倍。small API整理は比較的追従しやすいが、DAG orderingや必要waitは変更しない |

AdaptiveCpp／ROCmの上流修正でscheduler／driver側が改善する可能性はありますが、未分離の時間から改善率は見積もれません。SDMA順序修正を理由にDAG回避策を外すこともしません。CPU pool設定／layoutは主に本家側と共有できる候補で、GPU専用の別engineにする必要はありません。Strata本家を実行したり、その速度をruntime overheadのoracleにしたりしていません。

### 成果物、再開地点、停止

採用したコードはoffline専用の`tools/profile_pool_timing.py`とそのstandard-library testです。UTF-8 logからactual rowsを検証し、prefill／decode／全位置のhost accounting、payload／API counts、未分解分、独立理論上限をJSONへ出します。engineやGPUを起動せず、cacheやprofiling設定をlogの不在から推定しません。9 testsと現行3Bの3ログ・新正常診断で検証しました。`git diff --check`と日本語lintは成功し、全体lintはツール不足で未完了、REUSEは既存の`.agents/skills` checkoutの指摘が残ります。ローカルobserverやETW controllerをproductionへ採用していません。

再開の最小候補は、実envとworker threadを確認したうえで、cache-offのまま既存spin設定を1要因だけ変えるA/Bです。source実装はまだ選定していません。数値bitwise gate、同条件の反復、CPU-thread時間、必要なwake遅延を確認してから採用判断をします。最大host windowの原因分離には、bufferを小さくした専用ETW profileとGPU table対応の確認が必要です。これは次の別指示で行い、今回のRAM guardを緩めません。

再集計は`phase3p-saved-host-report.json/.txt`、開始保護は`phase3p-start-state.json`、新正常診断は`phase3p-normal-diag.*`、数値gateは`phase3p-normal-numerical.json`と`phase3p-normal-argmax.json`、process counterは`phase3p-process-counters.json`です。途中traceは`phase3p-etw-once.etl`、停止確認は`phase3p-etw-once.done`と`-poststatus.txt`、offline部分exportは`phase3p-offline/cli-once-report.md`です。これらは`build/windows-acpp/`に保持し、生成ログ・trace・model・実行ファイル・toolchainはcommitしません。正常版と使用header/runtimeのhash一致、ロード済みDLLの一致、推論source無変更を`phase3p-final-integrity.json`へ保存しました。共有SDKや復元用コピーには変更を加えていませんが、このJSONでその全file hashを再検証したとは扱いません。

性能の単一原因は確定していませんが、最大host window、CPU全体の負担、確認済みの上限と不足する証拠を整理した時点でPhase 3Pを終了します。計測の追加GPU実行と高速化実装には進まず、採用したoffline parserと日英文書だけをcommitして停止します。Phase 3D、cache再有効化、WMMA、Command Graph、VMM、native in-order、DPC++移行、driver／SDK更新には着手していません。

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

<a id="lane-affinity-measurement"></a>
## 2026-10-12: AdaptiveCppのlane照合を隔離して測定

AdaptiveCpp `0ea863abe1af4422593583f91e5608ef75044a16`の`multi_queue_executor::find_assigned_lane_index`は、DAG nodeに登録された親executorを子executorと比較していたため、先行nodeのlaneを見つけられませんでした。候補修正は所属・backend・device範囲を確認してから、nodeのexecution laneと子のqueueを照合します。kernel/copy laneの分離と必要なstream間依存、XeStrataのDAG方式、final-copy event待機は維持します。全laneを統合する別案はkernel後のD2Hで248,320要素すべてが古い0になったため不採用で、モデル性能測定には使用していません。

Windows R9700の隔離runtime（B）と正常版（A）で同じ13入力ID、41出力token、53位置を比較しました。診断A/Bでは全13,160,960 logitsと26,593,280 residual floatsがbit単位で一致し、非有限値は0です。dumpなしのfresh processをA1→B1→A2→B2→A3→B3の順に各3回実行し、Decode中央値はA 2.576→B 3.846 tok/s（+49.3%、1.493倍）、Routing/Inputのhost待機中央値は261.091→122.509 ms/位置、CPU callbackは100.624→101.673 ms/位置でした。待機はGPU処理・転送・runtimeを含み、GPU実行時間の内訳ではありません。固定したこのworkloadでの結果であり、10倍には達していません。

修正は`third_party/main/adaptivecpp/patches/0002-dag-lane-affinity.patch`として固定commitのAdaptiveCpp sourceだけへ適用する対象です。別source/build/prefixで再現・検査し、正常版installを上書きしません。正常版exe、SDK、driver、復元用コピーは維持しました。記録はGit対象外の`build/windows-acpp/lane-affinity-20261012T084944Z/report.ja.md`、`result.json`と`.tools/lane-affinity-runtime/source.diff`にあります。他のprompt、長いcontext、別GPU・量子化、実GPU障害時のcleanup、4 GiB最大allocation条件は今回の検査範囲外です。

<a id="expert-cache-and-wsl"></a>
## 2026-10-13: VRAMのexpert cacheを有効化、WSL 2の経路は不可

### expert cacheが常に0だった原因

AdaptiveCppのビルドでは`strata::gpu::mem_info`が常に失敗を返していました。`--expert-cache auto`は空きVRAMを0 GiBと読み、cacheは0 slotsになっていました。Phase 3Cで4 slotsしか試せなかったのも、この自動sizingが働かなかったためです。HIP backendのdeviceでは、HIPの`hipMemGetInfo`を実行時に読み込んで空きVRAMを取ります（buildはHIPをlinkしません）。HIPが答えないbackendでは、deviceの総量からこのprocessの確保量を引いた値を使います。

WindowsのHIPでは、device allocationがsystem commitも同じ量だけ消費します。4 GiBの`hipMalloc`でcommitの空きが4.1 GiB減り、`hipFree`で戻りました。物理RAMは消費しません。そのためWindowsでは、空きVRAMと「空きcommit − 4 GiB」の小さい方を空き容量とします。27 GiB分のslotsを一度に確保したときはcommitの空きが98 MiBまで落ちましたが、この上限を入れてからは確保が空きcommitの範囲に収まります。page fileを大きくすれば、cacheも大きくなります。

同じhello入力（13 input IDs、`--max-new 64`、context 256、workers 3、mmap、VRAM制限とno-XMXの指定なし）で、cache autoは7,759 slots（9.99 GiB）、hitは25,440要求中19,091（75.0%）でした。生成した41 tokenはcache-offと同じです。Decodeは3.22〜4.87 tok/sで、cache-offの3.11〜4.88 tok/sとほぼ変わりません。この短い実行ではcacheが空の状態から始まり、6,349個のexpertを生成中にmmapから読んでVRAMへ入れているためです。

### 残る下限: GPU側の層ごとの時間

cache-on、`STRATA_POOL_TIMING=1`のDecodeでは、CPU poolは0.09 ms/位置まで減り、Routing/Inputのhost待機は約115 ms/位置（1層あたり約2.4 ms）のまま残りました。`--no-pool`（CPU expertsなし）のDecodeも10.60 tok/s（約94 ms/token）です。同じengineはLinuxのRX 9060 XT（R9700の約半分のGPU）でdecode約49 tok/sなので、Windows経路の主な損失はGPU側の層ごとの時間です。Linux経路のcommand graph、doorbell、native experts、batched prefillはこの経路にありません。

native in-order queue（AdaptiveCppのDAG schedulerを通さない）も試しました。Decodeは4.88から6.02 tok/sになりましたが、headのlogits readbackが古い値を返し、53位置中19位置でargmaxが食い違いました。Routing待機も約128 ms/位置で、DAG方式と変わりません。不採用とし、DAG方式を維持します。

### WSL 2の経路

WSL 2（Ubuntu 24.04、ROCm 7.2.3、librocdxg）でXeStrataのLinux版を試しました。intel/llvm v7.1.1（HIP target付き）とengineのbuildは通りました。一方でR9700（gfx1201）では、`hipMalloc`は成功するものの、最初のqueue作成、kernel起動、event記録が完了しません。同じ最小HIPテストは内蔵の890M（gfx1150）では通ります。

librocdxgのdebug logでは、ROCm内部の最初のkernel dispatchをPM4へ変換して投入した後、R9700ではqueueのread pointerが0から進みません。librocdxgの`src/wddm/cmd_util.cpp`（ROCm/rocm-systemsへ移った現在の版も同じ）はgfx12の分岐を持たず、cache flushは「gfx10以降は共通」のbit配置、`PRIV`はgfx11だけに立てています。最新のlibrocdxg 1.2.2でも変わらず、gfx12にも`PRIV`を立てる試みではWindowsがbugcheck 0xD1（`dxgmms2.sys`）で停止しました。Windows driverは32.0.32015.2008（2026-09-28）で、AMDがWSLで保証する組み合わせ（Adrenalin 26.2.2とROCm 7.2.1）より新しい版です。

WSL経由のR9700はこの環境では使えないため、Windowsネイティブの経路を続けます。WSL 2でAMD GPUを`rocminfo`から検出するsetupの変更は、このbugcheckの危険があるためcommitしていません。intel/llvmのbuild後に`xmx_probe`がWSL上で約1時間45分CPUを使い続けた問題も、別に直す必要があります。

<a id="pool-graphs"></a>
## 2026-10-13: pool経路の層をHIP graphで再生

### 層ごとの時間の原因

AdaptiveCppのHIP backendで、起動の費用を小さなSYCLテストで測りました。中身が空のkernelでも、続けて投入すると1個あたり約38 µs（DAG方式）、26〜35 µs（native in-order）かかります。AdaptiveCppを通さない素のHIPで同じkernelを続けて起動すると1.7 µs、kernelごとにeventを記録すると34 µsでした。AdaptiveCppは操作ごとに完了eventを記録するので、Windows HIPではkernel 1個ごとに別のsubmissionが発生していると考えられます。後ろにeventがないkernelは、同期が成功を返しても結果に反映されないことがありました（Windows HIPの挙動。native in-orderでreadbackが古い値になる現象と同じ根の可能性があります）。

HIPのstream captureで、AdaptiveCppのnative in-order queueに投入したSYCL kernelをgraphへ記録できました。40 kernelのgraphを再生すると1個あたり3〜8 µsで、値も一致しました。

### 実装

- `strata::gpu`のcapture APIを、AdaptiveCppビルドではHIPのstream captureで実装しました（`begin_capture`、`end_capture`、`abandon_capture`、`is_capturing`、`graph_launch`、`graph_destroy`）。HIPは実行時に読み込み、buildはHIPをlinkしません。記録はnative in-order queueの1本のHIP streamで行い（`stream_create_capture`）、再生はengineのDAG queueのcustom operationとして`hipGraphLaunch`を呼ぶので、前後の操作との順序はAdaptiveCppが保ちます。
- `session_capture`に`doorbell`引数を加え、doorbellなしで`pre`と`post`を記録できるようにしました。`session_token_pool`は、渡されたgraphがあれば層ごとに`pre[l]`と`post[l]`を再生し、位置は`stage_token`で渡します。routingのD2H、CPU pool、hit path、dumpはgraphの外に残ります。
- AdaptiveCppビルドでは既定で有効です（`STRATA_POOL_GRAPHS=0`で無効）。ほかのビルドでは`STRATA_POOL_GRAPHS=1`のときだけ使います。

### 結果

`pre`は48層で2,197ノード（1層あたり約46）、`post`は96ノードです。hello入力、cache 0で、graphあり・なしを交互に2回ずつ測りました。

| | graphなし | graphあり |
| --- | --- | --- |
| Decode | 4.65、4.61 tok/s | 7.15、7.05 tok/s |
| Prefill | 4.08、4.10 tok/s | 5.98、6.00 tok/s |

全logitsはgraphなしとbit単位で一致しました（helloの53位置、計算問題の入力の66位置）。Routing待機は約115から約95 ms/位置に減り、CPUのsubmitは約14から約3 ms/位置になりました。残る1層あたり約2 msはkernel自体の時間が中心と考えられます（1層46ノードの起動は0.4 ms程度）。

このPCの条件では、expert cacheを併用すると遅くなりました。コミットの空きで約4,900 slots（expertの約20%）に限られ、profileで事前に埋めてもhit率は48%、Decodeは4.3 tok/sでした。hitの計算はgraphの外でGPUの層と直列に並び、kernelごとの起動費用もかかります。CPU poolはgraphありで約30 ms/位置なので、cacheで移せる分より増える分が大きくなります。

`--stage-timing`はAdaptiveCppビルドではすべて0を返し、kernel別のGPU時間はまだ測れていません。unverified：Linuxのfreeとcontrib-llvm（`session_capture`の変更は既定値で従来どおり）、NVIDIA、内蔵GPU、長いcontext、他の量子化。

<a id="gpu-time-per-token"></a>
## 2026-10-13: GPU時間の内訳

AdaptiveCppビルドでは計時用のeventがmarker kernelで、timestampを持たないため、`--stage-timing`は全区間0を返していました。計時用のeventだけ、custom operationの中でHIP eventを記録し、`hipEventElapsedTime`で区間を読むようにしました。`--stage-timing`のときはpool経路のgraphを使いません（graphの中に計時eventを置けないため）。この計時は区間ごとにsubmissionを増やすので、値は主にkernel数に比例する起動費用で、Decodeも2.64 tok/sまで落ちます。比率を見る診断用です。

pool経路のgraphで`--gpu-stages`も使えるようにしました（`session_capture`の`split`）。graphの中で測った1 tokenあたりのGPU時間は次のとおりです。

| 区間 | ms/token | ms/層 |
| --- | ---: | ---: |
| mixer（gr_read、attention、gr_write） | 21.9 | 0.46 |
| FFN前段とrouter | 12.6 | 0.26 |
| post（moe_finish、gr_write） | 1.0 | 0.02 |
| 合計 | 35.5 | 0.74 |

GDN層は0.44 ms/層、QSA層は0.50 ms/層です。prefixの差分による5区間の表は、差分法とsweep法でずれが大きく（router区間で-91%など）、区間ごとの細かい値には使えません。

graphありの通常実行では、Routing待機が約95 ms/位置、CPU poolが約30 ms/位置です。GPUの計算は約35 msなので、Routing待機の約60 msは層ごとの受け渡し（graph起動、3回のD2H、event待機、parts H2D、`post`の起動を、AdaptiveCppのDAG queueでそれぞれ投入する費用）と考えられます。次はこの受け渡しを減らします。

<a id="direct-loop"></a>
## 2026-10-13: 層の受け渡しを専用のHIP streamで直接行う

graphを使っても、層ごとに`pre`の起動、3回のD2H、event待機、parts H2D、`post`の起動を、それぞれAdaptiveCppのDAG queueへ投入していました。routingのD2Hを`pre[l]`の最後に、partsのH2Dを`post[l]`の最初に入れて記録し（`session_capture`の`y_miss`）、層のループの間はengineのqueueを通さず専用のHIP stream（`strata::gpu::direct_stream`）にgraphを直接起動します。1層は`pre`の起動、`hipStreamSynchronize`、CPU pool、`post`の起動だけになります。層の前にengine streamを1回drainし、層の後にdirect streamを1回同期して、embedding、PLE、headとの順序を保ちます。

hit path（expert cache）、層のdump、層ごとの同期を使うときは、copyを含まないgraphで従来の受け渡しを使います。copyを含むgraphが従来の受け渡しへ渡されると、copyが重複してhitのcombineを上書きするため、`session_token_pool`はエラーで止めます。

hello入力、cache 0で、Decodeは8.87、9.00 tok/s、Prefillは7.27、7.48 tok/sでした（直前のgraphだけの版は7.15、7.05 tok/sと5.98、6.00 tok/s）。全logitsは従来の受け渡しとbit単位で一致しました（helloの53位置、計算問題の66位置）。cache autoでは従来の受け渡しに戻り、生成した41 tokenも同じです。

Routing待機は約80 ms/位置、CPU poolは約35 ms/位置です。CPU poolを省いた診断（出力は正しくない）ではRouting待機が約54 msで、GPUの計算約35 msとの差の約19 msは、48回のgraph起動と完了待ちの費用（1層約0.4 ms）です。完了待ちを`hipStreamQuery`のspinにしても変わらず、wake-upの遅れではありません。CPU poolがあるときに増える約22 msは、GPUが待つ間のclock低下などが考えられますが、確かめていません。これらの診断用の切り替えはcommitしていません。

<a id="native-and-spec"></a>
## 2026-10-13: `--native`と推測デコード

### `--native`

これまでのWindowsの実行は、canonical packのdense重みと汎用kernelで動かしていました（`--native`なし）。Linuxのsetupが既定で使う`--native <shard 1>`は、projectionをGGUFから読みnative kernel（BF16 projection、融合GDN、split-K attention、block top-k、native QSA indexer、Q5_K head）を使います。同じhello入力で、Decodeは8.74、9.17から12.87、12.99 tok/s、Routing待機は73〜99から46 ms/位置になりました。生成した41 tokenと53位置のargmaxは同じで、logitsは最大1.31異なります（kernelの計算順序の違い）。layer graphは2,341から1,452ノードです。

### captureの一般化

engineのcaptureを使う処理（verify window、MTP drafter）は、普通のengine streamで記録します。AdaptiveCppビルドでは、記録中のstreamへの投入を`Runtime::stream`がそのstream用のnative in-order queueへ振り替え、HIPのstream captureで記録します。`stream_fork`の分岐先も同じqueueへ振り替えて直列に記録し、`stream_join`で戻します。tokenは引き続きpool経路（直接loop）で動き、`session_loop`とtoken graph（GPUがdoorbellを待つ）は使いません。captureは既定で有効です（`STRATA_ACPP_CAPTURE=0`で無効）。

### 推測デコード

verify window（`--spec`）は、残る前提（device residency table、`--expert-profile`、expert cache）を満たせば動きました。AdaptiveCppビルドでは、verify windowを常にsegmented（層ごとにhostがsegmentを起動）にします。doorbellの検査とsegmentなしのwindowはGPUにhostを待たせ、WindowsのdriverがTDRと扱う恐れがあるためです（`STRATA_VERIFY_SEGMENTED=0`で検査に戻せます）。

最初のMTPの下書きは148 ms/roundでした。下書きの待ちloopが1024回ごとに`stream_idle`を呼び、AdaptiveCppビルドの`stream_idle`は呼ぶたびにmarker kernelを投入していたため、待つ間に数千のsubmissionがGPUへ積まれていました。`strata::gpu`が各streamに最後に投入した操作（graph起動、copy、memset、event）のeventを覚え、`stream_idle`はそれを問い合わせるだけにしました。下書きは3.3 ms/roundになりました。この表はprocess終了時に破棄しません（runtimeの後片付けの後でeventを破棄して0xC0000005で落ちたため）。

hello入力、`--native`、expert cache auto（commitの空きで8,868〜8,923 slots）、`--expert-profile`、`--spec 4 --spec-min-p 0.5`、MTP、pool workersは既定（11 + host）で、Decodeは21.91、20.71 tok/sでした。生成した41 tokenは推測なしと同じです。12 roundで1 roundあたり3.33 token、下書きの採択率は90〜93%です。1 roundのうちCPU expertsが109〜117 ms、GPUの待ちが27〜29 msで、残る主な時間はCPU expertsです。CPU expertsはworker 3から11で148から115 ms/roundにしか減らず、mmapされたexpertの読み込み（RAM 27.6 GB、多くはSSD）に律速されていると考えられます。PrefillはcacheのあるときCPU経路のhitを通るため5 tok/s前後です。

pool workersはこれまで`--pool-workers 3`で測っていました。推測なしの`--native`でも既定のworker数で13.39、13.85 tok/sです。

<a id="cpu-side"></a>
## 2026-10-13: CPU側の時間とprefill

- pool workersは既定（物理core数から決まる11 + host）が速く、3から11で推測デコードのCPU expertsは148から115 ms/roundに減りました。比例しないのは、mmapされたexpertの読み込みに律速されているためと考えられます。
- `--resident-budget-gib 6`は、空きRAMから8 GiBの余裕を引くと0 GiBに切り詰められ、効きませんでした。
- expert cacheの大きさがそのまま効きます。同じ条件でcache 11.42 GiBのときDecode 21〜22 tok/s、13.35 GiBで23.00 tok/sでした。Windowsではdevice allocationがcommitを消費するため、cacheの上限はcommitの空き（page fileの大きさ）で決まります。
- `--prefill auto`（batched prompt path）はWindowsでもそのまま動きました。191 token（計算問題の入力を繰り返したもの）のpromptで、Prefillは18.73 tok/s、Decodeは18.76 tok/sでした。1 tokenずつのpromptでは各tokenがexpertをmmapから読み直し、同じ入力で空きRAMが227 MiBまで下がったため途中で止めました。12 tokenのpromptでは5.31から6.19 tok/sです。expertのhostからの転送はDMAを使わず（AdaptiveCppでは任意のhost memoryを登録できない）、12 tokenでは1,337個の転送に1,469 msかかりました。

このPCでの推奨構成は`--native`、`--expert-cache auto --expert-profile`、`--prefill auto`、`--spec 4 --spec-min-p 0.5 --mtp`、既定のpool workersです。Git対象外の`.tools/run-xestrata-fast.bat`がこの構成で起動します（context 4096、hello入力でDecode 18.19 tok/s）。

<a id="page-file"></a>
## 2026-10-13: page fileを広げてexpert cacheを全量に

page fileをC:の固定40〜48 GiB（初期40,960 MiB、最大49,152 MiB）にすると、commitの上限は約37,000から69,241 MiBになり、expert cache autoは20,434 slots（26.31 GiB、LinuxのB70の21,062に近い）を取りました。

profileでcacheを埋める間に、空きRAMが200 MiB前後まで下がりました。Windows用の`madvise`互換関数が`MADV_DONTNEED`を何もしない実装で、VRAMへ写したexpertのmapped pageがworking setに残り続けたためです。`MADV_DONTNEED`を`VirtualUnlock`（lockしていないpageをworking setから外し、file cacheのstandbyへ戻す）で実装し、`FileExpertSource::release`をWindowsで実装し、profileのfillで1個写すごとに手放すようにしました。Linuxの動作は変えていません。最小の空きRAMは18.4 GiBを保ちました。

hello入力、`--native`、cache auto（26.31 GiB）、`--expert-profile`、`--prefill auto`、`--spec 4 --spec-min-p 0.5`、MTPで、Decodeは35.86、36.16 tok/s、Prefillは13.74、13.67 tok/s、hitは96.6%（CPU expertsは1層あたり1.00個）でした。生成した41 tokenは同じです。191 tokenのpromptでは、Prefill 81.73 tok/s、Decode 40.73 tok/sでした。
