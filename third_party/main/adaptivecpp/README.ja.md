<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# AdaptiveCppのWindows向け局所パッチ

対象はAdaptiveCppのcommit `0ea863abe1af4422593583f91e5608ef75044a16`だけです。AdaptiveCpp本体は同梱せず、既存のローカルcheckoutを使います。ダウンロードは行いません。0001はevent handler、0002はlane照合の修正です。

`patches/0001-event-async-handler.patch`は、`sycl::event`のnode付きconstructorが受け取った`async_handler`をメンバーへ保存する1行の修正です。パッチはAdaptiveCppと同じBSD-2-Clauseで、元のcopyright・license headerを変えません。原文のライセンスは`LICENSE`に保存しています。XeStrataのビルド用Pythonコードとこの文書はLGPL-3.0-or-laterです。

## ビルド前の適用

`tools/adaptivecpp_build.py`をAdaptiveCpp単独ビルドの入口として使います。

```text
python tools/adaptivecpp_build.py --source PATH_TO_ADAPTIVECPP --build build/acpp-eventfix --prefix .tools/acpp-eventfix --receipt build/acpp-eventfix-receipt.json --cmake-arg=-G --cmake-arg=Ninja
```

compiler、CPU/GPU backend、LLVMやOpenMPの場所は`--cmake-arg=-DNAME=VALUE`で明示します。現在のWindows環境ではVisual Studioの開発用shellを使います。LLVMへの組込みビルドはこの単独ビルドとは別です。その場合も`--apply-only`を構成前に実行し、install後に`--verify-installed-only --prefix PATH`で確認します。

wrapperは次の順に処理します。

1. checkoutのHEADが上記SHAと一致し、commit中の`event.hpp`が固定SHA-256と一致することを確認します。
2. working headerが原文または修正済みの全文と一致することを確認します。LFとCRLFの正規の2通りだけを認め、混在や別の編集は拒否します。
3. 未適用なら`git apply --check`後に適用し、適用済みならreverse checkだけを行ってスキップします。ユーザーの変更をreset、revert、削除しません。
4. CMake configure、build、installを実行します。source、build、prefixの重複、既存CMake cacheの別source/prefix、未修正の既存install、既存receiptの上書きは拒否します。
5. `PREFIX/include/AdaptiveCpp/hipSYCL/sycl/event.hpp`が修正済み全文と一致することを確認します。commit SHA、patch・headerのSHA-256、適用結果をJSONへ記録します。

不整合やcommand失敗時は停止します。直接CMakeを実行するとこの検査は通らないため、更新時の入口にはしません。正常版のsource、build、prefixは復旧用として残し、別の場所で作業します。

commit中のLF原文のSHA-256は`0576c693880346dcd6d85ec3934be332c8e1678af7f126c267d603916338b2ad`です。WindowsのCRLF原文は`0eb54a547e68eaf0055f75431fbc8af7186e7221988d0829c333acec76797526`です。Git blobとworking fileのハッシュは区別します。パッチは空白だけのcontext行を避けるため後続contextを省いています。wrapperは全文ハッシュの一致を前提に`--unidiff-zero`を指定し、適用後の全文も検査します。未知のheaderへ無条件に適用するものではありません。

## 検証と範囲

`python -m unittest tools.test_adaptivecpp_build -v`は、ローカルGit fixtureで適用・スキップ・SHA違い・header改変・install不一致・保護条件を確認します。

`tests/platform/adaptivecpp_event_async.cpp`は、実際のinstalled headerとruntimeを使う隔離テストです。CPU backendだけの別prefixで実行し、GPU backendをinstall・探索させません。device queue、device allocation、kernel、DMAは使いません。制御されたsynthetic runtime errorを登録し、handlerの呼出し回数、元のdiagnostic、SYCL errc、複数error、event copy、再通知なし、throwするhandlerを確認します。同じCPU runtimeと未修正event headerでの`--expect-broken`はnegative controlです。

0001はqueue順序、DAG、正常時の演算、XeStrata本体を変えません。default生成eventのhandler挙動、process-global error一覧、cancelやDMA完了不明時のcleanupは別課題です。このテストをGPU障害時のcleanupや実モデル生成の検証とは扱いません。

<a id="lane-affinity-patch-0002"></a>
## 0002: lane照合の範囲と隔離ビルド

`patches/0002-dag-lane-affinity.patch`は同じ固定commitの`src/runtime/multi_queue_executor.cpp`を対象にし、runtime動作の変更は`find_assigned_lane_index`に限定します（ライセンス帰属表示も追加します）。親multi_queue_executorが登録されたDAG nodeについて、先行laneを子executorの所有判定ではなくnodeのexecution laneと子のqueueで照合します。所属・backend・device範囲の確認を先に行い、foreign nodeの未初期化laneは読まない設計です。kernel/copy laneを統合せず、必要なstream間依存やXeStrataのfinal-copy待機を外しません。原ソースとの差分は`.tools/lane-affinity-runtime/source.diff`に保存しています。

再現時は別checkoutで0001を適用した後、`python tools/adaptivecpp_lane_affinity.py --source PATH_TO_ISOLATED_ADAPTIVECPP --apply-only`を実行します。build/install後は同じsourceへ`--verify-only`を実行します。toolは固定commitと対象sourceの原文・適用後全文のSHA-256を確認し、不一致なら停止します。buildやinstallは行いません。別source/build/prefixだけで作業し、正常版のsource/build/install、既存の0001検証環境や復元用コピーは上書きしません。上の0001用手順だけで0002の適用・検証が済んだとは扱いません。新規構築物は今回の隔離DLLと別のbinaryになるため、使用前に数値gateが必要です。

Windows R9700で正常版Aと隔離runtime Bの53位置・41出力tokenを比較し、全13,160,960 logitsと26,593,280 residual floatsがbit単位で一致しました。A/B交互の各3回ではDecode中央値2.576→3.846 tok/s（+49.3%）、host Routing/Input待機261.091→122.509 ms/位置、CPU callback100.624→101.673 ms/位置でした。全lane統合の別案は248,320要素すべてが古い0となり不採用です。10倍の改善や他のworkloadでの再現を示す結果ではありません。詳細はGit対象外の`build/windows-acpp/lane-affinity-20261012T084944Z/report.ja.md`と`result.json`、進捗記録は`docs/windows-port-progress.ja.md`を参照してください。
