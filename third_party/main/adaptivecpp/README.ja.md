<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# AdaptiveCppの暫定eventパッチ

対象はAdaptiveCppのcommit `0ea863abe1af4422593583f91e5608ef75044a16`だけです。AdaptiveCpp本体は同梱せず、既存のローカルcheckoutを使います。ダウンロードは行いません。

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

このパッチはqueue順序、DAG、正常時の演算、XeStrata本体を変えません。default生成eventのhandler挙動、process-global error一覧、cancelやDMA完了不明時のcleanupは別課題です。このテストをGPU障害時のcleanupや実モデル生成の検証とは扱いません。
