<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# V100 32 GB 2枚: 新版同士の公式 usage ベンチ

XeStrata xe0.1.40.2.2 と Strata v0.1.42 を同一PCで交互に測定しました。
以前の PR #3 の短い独自課題ではなく、今回追加された `tools/bench_core.py` の
`ab_usage`、会話、サンプリング、集計を変更せずに使っています。
この環境では、日本語・英語・コーディングのすべてで通常版が速い結果でした。

[English](README.md)

## 結果

各2回の中央値。速度は各回・各会話のトークン数合計 / 時間合計を計算してから集計。
TTFTは各回のターン中央値をさらに2回で集計します。

| 測定 | Strata v0.1.42 | XeStrata xe0.1.40.2.2 |
| --- | ---: | ---: |
| 日本語 decode (tok/s) | 79.2 | 68.2 |
| 英語 decode (tok/s) | 88.5 | 80.5 |
| agent decode (tok/s) | 87.2 | 75.6 |
| 日本語 TTFT (秒) | 0.64 | 0.96 |
| 英語 TTFT (秒) | 0.66 | 0.84 |
| agent TTFT (秒) | 4.53 | 5.29 |
| agent prefill (tok/s) | 876 | 743 |

## 条件と再現方法

- Ryzen 9 9950X、128 GB RAM、Ubuntu 24.04.5、NVIDIA 580.178.04。
- Tesla V100 PCIe 32 GB 2枚、各150 W。PCIe転送プローブは約13.1 / 3.2 GB/s。
- IQ3_S GSQ-RCO、32K、INT8 KV、MTP spec 4 / min-p 0.5、expert-cache auto、prefill auto。
- 両エンジンとも24層ずつ、並列枠2、実際のリクエストは1本。
- 同じ固定expert profileを共有し、追加のV100最適化スイッチは設定していません。
- XeStrataはcontrib-llvm、CUDA 12.8 / sm_70。通常版もCUDA 12.8 / sm_70でビルド。
- 各エンジンのウォームアップ1応答を除外。順序はXe→通常版、通常版→Xe。
- 温度0.7、top_p 0.8、top_k 20、最大1,024トークン。日本語5、英語5、agent7ターン。
- 本測定68ターンすべて完了。GPUコア温度の最大は52℃。

`conditions.json` に測定コミット、`configs.json` に実際の起動設定があります。
`<HOME>` と `<GPU_UUID>` は公開用の置換です。環境に合わせて戻してください。
モデル・pack・MTP・expert profileのファイル本体とチェックサムは添付していません。

再現時は測定タグの `tools/bench_core.py` を使い、各設定から `Engine` を作り、
`ab_usage(out, "V100x2", engines, config_callback, 32768, load_corpus()["agent"], rounds=2, temperature=0.7)`
を呼びます。サーバーは各エンジン自身の `serve/server.py` を使います。
その後 `collect` で集計します。設定内のポートは各起動で空きポートに置き換わります。

## 公式CLIとの差分と制限

タグ時点の `bench.py:usage_cfg` は `--original-strata` 指定時に `split_pci`、
`layer_split`、`parallel`、`env` を引き継ぎません。また通常版の準備は先頭GPUを選びます。
そのため `--bench --original-strata` をそのまま実行した結果ではありません。
公式の `ab_usage` にエンジン別の完全な設定を渡して、2GPU条件を維持しました。
計測・プロンプト・集計の実装は変更していません。

公式CLIと同じく、2GPU分割では単体generate/serveの段階別測定を実施していません。
同時リクエストの負荷試験、品質評価、長い32K入力の速度測定でもありません。
サンプリングで回答が異なるため、同一出力の厳密なカーネル比較ではありません。
短い日本語質問では、prefillの固定処理時間の影響を強く受けます。

## 診断ファイル

- [集計値](matrix-usage.json)、[CSV](comparison.csv)、[ターン別の値とウォームアップ](runs.json)。
- [XeStrataエンジンログ](xe-engine.log)、[通常版エンジンログ](up-engine.log)。
- ログは合成ベンチのものだけです。個人の会話、認証情報、ユーザー名、GPU UUIDは含めません。
- XeStrataは2GPU分割とoneMath (cuBLAS)、joint_matrix経路を報告しています。
  これだけで各カーネルの効率や遅さの原因は断定できません。

本番サービスは試験後に復帰しています。このPRは記録のみで、エンジンの修正はありません。
