<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# V100 32 GB 1枚: 追加測定

PR #8 の作者コメントに応じたシングルGPUの追加測定です。
[English](README.md)

## 結果

各2回の中央値。会話は公式 `ab_usage`、固定入力は `ab_generate` を使いました。

| Measurement | Strata v0.1.42 | XeStrata xe0.1.40.2.2 |
| --- | ---: | ---: |
| Japanese decode (tok/s) | 65.1 | 64.9 |
| English decode (tok/s) | 78.8 | 76.6 |
| Agent decode (tok/s) | 77.5 | 72.2 |
| Japanese TTFT (s) | 0.54 | 0.68 |
| English TTFT (s) | 0.54 | 0.72 |
| Agent TTFT (s) | 3.87 | 4.25 |
| Agent prefill (tok/s) | 1029 | 937 |
| 1K generate decode (tok/s) | 78.9 | 80.0 |
| 1K generate prefill (tok/s) | 693 | 679 |

日本語の生成速度はほぼ同じでした。2GPU時より差は小さいですが、並列枠も変えたため、
GPU枚数だけの効果を分離した比較ではありません。

## 条件

CPU、RAM、ドライバー、ビルド、モデルは[2GPUの報告](../2026-10-11-community-v100-x2-official-usage/README.ja.md)と同じです。
PCIeが速いGPU0 (01:00.0) のV100のみ使用、150 W。GPU1の使用メモリは0 MiBでした。

- 会話: 32K、INT8 KV、1並列枠。日本語5・英語5・agent7ターンを各2回、計68ターン。
- 2GPU報告では2枠でした。今回も同時リクエストは1本です。
- 同じexpert profile、IQ3_S、MTP spec 4 / min-p 0.5、expert-cache auto、prefill auto。
- 各エンジンにウォームアップ1回。会話順序はXe→通常版、通常版→Xe。
- 会話の温度0.7、top_p 0.8、top_k 20、最大1,024トークン。
- 固定入力: 公式1K段、入力1,016トークン、出力256、8Kコンテキスト。
  各エンジンのウォームアップを除外し、通常版の最初の出力を残りの測定が `--spec-follow` で追います。
- 通常版のデフォルトのroute tail skipとPCIe分担自動調整は無効化していません。
  ログではPCIe比率が0.36から0.10へ変更されています。
- 単体serve段、4K以上の固定入力、並列負荷、品質評価は未測定です。

## Voltaコード確認の参考

XeStrataは両方の追従測定で256トークン中7トークンのargmax差を報告しました。
通常版自身の追従測定は0でした。異なるエンジン・既定値の比較なので、
この差をそのままVoltaの不具合や品質低下と判断することはできません。
速度と実行成功だけで数値的正しさを証明したという主張もしません。

カーネル診断はdense-bf16で既定約2,726 µs、no-xmx約22,512 µs、no-bf16-mma約2,909 µsでした。
表示上は既定がoneMath (cuBLAS) / joint_matrix、no-xmxがDP4aです。
これはログの経路名であり、実際の命令を逆アセンブルで確認した結果ではありません。
expert-gate-up/downはoffで、診断されていません。

## ファイル

- [会話集計](matrix-usage.json)、[ターン別記録](usage-turns.json)、[設定](configs.json)。
- [固定入力集計](matrix.json)、[固定入力ログ](generate-logs.txt)、[カーネル診断](kernel-bench.txt)。
- [XeStrataログ](xe-engine.log)、[通常版ログ](up-engine.log)。

個人のパスとGPU UUIDを置換しました。ログは合成ベンチのみです。
開始前の補助スクリプト構文エラーは修正して再実行し、測定には含めていません。
測定後、本番の2GPUチャットとLINE Botは復帰済みです。
