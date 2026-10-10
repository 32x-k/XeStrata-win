<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# 速さの測り方

[English](BENCH.md) | 日本語

XeStrata の速さは 3 つの測り方で測ります。
利用者の `xestrata --bench` と、開発者の upstream 追従の計測は、同じ実装（`tools/bench_core.py`）で、同じプロンプト・会話・設定を使います。
そのため、利用者の記録と開発機の記録を同じ表で比べられます。

| 測り方 | 動かし方 | 分かること |
| --- | --- | --- |
| generate | `strata` を 1 回ずつ起動し、段のプロンプトを読んで 256 トークン書く（greedy） | カーネルと推測デコードの速さ |
| serve | `strata --serve` に標準入力で 4 ターンのチャットを送る（段のプロンプトと、続きの質問 3 つ。1 ターン 256 トークン） | サーバーの出力ループと会話のチェックポイント（続きのターンの待ち時間） |
| 通常使用 | `serve/server.py` を起動し、`/v1/chat/completions` にストリーミングで送る | 利用者が使う経路全体（チャットテンプレート、思考、ツール呼び出し） |

段は、コーディングエージェントのプロンプト（XeStrata のソース 58dbbe5 を読ませたもの）の長さで、1K・4K・32K・64K・128K・192K・262K があります。
各段の設定は、setup がその文脈に書くもの（8K を超えると K/V を INT8、64K 以上で `--kv-resident 32768`）です。
通常使用の会話は、日本語と英語のチャット 5 ターンずつと、ツールを呼ぶコーディングエージェント 7 ステップです（1 ターン最大 1,024 トークン、温度 0.7・top_p 0.8・top_k 20）。

## xestrata --bench

入れたモデルの、この PC での速さを測ります。
測るのは IQ3_XXS です（リリースの表と追従の計測に並べるため）。RAM の少ない PC など、ほかの入れたモデルは `--model`（Qwen3.8-Flash-Next 以外は `--family` も）で選びます。

```sh
xestrata --bench                                  # 3 つの測り方と、機械の基本の速さ（RAM・PCIe・SSD）
xestrata --bench-verbose                          # さらに段階ごとの時間、カーネルごとの時間、GPU の負荷と温度、答えの指紋
xestrata --bench --bench-rounds 3                 # 各計測の回数（既定 2）
xestrata --bench --model IQ2_XS                   # IQ3_XXS 以外の入れたモデル（Coder は --model IQ1_M --family coder）
xestrata --bench --original-strata v0.1.41        # 同じモデルで upstream Strata のそのタグと交互に測る
```

- 時間は 40 分ほどです（`--bench-rounds` を 1 増やすと 20 分ほど増え、`--original-strata` を付けると約 2 倍）。
- 測っている間は GPU を使います。xestrata のサーバーが動いていれば、止めるように言って終わります。
- generate と serve の段は 64K・32K・1K のうち、設定の文脈に入るものです。通常使用は設定の文脈のまま測ります。
- 複数の GPU に層を分けた設定では、generate と serve は測らず、通常使用だけを測ります。
- 記録はデータフォルダの `records/bench-<日時>/` に書きます。どこにも送りません。

| ファイル | 中身 |
| --- | --- |
| `summary.md` | 報告に貼る要約（貼る前に中身を確かめてください） |
| `record.json` | 機械、選ばれた経路、計測の全部 |
| `tables.md` | 3 つの測り方の表 |
| `runs/`・`serve-runs/`・`usage-runs/` | 1 回ずつのログ |

### --original-strata

指定したタグの upstream Strata を git で取ってきて、データフォルダの `upstream/` に置き、XeStrata と交互に測ります。
エンジンは upstream の setup と同じ方法で用意します。

| GPU | upstream のエンジン |
| --- | --- |
| NVIDIA | リリースの配布物（SHA-256 を確かめる） |
| AMD | タグのソースを HIP でビルド |
| Intel | upstream の `sycl/` を Docker でビルド（sudo なしで docker が使えること） |

- NVIDIA の配布物と Intel の Docker イメージは自由ソフトウェアでないので、free のビルドでは NVIDIA と Intel の比較はしません。
- Intel では出力の速さを比べません。upstream の `sycl/` は、VRAM に入らないエキスパートを PCIe 越しに読むためです。

## upstream 追従の計測

upstream のリリースを取り込むとき（`.claude/skills/upstream-sync/SKILL.md`）は、B70・RTX 4070・RX 9060 XT のそれぞれで、3 つの測り方すべてで XeStrata と upstream を交互に測り、`judge.py` で判定します。

```sh
S=.claude/skills/upstream-sync/scripts
python $S/sync_prompts.py <pack> <ids>                                         # プロンプト（1 回）
XE="..." UP="..." XE_PROFILE=... UP_PROFILE=... DATA=<data> IDS=<ids> python $S/bench_generate.py $o/<file> <GPU>
XE="..." UP="..." XE_PROFILE=... UP_PROFILE=... DATA=<data> IDS=<ids> python $S/bench_serve.py $o/IQ3_XXS <GPU>
XE="..." UP="..." XE_ROOT=... UP_ROOT=... XE_PROFILE=... UP_PROFILE=... DATA=<data> PY=<python> \
  python $S/bench_usage.py $o/IQ3_XXS <GPU>
python $S/sync_collect.py $o/<file> xe-<commit> <tag>                         # matrix*.json と表
python $S/judge.py --prompt-only B70 $o/IQ3_XXS ...                           # ALL PASS か FAIL <n>
```

- モデルの既定は IQ3_XXS です。RAM の少ない PC では、setup が書いたモデルの設定を `CONFIG=xestrata-<モデル>.json` で渡すと、そのモデルのファイルと設定（RAM の予算、calibrate の結果）で両エンジンを測ります。
  ファイルを直接指定するときは `PACK`・`NATIVE`・`PLE`（と両エンジンに足すオプションの `EXTRA`）です。通常使用の文脈は `CONTEXT`（既定 262144）、段は `TIERS` で変えられます。
- `XE` と `UP` は各エンジンのバイナリかラッパーです。前に `env K=V ...` を付けられます。serve と通常使用のラッパーは標準入力を通すこと（`docker run -i`）。
- 回数は generate が 2 回、serve と通常使用が 3 回です（`ROUNDS` で変えられます）。serve と通常使用は各エンジンが自分のテキストを書くので、温度 0.7 では 1 回ごとに中央値から 10% 前後ずれます。
- generate は `--spec-follow` で両エンジンに upstream の答えをなぞらせ、テキストの違いを除きます。serve にはこの機能がありません。
- 合格の幅: generate と、serve・通常使用の最初のプロンプトの速さは、XeStrata が upstream より 1% 以上遅くないこと。serve・通常使用のそれ以外（続きのターンの待ち時間、出力、最初のトークンまでの時間）は 5% 以上悪くないこと。
- `--dry-run` を付けると、エンジンを動かさずに起動するコマンドを出します。

## リリースの表

リリースノートの速さの表は、版をまたいで比べられるように決まった方法で測ります（`.claude/skills/release/scripts/bench.sh`。IQ3_XXS の 1K〜262K の 6 段、交互に 2 回、`--spec-follow` なし）。
上の 2 つとは別の実装で、追従の計測を変えてもこちらは変えません。
