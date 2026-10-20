<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->

# Windows での使い方

Windows で AMD と Intel の GPU を使う配布用の zip（`XeStrata-<版>-windows-x64-amd-intel.zip`）の使い方です。
導入も起動も設定の変更も、zip の中の `xestrata-cli.bat` でします。
`xestrata-cli.bat` は受け取ったオプションをそのまま setup に渡すので、オプションは Linux の `xestrata` と同じです（`xestrata-cli.bat --help`）。

## 入れる

1. zip を浅いフォルダー（`C:\XeStrata` など）に展開します。中のパスが長いので、深いフォルダーでは展開に失敗することがあります。
1. `xestrata-cli.bat` を実行します（ダブルクリックでも動きます）。GPU とモデルの選択、ダウンロード、モデルの準備を端末で進め、終わると起動します。

GPU のドライバー（AMD は AMD Software、Intel は Intel のグラフィックスのドライバー）と [Visual C++ の再頒布可能パッケージ](https://aka.ms/vs/17/release/vc_redist.x64.exe)が要ります。
Visual C++ の再頒布可能パッケージがなければ、`xestrata-cli.bat` が知らせて止まります。.NET は要りません。
ディスクには、モデルのファイルとパックの分の空きが要ります。足りなければ setup が知らせて止まります。

## 起動と停止

- 2 回目からも同じ `xestrata-cli.bat` を実行します。入れてあるモデルがすぐ起動します。
- 起動するとブラウザで `http://127.0.0.1:8095` が開きます。アプリからは、OpenAI 互換の接続先として `http://127.0.0.1:8095/v1` を指定します。
- 止めるときは、モデルを動かしている端末のウィンドウを閉じます。
- `xestrata-cli.bat --gpu 1` は、この起動だけその GPU で動かします。GPU の番号は `xestrata-cli.bat --check` で見られます。

## 設定の変更

設定を変えるときは `xestrata-cli.bat --setup` を実行します。導入のときと同じ質問をもう一度たどります。
入れてあるモデルの大きさが最初から選ばれているので、Enter で進めば別のモデルをダウンロードし直すことはありません。モデルのファイルとパックはそのまま使います。
変えたいところだけをオプションで渡すこともできます。

| 変えたいもの | オプション |
| --- | --- |
| コンテキストの長さ | `--setup --context 131072` |
| KV キャッシュの精度 | `--setup --kv q4_0`（メモリが半分、精度が少し下がる） |
| 別の大きさ・別のモデル | `--setup --family gyro --model Gyro-M` など |
| ほかの PC やスマートフォンからつなぐ | `--setup --host 0.0.0.0 --api-key <秘密の文字列>` |
| ポート | `--port 8096` |
| 画像の読み取り | `--setup --vision yes`（GPU 版か CPU 版かを決めるときは `gpu` か `cpu`、やめるときは `no`） |
| ほかのプログラムのために空けておく VRAM | `--setup --vram-reserve-mib 3072` |
| モデルの置き場所 | `--setup --data-dir D:\XeStrata-data` |

`xestrata-cli.bat --calibrate` は、エンジンの設定のいくつかをこの PC で測って決めます（15〜30 分ほど）。結果は次の起動から使われます。

設定のファイルは `%APPDATA%\xestrata\xestrata-<モデル>.json` です。
setup が書かない項目（`env` に足した環境変数など）は `--setup` のあとも残りますが、エンジンのオプション（`args`）に手で足したものは `--setup` で捨てられます（どれを捨てたかは setup が表示します）。
エンジンのオプションは、できるだけ setup のオプションで変えてください。

## 持っているモデルのファイルを使う

ダウンロードしてあるモデルのファイルは、次のどちらかの方法で使えます。

- そのフォルダーのまま使う: `xestrata-cli.bat --family gyro --model Gyro-M --gguf-dir D:\models\Gyro` のように、`--gguf-dir` にファイルのあるフォルダーを渡します。コピーも移動もしません。
- データのフォルダーに入れる: setup が「The model files go in ...」と示すフォルダーに、元の名前のまま入れます。

どちらの方法でも、ファイル名は配布元と同じにしてください（`Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf`、分かれているものは `...-00001-of-00002.gguf` など）。

Gyro など、固定した版と照らし合わせるファイルは、初回に SHA-256 を確かめます。大きいファイルでは数分かかり、2 回目からは確かめません。
SHA-256 が配布元の版と合わないとき、setup はそのファイルを消さずに止まります。別の版のファイルなら、フォルダーの外に移してから setup をもう一度実行すると、固定した版をダウンロードします。

Gyro は SSD に置いてください。ファイルのおよそ半分は n-gram の表で、エンジンは 1 語ごとにその表を読みます。HDD では遅くなります。

## 置き場所

zip のフォルダーには書き込みません。

| もの | 場所 |
| --- | --- |
| 設定 | `%APPDATA%\xestrata` |
| ログとエンジンの作業フォルダー | `%LOCALAPPDATA%\xestrata\state` |
| モデルとパック（データのフォルダー） | 既定は `%LOCALAPPDATA%\xestrata\data`（`--data-dir` で変えられます） |

C: の空きが少ないときは、導入のときに `--data-dir` で別の SSD を指してください。

## 更新と削除

- 更新するときは、新しい zip を別のフォルダーに展開して、そちらの `xestrata-cli.bat` を実行します。設定とモデルはそのまま使います。
- 消すときは、`xestrata-cli.bat --remove-data` で XeStrata が置いたモデルのファイル、設定、ログを消し（先に確かめます）、そのあとで zip を展開したフォルダーを消します。
