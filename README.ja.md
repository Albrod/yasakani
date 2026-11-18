# YASAKANI（ヤサカニ）

[English](README.md) · 日本語

<a name="これは何か"></a>

YASAKANI は Windows PC の RTX 1 枚で Qwen3.8-27B を動かす推論エンジンです。初回のセットアップ後は、自分の PC だけでチャットやコード編集ができます。普段使っている開発ツールからも OpenAI 互換 API で接続できます。

**[Windows 版をダウンロード](https://github.com/yasakani-dev/yasakani/releases/download/v0.1.6/yasakani-quickstart-0.1.6.zip)** · [8エンジンとの速度比較](docs/MEASUREMENTS.ja.md#思考なしreadme-と同じ表反復等を除いた平均toks) · [コード編集の実演動画](https://github.com/yasakani-dev/yasakani/releases/download/v0.1.6/fast_live_uncut.mp4)

<a name="2-つのモデル"></a>
<a name="必要なもの"></a>
<a name="限界"></a>

## モデルと対応環境

![RTX 5090 1枚での JSONL 編集（思考なし）。推奨の品質モデルは本体 12.37 GB で 2,243 tok/s、最速モデルは本体 10.31 GB で 2,298 tok/s。文脈上限は 262,144 トークン。](docs/assets/hero-ja.svg)

普段使いには元モデルの予測に近い**品質モデル**をおすすめします。生成速度やファイルの小ささを優先する場合は**最速モデル**を選べます。[品質モデルの配布ページ](https://huggingface.co/yasakani-dev/Qwen3.8-27B-YASAKANI-quality) · [最速モデルの配布ページ](https://huggingface.co/yasakani-dev/Qwen3.8-27B-YASAKANI-speed)

| お使いの GPU | 選べるモデル | Windows での動作確認 |
|---|---|---|
| RTX 5090（32 GB） | 品質・最速 | 開発者が実機で検証済み |
| RTX 3090・3090 Ti・4090（24 GB） | 品質・最速 | ビルド済み、開発者の実機検証待ち |
| 対象の RTX 40・50 シリーズ（16 GB） | 最速。文脈長は空きメモリに合わせて調整 | ビルド済み、開発者の実機検証待ち |

Linux には寄稿者が検証したソースビルド手順があります。AMD と Mac は今後の対応目標です。[対象 GPU の型番と要件](docs/FAQ.ja.md#どの-gpuどの-os-で動く) · [Linux のビルド手順](BUILDING.md#linux-building-from-source)

**GPU が複数のとき**：1つのプロセスが使う GPU は1枚です。`--device N` で、`yasakani info` と nvidia-smi が N 番とする GPU を選びます。0.1.5 では GPU が2枚の PC で `--device 1` が `graph instantiate failed` で止まることがありました（[#20](https://github.com/yasakani-dev/yasakani/issues/20)）。0.1.6 で直しを入れましたが、2枚の GPU の PC ではまだ確かめていません。

## 使いはじめ

Windows 11（64 ビット）、Python 3.10 以上、NVIDIA ドライバー 580 以上、Visual C++ 2015–2022 ランタイム（x64）、ディスクの空き 20 GB を用意してください。

1. Windows 起動用 ZIP（v0.1.6）を `C:\yasakani` などの英数字のフォルダーに展開します。[配布ファイルと SHA-256](https://github.com/yasakani-dev/yasakani/releases/tag/v0.1.6)
2. **`START.cmd`** を開いてモデルを選びます。初回はエンジンとモデル一式を取得し、ファイルと GPU の自己点検まで行います。
3. メニューでブラウザーチャットかターミナルを選んで、質問を入力します。別のアプリから使う場合はサーバーを起動し、接続先を `http://127.0.0.1:8765/v1` に設定してください。

ターミナルで思考時間を短くするには `/effort medium`、思考をオフにするには `/think off` を入力します。[手動インストール](python/README.md#install) · [API の使い方](python/README.md#the-server)

## コミュニティの変換

[tools/convert/](tools/convert/) の変換器でほかの人が作ったファイルです。**このプロジェクトは確かめておらず、保守もしていません**。
動く状態に保つのは下に名前のある人です。新しい版に追いつかなくなった変換は、その版でこの一覧から外すことがあります。

**Arete-27B**（作成：[@audreyt](https://github.com/audreyt)）：Nous Research が Qwen3.8-27B をファインチューンした [NousResearch/Arete-27B](https://huggingface.co/NousResearch/Arete-27B) を、
コミュニティで変換したものです。品質モデルと同じ設定で作り、品質モデルの下書きモデルをそのまま使います。ファイルは
[audreyt/Arete-27B-YASAKANI-quality](https://huggingface.co/audreyt/Arete-27B-YASAKANI-quality/tree/659effbdf9dc103fcc053e3da7858bcff7f14cc9) のリビジョン `659effb` です。
全ファイルの SHA-256 はその `SHA256SUMS` にあります（モデルファイル `arete_quality.yskn`、12,365,242,368 バイト、
`541c9206ede621bd92e78ff09f838fca5e4660d834fa883b3d20cdb4ab56880a`）。

```bat
hf download audreyt/Arete-27B-YASAKANI-quality --revision 659effbdf9dc103fcc053e3da7858bcff7f14cc9 --local-dir D:\models\arete-quality
yasakani selfcheck --model-dir D:\models\arete-quality --dll D:\yasakani\yskn.dll
```

そのあとは [使いはじめ](#使いはじめ) と同じように `--model-dir D:\models\arete-quality` を付けて使います。次の値は、この変換を作った人が RTX 5090 で測ったものです。

* 元の BF16 の Arete-27B との近さ（`tools/quality/`、思考文 20 本、KV5 のキャッシュ、0.1.3 で測定）：KL 0.0116、第 1 候補一致率 95.64%。
  品質モデルを同じ方法で Qwen3.8-27B と比べた値は 0.0126 と 95.35% です。
* 0.1.4 で：`yasakani verify` は 11 ファイル中 11 ファイル、`yasakani selfcheck` は 3 件中 3 件が合格（出力は 0.1.3 と同じ）。
  先読みの 1 回の計算で進んだトークン数は、文章 2.40・思考あり 2.48・コード 4.68・CSV → JSON 15.04 です。思考文では品質モデルより少なく、作った人が同じカードと 0.1.3 で両方を測ると 2.48 対 3.24 でした。
  そのため思考文では、速さの表の値より遅くなると考えてください。
* そのほかの評価は、設定と注意点とともに[モデルカード](https://huggingface.co/audreyt/Arete-27B-YASAKANI-quality/tree/659effbdf9dc103fcc053e3da7858bcff7f14cc9)にあります。

<a name="速さ"></a>
<a name="014-で変わったこと"></a>

## 速度と品質

同じ PC の RTX 5090 1枚（定格600 W）で、1件ずつ生成して比較しました。思考なしでは反復などを除いた平均、思考ありでは課題ごとの最速回を使うと、最速モデルは掲載した7分野すべてで他の8エンジンを上回りました。特に差が出たのは入力の大部分を書き戻す編集です。コード編集では最速モデルが **2,025 tok/s**、MegaCapybara 1.39 Tiny が **441 tok/s** でした。

自分の PC の速さは `yasakani bench` で測れます。`yasakani bench --edit` は編集の課題3つ（コード編集・JSONL 編集・長い JSONL 編集）を動かし、速さ、1回の計算で進んだトークン数、答えが期待どおりかを表示します。編集の数字を支える広い写しの計算は計算能力 12.x（RTX 50 シリーズ）でだけ使い、測ったのは RTX 5090 だけです。[編集の速さの測り方](docs/FAQ.ja.md#編集の速さを自分のパソコンで測るには)

<details>
<summary>7分野の比較表を開く（思考なし・思考あり）</summary>

**思考なし**は各課題2回の平均です。反復や指示外のツール呼び出しが出た回は全エンジン共通の基準で除外しています。YASAKANI は11月15日に v0.1.4 で再測定しました（0.1.5・0.1.6 も GPU のコードは同じ）。単位は入力処理を除く生成 tok/s です。

| エンジン | コード | 物語 | 翻訳 | 構造化 | コード編集 | JSONL 編集 | 長い JSONL |
|---|---|---|---|---|---|---|---|
| **YASAKANI 最速モデル** | **512** | **195** | **479** | 882 | **2,025** | **2,298** | **1,214** |
| **YASAKANI 品質モデル** | 468 | 172 | 464 | **892** | 2,008 | 2,243 | 1,115 |
| MegaCapybara 1.39 Tiny | 465 | 164 | 268 | 599 | 441 | 658 | 666 |
| mkyt v0.3.2 | 395 | 181 | 374 | 555 | 423 | 634 | 639 |
| ignis v0.9.6 | 330 | 167 | 298 | 439 | 331 | 510 | 514 |
| NInfer（11/2 版） | 264 | 123 | 257 | 386 | 295 | 443 | 438 |
| vLLM 0.31.0 | 290 | 148 | 261 | 354 | 241 | 426 | 431 |
| SGLang 0.5.22 | 252 | 129 | 246 | 371 | 430 | 466 | 451 |
| llama.cpp b11700 | 242 | 133 | 232 | 337 | 244 | 360 | 362 |
| TensorFold 0.6.6 | 181 | 104 | 172 | 405 | 383 | 408 | 261 |

**思考あり**は課題ごとの最速回を分野別に集計しました。YASAKANI は11月8日の旧版で5回、比較先は2回から選んでいます。

| エンジン | コード | 物語 | 翻訳 | 構造化 | コード編集 | JSONL 編集 | 長い JSONL |
|---|---|---|---|---|---|---|---|
| **YASAKANI 最速モデル** | **337** | **287** | **497** | **536** | 667 | **1,108** | **704** |
| **YASAKANI 品質モデル** | 299 | 242 | 428 | 480 | **743** | 1,053 | 634 |
| MegaCapybara 1.39 Tiny | 290 | 206 | 308 | 429 | 550 | 663 | 516 |
| mkyt v0.3.2 | 244 | 196 | 348 | 384 | 378 | 603 | 454 |
| ignis v0.9.6 | 211 | 130 | 276 | 277 | 278 | 485 | 327 |
| NInfer（11/2 版） | 172 | 140 | 258 | 263 | 263 | 404 | 322 |
| vLLM 0.31.0 | 131 | 112 | 197 | 195 | 187 | 369 | 253 |
| SGLang 0.5.22 | 157 | 124 | 206 | 230 | 257 | 402 | 282 |
| llama.cpp b11700 | 140 | 122 | 184 | 197 | 206 | 349 | 221 |
| TensorFold 0.6.6 | 134 | 79 | 167 | 183 | 439 | 360 | 268 |

比較先は11月3日・6日に測定し、重み・量子化・KV・高速化には各エンジンの設定を使いました。[測定条件・全回平均](docs/MEASUREMENTS.ja.md) · [再現用の記録とスクリプト](bench/README.ja.md)

</details>

<a name="コード編集を最後まで"></a>

11月8日のカットなし録画では、最速モデルが Python ファイルを編集し、送信から **5.56 秒**で全文を返しました。生成コードは17,701件の入力で原本と同じ結果になりました。[時計付きの録画と検証記録](bench/20261108/demo/README_JA.md)

<a name="品質"></a>

品質モデルは、変換に使っていない思考文20本の20,460か所で、元の BF16 モデルと次のトークンの第1候補が **95.24%** 一致しました。同じ容量帯の GSQ-RCO IQ3_S は **93.64%** です。この比較での YASAKANI は FP16 KV を使用し、配布版の既定 KV5 でも **95.35%** でした。[13構成の品質比較と測定方法](docs/MEASUREMENTS.ja.md#思考文の品質比較)

<a name="長い文脈"></a>

文脈上限の262,144トークンを確保したとき、補助モデルを含む GPU メモリの使用量は品質モデルで約21.2 GiB、最速モデルで約19.1 GiB でした。長文での品質は128Kまで測っています。[KV5 の長文測定](docs/MEASUREMENTS.ja.md#最速モデルのkv5独自圧縮のキーバリューキャッシュ)

<a name="仕組みかんたんに"></a>
<a name="ドキュメントと開発参加"></a>

## 開発と支援

YASAKANI はモデルの圧縮形式と GPU の計算を一緒に設計しています。コード編集では、入力からそのまま使えそうな箇所を本体がまとめて検証するため、変えていない行を書き戻す時間を短縮できます。[技術解説](docs/TECHNOLOGY.ja.md) · [ファイル形式](docs/FORMAT.ja.md) · [変換ツール](tools/convert/)

<a name="エージェントとして使う"></a>

開発者向けにソースからのビルド手順、GPU の動作報告の方法、ブラウザーで使う任意のエージェント画面（24 GB 以上・旧版サーバー同梱）も公開しています。[ビルド](BUILDING.md) · [開発に参加](CONTRIBUTING.md) · [エージェントの導入](harness/HARNESS_README.txt)

<a name="支援"></a>

日本で開発したこのエンジンを、もっと多くの PC で使えるようにしたいと考えています。ご支援はまず24 GB・16 GB の RTX の実機検証と調整に充てます。掲載を希望する方は、このリポジトリと今後の版の起動時クレジットでご紹介します。[CAMPFIRE（日本在住の方）](https://camp-fire.jp/projects/992233/view) · [GitHub Sponsors](https://github.com/sponsors/yasakani-dev)

<a name="ライセンス"></a>

Apache License 2.0 で公開しています。[ライセンス](LICENSE) · [著作権表示](NOTICE) · [第三者の著作物](THIRD_PARTY_NOTICES.md) · [更新履歴と寄稿者](docs/RELEASE_NOTES.ja.md)
