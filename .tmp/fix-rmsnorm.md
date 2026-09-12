# 最終層リードアウト修正: マージ判断メモ

対象: `c628b90`（膜電位読み出し）とその後の訂正。
宛先: このブランチを `master` に入れるメンテナ。

結論から書く。**入れてよい。入れるべきである。** `c628b90` 単体は本番 38 万件学習を 3 epoch 目で破壊した。この訂正を載せないまま膜電位読み出しを残すと、同じ崩壊が再発する。スパイク読み出しへ revert するより、訂正済みの式の方が問題意識に対して正しい。

---

## 1. なぜマージが必要か

`c628b90` は「最終層をスパイク回数ではなく膜電位で読む」という変更である。狙いは妥当で、スパイク 0/1 が閾値下の情報を捨て、学習信号が痩せることへの対処だった。

しかし実装がスケールを取り違えた。膜電位は LIF クリップで ±20 まで振れる。それを毎フレーム RMSNorm して線形層へ渡した。2 epoch までは損失が下がる（78 → 66）が、3 epoch 目で epoch 平均が `1.05e25` になった。表示の崩れではない。`String(format: "%.4f", avgLoss)` に渡った `Float` そのものがその値である。チェックポイント間隔は 5 epoch なので、途中保存もなく学習成果は消えた。

この訂正をマージしない選択肢は二つしかない。

1. `c628b90` を残したまま再学習する。同じ崩壊が起きる。
2. `c628b90` を revert してスパイク回数に戻す。安定はするが、閾値下を捨てるという元の問題が残る。

訂正後の式はスパイクと同じ `[-1, 1]` に閉じ、閾値下はアナログのまま残る。安定と問題意識を両方満たす。**今入れなければ、次の大規模学習はまた 7 時間超えてから壊れる。**

---

## 2. 何が問題だったか

### 2.1 観測

38 万件（約 546 時間）の GPU CTC 学習:

| epoch | 音響損失 | 所要 |
|---|---|---|
| 1 | 78.4107 | 8498 秒 |
| 2 | 65.9578 | 8526 秒 |
| 3 | 1.048e25 | 8956 秒（ここで停止） |

1→2 は健全。3 は有限だが実用上 Inf と同じ。GPU CTC の番兵は `-1e30` で、バッチ平均に数件混ざるだけで epoch 平均がこの桁になる。ネットワークが尖ったあと、ロジットが壊れて経路尤度が番兵まで落ちた、と見るのが自然である。

学習時間が長い主因はこのリードアウトではない。`--cache-features` なしで毎 epoch 38 万本の WAV を再パースしていた。Disk I/O 待ちで CPU/GPU 使用率が低く見えるのはそのせい。本修正の対象外。

### 2.2 `c628b90` の実装がやっていたこと

最終層の積分直後の膜電位 \(v\) を timeSteps で平均し、**フレームごとに RMSNorm** してから `wOut` に掛けていた。

意図: スパイク 0/1 より膜電位の方が情報が多い。±20 のまま線形層に入れると発散するので正規化する。

実際:

- 隠れ層はハードリセット LIF のまま（発火で \(v \leftarrow 0\)）。読み出しだけアナログにした。
- RMSNorm はフレーム内の形だけ残し、**振幅を捨てる**。小さい発火もクリップ飽和も同じエネルギーになる。
- `1/\mathrm{rms}` の勾配は、最終層がほぼ沈黙したバッチで大きくなる。clipNorm 5.0 でも方向は壊せる。
- `wOut` の初期化はスパイク率 `[0, 1]` 用のまま。RMSNorm 後は単位ベクトルなので最初は動くが、学習が進んで膜電位がクリップに張り付くと CTC が先に壊れる。

層間電流の RMSNorm（上位層が沈黙しないためのもの）とは別物である。あれは残してよい。壊していたのは**読み出し側**の RMSNorm だけである。

### 2.3 SpikeYOLO との関係（誤診しやすい点）

着想は [Spiking-YOLO (Kim et al., 2019)](https://arxiv.org/html/1903.06530v2) の「スパイク回数より積算 \(V_{\mathrm{mem}}\) の方が精密」にある。問題意識（離散化で端数が落ちる）は同じでよい。実装の対応は成り立たない。

| | SpikeYOLO | `c628b90` |
|---|---|---|
| 設定 | DNN→SNN 変換の出力デコード | 代理勾配 BPTT + CTC |
| 時間 | 3500〜8000 step | 内部 4 step |
| リセット | \(V \leftarrow V - V_{\mathrm{th}}\)（余りが残る） | 隠れ層はハードリセット |
| スケール | \(V_{\mathrm{mem}}\) の大きさが出力 | 毎フレーム RMSNorm で振幅を捨てる |

CTC の経路周辺化は「どのフレームがどの文字か」を柔らかくする。膜電位の余り（振幅の端数）とは軸が違う。CTC があるから RMSNorm 付きハードリセットが SpikeYOLO 相当になる、ということはない。ロジットを作る前に消えた振幅は、CTC では戻らない。

---

## 3. 訂正後の式

隠れ層（層 0 と中間）は従来どおりハードリセット LIF。スパイクで通信する層は変えない。

**最終層だけ** 次の順である。1 層ネットワークでは層 0 が最終層なので、同じ式が層 0 に掛かる。`numLayers == 1` 用の別配線はない。

1. リークのみ: \(v \leftarrow \mathrm{clip}(\beta v + I, \pm 20)\)。`(1-s)` で 0 にしない。
2. 閾値単位で読んでクリップ: \(\mathrm{clip}(v / V_{\mathrm{th}}, -1, 1)\)
3. subtractive reset: \(v \leftarrow \mathrm{clip}(v - s\,V_{\mathrm{th}}, \pm 20)\)

timeSteps で平均して `wOut` に掛ける。フレーム RMSNorm はない。

| 状態 | 読み出し | 次内部ステップの \(v\) |
|---|---|---|
| 沈黙 \(v=0.4\) | 0.4（アナログ） | 0.4 がリーク |
| 発火 \(v=1.5\) | 1.0（商＝スパイク） | 0.5（余り） |
| 飽和 \(v=20\) | 1.0 | 19。ロジットは ±20 にならない |

発火はスパイクと同じ 1.0、閾値下は残る、余りは 4 内部ステップをまたぐ。`wOut` の入力は `[-1, 1]` に閉じるので、初期化スケールとも揃う。

学習（MLX `logitsBatch`）、推論（`SpikingNetwork.forward`）、量子化エンジンは同じ式である。MLX は層ループが 1 本で、最終層かどうかだけ分岐する。

---

## 4. 何を直して、何を直していないか

直した:

- 読み出し RMSNorm の除去
- 最終層の subtractive reset（余りを残す）
- 閾値単位クリップ（飽和がロジットを壊さない）
- MLX / Swift / 量子化の一致
- 1 層と多層を同じ層ループで扱う（`numLayers == 1` の別経路を廃止）
- 使われなくなった `spikeSumPtr` と量子化側の `membraneSum` / `readoutNorm` を削除

残した（意図的）:

- 層間電流の RMSNorm。上位層への信号を単位スケールにする別件。
- 隠れ層のハードリセット。再帰・層間はスパイク通信のまま。
- GPU CTC の番兵 `-1e30`。構造的に不能な 81 件は既に除外している。数値的に死ぬ経路のガードは CPU CTC にだけある。本修正の範囲外。
- `--cache-features`。学習時間の本命。本修正では触らない。
- CPU `BPTTTrainer`（交差エントロピー経路）はまだスパイク回数。本番の `train --device gpu` は MLX CTC を使う。`--device cpu` で音響学習すると推論と式が違う。フォールバックであり、手書き BPTT 逆伝播をこの訂正に巻き込むと別の欠陥を入れる。

---

## 5. 耐久性

次の組み合わせで、フォワードが一致し、損失が有限で、下がることをテストしている。

| 軸 | 確認 |
|---|---|
| 層数 | 1 層・2 層の MLX と Swift が `1e-3` で一致。3 層は重みの往復。本番は 3 層。 |
| 内部ステップ | `timeSteps = 4`（本番と同じ）。 |
| 発火 / 沈黙 / 飽和 | 沈黙はアナログ、発火 1.5 は読み出し 1.0 で余り 0.5、飽和 20 は読み出し 1.0。SIMD とスカラーがビット一致。 |
| compile / eager | ロジット差 0。5 ステップ後の `wRec` 差 `1e-7` 台。 |
| 系列長バケット | T=31/32/33/64。損失は有限。 |
| 学習率 | compile 済みステップが入力の LR を追う。 |
| CTC | 同一特徴でラベルを変えると損失が変わる。30 ステップで損失が下がる。 |
| 量子化 | 層 0 のみのエンジン。最終層式（余り + クリップ）に合わせた。多層量子化は元から持たない。 |

壊れにくい理由は単純で、読み出しが有界だからである。スパイク読み出しと同じレンジに閉じている。RMSNorm の `1/rms` も、±20 の線形層入力もない。最終層が沈黙してもクリップ勾配は `[-1, 1]` の内側だけ流れる。

それでも CTC 番兵は残る。YODAS のような誤ラベルで経路が構造的に死ぬサンプルは、平均損失をまた巨大にし得る。そのときは損失の印字が壊れるので、ネットワーク発散とは切り分けられる（1–2 epoch が既に巨大なら番兵漏れ、途中からなら重み崩壊）。本修正後は後者の主因を潰している。

Apple Silicon GPU（MLX compile）と CPU フォワード（Swift）は一致テスト済み。Linux / CUDA はリポジトリの対象外。

---

## 6. 机上ではないことの証明

一致テストと「損失が NaN にならない」だけでは、この式が正しいことは分からない。`c628b90` も 2 epoch までは損失が下がっていた。レビューでは次のテストを先に読む。

### 6.1 2 ニューロンのロジット恒等式

`MultiLayerSNNTests.testLastLayerLogitIsThresholdUnitsNotRawVoltageOrSpikeCount`

ネットワークを計算用紙にする。1 層、隠れ 2、出力 1、内部ステップ 1、`wIn = wRec = 0`、`wOut = 1`、`bOut = 0`。電流はバイアスだけ。

- ニューロン 0: \(I = 20\) → 膜電位はクリップ上限。発火する
- ニューロン 1: \(I = 0.4\) → 閾値未満。沈黙する

`wOut = 1` なのでロジットは読み出しの和そのもの。三通りの読み方が三つの数字になる。

| 読み方 | ロジット | 意味 |
|---|---|---|
| スパイク回数 | \(1 + 0 = 1.0\) | 閾値下 0.4 が消える。`c628b90` 以前 |
| 膜電位を生で足す | \(20 + 0.4 = 20.4\) | ±20 が線形層に入る。`c628b90` が 3 epoch 目で壊れた桁 |
| `clip(v/vTh, -1, 1)` | \(1.0 + 0.4 = 1.4\) | 発火は 1、沈黙のアナログは残る。この訂正 |

テストは **1.4 であること**、**1.0 ではないこと**（0.4 がロジットに届いている）、**20.4 ではないこと**（生の飽和が届いていない）を、Swift 推論と MLX `logitsBatch` の両方で見る。スパイクに戻しても、RMSNorm 前の生電圧に戻しても落ちる。数字が式そのものなので、コメントを信じなくても再現できる。

### 6.2 余りが次ステップに残ること

`LIFNeuronTests.testReadoutLayerKeepsRemainderAfterSpike`

\(I = 1.5\)、\(V_{\mathrm{th}} = 1\)。ハードリセットなら発火後の膜電位は 0。subtractive reset なら余り 0.5 が残る。

1. 発火: 読み出し 1.0、状態 \(v = 0.5\)
2. 入力 0 の次ステップ: \(\beta = 0.8\) なので \(v = 0.4\)、読み出し 0.4（沈黙のアナログ）

`testReadoutKeepsSubthresholdAndClipsSpike` はクリップ関数単体で、0.4 はそのまま、1.5 も 20 も 1.0、−20 は −1.0。

### 6.3 それ以外（配線が同じこと）

`testOneLayerForwardMatchesBetweenMLXAndPureSwift` と `testTwoLayerForwardMatchesBetweenMLXAndPureSwift` は、上の恒等式が学習側と推論側で同じ式であることを見る。compile 対 eager、CTC が下がること、系列長バケットは耐久用で、式の正しさの証拠ではない。

---

## 7. 検証コマンド

```text
swift test --filter 'LIFNeuronTests|MultiLayerSNNTests|SpikingNetworkTests|QuantizationTests|MLXCompileBPTTTests|MLXBPTTTests|MLXCompiledLearningRateTests|MLXCompileAuditVerificationTests'
```

このフィルタはリードアウト訂正とその周辺をまとめて回す。失敗 0。最初に見るべきは `testLastLayerLogitIsThresholdUnitsNotRawVoltageOrSpikeCount` で、期待値は 1.4 である。

大規模学習を再開するなら:

- この訂正入りのバイナリであること
- `--cache-features` を付ける（時間は別問題だが、付けないとまた 2.4 時間/epoch の I/O 待ちになる）
- 3 epoch 目の損失が有限で、1→2 と同程度の桁であること。`1e20` 台が出たらこの修正が効いていない
- チェックポイント間隔は 5 のままなので、短いランでは `checkpointEvery` を詰めた方が安全

`c628b90` 時点の重みは使わない。3 epoch 目で壊れた重みも、その直前の未保存状態も信用しない。

---

## 8. WAV 読みの配線

Go の `io.ReadCloser` をまねた `ReadCloser.swift` は消した。ファイルは `FileHandle`、メモリ上のバイト列は `WavParser.parse(bytes:)` である。

以前はヘッダの解釈が二つあった。`WavParser` はチャンクを歩いて `fmt` / `data` を探す。`WavStreamReader` は先頭 44 バイト固定で、LIST や fact が入った 16-bit ファイルを誤読する。学習は前者、ストリーマは未使用だった。

いまは形式の解釈が一本である。

```
WavFormat.parse(bytes:)     メモリ。data オフセットを返す
WavFormat.read(from:handle) ファイル。ハンドルを data 先頭に置く
WavPCM.decode(...)          どちらの経路も同じモノラル Float
```

- メモリ: `WavParser.parse(bytes:)` → `WavFormat.parse` → `WavPCM.decode`（テスト、`fromWavPairs`）
- ファイル: `SpeechDataset.loadWavFile` → `WavStreamReader` → `WavFormat.read` → チャンクごとに `WavPCM.decode`

`loadFeatures`、学習の評価、transcribe / segment / screen / mictrans のファイル入力はすべて `loadWavFile` に寄せた。全ファイルを `Data` に載せてから `[UInt8]` にコピーすることはしない。PCM `[Float]` は特徴抽出のため残る。

証明は `WavStreamReaderTests.testJunkChunkBeforeFmtMatchesParser`（JUNK のあとに fmt がある 16-bit で、パーサとストリーマの PCM が一致）と `testInt24MatchesParser` / `testFloat32MatchesParser`。44 バイト固定のままなら JUNK ファイルはゴミ PCM になる。

---

## 9. `@inline(__always)` を付ける範囲

LIF / SIMD / Filterbank / FFT には既に付いている。足りなかったのは DSP のホップ単位の小関数と PCM デコードの内側である。

超低遅延エンコーダでは全関数に付けた方が LLVM より速いことが多かった。あれは 48 kHz のサンプルループが小さく閉じていて、関数境界が LICM と SIMD を止めていた。SpikeTrans はそうではない。

- 学習 CTC は MLX / Metal。Swift の inline は効かない
- `SpikingNetwork.forward` は 10 ms に 1 回で、本体が太い。強制すると I キャッシュが悪化する
- 同一モジュールの `private` は WMO が既定でインラインする
- `mictrans` からライブラリをまたぐなら `@inlinable` が必要で、`__always` だけでは本体が相手モジュールに出ない

付けたのは次だけである。小さい、ループの中、LLVM がサイズで躊躇し得る、の三つが揃うところ。

- `StreamingFeatureFrontEnd`: `gainForRMS`, `pushRawFrame`, `updateGain`, `applyPreemphasis`, `ingestMelFromPreemph`, `writeThreeTap`, `pushTapIntoStack`
- `WavPCM.decode` とその 16/24/32-bit・float 本体、`int16LE` / `int32LE` / `isChunk`

`WavFormat.parse` / `read`、`FileHandle` の読み、`forward`、`predict` には付けない。

---

## 10. マージ後に残る仕事（この PR に入れない）

1. 特徴量キャッシュなしの学習時間。38 万件は `--cache-features` が前提。ヘルプにも推奨と書いてある。
2. GPU CTC の番兵混入ガード。CPU 側は `-inf` を `uCount * 5` に置き換えている。GPU 側にも同じ除外があれば、誤ラベル 1 件で epoch 平均が死なない。
3. CPU `BPTTTrainer` を最終層式に揃える。`--device cpu` を本番に使うなら必要。今は GPU 経路が正本。

これらをこのブランチに足すと、訂正の差分がまた膨らむ。リードアウトの式だけを先に入れ、大規模学習を再開できる状態にするのがこのマージの仕事である。
