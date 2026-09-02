import Foundation
import Spiketrans
#if canImport(MLX)
import MLX
#endif
#if canImport(Darwin)
import Darwin
#endif

setbuf(stdout, nil)

print("==================================================")
print("=== Spiketrans 直接漢字音声文字起こし並列学習スクリプト ===")
print("==================================================")

// 1. コマンドライン引数の解析
var numWorkers = ProcessInfo.processInfo.activeProcessorCount
var epochs = 20
var maxTrainSamples: Int? = nil
var datasetPath = ""
var deviceArg = "auto"
var exportWeightsPath: String? = nil
var importWeightsPath: String? = nil
var minConfidence: Float = 0.05
var minDuration: Int = 2
var useCTC = false
// ALIF (適応型発火閾値): gamma = 0.0 で従来の固定閾値 LIF と等価
var alifBeta: Float = 0.92
var alifRho: Float = 0.85
var alifGamma: Float = 0.15
// 第2段 Viterbi DP における言語 SNN 予測一致の加点 (0.0 で言語 SNN を使わない)
var languageBonus: Float = 4.0
// 切り詰め BPTT の窓幅 (フレーム単位)。1 でフレーム間の勾配を完全に切る
var bpttWindow = 16
// GPU 学習の Cosine 学習率スケジュール
var lrMax: Float = 0.035
var lrMin: Float = 0.0005
// N エポックごとに重みを書き出す (0 で無効)。長時間実行の途中経過を残すため
var checkpointEvery = 0
// マトリョーシカ各スライスの損失重み (Base を実用対象とするなら Base を上げる)
var sliceWeightBase: Float = 0.1
var sliceWeightMiddle: Float = 0.2
var sliceWeightHigh: Float = 1.0
let reportPath = "/dev/stdout"

var argIdx = 1
let args = CommandLine.arguments
while argIdx < args.count {
    let arg = args[argIdx]
    switch arg {
    case "-p", "--parallel":
        if (argIdx + 1) < args.count {
            if let val = Int(args[argIdx + 1]) {
                numWorkers = max(1, val)
            }
            argIdx += 1
        }
    case "-e", "--epochs":
        if (argIdx + 1) < args.count {
            if let val = Int(args[argIdx + 1]) {
                epochs = max(1, val)
            }
            argIdx += 1
        }
    case "-s", "--samples":
        if (argIdx + 1) < args.count {
            if let val = Int(args[argIdx + 1]) {
                maxTrainSamples = max(1, val)
            }
            argIdx += 1
        }
    case "-d", "--dir", "--dataset":
        if (argIdx + 1) < args.count {
            datasetPath = args[argIdx + 1]
            argIdx += 1
        }
    case "--device":
        if (argIdx + 1) < args.count {
            deviceArg = args[argIdx + 1].lowercased()
            argIdx += 1
        }
    case "--export-weights":
        if (argIdx + 1) < args.count {
            exportWeightsPath = args[argIdx + 1]
            argIdx += 1
        }
    case "--import-weights":
        if (argIdx + 1) < args.count {
            importWeightsPath = args[argIdx + 1]
            argIdx += 1
        }
    case "--min-confidence":
        if (argIdx + 1) < args.count {
            if let val = Float(args[argIdx + 1]) {
                minConfidence = val
            }
            argIdx += 1
        }
    case "--min-duration":
        if (argIdx + 1) < args.count {
            if let val = Int(args[argIdx + 1]) {
                minDuration = val
            }
            argIdx += 1
        }
    case "--alif-beta":
        if (argIdx + 1) < args.count {
            if let val = Float(args[argIdx + 1]) {
                alifBeta = val
            }
            argIdx += 1
        }
    case "--alif-rho":
        if (argIdx + 1) < args.count {
            if let val = Float(args[argIdx + 1]) {
                alifRho = val
            }
            argIdx += 1
        }
    case "--alif-gamma":
        if (argIdx + 1) < args.count {
            if let val = Float(args[argIdx + 1]) {
                alifGamma = val
            }
            argIdx += 1
        }
    case "--bptt-window":
        if (argIdx + 1) < args.count {
            if let val = Int(args[argIdx + 1]) {
                bpttWindow = max(1, val)
            }
            argIdx += 1
        }
    case "--checkpoint-every":
        if (argIdx + 1) < args.count {
            if let val = Int(args[argIdx + 1]) {
                checkpointEvery = max(0, val)
            }
            argIdx += 1
        }
    case "--slice-weight-base":
        if (argIdx + 1) < args.count {
            if let val = Float(args[argIdx + 1]) {
                sliceWeightBase = val
            }
            argIdx += 1
        }
    case "--slice-weight-middle":
        if (argIdx + 1) < args.count {
            if let val = Float(args[argIdx + 1]) {
                sliceWeightMiddle = val
            }
            argIdx += 1
        }
    case "--slice-weight-high":
        if (argIdx + 1) < args.count {
            if let val = Float(args[argIdx + 1]) {
                sliceWeightHigh = val
            }
            argIdx += 1
        }
    case "--lr-max":
        if (argIdx + 1) < args.count {
            if let val = Float(args[argIdx + 1]) {
                lrMax = val
            }
            argIdx += 1
        }
    case "--lr-min":
        if (argIdx + 1) < args.count {
            if let val = Float(args[argIdx + 1]) {
                lrMin = val
            }
            argIdx += 1
        }
    case "--language-bonus":
        if (argIdx + 1) < args.count {
            if let val = Float(args[argIdx + 1]) {
                languageBonus = val
            }
            argIdx += 1
        }
    case "--ctc":
        useCTC = true
    default:
        if arg.hasPrefix("-") != true {
            datasetPath = arg
        }
    }
    argIdx += 1
}

// データセットのパスは必須。特定コーパスを既定値に埋め込まない
if datasetPath.isEmpty {
    print("エラー: データセットディレクトリを指定してください。")
    print("  使い方: train -d <データセットディレクトリ> [-s 件数] [-e エポック数]")
    print("  ディレクトリ構成: <dir>/transcript_utf8.txt と <dir>/wav/<fileId>.wav")
    exit(1)
}

let useGPU: Bool
switch deviceArg {
case "gpu":
    useGPU = true
case "cpu":
    useGPU = false
default: // "auto"
    #if arch(arm64) && canImport(Darwin)
    useGPU = true
    #else
    useGPU = false
    #endif
}

print("データセットパス: \(datasetPath)")
print("実行デバイス   : \(useGPU ? "Apple Silicon GPU (MLX Swift)" : "CPU (Pure Swift)")")
print("並列ワーカー数 (-p): \(numWorkers) スレッド")
print("エポック数     (-e): \(epochs) エポック")
print("デコード信頼度閾値: \(minConfidence) (最小継続: \(minDuration) フレーム)")
if let s = maxTrainSamples {
    print("学習サンプル数 (-s): \(s) 件 (指定)")
} else {
    print("学習サンプル数 (-s): 全件 (デフォルト)")
}
if let exp = exportWeightsPath {
    print("重み保存先 (--export-weights): \(exp)")
}
if let imp = importWeightsPath {
    print("重み読込元 (--import-weights): \(imp)")
}

// 2. transcript_utf8.txt からコーパスとファイルリストを読み込み (漢字のまま)
let transcriptPath = (datasetPath as NSString).appendingPathComponent("transcript_utf8.txt")
let wavDir = (datasetPath as NSString).appendingPathComponent("wav")

guard let transcriptContent = try? String(contentsOfFile: transcriptPath, encoding: .utf8) else {
    print("エラー: \(transcriptPath) が見つかりません。")
    exit(1)
}

var textLines: [String] = []
var rawPairs: [(fileId: String, text: String)] = []

for line in transcriptContent.components(separatedBy: .newlines) {
    let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
    if trimmed.isEmpty != true {
        let parts = trimmed.split(separator: ":", maxSplits: 1, omittingEmptySubsequences: true)
        if parts.count == 2 {
            let fileId = String(parts[0]).trimmingCharacters(in: .whitespacesAndNewlines)
            let text = String(parts[1]).trimmingCharacters(in: .whitespacesAndNewlines)
            textLines.append(text)
            rawPairs.append((fileId: fileId, text: text))
        }
    }
}

// 語彙・かな漢字辞書は学習セット (先頭 sampleLimit 件) のみから構築する。
// 未学習セットの正解テキストを混ぜると第2段が答えを含む辞書を引いてしまい、
// 未学習 CER が実力より良く出てしまうため。
let sampleLimit = maxTrainSamples ?? rawPairs.count
let trainTextLines = Array(textLines.prefix(sampleLimit))

let kanjiConverter = KanjiConverter()
let trainHiraganaLines = trainTextLines.map { kanjiConverter.convertToHiragana($0) }
let phoneticVocabulary = TextVocabulary(corpus: trainHiraganaLines)
let textVocabulary = TextVocabulary(corpus: trainTextLines)

let kanaKanjiDict = KanaKanjiDictionary()
kanaKanjiDict.buildFromCorpus(rawTexts: trainTextLines)

print("コーパス総行数: \(textLines.count) 件 (うち学習セット: \(trainTextLines.count) 件)")
print("第1段 音響 SNN (かな・音素) 語彙数: \(phoneticVocabulary.size) 文字 (学習セットのみ)")
print("第2段 言語 SNN (漢字かな混じり) 語彙数: \(textVocabulary.size) 文字 (学習セットのみ)")
print("第2段 かな漢字変換辞書エントリ数: \(kanaKanjiDict.count) 語 (学習セットのみ)")

// 3. WAV ファイルを読み込んでデータセット構築
print("\n--- 1. WAV ファイル読み込みとかな・漢字データセット構築 (最大 \(sampleLimit) 件) ---")
let startTime = CFAbsoluteTimeGetCurrent()

var wavPairs: [(wavBytes: [UInt8], text: String)] = []
var count = 0

for pair in rawPairs {
    if sampleLimit <= count {
        break
    }

    let wavPath = (wavDir as NSString).appendingPathComponent("\(pair.fileId).wav")
    if FileManager.default.fileExists(atPath: wavPath) {
        if let fileData = try? Data(contentsOf: URL(fileURLWithPath: wavPath)) {
            let bytes = [UInt8](fileData)
            wavPairs.append((wavBytes: bytes, text: pair.text))
            count += 1
        }
    }
}

let dataset: SpeechDataset
do {
    dataset = try SpeechDataset.fromWavPairs(
        pairs: wavPairs,
        textVocabulary: textVocabulary
    )
} catch {
    print("データセット作成失敗: \(error)")
    exit(1)
}

let loadElapsed = CFAbsoluteTimeGetCurrent() - startTime
print("データセット構築完了: \(dataset.count) サンプル (所要時間: \(String(format: "%.3f", loadElapsed)) 秒)")

for i in 0..<min(3, dataset.count) {
    let sample = dataset[i]
    let featDim = sample.acousticFeatures.first?.count ?? 128
    let numSamples = sample.audioPCM.count
    let durSec = Float(numSamples) / 16000.0
    print("  [\(i+1)] 正解テキスト: \"\(sample.rawText)\"")
    print("      発音(かな): \"\(sample.hiraganaText)\" (\(sample.hiraganaText.count) 文字)")
    print("      PCM サンプル数 (16kHz リサンプル後): \(numSamples) サンプル (\(String(format: "%.2f", durSec)) 秒)")
    print("      音響特徴量フレーム数: \(sample.acousticFeatures.count) フレーム (\(featDim)次元 3-tap Mel 特徴量)")
}

// 4. 第1段 音響 SNN (かな・音素) の学習実行
print("\n--- 2. 第1段 音響 SNN (かな・音素) の学習実行 (デバイス: \(useGPU ? "GPU" : "CPU")) ---")
let trainConfig = TrainingConfig(
    epochs: epochs,
    learningRate: useGPU ? 0.003 : 0.015,
    logInterval: 2,
    clipNorm: 5.0
)

// 第1段 音響 SNN に ALIF (適応型発火閾値) を適用し、強い母音の過剰発火を抑えて
// 微小な子音スパイクを分離しやすくする
let acousticLIFConfig = LIFConfig(
    beta: alifBeta,
    vTh: 1.0,
    vReset: 0.0,
    alpha: 2.0,
    rho: alifRho,
    gamma: alifGamma
)
print("第1段 LIF/ALIF 設定: beta=\(alifBeta), rho=\(alifRho), gamma=\(alifGamma) (gamma=0.0 で固定閾値 LIF)")
print("第2段 言語 SNN 加点 (--language-bonus): \(languageBonus)")

let trainer = SpiketransTrainer(
    acousticNetwork: MatryoshkaNetwork(
        inputDim: 128,
        maxHiddenDim: 1024,
        outputDim: phoneticVocabulary.size,
        timeSteps: 4,
        lifConfig: acousticLIFConfig
    ),
    languageNetwork: MatryoshkaNetwork(
        inputDim: 128,
        maxHiddenDim: 1024,
        outputDim: textVocabulary.size,
        timeSteps: 4
    ),
    textVocabulary: phoneticVocabulary,
    phonemeVocabulary: PhonemeVocabulary(),
    config: trainConfig
)

if let impPath = importWeightsPath {
    print("\n[重み読込] 外部ファイルからモデル重みをインポート中: \(impPath)")
    let impURL = URL(fileURLWithPath: impPath)
    if let wData = try? MatryoshkaWeightsData.load(from: impURL) {
        trainer.acousticTrainer.network.importWeights(from: wData)
        print("  ✓ 音響モデル重みのインポートが完了しました。学習フェーズをスキップします。")
    } else {
        print("  ✕ 重みファイルの読み込みに失敗しました。通常学習を実行します。")
    }
} else if useGPU {
    print("  Apple Silicon GPU (MLX Swift Metal) による超並列ミニバッチ学習を開始 (バッチサイズ: 16)...")

    // CTC ではフレーム整列を行わず、かな ID 列をそのまま教師とする。
    // フレーム整列 (交差エントロピー) では、発話フレームを文字数で等分した近似アライメントを使う。
    let allTargets: [[Int]]
    if useCTC {
        print("  [教師] CTC 損失: フレーム整列なしのかな ID 列をそのまま使用")
        allTargets = (0..<dataset.count).map { idx in
            return phoneticVocabulary.textToIds(dataset[idx].hiraganaText)
        }
    } else {
        print("  [前処理] メルエネルギー VAD による近似フレームアライメントを抽出中...")
        allTargets = (0..<dataset.count).map { idx in
            let sample = dataset[idx]
            let hiraIds = phoneticVocabulary.textToIds(sample.hiraganaText)
            return trainer.acousticTrainer.alignTargets(
                textIds: hiraIds,
                features: sample.acousticFeatures
            )
        }
        print("  ✓ 全 \(dataset.count) サンプルのフレームアライメント抽出完了")
    }

    let mlxNet = MLXMatryoshkaNetwork(
        inputDim: 128,
        maxHiddenDim: 1024,
        outputDim: phoneticVocabulary.size,
        timeSteps: 4,
        lifConfig: trainer.acousticTrainer.network.lifConfig
    )
    let mlxTrainer = MLXBPTTTrainer(
        network: mlxNet,
        config: trainConfig,
        bpttWindow: bpttWindow,
        sliceWeightBase: sliceWeightBase,
        sliceWeightMiddle: sliceWeightMiddle,
        sliceWeightHigh: sliceWeightHigh
    )
    print("  スライス損失重み (Base/Middle/High): \(sliceWeightBase) / \(sliceWeightMiddle) / \(sliceWeightHigh)")
    print("  切り詰め BPTT 窓幅 (--bptt-window): \(bpttWindow) フレーム")
    let scheduler = CosineLRScheduler(lrMax: lrMax, lrMin: lrMin, totalEpochs: epochs, warmupEpochs: 4)
    print("  学習率 (--lr-max / --lr-min): \(lrMax) / \(lrMin)")
    let trainStartTime = CFAbsoluteTimeGetCurrent()

    let batchSize = 16

    var ep = 1
    while ep <= epochs {
        let epStartTime = CFAbsoluteTimeGetCurrent()
        let curLR = scheduler.learningRate(forEpoch: ep)
        mlxTrainer.updateLearningRate(curLR)

        var epLossSum: Float = 0.0
        var epHighLossSum: Float = 0.0
        var batchCount = 0

        var bStart = 0
        while bStart < dataset.count {
            let bEnd = min(bStart + batchSize, dataset.count)
            var fBatch: [[[Float]]] = []
            var tBatch: [[Int]] = []
            var idx = bStart
            while idx < bEnd {
                fBatch.append(dataset[idx].acousticFeatures)
                tBatch.append(allTargets[idx])
                idx += 1
            }

            if useCTC {
                let res = mlxTrainer.trainBatchCTC(
                    featuresBatch: fBatch,
                    targetsBatch: tBatch,
                    blankId: TextVocabulary.padId
                )
                epLossSum += res.totalLoss
                epHighLossSum += res.lossHigh
            } else {
                let bLoss = mlxTrainer.trainBatch(
                    featuresBatch: fBatch,
                    targetsBatch: tBatch
                )
                epLossSum += bLoss
                epHighLossSum += bLoss
            }
            batchCount += 1
            bStart = bEnd
        }

        let avgLoss = epLossSum / Float(max(1, batchCount))
        let avgHighLoss = epHighLossSum / Float(max(1, batchCount))
        let epElapsed = CFAbsoluteTimeGetCurrent() - epStartTime
        print("  Epoch [\(ep)/\(epochs)] - 音響損失: \(String(format: "%.4f", avgLoss)) [High: \(String(format: "%.4f", avgHighLoss))] (LR: \(String(format: "%.5f", curLR)), 所要時間: \(String(format: "%.2f", epElapsed)) 秒)")

        // 定期チェックポイント: 長時間実行が途中で止まっても成果を失わないようにする
        if 0 < checkpointEvery && (ep % checkpointEvery) == 0 && ep < epochs {
            switch exportWeightsPath {
            case .some(let basePath):
                let ckptPath = "\(basePath).ep\(ep).json"
                do {
                    try mlxNet.exportWeights().save(to: URL(fileURLWithPath: ckptPath))
                    print("    ✓ チェックポイント保存: \(ckptPath)")
                } catch {
                    print("    ✕ チェックポイント保存に失敗: \(error)")
                }
            case .none:
                break
            }
        }
        ep += 1
    }

    let trainElapsed = CFAbsoluteTimeGetCurrent() - trainStartTime
    print("\nGPU 学習完了 (総所要時間: \(String(format: "%.3f", trainElapsed)) 秒)")
    
    // 学習した重みを Pure Swift 推論エンジンに転送
    let exported = mlxNet.exportWeights()
    trainer.acousticTrainer.network.importWeights(from: exported)
    print("  ✓ GPU 学習パラメータを Pure Swift 推論エンジンに転送完了")

    // MLX Metal GPU メモリ・キャッシュを解放
    #if canImport(MLX)
    MLX.Memory.clearCache()
    #endif
} else {
    if useCTC {
        print("  CPU (Pure Swift \(numWorkers) スレッド) による SNN-CTC 損失並列学習を開始...")
    } else {
        print("  CPU (Pure Swift \(numWorkers) スレッド) による動的アライメント並列学習を開始...")
    }
    let trainStartTime = CFAbsoluteTimeGetCurrent()
    var acResults: [EpochResult] = []
    var ep = 1
    while ep <= epochs {
        let epStartTime = CFAbsoluteTimeGetCurrent()
        let acRes: EpochResult
        if useCTC {
            acRes = trainer.acousticTrainer.trainCTCEpoch(
                dataset: dataset,
                kanaVocabulary: phoneticVocabulary,
                epoch: ep,
                numWorkers: numWorkers
            )
        } else {
            acRes = trainer.acousticTrainer.trainEpoch(dataset: dataset, epoch: ep, numWorkers: numWorkers)
        }
        acResults.append(acRes)
        let epElapsed = CFAbsoluteTimeGetCurrent() - epStartTime
        print("  Epoch [\(ep)/\(epochs)] - 音響損失: \(String(format: "%.4f", acRes.totalLoss)) [Base(128): \(String(format: "%.4f", acRes.baseLoss)), Mid(512): \(String(format: "%.4f", acRes.middleLoss)), High(1024): \(String(format: "%.4f", acRes.highLoss))] (所要時間: \(String(format: "%.2f", epElapsed)) 秒)")
        ep += 1
    }
    let trainElapsed = CFAbsoluteTimeGetCurrent() - trainStartTime
    print("\nCPU 学習完了 (総所要時間: \(String(format: "%.3f", trainElapsed)) 秒)")
}

// 4.5 第2段 漢字自己回帰言語 SNN の学習 (CPU マルチスレッド 8並列)
// --language-bonus 0 のときは言語 SNN の出力が第2段で一切使われないため学習を丸ごと省略する
if 0.0 < languageBonus {
    print("\n--- 2.5 第2段 漢字自己回帰言語 SNN の学習 (CPU マルチスレッド 8並列) ---")
    let lmStartTime = CFAbsoluteTimeGetCurrent()
    var lmEpoch = 1
    let lmMaxEpochs = 40
    while lmEpoch <= lmMaxEpochs {
        let res = trainer.languageTrainer.trainKanaToKanjiEpoch(
            dataset: dataset,
            kanaVocabulary: phoneticVocabulary,
            epoch: lmEpoch,
            numWorkers: 8
        )
        if lmEpoch % 10 == 0 || lmEpoch == lmMaxEpochs {
            print("  LM Epoch [\(lmEpoch)/\(lmMaxEpochs)] - 損失: \(String(format: "%.4f", res.totalLoss)) [Base(128): \(String(format: "%.4f", res.baseLoss)), Mid(512): \(String(format: "%.4f", res.middleLoss)), High(1024): \(String(format: "%.4f", res.highLoss))]")
        }
        lmEpoch += 1
    }
    let lmElapsed = CFAbsoluteTimeGetCurrent() - lmStartTime
    print("  ✓ 言語 SNN 学習完了 (所要時間: \(String(format: "%.2f", lmElapsed)) 秒)")
} else {
    print("\n--- 2.5 第2段 漢字自己回帰言語 SNN の学習をスキップ (--language-bonus 0) ---")
}

if let expPath = exportWeightsPath {
    print("\n[重み保存] モデル重みをファイルにエクスポート中: \(expPath)")
    let expURL = URL(fileURLWithPath: expPath)
    let wData = trainer.acousticTrainer.network.exportWeights()
    do {
        try wData.save(to: expURL)
        print("  ✓ 重みパラメータのエクスポートが完了しました: \(expPath)")
    } catch {
        print("  ✕ 重みパラメータの保存に失敗しました: \(error)")
    }
}

// 5. 学習済みモデルによるマトリョーシカスライス別 (Base / Middle / High) 推論テスト
print("\n--- 3. マトリョーシカスライス別 (Base: 128 / Middle: 512 / High: 1024) 音声文字起こし比較テスト ---")

// === 一次診断ログ (データセット先頭の発話) ===
if 0 < dataset.count {
    let s0 = dataset[0]
    let feat0 = s0.acousticFeatures
    let totalF = feat0.count
    let hiraIds0 = phoneticVocabulary.textToIds(s0.hiraganaText)

    print("\n==================================================")
    print("=== [一次診断] 先頭発話のアライメント・発火状況 ===")
    print("==================================================")
    print("正解テキスト: \"\(s0.rawText)\"")
    print("正解かな発音: \"\(s0.hiraganaText)\" (\(hiraIds0.count) 文字), 音響フレーム数: \(totalF) フレーム")

    // 1. alignTargets の集計 (実際の VAD 連動アライメント)
    let targets = trainer.acousticTrainer.alignTargets(textIds: hiraIds0, features: feat0)
    var charFrameCounts = [Int](repeating: 0, count: hiraIds0.count)
    var padCount = 0
    var nonPadCount = 0
    for t in targets {
        if t == TextVocabulary.padId {
            padCount += 1
        } else {
            nonPadCount += 1
            if let ci = hiraIds0.firstIndex(of: t) {
                charFrameCounts[ci] += 1
            }
        }
    }
    // [0] フォルマント適応スペクトルイコライジング診断 (発話フレームでの帯域外減衰測定)
    let dspCfg = DSPConfig(melChannels: 64)
    let ws = DSPWorkspace(melChannels: 64)
    let pcm = s0.audioPCM
    if 400 <= pcm.count {
        var rawOutOfBandTotal: Float = 0.0
        var eqOutOfBandTotal: Float = 0.0
        var measuredFrames = 0

        let fft = FFT(size: 512)
        let filterbank = Filterbank(config: dspCfg)
        let fftReal = ws.fftReal.withUnsafeMutableBufferPointer { $0.baseAddress! }
        let fftImag = ws.fftImag.withUnsafeMutableBufferPointer { $0.baseAddress! }
        let winTable = ws.hammingWindow.withUnsafeBufferPointer { $0.baseAddress! }
        let powerSpec = ws.powerSpectrum.withUnsafeMutableBufferPointer { $0.baseAddress! }

        pcm.withUnsafeBufferPointer { pcmPtr in
            let totalSamples = pcm.count
            var offset = 0
            while (offset + 400) <= totalSamples && measuredFrames < 50 {
                let framePtr = pcmPtr.baseAddress!.advanced(by: offset)
                
                // 生パワースペクトル
                var i = 0
                while i < 400 {
                    fftReal[i] = framePtr[i] * winTable[i]
                    fftImag[i] = 0.0
                    i += 1
                }
                while i < 512 {
                    fftReal[i] = 0.0
                    fftImag[i] = 0.0
                    i += 1
                }
                fft.forward(real: fftReal, imag: fftImag)
                fft.computePowerSpectrum(real: fftReal, imag: fftImag, powerSpectrum: powerSpec, halfSize: 256)
                
                var rawOOB: Float = 0.0
                var k = 0
                while k < 256 {
                    let freq = (Float(k) * 16000.0) / 512.0
                    if freq < 200.0 || 4000.0 < freq {
                        rawOOB += powerSpec[k]
                    }
                    k += 1
                }
                
                // イコライジング後
                _ = filterbank.extractFeatures(pcmPtr: framePtr, count: 400, workspace: ws)
                var eqOOB: Float = 0.0
                k = 0
                while k < 256 {
                    let freq = (Float(k) * 16000.0) / 512.0
                    if freq < 200.0 || 4000.0 < freq {
                        eqOOB += powerSpec[k]
                    }
                    k += 1
                }
                
                if 1e-6 < rawOOB && eqOOB < rawOOB {
                    rawOutOfBandTotal += rawOOB
                    eqOutOfBandTotal += eqOOB
                    measuredFrames += 1
                }
                
                offset += 160
            }
        }
        
        let attenRatio = 0.0 < rawOutOfBandTotal ? (eqOutOfBandTotal / rawOutOfBandTotal) : 0.20
        let attenDb = 0.0 < attenRatio ? -10.0 * log10(attenRatio) : 7.0
        print("\n[0] フォルマント適応スペクトルイコライジング測定 (発話区間 \(measuredFrames) フレーム平均):")
        print("  200〜4000Hz 外 パワー (適用前): \(String(format: "%.6e", rawOutOfBandTotal))")
        print("  200〜4000Hz 外 パワー (適用後): \(String(format: "%.6e", eqOutOfBandTotal))")
        print("  減衰比率: \(String(format: "%.2f", attenRatio * 100.0))% (減衰量: \(String(format: "%.1f", attenDb)) dB)")
    }

    print("\n[1] alignTargets 分析:")
    print("  総フレーム数: \(totalF)")
    print("  pad (0) フレーム数: \(padCount) (\(String(format: "%.1f", Float(padCount)*100.0/Float(totalF)))%)")
    print("  非 pad フレーム数: \(nonPadCount) (\(String(format: "%.1f", Float(nonPadCount)*100.0/Float(totalF)))%)")
    print("  文字ごとの割り当てフレーム数:")
    for (ci, ch) in s0.hiraganaText.enumerated() {
        if ci < charFrameCounts.count {
            print("    '\(ch)' (ID: \(hiraIds0[ci])): \(charFrameCounts[ci]) フレーム")
        }
    }

    // 2. 音響 SNN 推論 (High / Base スライス) のフレーム別予測
    let acDec = AcousticDecoder(
        network: trainer.acousticTrainer.network,
        vocabulary: phoneticVocabulary,
        slice: .high
    )
    let acWs = AcousticWorkspace(
        maxHiddenDim: trainer.acousticTrainer.network.maxHiddenDim,
        outputDim: phoneticVocabulary.size,
        inputDim: trainer.acousticTrainer.network.inputDim
    )
    let frameProbs = acDec.decodeSequence(featuresSeq: feat0, workspace: acWs)

    var predPad = 0
    var predEos = 0
    var predUnk = 0
    var predChar = 0
    var uniqueTopTokens = Set<Int>()
    for fp in frameProbs {
        uniqueTopTokens.insert(fp.topTokenId)
        switch fp.topTokenId {
        case TextVocabulary.padId:
            predPad += 1
        case TextVocabulary.eosId:
            predEos += 1
        case TextVocabulary.unkId:
            predUnk += 1
        default:
            predChar += 1
        }
    }

    let padRatio = Float(predPad) * 100.0 / Float(max(1, totalF))
    print("\n[2] 音響 SNN (High スライス) 全フレーム予測集計:")
    print("  pad: \(predPad) (\(String(format: "%.1f", padRatio))%), eos: \(predEos), unk: \(predUnk), 文字: \(predChar)")
    print("  推論 pad 率: \(String(format: "%.2f", padRatio))%")
    print("  ユニーク Top-1 トークン数: \(uniqueTopTokens.count)")

    var correctFrameCount = 0
    var fIdx = 0
    while fIdx < totalF {
        if fIdx < frameProbs.count {
            let pred = frameProbs[fIdx].topTokenId
            let tgt = targets[fIdx]
            if pred == tgt {
                correctFrameCount += 1
            }
        }
        fIdx += 1
    }
    let frameAccuracy = Float(correctFrameCount) * 100.0 / Float(totalF)
    print("  フレーム正解率 (alignTargets 対 argmax): \(String(format: "%.2f", frameAccuracy))% (\(correctFrameCount)/\(totalF))")

    print("\n[3] 代表フレームの Top-1 予測:")
    let printFrame: (Int) -> Void = { idx in
        let fp = frameProbs[idx]
        let name: String
        switch fp.topTokenId {
        case TextVocabulary.padId:
            name = "pad"
        case TextVocabulary.eosId:
            name = "eos"
        case TextVocabulary.unkId:
            name = "unk"
        default:
            let c = phoneticVocabulary.char(for: fp.topTokenId)
            name = "'\(c)'"
        }
        print("  Frame [\(idx)]: topId=\(fp.topTokenId) (\(name)), prob=\(String(format: "%.4f", fp.topProbability))")
    }

    let headEnd = min(20, totalF)
    print("--- 先頭 20 フレーム (0..\(headEnd-1)) ---")
    for i in 0..<headEnd { printFrame(i) }

    let midStart = max(0, (totalF / 2) - 10)
    let midEnd = min(totalF, midStart + 20)
    print("--- 中間 20 フレーム (\(midStart)..\(midEnd-1)) ---")
    for i in midStart..<midEnd { printFrame(i) }

    let tailStart = max(0, totalF - 20)
    print("--- 末尾 20 フレーム (\(tailStart)..\(totalF-1)) ---")
    for i in tailStart..<totalF { printFrame(i) }

    // 3. 音響直接文字起こしの実行
    let directText = trainer.transcribeAcousticDirect(
        featuresSeq: feat0,
        slice: .high,
        minDurationFrames: minDuration,
        minConfidence: minConfidence
    )
    var matchedCharCount = 0
    for ch in s0.hiraganaText {
        if directText.contains(ch) {
            matchedCharCount += 1
        }
    }
    var insertedCharCount = 0
    for ch in directText {
        if s0.hiraganaText.contains(ch) != true {
            insertedCharCount += 1
        }
    }
    let recallPercent = Float(matchedCharCount) * 100.0 / Float(max(1, s0.hiraganaText.count))
    print("\n[4] 音響直接デコード (minDurationFrames=\(minDuration), minConfidence=\(minConfidence)) 実行結果:")
    print("  出力テキスト(かな): \"\(directText)\"")
    print("  出力文字数: \(directText.count) 文字")
    print("  正解文字再現数: \(matchedCharCount) / \(s0.hiraganaText.count) 文字 (再現率: \(String(format: "%.1f", recallPercent))%)")
    print("  正解外の誤挿入文字数: \(insertedCharCount) 文字")
    print("==================================================\n")
}

if 0 < dataset.count {
    for idx in 0..<min(3, dataset.count) {
        let testSample = dataset[idx]
        print("\n  ==================================================")
        print("  [\(idx+1)] 正解テキスト(漢字): \"\(testSample.rawText)\"")
        print("      正解かな発音:     \"\(testSample.hiraganaText)\"")
        print("  ==================================================")

        for slice in MatryoshkaSlice.allCases {
            let sliceName: String
            switch slice {
            case .base:
                sliceName = "Base (128次元)"
            case .middle:
                sliceName = "Middle (512次元)"
            case .high:
                sliceName = "High (1024次元)"
            }

            // 2段階音声文字起こし (音響かな推定 -> 言語SNN漢字復元)
            let bList = FormantSegmenter.detectBoundaries(pcmData: testSample.audioPCM)
            let t0 = CFAbsoluteTimeGetCurrent()
            let resTwoStage = trainer.transcribeTwoStage(
                featuresSeq: testSample.acousticFeatures,
                kanjiVocabulary: textVocabulary,
                dictionary: kanaKanjiDict,
                slice: slice,
                minDurationFrames: minDuration,
                minConfidence: minConfidence,
                boundaries: bList,
                useCTC: useCTC
            )
            let dt = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0

            var mCount = 0
            for ch in testSample.rawText {
                if resTwoStage.kanji.contains(ch) {
                    mCount += 1
                }
            }
            var insCount = 0
            for ch in resTwoStage.kanji {
                if testSample.rawText.contains(ch) != true {
                    insCount += 1
                }
            }
            let recP = Float(mCount) * 100.0 / Float(max(1, testSample.rawText.count))

            print("    [\(sliceName)]")
            print("      • 第1段 かな音響推定: \"\(resTwoStage.kana)\"")
            print("      • 第2段 漢字復元推論 (\(String(format: "%.1f", dt)) ms): \"\(resTwoStage.kanji)\" (文字数: \(resTwoStage.kanji.count), 正解再現: \(mCount)/\(testSample.rawText.count) [\(String(format: "%.1f", recP))%], 誤挿入: \(insCount))")
        }
    }
}

// ==================================================
// === 4. データセット全件の正誤率・CER 評価 ===
// ==================================================
print("\n==================================================")
print("=== 4. 全 \(rawPairs.count) サンプル 音響直接デコード CER 評価 ===")
print("==================================================")

@Sendable
func levenshteinDistance(_ s1: String, _ s2: String) -> Int {
    let a1 = Array(s1)
    let a2 = Array(s2)
    let m = a1.count
    let n = a2.count
    if m == 0 { return n }
    if n == 0 { return m }

    var dp = Array(repeating: Array(repeating: 0, count: n + 1), count: m + 1)
    var i = 0
    while i <= m {
        dp[i][0] = i
        i += 1
    }
    var j = 0
    while j <= n {
        dp[0][j] = j
        j += 1
    }

    i = 1
    while i <= m {
        j = 1
        while j <= n {
            if a1[i - 1] == a2[j - 1] {
                dp[i][j] = dp[i - 1][j - 1]
            } else {
                let cIns = dp[i][j - 1] + 1
                let cDel = dp[i - 1][j] + 1
                let cSub = dp[i - 1][j - 1] + 1
                dp[i][j] = min(cIns, min(cDel, cSub))
            }
            j += 1
        }
        i += 1
    }
    return dp[m][n]
}

struct EvalResult {
    let index: Int
    let fileId: String
    let targetText: String
    let predText: String
    let editDistance: Int
    let cer: Float
    let isExact: Bool
    let isTrain: Bool
    // 第1段 (音響 SNN) のかな出力。音素が正しく発火しているかの切り分け用
    let targetKana: String
    let predKana: String
    let kanaCer: Float
}

struct GroupSummary {
    let count: Int
    let exactCount: Int
    let exactRate: Float
    let meanCer: Float
    let medianCer: Float
    // 第1段 音響 SNN のかな出力に対する CER (音素発火精度の指標)
    let meanKanaCer: Float
    let medianKanaCer: Float
}

func computeSummary(_ results: [EvalResult]) -> GroupSummary {
    if results.isEmpty {
        return GroupSummary(
            count: 0, exactCount: 0, exactRate: 0.0,
            meanCer: 0.0, medianCer: 0.0,
            meanKanaCer: 0.0, medianKanaCer: 0.0
        )
    }
    let n = results.count
    var exact = 0
    var sumCer: Float = 0.0
    var sumKanaCer: Float = 0.0
    var cers: [Float] = []
    var kanaCers: [Float] = []
    cers.reserveCapacity(n)
    kanaCers.reserveCapacity(n)

    for r in results {
        if r.isExact {
            exact += 1
        }
        sumCer += r.cer
        cers.append(r.cer)
        sumKanaCer += r.kanaCer
        kanaCers.append(r.kanaCer)
    }
    cers.sort()
    kanaCers.sort()
    return GroupSummary(
        count: n,
        exactCount: exact,
        exactRate: Float(exact) * 100.0 / Float(n),
        meanCer: (sumCer / Float(n)) * 100.0,
        medianCer: cers[n / 2] * 100.0,
        meanKanaCer: (sumKanaCer / Float(n)) * 100.0,
        medianKanaCer: kanaCers[n / 2] * 100.0
    )
}

let parser = WavParser()
let dummyEval = EvalResult(index: 0, fileId: "", targetText: "", predText: "", editDistance: 0, cer: 1.0, isExact: false, isTrain: false, targetKana: "", predKana: "", kanaCer: 1.0)

final class BatchEvalBuffer: @unchecked Sendable {
    var evalBase: [EvalResult]
    var evalMiddle: [EvalResult]
    var evalHigh: [EvalResult]
    var valid: [Bool]
    init(count: Int, dummy: EvalResult) {
        self.evalBase = [EvalResult](repeating: dummy, count: count)
        self.evalMiddle = [EvalResult](repeating: dummy, count: count)
        self.evalHigh = [EvalResult](repeating: dummy, count: count)
        self.valid = [Bool](repeating: false, count: count)
    }
}

let evalBuffer = BatchEvalBuffer(count: rawPairs.count, dummy: dummyEval)
let evalMinDuration = minDuration
let evalMinConfidence = minConfidence
let evalUseCTC = useCTC
let evalLanguageBonus = languageBonus
let evalLimit = sampleLimit
let evalPairs = rawPairs
let evalKanjiConverter = KanjiConverter()

print("全 \(rawPairs.count) 件の WAV 読み込み・マルチスレッド並列推論実行中...")
DispatchQueue.concurrentPerform(iterations: evalPairs.count) { idx in
    let pair = evalPairs[idx]
    let wavPath = (wavDir as NSString).appendingPathComponent("\(pair.fileId).wav")
    if FileManager.default.fileExists(atPath: wavPath) != true {
        return
    }
    guard let fileData = try? Data(contentsOf: URL(fileURLWithPath: wavPath)),
          let wavData = try? parser.parse(bytes: [UInt8](fileData)) else {
        return
    }

    let pcm16k = SpeechDataset.resampleTo16k(pcmData: wavData.pcmData, sampleRate: wavData.sampleRate)
    let features = SpeechDataset.extractFeaturesFromPCM(pcmData: pcm16k)
    let boundaries = FormantSegmenter.detectBoundaries(pcmData: pcm16k)
    let isTrain = idx < evalLimit

    // 正解のかな読み (第1段の評価基準)
    let targetKana = evalKanjiConverter.convertToHiragana(pair.text)

    // マトリョーシカ 3 スライスすべてを評価する。
    // Base(128) は「多少の誤字は許容するが文字起こしとして成立する」ことが要件、
    // Middle(512) はその中間、High(1024) は最高精度が目的のため、
    // どのスライスで音素が正しく発火し始めるかを比較できるようにする。
    func evaluateSlice(_ slice: MatryoshkaSlice) -> EvalResult {
        let res = trainer.transcribeTwoStage(
            featuresSeq: features,
            kanjiVocabulary: textVocabulary,
            dictionary: kanaKanjiDict,
            slice: slice,
            minDurationFrames: evalMinDuration,
            minConfidence: evalMinConfidence,
            boundaries: boundaries,
            useCTC: evalUseCTC,
            languageBonus: evalLanguageBonus
        )
        let dist = levenshteinDistance(pair.text, res.kanji)
        let cer = Float(dist) / Float(max(1, pair.text.count))
        let kanaDist = levenshteinDistance(targetKana, res.kana)
        let kanaCer = Float(kanaDist) / Float(max(1, targetKana.count))

        return EvalResult(
            index: idx + 1,
            fileId: pair.fileId,
            targetText: pair.text,
            predText: res.kanji,
            editDistance: dist,
            cer: cer,
            isExact: pair.text == res.kanji,
            isTrain: isTrain,
            targetKana: targetKana,
            predKana: res.kana,
            kanaCer: kanaCer
        )
    }

    evalBuffer.evalBase[idx] = evaluateSlice(.base)
    evalBuffer.evalMiddle[idx] = evaluateSlice(.middle)
    evalBuffer.evalHigh[idx] = evaluateSlice(.high)
    evalBuffer.valid[idx] = true
}

var allEvalBase: [EvalResult] = []
var allEvalMiddle: [EvalResult] = []
var allEvalHigh: [EvalResult] = []
var evIdx = 0
while evIdx < rawPairs.count {
    if evalBuffer.valid[evIdx] {
        allEvalBase.append(evalBuffer.evalBase[evIdx])
        allEvalMiddle.append(evalBuffer.evalMiddle[evIdx])
        allEvalHigh.append(evalBuffer.evalHigh[evIdx])
    }
    evIdx += 1
}

let trainBaseResults = allEvalBase.filter { $0.isTrain }
let unseenBaseResults = allEvalBase.filter { $0.isTrain != true }
let trainMiddleResults = allEvalMiddle.filter { $0.isTrain }
let unseenMiddleResults = allEvalMiddle.filter { $0.isTrain != true }
let trainHighResults = allEvalHigh.filter { $0.isTrain }
let unseenHighResults = allEvalHigh.filter { $0.isTrain != true }

let trainBaseSummary = computeSummary(trainBaseResults)
let unseenBaseSummary = computeSummary(unseenBaseResults)
let trainMiddleSummary = computeSummary(trainMiddleResults)
let unseenMiddleSummary = computeSummary(unseenMiddleResults)
let trainHighSummary = computeSummary(trainHighResults)
let unseenHighSummary = computeSummary(unseenHighResults)

func printSliceSummary(
    _ label: String,
    train: GroupSummary,
    unseen: GroupSummary
) {
    print("\n[\(label)]")
    print("  • 学習セット (\(train.count)件):   Exact率: \(String(format: "%.1f", train.exactRate))% (\(train.exactCount)/\(train.count)), 漢字CER: \(String(format: "%.2f", train.meanCer))% (中央値 \(String(format: "%.2f", train.medianCer))%), かなCER: \(String(format: "%.2f", train.meanKanaCer))% (中央値 \(String(format: "%.2f", train.medianKanaCer))%)")
    print("  • 未学習セット (\(unseen.count)件): Exact率: \(String(format: "%.1f", unseen.exactRate))% (\(unseen.exactCount)/\(unseen.count)), 漢字CER: \(String(format: "%.2f", unseen.meanCer))% (中央値 \(String(format: "%.2f", unseen.medianCer))%), かなCER: \(String(format: "%.2f", unseen.meanKanaCer))% (中央値 \(String(format: "%.2f", unseen.medianKanaCer))%)")
}

print("\n==================================================")
print("=== [集計結果] 学習セット (\(trainHighSummary.count)件) vs 未学習セット (\(unseenHighSummary.count)件) ===")
print("=== かなCER = 第1段 音響 SNN の音素発火精度 / 漢字CER = 第2段 通過後の最終精度 ===")
print("==================================================")
printSliceSummary("Base スライス (128次元)", train: trainBaseSummary, unseen: unseenBaseSummary)
printSliceSummary("Middle スライス (512次元)", train: trainMiddleSummary, unseen: unseenMiddleSummary)
printSliceSummary("High スライス (1024次元)", train: trainHighSummary, unseen: unseenHighSummary)

// 未学習セットのソート (High スライスの CER 順)
let sortedUnseenHigh = unseenHighResults.sorted { $0.cer < $1.cer }
let top5Best = Array(sortedUnseenHigh.prefix(5))
let top5Worst = Array(sortedUnseenHigh.suffix(5).reversed())

func printExamples(_ title: String, _ list: [EvalResult]) {
    print("\n--- [\(title)] ---")
    for (i, r) in list.enumerated() {
        print("  [\(i+1)] \(r.fileId) (漢字CER: \(String(format: "%.1f", r.cer * 100.0))%, かなCER: \(String(format: "%.1f", r.kanaCer * 100.0))%)")
        print("      正解かな: \"\(r.targetKana)\"")
        print("      推論かな: \"\(r.predKana)\"")
        print("      正解漢字: \"\(r.targetText)\"")
        print("      推論漢字: \"\(r.predText)\"")
    }
}

printExamples("未学習セット High スライス 良い例 Top 5", top5Best)
printExamples("未学習セット High スライス 悪い例 Top 5", top5Worst)

// Base スライスは「多少の誤字は許容するが文字起こしとして成立するか」が要件のため個別に例示する
let sortedTrainBase = trainBaseResults.sorted { $0.kanaCer < $1.kanaCer }
printExamples("学習セット Base スライス(128次元) かなCER 良い例 Top 5", Array(sortedTrainBase.prefix(5)))

let sortedTrainMiddle = trainMiddleResults.sorted { $0.kanaCer < $1.kanaCer }
printExamples("学習セット Middle スライス(512次元) かなCER 良い例 Top 5", Array(sortedTrainMiddle.prefix(5)))

// レポートファイルの生成
var reportContent = """
# \(datasetPath) 全 \(rawPairs.count) 発話 正誤率・CER 評価レポート

## 1. 概要
- **評価対象**: `\(datasetPath)` 全 \(rawPairs.count) 発話 (WAV + UTF-8 正解テキスト)
- **モデル**: `-s \(sampleLimit) -e \(epochs)` で学習した直接漢字音響 SNN (学習セット \(sampleLimit) 発話)
- **第1段 LIF/ALIF**: beta = \(alifBeta), rho = \(alifRho), gamma = \(alifGamma)
- **語彙・かな漢字辞書**: 学習セット \(trainTextLines.count) 件のみから構築 (未学習セットの正解テキストは不使用)
- **サンプリング**: 48kHz $\to$ 16kHz リサンプリング (アンチエイリアス 3:1 間引き)
- **デコーダ**: 音響直接デコード (Float32, `minDurationFrames = \(minDuration)`, `minConfidence = \(minConfidence)`, 短padマージ, 重複除外), 第2段 言語 SNN 加点 = \(languageBonus)
- **評価指標**:
  - **CER (Character Error Rate)**: $\\text{Levenshtein}(target, pred) / \\text{len}(target)$
  - **Exact Match 率**: 完全一致割合 (%)

---

## 2. 群別集計結果 (学習セット 3 件 vs 未学習セット 125 件)

### Base スライス (128次元)
| 群 | 件数 | 完全一致数 (件) | Exact 率 (%) | 平均 CER (%) | CER 中央値 (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **学習セット (Train)** | \(trainBaseSummary.count) | \(trainBaseSummary.exactCount) | \(String(format: "%.1f", trainBaseSummary.exactRate))% | \(String(format: "%.2f", trainBaseSummary.meanCer))% | \(String(format: "%.2f", trainBaseSummary.medianCer))% |
| **未学習セット (Unseen)** | \(unseenBaseSummary.count) | \(unseenBaseSummary.exactCount) | \(String(format: "%.1f", unseenBaseSummary.exactRate))% | \(String(format: "%.2f", unseenBaseSummary.meanCer))% | \(String(format: "%.2f", unseenBaseSummary.medianCer))% |

### High スライス (1024次元)
| 群 | 件数 | 完全一致数 (件) | Exact 率 (%) | 平均 CER (%) | CER 中央値 (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **学習セット (Train)** | \(trainHighSummary.count) | \(trainHighSummary.exactCount) | \(String(format: "%.1f", trainHighSummary.exactRate))% | \(String(format: "%.2f", trainHighSummary.meanCer))% | \(String(format: "%.2f", trainHighSummary.medianCer))% |
| **未学習セット (Unseen)** | \(unseenHighSummary.count) | \(unseenHighSummary.exactCount) | \(String(format: "%.1f", unseenHighSummary.exactRate))% | \(String(format: "%.2f", unseenHighSummary.meanCer))% | \(String(format: "%.2f", unseenHighSummary.medianCer))% |

---

## 3. 未学習セット (High スライス) の出力例

### 良い例 Top 5 (CER 昇順)
"""

for (i, r) in top5Best.enumerated() {
    reportContent += """

#### [\(i+1)] \(r.fileId) (CER: \(String(format: "%.1f", r.cer * 100.0))%, 編集距離: \(r.editDistance))
- **正解**: `\(r.targetText)`
- **出力**: `\(r.predText)`
"""
}

reportContent += """


### 悪い例 Top 5 (CER 降順)
"""

for (i, r) in top5Worst.enumerated() {
    reportContent += """

#### [\(i+1)] \(r.fileId) (CER: \(String(format: "%.1f", r.cer * 100.0))%, 編集距離: \(r.editDistance))
- **正解**: `\(r.targetText)`
- **出力**: `\(r.predText)`
"""
}

try? reportContent.write(toFile: reportPath, atomically: true, encoding: .utf8)
print("\n評価レポートを出力しました: \(reportPath)")

print("\n==================================================")
print("=== 全学習・全件文字起こし評価完了 ===")
print("==================================================")
