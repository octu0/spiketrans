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

// 学習・評価の既定パラメータ。
// いずれも loanword128 での掃引で最良だった値を採用している。
enum Defaults {
    /// 連続フレームを束ねる倍率。10ms/フレームは CTC には細かすぎ、
    /// 逐次カーネル起動回数がそのまま学習時間に効く。4 で 40ms 相当。
    static let frameStack = 4
    /// 1 フレームあたりの 3-tap Mel 次元 (従来モード)
    static let melFrameDim = 128
    /// 音響 SNN への入力次元 (従来モード)
    static var acousticInputDim: Int { return melFrameDim * frameStack }

    /// Conv2DSubsampling フロントエンド (Phase 2)
    static let melChannels = 64
    static let convOutputDim = 128

    /// 多層 SNN 既定層数
    static let numLayers = 1

    /// 隠れ層の次元
    static let maxHiddenDim = 1024

    /// 切り詰め BPTT の窓幅 (フレーム単位)。
    /// 40ms/フレーム換算で 8 フレーム = 320ms, 12 フレーム = 480ms。
    /// 日本語の形態素・共調音結合 (250〜400ms) をカバーする適正窓幅。
    static let bpttWindow = 8

    /// Cosine 学習率スケジュール
    static let lrMax: Float = 0.003
    static let lrMin: Float = 0.0005

    /// LIF 設定。ALIF (gamma > 0) は損失・CER とも悪化したため無効。
    static let lifConfig = LIFConfig(beta: 0.92, vTh: 1.0, vReset: 0.0, alpha: 2.0, rho: 0.85, gamma: 0.0)

    /// N エポックごとの重み書き出し
    static let checkpointEvery = 10

    /// 学習率の暖機に使うステップ数の分母 (全ステップの 1/N を暖機に充てる)
    static let warmupStepDivisor = 50

    /// MLX のバッファキャッシュを捨てる間隔 (バッチ数)
    static let clearCacheEveryBatches = 200

    /// 学習セットから評価する最大件数。
    /// 学習データが数十万件になると全件評価だけで何時間もかかるため、
    /// 再現性の確認はこの件数の抽出で足りる (未学習セットは常に全件評価する)
    static let maxTrainEvalSamples = 5000

    /// ミニバッチサイズ
    static let batchSize = 64

    /// 第2段 言語 SNN の予測一致加点。
    /// 現状は第1段の出力に対して効果が測定できなかったため 0 (言語 SNN の学習ごと省略)。
    static let languageBonus: Float = 0.0

    /// 推論時の CTC ブランク割引率 (0.0 で無効)
    static let blankPenalty: Float = 0.0
}

/// 並列評価の既定ワーカー数は P コア数。静的分割で E コアを混ぜると
/// 遅いワーカーが末尾まで残り、全体が P コアのみより遅くなる
func performanceCoreCount() -> Int {
    var count: Int32 = 0
    var size = MemoryLayout<Int32>.size
    if sysctlbyname("hw.perflevel0.logicalcpu", &count, &size, nil, 0) == 0 && 0 < count {
        return Int(count)
    }
    return ProcessInfo.processInfo.activeProcessorCount
}

// 1. コマンドライン引数の解析
var numWorkers = performanceCoreCount()
var epochs = 20
var maxTrainSamples: Int? = nil
var datasetPath = ""
var deviceArg = "auto"
var exportWeightsPath: String? = nil
var importWeightsPath: String? = nil
var useConvSubsampling = false
var numLayers = Defaults.numLayers
var bpttWindow = Defaults.bpttWindow
var hiddenDim = Defaults.maxHiddenDim
var customLR: Float? = nil
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
                epochs = max(0, val)
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
    case "-o", "--export-weights":
        if (argIdx + 1) < args.count {
            exportWeightsPath = args[argIdx + 1]
            argIdx += 1
        }
    case "--import-weights":
        if (argIdx + 1) < args.count {
            importWeightsPath = args[argIdx + 1]
            argIdx += 1
        }
    case "--conv-subsampling", "--conv":
        useConvSubsampling = true
    case "-l", "--layers":
        if (argIdx + 1) < args.count {
            if let val = Int(args[argIdx + 1]) {
                numLayers = max(1, val)
            }
            argIdx += 1
        }
    case "-w", "--bptt-window":
        if (argIdx + 1) < args.count {
            if let val = Int(args[argIdx + 1]) {
                bpttWindow = max(1, val)
            }
            argIdx += 1
        }
    case "--hidden-dim":
        if (argIdx + 1) < args.count {
            if let val = Int(args[argIdx + 1]) {
                hiddenDim = max(64, val)
            }
            argIdx += 1
        }
    case "--lr":
        if (argIdx + 1) < args.count {
            if let val = Float(args[argIdx + 1]) {
                customLR = val
            }
            argIdx += 1
        }
    default:
        if arg.hasPrefix("-") != true {
            datasetPath = arg
        }
    }
    argIdx += 1
}

// データセットのパスは必須。特定コーパスを既定値に埋め込まない
if datasetPath.isEmpty {
    print("エラー: 学習マニフェスト (JSONL) を指定してください。")
    print("  使い方: train -d <マニフェスト.jsonl> [-s 件数] [-e エポック数] [--conv-subsampling] [--layers <N>] [-w <窓幅>]")
    print("  各行: {\"path\": \"/path/to/voice.wav\", \"text\": \"漢字かな混じりの発話テキスト\"}")
    print("  マニフェストは script/dataset/ の各コーパス用スクリプトで生成する")
    exit(1)
}

// 事前にインポート重みを読み込んで自動判定
var preloadedWeights: SpikingNetworkWeights? = nil
switch importWeightsPath {
case .some(let impPath):
    let impURL = URL(fileURLWithPath: impPath)
    if let wData = try? SpikingNetworkWeights.load(from: impURL) {
        preloadedWeights = wData
        if 1 < wData.numLayers && numLayers == Defaults.numLayers {
            numLayers = wData.numLayers
        }
        if wData.convSubsampling != nil && useConvSubsampling != true {
            useConvSubsampling = true
        }
        if hiddenDim == Defaults.maxHiddenDim {
            hiddenDim = wData.maxHiddenDim
        }
    }
case .none:
    break
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

let deviceDesc: String
switch useGPU {
case true:
    deviceDesc = "Apple Silicon GPU (MLX Swift)"
case false:
    deviceDesc = "CPU (Pure Swift)"
}
print("データセットパス: \(datasetPath)")
print("実行デバイス   : \(deviceDesc)")
print("並列ワーカー数 (-p): \(numWorkers) スレッド")
print("エポック数     (-e): \(epochs) エポック")
print("SNN 層数       (-l): \(numLayers) 層")
let frontendDesc: String
switch useConvSubsampling {
case true:
    frontendDesc = "2D-Conv Subsampling (64ch Mel 入力 -> 4x 時間圧縮)"
case false:
    frontendDesc = "3-tap Mel × \(Defaults.frameStack) フレーム束ね (128次元)"
}
print("フロントエンド : \(frontendDesc)")
print("BPTT 窓幅      (-w): \(bpttWindow) フレーム (\(bpttWindow * 40) ms 相当)")
print("隠れ層次元     : \(hiddenDim) 次元")
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

// 2. JSONL マニフェストから発話リストを読み込み (漢字のまま)。
// 各行が {"path": ..., "text": ...} の 1 発話。コーパス固有の構造は
// script/dataset/ の生成スクリプト側で吸収し、学習側はこの形式だけを知る。
struct ManifestEntry: Codable {
    let path: String
    let text: String
}

var effectiveManifestPath = datasetPath
var isDir: ObjCBool = false
if FileManager.default.fileExists(atPath: datasetPath, isDirectory: &isDir) {
    switch isDir.boolValue {
    case true:
        let transcriptCandidate = (datasetPath as NSString).appendingPathComponent("transcript_utf8.txt")
        if FileManager.default.fileExists(atPath: transcriptCandidate) {
            effectiveManifestPath = transcriptCandidate
        }
    case false:
        if (datasetPath as NSString).lastPathComponent == "recording_info.txt" {
            let dir = (datasetPath as NSString).deletingLastPathComponent
            let transcriptCandidate = (dir as NSString).appendingPathComponent("transcript_utf8.txt")
            if FileManager.default.fileExists(atPath: transcriptCandidate) {
                effectiveManifestPath = transcriptCandidate
            }
        }
    }
}

guard let manifestContent = try? String(contentsOfFile: effectiveManifestPath, encoding: .utf8) else {
    print("エラー: マニフェスト \(datasetPath) が読み込めません。")
    exit(1)
}

let manifestDir = (effectiveManifestPath as NSString).deletingLastPathComponent
let jsonDecoder = JSONDecoder()
var textLines: [String] = []
var rawPairs: [(path: String, fileId: String, text: String)] = []

// JSONL または ID:テキスト (transcript_utf8.txt 等) をパース
for line in manifestContent.components(separatedBy: "\n") {
    let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
    if trimmed.isEmpty != true {
        let decodedEntry = try? jsonDecoder.decode(ManifestEntry.self, from: Data(trimmed.utf8))
        switch decodedEntry {
        case .some(let entry):
            var wavPath = entry.path
            if wavPath.hasPrefix("/") != true {
                wavPath = (manifestDir as NSString).appendingPathComponent(wavPath)
            }
            let fileId = ((wavPath as NSString).lastPathComponent as NSString).deletingPathExtension
            textLines.append(entry.text)
            rawPairs.append((path: wavPath, fileId: fileId, text: entry.text))
        case .none:
            let colonIdx = trimmed.firstIndex(of: ":")
            switch colonIdx {
            case .some(let idx):
                let fileId = String(trimmed[..<idx]).trimmingCharacters(in: .whitespaces)
                let text = String(trimmed[trimmed.index(after: idx)...]).trimmingCharacters(in: .whitespaces)
                let wavSubdir = (manifestDir as NSString).appendingPathComponent("wav")
                let candidateWav1 = (wavSubdir as NSString).appendingPathComponent("\(fileId).wav")
                let candidateWav2 = (manifestDir as NSString).appendingPathComponent("\(fileId).wav")
                let wavPath: String
                switch FileManager.default.fileExists(atPath: candidateWav1) {
                case true:
                    wavPath = candidateWav1
                case false:
                    wavPath = candidateWav2
                }
                if FileManager.default.fileExists(atPath: wavPath) {
                    textLines.append(text)
                    rawPairs.append((path: wavPath, fileId: fileId, text: text))
                }
            case .none:
                print("エラー: マニフェストの行を解釈できません: \(trimmed.prefix(120))")
                exit(1)
            }
        }
    }
}

if rawPairs.isEmpty {
    print("エラー: 有効な発話データが見つかりませんでした: \(datasetPath)")
    exit(1)
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

// 3. 遅延データセット構築 (メタデータのみ保持し、特徴量はバッチ生成時に WAV から作る)
print("\n--- 1. かな・漢字データセット構築 (最大 \(sampleLimit) 件、遅延読み込み) ---")
let startTime = CFAbsoluteTimeGetCurrent()

let manifestPairs: [(path: String, text: String)] = rawPairs.prefix(sampleLimit).map { pair in
    return (path: pair.path, text: pair.text)
}
let dsFrameStack: Int
switch useConvSubsampling {
case true:
    dsFrameStack = 1
case false:
    dsFrameStack = Defaults.frameStack
}
let dataset = SpeechDataset.lazyFromManifest(
    pairs: manifestPairs,
    textVocabulary: textVocabulary,
    frameStack: dsFrameStack,
    useMel: useConvSubsampling,
    workers: numWorkers
)

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
    let featTypeDesc: String
    switch useConvSubsampling {
    case true:
        featTypeDesc = "64ch Mel スペクトログラム (Conv2D Subsampling 前)"
    case false:
        featTypeDesc = "\(featDim)次元 3-tap Mel 特徴量"
    }
    print("      音響特徴量フレーム数: \(sample.acousticFeatures.count) フレーム (\(featTypeDesc))")
}

// 4. 第1段 音響 SNN (かな・音素) の学習実行
let devShortDesc: String
switch useGPU {
case true:
    devShortDesc = "GPU"
case false:
    devShortDesc = "CPU"
}
print("\n--- 2. 第1段 音響 SNN (かな・音素) の学習実行 (デバイス: \(devShortDesc)) ---")
let effectiveLR: Float
switch customLR {
case .some(let lr):
    effectiveLR = lr
case .none:
    switch useGPU {
    case true:
        effectiveLR = Defaults.lrMax
    case false:
        effectiveLR = 0.015
    }
}
let trainConfig = TrainingConfig(
    epochs: epochs,
    learningRate: effectiveLR,
    logInterval: 2,
    clipNorm: 5.0
)

let acousticInputDim: Int
switch useConvSubsampling {
case true:
    acousticInputDim = Defaults.convOutputDim
case false:
    acousticInputDim = Defaults.acousticInputDim
}
switch useConvSubsampling {
case true:
    print("音響特徴量: 64ch Mel 入力 -> Conv2D Subsampling -> \(acousticInputDim) 次元 SNN 注入")
case false:
    print("音響特徴量: \(acousticInputDim) 次元 (\(Defaults.melFrameDim) 次元 3-tap Mel × \(Defaults.frameStack) フレーム束ね)")
}

print("第1段 LIF: beta = \(Defaults.lifConfig.beta), 層数 = \(numLayers), 隠れ層 = \(hiddenDim)")

let convSubsampler: Conv2DSubsampling?
let mlxConvSubsampler: MLXConv2DSubsampling?
switch useConvSubsampling {
case true:
    convSubsampler = Conv2DSubsampling(melChannels: Defaults.melChannels, outputDim: Defaults.convOutputDim)
    mlxConvSubsampler = MLXConv2DSubsampling(melChannels: Defaults.melChannels, outputDim: Defaults.convOutputDim)
case false:
    convSubsampler = nil
    mlxConvSubsampler = nil
}

let acousticNetwork = SpikingNetwork(
    numLayers: numLayers,
    inputDim: acousticInputDim,
    maxHiddenDim: hiddenDim,
    outputDim: phoneticVocabulary.size,
    timeSteps: 4,
    lifConfig: Defaults.lifConfig,
    convSubsampling: convSubsampler
)

let trainer = Trainer(
    acousticNetwork: acousticNetwork,
    languageNetwork: SpikingNetwork(
        inputDim: 128,
        maxHiddenDim: hiddenDim,
        outputDim: textVocabulary.size,
        timeSteps: 4
    ),
    textVocabulary: phoneticVocabulary,
    phonemeVocabulary: PhonemeVocabulary(),
    config: trainConfig
)

// 重みのインポート。-e 0 なら評価のみ、-e N なら読み込んだ重みから追加学習する
var importedWeights: SpikingNetworkWeights? = nil
switch importWeightsPath {
case .some(let impPath):
    print("\n[重み読込] 外部ファイルからモデル重みをインポート中: \(impPath)")
    let wData: SpikingNetworkWeights
    switch preloadedWeights {
    case .some(let pre):
        wData = pre
    case .none:
        let impURL = URL(fileURLWithPath: impPath)
        guard let loaded = try? SpikingNetworkWeights.load(from: impURL) else {
            print("  ✕ 重みファイルの読み込みに失敗しました。")
            exit(1)
        }
        wData = loaded
    }
    guard wData.outputDim == phoneticVocabulary.size,
          wData.inputDim == acousticInputDim else {
        print("  ✕ 重みの次元が現在の構成と一致しません (入力 \(wData.inputDim)/\(acousticInputDim), 出力 \(wData.outputDim)/\(phoneticVocabulary.size))。")
        exit(1)
    }
    trainer.acousticTrainer.network.importWeights(from: wData)
    importedWeights = wData
    switch epochs {
    case 0:
        print("  ✓ 音響モデル重みのインポート完了。評価のみ実行します (-e 0)。")
    default:
        print("  ✓ 音響モデル重みのインポート完了。この重みから \(epochs) エポックの追加学習を行います。")
    }
case .none:
    break
}

let mlxNet: MLXSpikingNetwork
let mlxTrainer: MLXBPTTTrainer
let scheduler: CosineLRScheduler

switch (epochs == 0, useGPU) {
case (true, _):
    // 追加学習なし。インポート済み重みで評価へ進む
    mlxNet = MLXSpikingNetwork(
        numLayers: numLayers,
        inputDim: acousticInputDim,
        maxHiddenDim: hiddenDim,
        outputDim: phoneticVocabulary.size,
        timeSteps: 4,
        lifConfig: trainer.acousticTrainer.network.lifConfig,
        convSubsampling: mlxConvSubsampler
    )
    mlxTrainer = MLXBPTTTrainer(network: mlxNet, config: trainConfig, bpttWindow: bpttWindow)
    scheduler = CosineLRScheduler(lrMax: Defaults.lrMax, lrMin: Defaults.lrMin, totalEpochs: 1)
case (false, true):
    print("  Apple Silicon GPU (MLX Swift Metal) による並列ミニバッチ学習を開始 (バッチサイズ: \(Defaults.batchSize))...")

    // 教師はフレームに整列していないかな ID 列。アライメントは CTC が周辺化する。
    print("  [教師] CTC 損失: フレーム整列なしのかな ID 列")
    let allTargets: [[Int]] = (0..<dataset.count).map { idx in
        return phoneticVocabulary.textToIds(dataset.hiraganaText(at: idx))
    }

    mlxNet = MLXSpikingNetwork(
        numLayers: numLayers,
        inputDim: acousticInputDim,
        maxHiddenDim: hiddenDim,
        outputDim: phoneticVocabulary.size,
        timeSteps: 4,
        lifConfig: trainer.acousticTrainer.network.lifConfig,
        convSubsampling: mlxConvSubsampler
    )
    switch importedWeights {
    case .some(let wData):
        mlxNet.importWeights(from: wData)
        print("  [追加学習] インポート済み重みを GPU 学習の初期値に設定")
    case .none:
        break
    }
    mlxTrainer = MLXBPTTTrainer(
        network: mlxNet,
        config: trainConfig,
        bpttWindow: bpttWindow
    )
    print("  切り詰め BPTT 窓幅: \(bpttWindow) フレーム (\(bpttWindow * 40) ms 相当)")
    let lrMax = customLR ?? Defaults.lrMax
    let lrMin: Float
    switch customLR {
    case .some(let lr):
        lrMin = lr / 6.0
    case .none:
        lrMin = Defaults.lrMin
    }
    scheduler = CosineLRScheduler(lrMax: lrMax, lrMin: lrMin, totalEpochs: epochs, warmupEpochs: 1)
    print("  学習率: \(lrMax) → \(lrMin)")
    let trainStartTime = CFAbsoluteTimeGetCurrent()

    let batchSize = Defaults.batchSize

    // CTC はフレーム数 T >= ラベル数 + 連続重複数 を要求する。
    // これを満たさないサンプル (早口・短尺音声に長い教師) は尤度が定義できず、
    // 番兵 -1e30 が損失に漏れて学習を汚染するため除外する。
    func ctcMinimumFrames(_ labels: [Int]) -> Int {
        var required = labels.count
        var i = 1
        while i < labels.count {
            if labels[i] == labels[i - 1] {
                required += 1
            }
            i += 1
        }
        return required
    }
    var feasibleIndices: [Int] = []
    var infeasibleCount = 0
    var di = 0
    while di < dataset.count {
        let frames = dataset.frameCount(at: di)
        let effectiveFrames: Int
        switch useConvSubsampling {
        case true:
            let t1 = (frames + 1) / 2
            effectiveFrames = (t1 + 1) / 2
        case false:
            effectiveFrames = frames
        }
        if ctcMinimumFrames(allTargets[di]) <= effectiveFrames {
            feasibleIndices.append(di)
        } else {
            infeasibleCount += 1
        }
        di += 1
    }
    if 0 < infeasibleCount {
        print("  CTC 整合不可のため学習から除外: \(infeasibleCount) 件 (フレーム数 < 必要ラベル長)")
    }

    // 長さ順にバッチを組む。バッチ内の最長フレーム数までパディングされるため、
    // 長さの近いサンプルをまとめると無駄な逐次ステップが減る。
    let lengthSortedIndices = feasibleIndices.sorted { a, b in
        return dataset.frameCount(at: a) < dataset.frameCount(at: b)
    }
    var paddedFrameTotal = 0
    var unsortedFrameTotal = 0
    var probeStart = 0
    while probeStart < lengthSortedIndices.count {
        let probeEnd = min(probeStart + batchSize, lengthSortedIndices.count)
        var sortedMax = 0
        var plainMax = 0
        var pi = probeStart
        while pi < probeEnd {
            sortedMax = max(sortedMax, dataset.frameCount(at: lengthSortedIndices[pi]))
            plainMax = max(plainMax, dataset.frameCount(at: feasibleIndices[pi]))
            pi += 1
        }
        paddedFrameTotal += sortedMax
        unsortedFrameTotal += plainMax
        probeStart = probeEnd
    }
    print("  長さ順バッチング: 逐次フレーム総数 \(unsortedFrameTotal) → \(paddedFrameTotal)")

    // バッチの構成 (どのサンプルをどのバッチに入れるか) は全エポック共通
    var batchGroups: [[Int]] = []
    var gStart = 0
    while gStart < lengthSortedIndices.count {
        let gEnd = min(gStart + batchSize, lengthSortedIndices.count)
        batchGroups.append(Array(lengthSortedIndices[gStart..<gEnd]))
        gStart = gEnd
    }

    // バッチの特徴量を WAV から並列生成する (遅延読み込みの実体)
    final class FeatureBatchBuffer: @unchecked Sendable {
        var items: [[[Float]]]
        init(count: Int) {
            self.items = [[[Float]]](repeating: [], count: count)
        }
    }
    let batchUseConv = useConvSubsampling
    let batchFrameStack: Int
    switch useConvSubsampling {
    case true:
        batchFrameStack = 1
    case false:
        batchFrameStack = Defaults.frameStack
    }
    let batchWorkers = numWorkers
    let batchDataset = dataset

    @Sendable func buildBatchFeatures(_ indices: [Int]) -> [[[Float]]] {
        let buffer = FeatureBatchBuffer(count: indices.count)
        let workerCount = max(1, min(batchWorkers, indices.count))
        DispatchQueue.concurrentPerform(iterations: workerCount) { worker in
            var i = worker
            while i < indices.count {
                let meta = batchDataset.metaSamples[indices[i]]
                buffer.items[i] = SpeechDataset.loadFeatures(
                    path: meta.path,
                    frameStack: batchFrameStack,
                    useMel: batchUseConv
                ).features
                i += workerCount
            }
        }
        return buffer.items
    }

    // GPU がバッチを学習している間に、次バッチの特徴量を CPU で先読みする
    final class PrefetchBox: @unchecked Sendable {
        var value: [[[Float]]] = []
    }

    // 学習率はバッチ単位で刻む。エポック単位だとデータが増えたときに
    // 暖機の割合が大きすぎる (6 エポック中 4 エポックが暖機など)
    let totalSteps = max(1, epochs * batchGroups.count)
    let warmupSteps = max(1, totalSteps / Defaults.warmupStepDivisor)
    print("  学習ステップ数: \(totalSteps) (暖機 \(warmupSteps) ステップ)")

    let checkpointInterval = min(Defaults.checkpointEvery, max(1, epochs / 6))
    print("  チェックポイント間隔: \(checkpointInterval) エポックごと")

    var globalStep = 0
    var ep = 1
    while ep <= epochs {
        let epStartTime = CFAbsoluteTimeGetCurrent()
        var curLR = scheduler.learningRate(
            step: globalStep + 1, totalSteps: totalSteps, warmupSteps: warmupSteps)

        var epLossSum: Float = 0.0
        var batchCount = 0

        var currentFeatures: [[[Float]]] = []
        if 0 < batchGroups.count {
            currentFeatures = buildBatchFeatures(batchGroups[0])
        }
        var bIdx = 0
        while bIdx < batchGroups.count {
            let prefetchBox = PrefetchBox()
            let prefetchGroup = DispatchGroup()
            if (bIdx + 1) < batchGroups.count {
                let nextIndices = batchGroups[bIdx + 1]
                prefetchGroup.enter()
                DispatchQueue.global(qos: .userInitiated).async {
                    prefetchBox.value = buildBatchFeatures(nextIndices)
                    prefetchGroup.leave()
                }
            }

            var tBatch: [[Int]] = []
            for sampleIdx in batchGroups[bIdx] {
                tBatch.append(allTargets[sampleIdx])
            }

            globalStep += 1
            curLR = scheduler.learningRate(
                step: globalStep, totalSteps: totalSteps, warmupSteps: warmupSteps)
            mlxTrainer.updateLearningRate(curLR)

            let res = mlxTrainer.trainBatchCTC(
                featuresBatch: currentFeatures,
                targetsBatch: tBatch,
                blankId: TextVocabulary.padId
            )
            epLossSum += res
            batchCount += 1

            // MLX は解放したバッファを形ごとに使い回すため、系列長が毎バッチ違うと
            // キャッシュが際限なく増えて Metal のリソース上限に達する。定期的に捨てる
            if (batchCount % Defaults.clearCacheEveryBatches) == 0 {
                MLX.Memory.clearCache()
            }

            prefetchGroup.wait()
            currentFeatures = prefetchBox.value
            bIdx += 1
        }

        let avgLoss = epLossSum / Float(max(1, batchCount))
        let epElapsed = CFAbsoluteTimeGetCurrent() - epStartTime
        print("  Epoch [\(ep)/\(epochs)] - 音響損失: \(String(format: "%.4f", avgLoss)) (LR: \(String(format: "%.5f", curLR)), 所要時間: \(String(format: "%.2f", epElapsed)) 秒)")

        // 定期チェックポイント: 長時間実行が途中で止まっても成果を失わないようにする。
        // エポック数が少ない大規模学習では 10 エポックごとだと 1 度も保存されないため、
        // 全体の 1/6 を上限に間隔を詰める
        if 0 < checkpointInterval && (ep % checkpointInterval) == 0 && ep < epochs {
            switch exportWeightsPath {
            case .some(let basePath):
                let ckptPath = "\(basePath).ep\(ep).json"
                do {
                    try mlxNet.exportWeights(vocabulary: phoneticVocabulary).save(to: URL(fileURLWithPath: ckptPath))
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
    let exported = mlxNet.exportWeights(vocabulary: phoneticVocabulary)
    trainer.acousticTrainer.network.importWeights(from: exported)
    print("  ✓ GPU 学習パラメータを Pure Swift 推論エンジンに転送完了")

    // MLX Metal GPU メモリ・キャッシュを解放
    #if canImport(MLX)
    MLX.Memory.clearCache()
    #endif
case (false, false):
    print("  CPU (Pure Swift \(numWorkers) スレッド) による SNN-CTC 損失並列学習を開始...")
    let trainStartTime = CFAbsoluteTimeGetCurrent()
    var acResults: [EpochResult] = []
    var ep = 1
    while ep <= epochs {
        let epStartTime = CFAbsoluteTimeGetCurrent()
        let acRes = trainer.acousticTrainer.trainCTCEpoch(
            dataset: dataset,
            kanaVocabulary: phoneticVocabulary,
            epoch: ep,
            numWorkers: numWorkers
        )
        acResults.append(acRes)
        let epElapsed = CFAbsoluteTimeGetCurrent() - epStartTime
        print("  Epoch [\(ep)/\(epochs)] - 音響損失: \(String(format: "%.4f", acRes.totalLoss)) (所要時間: \(String(format: "%.2f", epElapsed)) 秒)")
        ep += 1
    }
    let trainElapsed = CFAbsoluteTimeGetCurrent() - trainStartTime
    print("\nCPU 学習完了 (総所要時間: \(String(format: "%.3f", trainElapsed)) 秒)")
}

// 4.5 第2段 漢字自己回帰言語 SNN の学習 (CPU マルチスレッド)
// --language-bonus 0 のときは言語 SNN の出力が第2段で一切使われないため学習を丸ごと省略する
if 0.0 < Defaults.languageBonus {
    print("\n--- 2.5 第2段 漢字自己回帰言語 SNN の学習 (CPU マルチスレッド) ---")
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
            print("  LM Epoch [\(lmEpoch)/\(lmMaxEpochs)] - 損失: \(String(format: "%.4f", res.totalLoss))")
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
    let wData = trainer.acousticTrainer.network.exportWeights(vocabulary: phoneticVocabulary)
    do {
        try wData.save(to: expURL)
        print("  ✓ 重みパラメータのエクスポートが完了しました: \(expPath)")
    } catch {
        print("  ✕ 重みパラメータの保存に失敗しました: \(error)")
    }
}

// 5. 学習済みモデルによるスライス別 (Base / Middle / High) 推論テスト
print("\n--- 3. 音声文字起こしテスト ---")

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
        
        let attenRatio: Float
        switch 0.0 < rawOutOfBandTotal {
        case true:
            attenRatio = eqOutOfBandTotal / rawOutOfBandTotal
        case false:
            attenRatio = 0.20
        }
        let attenDb: Float
        switch 0.0 < attenRatio {
        case true:
            attenDb = -10.0 * log10(attenRatio)
        case false:
            attenDb = 7.0
        }
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

    // 2. 音響 SNN のフレーム別予測
    let acDec = AcousticDecoder(
        network: trainer.acousticTrainer.network,
        vocabulary: phoneticVocabulary
    )
    let acWs = AcousticWorkspace(
        maxHiddenDim: trainer.acousticTrainer.network.maxHiddenDim,
        outputDim: phoneticVocabulary.size,
        inputDim: trainer.acousticTrainer.network.inputDim,
        numLayers: trainer.acousticTrainer.network.numLayers
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

    let totalEvaluatedFrames = frameProbs.count
    let padRatio = Float(predPad) * 100.0 / Float(max(1, totalEvaluatedFrames))
    print("\n[2] 音響 SNN 全フレーム予測集計:")
    print("  pad: \(predPad) (\(String(format: "%.1f", padRatio))%), eos: \(predEos), unk: \(predUnk), 文字: \(predChar)")
    print("  推論 pad 率: \(String(format: "%.2f", padRatio))%")
    print("  ユニーク Top-1 トークン数: \(uniqueTopTokens.count)")

    var correctFrameCount = 0
    var fIdx = 0
    let checkLimit = min(totalF, totalEvaluatedFrames)
    while fIdx < checkLimit {
        let pred = frameProbs[fIdx].topTokenId
        let tgt = targets[fIdx]
        if pred == tgt {
            correctFrameCount += 1
        }
        fIdx += 1
    }
    let frameAccuracy = Float(correctFrameCount) * 100.0 / Float(max(1, checkLimit))
    print("  フレーム正解率 (alignTargets 対 argmax): \(String(format: "%.2f", frameAccuracy))% (\(correctFrameCount)/\(max(1, checkLimit)))")

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

    let headEnd = min(20, totalEvaluatedFrames)
    print("--- 先頭 \(headEnd) フレーム (0..\(max(0, headEnd - 1))) ---")
    for i in 0..<headEnd { printFrame(i) }

    let midStart = max(0, (totalEvaluatedFrames / 2) - 10)
    let midEnd = min(totalEvaluatedFrames, midStart + 20)
    print("--- 中間 \(max(0, midEnd - midStart)) フレーム (\(midStart)..\(max(midStart, midEnd - 1))) ---")
    for i in midStart..<midEnd { printFrame(i) }

    let tailStart = max(0, totalEvaluatedFrames - 20)
    print("--- 末尾 \(max(0, totalEvaluatedFrames - tailStart)) フレーム (\(tailStart)..\(max(tailStart, totalEvaluatedFrames - 1))) ---")
    for i in tailStart..<totalEvaluatedFrames { printFrame(i) }

    // 3. 音響直接文字起こしの実行
    let directText = trainer.transcribeAcousticDirect(
        featuresSeq: feat0,
        minDurationFrames: 3,
        minConfidence: 0.05
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
    print("\n[4] 音響直接デコード 実行結果:")
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

        // 2段階音声文字起こし (音響かな推定 -> 辞書 Viterbi 漢字復元)
        let bList = FormantSegmenter.detectBoundaries(pcmData: testSample.audioPCM)
        let t0 = CFAbsoluteTimeGetCurrent()
        let resTwoStage = trainer.transcribeTwoStage(
            featuresSeq: testSample.acousticFeatures,
            kanjiVocabulary: textVocabulary,
            dictionary: kanaKanjiDict,
            minDurationFrames: 3,
            minConfidence: 0.05,
            boundaries: bList,
            useCTC: true
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

        print("      • 第1段 かな音響推定: \"\(resTwoStage.kana)\"")
        print("      • 第2段 漢字復元推論 (\(String(format: "%.1f", dt)) ms): \"\(resTwoStage.kanji)\" (文字数: \(resTwoStage.kanji.count), 正解再現: \(mCount)/\(testSample.rawText.count) [\(String(format: "%.1f", recP))%], 誤挿入: \(insCount))")
    }
}

// ==================================================
// === 4. データセット全件の正誤率・CER 評価 ===
// ==================================================
print("\n==================================================")
print("=== 4. 音響直接デコード CER 評価 ===")
print("==================================================")

/// 句読点を除去した文字列 (句読点の寄与を分離して測るため)
@Sendable
func stripPunctuation(_ text: String) -> String {
    var out = ""
    for c in text {
        switch c {
        case "、", "。", "，", "．", "・":
            break
        default:
            out.append(c)
        }
    }
    return out
}

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
    /// 由来コーパス (WAV の親ディレクトリ名)。複数コーパス混合時の内訳集計用
    let corpus: String
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
    // 正解かなをそのまま第2段に入れた結果。第2段単体の実力を切り分けるための指標
    let goldKanaKanji: String
    let goldKanaKanjiCer: Float
    // 句読点を除いた比較 (句読点が CER にどれだけ寄与しているかの切り分け)
    let goldKanaKanjiCerNoPunct: Float
    let goldExactNoPunct: Bool
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
    // 正解かなを第2段に入れたときの漢字 CER (第2段単体の実力)
    let meanGoldKanjiCer: Float
    let goldExactRate: Float
    let meanGoldKanjiCerNoPunct: Float
    let goldExactRateNoPunct: Float
}

func computeSummary(_ results: [EvalResult]) -> GroupSummary {
    if results.isEmpty {
        return GroupSummary(
            count: 0, exactCount: 0, exactRate: 0.0,
            meanCer: 0.0, medianCer: 0.0,
            meanKanaCer: 0.0, medianKanaCer: 0.0,
            meanGoldKanjiCer: 0.0, goldExactRate: 0.0,
            meanGoldKanjiCerNoPunct: 0.0, goldExactRateNoPunct: 0.0
        )
    }
    let n = results.count
    var exact = 0
    var sumCer: Float = 0.0
    var sumKanaCer: Float = 0.0
    var sumGoldCer: Float = 0.0
    var goldExact = 0
    var sumGoldCerNoPunct: Float = 0.0
    var goldExactNoPunct = 0
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
        sumGoldCer += r.goldKanaKanjiCer
        if r.goldKanaKanji == r.targetText {
            goldExact += 1
        }
        sumGoldCerNoPunct += r.goldKanaKanjiCerNoPunct
        if r.goldExactNoPunct {
            goldExactNoPunct += 1
        }
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
        medianKanaCer: kanaCers[n / 2] * 100.0,
        meanGoldKanjiCer: (sumGoldCer / Float(n)) * 100.0,
        goldExactRate: Float(goldExact) * 100.0 / Float(n),
        meanGoldKanjiCerNoPunct: (sumGoldCerNoPunct / Float(n)) * 100.0,
        goldExactRateNoPunct: Float(goldExactNoPunct) * 100.0 / Float(n)
    )
}

let parser = WavParser()
let dummyEval = EvalResult(index: 0, fileId: "", corpus: "", targetText: "", predText: "", editDistance: 0, cer: 1.0, isExact: false, isTrain: false, targetKana: "", predKana: "", kanaCer: 1.0, goldKanaKanji: "", goldKanaKanjiCer: 1.0, goldKanaKanjiCerNoPunct: 1.0, goldExactNoPunct: false)

final class BatchEvalBuffer: @unchecked Sendable {
    var results: [EvalResult]
    var valid: [Bool]
    init(count: Int, dummy: EvalResult) {
        self.results = [EvalResult](repeating: dummy, count: count)
        self.valid = [Bool](repeating: false, count: count)
    }
}

// 評価対象の選定。
// 未学習セットは全件、学習セットは等間隔で間引く。
// 間引きは先頭からの連続ではなく全域から取るため、コーパスの偏りが出ない
var evalIndices: [Int] = []
let trainEvalTarget = min(Defaults.maxTrainEvalSamples, sampleLimit)
if trainEvalTarget < sampleLimit {
    // 比例配分で全域から均等に取る。指定件数がそのまま得られる
    var previousBucket = -1
    var selectIdx = 0
    while selectIdx < sampleLimit {
        let bucket = (selectIdx * trainEvalTarget) / sampleLimit
        if bucket != previousBucket {
            evalIndices.append(selectIdx)
            previousBucket = bucket
        }
        selectIdx += 1
    }
} else {
    var selectIdx = 0
    while selectIdx < sampleLimit {
        evalIndices.append(selectIdx)
        selectIdx += 1
    }
}
var unseenIdx = sampleLimit
while unseenIdx < rawPairs.count {
    evalIndices.append(unseenIdx)
    unseenIdx += 1
}

let evalBuffer = BatchEvalBuffer(count: evalIndices.count, dummy: dummyEval)
let evalLanguageBonus = Defaults.languageBonus
let evalPairs = evalIndices.map { rawPairs[$0] }
let evalIsTrain = evalIndices.map { $0 < sampleLimit }
let evalKanjiConverter = KanjiConverter()
let evalFrameStack = Defaults.frameStack
let evalUseConv = useConvSubsampling

let evalWorkers = max(1, numWorkers)
let unseenEvalCount = rawPairs.count - sampleLimit
if trainEvalTarget < sampleLimit {
    print("評価対象: 学習セット \(trainEvalTarget) 件 (\(sampleLimit) 件から均等抽出) + 未学習セット \(unseenEvalCount) 件 (全件)")
} else {
    print("評価対象: 学習セット \(trainEvalTarget) 件 + 未学習セット \(unseenEvalCount) 件 (いずれも全件)")
}
print("全 \(evalPairs.count) 件の WAV 読み込み・並列推論実行中 (\(evalWorkers) ワーカー)...")
let evalStartTime = Date()
DispatchQueue.concurrentPerform(iterations: evalWorkers) { worker in
    // 第2段デコーダはワーカー内で使い回し、発話ごとの再確保を避ける
    let goldDecoder = KanaKanjiDecoder(dictionary: kanaKanjiDict, languageBonus: 0.0)
    func processUtterance(_ idx: Int) {
        let pair = evalPairs[idx]
        let wavPath = pair.path
        if FileManager.default.fileExists(atPath: wavPath) != true {
            return
        }
        guard let fileData = try? Data(contentsOf: URL(fileURLWithPath: wavPath)),
              let wavData = try? parser.parse(bytes: [UInt8](fileData)) else {
            return
        }

        let pcm16k = SpeechDataset.resampleTo16k(pcmData: wavData.pcmData, sampleRate: wavData.sampleRate)
        let features: [[Float]]
        switch evalUseConv {
        case true:
            features = SpeechDataset.extractMelSpectrogram(pcmData: pcm16k)
        case false:
            features = SpeechDataset.extractFeaturesFromPCM(pcmData: pcm16k, frameStack: evalFrameStack)
        }
        let boundaries = FormantSegmenter.detectBoundaries(pcmData: pcm16k)
        let isTrain = evalIsTrain[idx]

        // 正解のかな読み (第1段の評価基準)
        let targetKana = evalKanjiConverter.convertToHiragana(pair.text)

        func evaluateUtterance() -> EvalResult {
            let res = trainer.transcribeTwoStage(
                featuresSeq: features,
                kanjiVocabulary: textVocabulary,
                dictionary: kanaKanjiDict,
                minDurationFrames: 3,
                minConfidence: 0.05,
                boundaries: boundaries,
                useCTC: true,
                languageBonus: evalLanguageBonus,
                blankPenalty: Defaults.blankPenalty
            )
            let dist = levenshteinDistance(pair.text, res.kanji)
            let cer = Float(dist) / Float(max(1, pair.text.count))
            let kanaDist = levenshteinDistance(targetKana, res.kana)
            let kanaCer = Float(kanaDist) / Float(max(1, targetKana.count))

            // 第2段単体の実力: 正解かなを入力したときの漢字復元
            let goldKanji = goldDecoder.decode(kanaText: targetKana)
            let goldDist = levenshteinDistance(pair.text, goldKanji)
            let goldCer = Float(goldDist) / Float(max(1, pair.text.count))

            let targetNoPunct = stripPunctuation(pair.text)
            let goldNoPunct = stripPunctuation(goldKanji)
            let goldDistNoPunct = levenshteinDistance(targetNoPunct, goldNoPunct)
            let goldCerNoPunct = Float(goldDistNoPunct) / Float(max(1, targetNoPunct.count))

            var corpusDir = ((pair.path as NSString).deletingLastPathComponent as NSString).lastPathComponent
            if corpusDir == "wav" {
                let parent = ((pair.path as NSString).deletingLastPathComponent as NSString).deletingLastPathComponent
                corpusDir = (parent as NSString).lastPathComponent
            }
            return EvalResult(
                index: idx + 1,
                fileId: pair.fileId,
                corpus: corpusDir,
                targetText: pair.text,
                predText: res.kanji,
                editDistance: dist,
                cer: cer,
                isExact: pair.text == res.kanji,
                isTrain: isTrain,
                targetKana: targetKana,
                predKana: res.kana,
                kanaCer: kanaCer,
                goldKanaKanji: goldKanji,
                goldKanaKanjiCer: goldCer,
                goldKanaKanjiCerNoPunct: goldCerNoPunct,
                goldExactNoPunct: targetNoPunct == goldNoPunct
            )
        }

        evalBuffer.results[idx] = evaluateUtterance()
        evalBuffer.valid[idx] = true
    }
    var idx = worker
    while idx < evalPairs.count {
        processUtterance(idx)
        idx += evalWorkers
    }
}
print(String(format: "評価所要時間: %.1f 秒 (%d ワーカー)", Date().timeIntervalSince(evalStartTime), evalWorkers))

var allEval: [EvalResult] = []
var evIdx = 0
while evIdx < evalPairs.count {
    if evalBuffer.valid[evIdx] {
        allEval.append(evalBuffer.results[evIdx])
    }
    evIdx += 1
}

let trainResults = allEval.filter { $0.isTrain }
let unseenResults = allEval.filter { $0.isTrain != true }

let trainSummary = computeSummary(trainResults)
let unseenSummary = computeSummary(unseenResults)

// 複数コーパス混合時はコーパスごとの内訳も出す
let corpusGroups = Dictionary(grouping: allEval, by: { $0.corpus })
func printCorpusBreakdown() {
    if corpusGroups.count < 2 {
        return
    }
    print("\n=== [コーパス別内訳] ===")
    for (name, results) in corpusGroups.sorted(by: { $0.key < $1.key }) {
        let cs = computeSummary(results)
        print("  • \(name) (\(cs.count)件): Exact率: \(String(format: "%.1f", cs.exactRate))%, 漢字CER: \(String(format: "%.2f", cs.meanCer))% (中央値 \(String(format: "%.2f", cs.medianCer))%), かなCER: \(String(format: "%.2f", cs.meanKanaCer))% (中央値 \(String(format: "%.2f", cs.medianKanaCer))%)")
    }
}

func printSliceSummary(
    _ label: String,
    train: GroupSummary,
    unseen: GroupSummary
) {
    print("\n[\(label)]")
    print("  • 学習セット (\(train.count)件):   Exact率: \(String(format: "%.1f", train.exactRate))% (\(train.exactCount)/\(train.count)), 漢字CER: \(String(format: "%.2f", train.meanCer))% (中央値 \(String(format: "%.2f", train.medianCer))%), かなCER: \(String(format: "%.2f", train.meanKanaCer))% (中央値 \(String(format: "%.2f", train.medianKanaCer))%)")
    print("  • 未学習セット (\(unseen.count)件): Exact率: \(String(format: "%.1f", unseen.exactRate))% (\(unseen.exactCount)/\(unseen.count)), 漢字CER: \(String(format: "%.2f", unseen.meanCer))% (中央値 \(String(format: "%.2f", unseen.medianCer))%), かなCER: \(String(format: "%.2f", unseen.meanKanaCer))% (中央値 \(String(format: "%.2f", unseen.medianKanaCer))%)")
    print("  ── 第2段単体 (正解かな入力時の漢字CER) ──")
    print("     句読点あり  学習: \(String(format: "%.2f", train.meanGoldKanjiCer))% (完全一致 \(String(format: "%.1f", train.goldExactRate))%) / 未学習: \(String(format: "%.2f", unseen.meanGoldKanjiCer))% (完全一致 \(String(format: "%.1f", unseen.goldExactRate))%)")
    print("     句読点除外  学習: \(String(format: "%.2f", train.meanGoldKanjiCerNoPunct))% (完全一致 \(String(format: "%.1f", train.goldExactRateNoPunct))%) / 未学習: \(String(format: "%.2f", unseen.meanGoldKanjiCerNoPunct))% (完全一致 \(String(format: "%.1f", unseen.goldExactRateNoPunct))%)")
}

// ==================================================
// === かな文字単位のエラー分析 (スライス別) ===
// ==================================================

/// 日本語かなの調音クラス。Base で破裂音・摩擦音が落ちているのかを切り分けるための分類。
func kanaArticulationClass(_ ch: Character) -> String {
    let voicelessPlosive: Set<Character> = [
        "か", "き", "く", "け", "こ", "た", "ち", "つ", "て", "と",
        "ぱ", "ぴ", "ぷ", "ぺ", "ぽ"
    ]
    let fricative: Set<Character> = [
        "さ", "し", "す", "せ", "そ", "は", "ひ", "ふ", "へ", "ほ"
    ]
    let voicedPlosive: Set<Character> = [
        "が", "ぎ", "ぐ", "げ", "ご", "だ", "ぢ", "づ", "で", "ど",
        "ば", "び", "ぶ", "べ", "ぼ", "ざ", "じ", "ず", "ぜ", "ぞ"
    ]
    let nasal: Set<Character> = ["な", "に", "ぬ", "ね", "の", "ま", "み", "む", "め", "も", "ん"]
    let approximant: Set<Character> = ["や", "ゆ", "よ", "ら", "り", "る", "れ", "ろ", "わ", "を"]
    let vowel: Set<Character> = ["あ", "い", "う", "え", "お"]
    let special: Set<Character> = ["っ", "ー", "ゃ", "ゅ", "ょ", "ぁ", "ぃ", "ぅ", "ぇ", "ぉ", "ゔ"]

    if voicelessPlosive.contains(ch) { return "無声破裂音 (k/t/p)" }
    if fricative.contains(ch) { return "摩擦音 (s/h)" }
    if voicedPlosive.contains(ch) { return "有声阻害音 (g/d/b/z)" }
    if nasal.contains(ch) { return "鼻音 (n/m/N)" }
    if approximant.contains(ch) { return "半母音・流音 (y/r/w)" }
    if vowel.contains(ch) { return "母音" }
    if special.contains(ch) { return "特殊 (促音・長音・拗音)" }
    return "その他"
}

/// 正解かなと推論かなを編集距離でアラインし、正解文字ごとの正解/置換/脱落を数える
func accumulateKanaErrors(
    target: String,
    pred: String,
    correct: inout [String: Int],
    substituted: inout [String: Int],
    deleted: inout [String: Int],
    insertedTotal: inout Int
) {
    let a = Array(target)
    let b = Array(pred)
    let m = a.count
    let n = b.count

    var dp = [[Int]](repeating: [Int](repeating: 0, count: n + 1), count: m + 1)
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
            var cost = 1
            if a[i - 1] == b[j - 1] {
                cost = 0
            }
            var best = dp[i - 1][j] + 1
            if dp[i][j - 1] + 1 < best { best = dp[i][j - 1] + 1 }
            if dp[i - 1][j - 1] + cost < best { best = dp[i - 1][j - 1] + cost }
            dp[i][j] = best
            j += 1
        }
        i += 1
    }

    // バックトレースして操作列を復元
    i = m
    j = n
    while 0 < i || 0 < j {
        if 0 < i && 0 < j {
            var cost = 1
            if a[i - 1] == b[j - 1] {
                cost = 0
            }
            if dp[i][j] == dp[i - 1][j - 1] + cost {
                let cls = kanaArticulationClass(a[i - 1])
                if cost == 0 {
                    correct[cls, default: 0] += 1
                } else {
                    substituted[cls, default: 0] += 1
                }
                i -= 1
                j -= 1
                continue
            }
        }
        if 0 < i && dp[i][j] == dp[i - 1][j] + 1 {
            deleted[kanaArticulationClass(a[i - 1]), default: 0] += 1
            i -= 1
            continue
        }
        insertedTotal += 1
        j -= 1
    }
}

func printKanaErrorAnalysis(_ label: String, _ results: [EvalResult]) {
    var correct: [String: Int] = [:]
    var substituted: [String: Int] = [:]
    var deleted: [String: Int] = [:]
    var insertedTotal = 0

    for r in results {
        accumulateKanaErrors(
            target: r.targetKana,
            pred: r.predKana,
            correct: &correct,
            substituted: &substituted,
            deleted: &deleted,
            insertedTotal: &insertedTotal
        )
    }

    print("\n[\(label)] 調音クラス別の正解率 (挿入合計: \(insertedTotal) 文字)")
    let classOrder = [
        "無声破裂音 (k/t/p)", "摩擦音 (s/h)", "有声阻害音 (g/d/b/z)",
        "鼻音 (n/m/N)", "半母音・流音 (y/r/w)", "母音", "特殊 (促音・長音・拗音)", "その他"
    ]
    for cls in classOrder {
        let c = correct[cls] ?? 0
        let sub = substituted[cls] ?? 0
        let del = deleted[cls] ?? 0
        let total = c + sub + del
        if total == 0 {
            continue
        }
        let rate = Float(c) * 100.0 / Float(total)
        print("  \(cls): 正解 \(String(format: "%.1f", rate))% (正解 \(c) / 置換 \(sub) / 脱落 \(del), 計 \(total))")
    }
}

// ==================================================
// === スライス別 推論レイテンシ計測 (単一スレッド) ===
// ==================================================
// 評価本体はマルチスレッドで走るためコア競合で per-utterance の値が歪む。
// レイテンシは単一スレッドで測り直す。
if 0 < rawPairs.count {
    let benchCount = min(20, rawPairs.count)
    var benchFeatures: [[[Float]]] = []
    var benchAudioSeconds: Double = 0.0
    var benchIdx = 0
    while benchIdx < benchCount {
        let pair = rawPairs[benchIdx]
        let wavPath = pair.path
        if let fileData = try? Data(contentsOf: URL(fileURLWithPath: wavPath)),
           let wavData = try? parser.parse(bytes: [UInt8](fileData)) {
            let pcm16k = SpeechDataset.resampleTo16k(pcmData: wavData.pcmData, sampleRate: wavData.sampleRate)
            benchAudioSeconds += Double(pcm16k.count) / 16000.0
            switch useConvSubsampling {
            case true:
                benchFeatures.append(SpeechDataset.extractMelSpectrogram(pcmData: pcm16k))
            case false:
                benchFeatures.append(SpeechDataset.extractFeaturesFromPCM(pcmData: pcm16k, frameStack: Defaults.frameStack))
            }
        }
        benchIdx += 1
    }

    print("\n==================================================")
    print("=== [推論レイテンシ] 単一スレッド, \(benchFeatures.count) 発話 (音声 \(String(format: "%.1f", benchAudioSeconds)) 秒) ===")
    print("==================================================")

    let benchNetwork = trainer.acousticTrainer.network
    do {
        let decoder = AcousticDecoder(
            network: benchNetwork,
            vocabulary: phoneticVocabulary,
            fallbackVocabulary: PhonemeVocabulary()
        )
        let ws = AcousticWorkspace(
            maxHiddenDim: benchNetwork.maxHiddenDim,
            outputDim: benchNetwork.outputDim,
            inputDim: benchNetwork.inputDim,
            numLayers: benchNetwork.numLayers
        )

        // 第1段 音響 SNN のみのフォワード時間
        let acousticStart = CFAbsoluteTimeGetCurrent()
        var totalFrames = 0
        for feats in benchFeatures {
            ws.reset()
            let probs = decoder.decodeSequence(featuresSeq: feats, workspace: ws)
            totalFrames += probs.count
        }
        let acousticElapsed = CFAbsoluteTimeGetCurrent() - acousticStart

        // 第1段 + CTC ビーム探索まで含めた文字起こし時間
        let fullStart = CFAbsoluteTimeGetCurrent()
        for feats in benchFeatures {
            let text = trainer.transcribeAcousticCTC(featuresSeq: feats, beamWidth: 16)
            if text.isEmpty && false {
                print("")
            }
        }
        let fullElapsed = CFAbsoluteTimeGetCurrent() - fullStart

        let n = Double(max(1, benchFeatures.count))
        let acousticMs = (acousticElapsed / n) * 1000.0
        let fullMs = (fullElapsed / n) * 1000.0
        let rtf = fullElapsed / max(1e-9, benchAudioSeconds)
        let speedup = max(1e-9, benchAudioSeconds) / fullElapsed

        print("[隠れ層 \(benchNetwork.maxHiddenDim)次元]")
        print("  音響 SNN のみ: \(String(format: "%.2f", acousticMs)) ms/発話 (\(totalFrames) フレーム処理)")
        print("  かな文字起こし全体 (SNN + CTC ビーム): \(String(format: "%.2f", fullMs)) ms/発話")
        print("  RTF: \(String(format: "%.4f", rtf)) (実時間の \(String(format: "%.0f", speedup)) 倍速)")
    }
}

// ==================================================
// === 第2段 Viterbi の選択内訳 (正解かな入力) ===
// ==================================================
// 正解かなを入れても誤る発話について、どの区間でどの語がどの経路・何点で
// 選ばれたかを出す。配点のどこが誤選択を招いているかを特定するため。
do {
    // 正解かな入力時の CER が高い順に数件を選ぶ
    let worstGold = trainResults
        .filter { 0.0 < $0.goldKanaKanjiCerNoPunct }
        .sorted { b, a in a.goldKanaKanjiCerNoPunct < b.goldKanaKanjiCerNoPunct }
    let sampleCount = min(3, worstGold.count)

    print("\n==================================================")
    print("=== [第2段 診断] 正解かな入力時の Viterbi 選択内訳 (誤り上位 \(sampleCount) 件) ===")
    print("==================================================")

    var d = 0
    while d < sampleCount {
        let r = worstGold[d]
        let diagDecoder = KanaKanjiDecoder(dictionary: kanaKanjiDict, languageBonus: 0.0)
        let produced = diagDecoder.decode(kanaText: r.targetKana)

        print("\n[\(r.fileId)] 句読点除外 CER: \(String(format: "%.1f", r.goldKanaKanjiCerNoPunct * 100.0))%")
        print("  正解かな: \"\(r.targetKana)\"")
        print("  正解漢字: \"\(r.targetText)\"")
        print("  第2段出力: \"\(produced)\"")
        print("  選択内訳:")
        for seg in diagDecoder.lastTrace {
            print("    \(seg.kanaRange) → \(seg.emitted)  [\(seg.kind)] \(String(format: "%+.1f", seg.stepScore))")
        }
        d += 1
    }
}

print("\n==================================================")
print("=== [かな文字単位エラー分析] 学習セット ===")
print("==================================================")
printKanaErrorAnalysis("学習セット", trainResults)
printKanaErrorAnalysis("未学習セット", unseenResults)

print("\n==================================================")
print("=== [集計結果] 学習セット (\(trainSummary.count)件) vs 未学習セット (\(unseenSummary.count)件) ===")
print("=== かなCER = 第1段 音響 SNN の音素発火精度 / 漢字CER = 第2段 通過後の最終精度 ===")
print("==================================================")
printSliceSummary("隠れ層 \(Defaults.maxHiddenDim)次元", train: trainSummary, unseen: unseenSummary)
printCorpusBreakdown()

// 未学習セットのソート (High スライスの CER 順)
let sortedUnseen = unseenResults.sorted { $0.cer < $1.cer }
let top5Best = Array(sortedUnseen.prefix(5))
let top5Worst = Array(sortedUnseen.suffix(5).reversed())

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

printExamples("未学習セット 良い例 Top 5", top5Best)
printExamples("未学習セット 悪い例 Top 5", top5Worst)



// レポートファイルの生成
var reportContent = """
# \(datasetPath) 正誤率・CER 評価レポート

## 1. 概要
- **評価対象**: `\(datasetPath)` \(evalPairs.count) 発話 (学習セットは間引き / 未学習セットは全件)
- **モデル**: `-s \(sampleLimit) -e \(epochs)` で学習した直接漢字音響 SNN (層数: \(numLayers), 隠れ層: \(hiddenDim), 学習セット \(sampleLimit) 発話)
- **第1段 LIF**: beta = \(Defaults.lifConfig.beta), rho = \(Defaults.lifConfig.rho), gamma = \(Defaults.lifConfig.gamma)
- **フロントエンド**: \(frontendDesc)
- **学習**: CTC 損失, 切り詰め BPTT 窓 \(bpttWindow) フレーム (\(bpttWindow * 40)ms 相当)
- **語彙・かな漢字辞書**: 学習セット \(trainTextLines.count) 件のみから構築 (未学習セットの正解テキストは不使用)
- **サンプリング**: 48kHz $\to$ 16kHz リサンプリング (アンチエイリアス 3:1 間引き)
- **デコーダ**: CTC Prefix Beam Search, 第2段 言語 SNN 加点 = \(Defaults.languageBonus)
- **評価指標**:
  - **CER (Character Error Rate)**: $\\text{Levenshtein}(target, pred) / \\text{len}(target)$
  - **Exact Match 率**: 完全一致割合 (%)

---

## 2. 群別集計結果

| 群 | 件数 | 完全一致数 (件) | Exact 率 (%) | 漢字 CER (%) | かな CER (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **学習セット (Train)** | \(trainSummary.count) | \(trainSummary.exactCount) | \(String(format: "%.1f", trainSummary.exactRate))% | \(String(format: "%.2f", trainSummary.meanCer))% | \(String(format: "%.2f", trainSummary.meanKanaCer))% |
| **未学習セット (Unseen)** | \(unseenSummary.count) | \(unseenSummary.exactCount) | \(String(format: "%.1f", unseenSummary.exactRate))% | \(String(format: "%.2f", unseenSummary.meanCer))% | \(String(format: "%.2f", unseenSummary.meanKanaCer))% |

---

## 3. 未学習セットの出力例

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
