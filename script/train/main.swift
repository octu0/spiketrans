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
// 変更する場合はここを直接書き換える (CLI 引数にはしない)。
enum Defaults {
    static let frameStack = StreamingFeatureFrontEnd.defaultStack
    static let melFrameDim = StreamingFeatureFrontEnd.tapDim
    static var acousticInputDim: Int { return melFrameDim * frameStack }

    /// 隠れ層の次元
    static let maxHiddenDim = 1024
    /// 音響 SNN の層数。層 0 が再帰層、層 1 以降は電流残差付きフィードフォワード層。
    /// JSUT 2000 × 20 epoch: 2 層 22.6% → 3 層 19.9% → 4 層 19.7% (未学習かな CER)。
    /// 混合 1.5 万件 × 3 epoch: 2 層 49.6% → 3 層 40.0%。幅 1536 も同程度の改善だが推論コストは
    /// 3 層 +13% に対し +44%。4 層・3 層+1536 は伸びず学習時間だけ増える
    static let numLayers = 3

    /// 切り詰め BPTT の窓幅 (フレーム単位)。
    /// 1 だとフレーム間の信用割り当てが消え、16 では発散した。
    static let bpttWindow = 4

    /// Cosine 学習率スケジュール
    static let lrMax: Float = 0.003
    static let lrMin: Float = 0.0005
    /// コサイン減衰に使うステップ数の上限。epoch 数を増やしても高い学習率の持続時間が
    /// 伸びないようにする。38 万件 (1 epoch 約 6,000 ステップ) で lrMax 付近を 4 epoch 超
    /// 維持すると損失が上がり始めたので、減衰を約 8 epoch で lrMin に到達させる
    static let lrDecaySteps = 50_000
    /// epoch 平均損失がこの回数続けて上がったら、最良の重みへ巻き戻して学習率を下げる
    static let lossRiseEpochs = 2
    /// 巻き戻しの学習率倍率と、巻き戻しの最大回数 (超えたら学習を打ち切る)
    static let lrRollbackFactor: Float = 0.5
    static let maxLRRollbacks = 2

    /// LIF 設定。ALIF (gamma > 0) は損失・CER とも悪化したため無効。
    static let lifConfig = LIFConfig(beta: 0.92, vTh: 1.0, vReset: 0.0, alpha: 2.0, rho: 0.85, gamma: 0.0)

    /// N エポックごとの重み書き出し
    static let checkpointEvery = 10

    /// 学習率の暖機に使うステップ数の分母 (全ステップの 1/N を暖機に充てる)
    static let warmupStepDivisor = 50
    /// 暖機ステップ数の下限
    static let minWarmupSteps = 100

    /// 系列長バケットをまとめる窓 (バッチ数)。
    /// compile 済みステップは形が変わるたびに GPU バッファを確保し直すため、バケットが毎バッチ入れ替わると
    /// 同じ形が続く場合の約 2 倍かかる (256 フレーム: 0.8 → 1.6 秒)。並びは混ぜたまま、この窓の中だけ
    /// 同じバケットが連続するように並べ替える。窓が 1 epoch の 1% 程度なら学習の順序はほぼ無作為のまま
    static let bucketRunWindow = 64

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
var batchSize = Defaults.batchSize
var datasetPath = ""
var englishDictPath = ""
var deviceArg = "auto"
var exportWeightsPath: String? = nil
var importWeightsPath: String? = nil
let reportPath = "/dev/stdout"

var argIdx = 1
let args = CommandLine.arguments
while argIdx < args.count {
    let arg = args[argIdx]
    switch arg {
    case "-h", "--help":
        print("使い方: train -d <マニフェスト.jsonl> [オプション]")
        print("オプション:")
        print("  -d, --dir, --dataset <パス>        学習マニフェスト (JSONL) のパス [必須]")
        print("  -b, --batch-size <Int>             ミニバッチサイズ (既定: 64)")
        print("  -e, --epochs <Int>                 エポック数 (既定: 20)")
        print("  -s, --samples <Int>                最大学習サンプル数 (制限なし)")
        print("  -p, --parallel <Int>               並列ワーカー数 (既定: P コア数)")
        print("  --device <auto|gpu|cpu>            実行デバイス (既定: auto)")
        print("  --export-weights <パス>            学習済み重みの出力先 JSON パス")
        print("  --import-weights <パス>            初期重みのインポート元 JSON パス")
        exit(0)
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
    case "-b", "--batch-size":
        if (argIdx + 1) < args.count {
            if let val = Int(args[argIdx + 1]) {
                batchSize = max(1, val)
            }
            argIdx += 1
        }
    case "-d", "--dir", "--dataset":
        if (argIdx + 1) < args.count {
            datasetPath = args[argIdx + 1]
            argIdx += 1
        }
    case "--english-dict":
        if (argIdx + 1) < args.count {
            englishDictPath = args[argIdx + 1]
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
    default:
        if arg.hasPrefix("-") != true {
            datasetPath = arg
        }
    }
    argIdx += 1
}

// データセットのパスは必須。特定コーパスを既定値に埋め込まない
if englishDictPath.isEmpty {
    print("エラー: 英語の発音辞書 (CMU 形式) を --english-dict で指定してください。")
    exit(1)
}
guard let englishDict = try? EnglishPronunciations(contentsOfFile: englishDictPath) else {
    print("エラー: 発音辞書 \(englishDictPath) が読み込めません。")
    exit(1)
}
print("英語の発音辞書: \(englishDictPath) (\(englishDict.count) 語)")

if datasetPath.isEmpty {
    print("エラー: 学習マニフェスト (JSONL) を指定してください。")
    print("  使い方: train -d <マニフェスト.jsonl> --english-dict <cmudict> [-s 件数] [-e エポック数] [-b バッチサイズ]")
    print("  詳細は train --help を参照してください。")
    print("  各行: {\"path\": \"/path/to/voice.wav\", \"text\": \"漢字かな混じりの発話テキスト\"}")
    print("  マニフェストは script/dataset/ の各コーパス用スクリプトで生成する")
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

let deviceDescription: String
if useGPU {
    deviceDescription = "Apple Silicon GPU (MLX Swift)"
} else {
    deviceDescription = "CPU (Pure Swift)"
}

print("データセットパス: \(datasetPath)")
print("実行デバイス   : \(deviceDescription)")
print("ミニバッチサイズ (-b): \(batchSize)")
print("並列ワーカー数 (-p): \(numWorkers) スレッド")
print("エポック数     (-e): \(epochs) エポック")
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

guard let manifestContent = try? String(contentsOfFile: datasetPath, encoding: .utf8) else {
    print("エラー: マニフェスト \(datasetPath) が読み込めません。")
    exit(1)
}

let manifestDir = (datasetPath as NSString).deletingLastPathComponent
let jsonDecoder = JSONDecoder()
var textLines: [String] = []
var rawPairs: [(path: String, fileId: String, text: String)] = []

// JSONL は改行 (LF) 区切り。CharacterSet.newlines で分けると U+2028 等の
// 行区切り文字でも切れてしまい、字幕由来のテキストを含む行が壊れる
for line in manifestContent.components(separatedBy: "\n") {
    let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
    if trimmed.isEmpty != true {
        guard let entry = try? jsonDecoder.decode(ManifestEntry.self, from: Data(trimmed.utf8)) else {
            print("エラー: マニフェストの行を解釈できません: \(trimmed.prefix(120))")
            exit(1)
        }
        // 相対パスはマニフェストのあるディレクトリ基準で解決する
        var wavPath = entry.path
        if wavPath.hasPrefix("/") != true {
            wavPath = (manifestDir as NSString).appendingPathComponent(wavPath)
        }
        let fileId = ((wavPath as NSString).lastPathComponent as NSString).deletingPathExtension
        textLines.append(entry.text)
        rawPairs.append((path: wavPath, fileId: fileId, text: entry.text))
    }
}

// 語彙・かな漢字辞書は学習セット (先頭 sampleLimit 件) のみから構築する。
// 未学習セットの正解テキストを混ぜると第2段が答えを含む辞書を引いてしまい、
// 未学習 CER が実力より良く出てしまうため。
let sampleLimit = maxTrainSamples ?? rawPairs.count
let trainTextLines = Array(textLines.prefix(sampleLimit))

let kanjiConverter = KanjiConverter(english: englishDict)
let trainHiraganaLines = trainTextLines.map { kanjiConverter.convertToHiragana($0) }

// 重みを持ち込む場合は同梱の語彙を使う。学習時と別のマニフェストでも
// 出力層の ID 割当が変わらず、評価や追加学習ができる
var importedWeights: SpikingNetworkWeights? = nil
if let impPath = importWeightsPath {
    print("\n[重み読込] 外部ファイルからモデル重みをインポート中: \(impPath)")
    guard let wData = try? SpikingNetworkWeights.load(from: URL(fileURLWithPath: impPath)) else {
        print("  ✕ 重みファイルの読み込みに失敗しました。")
        exit(1)
    }
    importedWeights = wData
}
let phoneticVocabulary: TextVocabulary
switch importedWeights?.vocabulary {
case .some(let embedded):
    phoneticVocabulary = embedded
    print("  同梱語彙を使用: \(embedded.size) 文字")
case .none:
    phoneticVocabulary = TextVocabulary(corpus: trainHiraganaLines)
}
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
let dataset = SpeechDataset.lazyFromManifest(
    pairs: manifestPairs,
    textVocabulary: textVocabulary,
    frameStack: Defaults.frameStack,
    workers: numWorkers,
    english: englishDict
)

let loadElapsed = CFAbsoluteTimeGetCurrent() - startTime
print("データセット構築完了: \(dataset.count) サンプル (所要時間: \(String(format: "%.3f", loadElapsed)) 秒)")
for i in 0..<min(3, dataset.count) {
    let sample = dataset.sample(at: i, loadPCM: true)
    let featDim = sample.acousticFeatures.first?.count ?? 128
    let numSamples = sample.audioPCM.count
    let durSec = Float(numSamples) / 16000.0
    print("  [\(i+1)] 正解テキスト: \"\(sample.rawText)\"")
    print("      発音(かな): \"\(sample.hiraganaText)\" (\(sample.hiraganaText.count) 文字)")
    print("      PCM サンプル数 (16kHz リサンプル後): \(numSamples) サンプル (\(String(format: "%.2f", durSec)) 秒)")
    print("      音響特徴量フレーム数: \(sample.acousticFeatures.count) フレーム (\(featDim)次元 3-tap Mel 特徴量)")
}

// 4. 第1段 音響 SNN (かな・音素) の学習実行
let trainDeviceLabel: String
let trainLR: Float
if useGPU {
    trainDeviceLabel = "GPU"
    trainLR = 0.003
} else {
    trainDeviceLabel = "CPU"
    trainLR = 0.015
}
print("\n--- 2. 第1段 音響 SNN (かな・音素) の学習実行 (デバイス: \(trainDeviceLabel)) ---")
let trainConfig = TrainingConfig(
    epochs: epochs,
    learningRate: trainLR,
    logInterval: 2,
    clipNorm: 5.0
)

let acousticInputDim = Defaults.acousticInputDim
print("音響特徴量: \(acousticInputDim) 次元 (\(Defaults.melFrameDim) 次元 3-tap Mel × \(Defaults.frameStack) フレーム束ね)")

print("第1段 LIF: beta = \(Defaults.lifConfig.beta), 層数 = \(Defaults.numLayers)")

let trainer = Trainer(
    acousticNetwork: SpikingNetwork(
        numLayers: Defaults.numLayers,
        inputDim: acousticInputDim,
        maxHiddenDim: Defaults.maxHiddenDim,
        outputDim: phoneticVocabulary.size,
        timeSteps: 4,
        lifConfig: Defaults.lifConfig
    ),
    languageNetwork: SpikingNetwork(
        inputDim: 128,
        maxHiddenDim: Defaults.maxHiddenDim,
        outputDim: textVocabulary.size,
        timeSteps: 4
    ),
    textVocabulary: phoneticVocabulary,
    phonemeVocabulary: PhonemeVocabulary(),
    config: trainConfig
)

// 重みのインポート。-e 0 なら評価のみ、-e N なら読み込んだ重みから追加学習する
if let wData = importedWeights {
    guard wData.outputDim == phoneticVocabulary.size,
          wData.inputDim == acousticInputDim,
          wData.numLayers == Defaults.numLayers else {
        print("  ✕ 重みの次元が現在の構成と一致しません (入力 \(wData.inputDim)/\(acousticInputDim), 出力 \(wData.outputDim)/\(phoneticVocabulary.size), 層数 \(wData.numLayers)/\(Defaults.numLayers))。")
        exit(1)
    }
    trainer.acousticTrainer.network.importWeights(from: wData)
    if epochs == 0 {
        print("  ✓ 音響モデル重みのインポート完了。評価のみ実行します (-e 0)。")
    } else {
        print("  ✓ 音響モデル重みのインポート完了。この重みから \(epochs) エポックの追加学習を行います。")
    }
}

if epochs == 0 {
    // 追加学習なし。インポート済み重みで評価へ進む
} else {
    if useGPU {
        print("  Apple Silicon GPU (MLX Swift Metal) による並列ミニバッチ学習を開始 (バッチサイズ: \(batchSize))...")

        // 教師はフレームに整列していないかな ID 列。アライメントは CTC が周辺化する。
        // 発話フレームを文字数で等分する近似アライメント + 交差エントロピーでは、
        // 教師ラベル自体が誤っているため学習セットすら再現できなかった。
        print("  [教師] CTC 損失: フレーム整列なしのかな ID 列")
        let allTargets: [[Int]] = (0..<dataset.count).map { idx in
            return phoneticVocabulary.textToIds(dataset.hiraganaText(at: idx))
        }

        let mlxNet = MLXSpikingNetwork(
            numLayers: Defaults.numLayers,
            inputDim: acousticInputDim,
            maxHiddenDim: Defaults.maxHiddenDim,
            outputDim: phoneticVocabulary.size,
            timeSteps: 4,
            lifConfig: trainer.acousticTrainer.network.lifConfig
        )
        if let wData = importedWeights {
            mlxNet.importWeights(from: wData)
            print("  [追加学習] インポート済み重みを GPU 学習の初期値に設定")
        }
        let mlxTrainer = MLXBPTTTrainer(
            network: mlxNet,
            config: trainConfig,
            bpttWindow: Defaults.bpttWindow
        )
        print("  切り詰め BPTT 窓幅: \(Defaults.bpttWindow) フレーム")
        let scheduler = CosineLRScheduler(lrMax: Defaults.lrMax, lrMin: Defaults.lrMin, totalEpochs: epochs, warmupEpochs: 1)
        print("  学習率: \(Defaults.lrMax) → \(Defaults.lrMin)")
        let trainStartTime = CFAbsoluteTimeGetCurrent()

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
        if ctcMinimumFrames(allTargets[di]) <= frames {
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

    // バッチの構成 (どのサンプルをどのバッチに入れるか) は全エポック共通。
    // 長い系列の件数を減らして総フレーム数を揃える案は、eager の所要時間が件数ではなく
    // フレーム数 (演算ノード数) で決まるため逆に遅くなった (15k × 2 epoch で 303 → 449 秒)
    var batchGroups: [[Int]] = []
    var gStart = 0
    while gStart < lengthSortedIndices.count {
        let gEnd = min(gStart + batchSize, lengthSortedIndices.count)
        batchGroups.append(Array(lengthSortedIndices[gStart..<gEnd]))
        gStart = gEnd
    }

    // 系列長バケット (32 フレーム単位に切り上げた最長フレーム数) を、window バッチの窓の中で昇順にまとめる
    func paddedFrameCount(of group: [Int]) -> Int {
        var maxFrames = 0
        for idx in group {
            maxFrames = max(maxFrames, dataset.frameCount(at: idx))
        }
        return ((maxFrames + 31) / 32) * 32
    }
    func groupBucketsWithinWindows(_ groups: inout [[Int]], window: Int) {
        var start = 0
        while start < groups.count {
            let end = min(start + window, groups.count)
            let keyed = groups[start..<end].map { group in (key: paddedFrameCount(of: group), group: group) }
            let sortedWindow = keyed.sorted { a, b in a.key < b.key }.map { $0.group }
            groups.replaceSubrange(start..<end, with: sortedWindow)
            start = end
        }
    }

    // バッチの特徴量を WAV から並列生成する (遅延読み込みの実体)
    final class FeatureBatchBuffer: @unchecked Sendable {
        var items: [[[Float]]]
        init(count: Int) {
            self.items = [[[Float]]](repeating: [], count: count)
        }
    }
    let activeWorkers = numWorkers
    @Sendable func buildBatchFeatures(_ indices: [Int]) -> [[[Float]]] {
        let buffer = FeatureBatchBuffer(count: indices.count)
        let workerCount = max(1, min(activeWorkers, indices.count))
        DispatchQueue.concurrentPerform(iterations: workerCount) { worker in
            var i = worker
            while i < indices.count {
                let meta = dataset.metaSamples[indices[i]]
                buffer.items[i] = SpeechDataset.loadFeatures(
                    path: meta.path,
                    frameStack: Defaults.frameStack,
                    loadPCM: false
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
    // 総ステップの 1/50 だが下限を設ける。2000 件 × 20 epoch (640 ステップ) では暖機 12 ステップとなり
    // lr 0.003 で暖機中に発散した。100 万件規模 (暖機 1000 ステップ超) には影響しない
    let warmupSteps = min(max(Defaults.minWarmupSteps, totalSteps / Defaults.warmupStepDivisor), max(1, totalSteps / 2))
    // 減衰はステップ数で決める。総ステップが上限を超える長い学習では、上限に達した後は lrMin で続ける
    let decaySteps = min(totalSteps - warmupSteps, Defaults.lrDecaySteps)
    let scheduleSteps = warmupSteps + decaySteps
    print("  学習ステップ数: \(totalSteps) (暖機 \(warmupSteps) ステップ、減衰 \(decaySteps) ステップ、以後 lrMin)")

    let checkpointInterval = min(Defaults.checkpointEvery, max(1, epochs / 6))
    print("  チェックポイント間隔: \(checkpointInterval) エポックごと")

    // 損失の上昇監視。epoch 平均損失が続けて上がるのは学習率が大きすぎる合図で、
    // そのまま続けても戻らない (28 epoch で ep4 の 64.2 から ep13 の 69.6 まで上がり、
    // 最後まで ep4 に戻らなかった)。最良の重みへ巻き戻して学習率を半分にする
    var bestLoss = Float.greatestFiniteMagnitude
    var bestEpoch = 0
    var bestWeights: SpikingNetworkWeights? = nil
    var prevLoss = Float.greatestFiniteMagnitude
    var riseStreak = 0
    var lrScale: Float = 1.0
    var rollbacks = 0

    var globalStep = 0
    var ep = 1
    while ep <= epochs {
        let epStartTime = CFAbsoluteTimeGetCurrent()
        // バッチの並びは毎エポック混ぜる。長さ順のまま流すと 1 エポックの前半は
        // 短い断片ばかり、後半は長い発話ばかりになり、100 万件規模では最初の
        // 数千ステップが 1 秒未満の断片だけで埋まって blank 一色に崩れる
        batchGroups.shuffle()
        groupBucketsWithinWindows(&batchGroups, window: Defaults.bucketRunWindow)
        var curLR = scheduler.learningRate(
            step: globalStep + 1, totalSteps: scheduleSteps, warmupSteps: warmupSteps) * lrScale

        var epLossSum: Float = 0.0
        var batchCount = 0
        // 時間の内訳: GPU の学習ステップ、先読み待ち (特徴量の読み込みが GPU より遅いとき)、
        // compile 済み / eager (系列長が compiledMaxFrames 超) の別、系列長バケットごとの時間
        var gpuSeconds = 0.0
        var waitSeconds = 0.0
        var eagerBatches = 0
        var eagerSeconds = 0.0
        var bucketSeconds: [Int: (count: Int, seconds: Double)] = [:]
        var bucketSwitches = 0
        var previousPaddedFrames = 0

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
                step: globalStep, totalSteps: scheduleSteps, warmupSteps: warmupSteps) * lrScale
            mlxTrainer.updateLearningRate(curLR)

            var maxFrames = 0
            for seq in currentFeatures {
                if maxFrames < seq.count {
                    maxFrames = seq.count
                }
            }
            let paddedFrames = ((maxFrames + 31) / 32) * 32
            let gpuStart = CFAbsoluteTimeGetCurrent()
            let res = mlxTrainer.trainBatchCTC(
                featuresBatch: currentFeatures,
                targetsBatch: tBatch,
                blankId: TextVocabulary.padId
            )
            let gpuElapsed = CFAbsoluteTimeGetCurrent() - gpuStart
            gpuSeconds += gpuElapsed
            if compiledMaxFrames < paddedFrames {
                eagerBatches += 1
                eagerSeconds += gpuElapsed
            }
            let prev = bucketSeconds[paddedFrames] ?? (count: 0, seconds: 0.0)
            bucketSeconds[paddedFrames] = (count: prev.count + 1, seconds: prev.seconds + gpuElapsed)
            if paddedFrames != previousPaddedFrames {
                bucketSwitches += 1
                previousPaddedFrames = paddedFrames
            }
            epLossSum += res
            batchCount += 1

            let waitStart = CFAbsoluteTimeGetCurrent()
            prefetchGroup.wait()
            waitSeconds += CFAbsoluteTimeGetCurrent() - waitStart
            currentFeatures = prefetchBox.value
            bIdx += 1
        }

        let avgLoss = epLossSum / Float(max(1, batchCount))
        let epElapsed = CFAbsoluteTimeGetCurrent() - epStartTime
        print("  Epoch [\(ep)/\(epochs)] - 音響損失: \(String(format: "%.4f", avgLoss)) (LR: \(String(format: "%.5f", curLR)), 所要時間: \(String(format: "%.2f", epElapsed)) 秒)")
        print("    内訳: GPU \(String(format: "%.0f", gpuSeconds)) 秒 (うち eager \(eagerBatches) バッチ \(String(format: "%.0f", eagerSeconds)) 秒) / 先読み待ち \(String(format: "%.0f", waitSeconds)) 秒 / \(batchCount) バッチ / バケット切替 \(bucketSwitches) 回")
        print("    MLX メモリ: 使用中 \(MLX.Memory.activeMemory >> 20) MB / キャッシュ \(MLX.Memory.cacheMemory >> 20) MB / ピーク \(MLX.GPU.peakMemory >> 20) MB")
        let topBuckets = bucketSeconds.sorted { a, b in b.value.seconds < a.value.seconds }.prefix(6)
        var bucketLine = "    系列長バケット (フレーム: バッチ数 / 秒):"
        for (frames, stat) in topBuckets {
            bucketLine += " \(frames): \(stat.count) / \(String(format: "%.0f", stat.seconds))"
        }
        print(bucketLine)

        if avgLoss < bestLoss {
            bestLoss = avgLoss
            bestEpoch = ep
            bestWeights = mlxNet.exportWeights(vocabulary: phoneticVocabulary)
        }
        if prevLoss < avgLoss {
            riseStreak += 1
        } else {
            riseStreak = 0
        }
        prevLoss = avgLoss

        if Defaults.lossRiseEpochs <= riseStreak {
            if let best = bestWeights, rollbacks < Defaults.maxLRRollbacks {
                rollbacks += 1
                lrScale *= Defaults.lrRollbackFactor
                riseStreak = 0
                prevLoss = bestLoss
                mlxTrainer.rollback(to: best)
                print("    ↩ 損失が \(Defaults.lossRiseEpochs) epoch 続けて上昇。epoch \(bestEpoch) の重み (損失 \(String(format: "%.4f", bestLoss))) へ巻き戻し、学習率を \(String(format: "%.2f", lrScale)) 倍にする (\(rollbacks)/\(Defaults.maxLRRollbacks) 回目)")
            } else {
                print("    ■ 損失が \(Defaults.lossRiseEpochs) epoch 続けて上昇し、巻き戻しの回数も使い切ったので学習を打ち切る (最良は epoch \(bestEpoch))")
                break
            }
        }

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

    // 最後の epoch が最良でなければ、最良の重みで終える
    if let best = bestWeights, bestLoss < prevLoss {
        mlxNet.importWeights(from: best)
        print("  最終 epoch より epoch \(bestEpoch) (損失 \(String(format: "%.4f", bestLoss))) の方が良いので、その重みを採用する")
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
    } else {
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
}

// languageBonus == 0 のときは言語 SNN を第2段で使わないため学習を省略する
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
    print("\n--- 2.5 第2段 漢字自己回帰言語 SNN の学習をスキップ (languageBonus = 0) ---")
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
    let s0 = dataset.sample(at: 0, loadPCM: true)
    let feat0 = s0.acousticFeatures
    let totalF = feat0.count
    let hiraIds0 = phoneticVocabulary.textToIds(s0.hiraganaText)

    print("\n==================================================")
    print("=== [一次診断] 先頭発話のアライメント・発火状況 ===")
    print("==================================================")
    print("正解テキスト: \"\(s0.rawText)\"")
    print("正解かな発音: \"\(s0.hiraganaText)\" (\(hiraIds0.count) 文字), 音響フレーム数: \(totalF) フレーム")

    // 1. alignTargets の集計 (Mel エネルギーによる発話フレーム配分)
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
                filterbank.extractFeatures(pcmPtr: framePtr, count: 400, workspace: ws)
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
        if 0.0 < rawOutOfBandTotal {
            attenRatio = eqOutOfBandTotal / rawOutOfBandTotal
        } else {
            attenRatio = 0.20
        }
        let attenDb: Float
        if 0.0 < attenRatio {
            attenDb = -10.0 * log10(attenRatio)
        } else {
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
        network: trainer.acousticTrainer.network
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

    let padRatio = Float(predPad) * 100.0 / Float(max(1, totalF))
    print("\n[2] 音響 SNN 全フレーム予測集計:")
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
        let testSample = dataset.sample(at: idx, loadPCM: true)
        print("\n  ==================================================")
        print("  [\(idx+1)] 正解テキスト(漢字): \"\(testSample.rawText)\"")
        print("      正解かな発音:     \"\(testSample.hiraganaText)\"")
        print("  ==================================================")

        let t0 = CFAbsoluteTimeGetCurrent()
        let resTwoStage = trainer.transcribeTwoStage(
            featuresSeq: testSample.acousticFeatures,
            kanjiVocabulary: textVocabulary,
            dictionary: kanaKanjiDict,
            minDurationFrames: 3,
            minConfidence: 0.05,
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
let evalKanjiConverter = KanjiConverter(english: englishDict)
let evalFrameStack = Defaults.frameStack

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
        guard let wavData = SpeechDataset.loadWavFile(path: wavPath) else {
            return
        }

        let pcm16k = SpeechDataset.resampleTo16k(pcmData: wavData.pcmData, sampleRate: wavData.sampleRate)
        let features = SpeechDataset.extractFeaturesFromPCM(pcmData: pcm16k, frameStack: evalFrameStack)
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
        if let wavData = SpeechDataset.loadWavFile(path: wavPath) {
            let pcm16k = SpeechDataset.resampleTo16k(pcmData: wavData.pcmData, sampleRate: wavData.sampleRate)
            benchAudioSeconds += Double(pcm16k.count) / 16000.0
            benchFeatures.append(SpeechDataset.extractFeaturesFromPCM(pcmData: pcm16k, frameStack: Defaults.frameStack))
        }
        benchIdx += 1
    }

    print("\n==================================================")
    print("=== [推論レイテンシ] 単一スレッド, \(benchFeatures.count) 発話 (音声 \(String(format: "%.1f", benchAudioSeconds)) 秒) ===")
    print("==================================================")

    let benchNetwork = trainer.acousticTrainer.network
    do {
        let decoder = AcousticDecoder(
            network: benchNetwork
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

// 未学習セットが小さいときは全件を出す。配信の評価セットで、区間ごとの音の条件と CER を突き合わせるため
if unseenResults.count <= 200 {
    print("\n--- [未学習セット 全件] fileId かなCER 漢字CER ---")
    for r in unseenResults {
        print("  \(r.fileId)\t\(String(format: "%.1f", r.kanaCer * 100.0))\t\(String(format: "%.1f", r.cer * 100.0))")
    }
}



// レポートファイルの生成
var reportContent = """
# \(datasetPath) 正誤率・CER 評価レポート

## 1. 概要
- **評価対象**: `\(datasetPath)` \(evalPairs.count) 発話 (学習セットは間引き / 未学習セットは全件)
- **モデル**: `-s \(sampleLimit) -e \(epochs)` で学習した直接漢字音響 SNN (学習セット \(sampleLimit) 発話)
- **第1段 LIF**: beta = \(Defaults.lifConfig.beta), rho = \(Defaults.lifConfig.rho), gamma = \(Defaults.lifConfig.gamma)
- **特徴量**: \(Defaults.melFrameDim) 次元 3-tap Mel × \(Defaults.frameStack) フレーム束ね = \(Defaults.acousticInputDim) 次元
- **学習**: CTC 損失, 切り詰め BPTT 窓 \(Defaults.bpttWindow), 学習率 \(Defaults.lrMax) → \(Defaults.lrMin)
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
