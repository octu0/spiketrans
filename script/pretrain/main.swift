import Foundation
import MLX
import Spiketrans

// 自己教師ありの事前学習。ラベル無しの音声だけを使う。出力した重みを train の --import-weights に渡すと、
// 隠れ層をそのまま使い、読み出しを作り直して CTC の学習を最初の学習率から始める。
//   APC (既定): 最終層の読み出しから Config.predictShifts フレーム先の入力特徴量を予測 (L1)
//     pretrain -d <マニフェスト.jsonl (path だけ使う)> -e <epoch> --export-weights <json> [-b 64] [-p 8] [-s 件数]
//   HuBERT 型: 先に k-means で特徴量のクラス中心を作り、入力の区間を隠して、隠したフレームのクラス番号を当てる (交差エントロピー)
//     pretrain -d <マニフェスト> --kmeans <中心.json>                 (中心を作って終わる)
//     pretrain -d <マニフェスト> -e <epoch> --targets <中心.json> --export-weights <json>
//   HuBERT 型の 2 周目: 事前学習済みの先生 (--teacher) の層 Config.teacherLayer の発火率を k-means して目標にし、新しいモデルを学習
//     pretrain -d <マニフェスト> --kmeans <中心.json> --teacher <重み.json>
//     pretrain -d <マニフェスト> -e <epoch> --targets <中心.json> --teacher <重み.json> --export-weights <json>
//
// ネットワークの構成は train の既定 (8 層・幅 1024・512 次元入力・同じ LIF) と揃える

setvbuf(stdout, nil, _IOLBF, 0)

enum Config {
    static let numLayers = 8
    static let maxHiddenDim = 1024
    static let timeSteps = 4
    static let bpttWindow = 4
    static let lifConfig = LIFConfig(beta: 0.92, vTh: 1.0, vReset: 0.0, alpha: 2.0, rho: 0.85, gamma: 0.0)
    static let lrMax: Float = 0.003
    static let lrMin: Float = 0.0005
    /// 1 発話から切り出す最大フレーム数 (compile 済みの経路に乗る上限 = 5.12 秒)
    static let cropFrames = 128
    /// これより短い発話は使わない (予測先を引くと損失を取れるフレームがほとんど残らない)
    static let minFrames = 16
    static let logEvery = 200
    /// 予測先 (フレーム数、40 ms 単位)。複数あれば同時に予測して損失を平均する
    static let predictShifts = [SpikingNetworkWeights.predictShift]
    /// HuBERT 型: クラス数、k-means に使うフレーム数と反復回数
    static let clusterCount = 256
    /// HuBERT 型の 2 周目: 先生の目標にする層 (8 層中 6 番目) と、そのときのクラス数 (本家 HuBERT の 2 周目と同じ 500)
    static let teacherLayer = 5
    static let teacherClusterCount = 500
    static let teacherKmeansFrames = 100_000
    static let kmeansFrames = 200_000
    static let kmeansIterations = 25
    /// HuBERT 型: 隠す区間。各フレームを確率 maskStartProb で開始点にし、maskSpan フレーム (200 ms) 隠す (全体の約 3 割)
    static let maskStartProb = 0.065
    static let maskSpan = 5
}

var manifestPath = ""
var exportPath = ""
var epochs = 1
var batchSize = 64
var workers = 8
var maxItems = Int.max
var kmeansOutPath = ""
var targetsPath = ""
var teacherPath = ""
var argIdx = 1
let args = CommandLine.arguments
while argIdx < args.count {
    let arg = args[argIdx]
    var value = ""
    if (argIdx + 1) < args.count {
        value = args[argIdx + 1]
    }
    switch arg {
    case "-d":
        manifestPath = value
        argIdx += 1
    case "-e":
        epochs = max(1, Int(value) ?? epochs)
        argIdx += 1
    case "-b":
        batchSize = max(1, Int(value) ?? batchSize)
        argIdx += 1
    case "-p":
        workers = max(1, Int(value) ?? workers)
        argIdx += 1
    case "-s":
        maxItems = max(1, Int(value) ?? maxItems)
        argIdx += 1
    case "--export-weights":
        exportPath = value
        argIdx += 1
    case "--kmeans":
        kmeansOutPath = value
        argIdx += 1
    case "--targets":
        targetsPath = value
        argIdx += 1
    case "--teacher":
        teacherPath = value
        argIdx += 1
    default:
        break
    }
    argIdx += 1
}
if manifestPath.isEmpty || (exportPath.isEmpty && kmeansOutPath.isEmpty) {
    print("使い方: pretrain -d <マニフェスト.jsonl> -e <epoch> --export-weights <重み.json> [--targets <中心.json>] [-b 64] [-p 8] [-s 件数]")
    print("        pretrain -d <マニフェスト.jsonl> --kmeans <中心.json>")
    exit(1)
}

// マニフェスト (path だけ読む)。長さは WAV のバイト数から見積もり (16 kHz・16 bit・mono で 1 秒 32,000 バイト)、
// 切り出し後のフレーム数 (32 単位) でバケットに分けて、バッチ内の 0 埋めを減らす
struct Row: Decodable {
    let path: String
}
guard let content = try? String(contentsOfFile: manifestPath, encoding: .utf8) else {
    print("エラー: マニフェストが読めません: \(manifestPath)")
    exit(1)
}
var paths: [String] = []
var buckets: [Int] = []
for line in content.components(separatedBy: "\n") {
    let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
    if trimmed.isEmpty {
        continue
    }
    guard let row = try? JSONDecoder().decode(Row.self, from: Data(trimmed.utf8)) else {
        continue
    }
    guard let attrs = try? FileManager.default.attributesOfItem(atPath: row.path),
          let size = attrs[.size] as? NSNumber else {
        continue
    }
    let frames = Int(Double(size.intValue) / 32000.0 * 25.0)
    if frames < Config.minFrames {
        continue
    }
    paths.append(row.path)
    buckets.append(((min(frames, Config.cropFrames) + 31) / 32) * 32)
    if maxItems <= paths.count {
        break
    }
}
if paths.isEmpty {
    print("エラー: 使える音声がありません")
    exit(1)
}
let itemPaths = paths
let itemBuckets = buckets
let batchItems = batchSize
let loadWorkers = workers
var totalSeconds = 0.0
for b in buckets {
    totalSeconds += Double(b) * 0.04
}
// 2 周目の先生 (重みは固定、目標を作るだけ)
var teacherTrainer: MLXBPTTTrainer? = nil
if teacherPath.isEmpty != true {
    guard let tw = try? SpikingNetworkWeights.load(from: URL(fileURLWithPath: teacherPath)) else {
        print("エラー: 先生の重みが読めません: \(teacherPath)")
        exit(1)
    }
    teacherTrainer = MLXBPTTTrainer(network: MLXSpikingNetwork(weights: tw), bpttWindow: Config.bpttWindow)
    print("先生: \(teacherPath) の層 \(Config.teacherLayer) の発火率を目標にする")
}

// k-means: 無作為に選んだ発話から最大 kmeansFrames フレームを集めて中心を作り、保存して終わる。
// 先生があれば、そのフレームの特徴量ではなく先生の層の発火率を集める
if kmeansOutPath.isEmpty != true {
    var frames: [[Float]] = []
    var order = Array(0..<paths.count).shuffled()
    let perFile = 50
    var target = Config.kmeansFrames
    var classes = Config.clusterCount
    if teacherTrainer != nil {
        target = Config.teacherKmeansFrames
        classes = Config.teacherClusterCount
    }
    while frames.count < target && order.isEmpty != true {
        let idx = order.removeLast()
        var feats = SpeechDataset.loadFeatures(path: paths[idx], frameStack: StreamingFeatureFrontEnd.defaultStack).features
        if feats.isEmpty {
            continue
        }
        if Config.cropFrames < feats.count {
            feats = Array(feats[0..<Config.cropFrames])
        }
        var rows = feats
        if let teacher = teacherTrainer {
            let x = MLXArray(feats.flatMap { $0 }, [1, feats.count, feats[0].count])
            let rates = teacher.layerRatesEager(network: teacher.network, features: x, layer: Config.teacherLayer)
            eval(rates)
            let h = rates.shape[2]
            let flat = rates.asArray(Float.self)
            rows = []
            var t = 0
            while t < feats.count {
                rows.append(Array(flat[(t * h)..<((t + 1) * h)]))
                t += 1
            }
        }
        var k = 0
        while k < min(perFile, rows.count) {
            frames.append(rows[Int.random(in: 0..<rows.count)])
            k += 1
        }
    }
    print("k-means: \(frames.count) フレーム (\(frames[0].count) 次元) → \(classes) クラス、\(Config.kmeansIterations) 回")
    let started = CFAbsoluteTimeGetCurrent()
    let book = ClusterCodebook.fit(frames: frames, k: classes, iterations: Config.kmeansIterations)
    let arrays = book.arrays()
    let ids = ClusterCodebook.assign(features: MLXArray(frames.prefix(50_000).flatMap { $0 }, [min(50_000, frames.count), frames[0].count]), codebook: arrays)
    eval(ids)
    var usage = [Int](repeating: 0, count: book.count)
    for i in ids.asArray(Int32.self) {
        usage[Int(i)] += 1
    }
    let used = usage.filter { 0 < $0 }.count
    print("  \(String(format: "%.0f", CFAbsoluteTimeGetCurrent() - started)) 秒、使われたクラス \(used) / \(book.count)、最大クラスの割合 \(String(format: "%.1f", Double(usage.max() ?? 0) * 100.0 / Double(max(1, ids.size))))%")
    do {
        try book.save(to: URL(fileURLWithPath: kmeansOutPath))
        print("  ✓ 中心を保存: \(kmeansOutPath)")
    } catch {
        print("  ✕ 中心の保存に失敗: \(error)")
        exit(1)
    }
    exit(0)
}

var codebookArrays: [MLXArray] = []
if targetsPath.isEmpty != true {
    guard let book = try? ClusterCodebook.load(from: URL(fileURLWithPath: targetsPath)) else {
        print("エラー: クラス中心が読めません: \(targetsPath)")
        exit(1)
    }
    codebookArrays = book.arrays()
    var round = ""
    if teacherTrainer != nil {
        round = " 2 周目"
    }
    print("事前学習 (HuBERT 型\(round)、\(book.count) クラス、\(Config.maskSpan) フレームの区間を開始確率 \(Config.maskStartProb) で隠し、隠したフレームのクラスを当てる)")
} else {
    print("事前学習 (APC、\(Config.predictShifts.map { String($0) }.joined(separator: "・")) フレーム先の特徴量を予測)")
}
print("  音声 \(paths.count) 件 (切り出し後 \(String(format: "%.1f", totalSeconds / 3600.0)) 時間/epoch) / epoch \(epochs) / バッチ \(batchSize)")

let inputDim = StreamingFeatureFrontEnd.acousticInputDim()
var classCount = 0
if codebookArrays.isEmpty != true {
    classCount = codebookArrays[2].shape[0]
}
let network = MLXSpikingNetwork(
    numLayers: Config.numLayers,
    inputDim: inputDim,
    maxHiddenDim: Config.maxHiddenDim,
    outputDim: 1,
    timeSteps: Config.timeSteps,
    lifConfig: Config.lifConfig,
    predictionHead: true,
    predictionTargets: Config.predictShifts.count,
    predictionClasses: classCount
)
let trainer = MLXBPTTTrainer(
    network: network,
    config: TrainingConfig(epochs: epochs, learningRate: Config.lrMax),
    bpttWindow: Config.bpttWindow
)
print("  ネットワーク: \(Config.numLayers) 層・幅 \(Config.maxHiddenDim)・入力 \(inputDim) 次元")

// バッチを組む: バケットごとに混ぜて batchSize ずつ切り、バッチの順も混ぜる
func makeBatches() -> [[Int]] {
    var byBucket: [Int: [Int]] = [:]
    var i = 0
    while i < itemPaths.count {
        byBucket[itemBuckets[i], default: []].append(i)
        i += 1
    }
    var batches: [[Int]] = []
    for (_, items) in byBucket {
        let shuffled = items.shuffled()
        var start = 0
        while start < shuffled.count {
            let end = min(start + batchItems, shuffled.count)
            batches.append(Array(shuffled[start..<end]))
            start = end
        }
    }
    return batches.shuffled()
}

final class BatchBuffer: @unchecked Sendable {
    var items: [[[Float]]]
    init(count: Int) {
        self.items = [[[Float]]](repeating: [], count: count)
    }
}

@Sendable func loadBatch(_ indices: [Int]) -> [[[Float]]] {
    let buffer = BatchBuffer(count: indices.count)
    let workerCount = max(1, min(loadWorkers, indices.count))
    DispatchQueue.concurrentPerform(iterations: workerCount) { worker in
        var rng = SystemRandomNumberGenerator()
        var k = worker
        while k < indices.count {
            buffer.items[k] = autoreleasepool {
                let feats = SpeechDataset.loadFeatures(path: itemPaths[indices[k]], frameStack: StreamingFeatureFrontEnd.defaultStack).features
                if feats.count <= Config.cropFrames {
                    return feats
                }
                let start = Int.random(in: 0...(feats.count - Config.cropFrames), using: &rng)
                return Array(feats[start..<(start + Config.cropFrames)])
            }
            k += workerCount
        }
    }
    return buffer.items.filter { Config.minFrames <= $0.count }
}

final class PrefetchBox: @unchecked Sendable {
    var value: [[[Float]]] = []
}

let batchesPerEpoch = makeBatches().count
let totalSteps = max(1, epochs * batchesPerEpoch)
let warmupSteps = max(100, totalSteps / 50)
let scheduler = CosineLRScheduler(lrMax: Config.lrMax, lrMin: Config.lrMin, totalEpochs: epochs, warmupEpochs: 1)
print("  学習ステップ数: \(totalSteps) (暖機 \(warmupSteps) ステップ、学習率 \(Config.lrMax) → \(Config.lrMin))")

var globalStep = 0
var ep = 1
let startAll = CFAbsoluteTimeGetCurrent()
while ep <= epochs {
    let batches = makeBatches()
    let epStart = CFAbsoluteTimeGetCurrent()
    var lossSum: Float = 0.0
    var windowSum: Float = 0.0
    var windowCount = 0
    var count = 0
    var waitSeconds = 0.0
    var current = loadBatch(batches[0])
    var b = 0
    while b < batches.count {
        let box = PrefetchBox()
        let group = DispatchGroup()
        if (b + 1) < batches.count {
            let next = batches[b + 1]
            group.enter()
            DispatchQueue.global(qos: .userInitiated).async {
                box.value = loadBatch(next)
                group.leave()
            }
        }
        globalStep += 1
        trainer.updateLearningRate(scheduler.learningRate(step: globalStep, totalSteps: totalSteps, warmupSteps: warmupSteps))
        var loss: Float = 0.0
        if codebookArrays.isEmpty {
            loss = trainer.trainBatchAPC(featuresBatch: current, shifts: Config.predictShifts)
        } else {
            loss = trainer.trainBatchMaskedCluster(featuresBatch: current, codebook: codebookArrays,
                                                   startProb: Config.maskStartProb, spanFrames: Config.maskSpan,
                                                   teacher: teacherTrainer, teacherLayer: Config.teacherLayer)
        }
        if loss.isFinite != true {
            print("  ✕ 損失が有限でなくなりました (step \(globalStep))。止めます")
            exit(1)
        }
        lossSum += loss
        windowSum += loss
        windowCount += 1
        count += 1
        if globalStep % Config.logEvery == 0 {
            let elapsed = CFAbsoluteTimeGetCurrent() - epStart
            print("    step \(globalStep): 予測損失 \(String(format: "%.4f", windowSum / Float(max(1, windowCount)))) / epoch 内 \(b + 1)/\(batches.count) / \(String(format: "%.0f", elapsed)) 秒 (先読み待ち \(String(format: "%.0f", waitSeconds)) 秒)")
            fflush(stdout)
            windowSum = 0.0
            windowCount = 0
        }
        let waitStart = CFAbsoluteTimeGetCurrent()
        group.wait()
        waitSeconds += CFAbsoluteTimeGetCurrent() - waitStart
        current = box.value
        b += 1
    }
    let epElapsed = CFAbsoluteTimeGetCurrent() - epStart
    print("  Epoch [\(ep)/\(epochs)] - 予測損失: \(String(format: "%.4f", lossSum / Float(max(1, count)))) (所要時間: \(String(format: "%.0f", epElapsed)) 秒、先読み待ち \(String(format: "%.0f", waitSeconds)) 秒)")
    fflush(stdout)
    var ckpt = exportPath
    if ep < epochs {
        ckpt = exportPath + ".ep\(ep).json"
    }
    do {
        try network.exportWeights().save(to: URL(fileURLWithPath: ckpt))
        print("    ✓ 重み保存: \(ckpt)")
    } catch {
        print("    ✕ 重み保存に失敗: \(error)")
    }
    ep += 1
}
print("事前学習完了 (\(String(format: "%.0f", CFAbsoluteTimeGetCurrent() - startAll)) 秒)")
