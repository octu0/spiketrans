import Foundation
import MLX
import Spiketrans

// 学習済み重みで発話を流し、層ごとの電流 RMS・発火率・膜電位の上限クリップ率を出す。
// 残差の尺度が層数に応じて増えていないかを見る診断
//   layerprobe --weights <json> -d <manifest.jsonl> [-s <件数>]

var weightsPath = ""
var manifestPath = ""
var maxItems = 50
var argIdx = 1
let args = CommandLine.arguments
while argIdx < args.count {
    switch args[argIdx] {
    case "--weights":
        if (argIdx + 1) < args.count {
            weightsPath = args[argIdx + 1]
            argIdx += 1
        }
    case "-d":
        if (argIdx + 1) < args.count {
            manifestPath = args[argIdx + 1]
            argIdx += 1
        }
    case "-s":
        if (argIdx + 1) < args.count {
            maxItems = max(1, Int(args[argIdx + 1]) ?? maxItems)
            argIdx += 1
        }
    default:
        break
    }
    argIdx += 1
}
if weightsPath.isEmpty || manifestPath.isEmpty {
    print("使い方: layerprobe --weights <重み.json> -d <マニフェスト.jsonl> [-s <件数>]")
    exit(1)
}

guard let weights = try? SpikingNetworkWeights.load(from: URL(fileURLWithPath: weightsPath)) else {
    print("エラー: 重みが読めません: \(weightsPath)")
    exit(1)
}
let network = MLXSpikingNetwork(weights: weights)
let trainer = MLXBPTTTrainer(network: network, bpttWindow: 4)
let frameStack = StreamingFeatureFrontEnd.frameStack(forInputDim: weights.inputDim)
let longContext = StreamingFeatureFrontEnd.hasLongContext(inputDim: weights.inputDim)
print("重み: \(weightsPath)")
print("層数 \(weights.numLayers) / 幅 \(weights.maxHiddenDim) / 入力 \(weights.inputDim) / 上位層の再帰 \(weights.hasUpperRecurrence) / 減衰率の学習 \(weights.hasLearnedBeta)")

struct Row: Decodable {
    let path: String
}
guard let content = try? String(contentsOfFile: manifestPath, encoding: .utf8) else {
    print("エラー: マニフェストが読めません: \(manifestPath)")
    exit(1)
}
var paths: [String] = []
for line in content.split(separator: "\n") {
    let trimmed = line.trimmingCharacters(in: .whitespaces)
    if trimmed.isEmpty {
        continue
    }
    if let row = try? JSONDecoder().decode(Row.self, from: Data(trimmed.utf8)) {
        paths.append(row.path)
    }
    if maxItems <= paths.count {
        break
    }
}

let numLayers = weights.numLayers
var sumCurrent = [Double](repeating: 0.0, count: numLayers)
var sumOwn = [Double](repeating: 0.0, count: numLayers)
var sumSpike = [Double](repeating: 0.0, count: numLayers)
var sumClipped = [Double](repeating: 0.0, count: numLayers)
var used = 0
var totalFrames = 0
for path in paths {
    let (_, features) = SpeechDataset.loadFeatures(path: path, frameStack: frameStack, longContext: longContext)
    if features.isEmpty {
        continue
    }
    var flat: [Float] = []
    flat.reserveCapacity(features.count * weights.inputDim)
    for frame in features {
        flat.append(contentsOf: frame)
    }
    let stats = trainer.layerStatistics(
        network: network,
        features: MLXArray(flat, [features.count, weights.inputDim])
    )
    var l = 0
    while l < numLayers {
        sumCurrent[l] += Double(stats[l].currentRMS)
        sumOwn[l] += Double(stats[l].ownCurrentRMS)
        sumSpike[l] += Double(stats[l].spikeRate)
        sumClipped[l] += Double(stats[l].clippedRate)
        l += 1
    }
    used += 1
    totalFrames += features.count
}
if used == 0 {
    print("エラー: 特徴量を作れた発話がありません")
    exit(1)
}
print("発話 \(used) 件 / \(totalFrames) フレーム (閾値 vTh = \(weights.vTh), 上限 \(LIFNeuronEngine.vClampMax))")
print("層    電流 RMS (残差込み)   この層の項 RMS   発火率    上限クリップ率")
var l = 0
while l < numLayers {
    let n = Double(used)
    print(String(
        format: "%-4d %20.3f %16.3f %10.4f %14.4f",
        l, sumCurrent[l] / n, sumOwn[l] / n, sumSpike[l] / n, sumClipped[l] / n
    ))
    l += 1
}
