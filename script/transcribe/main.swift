import Foundation
import Spiketrans

setbuf(stdout, nil)

var wavPath = ""
var weightsPath = ""
var dictPath = ""          // 第2段辞書を構築するテキスト (1 行 1 文)
var chunkSeconds = 0.0     // 0 で分割なし
var maxSeconds = 0.0       // 0 で全体

var argIdx = 1
let args = CommandLine.arguments
while argIdx < args.count {
    let arg = args[argIdx]
    switch arg {
    case "-i", "--input":
        if (argIdx + 1) < args.count {
            wavPath = args[argIdx + 1]
            argIdx += 1
        }
    case "-w", "--weights":
        if (argIdx + 1) < args.count {
            weightsPath = args[argIdx + 1]
            argIdx += 1
        }
    case "-d", "--dictionary":
        if (argIdx + 1) < args.count {
            dictPath = args[argIdx + 1]
            argIdx += 1
        }
    case "--chunk-seconds":
        if (argIdx + 1) < args.count {
            if let v = Double(args[argIdx + 1]) {
                chunkSeconds = max(0.0, v)
            }
            argIdx += 1
        }
    case "--max-seconds":
        if (argIdx + 1) < args.count {
            if let v = Double(args[argIdx + 1]) {
                maxSeconds = max(0.0, v)
            }
            argIdx += 1
        }
    default:
        break
    }
    argIdx += 1
}

if wavPath.isEmpty || weightsPath.isEmpty {
    print("使い方: transcribe -i <音声.wav> -w <重み.json> [-d <辞書テキスト>] [--chunk-seconds N] [--max-seconds N]")
    exit(1)
}

/// 常駐メモリ (MB)
func residentMemoryMB() -> Double {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
    let result = withUnsafeMutablePointer(to: &info) { infoPtr in
        infoPtr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { intPtr in
            task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), intPtr, &count)
        }
    }
    if result != KERN_SUCCESS {
        return 0.0
    }
    return Double(info.resident_size) / (1024.0 * 1024.0)
}

func report(_ label: String) {
    print(String(format: "  [メモリ] %@: %.0f MB", label, residentMemoryMB()))
}

print("==================================================")
print("=== 文字起こし ===")
print("==================================================")

// 1. 音声の読み込み
let loadStart = CFAbsoluteTimeGetCurrent()
guard let fileData = try? Data(contentsOf: URL(fileURLWithPath: wavPath)),
      let wavData = try? WavParser().parse(bytes: [UInt8](fileData)) else {
    print("エラー: 音声ファイルを読み込めません: \(wavPath)")
    exit(1)
}
var pcm = SpeechDataset.resampleTo16k(pcmData: wavData.pcmData, sampleRate: wavData.sampleRate)
if 0.0 < maxSeconds {
    let limit = Int(maxSeconds * 16000.0)
    if limit < pcm.count {
        pcm = Array(pcm[0..<limit])
    }
}
let audioSeconds = Double(pcm.count) / 16000.0
print(String(format: "音声: %.1f 分 (%.0f 秒), 読み込み %.1f 秒",
             audioSeconds / 60.0, audioSeconds, CFAbsoluteTimeGetCurrent() - loadStart))
report("音声読み込み後")

// 2. モデルの読み込み
guard let weights = try? SpikingNetworkWeights.load(from: URL(fileURLWithPath: weightsPath)) else {
    print("エラー: 重みを読み込めません: \(weightsPath)")
    exit(1)
}
print("第1段: 入力 \(weights.inputDim) 次元 / 隠れ \(weights.maxHiddenDim) / 出力 \(weights.outputDim), beta = \(weights.beta)")

// 3. 語彙と第2段辞書。
// 語彙は重みに同梱されているものを優先し、無い場合だけテキストから作り直す。
var corpusLines: [String] = []
if dictPath.isEmpty != true {
    guard let dictContent = try? String(contentsOfFile: dictPath, encoding: .utf8) else {
        print("エラー: テキストを読み込めません: \(dictPath)")
        exit(1)
    }
    for line in dictContent.components(separatedBy: "\n") {
        let tr = line.trimmingCharacters(in: .whitespacesAndNewlines)
        if tr.isEmpty != true {
            corpusLines.append(tr)
        }
    }
}

let kanjiConverter = KanjiConverter()
let phoneticVocabulary: TextVocabulary
switch weights.vocabulary {
case .some(let embedded):
    phoneticVocabulary = embedded
    print("かな語彙: \(phoneticVocabulary.size) 文字 (重みに同梱)")
case .none:
    if corpusLines.isEmpty {
        print("エラー: この重みにはかな語彙が含まれていません。学習時と同じテキストを -d に指定してください。")
        exit(1)
    }
    phoneticVocabulary = TextVocabulary(corpus: corpusLines.map { kanjiConverter.convertToHiragana($0) })
    print("かな語彙: \(phoneticVocabulary.size) 文字 (\(corpusLines.count) 行から再構築)")
}
if phoneticVocabulary.size != weights.outputDim {
    print("エラー: 語彙数 \(phoneticVocabulary.size) が重みの出力次元 \(weights.outputDim) と一致しません。")
    exit(1)
}

let kanaKanjiDict = KanaKanjiDictionary()
if corpusLines.isEmpty != true {
    kanaKanjiDict.buildFromCorpus(rawTexts: corpusLines)
    print("第2段辞書: \(kanaKanjiDict.count) 語")
    report("辞書構築後")
} else {
    print("第2段辞書: なし (かなのみ出力します)")
}

// 4. 分割単位の決定
let frameStack = 4
var segments: [(start: Int, count: Int)] = []
if chunkSeconds <= 0.0 {
    segments.append((start: 0, count: pcm.count))
} else {
    let chunkSamples = Int(chunkSeconds * 16000.0)
    var offset = 0
    while offset < pcm.count {
        let n = min(chunkSamples, pcm.count - offset)
        segments.append((start: offset, count: n))
        offset += n
    }
}
print("分割: \(segments.count) チャンク" + (chunkSeconds <= 0.0 ? " (分割なし)" : " (\(Int(chunkSeconds)) 秒ごと)"))

// 5. 文字起こし
let network = SpikingNetwork(
    inputDim: weights.inputDim,
    maxHiddenDim: weights.maxHiddenDim,
    outputDim: weights.outputDim,
    timeSteps: weights.timeSteps,
    lifConfig: weights.lifConfig
)
network.importWeights(from: weights)
report("モデル読み込み後")

let decoder = AcousticDecoder(
    network: network,
    vocabulary: phoneticVocabulary,
    fallbackVocabulary: PhonemeVocabulary()
)
let workspace = AcousticWorkspace(
    maxHiddenDim: network.maxHiddenDim,
    outputDim: network.outputDim,
    inputDim: network.inputDim
)
let beamDecoder = CTCBeamDecoder(
    vocabulary: phoneticVocabulary,
    blankId: TextVocabulary.padId,
    beamWidth: 16
)

var kanaParts: [String] = []
var totalFrames = 0
var peakMemory = residentMemoryMB()
let inferStart = CFAbsoluteTimeGetCurrent()

var segIdx = 0
while segIdx < segments.count {
    let seg = segments[segIdx]
    let segPCM = Array(pcm[seg.start..<(seg.start + seg.count)])
    let features = SpeechDataset.extractFeaturesFromPCM(pcmData: segPCM, frameStack: frameStack)
    totalFrames += features.count

    if 0 < features.count {
        let frameProbs = decoder.decodeSequence(featuresSeq: features, workspace: workspace)
        let outDim = network.outputDim
        var logProbs = [[Float]](
            repeating: [Float](repeating: 0.0, count: outDim),
            count: frameProbs.count
        )
        var f = 0
        while f < frameProbs.count {
            let probs = frameProbs[f].probabilities
            var c = 0
            while c < outDim {
                logProbs[f][c] = log(max(1e-30, probs[c]))
                c += 1
            }
            f += 1
        }
        kanaParts.append(beamDecoder.decode(logProbs: logProbs).text)
    }

    let mem = residentMemoryMB()
    if peakMemory < mem {
        peakMemory = mem
    }
    if (segIdx % 10) == 0 || segIdx == (segments.count - 1) {
        let done = Double(seg.start + seg.count) / 16000.0
        let elapsed = CFAbsoluteTimeGetCurrent() - inferStart
        print(String(format: "  [%d/%d] %.0f 秒処理 / 経過 %.1f 秒 / メモリ %.0f MB",
                     segIdx + 1, segments.count, done, elapsed, mem))
    }
    segIdx += 1
}

let inferElapsed = CFAbsoluteTimeGetCurrent() - inferStart
let kanaText = kanaParts.joined()
print("")
print("==================================================")
print(String(format: "第1段 完了: %.1f 秒 (音声 %.0f 秒, RTF %.4f, 実時間の %.0f 倍速)",
             inferElapsed, audioSeconds, inferElapsed / audioSeconds, audioSeconds / inferElapsed))
print("  処理フレーム数: \(totalFrames), かな文字数: \(kanaText.count)")
print(String(format: "  ピークメモリ: %.0f MB", peakMemory))

// 6. 第2段 (辞書がある場合)
if 0 < kanaText.count && 0 < kanaKanjiDict.count {
    let stage2Start = CFAbsoluteTimeGetCurrent()
    let kanaDecoder = KanaKanjiDecoder(dictionary: kanaKanjiDict, languageBonus: 0.0)
    var kanjiParts: [String] = []
    var pIdx = 0
    while pIdx < kanaParts.count {
        kanjiParts.append(kanaDecoder.decode(kanaText: kanaParts[pIdx]))
        pIdx += 1
    }
    let kanjiText = kanjiParts.joined()
    let stage2Elapsed = CFAbsoluteTimeGetCurrent() - stage2Start
    print(String(format: "第2段 完了: %.1f 秒 (漢字 %d 文字)", stage2Elapsed, kanjiText.count))
    let mem = residentMemoryMB()
    if peakMemory < mem {
        peakMemory = mem
    }
    print(String(format: "  ピークメモリ: %.0f MB", peakMemory))
    print("")
    print("--- 冒頭 300 文字 ---")
    print(String(kanjiText.prefix(300)))
}
print("")
print("--- かな冒頭 200 文字 ---")
print(String(kanaText.prefix(200)))
