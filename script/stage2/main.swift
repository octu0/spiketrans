import Foundation
import Spiketrans
import MLX

setbuf(stdout, nil)

// 第2段 (かな漢字変換) だけを評価する。
// 正解のかなを入力して漢字を復元させるため音響 SNN を必要とせず、
// スコア設計を反復するあいだ数秒で結果が得られる。

var datasetPath = ""
var dictSamples = 0        // 辞書・語彙の構築に使う行数 (0 で全件)
var showExamples = 3
var nbest = 0              // 0 で N-best 評価なし
var lmWeightsPath = ""     // 言語モデル重みのパス。無ければ学習して保存する
var lmEpochs = 60
var lmWeight: Float = 0.5  // 再スコアリングにおける言語モデルの重み
var extraCorpusPath = ""   // 追加・汎用コーパステキストのパス

var argIdx = 1
let args = CommandLine.arguments
while argIdx < args.count {
    let arg = args[argIdx]
    switch arg {
    case "-d", "--dir", "--dataset":
        if (argIdx + 1) < args.count {
            datasetPath = args[argIdx + 1]
            argIdx += 1
        }
    case "-s", "--samples":
        if (argIdx + 1) < args.count {
            if let v = Int(args[argIdx + 1]) {
                dictSamples = max(0, v)
            }
            argIdx += 1
        }
    case "--extra-corpus":
        if (argIdx + 1) < args.count {
            extraCorpusPath = args[argIdx + 1]
            argIdx += 1
        }
    case "--nbest":
        if (argIdx + 1) < args.count {
            if let v = Int(args[argIdx + 1]) {
                nbest = max(0, v)
            }
            argIdx += 1
        }
    case "--lm-weights":
        if (argIdx + 1) < args.count {
            lmWeightsPath = args[argIdx + 1]
            argIdx += 1
        }
    case "--lm-epochs":
        if (argIdx + 1) < args.count {
            if let v = Int(args[argIdx + 1]) {
                lmEpochs = max(1, v)
            }
            argIdx += 1
        }
    case "--lm-weight":
        if (argIdx + 1) < args.count {
            if let v = Float(args[argIdx + 1]) {
                lmWeight = v
            }
            argIdx += 1
        }
    case "--examples":
        if (argIdx + 1) < args.count {
            if let v = Int(args[argIdx + 1]) {
                showExamples = max(0, v)
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

if datasetPath.isEmpty {
    print("エラー: データセットディレクトリを指定してください。")
    print("  使い方: stage2 -d <データセットディレクトリ> [-s 辞書構築行数] [--examples 件数]")
    exit(1)
}

let transcriptPath = (datasetPath as NSString).appendingPathComponent("transcript_utf8.txt")
guard let content = try? String(contentsOfFile: transcriptPath, encoding: .utf8) else {
    print("エラー: \(transcriptPath) が見つかりません。")
    exit(1)
}

var fileIds: [String] = []
var texts: [String] = []
for line in content.components(separatedBy: .newlines) {
    let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
    if trimmed.isEmpty {
        continue
    }
    let parts = trimmed.split(separator: ":", maxSplits: 1, omittingEmptySubsequences: true)
    if parts.count == 2 {
        fileIds.append(String(parts[0]).trimmingCharacters(in: .whitespacesAndNewlines))
        texts.append(String(parts[1]).trimmingCharacters(in: .whitespacesAndNewlines))
    }
}

let dictLimit: Int
if dictSamples == 0 {
    dictLimit = texts.count
} else {
    dictLimit = min(dictSamples, texts.count)
}
let dictTexts = Array(texts.prefix(dictLimit))

let converter = KanjiConverter()
let dictionary = KanaKanjiDictionary()
dictionary.buildFromCorpus(rawTexts: dictTexts, converter: converter)

var extraTexts: [String] = []
if extraCorpusPath.isEmpty != true {
    if let extraContent = try? String(contentsOfFile: extraCorpusPath, encoding: .utf8) {
        for line in extraContent.components(separatedBy: .newlines) {
            let tr = line.trimmingCharacters(in: .whitespacesAndNewlines)
            if tr.isEmpty != true {
                extraTexts.append(tr)
            }
        }
        dictionary.buildFromCorpus(rawTexts: extraTexts, converter: converter)
        print("追加コーパス読込: \(extraCorpusPath) (\(extraTexts.count) 行)")
    }
}

print("データセット: \(datasetPath)")
print("総行数: \(texts.count) 行 / 辞書構築: \(dictTexts.count) 行")
print("辞書エントリ数: \(dictionary.count) 語")

func levenshtein(_ s1: String, _ s2: String) -> Int {
    let a = Array(s1)
    let b = Array(s2)
    if a.isEmpty { return b.count }
    if b.isEmpty { return a.count }
    var prev = Array(0...b.count)
    var cur = [Int](repeating: 0, count: b.count + 1)
    var i = 1
    while i <= a.count {
        cur[0] = i
        var j = 1
        while j <= b.count {
            var cost = 1
            if a[i - 1] == b[j - 1] {
                cost = 0
            }
            var best = prev[j] + 1
            if cur[j - 1] + 1 < best { best = cur[j - 1] + 1 }
            if prev[j - 1] + cost < best { best = prev[j - 1] + cost }
            cur[j] = best
            j += 1
        }
        prev = cur
        i += 1
    }
    return prev[b.count]
}

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

struct Result {
    let fileId: String
    let target: String
    let produced: String
    let cer: Float
    let cerNoPunct: Float
    let isExact: Bool
    let isExactNoPunct: Bool
    let isDictSource: Bool
}

let decoder = KanaKanjiDecoder(dictionary: dictionary, languageBonus: 0.0)
var results: [Result] = []
let start = Date()

var idx = 0
while idx < texts.count {
    let target = texts[idx]
    let kana = converter.convertToHiragana(target)
    let produced = decoder.decode(kanaText: kana)

    let dist = levenshtein(target, produced)
    let cer = Float(dist) / Float(max(1, target.count))
    let tNo = stripPunctuation(target)
    let pNo = stripPunctuation(produced)
    let distNo = levenshtein(tNo, pNo)
    let cerNo = Float(distNo) / Float(max(1, tNo.count))

    results.append(Result(
        fileId: fileIds[idx],
        target: target,
        produced: produced,
        cer: cer,
        cerNoPunct: cerNo,
        isExact: target == produced,
        isExactNoPunct: tNo == pNo,
        isDictSource: idx < dictLimit
    ))
    idx += 1
}

let elapsed = Date().timeIntervalSince(start)

func summarize(_ list: [Result], _ label: String) {
    if list.isEmpty {
        return
    }
    let n = Float(list.count)
    var sumCer: Float = 0.0
    var sumCerNo: Float = 0.0
    var exact = 0
    var exactNo = 0
    for r in list {
        sumCer += r.cer
        sumCerNo += r.cerNoPunct
        if r.isExact { exact += 1 }
        if r.isExactNoPunct { exactNo += 1 }
    }
    print("[\(label)] \(list.count) 件")
    print("  句読点あり  CER: \(String(format: "%.2f", sumCer / n * 100.0))%  完全一致: \(String(format: "%.1f", Float(exact) / n * 100.0))%")
    print("  句読点除外  CER: \(String(format: "%.2f", sumCerNo / n * 100.0))%  完全一致: \(String(format: "%.1f", Float(exactNo) / n * 100.0))%")
}

print("\n==================================================")
print("=== 第2段単体 (正解かな入力) 評価 ===")
print("==================================================")
summarize(results.filter { $0.isDictSource }, "辞書構築に使った行")
let unseen = results.filter { $0.isDictSource != true }
summarize(unseen, "辞書構築に使っていない行")
print("\n処理時間: \(String(format: "%.2f", elapsed)) 秒 (\(String(format: "%.1f", Double(results.count) / elapsed)) 行/秒)")

// N-best のオラクル評価: 候補の中で最も正解に近いものを選べた場合の上限
if 0 < nbest {
    let nbStart = Date()
    var sumOracle: Float = 0.0
    var oracleExact = 0
    var sumTop1: Float = 0.0
    var rankSum = 0
    var evaluated = 0

    var i2 = 0
    while i2 < texts.count {
        if i2 < dictLimit {
            let target = stripPunctuation(texts[i2])
            let kana = converter.convertToHiragana(texts[i2])
            let cands = decoder.decodeNBest(kanaText: kana, beamWidth: nbest)
            if cands.isEmpty != true {
                var bestCer = Float.greatestFiniteMagnitude
                var bestRank = 0
                var r = 0
                while r < cands.count {
                    let p = stripPunctuation(cands[r].text)
                    let c = Float(levenshtein(target, p)) / Float(max(1, target.count))
                    if c < bestCer {
                        bestCer = c
                        bestRank = r
                    }
                    r += 1
                }
                let top1 = stripPunctuation(cands[0].text)
                sumTop1 += Float(levenshtein(target, top1)) / Float(max(1, target.count))
                sumOracle += bestCer
                if bestCer == 0.0 {
                    oracleExact += 1
                }
                rankSum += bestRank
                evaluated += 1
            }
        }
        i2 += 1
    }

    let nf = Float(max(1, evaluated))
    print("\n==================================================")
    print("=== N-best オラクル評価 (K = \(nbest), 句読点除外) ===")
    print("==================================================")
    print("  Top-1     CER: \(String(format: "%.2f", sumTop1 / nf * 100.0))%")
    print("  オラクル  CER: \(String(format: "%.2f", sumOracle / nf * 100.0))%  完全一致: \(String(format: "%.1f", Float(oracleExact) / nf * 100.0))%")
    print("  最良候補の平均順位: \(String(format: "%.2f", Float(rankSum) / nf))")
    print("  処理時間: \(String(format: "%.2f", Date().timeIntervalSince(nbStart))) 秒")
}

// 言語モデルによる N-best 再スコアリング
if lmWeightsPath.isEmpty != true && 0 < nbest {
    let allLmTexts = dictTexts + extraTexts
    let kanjiVocabulary = TextVocabulary(corpus: allLmTexts)
    let lm = CharLanguageModel.make(vocabulary: kanjiVocabulary)
    let weightsURL = URL(fileURLWithPath: lmWeightsPath)

    if FileManager.default.fileExists(atPath: lmWeightsPath) {
        if let w = try? SpikingNetworkWeights.load(from: weightsURL) {
            lm.importWeights(w)
            print("\n言語モデル重みを読み込みました: \(lmWeightsPath)")
        } else {
            print("\n警告: 言語モデル重みの読み込みに失敗しました")
        }
    } else {
        print("\n==================================================")
        print("=== 文字レベル言語モデルの学習 (語彙 \(kanjiVocabulary.size) 文字) ===")
        print("==================================================")

        let mlxNet = MLXSpikingNetwork(
            inputDim: kanjiVocabulary.size,
            maxHiddenDim: lm.network.maxHiddenDim,
            outputDim: kanjiVocabulary.size,
            timeSteps: lm.network.timeSteps,
            lifConfig: lm.network.lifConfig
        )
        let trainer = MLXBPTTTrainer(network: mlxNet, config: TrainingConfig(learningRate: 0.003), bpttWindow: 8)
        let scheduler = CosineLRScheduler(lrMax: 0.003, lrMin: 0.0005, totalEpochs: lmEpochs, warmupEpochs: 3)

        var pairs: [(features: [[Float]], targets: [Int])] = []
        for t in allLmTexts {
            let pair = CharLanguageModel.makeTrainingPair(text: t, vocabulary: kanjiVocabulary)
            if pair.features.isEmpty != true {
                pairs.append(pair)
            }
        }
        // 長さ順に並べてバッチ内のパディングを減らす
        pairs.sort { a, b in a.features.count < b.features.count }
        print("学習系列: \(pairs.count) 件")

        let batchSize = 16
        var ep = 1
        while ep <= lmEpochs {
            let epStart = Date()
            trainer.updateLearningRate(scheduler.learningRate(forEpoch: ep))
            var lossSum: Float = 0.0
            var batches = 0
            var bStart = 0
            while bStart < pairs.count {
                let bEnd = min(bStart + batchSize, pairs.count)
                var fb: [[[Float]]] = []
                var tb: [[Int]] = []
                var k = bStart
                while k < bEnd {
                    fb.append(pairs[k].features)
                    tb.append(pairs[k].targets)
                    k += 1
                }
                lossSum += trainer.trainBatch(featuresBatch: fb, targetsBatch: tb)
                batches += 1
                bStart = bEnd
            }
            if ep % 10 == 0 || ep == lmEpochs || ep == 1 {
                let avg = lossSum / Float(max(1, batches))
                print("  Epoch [\(ep)/\(lmEpochs)] 損失: \(String(format: "%.4f", avg)) (\(String(format: "%.1f", Date().timeIntervalSince(epStart))) 秒)")
            }
            ep += 1
        }

        lm.importWeights(mlxNet.exportWeights())
        try? lm.exportWeights().save(to: weightsURL)
        print("  言語モデル重みを保存しました: \(lmWeightsPath)")
        MLX.Memory.clearCache()
    }

    // 再スコアリング評価
    print("\n==================================================")
    print("=== 言語モデル再スコアリング (K = \(nbest), 重み \(lmWeight), 句読点除外) ===")
    print("==================================================")

    let rsStart = Date()

    struct RescoreStats {
        var sumBase: Float = 0.0
        var sumRescored: Float = 0.0
        var exactBase: Int = 0
        var exactRescored: Int = 0
        var count: Int = 0

        mutating func update(target: String, basePred: String, rescoredPred: String) {
            let baseCer = Float(levenshtein(target, basePred)) / Float(max(1, target.count))
            let rescoredCer = Float(levenshtein(target, rescoredPred)) / Float(max(1, target.count))
            sumBase += baseCer
            sumRescored += rescoredCer
            if baseCer == 0.0 { exactBase += 1 }
            if rescoredCer == 0.0 { exactRescored += 1 }
            count += 1
        }

        func report(label: String) {
            if count <= 0 {
                return
            }
            let nf = Float(count)
            print("[\(label)] \(count) 件")
            print("  再スコアリング前  CER: \(String(format: "%.2f", sumBase / nf * 100.0))%  完全一致: \(String(format: "%.1f", Float(exactBase) / nf * 100.0))%")
            print("  再スコアリング後  CER: \(String(format: "%.2f", sumRescored / nf * 100.0))%  完全一致: \(String(format: "%.1f", Float(exactRescored) / nf * 100.0))%")
        }
    }

    var statsTrain = RescoreStats()
    var statsUnseen = RescoreStats()

    var ri = 0
    while ri < texts.count {
        let target = stripPunctuation(texts[ri])
        let kana = converter.convertToHiragana(texts[ri])
        let cands = decoder.decodeNBest(kanaText: kana, beamWidth: nbest)
        if cands.isEmpty {
            ri += 1
            continue
        }

        var bestIdx = 0
        var bestScore = -Float.greatestFiniteMagnitude
        var c = 0
        while c < cands.count {
            let combined = cands[c].score + (lmWeight * lm.logProbability(of: cands[c].text))
            if bestScore < combined {
                bestScore = combined
                bestIdx = c
            }
            c += 1
        }

        let basePred = stripPunctuation(cands[0].text)
        let rescoredPred = stripPunctuation(cands[bestIdx].text)

        if ri < dictLimit {
            statsTrain.update(target: target, basePred: basePred, rescoredPred: rescoredPred)
        } else {
            statsUnseen.update(target: target, basePred: basePred, rescoredPred: rescoredPred)
        }

        ri += 1
    }

    statsTrain.report(label: "辞書構築に使った行")
    if dictLimit < texts.count {
        statsUnseen.report(label: "辞書構築に使っていない行")
    }
    print("  処理時間: \(String(format: "%.1f", Date().timeIntervalSince(rsStart))) 秒")
}

if 0 < showExamples {
    // 句読点だけが誤っている例。句読点以外は完全一致しているもの
    let punctOnly = results.filter { $0.isDictSource && $0.isExactNoPunct && ($0.isExact != true) }
    print("\n--- 句読点のみ誤り: \(punctOnly.count) 件 (辞書構築に使った行の \(String(format: "%.1f", Float(punctOnly.count) * 100.0 / Float(max(1, results.filter { $0.isDictSource }.count))))%) ---")
    for r in punctOnly.prefix(showExamples * 3) {
        print("  正解: \"\(r.target)\"")
        print("  出力: \"\(r.produced)\"")
    }

    let worst = results
        .filter { $0.isDictSource }
        .sorted { b, a in a.cerNoPunct < b.cerNoPunct }
        .prefix(showExamples)

    print("\n--- 誤り上位 \(worst.count) 件 (辞書構築に使った行) ---")
    for r in worst {
        print("\n[\(r.fileId)] 句読点除外 CER: \(String(format: "%.1f", r.cerNoPunct * 100.0))%")
        print("  正解かな: \"\(converter.convertToHiragana(r.target))\"")
        print("  正解漢字: \"\(r.target)\"")
        print("  出力:     \"\(r.produced)\"")
        let traceDecoder = KanaKanjiDecoder(dictionary: dictionary, languageBonus: 0.0)
        _ = traceDecoder.decode(kanaText: converter.convertToHiragana(r.target))
        for seg in traceDecoder.lastTrace {
            print("    \(seg.kanaRange) → \(seg.emitted)  [\(seg.kind)] \(String(format: "%+.2f", seg.stepScore))")
        }
    }
}
