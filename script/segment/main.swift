import Foundation
import Spiketrans

setbuf(stdout, nil)

// 長い音声を VAD で発話単位に切り、書き起こしを各区間に割り当てて学習マニフェストを作る。
//
// 入力は「1 本の WAV + その全文の書き起こし」。区間の切れ目は VAD の無音、文の切れ目は
// 句点・感嘆符・疑問符・三点リーダなど。文を区間へ割り当てるのは、区間の音声長と
// 文の文字数の比 (文字/秒) が全区間で一定に近くなる分割を動的計画法で選ぶ。
// 音響モデルによる強制整列ではないので、文字/秒が大きく外れた区間は確認用に出力する。
//
// 使い方: segment -i <音声.wav> -t <書き起こし.txt or 文字列> -o <出力ディレクトリ> [--max-seconds 15]
//         標準出力に {"path","text"} の JSONL を書く

/// 区間を結合する無音の上限 (秒)。これより短い間は同じ発話とみなす
let mergeGapSeconds: Float = 0.35
/// 単独では短すぎて捨てる区間 (秒)
let minSegmentSeconds: Float = 0.6
/// 1 区間の上限 (秒)。これを超える区間は内部でいちばん長い無音で割る
var maxSegmentSeconds: Float = 15.0
/// 割り当て後に「要確認」として出す文字密度の範囲 (文字/秒)
let plausibleCharsPerSecond: ClosedRange<Float> = 3.0...12.0

var wavPath = ""
var textArg = ""
var outDir = ""
var argIdx = 1
let args = CommandLine.arguments
while argIdx < args.count {
    switch args[argIdx] {
    case "-i", "--input":
        if (argIdx + 1) < args.count {
            wavPath = args[argIdx + 1]
            argIdx += 1
        }
    case "-t", "--text":
        if (argIdx + 1) < args.count {
            textArg = args[argIdx + 1]
            argIdx += 1
        }
    case "-o", "--output":
        if (argIdx + 1) < args.count {
            outDir = args[argIdx + 1]
            argIdx += 1
        }
    case "--max-seconds":
        if (argIdx + 1) < args.count {
            if let v = Float(args[argIdx + 1]) {
                maxSegmentSeconds = max(2.0, v)
            }
            argIdx += 1
        }
    default:
        break
    }
    argIdx += 1
}
if wavPath.isEmpty || textArg.isEmpty || outDir.isEmpty {
    FileHandle.standardError.write("使い方: segment -i <音声.wav> -t <書き起こし> -o <出力ディレクトリ> [--max-seconds 15]\n".data(using: .utf8)!)
    exit(1)
}

func warn(_ message: String) {
    FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
}

// 書き起こしはファイルパスでも直接の文字列でもよい
var transcript = textArg
if FileManager.default.fileExists(atPath: textArg) {
    guard let t = try? String(contentsOfFile: textArg, encoding: .utf8) else {
        warn("エラー: 書き起こしを読めません: \(textArg)")
        exit(1)
    }
    transcript = t
}

guard let fileData = try? Data(contentsOf: URL(fileURLWithPath: wavPath)),
      let wav = try? WavParser().parse(bytes: [UInt8](fileData)) else {
    warn("エラー: 音声を読めません: \(wavPath)")
    exit(1)
}
let pcm = SpeechDataset.resampleTo16k(pcmData: wav.pcmData, sampleRate: wav.sampleRate)
let sampleRate: Float = 16000.0
let totalSeconds = Float(pcm.count) / sampleRate

// ---- 1. VAD で区間を得て、短い間は結合し、長すぎる区間は無音で割る ----

let dspConfig = DSPConfig()
let vad = VAD(config: dspConfig)
let workspace = DSPWorkspace(maxFrameSize: 1024, lpcOrder: dspConfig.lpcOrder, melChannels: dspConfig.melChannels)

// VAD は発話単位の RMS 正規化を前提にしているので、小さい録音は持ち上げる
var sumSq: Float = 0.0
var i = 0
while i < pcm.count {
    sumSq += pcm[i] * pcm[i]
    i += 1
}
let rms = sqrt(sumSq / Float(max(1, pcm.count)))
var scaled = pcm
if 1e-6 < rms {
    let gain = min(0.05 / rms, 20.0)
    i = 0
    while i < scaled.count {
        scaled[i] *= gain
        i += 1
    }
}

let rawSegments = vad.segmentUtterances(pcmData: scaled, workspace: workspace)
if rawSegments.isEmpty {
    warn("エラー: 有声区間が見つかりません")
    exit(1)
}

struct Span {
    var start: Int
    var end: Int
    var seconds: Float {
        return Float(end - start) / sampleRate
    }
}

// 1a. 近い区間を結合
var merged: [Span] = []
for seg in rawSegments {
    if let last = merged.last, Float(seg.startIndex - last.end) / sampleRate < mergeGapSeconds {
        merged[merged.count - 1].end = max(last.end, seg.endIndex)
    } else {
        merged.append(Span(start: seg.startIndex, end: seg.endIndex))
    }
}

// 1b. 長すぎる区間は、内部でエネルギーが最も低い 200ms 窓で割る (再帰的に)
func lowestEnergySplit(_ span: Span, in samples: [Float]) -> Int {
    let window = Int(0.2 * sampleRate)
    let hop = Int(0.05 * sampleRate)
    // 端から 1 秒以内では割らない
    let margin = Int(1.0 * sampleRate)
    var best = (span.start + span.end) / 2
    var bestEnergy = Float.greatestFiniteMagnitude
    var pos = span.start + margin
    while (pos + window) <= (span.end - margin) {
        var e: Float = 0.0
        var k = pos
        while k < pos + window {
            e += samples[k] * samples[k]
            k += 1
        }
        if e < bestEnergy {
            bestEnergy = e
            best = pos + window / 2
        }
        pos += hop
    }
    return best
}

var spans: [Span] = []
var queue = merged
while queue.isEmpty != true {
    let span = queue.removeFirst()
    if maxSegmentSeconds < span.seconds {
        let cut = lowestEnergySplit(span, in: scaled)
        if span.start < cut && cut < span.end {
            queue.insert(Span(start: cut, end: span.end), at: 0)
            queue.insert(Span(start: span.start, end: cut), at: 0)
            continue
        }
    }
    spans.append(span)
}
spans = spans.filter { minSegmentSeconds <= $0.seconds }
if spans.isEmpty {
    warn("エラー: 十分な長さの区間がありません")
    exit(1)
}

// ---- 2. 書き起こしを文に分け、文字/秒が揃うように区間へ割り当てる ----

/// 文の切れ目になる記号。記号自身は前の文に残す
let sentenceBreaks: Set<Character> = ["。", "！", "？", "!", "?", "…", "\n"]
var sentences: [String] = []
var current = ""
for ch in transcript {
    current.append(ch)
    if sentenceBreaks.contains(ch) {
        let trimmed = current.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty != true {
            sentences.append(trimmed)
        }
        current = ""
    }
}
let tail = current.trimmingCharacters(in: .whitespacesAndNewlines)
if tail.isEmpty != true {
    sentences.append(tail)
}
// 文が区間より少ないときは読点でも切る
if sentences.count < spans.count {
    var finer: [String] = []
    for s in sentences {
        var buf = ""
        for ch in s {
            buf.append(ch)
            if ch == "、" || ch == "," {
                finer.append(buf)
                buf = ""
            }
        }
        if buf.isEmpty != true {
            finer.append(buf)
        }
    }
    sentences = finer
}

/// かな換算の文字数 (記号・空白を除く) で密度を測る
func speechLength(_ s: String, converter: KanjiConverter) -> Int {
    return converter.kanaOnly(converter.convertToHiragana(s)).count
}
let converter = KanjiConverter()
let sentenceLengths = sentences.map { speechLength($0, converter: converter) }
let totalChars = sentenceLengths.reduce(0, +)
let totalSpeech = spans.reduce(Float(0.0)) { $0 + $1.seconds }
let targetRate = Float(totalChars) / max(0.1, totalSpeech)

// 動的計画法: 文 i..<j を区間 k に割り当てたときのコスト = (文字数 - 区間秒 × 全体の文字/秒)^2 / 区間秒
// 各区間に少なくとも 1 文、文の順序は保つ
let sCount = sentences.count
let kCount = spans.count
if sCount < kCount {
    warn("警告: 文の数 (\(sCount)) が区間の数 (\(kCount)) より少ないため、区間を文の数に合わせて結合します")
}
let effectiveK = min(kCount, sCount)
// 区間数を文数以下に落とす必要があるときは、短い区間を隣とまとめる
while effectiveK < spans.count {
    var minIdx = 0
    var minLen = Float.greatestFiniteMagnitude
    var idx = 0
    while idx < spans.count {
        if spans[idx].seconds < minLen {
            minLen = spans[idx].seconds
            minIdx = idx
        }
        idx += 1
    }
    let neighbor: Int
    if minIdx == 0 {
        neighbor = 1
    } else {
        neighbor = minIdx - 1
    }
    let lo = min(minIdx, neighbor)
    let hi = max(minIdx, neighbor)
    spans[lo].end = spans[hi].end
    spans.remove(at: hi)
}

var prefix = [Int](repeating: 0, count: sCount + 1)
i = 0
while i < sCount {
    prefix[i + 1] = prefix[i] + sentenceLengths[i]
    i += 1
}
let inf = Float.greatestFiniteMagnitude
var cost = [[Float]](repeating: [Float](repeating: inf, count: sCount + 1), count: effectiveK + 1)
var choice = [[Int]](repeating: [Int](repeating: 0, count: sCount + 1), count: effectiveK + 1)
cost[0][0] = 0.0
var k = 1
while k <= effectiveK {
    let seconds = spans[k - 1].seconds
    let expected = seconds * targetRate
    var j = k
    while j <= sCount - (effectiveK - k) {
        var best = inf
        var bestI = 0
        var ii = k - 1
        while ii < j {
            if cost[k - 1][ii] < inf {
                let chars = Float(prefix[j] - prefix[ii])
                let diff = chars - expected
                let c = cost[k - 1][ii] + (diff * diff) / max(0.5, seconds)
                if c < best {
                    best = c
                    bestI = ii
                }
            }
            ii += 1
        }
        cost[k][j] = best
        choice[k][j] = bestI
        j += 1
    }
    k += 1
}

var boundaries = [Int](repeating: 0, count: effectiveK + 1)
boundaries[effectiveK] = sCount
k = effectiveK
while 0 < k {
    boundaries[k - 1] = choice[k][boundaries[k]]
    k -= 1
}

// ---- 3. 区間ごとに WAV を書き、マニフェスト行を出す ----

func writeWav(_ samples: ArraySlice<Float>, to path: String) -> Bool {
    var data = Data()
    let dataBytes = UInt32(samples.count * 2)
    func appendU32(_ v: UInt32) {
        var le = v.littleEndian
        data.append(Data(bytes: &le, count: 4))
    }
    func appendU16(_ v: UInt16) {
        var le = v.littleEndian
        data.append(Data(bytes: &le, count: 2))
    }
    data.append("RIFF".data(using: .ascii)!)
    appendU32(36 + dataBytes)
    data.append("WAVE".data(using: .ascii)!)
    data.append("fmt ".data(using: .ascii)!)
    appendU32(16)
    appendU16(1)
    appendU16(1)
    appendU32(16000)
    appendU32(16000 * 2)
    appendU16(2)
    appendU16(16)
    data.append("data".data(using: .ascii)!)
    appendU32(dataBytes)
    var pcm16 = [Int16](repeating: 0, count: samples.count)
    var n = 0
    for v in samples {
        let clamped = max(-1.0, min(1.0, v))
        pcm16[n] = Int16(clamped * 32767.0)
        n += 1
    }
    pcm16.withUnsafeBufferPointer { buf in
        data.append(Data(buffer: buf))
    }
    do {
        try data.write(to: URL(fileURLWithPath: path), options: .atomic)
        return true
    } catch {
        return false
    }
}

try? FileManager.default.createDirectory(atPath: outDir, withIntermediateDirectories: true)
let baseName = ((wavPath as NSString).lastPathComponent as NSString).deletingPathExtension
var written = 0
var flagged = 0
k = 0
while k < effectiveK {
    let span = spans[k]
    let text = sentences[boundaries[k]..<boundaries[k + 1]].joined()
    let chars = prefix[boundaries[k + 1]] - prefix[boundaries[k]]
    let rate = Float(chars) / span.seconds
    let outPath = (outDir as NSString).appendingPathComponent(String(format: "%@_seg%03d.wav", baseName, k + 1))
    // 元の (正規化前の) 音声を書く
    if writeWav(pcm[span.start..<span.end], to: outPath) != true {
        warn("エラー: 書き出しに失敗: \(outPath)")
        exit(1)
    }
    let escaped = text.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "\"", with: "\\\"")
    print("{\"path\": \"\(outPath)\", \"text\": \"\(escaped)\"}")
    if plausibleCharsPerSecond.contains(rate) != true {
        flagged += 1
        warn(String(format: "  要確認 seg%03d: %.1f 秒 / %d 文字 = %.1f 文字/秒: %@", k + 1, span.seconds, chars, rate, String(text.prefix(40))))
    }
    written += 1
    k += 1
}
warn(String(format: "完了: %.1f 秒の音声を %d 区間に分割 (全体 %.1f 文字/秒、要確認 %d 区間)", totalSeconds, written, targetRate, flagged))
