import Foundation
import Spiketrans

setbuf(stdout, nil)

// 長い音声を発話単位に切り、書き起こしを各区間に付けて学習マニフェストを作る。
//
// 書き起こしは 2 形式を受け付ける。
//   時刻付き: SRT (`HH:MM:SS,mmm --> HH:MM:SS,mmm` の次の行から空行までが本文)。
//             時刻どおりに切る。1 区間が --max-seconds を超えるときだけ、その中を VAD で割り文を配分する
//   全文のみ: 区間の切れ目は VAD の無音 (transcribe と同じ SpeechChunker)、文の切れ目は
//             句点・感嘆符・疑問符・三点リーダ。区間の秒数と文字数の比が全区間で揃うよう
//             動的計画法で文を区間へ配分する。音響モデルによる強制整列ではないので、
//             文字/秒が外れた区間は「要確認」として出す
//
// 使い方: segment -i <音声.wav> -t <書き起こし.srt / .txt / 文字列> -o <出力ディレクトリ> [--max-seconds 15]
//         標準出力に {"path","text"} の JSONL を書く

/// 区間を結合する無音の上限 (秒)。これより短い間は同じ発話とみなす
let mergeGapSeconds: Float = 0.35
/// 単独では短すぎて捨てる区間 (秒)
let minSegmentSeconds: Float = 0.6
/// 1 区間の上限 (秒)。これを超える区間は内部でいちばん静かな窓で割る
var maxSegmentSeconds: Float = 15.0
/// 割り当て後に「要確認」として出す文字密度の範囲 (文字/秒)
let plausibleCharsPerSecond: ClosedRange<Float> = 3.0...12.0
let sampleRate: Float = 16000.0

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

func warn(_ message: String) {
    FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
}

if wavPath.isEmpty || textArg.isEmpty || outDir.isEmpty {
    warn("使い方: segment -i <音声.wav> -t <書き起こし> -o <出力ディレクトリ> [--max-seconds 15]")
    exit(1)
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
let totalSeconds = Float(pcm.count) / sampleRate
let converter = KanjiConverter()
let chunker = SpeechChunker(
    mergeGapSeconds: mergeGapSeconds,
    minSegmentSeconds: minSegmentSeconds,
    maxSegmentSeconds: maxSegmentSeconds
)

extension SpeechChunker.Span {
    var seconds: Float {
        return Float(end - start) / 16000.0
    }
}

// ---- 文の分割と、区間への配分 ----

/// 文の切れ目になる記号。記号自身は前の文に残す
let sentenceBreaks: Set<Character> = ["。", "！", "？", "!", "?", "…", "\n"]

func splitSentences(_ text: String, atLeast minimum: Int) -> [String] {
    var sentences: [String] = []
    var current = ""
    for ch in text {
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
    if sentences.count < minimum {
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
    return sentences
}

/// かな換算の文字数 (記号・空白を除く) で密度を測る
func speechLength(_ s: String) -> Int {
    return converter.kanaOnly(converter.convertToHiragana(s)).count
}

/// 文を区間へ順序を保って配分する。区間数が文数より多いときは短い区間を隣とまとめる。
/// 動的計画法: 文 i..<j を区間 k に置くコスト = (文字数 - 区間秒 × 全体の文字/秒)^2 / 区間秒
func assign(sentences: [String], to inputSpans: [SpeechChunker.Span]) -> [(span: SpeechChunker.Span, text: String, chars: Int)] {
    var spans = inputSpans
    let sCount = sentences.count
    if sCount == 0 || spans.isEmpty {
        return []
    }
    if sCount < spans.count {
        warn("警告: 文の数 (\(sCount)) が区間の数 (\(spans.count)) より少ないため、区間を文の数に合わせて結合します")
    }
    let kCount = min(spans.count, sCount)
    while kCount < spans.count {
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

    let lengths = sentences.map { speechLength($0) }
    var prefix = [Int](repeating: 0, count: sCount + 1)
    var i = 0
    while i < sCount {
        prefix[i + 1] = prefix[i] + lengths[i]
        i += 1
    }
    let totalSpeech = spans.reduce(Float(0.0)) { $0 + $1.seconds }
    let targetRate = Float(prefix[sCount]) / max(0.1, totalSpeech)

    let inf = Float.greatestFiniteMagnitude
    var cost = [[Float]](repeating: [Float](repeating: inf, count: sCount + 1), count: kCount + 1)
    var choice = [[Int]](repeating: [Int](repeating: 0, count: sCount + 1), count: kCount + 1)
    cost[0][0] = 0.0
    var k = 1
    while k <= kCount {
        let seconds = spans[k - 1].seconds
        let expected = seconds * targetRate
        var j = k
        while j <= sCount - (kCount - k) {
            var best = inf
            var bestI = 0
            var ii = k - 1
            while ii < j {
                if cost[k - 1][ii] < inf {
                    let diff = Float(prefix[j] - prefix[ii]) - expected
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
    var boundaries = [Int](repeating: 0, count: kCount + 1)
    boundaries[kCount] = sCount
    k = kCount
    while 0 < k {
        boundaries[k - 1] = choice[k][boundaries[k]]
        k -= 1
    }

    var result: [(span: SpeechChunker.Span, text: String, chars: Int)] = []
    k = 0
    while k < kCount {
        result.append((
            span: spans[k],
            text: sentences[boundaries[k]..<boundaries[k + 1]].joined(),
            chars: prefix[boundaries[k + 1]] - prefix[boundaries[k]]
        ))
        k += 1
    }
    return result
}

// ---- 区間の決定 ----

/// SRT の 1 キュー
struct TimedLine {
    let start: Float
    let end: Float
    let text: String
}

/// `HH:MM:SS,mmm` (区切りは , か .) を秒に。形式が違えば nil
func parseSRTTime(_ s: String) -> Float? {
    let t = s.trimmingCharacters(in: .whitespaces).replacingOccurrences(of: ",", with: ".")
    let parts = t.components(separatedBy: ":")
    if parts.count != 3 {
        return nil
    }
    guard let h = Float(parts[0]), let m = Float(parts[1]), let sec = Float(parts[2]) else {
        return nil
    }
    return (h * 3600.0) + (m * 60.0) + sec
}

/// SRT を読む。連番行は無視し、時刻行の次から空行までを本文とする (複数行は連結)
func parseSRT(_ text: String) -> [TimedLine] {
    var cues: [TimedLine] = []
    var pendingStart: Float = 0.0
    var pendingEnd: Float = 0.0
    var inCue = false
    var body = ""

    func closeCue() {
        let trimmed = body.trimmingCharacters(in: .whitespacesAndNewlines)
        if inCue && trimmed.isEmpty != true && pendingStart < pendingEnd {
            cues.append(TimedLine(start: pendingStart, end: pendingEnd, text: trimmed))
        }
        inCue = false
        body = ""
    }

    for raw in text.components(separatedBy: "\n") {
        let line = raw.trimmingCharacters(in: .whitespaces)
        if line.isEmpty {
            closeCue()
            continue
        }
        if line.contains("-->") {
            closeCue()
            let parts = line.components(separatedBy: "-->")
            if parts.count == 2, let s = parseSRTTime(parts[0]), let e = parseSRTTime(parts[1]) {
                pendingStart = s
                pendingEnd = e
                inCue = true
            }
            continue
        }
        if inCue {
            body += line
        }
    }
    closeCue()
    return cues
}

/// 区間の由来。cue は SRT の時刻どおり (整列は正しい)、vad は VAD 分割 + 文の配分 (整列は推定)
var pieces: [(span: SpeechChunker.Span, text: String, chars: Int, exact: Bool)] = []
let timed = parseSRT(transcript)
let modeDescription: String
if timed.isEmpty {
    // 全文のみ: VAD で切って文を配分する
    modeDescription = "VAD 分割 + 文の配分"
    let spans = chunker.chunk(pcm: pcm)
    if spans.isEmpty {
        warn("エラー: 十分な長さの有声区間がありません")
        exit(1)
    }
    pieces = assign(sentences: splitSentences(transcript, atLeast: spans.count), to: spans).map {
        (span: $0.span, text: $0.text, chars: $0.chars, exact: false)
    }
} else {
    modeDescription = "SRT"
    var skipped = 0
    for line in timed {
        let start = max(0, Int(line.start * sampleRate))
        let end = min(pcm.count, Int(line.end * sampleRate))
        if end <= start {
            skipped += 1
            continue
        }
        let span = SpeechChunker.Span(start: start, end: end)
        if span.seconds <= maxSegmentSeconds {
            pieces.append((span: span, text: line.text, chars: speechLength(line.text), exact: true))
            continue
        }
        // 上限を超える行だけ、その中を VAD で割って文を配分する
        let sub = chunker.chunk(pcm: Array(pcm[start..<end])).map {
            SpeechChunker.Span(start: $0.start + start, end: $0.end + start)
        }
        if sub.isEmpty {
            pieces.append((span: span, text: line.text, chars: speechLength(line.text), exact: true))
            continue
        }
        pieces += assign(sentences: splitSentences(line.text, atLeast: sub.count), to: sub).map {
            (span: $0.span, text: $0.text, chars: $0.chars, exact: false)
        }
    }
    if 0 < skipped {
        warn("警告: 音声の範囲外などで \(skipped) 行を飛ばしました")
    }
}

// ---- WAV とマニフェスト行の出力 ----

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
var flagged = 0
var totalChars = 0
var totalSpeech: Float = 0.0
var index = 0
while index < pieces.count {
    let piece = pieces[index]
    let rate = Float(piece.chars) / piece.span.seconds
    let outPath = (outDir as NSString).appendingPathComponent(String(format: "%@_seg%03d.wav", baseName, index + 1))
    if writeWav(pcm[piece.span.start..<piece.span.end], to: outPath) != true {
        warn("エラー: 書き出しに失敗: \(outPath)")
        exit(1)
    }
    let escaped = piece.text.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "\"", with: "\\\"")
    // source と cps (文字/秒) は呼び出し側が整列の怪しい区間を落とすのに使う。学習側は path と text だけ読む
    var source = "vad"
    if piece.exact {
        source = "cue"
    }
    print(String(format: "{\"path\": \"%@\", \"text\": \"%@\", \"source\": \"%@\", \"cps\": %.2f}", outPath, escaped, source, rate))
    if piece.exact != true && plausibleCharsPerSecond.contains(rate) != true {
        flagged += 1
        warn(String(format: "  要確認 seg%03d [%.1f -> %.1f]: %d 文字 = %.1f 文字/秒: %@",
                    index + 1, Float(piece.span.start) / sampleRate, Float(piece.span.end) / sampleRate,
                    piece.chars, rate, String(piece.text.prefix(40))))
    }
    totalChars += piece.chars
    totalSpeech += piece.span.seconds
    index += 1
}
warn(String(format: "完了 (%@): %.1f 秒の音声を %d 区間に分割 (全体 %.1f 文字/秒、要確認 %d 区間)",
            modeDescription, totalSeconds, pieces.count, Float(totalChars) / max(0.1, totalSpeech), flagged))
