import Foundation
import Spiketrans

setbuf(stdout, nil)

// 学習マニフェストの音声を DSP でスクリーニングし、統計を付けた JSONL を出す。
//
// 字幕を自動整列したコーパスは「無音ばかり」「BGM だけ」「音声長に対して文字が少なすぎる」
// 区間を多く含み、そのまま学習に混ぜると blank 一色に崩れる。VAD で有声時間を測り、
// 文字密度を総時間ではなく有声時間で取り直し、ピッチとフォルマントで「声として成立
// しているか」を見る。閾値で絞るのは train.sh 側で、ここは計測だけを行う。
//
// 使い方: screen -i <manifest.jsonl> -o <screened.jsonl> [-p ワーカー数]

/// 1 発話ぶんの計測値
struct ScreenStats {
    var durationSeconds: Float = 0.0
    /// VAD で有声と判定されたフレームの割合
    var speechRatio: Float = 0.0
    /// 先頭・末尾の無音長 (秒)
    var leadSilenceSeconds: Float = 0.0
    var tailSilenceSeconds: Float = 0.0
    /// テキスト文字数 / 有声秒
    var charsPerSpeechSecond: Float = 0.0
    /// 有声フレームのうちピッチが取れた割合
    var voicedRatio: Float = 0.0
    /// 有声フレームのうち F1/F2 が母音らしい帯域に収まった割合
    var formantRatio: Float = 0.0
    /// 有声区間と無声区間の RMS 比 (dB)
    var snrDb: Float = 0.0
    /// 振幅 0.99 を超えたサンプルの割合 (クリッピング)
    var clipRatio: Float = 0.0
    var peak: Float = 0.0
}

struct ManifestEntry: Codable {
    let path: String
    let text: String
}

var inputPath = ""
var outputPath = ""
var numWorkers = 8
var argIdx = 1
let args = CommandLine.arguments
while argIdx < args.count {
    switch args[argIdx] {
    case "-i", "--input":
        if (argIdx + 1) < args.count {
            inputPath = args[argIdx + 1]
            argIdx += 1
        }
    case "-o", "--output":
        if (argIdx + 1) < args.count {
            outputPath = args[argIdx + 1]
            argIdx += 1
        }
    case "-p", "--parallel":
        if (argIdx + 1) < args.count {
            if let v = Int(args[argIdx + 1]) {
                numWorkers = max(1, v)
            }
            argIdx += 1
        }
    default:
        break
    }
    argIdx += 1
}
if inputPath.isEmpty || outputPath.isEmpty {
    print("使い方: screen -i <manifest.jsonl> -o <screened.jsonl> [-p ワーカー数]")
    exit(1)
}

guard let content = try? String(contentsOfFile: inputPath, encoding: .utf8) else {
    print("エラー: マニフェストを読めません: \(inputPath)")
    exit(1)
}
let decoder = JSONDecoder()
var entries: [ManifestEntry] = []
var rawLines: [String] = []
for line in content.components(separatedBy: "\n") {
    let trimmed = line.trimmingCharacters(in: .whitespaces)
    if trimmed.isEmpty {
        continue
    }
    guard let entry = try? decoder.decode(ManifestEntry.self, from: Data(trimmed.utf8)) else {
        print("エラー: 行を解釈できません: \(trimmed.prefix(100))")
        exit(1)
    }
    entries.append(entry)
    rawLines.append(trimmed)
}
print("入力: \(entries.count) 件 / ワーカー \(numWorkers)")

let dspConfig = DSPConfig()
let frameSize = dspConfig.frameSize
let hopSize = dspConfig.hopSize
let sampleRate = Float(dspConfig.sampleRate)
let pitchWindow = dspConfig.maxPitchLag * 2

/// 1 ファイルを計測する。ワーカーごとに VAD と作業領域を持つ
final class Screener {
    let vad = VAD(config: dspConfig)
    let pitch = PitchDetector(config: dspConfig)
    let lpc = LPC(config: dspConfig)
    let solver = DurandKernerSolver()
    let formants = FormantExtractor(sampleRate: sampleRate)
    let workspace = DSPWorkspace(maxFrameSize: 1024, lpcOrder: dspConfig.lpcOrder, melChannels: dspConfig.melChannels)

    func measure(path: String, textLength: Int) -> ScreenStats? {
        guard let wav = SpeechDataset.loadWavFile(path: path) else {
            return nil
        }
        let pcm = SpeechDataset.resampleTo16k(pcmData: wav.pcmData, sampleRate: wav.sampleRate)
        var stats = ScreenStats()
        stats.durationSeconds = Float(pcm.count) / sampleRate
        if pcm.count < frameSize {
            return stats
        }

        // 振幅
        var peak: Float = 0.0
        var clipped = 0
        var i = 0
        while i < pcm.count {
            let a = abs(pcm[i])
            if peak < a {
                peak = a
            }
            if 0.99 < a {
                clipped += 1
            }
            i += 1
        }
        stats.peak = peak
        stats.clipRatio = Float(clipped) / Float(pcm.count)

        // 発話単位の RMS 正規化と同じ前提で VAD を掛ける (小さい録音でも無音扱いにしない)
        var sumSq: Float = 0.0
        i = 0
        while i < pcm.count {
            sumSq += pcm[i] * pcm[i]
            i += 1
        }
        let rms = sqrt(sumSq / Float(pcm.count))
        var gain: Float = 1.0
        if 1e-6 < rms {
            gain = min(0.05 / rms, 20.0)
        }
        var scaled = pcm
        i = 0
        while i < scaled.count {
            scaled[i] *= gain
            i += 1
        }

        var frameCount = 0
        var speechFrames = 0
        var voicedFrames = 0
        var formantFrames = 0
        var firstSpeech = -1
        var lastSpeech = -1
        var speechEnergy: Float = 0.0
        var silenceEnergy: Float = 0.0
        var silenceFrames = 0

        scaled.withUnsafeBufferPointer { buf in
            let base = buf.baseAddress!
            var offset = 0
            while (offset + frameSize) <= scaled.count {
                let framePtr = base.advanced(by: offset)
                let v = vad.processFrame(ptr: framePtr, count: frameSize, workspace: workspace)
                if v.isSpeech {
                    speechFrames += 1
                    speechEnergy += v.rms * v.rms
                    if firstSpeech < 0 {
                        firstSpeech = frameCount
                    }
                    lastSpeech = frameCount
                    // ピッチ推定は最大ラグ (320 サンプル) より長い窓が要るので 2 フレームぶん見る
                    if (offset + pitchWindow) <= scaled.count {
                        let p = pitch.detectPitch(ptr: framePtr, count: pitchWindow, workspace: workspace)
                        if p.isVoiced {
                            voicedFrames += 1
                        }
                    }
                    if lpc.computeCoefficients(ptr: framePtr, count: frameSize, workspace: workspace) {
                        let solved = workspace.lpcCoeffs.withUnsafeBufferPointer { cPtr in
                            solver.solve(coefficients: cPtr.baseAddress!, order: dspConfig.lpcOrder, workspace: workspace)
                        }
                        if solved {
                            let f = workspace.durandKernerCurr.withUnsafeBufferPointer { rPtr in
                                formants.extractFormants(roots: rPtr.baseAddress!, count: dspConfig.lpcOrder)
                            }
                            if 2 <= f.count && 200.0 <= f.f1 && f.f1 <= 1100.0 && 600.0 <= f.f2 && f.f2 <= 3200.0 {
                                formantFrames += 1
                            }
                        }
                    }
                } else {
                    silenceEnergy += v.rms * v.rms
                    silenceFrames += 1
                }
                frameCount += 1
                offset += hopSize
            }
        }

        if frameCount == 0 {
            return stats
        }
        let hopSeconds = Float(hopSize) / sampleRate
        stats.speechRatio = Float(speechFrames) / Float(frameCount)
        if 0 <= firstSpeech {
            stats.leadSilenceSeconds = Float(firstSpeech) * hopSeconds
            stats.tailSilenceSeconds = Float(frameCount - 1 - lastSpeech) * hopSeconds
        } else {
            stats.leadSilenceSeconds = stats.durationSeconds
            stats.tailSilenceSeconds = stats.durationSeconds
        }
        let speechSeconds = Float(speechFrames) * hopSeconds
        if 0.0 < speechSeconds {
            stats.charsPerSpeechSecond = Float(textLength) / speechSeconds
            stats.voicedRatio = Float(voicedFrames) / Float(speechFrames)
            stats.formantRatio = Float(formantFrames) / Float(speechFrames)
        }
        if 0 < speechFrames && 0 < silenceFrames {
            let sp = speechEnergy / Float(speechFrames)
            let si = max(1e-10, silenceEnergy / Float(silenceFrames))
            stats.snrDb = 10.0 * log10(max(1e-10, sp) / si)
        }
        return stats
    }
}

final class ResultBuffer: @unchecked Sendable {
    var lines: [String?]
    var failed = 0
    let lock = NSLock()
    init(count: Int) {
        self.lines = [String?](repeating: nil, count: count)
    }
}

func formatStats(_ s: ScreenStats) -> String {
    return String(
        format: "\"dur\":%.2f,\"speech\":%.3f,\"lead\":%.2f,\"tail\":%.2f,\"cps\":%.2f,\"voiced\":%.3f,\"formant\":%.3f,\"snr\":%.1f,\"clip\":%.5f,\"peak\":%.3f",
        s.durationSeconds, s.speechRatio, s.leadSilenceSeconds, s.tailSilenceSeconds,
        s.charsPerSpeechSecond, s.voicedRatio, s.formantRatio, s.snrDb, s.clipRatio, s.peak
    )
}

let startedAt = Date()
let results = ResultBuffer(count: entries.count)
let progressEvery = max(1, entries.count / 20)
DispatchQueue.concurrentPerform(iterations: numWorkers) { worker in
    let screener = Screener()
    var i = worker
    while i < entries.count {
        let entry = entries[i]
        let statsLine: String
        switch screener.measure(path: entry.path, textLength: entry.text.count) {
        case .some(let s):
            statsLine = formatStats(s)
        case .none:
            statsLine = ""
            results.lock.lock()
            results.failed += 1
            results.lock.unlock()
        }
        // 元の行の末尾 "}" の前に統計を差し込む
        let raw = rawLines[i]
        if statsLine.isEmpty {
            results.lines[i] = nil
        } else {
            let body = raw.dropLast()
            results.lines[i] = String(body) + "," + statsLine + "}"
        }
        if (i % progressEvery) == 0 && worker == 0 {
            let elapsed = Date().timeIntervalSince(startedAt)
            print(String(format: "  %d / %d (%.0f 秒)", i, entries.count, elapsed))
        }
        i += numWorkers
    }
}

var output = ""
output.reserveCapacity(content.utf8.count + entries.count * 160)
var written = 0
for line in results.lines {
    if let l = line {
        output.append(l)
        output.append("\n")
        written += 1
    }
}
do {
    try output.write(toFile: outputPath, atomically: true, encoding: .utf8)
} catch {
    print("エラー: 出力を書けません: \(outputPath)")
    exit(1)
}
print(String(format: "完了: %d 件を出力 (読めなかった音声 %d 件、%.0f 秒)", written, results.failed, Date().timeIntervalSince(startedAt)))
