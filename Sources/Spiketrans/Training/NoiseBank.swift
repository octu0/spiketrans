import Foundation

/// 学習時にきれいな発話へ重ねる雑音 (配信のゲーム音・BGM・効果音) のバンク。
///
/// 実況配信の誤りは声と同じ帯域に乗る非定常なゲーム音によるもので、EQ では分けられない。
/// 学習データ側で同じ条件を経験させるため、マニフェスト (`{"path": ...}` の jsonl) の WAV を
/// 16 kHz mono の PCM として保持し、発話ごとに確率 `probability` で 1 クリップを選び、
/// 発話の RMS に対して `snrRange` dB の SNR になるよう縮尺して足す。推論側は変更しない。
public final class NoiseBank: @unchecked Sendable {
    public static let sampleRate = 16000
    /// 1 発話に雑音を重ねる確率
    public static let probability: Double = 0.5
    /// 重ねるときの SNR (dB)。一様乱数で選ぶ
    public static let snrRange: ClosedRange<Float> = 5.0...20.0

    private let clips: [[Float]]

    public var count: Int {
        return clips.count
    }

    public var totalSeconds: Double {
        var n = 0
        for clip in clips {
            n += clip.count
        }
        return Double(n) / Double(Self.sampleRate)
    }

    /// クリップの PCM (16 kHz mono) から作る。空のクリップは捨てる
    public init(clips: [[Float]]) {
        self.clips = clips.filter { $0.isEmpty != true }
    }

    /// jsonl マニフェスト (`path` キー) の WAV を読み込む。読めないもの・16 kHz でないものは飛ばす
    public convenience init(manifestPath: String) throws {
        let content = try String(contentsOfFile: manifestPath, encoding: .utf8)
        var clips: [[Float]] = []
        for line in content.split(separator: "\n") {
            guard let obj = try? JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any],
                  let path = obj["path"] as? String else {
                continue
            }
            guard let wav = SpeechDataset.loadWavFile(path: path) else {
                continue
            }
            let pcm = SpeechDataset.resampleTo16k(pcmData: wav.pcmData, sampleRate: wav.sampleRate)
            if pcm.isEmpty != true {
                clips.append(pcm)
            }
        }
        self.init(clips: clips)
    }

    /// 発話 PCM (16 kHz) に雑音を重ねる。確率で素通し、発話が無音なら素通し
    public func mix(_ speech: [Float]) -> [Float] {
        var rng = SystemRandomNumberGenerator()
        return mix(speech, using: &rng)
    }

    public func mix<G: RandomNumberGenerator>(_ speech: [Float], using rng: inout G) -> [Float] {
        if clips.isEmpty || speech.isEmpty {
            return speech
        }
        if Self.probability < Double.random(in: 0.0..<1.0, using: &rng) {
            return speech
        }
        let snrDB = Float.random(in: Self.snrRange, using: &rng)
        let clip = clips[Int.random(in: 0..<clips.count, using: &rng)]
        let offset = Int.random(in: 0..<clip.count, using: &rng)
        return Self.mix(speech: speech, noise: clip, offset: offset, snrDB: snrDB)
    }

    /// 決定的な合成。`noise` は `offset` から始めて足りなければ先頭に巻き戻して繰り返す。
    /// 雑音の縮尺は 発話 RMS / 雑音 RMS / 10^(SNR/20)
    public static func mix(speech: [Float], noise: [Float], offset: Int, snrDB: Float) -> [Float] {
        if noise.isEmpty || speech.isEmpty {
            return speech
        }
        let speechRMS = rms(speech)
        if speechRMS <= 1e-6 {
            return speech
        }
        // 発話の長さぶんの雑音を切り出す
        var segment = [Float](repeating: 0.0, count: speech.count)
        var j = offset % noise.count
        var i = 0
        while i < speech.count {
            segment[i] = noise[j]
            j += 1
            if noise.count <= j {
                j = 0
            }
            i += 1
        }
        let noiseRMS = rms(segment)
        if noiseRMS <= 1e-6 {
            return speech
        }
        let gain = speechRMS / noiseRMS / powf(10.0, snrDB / 20.0)
        var out = speech
        i = 0
        while i < out.count {
            out[i] += segment[i] * gain
            i += 1
        }
        return out
    }

    static func rms(_ x: [Float]) -> Float {
        if x.isEmpty {
            return 0.0
        }
        var acc: Double = 0.0
        for v in x {
            acc += Double(v) * Double(v)
        }
        return Float((acc / Double(x.count)).squareRoot())
    }
}
