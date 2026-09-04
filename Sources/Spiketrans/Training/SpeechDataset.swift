import Foundation

/// 1つの音声・テキスト学習サンプル
public struct AudioTextSample: Sendable {
    public let audioPCM: [Float]
    public let rawText: String
    public let hiraganaText: String
    public let textIds: [Int]
    public let phonemeIds: [Int]
    public let acousticFeatures: [[Float]] // [frames][128]

    public init(
        audioPCM: [Float],
        rawText: String,
        hiraganaText: String = "",
        textIds: [Int],
        phonemeIds: [Int] = [],
        acousticFeatures: [[Float]]
    ) {
        self.audioPCM = audioPCM
        self.rawText = rawText
        self.hiraganaText = hiraganaText
        self.textIds = textIds
        self.phonemeIds = phonemeIds
        self.acousticFeatures = acousticFeatures
    }
}

/// 音声・漢字テキスト学習用データセット (音素・かな音響学習 & 言語SNN統合)
///
/// 2 つのモードを持つ。
/// - 即時モード: 全サンプルの特徴量を保持する (テスト・小規模データ用)
/// - 遅延モード: メタデータだけ保持し、アクセス時に WAV から特徴量を生成する。
///   メモリ消費がデータ量と切り離されるため、大規模コーパスはこちらを使う
public final class SpeechDataset: @unchecked Sendable {
    /// 遅延モードの 1 発話ぶんのメタデータ
    public struct SampleMeta: Sendable {
        public let path: String
        public let rawText: String
        public let hiraganaText: String
        public let textIds: [Int]
        public let phonemeIds: [Int]
        public let frameCount: Int
    }

    public let samples: [AudioTextSample]
    public let metaSamples: [SampleMeta]
    public let lazyFrameStack: Int
    private let isLazy: Bool

    public init(samples: [AudioTextSample]) {
        self.samples = samples
        self.metaSamples = []
        self.lazyFrameStack = 1
        self.isLazy = false
    }

    public init(metaSamples: [SampleMeta], frameStack: Int) {
        self.samples = []
        self.metaSamples = metaSamples
        self.lazyFrameStack = frameStack
        self.isLazy = true
    }

    public var count: Int {
        if isLazy {
            return metaSamples.count
        }
        return samples.count
    }

    /// 特徴量を生成せずにフレーム数を返す (長さソート・CTC 整合判定用)
    public func frameCount(at index: Int) -> Int {
        if isLazy {
            return metaSamples[index].frameCount
        }
        return samples[index].acousticFeatures.count
    }

    /// 特徴量を生成せずにかな読みを返す (CTC 教師列の構築用)
    public func hiraganaText(at index: Int) -> String {
        if isLazy {
            return metaSamples[index].hiraganaText
        }
        return samples[index].hiraganaText
    }

    /// 遅延モードでは呼び出しごとに WAV を読み特徴量を生成する。
    /// 返り値を保持しない限りメモリには残らない
    public subscript(index: Int) -> AudioTextSample {
        if isLazy != true {
            return samples[index]
        }
        let meta = metaSamples[index]
        let (pcm16k, features) = Self.loadFeatures(path: meta.path, frameStack: lazyFrameStack)
        return AudioTextSample(
            audioPCM: pcm16k,
            rawText: meta.rawText,
            hiraganaText: meta.hiraganaText,
            textIds: meta.textIds,
            phonemeIds: meta.phonemeIds,
            acousticFeatures: features
        )
    }

    /// WAV ファイルを読み込んで 16kHz PCM と音響特徴量を生成する
    public static func loadFeatures(path: String, frameStack: Int) -> (pcm: [Float], features: [[Float]]) {
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)),
              let wav = try? WavParser().parse(bytes: [UInt8](data)) else {
            return ([], [])
        }
        let pcm16k = resampleTo16k(pcmData: wav.pcmData, sampleRate: wav.sampleRate)
        return (pcm16k, extractFeaturesFromPCM(pcmData: pcm16k, frameStack: frameStack))
    }

    /// マニフェストのペアから遅延データセットを構築する。
    /// 並列に全 WAV を一度読んでフレーム数を確定し、特徴量は破棄する
    public static func lazyFromManifest(
        pairs: [(path: String, text: String)],
        textVocabulary: TextVocabulary,
        phonemeVocabulary: PhonemeVocabulary = PhonemeVocabulary(),
        frameStack: Int = 1,
        workers: Int = 8
    ) -> SpeechDataset {
        final class MetaBuffer: @unchecked Sendable {
            var items: [SampleMeta?]
            init(count: Int) {
                self.items = [SampleMeta?](repeating: nil, count: count)
            }
        }
        let buffer = MetaBuffer(count: pairs.count)
        let workerCount = max(1, workers)
        DispatchQueue.concurrentPerform(iterations: workerCount) { worker in
            let converter = KanjiConverter(vocabulary: phonemeVocabulary)
            var i = worker
            while i < pairs.count {
                let pair = pairs[i]
                let (_, features) = loadFeatures(path: pair.path, frameStack: frameStack)
                if 0 < features.count {
                    buffer.items[i] = SampleMeta(
                        path: pair.path,
                        rawText: pair.text,
                        hiraganaText: converter.convertToHiragana(pair.text),
                        textIds: textVocabulary.textToIds(pair.text),
                        phonemeIds: converter.toPhonemeTokenIds(pair.text),
                        frameCount: features.count
                    )
                }
                i += workerCount
            }
        }
        var metas: [SampleMeta] = []
        metas.reserveCapacity(pairs.count)
        for item in buffer.items {
            if let meta = item {
                metas.append(meta)
            }
        }
        return SpeechDataset(metaSamples: metas, frameStack: frameStack)
    }

    /// テキスト中の全発音から音素トークン ID 列を抽出
    public static func extractFallbackPhonemeIds(text: String, phonemeVocabulary: PhonemeVocabulary = PhonemeVocabulary()) -> [Int] {
        let converter = KanjiConverter(vocabulary: phonemeVocabulary)
        return converter.toPhonemeTokenIds(text)
    }

    /// 任意のサンプリングレートの PCM データを 16kHz にリサンプリング (48kHz時は 3 サンプル平均のアンチエイリアシング間引き)
    public static func resampleTo16k(pcmData: [Float], sampleRate: Int) -> [Float] {
        if sampleRate == 16000 || pcmData.isEmpty {
            return pcmData
        }

        if sampleRate == 48000 {
            let outCount = pcmData.count / 3
            var out = [Float](repeating: 0.0, count: outCount)
            var m = 0
            while m < outCount {
                let srcIdx = m * 3
                let s0 = pcmData[srcIdx]
                let s1 = (srcIdx + 1 < pcmData.count) ? pcmData[srcIdx + 1] : s0
                let s2 = (srcIdx + 2 < pcmData.count) ? pcmData[srcIdx + 2] : s1
                // 3 サンプル平均ローパスフィルタによるエイリアシング防止
                out[m] = (s0 + s1 + s2) / 3.0
                m += 1
            }
            return out
        }

        // 一般的なサンプリングレート比率の場合 (線形補間)
        let ratio = Float(sampleRate) / 16000.0
        let outCount = Int(Float(pcmData.count) / ratio)
        if outCount <= 0 {
            return []
        }
        var out = [Float](repeating: 0.0, count: outCount)
        var m = 0
        while m < outCount {
            let srcPos = Float(m) * ratio
            let i0 = Int(srcPos)
            let i1 = min(pcmData.count - 1, i0 + 1)
            let frac = srcPos - Float(i0)
            if i0 < pcmData.count {
                out[m] = (1.0 - frac) * pcmData[i0] + frac * pcmData[i1]
            }
            m += 1
        }
        return out
    }

    /// WAV バイト列と漢字テキストのペア配列からデータセットを直接構築 (48kHz 等は 16kHz に自動リサンプル)
    public static func fromWavPairs(
        pairs: [(wavBytes: [UInt8], text: String)],
        textVocabulary: TextVocabulary,
        phonemeVocabulary: PhonemeVocabulary = PhonemeVocabulary(),
        frameStack: Int = 1
    ) throws -> SpeechDataset {
        let parser = WavParser()
        var sampleList: [AudioTextSample] = []
        var pIdx = 0

        while pIdx < pairs.count {
            let pair = pairs[pIdx]
            let wavData = try parser.parse(bytes: pair.wavBytes)
            let pcm16k = resampleTo16k(pcmData: wavData.pcmData, sampleRate: wavData.sampleRate)
            let textIds = textVocabulary.textToIds(pair.text)
            let converter = KanjiConverter(vocabulary: phonemeVocabulary)
            let hiraganaText = converter.convertToHiragana(pair.text)
            let phonemeIds = converter.toPhonemeTokenIds(pair.text)
            let featuresSeq = extractFeaturesFromPCM(pcmData: pcm16k, frameStack: frameStack)

            if 0 < featuresSeq.count {
                sampleList.append(AudioTextSample(
                    audioPCM: pcm16k,
                    rawText: pair.text,
                    hiraganaText: hiraganaText,
                    textIds: textIds,
                    phonemeIds: phonemeIds,
                    acousticFeatures: featuresSeq
                ))
            }

            pIdx += 1
        }

        return SpeechDataset(samples: sampleList)
    }

    /// PCM データとテキストのペア配列から直接構築
    public static func fromPCMPairs(
        pairs: [(pcmData: [Float], text: String)],
        textVocabulary: TextVocabulary,
        phonemeVocabulary: PhonemeVocabulary = PhonemeVocabulary(),
        frameStack: Int = 1
    ) -> SpeechDataset {
        var sampleList: [AudioTextSample] = []
        var pIdx = 0

        while pIdx < pairs.count {
            let pair = pairs[pIdx]
            let textIds = textVocabulary.textToIds(pair.text)
            let phonemeIds = extractFallbackPhonemeIds(text: pair.text, phonemeVocabulary: phonemeVocabulary)
            let featuresSeq = extractFeaturesFromPCM(pcmData: pair.pcmData, frameStack: frameStack)

            if 0 < featuresSeq.count {
                sampleList.append(AudioTextSample(
                    audioPCM: pair.pcmData,
                    rawText: pair.text,
                    textIds: textIds,
                    phonemeIds: phonemeIds,
                    acousticFeatures: featuresSeq
                ))
            }

            pIdx += 1
        }

        return SpeechDataset(samples: sampleList)
    }

    /// PCM 配列から 128次元音響特徴量系列 (Preemphasis + 64ch Mel + 3-tap 平滑/差分) を抽出
    /// 発話単位のレベル正規化の目標 RMS。JSUT (スタジオ正規化済み) の実測値に合わせ、
    /// 録音レベルがバラバラな実録音 (Common Voice・配信音声) を同じ入力電流レンジへ揃える
    static let targetRMS: Float = 0.05
    /// 正規化ゲインの上限。ほぼ無音の音声でノイズだけを増幅しないための歯止め
    static let maxGain: Float = 20.0

    public static func extractFeaturesFromPCM(pcmData: [Float], frameStack: Int = 1) -> [[Float]] {
        let totalSamples = pcmData.count
        if totalSamples < 400 {
            return []
        }

        // 0. 発話単位の RMS 正規化。静かな録音は入力電流が不足してスパイクが立たない
        var sumSquares: Float = 0.0
        var rIdx = 0
        while rIdx < totalSamples {
            sumSquares += pcmData[rIdx] * pcmData[rIdx]
            rIdx += 1
        }
        let rms = sqrtf(sumSquares / Float(totalSamples))
        var gain: Float = 1.0
        if 1e-6 < rms {
            gain = min(Self.targetRMS / rms, Self.maxGain)
        }

        // 1. PCM プリエンファシス (2-tap, coeff: 0.97)
        var preemph = [Float](repeating: 0.0, count: totalSamples)
        preemph[0] = pcmData[0] * gain
        var pIdx = 1
        while pIdx < totalSamples {
            preemph[pIdx] = (pcmData[pIdx] - (0.97 * pcmData[pIdx - 1])) * gain
            pIdx += 1
        }

        // 2. 64ch Mel Filterbank 抽出
        let dspConfig = DSPConfig(melChannels: 64)
        let filterbank = Filterbank(config: dspConfig)
        let workspace = DSPWorkspace(melChannels: 64)

        var rawMelSeq: [[Float]] = []
        var offset = 0
        let frameSize = dspConfig.frameSize
        let hopSize = dspConfig.hopSize

        preemph.withUnsafeBufferPointer { pcmPtr in
            let basePcm = pcmPtr.baseAddress!
            while (offset + frameSize) <= totalSamples {
                let framePtr = basePcm.advanced(by: offset)
                let feat = filterbank.extractFeatures(
                    pcmPtr: framePtr,
                    count: frameSize,
                    workspace: workspace
                )
                rawMelSeq.append(feat)
                offset += hopSize
            }
        }

        let numFrames = rawMelSeq.count
        if numFrames <= 0 {
            return []
        }

        // 3. Mel 系列の時間 3-tap カーネル適用 (平滑 s & 差分 d → 128 次元)
        var featuresSeq: [[Float]] = []
        featuresSeq.reserveCapacity(numFrames)

        var t = 0
        while t < numFrames {
            let prevIdx = (t == 0) ? 0 : t - 1
            let currIdx = t
            let nextIdx = (t == numFrames - 1) ? (numFrames - 1) : t + 1

            let prevMel = rawMelSeq[prevIdx]
            let currMel = rawMelSeq[currIdx]
            let nextMel = rawMelSeq[nextIdx]

            var feat128 = [Float](repeating: 0.0, count: 128)
            var c = 0
            while c < 64 {
                let xPrev = prevMel[c]
                let xCurr = currMel[c]
                let xNext = nextMel[c]

                // 平滑化: s[t] = 0.25*x[t-1] + 0.5*x[t] + 0.25*x[t+1]
                feat128[c] = (0.25 * xPrev) + (0.5 * xCurr) + (0.25 * xNext)
                // 差分: d[t] = 0.5*(x[t+1] - x[t-1])
                feat128[64 + c] = 0.5 * (xNext - xPrev)

                c += 1
            }

            featuresSeq.append(feat128)
            t += 1
        }

        return stackFrames(featuresSeq, stack: frameStack)
    }

    /// 連続する stack フレームを 1 フレームに束ねて時間解像度を落とす
    ///
    /// hopSize=160 (16kHz) では 1 フレーム 10ms と CTC には過剰に細かく、
    /// SNN の逐次ステップ数がそのまま学習時間に効く。3 フレーム束ねて 30ms 相当に
    /// すると情報を捨てずに逐次ステップを 1/3 にできる。
    /// stack = 1 のときは何もしない。
    public static func stackFrames(_ featuresSeq: [[Float]], stack: Int) -> [[Float]] {
        if stack <= 1 || featuresSeq.isEmpty {
            return featuresSeq
        }

        let frameDim = featuresSeq[0].count
        let outCount = featuresSeq.count / stack
        if outCount <= 0 {
            // 束ねるには短すぎる場合は 1 フレームに全部詰めてゼロ埋め
            var single = [Float](repeating: 0.0, count: frameDim * stack)
            var f = 0
            while f < featuresSeq.count {
                let src = featuresSeq[f]
                var d = 0
                while d < frameDim {
                    single[(f * frameDim) + d] = src[d]
                    d += 1
                }
                f += 1
            }
            return [single]
        }

        var out = [[Float]](
            repeating: [Float](repeating: 0.0, count: frameDim * stack),
            count: outCount
        )
        var o = 0
        while o < outCount {
            var k = 0
            while k < stack {
                let src = featuresSeq[(o * stack) + k]
                let offset = k * frameDim
                var d = 0
                while d < frameDim {
                    out[o][offset + d] = src[d]
                    d += 1
                }
                k += 1
            }
            o += 1
        }
        return out
    }
}
