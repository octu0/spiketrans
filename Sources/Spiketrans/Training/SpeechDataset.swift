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

        public init(
            path: String,
            rawText: String,
            hiraganaText: String,
            textIds: [Int],
            phonemeIds: [Int],
            frameCount: Int
        ) {
            self.path = (path as NSString).standardizingPath
            self.rawText = rawText
            self.hiraganaText = hiraganaText
            self.textIds = textIds
            self.phonemeIds = phonemeIds
            self.frameCount = frameCount
        }
    }

    public let samples: [AudioTextSample]
    public let metaSamples: [SampleMeta]
    public let lazyFrameStack: Int
    public let cache: FeatureDiskCache?
    public let maxCacheGigabytes: Double
    private let isLazy: Bool

    public init(samples: [AudioTextSample]) {
        self.samples = samples
        self.metaSamples = []
        self.lazyFrameStack = 1
        self.cache = nil
        self.maxCacheGigabytes = 0.0
        self.isLazy = false
    }

    public init(
        metaSamples: [SampleMeta],
        frameStack: Int,
        cache: FeatureDiskCache? = nil,
        maxCacheGigabytes: Double = 0.0
    ) {
        self.samples = []
        self.metaSamples = metaSamples
        self.lazyFrameStack = frameStack
        self.cache = cache
        self.maxCacheGigabytes = maxCacheGigabytes
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

    /// 特徴量を生成せずに漢字テキストトークン ID 列を返す (言語 SNN 学習用)
    public func textIds(at index: Int) -> [Int] {
        if isLazy {
            return metaSamples[index].textIds
        }
        return samples[index].textIds
    }

    /// 遅延モードでは呼び出しごとに WAV を読み特徴量を生成する。
    /// キャッシュが存在する場合はディスクから特徴量を直読する (PCM はオンデマンドでのみ生成)
    public subscript(index: Int) -> AudioTextSample {
        return sample(at: index, loadPCM: false)
    }

    /// インデックス指定でサンプルを取得する。
    /// - Parameter loadPCM: キャッシュヒット時でも PCM データをデコードするかどうか。
    ///   通常の学習（SNN / CTC）では特徴量のみが必要なため `false` を推奨し、
    ///   フォルマント境界検出等の診断処理でのみ `true` を指定する。
    public func sample(at index: Int, loadPCM: Bool = false) -> AudioTextSample {
        if isLazy != true {
            return samples[index]
        }
        let meta = metaSamples[index]
        let (pcm16k, features) = Self.loadFeatures(
            path: meta.path,
            frameStack: lazyFrameStack,
            cache: cache,
            loadPCM: loadPCM
        )
        return AudioTextSample(
            audioPCM: pcm16k,
            rawText: meta.rawText,
            hiraganaText: meta.hiraganaText,
            textIds: meta.textIds,
            phonemeIds: meta.phonemeIds,
            acousticFeatures: features
        )
    }

    /// WAV ファイルまたはディスクキャッシュから音響特徴量をロードする
    public static func loadFeatures(
        path: String,
        frameStack: Int,
        cache: FeatureDiskCache? = nil,
        loadPCM: Bool = false
    ) -> (pcm: [Float], features: [[Float]]) {
        if let c = cache, let cached = c.load(path: path, frameStack: frameStack) {
            var pcm16k: [Float] = []
            if loadPCM {
                if let data = try? Data(contentsOf: URL(fileURLWithPath: path)),
                   let wav = try? WavParser().parse(bytes: [UInt8](data)) {
                    pcm16k = resampleTo16k(pcmData: wav.pcmData, sampleRate: wav.sampleRate)
                }
            }
            return (pcm16k, cached)
        }

        guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)),
              let wav = try? WavParser().parse(bytes: [UInt8](data)) else {
            return ([], [])
        }
        let pcm16k = resampleTo16k(pcmData: wav.pcmData, sampleRate: wav.sampleRate)
        let features = extractFeaturesFromPCM(pcmData: pcm16k, frameStack: frameStack)
        if let c = cache {
            if 0 < features.count {
                c.save(path: path, frameStack: frameStack, features: features)
            }
        }
        return (pcm16k, features)
    }

    /// サンプル一覧からフレーム数の降順（長い順）に、指定容量（GB）に収まる長尺音声をキャッシュ対象として選別する
    public static func selectQuotaPaths(
        metas: [SampleMeta],
        frameStack: Int,
        maxGigabytes: Double
    ) -> Set<String> {
        if maxGigabytes <= 0.0 {
            var allPaths = Set<String>()
            allPaths.reserveCapacity(metas.count)
            for m in metas {
                allPaths.insert(m.path)
            }
            return allPaths
        }

        let maxBytesDouble = maxGigabytes * 1024.0 * 1024.0 * 1024.0
        let maxBytes: Int64
        if Double(Int64.max) < maxBytesDouble {
            maxBytes = Int64.max
        } else {
            maxBytes = Int64(maxBytesDouble)
        }

        // 長尺（frameCount が大きい）順にソート (比較演算子は < のみ使用)
        let sorted = metas.sorted { a, b in
            if a.frameCount < b.frameCount {
                return false
            }
            if b.frameCount < a.frameCount {
                return true
            }
            return a.path < b.path
        }

        var seenPaths = Set<String>()
        seenPaths.reserveCapacity(sorted.count)
        var accumulatedBytes: Int64 = 0
        var selected = Set<String>()

        for item in sorted {
            if seenPaths.insert(item.path).inserted != true {
                continue
            }
            let itemBytes = FeatureDiskCache.estimateFileBytes(frameCount: item.frameCount, frameStack: frameStack)
            if itemBytes <= 0 {
                continue
            }
            let (nextTotal, overflow) = accumulatedBytes.addingReportingOverflow(itemBytes)
            if overflow != true {
                if nextTotal <= maxBytes {
                    accumulatedBytes = nextTotal
                    selected.insert(item.path)
                }
            }
        }

        return selected
    }

    /// マニフェストのペアから遅延データセットを構築する。
    /// キャッシュが存在する場合は 32 バイトヘッダーのみを読みフレーム数を瞬時に確定する
    public static func lazyFromManifest(
        pairs: [(path: String, text: String)],
        textVocabulary: TextVocabulary,
        phonemeVocabulary: PhonemeVocabulary = PhonemeVocabulary(),
        frameStack: Int = 1,
        workers: Int = 8,
        cache: FeatureDiskCache? = nil,
        maxCacheGigabytes: Double = 0.0
    ) -> SpeechDataset {
        final class MetaBuffer: @unchecked Sendable {
            var items: [SampleMeta?]
            init(count: Int) {
                self.items = [SampleMeta?](repeating: nil, count: count)
            }
        }
        let buffer = MetaBuffer(count: pairs.count)
        let workerCount = max(1, workers)

        // 容量制限（クォータ）が有効な場合、初期メタデータ走査時はキャッシュ保存を行わない
        // （全サンプルのフレーム数が確定した後に、長尺上位のみを選別してキャッシュするため）
        // また、過去のホワイトリストによってヘッダー読み込みが拒絶されないよう allowedPaths をリセットする
        let initialCache: FeatureDiskCache?
        if 0.0 < maxCacheGigabytes {
            cache?.allowedPaths = nil
            initialCache = nil
        } else {
            initialCache = cache
        }

        DispatchQueue.concurrentPerform(iterations: workerCount) { worker in
            let converter = KanjiConverter(vocabulary: phonemeVocabulary)
            var i = worker
            while i < pairs.count {
                let pair = pairs[i]
                var frameCount = 0
                if let c = cache, let cachedFrames = c.getFrameCount(path: pair.path, frameStack: frameStack) {
                    frameCount = cachedFrames
                } else {
                    let (_, features) = loadFeatures(
                        path: pair.path,
                        frameStack: frameStack,
                        cache: initialCache,
                        loadPCM: false
                    )
                    frameCount = features.count
                }

                if 0 < frameCount {
                    buffer.items[i] = SampleMeta(
                        path: pair.path,
                        rawText: pair.text,
                        hiraganaText: converter.convertToHiragana(pair.text),
                        textIds: textVocabulary.textToIds(pair.text),
                        phonemeIds: converter.toPhonemeTokenIds(pair.text),
                        frameCount: frameCount
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

        if let c = cache {
            if 0.0 < maxCacheGigabytes {
                let allowed = selectQuotaPaths(
                    metas: metas,
                    frameStack: frameStack,
                    maxGigabytes: maxCacheGigabytes
                )
                c.allowedPaths = allowed

                // クォータ対象外となったサンプルの旧キャッシュが存在する場合はディスクから削除し、
                // 指定された最大容量制約を確実に遵守する (同一パスの重複削除を防止)
                var checkedPaths = Set<String>()
                checkedPaths.reserveCapacity(metas.count)
                for meta in metas {
                    if checkedPaths.insert(meta.path).inserted {
                        if allowed.contains(meta.path) != true {
                            c.remove(path: meta.path, frameStack: frameStack)
                        }
                    }
                }
            } else {
                c.allowedPaths = nil
            }
        }

        return SpeechDataset(
            metaSamples: metas,
            frameStack: frameStack,
            cache: cache,
            maxCacheGigabytes: maxCacheGigabytes
        )
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
                let s1: Float
                if (srcIdx + 1) < pcmData.count {
                    s1 = pcmData[srcIdx + 1]
                } else {
                    s1 = s0
                }
                let s2: Float
                if (srcIdx + 2) < pcmData.count {
                    s2 = pcmData[srcIdx + 2]
                } else {
                    s2 = s1
                }
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

    /// 学習スクリプトとストリーミングの既定束ね数。SNN `inputDim` は `acousticInputDim()`。
    public static let defaultFrameStack = StreamingFeatureFrontEnd.defaultStack

    public static func acousticInputDim(frameStack: Int = defaultFrameStack) -> Int {
        return StreamingFeatureFrontEnd.acousticInputDim(stack: frameStack)
    }

    /// クリップ全体の RMS でゲインを固定し、`StreamingFeatureFrontEnd` で特徴化する。
    /// データセット・`transcribe`・`mictrans` の入口。ストリーミングと Mel / 3-tap / 束ねを共有する。
    public static func extractFeaturesFromPCM(pcmData: [Float], frameStack: Int = 1) -> [[Float]] {
        let totalSamples = pcmData.count
        if totalSamples < 400 {
            return []
        }

        var sumSquares: Float = 0.0
        var rIdx = 0
        while rIdx < totalSamples {
            sumSquares += pcmData[rIdx] * pcmData[rIdx]
            rIdx += 1
        }
        let rms = sqrtf(sumSquares / Float(totalSamples))
        let front = StreamingFeatureFrontEnd(frameStack: max(1, frameStack))
        front.setGain(StreamingFeatureFrontEnd.gainForRMS(rms))
        front.beginUtterance()

        let frameSize = front.frameSize
        let hopSize = front.hopSize
        var out: [[Float]] = []
        var offset = 0
        pcmData.withUnsafeBufferPointer { buf in
            let base = buf.baseAddress!
            while (offset + frameSize) <= totalSamples {
                if let stacked = front.pushRawFrame(
                    pcmPtr: base.advanced(by: offset),
                    count: frameSize
                ) {
                    out.append(Array(stacked))
                }
                offset += hopSize
            }
        }
        if let last = front.flush() {
            out.append(Array(last))
        }
        return out
    }
}
