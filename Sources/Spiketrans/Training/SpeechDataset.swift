import Foundation

/// 1つの音声・テキスト学習サンプル
public struct AudioTextSample: Sendable {
    public let audioPCM: [Float]
    public let rawText: String
    public let textIds: [Int]
    public let phonemeIds: [Int] // フォールバック音素教師（ひらがな・カタカナのみ、漢字はpad）
    public let acousticFeatures: [[Float]] // [frames][64]

    public init(
        audioPCM: [Float],
        rawText: String,
        textIds: [Int],
        phonemeIds: [Int] = [],
        acousticFeatures: [[Float]]
    ) {
        self.audioPCM = audioPCM
        self.rawText = rawText
        self.textIds = textIds
        self.phonemeIds = phonemeIds
        self.acousticFeatures = acousticFeatures
    }
}

/// 音声・漢字テキスト学習用データセット (読み変換不要・直接漢字テキストを学習)
public final class SpeechDataset: @unchecked Sendable {
    public let samples: [AudioTextSample]

    public init(samples: [AudioTextSample]) {
        self.samples = samples
    }

    public var count: Int {
        return samples.count
    }

    public subscript(index: Int) -> AudioTextSample {
        return samples[index]
    }

    /// テキスト中のひらがな・カタカナのみを音素化（漢字は読み推定せず pad）
    public static func extractFallbackPhonemeIds(text: String, phonemeVocabulary: PhonemeVocabulary = PhonemeVocabulary()) -> [Int] {
        var phonemeTokens: [Int] = []
        let converter = KanjiConverter(vocabulary: phonemeVocabulary)
        let normalized = converter.normalizeKana(text)

        for c in normalized {
            let scalarVal = c.unicodeScalars.first?.value ?? 0
            if 0x3041 <= scalarVal && scalarVal <= 0x3096 || c == "ー" {
                let pList = phonemeVocabulary.kanaToPhonemes(String(c))
                var pIdx = 0
                while pIdx < pList.count {
                    phonemeTokens.append(phonemeVocabulary.id(for: pList[pIdx]))
                    pIdx += 1
                }
            } else {
                phonemeTokens.append(PhonemeVocabulary.padId)
            }
        }
        return phonemeTokens
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
        phonemeVocabulary: PhonemeVocabulary = PhonemeVocabulary()
    ) throws -> SpeechDataset {
        let parser = WavParser()
        var sampleList: [AudioTextSample] = []
        var pIdx = 0

        while pIdx < pairs.count {
            let pair = pairs[pIdx]
            let wavData = try parser.parse(bytes: pair.wavBytes)
            let pcm16k = resampleTo16k(pcmData: wavData.pcmData, sampleRate: wavData.sampleRate)
            let textIds = textVocabulary.textToIds(pair.text)
            let phonemeIds = extractFallbackPhonemeIds(text: pair.text, phonemeVocabulary: phonemeVocabulary)
            let featuresSeq = extractFeaturesFromPCM(pcmData: pcm16k)

            if 0 < featuresSeq.count {
                sampleList.append(AudioTextSample(
                    audioPCM: pcm16k,
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

    /// PCM データとテキストのペア配列から直接構築
    public static func fromPCMPairs(
        pairs: [(pcmData: [Float], text: String)],
        textVocabulary: TextVocabulary,
        phonemeVocabulary: PhonemeVocabulary = PhonemeVocabulary()
    ) -> SpeechDataset {
        var sampleList: [AudioTextSample] = []
        var pIdx = 0

        while pIdx < pairs.count {
            let pair = pairs[pIdx]
            let textIds = textVocabulary.textToIds(pair.text)
            let phonemeIds = extractFallbackPhonemeIds(text: pair.text, phonemeVocabulary: phonemeVocabulary)
            let featuresSeq = extractFeaturesFromPCM(pcmData: pair.pcmData)

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
    public static func extractFeaturesFromPCM(pcmData: [Float]) -> [[Float]] {
        let totalSamples = pcmData.count
        if totalSamples < 400 {
            return []
        }

        // 1. PCM プリエンファシス (2-tap, coeff: 0.97)
        var preemph = [Float](repeating: 0.0, count: totalSamples)
        preemph[0] = pcmData[0]
        var pIdx = 1
        while pIdx < totalSamples {
            preemph[pIdx] = pcmData[pIdx] - (0.97 * pcmData[pIdx - 1])
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

        return featuresSeq
    }
}
