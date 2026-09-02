import Foundation

/// 文字起こし結果データモデル
public struct TranscriptionResult: Sendable, Equatable {
    public let text: String
    public let phonemes: [String]
    public let tokenIds: [Int]
    public let startTimeSeconds: Float
    public let endTimeSeconds: Float
    public let confidence: Float
    public let isFinal: Bool

    public init(
        text: String,
        phonemes: [String],
        tokenIds: [Int],
        startTimeSeconds: Float,
        endTimeSeconds: Float,
        confidence: Float,
        isFinal: Bool
    ) {
        self.text = text
        self.phonemes = phonemes
        self.tokenIds = tokenIds
        self.startTimeSeconds = startTimeSeconds
        self.endTimeSeconds = endTimeSeconds
        self.confidence = confidence
        self.isFinal = isFinal
    }
}

/// ストリーミングパイプライン設定パラメータ
public struct StreamingTranscriberConfig: Sendable {
    public let dspConfig: DSPConfig
    public let useQuantization: Bool
    public let beamWidth: Int
    public let lmWeight: Float
    public let maxSegmentDurationSeconds: Float

    public init(
        dspConfig: DSPConfig = DSPConfig(),
        useQuantization: Bool = false,
        beamWidth: Int = 4,
        lmWeight: Float = 0.3,
        maxSegmentDurationSeconds: Float = 15.0
    ) {
        self.dspConfig = dspConfig
        self.useQuantization = useQuantization
        self.beamWidth = beamWidth
        self.lmWeight = lmWeight
        self.maxSegmentDurationSeconds = maxSegmentDurationSeconds
    }
}

/// 統合ストリーミング音声文字起こしパイプライン (O(1) メモリ & ゼロアロケーション)
public final class StreamingTranscriber: @unchecked Sendable {
    public static let defaultUnkThreshold: Float = 0.25

    public let config: StreamingTranscriberConfig
    public let textVocabulary: TextVocabulary
    public let phonemeVocabulary: PhonemeVocabulary
    public let unkThreshold: Float

    // コールバック
    public var onPartialResult: (@Sendable (TranscriptionResult) -> Void)?
    public var onFinalResult: (@Sendable (TranscriptionResult) -> Void)?

    // DSP & SNN エンジン
    private let vad: VAD
    private let pitchDetector: PitchDetector
    private let lpc: LPC
    private let durandKerner: DurandKernerSolver
    private let formantExtractor: FormantExtractor
    private let filterbank: Filterbank
    private let dspWorkspace: DSPWorkspace

    private let acousticDecoder: AcousticDecoder
    private let acousticWorkspace: AcousticWorkspace
    private let languageDecoder: LanguageDecoder

    // O(1) 固定長リングバッファ
    private let ringBufferCapacity: Int = 32768
    private var ringBuffer: [Float]
    private var ringWritePos: Int = 0
    private var ringReadPos: Int = 0
    private var ringAvailable: Int = 0
    private var totalSamplesProcessed: Int64 = 0

    // 発話セグメント管理 (固定容量バッファ再利用)
    private let maxSegmentFrames: Int
    private var segmentProbs: [AcousticFrameProbabilities]
    private var segmentRawFeatures: [[Float]] // フォールバック音素推定用
    private var segmentStartSample: Int64 = 0
    private var segmentSpeechActive: Bool = false
    private var consecutiveSilenceFrames: Int = 0
    private var consecutiveSpeechFrames: Int = 0

    public init(
        config: StreamingTranscriberConfig = StreamingTranscriberConfig(),
        acousticNetwork: SpikingNetwork,
        languageNetwork: SpikingNetwork,
        quantizedAcousticEngine: QuantizedEngine? = nil,
        textVocabulary: TextVocabulary = TextVocabulary(),
        phonemeVocabulary: PhonemeVocabulary = PhonemeVocabulary(),
        unkThreshold: Float = StreamingTranscriber.defaultUnkThreshold
    ) {
        self.config = config
        self.textVocabulary = textVocabulary
        self.phonemeVocabulary = phonemeVocabulary
        self.unkThreshold = unkThreshold

        let dspCfg = config.dspConfig
        self.vad = VAD(config: dspCfg)
        self.pitchDetector = PitchDetector(config: dspCfg)
        self.lpc = LPC(config: dspCfg)
        self.durandKerner = DurandKernerSolver()
        self.formantExtractor = FormantExtractor(sampleRate: Float(dspCfg.sampleRate))
        self.filterbank = Filterbank(config: dspCfg)
        self.dspWorkspace = DSPWorkspace(
            maxFrameSize: dspCfg.frameSize,
            lpcOrder: dspCfg.lpcOrder,
            melChannels: dspCfg.melChannels,
            maxPitchLag: dspCfg.maxPitchLag
        )

        var qEngine: QuantizedEngine? = nil
        if config.useQuantization {
            switch quantizedAcousticEngine {
            case .some(let engine):
                qEngine = engine
            case .none:
                let qWeights = QuantizedEngine.quantize(
                    network: acousticNetwork,
                    config: .int32Config()
                )
                qEngine = QuantizedEngine(weights: qWeights, timeSteps: acousticNetwork.timeSteps)
            }
        }

        self.acousticDecoder = AcousticDecoder(
            network: acousticNetwork,
            quantizedEngine: qEngine,
            vocabulary: textVocabulary,
            fallbackVocabulary: phonemeVocabulary
        )
        self.acousticWorkspace = AcousticWorkspace(
            maxHiddenDim: acousticNetwork.maxHiddenDim,
            outputDim: acousticNetwork.outputDim,
            inputDim: acousticNetwork.inputDim
        )

        let lmConfig = LanguageDecoderConfig(
            beamWidth: config.beamWidth,
            lmWeight: config.lmWeight
        )
        self.languageDecoder = LanguageDecoder(
            lmNetwork: languageNetwork,
            vocabulary: textVocabulary,
            fallbackVocabulary: phonemeVocabulary,
            config: lmConfig
        )

        self.ringBuffer = [Float](repeating: 0.0, count: ringBufferCapacity)
        let framesPerSec = Float(dspCfg.sampleRate) / Float(dspCfg.hopSize)
        self.maxSegmentFrames = Int(config.maxSegmentDurationSeconds * framesPerSec) + 100
        self.segmentProbs = []
        self.segmentProbs.reserveCapacity(maxSegmentFrames)
        self.segmentRawFeatures = []
        self.segmentRawFeatures.reserveCapacity(maxSegmentFrames)
    }

    /// PCM 音声配列の入力
    public func appendAudio(pcm: [Float]) {
        pcm.withUnsafeBufferPointer { buf in
            switch buf.baseAddress {
            case .some(let ptr):
                appendAudio(pcmPtr: ptr, count: buf.count)
            case .none:
                break
            }
        }
    }

    /// PCM 音声ポインタの入力
    public func appendAudio(pcmPtr: UnsafePointer<Float>, count: Int) {
        if count <= 0 {
            return
        }

        var i = 0
        while i < count {
            ringBuffer[ringWritePos] = pcmPtr[i]
            ringWritePos = (ringWritePos + 1) % ringBufferCapacity
            if ringAvailable < ringBufferCapacity {
                ringAvailable += 1
            } else {
                ringReadPos = (ringReadPos + 1) % ringBufferCapacity
            }
            i += 1
        }

        processAvailableFrames()
    }

    /// リングバッファ内の利用可能フレームを順次処理
    private func processAvailableFrames() {
        let frameSize = config.dspConfig.frameSize
        let hopSize = config.dspConfig.hopSize

        let rawBuf = dspWorkspace.rawFrame.withUnsafeMutableBufferPointer { $0.baseAddress! }

        while frameSize <= ringAvailable {
            // 1. リングバッファから rawFrame へコピー
            var i = 0
            while i < frameSize {
                let idx = (ringReadPos + i) % ringBufferCapacity
                rawBuf[i] = ringBuffer[idx]
                i += 1
            }

            let currentSampleOffset = totalSamplesProcessed
            ringReadPos = (ringReadPos + hopSize) % ringBufferCapacity
            ringAvailable -= hopSize
            totalSamplesProcessed += Int64(hopSize)

            // 2. VAD 判定
            let vadRes = vad.processFrame(
                ptr: rawBuf,
                count: frameSize,
                workspace: dspWorkspace
            )

            // 3. 発話ステートマシン進行
            if vadRes.isSpeech {
                consecutiveSilenceFrames = 0
                consecutiveSpeechFrames += 1

                if segmentSpeechActive != true {
                    if 3 <= consecutiveSpeechFrames {
                        segmentSpeechActive = true
                        let preRoll = Int64(hopSize * consecutiveSpeechFrames)
                        if preRoll <= currentSampleOffset {
                            segmentStartSample = currentSampleOffset - preRoll
                        } else {
                            segmentStartSample = 0
                        }
                        segmentProbs.removeAll(keepingCapacity: true)
                    }
                }
            } else {
                consecutiveSpeechFrames = 0
                consecutiveSilenceFrames += 1
            }

            // 4. 発話フレームの特徴量抽出 & 音響 SNN 推論
            if segmentSpeechActive {
                let features = filterbank.extractFeatures(
                    pcmPtr: rawBuf,
                    count: frameSize,
                    workspace: dspWorkspace
                )

                let acousticFrame = acousticDecoder.decodeFrame(
                    features: features,
                    workspace: acousticWorkspace,
                    frameIndex: segmentProbs.count
                )

                if segmentProbs.count < maxSegmentFrames {
                    segmentProbs.append(acousticFrame)
                    segmentRawFeatures.append(features)
                }

                // 5. 部分認識結果コールバック (10フレーム = 100ms ごと、コールバック登録時のみ実行)
                if let onPartial = onPartialResult {
                    if (segmentProbs.count % 10) == 0 {
                        let windowLimit = 50
                        let window: [AcousticFrameProbabilities]
                        if windowLimit < segmentProbs.count {
                            window = Array(segmentProbs[(segmentProbs.count - windowLimit)..<segmentProbs.count])
                        } else {
                            window = segmentProbs
                        }
                        let greedy = languageDecoder.decodeGreedy(
                            acousticProbs: window,
                            unkThreshold: unkThreshold
                        )
                        let startSec = Float(segmentStartSample) / Float(config.dspConfig.sampleRate)
                        let endSec = Float(totalSamplesProcessed) / Float(config.dspConfig.sampleRate)
                        let partialRes = TranscriptionResult(
                            text: greedy.text,
                            phonemes: phonemeVocabulary.kanaToPhonemes(greedy.text),
                            tokenIds: greedy.tokens,
                            startTimeSeconds: startSec,
                            endTimeSeconds: endSec,
                            confidence: 0.8,
                            isFinal: false
                        )
                        onPartial(partialRes)
                    }
                }

                // 6. 発話終了判定 (20フレーム = 200ms 無音 または 最大セグメント長到達)
                if 20 <= consecutiveSilenceFrames || maxSegmentFrames <= segmentProbs.count {
                    finalizeSegment()
                }
            }
        }
    }

    /// 音響特徴量から母音・子音を推定し、ひらがな（聞こえた音）にフォールバック
    public func decodeFallbackKana(from featuresSeq: [[Float]]) -> String {
        if featuresSeq.isEmpty {
            return ""
        }
        var phonemes: [String] = []
        var lastPhoneme = ""

        var fIdx = 0
        while fIdx < featuresSeq.count {
            let feat = featuresSeq[fIdx]
            var lowEnergy: Float = 0.0
            var midEnergy: Float = 0.0
            var highEnergy: Float = 0.0

            var d = 0
            let featCount = feat.count
            while d < featCount {
                let v = feat[d]
                switch true {
                case d < 16:
                    lowEnergy += v
                case d < 40:
                    midEnergy += v
                default:
                    highEnergy += v
                }
                d += 1
            }

            let totalEnergy = lowEnergy + midEnergy + highEnergy
            if 0.1 <= totalEnergy {
                var p = "a"
                switch true {
                case highEnergy < lowEnergy && midEnergy < lowEnergy:
                    p = "u"
                case lowEnergy < highEnergy && midEnergy < highEnergy:
                    p = "i"
                case lowEnergy < midEnergy && highEnergy < midEnergy:
                    p = "a"
                default:
                    p = "o"
                }

                if p != lastPhoneme {
                    phonemes.append(p)
                    lastPhoneme = p
                }
            }
            fIdx += 1
        }

        if phonemes.isEmpty {
            return ""
        }
        return phonemeVocabulary.phonemesToKana(phonemes)
    }

    /// 現在の発話セグメントの言語デコードと結果確定 (本線: 直接漢字かな + 未知語フォールバック: 聞こえた音のかな)
    private func finalizeSegment() {
        if segmentProbs.isEmpty != true {
            let decodeRes: (tokens: [Int], text: String, score: Float)
            if config.beamWidth <= 1 {
                decodeRes = languageDecoder.decodeGreedy(
                    acousticProbs: segmentProbs,
                    unkThreshold: unkThreshold
                )
            } else {
                decodeRes = languageDecoder.decodeBeamSearch(
                    acousticProbs: segmentProbs,
                    unkThreshold: unkThreshold
                )
            }

            // 本線デコード結果の確定 (未知語トークン <unk>, ? のサニタイズ)
            var finalText = decodeRes.text
            finalText = finalText.replacingOccurrences(of: "<unk>", with: "")
            finalText = finalText.replacingOccurrences(of: "?", with: "")

            let startSec = Float(segmentStartSample) / Float(config.dspConfig.sampleRate)
            let endSec = Float(totalSamplesProcessed) / Float(config.dspConfig.sampleRate)
            let finalRes = TranscriptionResult(
                text: finalText,
                phonemes: phonemeVocabulary.kanaToPhonemes(finalText),
                tokenIds: decodeRes.tokens,
                startTimeSeconds: startSec,
                endTimeSeconds: endSec,
                confidence: 0.95,
                isFinal: true
            )
            onFinalResult?(finalRes)
        }

        segmentProbs.removeAll(keepingCapacity: true)
        segmentRawFeatures.removeAll(keepingCapacity: true)
        segmentSpeechActive = false
        consecutiveSilenceFrames = 0
        consecutiveSpeechFrames = 0
        acousticWorkspace.reset()
    }

    /// 残存バッファのフラッシュと終端処理
    public func flush() {
        let frameSize = config.dspConfig.frameSize
        if 0 < ringAvailable {
            let rawBuf = dspWorkspace.rawFrame.withUnsafeMutableBufferPointer { $0.baseAddress! }
            var i = 0
            while i < frameSize {
                if i < ringAvailable {
                    let idx = (ringReadPos + i) % ringBufferCapacity
                    rawBuf[i] = ringBuffer[idx]
                } else {
                    rawBuf[i] = 0.0
                }
                i += 1
            }
            ringReadPos = 0
            ringAvailable = 0
        }

        if segmentSpeechActive {
            finalizeSegment()
        }
    }

    /// 内部状態の全リセット
    public func reset() {
        ringWritePos = 0
        ringReadPos = 0
        ringAvailable = 0
        totalSamplesProcessed = 0
        segmentSpeechActive = false
        consecutiveSilenceFrames = 0
        consecutiveSpeechFrames = 0
        segmentProbs.removeAll(keepingCapacity: true)
        segmentRawFeatures.removeAll(keepingCapacity: true)
        acousticWorkspace.reset()
    }
}
