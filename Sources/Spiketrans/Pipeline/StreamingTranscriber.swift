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
    public let enableFillerRemoval: Bool
    public let fillerMode: FillerFilterMode
    public let enableITN: Bool
    public let enableParagraphSegmentation: Bool
    public let paragraphPauseThreshold: Float

    public init(
        dspConfig: DSPConfig = DSPConfig(),
        useQuantization: Bool = false,
        beamWidth: Int = 4,
        lmWeight: Float = 0.3,
        maxSegmentDurationSeconds: Float = 15.0,
        enableFillerRemoval: Bool = false,
        fillerMode: FillerFilterMode = .remove,
        enableITN: Bool = false,
        enableParagraphSegmentation: Bool = false,
        paragraphPauseThreshold: Float = 1.2
    ) {
        self.dspConfig = dspConfig
        self.useQuantization = useQuantization
        self.beamWidth = beamWidth
        self.lmWeight = lmWeight
        self.maxSegmentDurationSeconds = maxSegmentDurationSeconds
        self.enableFillerRemoval = enableFillerRemoval
        self.fillerMode = fillerMode
        self.enableITN = enableITN
        self.enableParagraphSegmentation = enableParagraphSegmentation
        self.paragraphPauseThreshold = paragraphPauseThreshold
    }
}

/// ストリーミング文字起こし。VAD で発話を切り、音響デコードして後処理する。
public final class StreamingTranscriber: @unchecked Sendable {
    public let config: StreamingTranscriberConfig
    public let textVocabulary: TextVocabulary
    public let phonemeVocabulary: PhonemeVocabulary

    // コールバック
    public var onPartialResult: (@Sendable (TranscriptionResult) -> Void)?
    public var onFinalResult: (@Sendable (TranscriptionResult) -> Void)?
    public var onParagraphResult: (@Sendable (ParagraphSegment) -> Void)?

    private let vad: VAD
    private let dspWorkspace: DSPWorkspace

    private let acousticDecoder: AcousticDecoder
    private let acousticWorkspace: AcousticWorkspace
    private let languageDecoder: LanguageDecoder

    // テキスト後処理エンジン (フィラー除去・ITN・段落分け)
    private let fillerFilter: FillerWordFilter
    private let normalizer: InverseTextNormalizer
    private let paragraphSegmenter: ParagraphSegmenter

    // O(1) 固定長リングバッファ
    private let ringBufferCapacity: Int = 32768
    private var ringBuffer: [Float]
    private var ringWritePos: Int = 0
    private var ringReadPos: Int = 0
    private var ringAvailable: Int = 0
    private var totalSamplesProcessed: Int64 = 0

    // 発話セグメント管理 (固定容量バッファ再利用)
    private let maxSegmentHops: Int
    private var segmentHopCount: Int = 0
    private var segmentProbs: [AcousticFrameProbabilities]
    private var segmentRawFeatures: [[Float]] // 本線が空のとき fallbackKana に渡す
    private var segmentStartSample: Int64 = 0
    private var segmentSpeechActive: Bool = false
    private var consecutiveSilenceFrames: Int = 0
    private var consecutiveSpeechFrames: Int = 0
    private let featureFrontEnd: StreamingFeatureFrontEnd
    private let partialEveryStacked: Int

    public init(
        config: StreamingTranscriberConfig = StreamingTranscriberConfig(),
        acousticNetwork: SpikingNetwork,
        languageNetwork: SpikingNetwork,
        quantizedAcousticEngine: QuantizedEngine? = nil,
        textVocabulary: TextVocabulary = TextVocabulary(),
        phonemeVocabulary: PhonemeVocabulary = PhonemeVocabulary()
    ) {
        self.config = config
        self.textVocabulary = textVocabulary
        self.phonemeVocabulary = phonemeVocabulary

        let dspCfg = config.dspConfig
        self.vad = VAD(config: dspCfg)
        self.dspWorkspace = DSPWorkspace(
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
            quantizedEngine: qEngine
        )
        self.acousticWorkspace = AcousticWorkspace(
            maxHiddenDim: acousticNetwork.maxHiddenDim,
            outputDim: acousticNetwork.outputDim,
            inputDim: acousticNetwork.inputDim,
            numLayers: acousticNetwork.numLayers
        )

        let lmConfig = LanguageDecoderConfig(
            beamWidth: config.beamWidth,
            lmWeight: config.lmWeight
        )
        self.languageDecoder = LanguageDecoder(
            lmNetwork: languageNetwork,
            vocabulary: textVocabulary,
            config: lmConfig
        )

        self.fillerFilter = FillerWordFilter(mode: config.fillerMode)
        self.normalizer = InverseTextNormalizer()
        self.paragraphSegmenter = ParagraphSegmenter(pauseThresholdSeconds: config.paragraphPauseThreshold)

        self.ringBuffer = [Float](repeating: 0.0, count: ringBufferCapacity)
        let framesPerSec = Float(dspCfg.sampleRate) / Float(dspCfg.hopSize)
        self.maxSegmentHops = Int(config.maxSegmentDurationSeconds * framesPerSec) + 100
        var stack = acousticNetwork.inputDim / StreamingFeatureFrontEnd.tapDim
        if stack < 1 {
            stack = 1
        }
        let stackedCap = (maxSegmentHops / stack) + 4
        self.segmentProbs = []
        self.segmentProbs.reserveCapacity(stackedCap)
        self.segmentRawFeatures = []
        self.segmentRawFeatures.reserveCapacity(stackedCap)
        self.featureFrontEnd = StreamingFeatureFrontEnd(
            frameStack: stack,
            dspConfig: dspCfg
        )
        // 束ね前は 10 ホップごと (100 ms)。束ね後も同じ実時間。
        let partialEvery = 10 / stack
        if partialEvery < 1 {
            self.partialEveryStacked = 1
        } else {
            self.partialEveryStacked = partialEvery
        }
    }

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

    private func processAvailableFrames() {
        let frameSize = config.dspConfig.frameSize
        let hopSize = config.dspConfig.hopSize

        let rawBuf = dspWorkspace.rawFrame.withUnsafeMutableBufferPointer { $0.baseAddress! }

        while frameSize <= ringAvailable {
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

            // セグメンテーション VAD は raw。エネルギー閾値を screen/segment と揃える。
            let vadRes = vad.processFrame(
                ptr: rawBuf,
                count: frameSize,
                workspace: dspWorkspace
            )

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
                        segmentRawFeatures.removeAll(keepingCapacity: true)
                        segmentHopCount = 0
                        featureFrontEnd.beginUtterance()
                    }
                }
            } else {
                consecutiveSpeechFrames = 0
                consecutiveSilenceFrames += 1
            }

            if segmentSpeechActive {
                segmentHopCount += 1
                if let features = featureFrontEnd.pushRawFrame(pcmPtr: rawBuf, count: frameSize) {
                    ingestStackedFeatures(features)
                }

                if 20 <= consecutiveSilenceFrames || maxSegmentHops <= segmentHopCount {
                    finalizeSegment()
                }
            }
        }
    }

    private func ingestStackedFeatures(_ features: [Float]) {
        let acousticFrame = acousticDecoder.decodeFrame(
            features: features,
            workspace: acousticWorkspace,
            frameIndex: segmentProbs.count
        )
        segmentProbs.append(acousticFrame)
        segmentRawFeatures.append(features)

        if let onPartial = onPartialResult {
            if (segmentProbs.count % partialEveryStacked) == 0 {
                let windowLimit = 50
                let window: [AcousticFrameProbabilities]
                if windowLimit < segmentProbs.count {
                    window = Array(segmentProbs[(segmentProbs.count - windowLimit)..<segmentProbs.count])
                } else {
                    window = segmentProbs
                }
                let greedy = languageDecoder.decodeGreedy(
                    acousticProbs: window
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
    }

    public func decodeFallbackKana(from featuresSeq: [[Float]]) -> String {
        return phonemeVocabulary.fallbackKana(fromFeatureFrames: featuresSeq)
    }

    /// 発話を確定する。本線が空なら `decodeFallbackKana`。
    private func finalizeSegment() {
        if let flushed = featureFrontEnd.flush() {
            ingestStackedFeatures(flushed)
        }
        if segmentProbs.isEmpty != true {
            let decodeRes: (tokens: [Int], text: String, score: Float)
            if config.beamWidth <= 1 {
                decodeRes = languageDecoder.decodeGreedy(
                    acousticProbs: segmentProbs
                )
            } else {
                decodeRes = languageDecoder.decodeBeamSearch(
                    acousticProbs: segmentProbs
                )
            }

            var finalText = decodeRes.text
            finalText = finalText.replacingOccurrences(of: "<unk>", with: "")
            finalText = finalText.replacingOccurrences(of: "?", with: "")

            if finalText.isEmpty {
                finalText = decodeFallbackKana(from: segmentRawFeatures)
            }

            if config.enableFillerRemoval {
                finalText = fillerFilter.filter(finalText)
            }
            if config.enableITN {
                finalText = normalizer.normalize(finalText)
            }

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

            if config.enableParagraphSegmentation {
                if let paragraph = paragraphSegmenter.appendSegment(
                    text: finalText,
                    startSeconds: startSec,
                    endSeconds: endSec
                ) {
                    onParagraphResult?(paragraph)
                }
            }
        }

        segmentProbs.removeAll(keepingCapacity: true)
        segmentRawFeatures.removeAll(keepingCapacity: true)
        segmentHopCount = 0
        segmentSpeechActive = false
        consecutiveSilenceFrames = 0
        consecutiveSpeechFrames = 0
        acousticWorkspace.reset()
        featureFrontEnd.beginUtterance()
    }

    /// 端数ホップ (frameSize 未満) は捨て、開いている発話を確定する。
    public func flush() {
        ringReadPos = 0
        ringAvailable = 0

        if segmentSpeechActive {
            finalizeSegment()
        }

        if config.enableParagraphSegmentation {
            if let finalParagraph = paragraphSegmenter.flush() {
                onParagraphResult?(finalParagraph)
            }
        }
    }

    public func reset() {
        ringWritePos = 0
        ringReadPos = 0
        ringAvailable = 0
        totalSamplesProcessed = 0
        segmentSpeechActive = false
        consecutiveSilenceFrames = 0
        consecutiveSpeechFrames = 0
        segmentHopCount = 0
        segmentProbs.removeAll(keepingCapacity: true)
        segmentRawFeatures.removeAll(keepingCapacity: true)
        acousticWorkspace.reset()
        paragraphSegmenter.reset()
        featureFrontEnd.beginUtterance()
    }
}
