import Foundation

/// 音響 SNN と言語 SNN の推論入口。学習は MLXBPTTTrainer が行い、重みをここへ転送する。
public final class Trainer: @unchecked Sendable {
    public let acousticNetwork: SpikingNetwork
    public let languageNetwork: SpikingNetwork
    public let textVocabulary: TextVocabulary
    public let phonemeVocabulary: PhonemeVocabulary

    public init(
        acousticNetwork: SpikingNetwork,
        languageNetwork: SpikingNetwork,
        textVocabulary: TextVocabulary,
        phonemeVocabulary: PhonemeVocabulary = PhonemeVocabulary()
    ) {
        self.textVocabulary = textVocabulary
        self.phonemeVocabulary = phonemeVocabulary
        self.acousticNetwork = acousticNetwork
        self.languageNetwork = languageNetwork
    }

    /// 音響入力は `SpeechDataset.acousticInputDim()`、言語入力は 128 次元のトークン埋め込み。
    public static func makeDefault(
        textVocabulary: TextVocabulary,
        phonemeVocabulary: PhonemeVocabulary = PhonemeVocabulary()
    ) -> Trainer {
        let acNet = SpikingNetwork(
            inputDim: SpeechDataset.acousticInputDim(),
            maxHiddenDim: 1024,
            outputDim: textVocabulary.size,
            timeSteps: 4
        )
        let lmNet = SpikingNetwork(
            inputDim: 128,
            maxHiddenDim: 1024,
            outputDim: textVocabulary.size,
            timeSteps: 4
        )
        return Trainer(
            acousticNetwork: acNet,
            languageNetwork: lmNet,
            textVocabulary: textVocabulary,
            phonemeVocabulary: phonemeVocabulary
        )
    }
}

/// 推論実行精度モード
public enum ExecutionPrecision: String, Sendable, CaseIterable {
    case float32 = "Float32"
    case float16 = "Float16"
    case int32   = "Int32"
    case int16   = "Int16"
}

extension Trainer {
    private func acousticFrameStack() -> Int {
        return StreamingFeatureFrontEnd.frameStack(forInputDim: acousticNetwork.inputDim)
    }

    private func acousticLongContext() -> Bool {
        return StreamingFeatureFrontEnd.hasLongContext(inputDim: acousticNetwork.inputDim)
    }

    /// PCM から音響 greedy でテキストを返す。既定の LanguageDecoder は lmWeight=0。
    public func transcribe(
        pcmData: [Float],
        precision: ExecutionPrecision = .float32
    ) -> String {
        let featuresSeq = SpeechDataset.extractFeaturesFromPCM(
            pcmData: pcmData,
            frameStack: acousticFrameStack(),
            longContext: acousticLongContext()
        )
        if featuresSeq.isEmpty {
            return ""
        }

        var qEngine: QuantizedEngine? = nil
        switch precision {
        case .float32:
            break
        case .float16:
            let qWeights = QuantizedEngine.quantize(
                network: acousticNetwork,
                config: .float16Config()
            )
            qEngine = QuantizedEngine(weights: qWeights, timeSteps: acousticNetwork.timeSteps)
        case .int32:
            let qWeights = QuantizedEngine.quantize(
                network: acousticNetwork,
                config: .int32Config()
            )
            qEngine = QuantizedEngine(weights: qWeights, timeSteps: acousticNetwork.timeSteps)
        case .int16:
            let qWeights = QuantizedEngine.quantize(
                network: acousticNetwork,
                config: .int16Config()
            )
            qEngine = QuantizedEngine(weights: qWeights, timeSteps: acousticNetwork.timeSteps)
        }

        let acDecoder = AcousticDecoder(
            network: acousticNetwork,
            quantizedEngine: qEngine
        )
        let acWorkspace = AcousticWorkspace(
            maxHiddenDim: acousticNetwork.maxHiddenDim,
            outputDim: acousticNetwork.outputDim,
            inputDim: acousticNetwork.inputDim,
            numLayers: acousticNetwork.numLayers
        )

        let lmDecoder = LanguageDecoder(
            lmNetwork: languageNetwork,
            vocabulary: textVocabulary
        )

        var acousticProbs: [AcousticFrameProbabilities] = []
        acousticProbs.reserveCapacity(featuresSeq.count)

        var f = 0
        while f < featuresSeq.count {
            let feat = featuresSeq[f]
            let frame = acDecoder.decodeFrame(
                features: feat,
                workspace: acWorkspace,
                frameIndex: f
            )
            acousticProbs.append(frame)
            f += 1
        }

        let greedyRes = lmDecoder.decodeGreedy(
            acousticProbs: acousticProbs
        )

        return greedyRes.text
    }

    /// 音響 SNN のみ。低信頼度は pad、短い pad を挟んだ同一文字をマージし、最短フレーム未満を落とす。
    public func transcribeAcousticDirect(
        pcmData: [Float],
        minDurationFrames: Int = 3,
        minConfidence: Float = 0.45
    ) -> String {
        let pcm16k = SpeechDataset.resampleTo16k(pcmData: pcmData, sampleRate: 16000)
        let featuresSeq = SpeechDataset.extractFeaturesFromPCM(
            pcmData: pcm16k,
            frameStack: acousticFrameStack(),
            longContext: acousticLongContext()
        )
        return transcribeAcousticDirect(
            featuresSeq: featuresSeq,
            minDurationFrames: minDurationFrames,
            minConfidence: minConfidence
        )
    }

    public func transcribeAcousticDirect(
        featuresSeq: [[Float]],
        minDurationFrames: Int = 3,
        minConfidence: Float = 0.45
    ) -> String {
        if featuresSeq.isEmpty {
            return ""
        }

        let acDecoder = AcousticDecoder(
            network: acousticNetwork
        )
        let acWorkspace = AcousticWorkspace(
            maxHiddenDim: acousticNetwork.maxHiddenDim,
            outputDim: acousticNetwork.outputDim,
            inputDim: acousticNetwork.inputDim,
            numLayers: acousticNetwork.numLayers
        )

        let frameProbs = acDecoder.decodeSequence(
            featuresSeq: featuresSeq,
            workspace: acWorkspace
        )
        var rawTokens: [Int] = []
        rawTokens.reserveCapacity(featuresSeq.count)

        var f = 0
        while f < featuresSeq.count {
            let frame = frameProbs[f]
            // 1. 低信頼度フレームは pad として扱う
            if frame.topProbability < minConfidence {
                rawTokens.append(TextVocabulary.padId)
            } else {
                rawTokens.append(frame.topTokenId)
            }
            f += 1
        }

        // 2. ランレングス抽出
        struct Segment {
            var token: Int
            var count: Int
        }
        var segments: [Segment] = []
        if 0 < rawTokens.count {
            var currToken = rawTokens[0]
            var currCount = 1
            var i = 1
            while i < rawTokens.count {
                let tok = rawTokens[i]
                if tok == currToken {
                    currCount += 1
                } else {
                    segments.append(Segment(token: currToken, count: currCount))
                    currToken = tok
                    currCount = 1
                }
                i += 1
            }
            segments.append(Segment(token: currToken, count: currCount))
        }

        // 3. 短い pad / 制御トークン (<= 2フレーム) を挟んだ同一文字のマージ
        var mergedSegments: [Segment] = []
        var sIdx = 0
        while sIdx < segments.count {
            let seg = segments[sIdx]
            if seg.token < 4 && seg.count <= 2 {
                // 前後が同じ文字か確認
                if 0 < mergedSegments.count && (sIdx + 1) < segments.count {
                    let prevToken = mergedSegments[mergedSegments.count - 1].token
                    let nextSeg = segments[sIdx + 1]
                    if 4 <= prevToken && prevToken == nextSeg.token {
                        // マージして同一文字の継続とする
                        mergedSegments[mergedSegments.count - 1].count += nextSeg.count
                        sIdx += 2
                        continue
                    }
                }
            }
            mergedSegments.append(seg)
            sIdx += 1
        }

        // 4. 有効文字の抽出 (4 <= token かつ minDurationFrames 以上, 連続重複除外)
        var collapsedTokens: [Int] = []
        var lastEmittedToken = TextVocabulary.padId
        var mIdx = 0
        while mIdx < mergedSegments.count {
            let mSeg = mergedSegments[mIdx]
            if 4 <= mSeg.token && minDurationFrames <= mSeg.count {
                if mSeg.token != lastEmittedToken {
                    collapsedTokens.append(mSeg.token)
                    lastEmittedToken = mSeg.token
                }
            }
            mIdx += 1
        }

        return textVocabulary.idsToText(collapsedTokens)
    }

    /// `AcousticDecoder` のフレーム対数確率を CTC ビームで文字列にする。
    public func transcribeAcousticCTC(
        featuresSeq: [[Float]],
        beamWidth: Int = 16,
        blankPenalty: Float = 0.0
    ) -> String {
        if featuresSeq.isEmpty {
            return ""
        }

        let network = acousticNetwork
        let acDecoder = AcousticDecoder(
            network: network
        )
        let acWorkspace = AcousticWorkspace(
            maxHiddenDim: network.maxHiddenDim,
            outputDim: network.outputDim,
            inputDim: network.inputDim,
            numLayers: network.numLayers
        )

        let frameProbs = acDecoder.decodeSequence(
            featuresSeq: featuresSeq,
            workspace: acWorkspace
        )

        let outDim = network.outputDim
        var logProbs = [[Float]](
            repeating: [Float](repeating: 0.0, count: outDim),
            count: frameProbs.count
        )
        // ブランク割引: 疎なスパイクでは不確実フレームがブランクに寄りやすく
        // 文字脱落を招くため、ブランク確率を一定率で割り引く
        let blankDiscount = max(0.01, 1.0 - blankPenalty)
        var f = 0
        while f < frameProbs.count {
            let probs = frameProbs[f].probabilities
            var c = 0
            while c < outDim {
                var prob = probs[c]
                if c == TextVocabulary.padId && 0.0 < blankPenalty {
                    prob *= blankDiscount
                }
                logProbs[f][c] = log(max(1e-30, prob))
                c += 1
            }
            f += 1
        }

        let decoder = CTCBeamDecoder(
            vocabulary: textVocabulary,
            blankId: TextVocabulary.padId,
            beamWidth: beamWidth
        )
        return decoder.decode(logProbs: logProbs).text
    }

    /// 第1段でかな、第2段で漢字かな混じり。`useCTC` なら第1段はビーム CTC。
    public func transcribeTwoStage(
        featuresSeq: [[Float]],
        kanjiVocabulary: TextVocabulary,
        dictionary: KanaKanjiDictionary? = nil,
        minDurationFrames: Int = 3,
        minConfidence: Float = 0.05,
        useCTC: Bool = false,
        languageBonus: Float = 4.0,
        blankPenalty: Float = 0.0
    ) -> (kana: String, kanji: String) {
        let kanaText: String
        if useCTC {
            kanaText = transcribeAcousticCTC(featuresSeq: featuresSeq, beamWidth: 16, blankPenalty: blankPenalty)
        } else {
            kanaText = transcribeAcousticDirect(
                featuresSeq: featuresSeq,
                minDurationFrames: minDurationFrames,
                minConfidence: minConfidence
            )
        }

        var kanjiText = ""
        switch dictionary {
        case .some(let dict):
            // 辞書 Viterbi DP に、学習済み第2段 言語 SNN の予測も手掛かりとして与える
            let lmDecoder = LanguageDecoder(
                lmNetwork: languageNetwork,
                vocabulary: kanjiVocabulary
            )
            let decoder = KanaKanjiDecoder(
                dictionary: dict,
                languageDecoder: lmDecoder,
                kanaVocabulary: textVocabulary,
                languageBonus: languageBonus
            )
            kanjiText = decoder.decode(kanaText: kanaText)
        case .none:
            let decoder = LanguageDecoder(
                lmNetwork: languageNetwork,
                vocabulary: kanjiVocabulary
            )
            kanjiText = decoder.decodeKanaToKanji(
                kanaText: kanaText,
                kanaVocabulary: textVocabulary
            )
        }

        return (kana: kanaText, kanji: kanjiText)
    }
}
