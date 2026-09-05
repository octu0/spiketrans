import Foundation

/// 統合学習サマリー
public struct TrainingSummary: Sendable {
    public let acousticEpochs: [EpochResult]
    public let languageEpochs: [EpochResult]
    public let finalAcousticLoss: Float
    public let finalLanguageLoss: Float

    public init(
        acousticEpochs: [EpochResult],
        languageEpochs: [EpochResult],
        finalAcousticLoss: Float,
        finalLanguageLoss: Float
    ) {
        self.acousticEpochs = acousticEpochs
        self.languageEpochs = languageEpochs
        self.finalAcousticLoss = finalAcousticLoss
        self.finalLanguageLoss = finalLanguageLoss
    }
}

/// 第1段 音響 SNN と第2段 言語 SNN を束ねる学習・推論の基盤
public final class Trainer: @unchecked Sendable {
    public let acousticTrainer: AcousticTrainer
    public let languageTrainer: LanguageTrainer
    public let textVocabulary: TextVocabulary
    public let phonemeVocabulary: PhonemeVocabulary

    public init(
        acousticNetwork: SpikingNetwork,
        languageNetwork: SpikingNetwork,
        textVocabulary: TextVocabulary,
        phonemeVocabulary: PhonemeVocabulary = PhonemeVocabulary(),
        config: TrainingConfig = TrainingConfig()
    ) {
        self.textVocabulary = textVocabulary
        self.phonemeVocabulary = phonemeVocabulary
        self.acousticTrainer = AcousticTrainer(network: acousticNetwork, config: config)
        self.languageTrainer = LanguageTrainer(
            network: languageNetwork,
            textVocabulary: textVocabulary,
            config: config
        )
    }

    /// デフォルトネットワーク構成で簡単に初期化するファクトリ (音響・言語とも直接TextVocabularyを出力)
    public static func makeDefault(
        textVocabulary: TextVocabulary,
        phonemeVocabulary: PhonemeVocabulary = PhonemeVocabulary(),
        config: TrainingConfig = TrainingConfig()
    ) -> Trainer {
        let acNet = SpikingNetwork(
            inputDim: 128,
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
            phonemeVocabulary: phonemeVocabulary,
            config: config
        )
    }

    /// データセットを用いた音響・言語両モデルの統合並列学習
    public func fit(dataset: SpeechDataset, numWorkers: Int = 1) -> TrainingSummary {
        let acResults = acousticTrainer.train(dataset: dataset, numWorkers: numWorkers)
        let lmResults = languageTrainer.train(dataset: dataset, numWorkers: numWorkers)

        var finalAcLoss: Float = 0.0
        if 0 < acResults.count {
            finalAcLoss = acResults[acResults.count - 1].totalLoss
        }

        var finalLmLoss: Float = 0.0
        if 0 < lmResults.count {
            finalLmLoss = lmResults[lmResults.count - 1].totalLoss
        }

        return TrainingSummary(
            acousticEpochs: acResults,
            languageEpochs: lmResults,
            finalAcousticLoss: finalAcLoss,
            finalLanguageLoss: finalLmLoss
        )
    }

}

/// 推論実行精度モード
public enum ExecutionPrecision: String, Sendable, CaseIterable {
    case float32 = "Float32"
    case int32   = "Int32"
    case int16   = "Int16"
}

extension Trainer {
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

    /// PCM 音声から直接漢字・かなテキストを文字起こし (本線: TextVocabulary + Language SNN 自己回帰 + 未知語フォールバック)
    public func transcribe(
        pcmData: [Float],
        precision: ExecutionPrecision = .float32,
        unkThreshold: Float = 0.25
    ) -> String {
        let featuresSeq = SpeechDataset.extractFeaturesFromPCM(pcmData: pcmData)
        if featuresSeq.isEmpty {
            return ""
        }

        var qEngine: QuantizedEngine? = nil
        switch precision {
        case .float32:
            break
        case .int32:
            let qWeights = QuantizedEngine.quantize(
                network: acousticTrainer.network,
                config: .int32Config()
            )
            qEngine = QuantizedEngine(weights: qWeights, timeSteps: acousticTrainer.network.timeSteps)
        case .int16:
            let qWeights = QuantizedEngine.quantize(
                network: acousticTrainer.network,
                config: .int16Config()
            )
            qEngine = QuantizedEngine(weights: qWeights, timeSteps: acousticTrainer.network.timeSteps)
        }

        let acDecoder = AcousticDecoder(
            network: acousticTrainer.network,
            quantizedEngine: qEngine,
            vocabulary: textVocabulary,
            fallbackVocabulary: phonemeVocabulary
        )
        let acWorkspace = AcousticWorkspace(
            maxHiddenDim: acousticTrainer.network.maxHiddenDim,
            outputDim: acousticTrainer.network.outputDim,
            inputDim: acousticTrainer.network.inputDim,
            numLayers: acousticTrainer.network.numLayers
        )

        let lmDecoder = LanguageDecoder(
            lmNetwork: languageTrainer.network,
            vocabulary: textVocabulary,
            fallbackVocabulary: phonemeVocabulary
        )

        let acousticProbs = acDecoder.decodeSequence(
            featuresSeq: featuresSeq,
            workspace: acWorkspace
        )

        let greedyRes = lmDecoder.decodeGreedy(
            acousticProbs: acousticProbs,
            unkThreshold: unkThreshold
        )

        return greedyRes.text
    }

    /// 音響 SNN のみによる直接文字起こし (低信頼度pad化, 短padマージ, CTC collapse, 最小持続フレーム判定)
    public func transcribeAcousticDirect(
        pcmData: [Float],
        minDurationFrames: Int = 3,
        minConfidence: Float = 0.45
    ) -> String {
        let pcm16k = SpeechDataset.resampleTo16k(pcmData: pcmData, sampleRate: 16000)
        let featuresSeq = SpeechDataset.extractFeaturesFromPCM(pcmData: pcm16k)
        let boundaries = FormantSegmenter.detectBoundaries(pcmData: pcm16k)
        return transcribeAcousticDirect(
            featuresSeq: featuresSeq,
            minDurationFrames: minDurationFrames,
            minConfidence: minConfidence,
            boundaries: boundaries
        )
    }

    /// 特徴量系列から音響 SNN のみによる直接文字起こし
    public func transcribeAcousticDirect(
        featuresSeq: [[Float]],
        minDurationFrames: Int = 3,
        minConfidence: Float = 0.45,
        boundaries: [Int]? = nil
    ) -> String {
        if featuresSeq.isEmpty {
            return ""
        }

        let acDecoder = AcousticDecoder(
            network: acousticTrainer.network,
            vocabulary: textVocabulary,
            fallbackVocabulary: phonemeVocabulary
        )
        let acWorkspace = AcousticWorkspace(
            maxHiddenDim: acousticTrainer.network.maxHiddenDim,
            outputDim: acousticTrainer.network.outputDim,
            inputDim: acousticTrainer.network.inputDim,
            numLayers: acousticTrainer.network.numLayers
        )

        let frameProbs = acDecoder.decodeSequence(
            featuresSeq: featuresSeq,
            workspace: acWorkspace,
            boundaries: boundaries
        )
        var rawTokens: [Int] = []
        rawTokens.reserveCapacity(frameProbs.count)

        var f = 0
        while f < frameProbs.count {
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

    /// 音響 SNN の対数確率から CTC プレフィックスビーム探索文字起こしを実行
    ///
    /// フォワードには Event-driven 疎スパイク推論 (`AcousticDecoder`) を用いる。
    /// BPTT 用の密なフォワードと違い発話ごとの巨大キャッシュを確保せず、
    /// 学習側と同じ sliceNorm を適用するためスライス間のスケールも一致する。
    public func transcribeAcousticCTC(
        featuresSeq: [[Float]],
        beamWidth: Int = 16,
        blankPenalty: Float = 0.0
    ) -> String {
        if featuresSeq.isEmpty {
            return ""
        }

        let network = acousticTrainer.network
        let acDecoder = AcousticDecoder(
            network: network,
            vocabulary: textVocabulary,
            fallbackVocabulary: phonemeVocabulary
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

    /// 2段階音声文字起こし (第1段 音響 SNN かな推定 -> 第2段 漢字かな混じり文復元)
    public func transcribeTwoStage(
        featuresSeq: [[Float]],
        kanjiVocabulary: TextVocabulary,
        dictionary: KanaKanjiDictionary? = nil,
        minDurationFrames: Int = 3,
        minConfidence: Float = 0.05,
        boundaries: [Int]? = nil,
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
                minConfidence: minConfidence,
                boundaries: boundaries
            )
        }

        var kanjiText = ""
        switch dictionary {
        case .some(let dict):
            // 辞書 Viterbi DP に、学習済み第2段 言語 SNN の予測も手掛かりとして与える
            let lmDecoder = LanguageDecoder(
                lmNetwork: languageTrainer.network,
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
                lmNetwork: languageTrainer.network,
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
