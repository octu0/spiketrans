import Foundation

/// ビーム探索における仮説候補
public struct BeamHypothesis: Sendable {
    public let tokenIds: [Int]
    public let score: Float
    public let lmVState: [Float]
    public let lmSState: [Float]
    public let isFinished: Bool

    public init(
        tokenIds: [Int],
        score: Float,
        lmVState: [Float],
        lmSState: [Float],
        isFinished: Bool
    ) {
        self.tokenIds = tokenIds
        self.score = score
        self.lmVState = lmVState
        self.lmSState = lmSState
        self.isFinished = isFinished
    }
}

/// 言語デコーダ設定パラメータ
public struct LanguageDecoderConfig: Sendable {
    public let beamWidth: Int
    public let maxSequenceLength: Int
    public let lmWeight: Float         // α (言語モデル重み)
    public let wordBonus: Float        // β (単語挿入ボーナス)
    public let blankPenalty: Float

    public init(
        beamWidth: Int = 4,
        maxSequenceLength: Int = 128,
        lmWeight: Float = 0.0,
        wordBonus: Float = 0.0,
        blankPenalty: Float = 0.0
    ) {
        self.beamWidth = beamWidth
        self.maxSequenceLength = maxSequenceLength
        self.lmWeight = lmWeight
        self.wordBonus = wordBonus
        self.blankPenalty = blankPenalty
    }
}

/// 第2段 自己回帰言語 SNN デコーダ (先行文字トークン文脈遷移 + 音響言語結合デコード)
public final class LanguageDecoder: @unchecked Sendable {
    public let lmNetwork: MatryoshkaNetwork
    public let vocabulary: TextVocabulary
    public let fallbackVocabulary: PhonemeVocabulary
    public let config: LanguageDecoderConfig
    public let inputDim: Int

    public init(
        lmNetwork: MatryoshkaNetwork,
        vocabulary: TextVocabulary = TextVocabulary(),
        fallbackVocabulary: PhonemeVocabulary = PhonemeVocabulary(),
        config: LanguageDecoderConfig = LanguageDecoderConfig()
    ) {
        self.lmNetwork = lmNetwork
        self.vocabulary = vocabulary
        self.fallbackVocabulary = fallbackVocabulary
        self.config = config
        self.inputDim = lmNetwork.inputDim
    }

    /// トークン ID から決定論的埋め込み特徴量ベクトル (inputDim 次元) を生成
    @inline(__always)
    public func buildTokenFeature(tokenId: Int) -> [Float] {
        var feat = [Float](repeating: 0.0, count: inputDim)
        var d = 0
        while d < inputDim {
            let angle = Float(tokenId * 17 + d * 11 + 3)
            feat[d] = (sin(angle) * 0.5) + 0.5
            d += 1
        }
        return feat
    }

    /// 貪欲法 (Greedy) による音響+言語結合デコード (直接漢字かな + 未知語フォールバック)
    public func decodeGreedy(
        acousticProbs: [AcousticFrameProbabilities],
        slice: MatryoshkaSlice = .base,
        unkThreshold: Float = 0.25
    ) -> (tokens: [Int], text: String, score: Float) {
        if acousticProbs.isEmpty {
            return ([], "", 0.0)
        }

        let hSize = min(slice.rawValue, lmNetwork.maxHiddenDim)
        let outDim = lmNetwork.outputDim
        var vLM = [Float](repeating: 0.0, count: hSize)
        var sLM = [Float](repeating: 0.0, count: hSize)
        var spikeSumLM = [Float](repeating: 0.0, count: hSize)
        var logitsLM = [Float](repeating: 0.0, count: outDim)
        var probsLM = [Float](repeating: 0.0, count: outDim)

        var prevToken = TextVocabulary.sosId
        var accumulatedScore: Float = 0.0
        var selectedTokens: [Int] = []
        var lastNonBlankToken: Int? = nil
        var lastTokenWasBlank = true

        var fIdx = 0
        while fIdx < acousticProbs.count {
            if config.maxSequenceLength <= selectedTokens.count {
                break
            }

            let fp = acousticProbs[fIdx]

            // 1. 先行トークン埋め込みによる言語 SNN 推論
            let tokenFeatures = buildTokenFeature(tokenId: prevToken)

            lmNetwork.forwardSlice(
                features: tokenFeatures,
                slice: slice,
                vPrev: &vLM,
                sPrev: &sLM,
                spikeSum: &spikeSumLM,
                logits: &logitsLM,
                probabilities: &probsLM
            )

            // 2. 結合スコア最大トークンの選択
            var bestToken = fp.topTokenId
            var bestScore: Float = -Float.greatestFiniteMagnitude

            let maxTokId = min(fp.probabilities.count, probsLM.count)
            var tokId = 0
            while tokId < maxTokId {
                let pAc = max(1e-7, fp.probabilities[tokId])
                let pLm = max(1e-7, probsLM[tokId])
                var score: Float = 0.0

                switch tokId {
                case TextVocabulary.padId:
                    score = log(pAc) - config.blankPenalty
                case TextVocabulary.unkId:
                    score = log(pAc) - 1.0 // 未知トークンのペナルティ
                case TextVocabulary.eosId:
                    if acousticProbs.count - 5 <= fIdx {
                        score = log(pAc) + (config.lmWeight * log(pLm))
                    } else {
                        score = -Float.greatestFiniteMagnitude // 系列の途中では eos を除外
                    }
                default:
                    score = log(pAc) + (config.lmWeight * log(pLm)) + config.wordBonus
                }

                if bestScore < score {
                    bestScore = score
                    bestToken = tokId
                }

                tokId += 1
            }

            if bestToken == TextVocabulary.eosId {
                break
            }

            // 3. CTC 重複圧縮を考慮したトークン蓄積
            if bestToken == TextVocabulary.padId {
                lastTokenWasBlank = true
            } else {
                var shouldAppend = false
                switch lastNonBlankToken {
                case .none:
                    shouldAppend = true
                case .some(let prev):
                    if prev != bestToken || lastTokenWasBlank {
                        shouldAppend = true
                    }
                }

                if shouldAppend {
                    selectedTokens.append(bestToken)
                    accumulatedScore += bestScore
                    prevToken = bestToken
                    lastNonBlankToken = bestToken
                }
                lastTokenWasBlank = false
            }

            fIdx += 1
        }

        let text = vocabulary.idsToText(selectedTokens)
        return (tokens: selectedTokens, text: text, score: accumulatedScore)
    }

    /// ビーム探索 (Beam Search) による音響+言語結合デコード
    public func decodeBeamSearch(
        acousticProbs: [AcousticFrameProbabilities],
        slice: MatryoshkaSlice = .base,
        unkThreshold: Float = 0.25
    ) -> (tokens: [Int], text: String, score: Float) {
        if acousticProbs.isEmpty {
            return ([], "", 0.0)
        }

        let hSize = min(slice.rawValue, lmNetwork.maxHiddenDim)
        let outDim = lmNetwork.outputDim
        var beams: [BeamHypothesis] = [
            BeamHypothesis(
                tokenIds: [],
                score: 0.0,
                lmVState: [Float](repeating: 0.0, count: hSize),
                lmSState: [Float](repeating: 0.0, count: hSize),
                isFinished: false
            )
        ]

        var spikeSumLM = [Float](repeating: 0.0, count: hSize)
        var logitsLM = [Float](repeating: 0.0, count: outDim)
        var probsLM = [Float](repeating: 0.0, count: outDim)

        var fIdx = 0
        while fIdx < acousticProbs.count {
            let fp = acousticProbs[fIdx]
            var candidates: [BeamHypothesis] = []

            var bIdx = 0
            while bIdx < beams.count {
                let hyp = beams[bIdx]
                if hyp.isFinished {
                    candidates.append(hyp)
                    bIdx += 1
                    continue
                }

                var prevTok = TextVocabulary.sosId
                if hyp.tokenIds.isEmpty != true {
                    prevTok = hyp.tokenIds[hyp.tokenIds.count - 1]
                }

                var vLM = hyp.lmVState
                var sLM = hyp.lmSState
                let tokenFeatures = buildTokenFeature(tokenId: prevTok)

                lmNetwork.forwardSlice(
                    features: tokenFeatures,
                    slice: slice,
                    vPrev: &vLM,
                    sPrev: &sLM,
                    spikeSum: &spikeSumLM,
                    logits: &logitsLM,
                    probabilities: &probsLM
                )

                // 各候補トークンを展開
                let maxTokId = min(fp.probabilities.count, probsLM.count)
                var tokId = 0
                while tokId < maxTokId {
                    let pAc = max(1e-7, fp.probabilities[tokId])
                    let pLm = max(1e-7, probsLM[tokId])

                    switch tokId {
                    case TextVocabulary.padId:
                        let stepScore = log(pAc) - config.blankPenalty
                        candidates.append(BeamHypothesis(
                            tokenIds: hyp.tokenIds,
                            score: hyp.score + stepScore,
                            lmVState: vLM,
                            lmSState: sLM,
                            isFinished: false
                        ))
                    case TextVocabulary.unkId:
                        let stepScore = log(pAc) - 1.0
                        candidates.append(BeamHypothesis(
                            tokenIds: hyp.tokenIds,
                            score: hyp.score + stepScore,
                            lmVState: vLM,
                            lmSState: sLM,
                            isFinished: false
                        ))
                    case TextVocabulary.eosId:
                        let stepScore = log(pAc) + (config.lmWeight * log(pLm))
                        candidates.append(BeamHypothesis(
                            tokenIds: hyp.tokenIds,
                            score: hyp.score + stepScore,
                            lmVState: vLM,
                            lmSState: sLM,
                            isFinished: true
                        ))
                    default:
                        let stepScore = log(pAc) + (config.lmWeight * log(pLm)) + config.wordBonus
                        var newTokens = hyp.tokenIds
                        var shouldAppend = false
                        switch hyp.tokenIds.last {
                        case .none:
                            shouldAppend = true
                        case .some(let prev):
                            if prev != tokId {
                                shouldAppend = true
                            }
                        }

                        if shouldAppend {
                            newTokens.append(tokId)
                        }

                        candidates.append(BeamHypothesis(
                            tokenIds: newTokens,
                            score: hyp.score + stepScore,
                            lmVState: vLM,
                            lmSState: sLM,
                            isFinished: false
                        ))
                    }

                    tokId += 1
                }

                bIdx += 1
            }

            // 枝刈り (スコア降順ソート: < を使用)
            candidates.sort { (a, b) -> Bool in
                b.score < a.score
            }

            var nextBeams: [BeamHypothesis] = []
            var cIdx = 0
            let limit = min(config.beamWidth, candidates.count)
            while cIdx < limit {
                nextBeams.append(candidates[cIdx])
                cIdx += 1
            }
            beams = nextBeams

            fIdx += 1
        }

        let bestHyp = beams[0]
        let text = vocabulary.idsToText(bestHyp.tokenIds)
        return (tokens: bestHyp.tokenIds, text: text, score: bestHyp.score)
    }

    /// ひらがな・音素テキスト系列から自己回帰言語 SNN により漢字かな混じりテキストを生成
    public func decodeKanaToKanji(
        kanaText: String,
        kanaVocabulary: TextVocabulary,
        slice: MatryoshkaSlice = .high
    ) -> String {
        if kanaText.isEmpty {
            return ""
        }

        let kanaIds = kanaVocabulary.textToIds(kanaText)
        if kanaIds.isEmpty {
            return ""
        }

        let hSize = min(slice.rawValue, lmNetwork.maxHiddenDim)
        let outDim = lmNetwork.outputDim
        var vLM = [Float](repeating: 0.0, count: hSize)
        var sLM = [Float](repeating: 0.0, count: hSize)
        var spikeSumLM = [Float](repeating: 0.0, count: hSize)
        var logitsLM = [Float](repeating: 0.0, count: outDim)
        var probsLM = [Float](repeating: 0.0, count: outDim)

        var outputTokens: [Int] = []
        var kIdx = 0
        while kIdx < kanaIds.count {
            let kId = kanaIds[kIdx]
            let feat = buildTokenFeature(tokenId: kId)

            lmNetwork.forwardSlice(
                features: feat,
                slice: slice,
                vPrev: &vLM,
                sPrev: &sLM,
                spikeSum: &spikeSumLM,
                logits: &logitsLM,
                probabilities: &probsLM
            )

            // Top-1 漢字トークン（4 <= ID）の選択
            var bestTokId = -1
            var bestProb: Float = -1.0
            var tId = 4
            while tId < outDim {
                let p = probsLM[tId]
                if bestProb < p {
                    bestProb = p
                    bestTokId = tId
                }
                tId += 1
            }

            if 4 <= bestTokId {
                outputTokens.append(bestTokId)
            }
            kIdx += 1
        }

        return vocabulary.idsToText(outputTokens)
    }
}
