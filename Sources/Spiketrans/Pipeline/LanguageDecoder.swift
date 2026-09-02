import Foundation

/// ビーム探索における仮説候補
public struct BeamHypothesis: Sendable {
    public let tokenIds: [Int]
    public let score: Float
    public let lmVState: [Float]
    public let lmSState: [Float]
    public let lmAState: [Float]
    public let isFinished: Bool

    public init(
        tokenIds: [Int],
        score: Float,
        lmVState: [Float],
        lmSState: [Float],
        lmAState: [Float] = [],
        isFinished: Bool
    ) {
        self.tokenIds = tokenIds
        self.score = score
        self.lmVState = lmVState
        self.lmSState = lmSState
        self.lmAState = lmAState
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

/// 第2段 自己回帰言語 SNN デコーダ
public final class LanguageDecoder: @unchecked Sendable {
    public let lmNetwork: SpikingNetwork
    public let vocabulary: TextVocabulary
    public let fallbackVocabulary: PhonemeVocabulary
    public let config: LanguageDecoderConfig

    public init(
        lmNetwork: SpikingNetwork,
        vocabulary: TextVocabulary = TextVocabulary(),
        fallbackVocabulary: PhonemeVocabulary = PhonemeVocabulary(),
        config: LanguageDecoderConfig = LanguageDecoderConfig()
    ) {
        self.lmNetwork = lmNetwork
        self.vocabulary = vocabulary
        self.fallbackVocabulary = fallbackVocabulary
        self.config = config
    }

    /// トークン ID からワンホット風埋め込み特徴量を生成
    private func buildTokenFeature(tokenId: Int) -> [Float] {
        var feat = [Float](repeating: 0.0, count: lmNetwork.inputDim)
        if tokenId < lmNetwork.inputDim {
            feat[tokenId] = 1.0
        } else {
            // ハッシュ分散埋め込み
            let idx = abs(tokenId.hashValue) % lmNetwork.inputDim
            feat[idx] = 1.0
        }
        return feat
    }

    /// 貪欲法 (Greedy) による音響+言語結合デコード (直接漢字かな + 未知語フォールバック)
    public func decodeGreedy(
        acousticProbs: [AcousticFrameProbabilities],
        unkThreshold: Float = 0.25
    ) -> (tokens: [Int], text: String, score: Float) {
        if acousticProbs.isEmpty {
            return ([], "", 0.0)
        }

        let hSize = lmNetwork.maxHiddenDim
        let outDim = lmNetwork.outputDim
        var vLM = [Float](repeating: 0.0, count: hSize)
        var sLM = [Float](repeating: 0.0, count: hSize)
        var aLM = [Float](repeating: 0.0, count: hSize)
        var spikeSumLM = [Float](repeating: 0.0, count: hSize)
        var logitsLM = [Float](repeating: 0.0, count: outDim)
        var probsLM = [Float](repeating: 0.0, count: outDim)

        let scratchLM = ForwardScratch(maxHiddenDim: hSize)

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

            lmNetwork.forward(
                features: tokenFeatures,
                vPrev: &vLM,
                sPrev: &sLM,
                aPrev: &aLM,
                spikeSum: &spikeSumLM,
                logits: &logitsLM,
                probabilities: &probsLM,
                scratch: scratchLM
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
                    // 系列の途中で eos を選ぶと出力が途中で切れるため終端付近のみ許可
                    if acousticProbs.count - 5 <= fIdx {
                        score = log(pAc) + (config.lmWeight * log(pLm))
                    } else {
                        score = -Float.greatestFiniteMagnitude
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
        unkThreshold: Float = 0.25
    ) -> (tokens: [Int], text: String, score: Float) {
        if acousticProbs.isEmpty {
            return ([], "", 0.0)
        }

        let hSize = lmNetwork.maxHiddenDim
        let outDim = lmNetwork.outputDim
        var beams: [BeamHypothesis] = [
            BeamHypothesis(
                tokenIds: [],
                score: 0.0,
                lmVState: [Float](repeating: 0.0, count: hSize),
                lmSState: [Float](repeating: 0.0, count: hSize),
                lmAState: [Float](repeating: 0.0, count: hSize),
                isFinished: false
            )
        ]

        var spikeSumLM = [Float](repeating: 0.0, count: hSize)
        var logitsLM = [Float](repeating: 0.0, count: outDim)
        var probsLM = [Float](repeating: 0.0, count: outDim)
        let scratchLM = ForwardScratch(maxHiddenDim: hSize)

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
                var aLM = hyp.lmAState
                if aLM.isEmpty {
                    aLM = [Float](repeating: 0.0, count: hSize)
                }
                let tokenFeatures = buildTokenFeature(tokenId: prevTok)

                lmNetwork.forward(
                    features: tokenFeatures,
                    vPrev: &vLM,
                    sPrev: &sLM,
                    aPrev: &aLM,
                    spikeSum: &spikeSumLM,
                    logits: &logitsLM,
                    probabilities: &probsLM,
                    scratch: scratchLM
                )

                // 各候補トークンを展開
                let maxTokId = min(fp.probabilities.count, probsLM.count)
                var tokId = 0
                while tokId < maxTokId {
                    let pAc = max(1e-7, fp.probabilities[tokId])
                    let pLm = max(1e-7, probsLM[tokId])

                    switch tokId {
                    case TextVocabulary.padId:
                        let newScore = hyp.score + log(pAc) - config.blankPenalty
                        candidates.append(BeamHypothesis(
                            tokenIds: hyp.tokenIds,
                            score: newScore,
                            lmVState: vLM,
                            lmSState: sLM,
                            lmAState: aLM,
                            isFinished: false
                        ))
                    case TextVocabulary.unkId:
                        // 未知トークンのペナルティ (系列には追加しない)
                        let newScore = hyp.score + log(pAc) - 1.0
                        candidates.append(BeamHypothesis(
                            tokenIds: hyp.tokenIds,
                            score: newScore,
                            lmVState: vLM,
                            lmSState: sLM,
                            lmAState: aLM,
                            isFinished: false
                        ))
                    case TextVocabulary.eosId:
                        // 系列の途中で確定させると出力が途中で切れるため終端付近のみ許可
                        if acousticProbs.count - 5 <= fIdx {
                            let newScore = hyp.score + log(pAc) + (config.lmWeight * log(pLm))
                            candidates.append(BeamHypothesis(
                                tokenIds: hyp.tokenIds,
                                score: newScore,
                                lmVState: vLM,
                                lmSState: sLM,
                                lmAState: aLM,
                                isFinished: true
                            ))
                        }
                    default:
                        var newTokens = hyp.tokenIds
                        // 連続同一トークンの重複圧縮判定 (直前が同じトークンかつ直前が pad でない場合はマージ)
                        var shouldAppend = true
                        if newTokens.isEmpty != true {
                            let lastTok = newTokens[newTokens.count - 1]
                            if lastTok == tokId {
                                shouldAppend = false
                            }
                        }

                        if shouldAppend {
                            newTokens.append(tokId)
                        }

                        let newScore = hyp.score + log(pAc) + (config.lmWeight * log(pLm)) + config.wordBonus
                        candidates.append(BeamHypothesis(
                            tokenIds: newTokens,
                            score: newScore,
                            lmVState: vLM,
                            lmSState: sLM,
                            lmAState: aLM,
                            isFinished: false
                        ))
                    }

                    tokId += 1
                }

                bIdx += 1
            }

            // スコア順にソートして Top-K を残す
            candidates.sort { hypB, hypA in
                hypA.score < hypB.score
            }

            beams = Array(candidates.prefix(config.beamWidth))
            fIdx += 1
        }

        // 最良仮説を選択
        var bestHyp = beams[0]
        var b = 1
        while b < beams.count {
            if bestHyp.score < beams[b].score {
                bestHyp = beams[b]
            }
            b += 1
        }

        let text = vocabulary.idsToText(bestHyp.tokenIds)
        return (tokens: bestHyp.tokenIds, text: text, score: bestHyp.score)
    }

    /// 音響 SNN の推定したかな文字列から直接漢字かな混じり文を自己回帰復元
    public func decodeKanaToKanji(
        kanaText: String,
        kanaVocabulary: TextVocabulary
    ) -> String {
        if kanaText.isEmpty {
            return ""
        }

        let kanaIds = kanaVocabulary.textToIds(kanaText)
        if kanaIds.isEmpty {
            return ""
        }

        let hSize = lmNetwork.maxHiddenDim
        let outDim = lmNetwork.outputDim
        var vLM = [Float](repeating: 0.0, count: hSize)
        var sLM = [Float](repeating: 0.0, count: hSize)
        var aLM = [Float](repeating: 0.0, count: hSize)
        var spikeSumLM = [Float](repeating: 0.0, count: hSize)
        var logitsLM = [Float](repeating: 0.0, count: outDim)
        var probsLM = [Float](repeating: 0.0, count: outDim)

        let scratchLM = ForwardScratch(maxHiddenDim: hSize)

        var outputTokens: [Int] = []
        var kIdx = 0
        while kIdx < kanaIds.count {
            let kId = kanaIds[kIdx]
            let feat = buildTokenFeature(tokenId: kId)

            lmNetwork.forward(
                features: feat,
                vPrev: &vLM,
                sPrev: &sLM,
                aPrev: &aLM,
                spikeSum: &spikeSumLM,
                logits: &logitsLM,
                probabilities: &probsLM,
                scratch: scratchLM
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

    /// かな 1 文字ごとに言語 SNN が予測する漢字 1 文字を返す (予測不能な位置は nil)
    ///
    /// かな位置と 1:1 対応するため、第2段 Viterbi DP の区間スコアに
    /// 言語 SNN の予測を局所的な手掛かりとして加算できる。
    public func predictKanjiPerKana(
        kanaText: String,
        kanaVocabulary: TextVocabulary
    ) -> [Character?] {
        let kanaChars = Array(kanaText)
        if kanaChars.isEmpty {
            return []
        }

        var hints = [Character?](repeating: nil, count: kanaChars.count)

        let hSize = lmNetwork.maxHiddenDim
        let outDim = lmNetwork.outputDim
        var vLM = [Float](repeating: 0.0, count: hSize)
        var sLM = [Float](repeating: 0.0, count: hSize)
        var aLM = [Float](repeating: 0.0, count: hSize)
        var spikeSumLM = [Float](repeating: 0.0, count: hSize)
        var logitsLM = [Float](repeating: 0.0, count: outDim)
        var probsLM = [Float](repeating: 0.0, count: outDim)
        let scratchLM = ForwardScratch(maxHiddenDim: hSize)

        var kIdx = 0
        while kIdx < kanaChars.count {
            let ids = kanaVocabulary.textToIds(String(kanaChars[kIdx]))
            var kId = TextVocabulary.unkId
            if ids.isEmpty != true {
                kId = ids[0]
            }

            lmNetwork.forward(
                features: buildTokenFeature(tokenId: kId),
                vPrev: &vLM,
                sPrev: &sLM,
                aPrev: &aLM,
                spikeSum: &spikeSumLM,
                logits: &logitsLM,
                probabilities: &probsLM,
                scratch: scratchLM
            )

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
                let predicted = vocabulary.idsToText([bestTokId])
                if predicted.count == 1 {
                    hints[kIdx] = Array(predicted)[0]
                }
            }
            kIdx += 1
        }

        return hints
    }
}
