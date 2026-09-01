import Foundation

/// CTC プレフィックスビーム仮説
public struct CTCPrefixHypothesis: Sendable {
    public var prefix: [Int]         // トークン ID 系列
    public var pBlank: Float         // 末尾が Blank で終わる確率 (対数)
    public var pNonBlank: Float      // 末尾が Non-Blank で終わる確率 (対数)
    public var totalProb: Float {
        CTCLossCalculator.logAdd(pBlank, pNonBlank)
    }

    public init(prefix: [Int], pBlank: Float, pNonBlank: Float) {
        self.prefix = prefix
        self.pBlank = pBlank
        self.pNonBlank = pNonBlank
    }
}

/// Pure Swift SNN CTC プレフィックスビーム探索デコーダ (CTC Prefix Beam Search)
public final class CTCBeamDecoder: @unchecked Sendable {
    public let vocabulary: TextVocabulary
    public let blankId: Int
    public let beamWidth: Int
    public let lmWeight: Float
    public let lengthBonus: Float

    public init(
        vocabulary: TextVocabulary,
        blankId: Int = 0,
        beamWidth: Int = 16,
        lmWeight: Float = 0.3,
        lengthBonus: Float = 0.1
    ) {
        self.vocabulary = vocabulary
        self.blankId = blankId
        self.beamWidth = beamWidth
        self.lmWeight = lmWeight
        self.lengthBonus = lengthBonus
    }

    /// 対数確率系列 (T x V) から CTC プレフィックスビーム探索により最尤テキストをデコード
    public func decode(logProbs: [[Float]]) -> (text: String, tokens: [Int], score: Float) {
        let tCount = logProbs.count
        if tCount <= 0 {
            return ("", [], -.infinity)
        }

        let vCount = logProbs[0].count

        // 初期仮説: 空プレフィックス (pBlank = 0, pNonBlank = -inf)
        var beams: [[Int]: CTCPrefixHypothesis] = [
            []: CTCPrefixHypothesis(prefix: [], pBlank: 0.0, pNonBlank: -.infinity)
        ]

        var t = 0
        while t < tCount {
            let lp = logProbs[t]
            var nextBeams: [[Int]: CTCPrefixHypothesis] = [:]

            for (_, hyp) in beams {
                let pBlankCurr = lp[blankId]

                // 1. Blank トークンの遷移 (プレフィックス長は不変)
                let pBNext = hyp.totalProb + pBlankCurr
                var existingSame = nextBeams[hyp.prefix] ?? CTCPrefixHypothesis(prefix: hyp.prefix, pBlank: -.infinity, pNonBlank: -.infinity)
                existingSame.pBlank = CTCLossCalculator.logAdd(existingSame.pBlank, pBNext)
                nextBeams[hyp.prefix] = existingSame

                // 2. Non-Blank トークンの遷移
                var c = 1
                while c < vCount {
                    let pChar = lp[c]
                    var endToken = -1
                    if 0 < hyp.prefix.count {
                        endToken = hyp.prefix[hyp.prefix.count - 1]
                    }

                    if c == endToken {
                        // 直前と同じ文字の場合:
                        // Blank を挟んだ遷移 (pBlank から) は新規文字追加
                        var newPref = hyp.prefix
                        newPref.append(c)
                        let pNew = hyp.pBlank + pChar
                        var existingNew = nextBeams[newPref] ?? CTCPrefixHypothesis(prefix: newPref, pBlank: -.infinity, pNonBlank: -.infinity)
                        existingNew.pNonBlank = CTCLossCalculator.logAdd(existingNew.pNonBlank, pNew)
                        nextBeams[newPref] = existingNew

                        // Blank を挟まない遷移 (pNonBlank から) は同一文字の継続 (プレフィックス不変)
                        let pCont = hyp.pNonBlank + pChar
                        var exSame = nextBeams[hyp.prefix] ?? CTCPrefixHypothesis(prefix: hyp.prefix, pBlank: -.infinity, pNonBlank: -.infinity)
                        exSame.pNonBlank = CTCLossCalculator.logAdd(exSame.pNonBlank, pCont)
                        nextBeams[hyp.prefix] = exSame
                    } else {
                        // 異なる文字の場合: 常に新規文字追加
                        var newPref = hyp.prefix
                        newPref.append(c)
                        let pNew = hyp.totalProb + pChar
                        var existingNew = nextBeams[newPref] ?? CTCPrefixHypothesis(prefix: newPref, pBlank: -.infinity, pNonBlank: -.infinity)
                        existingNew.pNonBlank = CTCLossCalculator.logAdd(existingNew.pNonBlank, pNew)
                        nextBeams[newPref] = existingNew
                    }
                    c += 1
                }
            }

            // 枝刈り (Top-K ビーム選択)
            var sortedList = Array(nextBeams.values)
            sortedList.sort { a, b in
                b.totalProb < a.totalProb
            }

            beams.removeAll(keepingCapacity: true)
            var k = 0
            let limit = min(beamWidth, sortedList.count)
            while k < limit {
                let h = sortedList[k]
                beams[h.prefix] = h
                k += 1
            }

            t += 1
        }

        // 最良仮説の抽出
        var bestHyp: CTCPrefixHypothesis? = nil
        var bestScore: Float = -.infinity

        for (_, hyp) in beams {
            let score = hyp.totalProb + (Float(hyp.prefix.count) * lengthBonus)
            if bestScore < score {
                bestScore = score
                bestHyp = hyp
            }
        }

        switch bestHyp {
        case .some(let h):
            let text = vocabulary.idsToText(h.prefix)
            return (text, h.prefix, h.totalProb)
        case .none:
            return ("", [], -.infinity)
        }
    }

    /// 単純貪欲法 (Greedy Best Path) による CTC デコード (最速・ゼロアロケーション)
    public func decodeGreedy(logProbs: [[Float]]) -> (text: String, tokens: [Int]) {
        let tCount = logProbs.count
        if tCount <= 0 {
            return ("", [])
        }

        var collapsedTokens: [Int] = []
        var lastToken = blankId

        var t = 0
        while t < tCount {
            let lp = logProbs[t]
            var bestId = 0
            var bestLp = lp[0]
            var c = 1
            while c < lp.count {
                if bestLp < lp[c] {
                    bestLp = lp[c]
                    bestId = c
                }
                c += 1
            }

            if bestId != blankId {
                if bestId != lastToken {
                    collapsedTokens.append(bestId)
                }
            }
            lastToken = bestId
            t += 1
        }

        let text = vocabulary.idsToText(collapsedTokens)
        return (text, collapsedTokens)
    }
}
