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
    /// 文字数に応じた対数スコアへの加点。対数確率と単位が異なるため既定は 0.0。
    /// 正値にすると未収束モデルで長い繰り返し出力が選ばれやすくなる。
    public let lengthBonus: Float
    /// 1 フレームあたりに展開する非 Blank トークンの上限 (確率上位から選択)
    public let tokenPruneCount: Int

    public init(
        vocabulary: TextVocabulary,
        blankId: Int = 0,
        beamWidth: Int = 16,
        lmWeight: Float = 0.3,
        lengthBonus: Float = 0.0,
        tokenPruneCount: Int = 8
    ) {
        self.vocabulary = vocabulary
        self.blankId = blankId
        self.beamWidth = beamWidth
        self.lmWeight = lmWeight
        self.lengthBonus = lengthBonus
        self.tokenPruneCount = max(1, tokenPruneCount)
    }

    /// 対数確率系列 (T x V) から CTC プレフィックスビーム探索により最尤テキストをデコード。
    /// 実体は `CTCStreamingDecoder` で、全フレームを流し込んでから最良仮説を取り出す
    public func decode(logProbs: [[Float]]) -> (text: String, tokens: [Int], score: Float) {
        if logProbs.isEmpty {
            return ("", [], -.infinity)
        }
        let streaming = makeStreamingDecoder()
        streaming.push(frames: logProbs)
        return streaming.best
    }

    /// 同じ設定でフレーム単位に進められるデコーダを作る
    public func makeStreamingDecoder() -> CTCStreamingDecoder {
        return CTCStreamingDecoder(
            vocabulary: vocabulary,
            blankId: blankId,
            beamWidth: beamWidth,
            lengthBonus: lengthBonus,
            tokenPruneCount: tokenPruneCount
        )
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
