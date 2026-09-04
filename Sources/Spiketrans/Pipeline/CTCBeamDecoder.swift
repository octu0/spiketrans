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

    /// プレフィックスを整数 ID に内在化する trie。
    ///
    /// 仮説を `[Int]` 配列で持つと、辞書を引くたびにプレフィックス全体のハッシュ計算
    /// (O(長さ)) と配列コピーが発生する。trie ノード ID なら検索も追加も O(1) になる。
    private struct PrefixTrie {
        var parents: [Int32] = [-1]
        var tokens: [Int32] = [-1]
        var depths: [Int32] = [0]
        /// (親ノード, トークン) -> 子ノード
        var children: [Int64: Int32] = [:]

        /// 末尾にトークンを足したノードを返す (未登録なら作る)
        mutating func child(of node: Int32, token: Int) -> Int32 {
            let key = (Int64(node) << 32) | Int64(token)
            switch children[key] {
            case .some(let existing):
                return existing
            case .none:
                let newNode = Int32(parents.count)
                parents.append(node)
                tokens.append(Int32(token))
                depths.append(depths[Int(node)] + 1)
                children[key] = newNode
                return newNode
            }
        }

        /// ノードからルートまで遡ってトークン列を復元する
        func sequence(of node: Int32) -> [Int] {
            var result = [Int](repeating: 0, count: Int(depths[Int(node)]))
            var cursor = node
            var pos = result.count - 1
            while 0 <= pos {
                result[pos] = Int(tokens[Int(cursor)])
                cursor = parents[Int(cursor)]
                pos -= 1
            }
            return result
        }
    }

    /// trie ノードで表したビーム仮説
    private struct NodeHypothesis {
        var node: Int32
        var pBlank: Float
        var pNonBlank: Float
        var totalProb: Float {
            CTCLossCalculator.logAdd(pBlank, pNonBlank)
        }
    }

    /// 対数確率系列 (T x V) から CTC プレフィックスビーム探索により最尤テキストをデコード
    public func decode(logProbs: [[Float]]) -> (text: String, tokens: [Int], score: Float) {
        let tCount = logProbs.count
        if tCount <= 0 {
            return ("", [], -.infinity)
        }

        let vCount = logProbs[0].count

        var trie = PrefixTrie()
        let rootNode: Int32 = 0

        // 初期仮説: 空プレフィックス (pBlank = 0, pNonBlank = -inf)
        var beams: [NodeHypothesis] = [
            NodeHypothesis(node: rootNode, pBlank: 0.0, pNonBlank: -.infinity)
        ]
        // ノード -> nextList 内の位置。フレームごとに再利用する
        var nextIndex: [Int32: Int] = [:]
        var nextList: [NodeHypothesis] = []

        // 展開候補トークンのバッファ (毎フレーム再利用してアロケーションを避ける)
        var candidateTokens = [Int](repeating: 0, count: max(1, vCount))
        var candidateScores = [Float](repeating: -.infinity, count: max(1, vCount))

        var t = 0
        while t < tCount {
            let lp = logProbs[t]
            nextIndex.removeAll(keepingCapacity: true)
            nextList.removeAll(keepingCapacity: true)

            // ノードに対応する仮説の位置を返す (無ければ -inf で作る)
            func slot(for node: Int32) -> Int {
                switch nextIndex[node] {
                case .some(let idx):
                    return idx
                case .none:
                    let idx = nextList.count
                    nextList.append(NodeHypothesis(node: node, pBlank: -.infinity, pNonBlank: -.infinity))
                    nextIndex[node] = idx
                    return idx
                }
            }

            // 全語彙を展開すると T × beam × V 回のプレフィックス辞書操作となり非常に遅いため、
            // このフレームで確率上位の tokenPruneCount 個だけを展開対象とする。
            var candidateCount = 0
            var c = 1
            while c < vCount {
                let p = lp[c]
                if candidateCount < tokenPruneCount {
                    // まだ空きがあるので挿入位置を探して挿入
                    var insertAt = candidateCount
                    while 0 < insertAt && candidateScores[insertAt - 1] < p {
                        candidateScores[insertAt] = candidateScores[insertAt - 1]
                        candidateTokens[insertAt] = candidateTokens[insertAt - 1]
                        insertAt -= 1
                    }
                    candidateScores[insertAt] = p
                    candidateTokens[insertAt] = c
                    candidateCount += 1
                } else {
                    if candidateScores[candidateCount - 1] < p {
                        var insertAt = candidateCount - 1
                        while 0 < insertAt && candidateScores[insertAt - 1] < p {
                            candidateScores[insertAt] = candidateScores[insertAt - 1]
                            candidateTokens[insertAt] = candidateTokens[insertAt - 1]
                            insertAt -= 1
                        }
                        candidateScores[insertAt] = p
                        candidateTokens[insertAt] = c
                    }
                }
                c += 1
            }

            let pBlankCurr = lp[blankId]
            var bi = 0
            while bi < beams.count {
                let hyp = beams[bi]

                // 1. Blank トークンの遷移 (プレフィックス長は不変)
                let pBNext = hyp.totalProb + pBlankCurr
                let sameSlot = slot(for: hyp.node)
                nextList[sameSlot].pBlank = CTCLossCalculator.logAdd(nextList[sameSlot].pBlank, pBNext)

                // 2. Non-Blank トークンの遷移 (確率上位の候補のみ展開)
                let endToken = Int(trie.tokens[Int(hyp.node)])
                var ci = 0
                while ci < candidateCount {
                    let c = candidateTokens[ci]
                    let pChar = lp[c]
                    let childNode = trie.child(of: hyp.node, token: c)
                    if c == endToken {
                        // 直前と同じ文字の場合:
                        // Blank を挟んだ遷移 (pBlank から) は新規文字追加
                        let newSlot = slot(for: childNode)
                        nextList[newSlot].pNonBlank = CTCLossCalculator.logAdd(
                            nextList[newSlot].pNonBlank, hyp.pBlank + pChar)

                        // Blank を挟まない遷移 (pNonBlank から) は同一文字の継続 (プレフィックス不変)
                        let contSlot = slot(for: hyp.node)
                        nextList[contSlot].pNonBlank = CTCLossCalculator.logAdd(
                            nextList[contSlot].pNonBlank, hyp.pNonBlank + pChar)
                    } else {
                        // 異なる文字の場合: 常に新規文字追加
                        let newSlot = slot(for: childNode)
                        nextList[newSlot].pNonBlank = CTCLossCalculator.logAdd(
                            nextList[newSlot].pNonBlank, hyp.totalProb + pChar)
                    }
                    ci += 1
                }
                bi += 1
            }

            // 枝刈り (Top-K ビーム選択)。
            // 同点時はノード ID 昇順で決めるため、実行ごとに結果が揺れない
            nextList.sort { a, b in
                if a.totalProb != b.totalProb {
                    return b.totalProb < a.totalProb
                }
                return a.node < b.node
            }

            beams.removeAll(keepingCapacity: true)
            var k = 0
            let limit = min(beamWidth, nextList.count)
            while k < limit {
                beams.append(nextList[k])
                k += 1
            }

            t += 1
        }

        // 最良仮説の抽出
        var bestNode: Int32 = -1
        var bestTotal: Float = -.infinity
        var bestScore: Float = -.infinity

        var bIdx = 0
        while bIdx < beams.count {
            let hyp = beams[bIdx]
            let total = hyp.totalProb
            let score = total + (Float(trie.depths[Int(hyp.node)]) * lengthBonus)
            if bestScore < score {
                bestScore = score
                bestTotal = total
                bestNode = hyp.node
            }
            bIdx += 1
        }

        if bestNode < 0 {
            return ("", [], -.infinity)
        }
        let tokens = trie.sequence(of: bestNode)
        return (vocabulary.idsToText(tokens), tokens, bestTotal)
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
