import Foundation

/// プレフィックスを整数 ID に内在化する trie。
///
/// 仮説を `[Int]` 配列で持つと、辞書を引くたびにプレフィックス全体のハッシュ計算
/// (O(長さ)) と配列コピーが発生する。trie ノード ID なら検索も追加も O(1) になる。
struct CTCPrefixTrie {
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

    mutating func removeAll() {
        parents = [-1]
        tokens = [-1]
        depths = [0]
        children.removeAll(keepingCapacity: true)
    }
}

/// trie ノードで表したビーム仮説
struct CTCNodeHypothesis {
    var node: Int32
    var pBlank: Float
    var pNonBlank: Float
    var totalProb: Float {
        CTCLossCalculator.logAdd(pBlank, pNonBlank)
    }
}

/// フレーム単位で進める CTC プレフィックスビーム探索。
///
/// ビームと trie をインスタンスに保持するため、音声が届いたぶんだけ `push` して
/// 途中結果を `best` で取り出せる。全フレームが揃うのを待つ必要がない。
/// 発話の区切りでは `reset()` で状態を捨てる。
public final class CTCStreamingDecoder: @unchecked Sendable {
    public let vocabulary: TextVocabulary
    public let blankId: Int
    public let beamWidth: Int
    /// 文字数に応じた対数スコアへの加点。対数確率と単位が異なるため既定は 0.0
    public let lengthBonus: Float
    /// 1 フレームあたりに展開する非 Blank トークンの上限 (確率上位から選択)
    public let tokenPruneCount: Int

    /// 投入済みのフレーム数
    public private(set) var frameCount = 0

    private var trie = CTCPrefixTrie()
    private var beams: [CTCNodeHypothesis]
    private var nextIndex: [Int32: Int] = [:]
    private var nextList: [CTCNodeHypothesis] = []
    private var candidateTokens: [Int] = []
    private var candidateScores: [Float] = []

    public init(
        vocabulary: TextVocabulary,
        blankId: Int = 0,
        beamWidth: Int = 16,
        lengthBonus: Float = 0.0,
        tokenPruneCount: Int = 8
    ) {
        self.vocabulary = vocabulary
        self.blankId = blankId
        self.beamWidth = beamWidth
        self.lengthBonus = lengthBonus
        self.tokenPruneCount = max(1, tokenPruneCount)
        // 初期仮説: 空プレフィックス (pBlank = 0, pNonBlank = -inf)
        self.beams = [CTCNodeHypothesis(node: 0, pBlank: 0.0, pNonBlank: -.infinity)]
    }

    /// 状態を初期化する。発話の区切りで呼ぶ
    public func reset() {
        trie.removeAll()
        beams = [CTCNodeHypothesis(node: 0, pBlank: 0.0, pNonBlank: -.infinity)]
        nextIndex.removeAll(keepingCapacity: true)
        nextList.removeAll(keepingCapacity: true)
        frameCount = 0
    }

    /// 1 フレームの対数確率でビームを進める
    public func push(frame logProbs: [Float]) {
        let vCount = logProbs.count
        if vCount <= 1 {
            return
        }
        if candidateTokens.count < vCount {
            candidateTokens = [Int](repeating: 0, count: vCount)
            candidateScores = [Float](repeating: -.infinity, count: vCount)
        }

        nextIndex.removeAll(keepingCapacity: true)
        nextList.removeAll(keepingCapacity: true)

        // 全語彙を展開すると frame × beam × V 回のノード操作となり非常に遅いため、
        // このフレームで確率上位の tokenPruneCount 個だけを展開対象とする
        var candidateCount = 0
        var c = 1
        while c < vCount {
            let p = logProbs[c]
            if candidateCount < tokenPruneCount {
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

        let pBlankCurr = logProbs[blankId]
        var bi = 0
        while bi < beams.count {
            let hyp = beams[bi]

            // 1. Blank トークンの遷移 (プレフィックス長は不変)
            let sameSlot = slot(for: hyp.node)
            nextList[sameSlot].pBlank = CTCLossCalculator.logAdd(
                nextList[sameSlot].pBlank, hyp.totalProb + pBlankCurr)

            // 2. Non-Blank トークンの遷移 (確率上位の候補のみ展開)
            let endToken = Int(trie.tokens[Int(hyp.node)])
            var ci = 0
            while ci < candidateCount {
                let token = candidateTokens[ci]
                let pChar = logProbs[token]
                let childNode = trie.child(of: hyp.node, token: token)
                if token == endToken {
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
        frameCount += 1
    }

    /// 複数フレームをまとめて進める
    public func push(frames logProbsSeq: [[Float]]) {
        var t = 0
        while t < logProbsSeq.count {
            push(frame: logProbsSeq[t])
            t += 1
        }
    }

    /// 現時点で最良の仮説
    public var best: (text: String, tokens: [Int], score: Float) {
        var bestNode: Int32 = -1
        var bestTotal: Float = -.infinity
        var bestScore: Float = -.infinity

        var i = 0
        while i < beams.count {
            let hyp = beams[i]
            let total = hyp.totalProb
            let score = total + (Float(trie.depths[Int(hyp.node)]) * lengthBonus)
            if bestScore < score {
                bestScore = score
                bestTotal = total
                bestNode = hyp.node
            }
            i += 1
        }

        if bestNode < 0 {
            return ("", [], -.infinity)
        }
        let tokens = trie.sequence(of: bestNode)
        return (vocabulary.idsToText(tokens), tokens, bestTotal)
    }

    /// ノードに対応する仮説の位置を返す (無ければ -inf で作る)
    private func slot(for node: Int32) -> Int {
        switch nextIndex[node] {
        case .some(let idx):
            return idx
        case .none:
            let idx = nextList.count
            nextList.append(CTCNodeHypothesis(node: node, pBlank: -.infinity, pNonBlank: -.infinity))
            nextIndex[node] = idx
            return idx
        }
    }
}
