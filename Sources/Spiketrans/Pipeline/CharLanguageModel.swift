import Foundation

/// 言語モデル・文脈スコアラーの共通インターフェイス
public protocol LanguageModelScorer: Sendable {
    /// 与えられたテキスト (漢字かな混じり文 Y) の対数確率 log P_LM(Y) を算出
    func logProbability(of text: String) -> Float
}

/// 漢字かな混じり文の文字レベル自己回帰言語モデル
///
/// 直前の文字を入力に次の文字の分布を出す SNN。文全体の対数確率
/// `Σ log P(c_t | c_1..c_{t-1})` を返せるため、第2段の N-best 候補を
/// 文脈込みで再スコアリングできる。学習はテキストのみで完結する。
public final class CharLanguageModel: LanguageModelScorer, @unchecked Sendable {
    public let network: SpikingNetwork
    public let vocabulary: TextVocabulary

    public init(network: SpikingNetwork, vocabulary: TextVocabulary) {
        self.network = network
        self.vocabulary = vocabulary
    }

    /// 語彙サイズから未学習のモデルを作る
    public static func make(vocabulary: TextVocabulary, hiddenDim: Int = 1024, timeSteps: Int = 4) -> CharLanguageModel {
        let net = SpikingNetwork(
            inputDim: vocabulary.size,
            maxHiddenDim: hiddenDim,
            outputDim: vocabulary.size,
            timeSteps: timeSteps
        )
        return CharLanguageModel(network: net, vocabulary: vocabulary)
    }

    /// 文字 ID をワンホット特徴量にする
    static func oneHot(_ tokenId: Int, size: Int) -> [Float] {
        var v = [Float](repeating: 0.0, count: size)
        if 0 <= tokenId && tokenId < size {
            v[tokenId] = 1.0
        }
        return v
    }

    /// 学習用の (入力系列, 目標系列) を作る。
    /// 入力は sos + 文の先頭から末尾ひとつ前、目標は文そのもの。
    public static func makeTrainingPair(text: String, vocabulary: TextVocabulary) -> (features: [[Float]], targets: [Int]) {
        let ids = vocabulary.textToIds(text)
        if ids.isEmpty {
            return ([], [])
        }
        var features: [[Float]] = []
        features.reserveCapacity(ids.count)
        var prev = TextVocabulary.sosId
        var i = 0
        while i < ids.count {
            features.append(oneHot(prev, size: vocabulary.size))
            prev = ids[i]
            i += 1
        }
        return (features, ids)
    }

    /// 文の対数確率。文字数で割らない生の合計値を返す。
    public func logProbability(of text: String) -> Float {
        let ids = vocabulary.textToIds(text)
        if ids.isEmpty {
            return 0.0
        }

        let hSize = network.maxHiddenDim
        let outDim = network.outputDim
        var vPrev = [Float](repeating: 0.0, count: hSize)
        var sPrev = [Float](repeating: 0.0, count: hSize)
        var aPrev = [Float](repeating: 0.0, count: hSize)
        var spikeSum = [Float](repeating: 0.0, count: hSize)
        var logits = [Float](repeating: 0.0, count: outDim)
        var probs = [Float](repeating: 0.0, count: outDim)
        let scratch = ForwardScratch(maxHiddenDim: hSize)

        var oneHotInput = [Float](repeating: 0.0, count: network.inputDim)
        var total: Float = 0.0
        var prev = TextVocabulary.sosId
        var i = 0
        while i < ids.count {
            if 0 <= prev && prev < network.inputDim {
                oneHotInput[prev] = 1.0
            }
            network.forward(
                features: oneHotInput,
                vPrev: &vPrev,
                sPrev: &sPrev,
                aPrev: &aPrev,
                spikeSum: &spikeSum,
                logits: &logits,
                probabilities: &probs,
                scratch: scratch
            )
            if 0 <= prev && prev < network.inputDim {
                oneHotInput[prev] = 0.0
            }
            let target = ids[i]
            if 0 <= target && target < outDim {
                total += log(max(1e-12, probs[target]))
            }
            prev = target
            i += 1
        }
        return total
    }

    /// 重みの読み込み
    public func importWeights(_ weights: SpikingNetworkWeights) {
        network.importWeights(from: weights)
    }

    /// 重みの書き出し
    public func exportWeights() -> SpikingNetworkWeights {
        return network.exportWeights()
    }
}

/// 辞書内の単語 Bigram / Unigram 統計を用いた統計的文脈スコアラー (O(1) メモリ & 高速)
public final class StatisticalNGramScorer: LanguageModelScorer, @unchecked Sendable {
    public let dictionary: KanaKanjiDictionary
    public let converter: KanjiConverter
    public let wordWeight: Float

    public init(
        dictionary: KanaKanjiDictionary,
        converter: KanjiConverter = KanjiConverter(),
        wordWeight: Float = 1.0
    ) {
        self.dictionary = dictionary
        self.converter = converter
        self.wordWeight = wordWeight
    }

    /// 文の対数確率 log P_LM(Y) を計算
    public func logProbability(of text: String) -> Float {
        if text.isEmpty {
            return 0.0
        }

        var totalLogProb: Float = 0.0
        let surfaces = converter.tokenizeSurfaces(text)
        if surfaces.isEmpty {
            return 0.0
        }

        var prevSurface = ""
        var i = 0
        while i < surfaces.count {
            let surface = surfaces[i].trimmingCharacters(in: .whitespacesAndNewlines)
            if surface.isEmpty != true {
                var isPunct = false
                if surface.count == 1 {
                    switch surface.first {
                    case .some(let ch):
                        isPunct = punctuationCharacters.contains(ch)
                    case .none:
                        isPunct = false
                    }
                }
                if isPunct != true {
                    let bigramProb = dictionary.logBigram(from: prevSurface, to: surface)
                    totalLogProb += wordWeight * bigramProb
                    prevSurface = surface
                }
            }
            i += 1
        }

        return totalLogProb
    }
}
