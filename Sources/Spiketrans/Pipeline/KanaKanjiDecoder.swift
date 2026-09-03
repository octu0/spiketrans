import Foundation

/// 句読点として扱う文字。第2段の各所で共有する
let punctuationCharacters: Set<Character> = ["、", "。", "，", "．"]

/// 形態素・語彙エントリ
public struct KanaKanjiEntry: Sendable {
    public let reading: String       // ひらがな読み (例: "とつぜん", "ぱにくって")
    public let surface: String       // 表記 (例: "突然", "パニクって")
    public let frequency: Int        // 出現頻度
    public let isParticle: Bool      // 助詞・文末表現フラグ

    public init(reading: String, surface: String, frequency: Int = 1, isParticle: Bool = false) {
        self.reading = reading
        self.surface = surface
        self.frequency = frequency
        self.isParticle = isParticle
    }
}

/// かな漢字変換用形態素辞書
///
/// スコアは対数確率で表す。表層の出現回数から P(表層 | 読み) を、
/// 語の連接回数から P(表層 | 直前の表層) を推定する。
public final class KanaKanjiDictionary: @unchecked Sendable {
    /// Bigram と Unigram の補間係数 (1.0 で Bigram のみ)
    private let bigramInterpolation: Float = 0.7

    private var entriesByReading: [String: [KanaKanjiEntry]] = [:]
    /// 読みごとの総出現回数
    private var readingTotals: [String: Int] = [:]
    /// 語の総出現回数
    private var surfaceTotals: [String: Int] = [:]
    /// 語の連接回数
    private var bigramCounts: [String: [String: Int]] = [:]
    /// 語彙数 (平滑化の分母に使う)
    private var vocabularySize: Int = 0
    /// 読みの長さごとのインデックス
    private var readingsByLength: [Int: [String]] = [:]
    /// 1 文字をワイルドカードにしたキーから読みを引くインデックス。
    /// 許容距離が 0.8 未満なら候補は「同じ長さで 1 文字だけ異なる読み」に限られるため、
    /// 辞書を走査せず定数個のハッシュ参照で候補を絞れる。
    private var oneSubstitutionIndex: [String: [String]] = [:]
    /// 全トークンの出現回数
    private var totalTokens: Int = 0
    /// 語の直後に読点が続く確率。句読点は音響側に存在しないため連接統計から復元する
    public private(set) var commaAfterRate: [String: Float] = [:]
    /// 文末に付く句読点とその出現率
    public private(set) var sentenceFinalPunctuation: Character? = nil
    public private(set) var sentenceFinalRate: Float = 0.0

    public init() {}

    public var count: Int {
        return entriesByReading.count
    }

    public var allReadings: [String] {
        return Array(entriesByReading.keys)
    }

    /// エントリを追加 (同じ読み・表層の組は出現回数を加算する)
    public func addEntry(_ entry: KanaKanjiEntry) {
        let reading = entry.reading
        var list = entriesByReading[reading] ?? []

        var found = false
        var i = 0
        while i < list.count {
            if list[i].surface == entry.surface {
                list[i] = KanaKanjiEntry(
                    reading: reading,
                    surface: entry.surface,
                    frequency: list[i].frequency + entry.frequency,
                    isParticle: list[i].isParticle
                )
                found = true
                break
            }
            i += 1
        }
        if found != true {
            list.append(entry)
        }

        if entriesByReading[reading] == nil {
            readingsByLength[reading.count, default: []].append(reading)
            for key in Self.wildcardKeys(for: reading) {
                oneSubstitutionIndex[key, default: []].append(reading)
            }
        }
        entriesByReading[reading] = list
        readingTotals[reading, default: 0] += entry.frequency
        surfaceTotals[entry.surface, default: 0] += entry.frequency
        totalTokens += entry.frequency
    }

    /// 語の連接を記録
    public func addBigram(from: String, to: String) {
        var map = bigramCounts[from] ?? [:]
        map[to, default: 0] += 1
        bigramCounts[from] = map
    }

    /// 1 文字をワイルドカードに置き換えたキー列
    static func wildcardKeys(for reading: String) -> [String] {
        let chars = Array(reading)
        var keys: [String] = []
        keys.reserveCapacity(chars.count)
        var i = 0
        while i < chars.count {
            var key = ""
            var j = 0
            while j < chars.count {
                if j == i {
                    key.append("*")
                } else {
                    key.append(chars[j])
                }
                j += 1
            }
            keys.append(key)
            i += 1
        }
        return keys
    }

    /// log P(読み | 表層)
    ///
    /// 語がその読みで現れる割合。表層が決まれば読みはほぼ一意なので通常 0 に近い。
    public func logReadingGivenSurface(reading: String, surface: String) -> Float {
        let surfaceTotal = surfaceTotals[surface] ?? 0
        if surfaceTotal <= 0 {
            return -.infinity
        }
        var pairCount = 0
        for e in entriesByReading[reading] ?? [] {
            if e.surface == surface {
                pairCount = e.frequency
                break
            }
        }
        if pairCount <= 0 {
            return -.infinity
        }
        return log(Float(pairCount) / Float(surfaceTotal))
    }

    /// log P(表層 | 直前の表層)
    ///
    /// Bigram と Unigram の補間平滑化。未観測の連接でも Unigram に退避するため、
    /// 加算平滑化のように全ての未観測連接が同じ値に張り付くことがない。
    public func logBigram(from: String, to: String) -> Float {
        let total = Float(max(1, totalTokens))
        let unigram = Float(surfaceTotals[to] ?? 0) / total

        let prevTotal = Float(surfaceTotals[from] ?? 0)
        var bigram: Float = 0.0
        if 0 < prevTotal {
            bigram = Float(bigramCounts[from]?[to] ?? 0) / prevTotal
        }

        let mixed = (bigramInterpolation * bigram) + ((1.0 - bigramInterpolation) * unigram)
        if mixed <= 0.0 {
            // 辞書に存在しない語への遷移
            return log(1.0 / (total * total))
        }
        return log(mixed)
    }

    /// 読み完全一致の候補を取得
    public func lookupExact(reading: String) -> [KanaKanjiEntry] {
        return entriesByReading[reading] ?? []
    }

    /// コーパステキストから語彙辞書および Bigram 連接確率を自動構築
    ///
    /// 表層と読みは同一の形態素解析結果から取り出す。読みを別経路で推定すると
    /// 第1段が出力するかなと辞書の読みが一致せず、辞書引きが成立しない。
    public func buildFromCorpus(rawTexts: [String], converter: KanjiConverter = KanjiConverter()) {
        var commaFollows: [String: Int] = [:]
        var wordOccurrences: [String: Int] = [:]
        var finalPunctCounts: [Character: Int] = [:]
        var sentenceCount = 0

        for text in rawTexts {
            var prevWord: String? = nil

            // 文末の句読点を集計
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty != true {
                sentenceCount += 1
                if let last = trimmed.last {
                    if punctuationCharacters.contains(last) {
                        finalPunctCounts[last, default: 0] += 1
                    }
                }
            }

            // 読みを持つ語と句読点を同じトークン列上で辿り、語の直後の読点を数える
            var lastContentWord: String? = nil
            for token in converter.tokenize(text) {
                let surface = token.surface.trimmingCharacters(in: .whitespacesAndNewlines)
                if surface.isEmpty {
                    continue
                }

                // 句読点トークンは読みを持たない。直前の語との連接として記録する
                if surface.count == 1 {
                    if let ch = surface.first {
                        if punctuationCharacters.contains(ch) {
                            switch lastContentWord {
                            case .some(let w):
                                if ch == "、" {
                                    commaFollows[w, default: 0] += 1
                                }
                            case .none:
                                break
                            }
                            continue
                        }
                    }
                }

                let reading = token.reading
                if reading.isEmpty {
                    continue
                }

                addEntry(KanaKanjiEntry(reading: reading, surface: surface, frequency: 1))
                wordOccurrences[surface, default: 0] += 1
                lastContentWord = surface

                switch prevWord {
                case .some(let pWord):
                    addBigram(from: pWord, to: surface)
                case .none:
                    ()
                }

                prevWord = surface
            }
        }

        // 文末句読点: 最も多い句読点とその出現率
        var bestFinal: Character? = nil
        var bestFinalCount = 0
        for (ch, cnt) in finalPunctCounts {
            if bestFinalCount < cnt {
                bestFinalCount = cnt
                bestFinal = ch
            }
        }
        sentenceFinalPunctuation = bestFinal
        if 0 < sentenceCount {
            sentenceFinalRate = Float(bestFinalCount) / Float(sentenceCount)
        }

        // 読点の連接率
        for (w, cnt) in commaFollows {
            let total = wordOccurrences[w] ?? 0
            if 0 < total {
                commaAfterRate[w] = Float(cnt) / Float(total)
            }
        }

        vocabularySize = surfaceTotals.count
    }


    /// 音素調音距離によるファジー検索
    ///
    /// 挿入・削除のコストは 1 文字 0.8 なので、許容距離が 0.8 未満なら
    /// 候補は読み長が一致するものだけに限られる。その場合は対角比較で
    /// 距離を求められ、コスト超過時点で打ち切れる。
    public func lookupFuzzyPhonetic(reading: String, maxPhoneticDist: Float = 1.25) -> [(entry: KanaKanjiEntry, dist: Float)] {
        var results: [(entry: KanaKanjiEntry, dist: Float)] = []
        let queryChars = Array(reading)
        let rLen = queryChars.count

        // 挿入・削除は 1 文字 0.8 かかるため、許容距離が 0.8 未満なら
        // 候補は「同じ長さで 1 文字だけ異なる読み」に限られる。
        // その場合はワイルドカード索引で候補を直接引ける。
        if maxPhoneticDist < 0.8 {
            var seen = Set<String>()
            for key in Self.wildcardKeys(for: reading) {
                for candReading in oneSubstitutionIndex[key] ?? [] {
                    if candReading == reading || seen.contains(candReading) {
                        continue
                    }
                    seen.insert(candReading)
                    let dist = diagonalDistance(queryChars, Array(candReading), limit: maxPhoneticDist)
                    if dist <= maxPhoneticDist {
                        for e in entriesByReading[candReading] ?? [] {
                            results.append((entry: e, dist: dist))
                        }
                    }
                }
            }
            return results
        }

        // 許容距離が大きい場合は長さ範囲を絞って編集距離で評価する
        let maxLengthDiff = Int(maxPhoneticDist / 0.8)
        var length = rLen - maxLengthDiff
        while length <= rLen + maxLengthDiff {
            if length <= 0 {
                length += 1
                continue
            }
            for candReading in readingsByLength[length] ?? [] {
                let dist = phoneticDistance(reading, candReading)
                if dist <= maxPhoneticDist {
                    for e in entriesByReading[candReading] ?? [] {
                        results.append((entry: e, dist: dist))
                    }
                }
            }
            length += 1
        }

        return results
    }

    /// 同じ長さの読み同士を対角整列で比較する。累積コストが limit を超えたら打ち切る。
    ///
    /// 挿入・削除が 0.8 なので、limit が 1.6 未満なら対角以外の整列は成立しない。
    private func diagonalDistance(_ a: [Character], _ b: [Character], limit: Float) -> Float {
        var total: Float = 0.0
        var i = 0
        while i < a.count {
            if a[i] != b[i] {
                total += charSubstitutionCost(a[i], b[i])
                if limit < total {
                    return .infinity
                }
            }
            i += 1
        }
        return total
    }

    /// 音素・調音位置類似度に基づく連続編集距離
    public func phoneticDistance(_ s1: String, _ s2: String) -> Float {
        let a1 = Array(s1)
        let a2 = Array(s2)
        let m = a1.count
        let n = a2.count
        if m == 0 { return Float(n) * 0.8 }
        if n == 0 { return Float(m) * 0.8 }

        var dp = [[Float]](repeating: [Float](repeating: 0.0, count: n + 1), count: m + 1)
        var i = 0
        while i <= m {
            dp[i][0] = Float(i) * 0.8
            i += 1
        }
        var j = 0
        while j <= n {
            dp[0][j] = Float(j) * 0.8
            j += 1
        }

        i = 1
        while i <= m {
            j = 1
            while j <= n {
                let c1 = a1[i - 1]
                let c2 = a2[j - 1]
                let substCost = charSubstitutionCost(c1, c2)

                let d1 = dp[i - 1][j] + 0.8
                let d2 = dp[i][j - 1] + 0.8
                let d3 = dp[i - 1][j - 1] + substCost

                var minVal = d1
                if d2 < minVal { minVal = d2 }
                if d3 < minVal { minVal = d3 }
                dp[i][j] = minVal
                j += 1
            }
            i += 1
        }
        return dp[m][n]
    }

    /// 2文字間の調音位置類似コスト
    private func charSubstitutionCost(_ c1: Character, _ c2: Character) -> Float {
        if c1 == c2 {
            return 0.0
        }

        // 清音・濁音・半濁音の調音類似ペア
        let pairStr = "\(c1)\(c2)"
        let revPairStr = "\(c2)\(c1)"

        let voicedPairs: Set<String> = [
            "かが", "きぎ", "くぐ", "けげ", "こご",
            "さざ", "しじ", "すず", "せぜ", "そぞ",
            "ただ", "ちぢ", "つづ", "てで", "とど",
            "はば", "ひび", "ふぶ", "へべ", "ほぼ",
            "はぱ", "ひぴ", "ふぷ", "へぺ", "ほぽ",
            "ばぱ", "びぴ", "ぶぷ", "べぺ", "ぼぽ"
        ]

        if voicedPairs.contains(pairStr) || voicedPairs.contains(revPairStr) {
            return 0.35
        }

        // 促音・長音・撥音・重複母音のゆらぎ
        let specialChars: Set<Character> = ["っ", "ー", "ん", "う", "い"]
        if specialChars.contains(c1) || specialChars.contains(c2) {
            return 0.40
        }

        // 類似母音ペア
        let vowelPairs: Set<String> = ["おう", "うお", "えい", "いえ", "あお", "おあ"]
        if vowelPairs.contains(pairStr) || vowelPairs.contains(revPairStr) {
            return 0.45
        }

        return 1.0
    }
}

/// 第2段 堅牢なかな漢字ビーム探索統合デコーダ (音素調音距離 ＆ Bigram 連接確率大域最適化)
public final class KanaKanjiDecoder: @unchecked Sendable {
    /// 辞書引きの最大読み長
    private static let maxWordLength = 12
    /// ファジー一致を許す最小読み長
    private static let minFuzzyLength = 3
    /// ファジー一致で許容する最大調音距離
    private static let maxFuzzyDistance: Float = 0.6
    /// 調音距離 1 あたりのコスト。
    /// かなの取り違えが起きる確率を対数で表す。完全一致を明確に優先させる。
    private static let fuzzyDistanceCost: Float = -12.0
    /// 辞書に無いかなを 1 文字そのまま通すときの対数確率
    private static let unknownCharLogProb: Float = -12.0
    /// Viterbi のビーム幅。
    /// スコアが直前の語に依存するため位置ごとに 1 状態では厳密解にならない。
    /// 複数状態を保持することで、直前の語が異なる有望な経路を残す。
    private static let defaultBeamWidth = 5

    public let dictionary: KanaKanjiDictionary
    public let languageDecoder: LanguageDecoder?
    /// 言語 SNN に入力するかな側の語彙 (languageDecoder 併用時に必須)
    public let kanaVocabulary: TextVocabulary?
    /// 言語 SNN のスライス
    /// 言語 SNN の予測と一致した 1 文字あたりの加点 (0.0 で言語 SNN を無効化)
    public let languageBonus: Float
    /// Bigram 項の重み
    public let bigramWeight: Float
    /// 語の直後に読点を付ける連接率の閾値
    public let commaThreshold: Float
    /// 文末句読点を付ける出現率の閾値
    public let sentenceFinalThreshold: Float

    /// 直近の decode で選ばれた区間の内訳 (診断用)
    public private(set) var lastTrace: [TraceSegment] = []

    public init(
        dictionary: KanaKanjiDictionary,
        languageDecoder: LanguageDecoder? = nil,
        kanaVocabulary: TextVocabulary? = nil,
        languageBonus: Float = 4.0,
        bigramWeight: Float = 1.0,
        commaThreshold: Float = 0.5,
        sentenceFinalThreshold: Float = 0.5
    ) {
        self.dictionary = dictionary
        self.languageDecoder = languageDecoder
        self.kanaVocabulary = kanaVocabulary
        self.languageBonus = languageBonus
        self.bigramWeight = bigramWeight
        self.commaThreshold = commaThreshold
        self.sentenceFinalThreshold = sentenceFinalThreshold
    }

    /// 区間 [start, end) のかなに対する言語 SNN 予測と表記の一致文字数を数える
    private func languageAgreement(
        surface: String,
        hints: [Character?],
        start: Int,
        end: Int
    ) -> Float {
        if hints.isEmpty {
            return 0.0
        }

        var available: [Character] = []
        available.reserveCapacity(end - start)
        var h = start
        while h < end {
            if h < hints.count {
                switch hints[h] {
                case .some(let c):
                    available.append(c)
                case .none:
                    break
                }
            }
            h += 1
        }
        if available.isEmpty {
            return 0.0
        }

        var matched: Float = 0.0
        for ch in surface {
            var found = -1
            var a = 0
            while a < available.count {
                if available[a] == ch {
                    found = a
                    break
                }
                a += 1
            }
            if 0 <= found {
                available.remove(at: found)
                matched += 1.0
            }
        }
        return matched
    }

    /// Viterbi が選んだ区間の内訳
    public struct TraceSegment: Sendable {
        public let kanaRange: String
        public let emitted: String
        public let kind: String
        public let stepScore: Float
    }

    /// 語の直後に付ける読点。学習コーパスで過半数がその語の後に読点を伴う場合に付与する
    private func commaSuffix(after surface: String) -> String {
        switch dictionary.commaAfterRate[surface] {
        case .some(let rate):
            if commaThreshold <= rate {
                return "、"
            }
            return ""
        case .none:
            return ""
        }
    }

    /// 文末句読点。学習コーパスでの出現率が閾値を超える場合に付与する
    private func sentenceFinalSuffix() -> String {
        switch dictionary.sentenceFinalPunctuation {
        case .some(let ch):
            if sentenceFinalThreshold <= dictionary.sentenceFinalRate {
                return String(ch)
            }
            return ""
        case .none:
            return ""
        }
    }

    /// K-best 候補とそのスコア
    public struct Candidate: Sendable {
        public let text: String
        public let score: Float
    }

    /// 上位 K 件の候補を返す。
    ///
    /// 言語モデルによる再スコアリングの入力に使う。各位置で上位 K 状態を保持する
    /// ビーム付き Viterbi で、K = 1 なら通常の Viterbi と一致する。
    public func decodeNBest(kanaText: String, beamWidth: Int) -> [Candidate] {
        if kanaText.isEmpty {
            return []
        }
        let k = max(1, beamWidth)
        let chars = Array(kanaText)
        let n = chars.count

        struct BeamState {
            var score: Float
            var text: String
            var lastWord: String
            var trace: [TraceSegment]
        }

        var beams = [[BeamState]](repeating: [], count: n + 1)
        beams[0] = [BeamState(score: 0.0, text: "", lastWord: "", trace: [])]

        /// 候補を追加し、上位 K 件だけ残す
        func push(_ index: Int, _ state: BeamState) {
            var list = beams[index]
            list.append(state)
            list.sort { b, a in a.score < b.score }
            if k < list.count {
                list.removeLast(list.count - k)
            }
            beams[index] = list
        }

        var i = 0
        while i < n {
            if beams[i].isEmpty {
                i += 1
                continue
            }
            let currentStates = beams[i]

            let maxL = min(Self.maxWordLength, n - i)
            var l = maxL
            while 1 <= l {
                let subStr = String(chars[i..<(i + l)])
                let exactEntries = dictionary.lookupExact(reading: subStr)

                for entry in exactEntries {
                    let emissionLogProb = dictionary.logReadingGivenSurface(reading: subStr, surface: entry.surface)
                    if emissionLogProb.isFinite != true {
                        continue
                    }
                    let emitted = entry.surface + commaSuffix(after: entry.surface)
                    for st in currentStates {
                        let bigramLogProb = dictionary.logBigram(from: st.lastWord, to: entry.surface)
                        let stepScore = emissionLogProb + (bigramWeight * bigramLogProb)
                        push(i + l, BeamState(
                            score: st.score + stepScore,
                            text: st.text + emitted,
                            lastWord: entry.surface,
                            trace: st.trace + [TraceSegment(
                                kanaRange: subStr, emitted: emitted,
                                kind: "完全一致", stepScore: stepScore
                            )]
                        ))
                    }
                }

                if Self.minFuzzyLength <= l && exactEntries.isEmpty {
                    let fuzzyList = dictionary.lookupFuzzyPhonetic(reading: subStr, maxPhoneticDist: Self.maxFuzzyDistance)
                    for item in fuzzyList {
                        let entry = item.entry
                        let emissionLogProb = dictionary.logReadingGivenSurface(reading: entry.reading, surface: entry.surface)
                        if emissionLogProb.isFinite != true {
                            continue
                        }
                        let emitted = entry.surface + commaSuffix(after: entry.surface)
                        let fuzzyCost = Self.fuzzyDistanceCost * item.dist
                        for st in currentStates {
                            let bigramLogProb = dictionary.logBigram(from: st.lastWord, to: entry.surface)
                            let stepScore = emissionLogProb + (bigramWeight * bigramLogProb) + fuzzyCost
                            push(i + l, BeamState(
                                score: st.score + stepScore,
                                text: st.text + emitted,
                                lastWord: entry.surface,
                                trace: st.trace + [TraceSegment(
                                    kanaRange: subStr, emitted: emitted,
                                    kind: "ファジー(距離 \(String(format: "%.2f", item.dist)))",
                                    stepScore: stepScore
                                )]
                            ))
                        }
                    }
                }

                l -= 1
            }

            let singleChar = String(chars[i])
            for st in currentStates {
                push(i + 1, BeamState(
                    score: st.score + Self.unknownCharLogProb,
                    text: st.text + singleChar,
                    lastWord: singleChar,
                    trace: st.trace + [TraceSegment(
                        kanaRange: singleChar, emitted: singleChar,
                        kind: "1文字スルー", stepScore: Self.unknownCharLogProb
                    )]
                ))
            }

            i += 1
        }

        let suffix = sentenceFinalSuffix()
        if let best = beams[n].first {
            lastTrace = best.trace
        }
        return beams[n].map { st in
            var restored = st.text
            if let last = restored.last {
                if punctuationCharacters.contains(last) {
                    restored.removeLast()
                }
            }
            return Candidate(text: restored + suffix, score: st.score)
        }
    }

    /// 音響 SNN の出力かな文字列から漢字かな混じり文を復元する
    ///
    /// スコアが直前の語に依存するため、位置ごとに 1 状態だけ残す Viterbi では
    /// 厳密解にならない。ビーム付き探索の最良候補を返す。
    public func decode(kanaText: String) -> String {
        let candidates = decodeNBest(kanaText: kanaText, beamWidth: Self.defaultBeamWidth)
        switch candidates.first {
        case .some(let best):
            return best.text
        case .none:
            return ""
        }
    }
}
