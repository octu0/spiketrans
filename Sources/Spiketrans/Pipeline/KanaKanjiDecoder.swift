import Foundation

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
    /// 句読点として扱う文字
    static let punctuation: Set<Character> = ["、", "。", "，", "．"]

    /// Bigram の加算平滑化パラメータ
    private let bigramSmoothing: Float = 0.1

    private var entriesByReading: [String: [KanaKanjiEntry]] = [:]
    /// 読みごとの総出現回数
    private var readingTotals: [String: Int] = [:]
    /// 語の総出現回数
    private var surfaceTotals: [String: Int] = [:]
    /// 語の連接回数
    private var bigramCounts: [String: [String: Int]] = [:]
    /// 語彙数 (平滑化の分母に使う)
    private var vocabularySize: Int = 0
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

        entriesByReading[reading] = list
        readingTotals[reading, default: 0] += entry.frequency
        surfaceTotals[entry.surface, default: 0] += entry.frequency
    }

    /// 語の連接を記録
    public func addBigram(from: String, to: String) {
        var map = bigramCounts[from] ?? [:]
        map[to, default: 0] += 1
        bigramCounts[from] = map
    }

    /// log P(表層 | 読み)
    public func logEmission(reading: String, surface: String) -> Float {
        let total = readingTotals[reading] ?? 0
        if total <= 0 {
            return -.infinity
        }
        var surfaceCount = 0
        for e in entriesByReading[reading] ?? [] {
            if e.surface == surface {
                surfaceCount = e.frequency
                break
            }
        }
        if surfaceCount <= 0 {
            return -.infinity
        }
        return log(Float(surfaceCount) / Float(total))
    }

    /// log P(表層 | 直前の表層)。加算平滑化のため未観測の連接でも有限値を返す
    public func logBigram(from: String, to: String) -> Float {
        let v = Float(max(1, vocabularySize))
        let prevTotal = Float(surfaceTotals[from] ?? 0)
        let pairCount = Float(bigramCounts[from]?[to] ?? 0)
        let numerator = pairCount + bigramSmoothing
        let denominator = prevTotal + (bigramSmoothing * v)
        return log(numerator / denominator)
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
                    if Self.punctuation.contains(last) {
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
                        if Self.punctuation.contains(ch) {
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


    /// 音素調音距離テーブルによるファジー検索 (Phonetic Distance <= maxPhoneticDist)
    public func lookupFuzzyPhonetic(reading: String, maxPhoneticDist: Float = 1.25) -> [(entry: KanaKanjiEntry, dist: Float)] {
        var results: [(entry: KanaKanjiEntry, dist: Float)] = []
        let rLen = reading.count

        for (candReading, candEntries) in entriesByReading {
            let diff = abs(candReading.count - rLen)
            if 3 < diff {
                continue
            }

            let dist = phoneticDistance(reading, candReading)
            if dist <= maxPhoneticDist {
                for e in candEntries {
                    results.append((entry: e, dist: dist))
                }
            }
        }

        return results
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

    /// 音響 SNN の出力かな文字列から Viterbi DP 大域最適化により漢字かな混じり文を復元
    public func decode(kanaText: String) -> String {
        if kanaText.isEmpty {
            return ""
        }

        let chars = Array(kanaText)
        let n = chars.count

        // 第2段 言語 SNN による、かな 1 文字ごとの漢字予測 (DP の区間スコアに加算)
        var lmHints: [Character?] = []
        switch languageDecoder {
        case .some(let lmDecoder):
            switch kanaVocabulary {
            case .some(let kanaVocab):
                if 0.0 < languageBonus {
                    lmHints = lmDecoder.predictKanjiPerKana(
                        kanaText: kanaText,
                        kanaVocabulary: kanaVocab
                    )
                }
            case .none:
                break
            }
        case .none:
            break
        }

        struct DPState {
            var score: Float
            var text: String
            var lastWord: String
            // トレース用: この状態に至った直前位置・出力・経路種別・区間スコア
            var prevIndex: Int
            var emitted: String
            var kind: String
            var stepScore: Float
        }

        var dp = [DPState](
            repeating: DPState(
                score: -.infinity, text: "", lastWord: "",
                prevIndex: -1, emitted: "", kind: "", stepScore: 0.0
            ),
            count: n + 1
        )
        dp[0] = DPState(
            score: 0.0, text: "", lastWord: "",
            prevIndex: -1, emitted: "", kind: "start", stepScore: 0.0
        )

        var i = 0
        while i < n {
            if dp[i].score.isInfinite {
                i += 1
                continue
            }

            let curScore = dp[i].score
            let curText = dp[i].text
            let curLastWord = dp[i].lastWord

            // 1. 辞書引きによる語の遷移。助詞も辞書に載っているため特別扱いはしない
            let maxL = min(Self.maxWordLength, n - i)
            var l = maxL
            while 1 <= l {
                let subStr = String(chars[i..<(i + l)])
                let exactEntries = dictionary.lookupExact(reading: subStr)

                for entry in exactEntries {
                    let emissionLogProb = dictionary.logEmission(reading: subStr, surface: entry.surface)
                    if emissionLogProb.isFinite != true {
                        continue
                    }
                    let bigramLogProb = dictionary.logBigram(from: curLastWord, to: entry.surface)
                    let lmScore = languageBonus * languageAgreement(
                        surface: entry.surface,
                        hints: lmHints,
                        start: i,
                        end: i + l
                    )
                    let nextScore = curScore + emissionLogProb + (bigramWeight * bigramLogProb) + lmScore

                    if dp[i + l].score < nextScore {
                        let emitted = entry.surface + commaSuffix(after: entry.surface)
                        dp[i + l] = DPState(
                            score: nextScore, text: curText + emitted, lastWord: entry.surface,
                            prevIndex: i, emitted: emitted, kind: "完全一致",
                            stepScore: nextScore - curScore
                        )
                    }
                }

                // 2. 完全一致が無い区間のみ、音素調音距離によるファジー一致を許す。
                //    第1段のかな誤りを吸収するための経路で、距離に比例したコストを課す。
                if Self.minFuzzyLength <= l && exactEntries.isEmpty {
                    let fuzzyList = dictionary.lookupFuzzyPhonetic(reading: subStr, maxPhoneticDist: Self.maxFuzzyDistance)
                    for item in fuzzyList {
                        let entry = item.entry
                        let emissionLogProb = dictionary.logEmission(reading: entry.reading, surface: entry.surface)
                        if emissionLogProb.isFinite != true {
                            continue
                        }
                        let bigramLogProb = dictionary.logBigram(from: curLastWord, to: entry.surface)
                        let lmScore = languageBonus * languageAgreement(
                            surface: entry.surface,
                            hints: lmHints,
                            start: i,
                            end: i + l
                        )
                        let nextScore = curScore
                            + emissionLogProb
                            + (bigramWeight * bigramLogProb)
                            + (Self.fuzzyDistanceCost * item.dist)
                            + lmScore

                        if dp[i + l].score < nextScore {
                            let emitted = entry.surface + commaSuffix(after: entry.surface)
                            dp[i + l] = DPState(
                                score: nextScore, text: curText + emitted, lastWord: entry.surface,
                                prevIndex: i, emitted: emitted,
                                kind: "ファジー(距離 \(String(format: "%.2f", item.dist)))",
                                stepScore: nextScore - curScore
                            )
                        }
                    }
                }

                l -= 1
            }

            // 3. 辞書に無いかなは 1 文字そのまま通す。語を 1 つ使うより高コストにする
            let singleChar = String(chars[i])
            let passScore = curScore + Self.unknownCharLogProb
            if dp[i + 1].score < passScore {
                dp[i + 1] = DPState(
                    score: passScore, text: curText + singleChar, lastWord: singleChar,
                    prevIndex: i, emitted: singleChar, kind: "1文字スルー",
                    stepScore: Self.unknownCharLogProb
                )
            }

            i += 1
        }

        // 選択経路を復元
        var trace: [TraceSegment] = []
        var walk = n
        while 0 < walk {
            let st = dp[walk]
            if st.prevIndex < 0 {
                break
            }
            trace.append(TraceSegment(
                kanaRange: String(chars[st.prevIndex..<walk]),
                emitted: st.emitted,
                kind: st.kind,
                stepScore: st.stepScore
            ))
            walk = st.prevIndex
        }
        lastTrace = trace.reversed()

        var restored = dp[n].text
        // 文末の句読点が重複しないように整える
        if let last = restored.last {
            if Self.punctuationChars.contains(last) {
                restored.removeLast()
            }
        }
        return restored + sentenceFinalSuffix()
    }

    /// 句読点として扱う文字
    private static let punctuationChars: Set<Character> = ["、", "。", "，", "．"]

    /// 辞書引きの最大読み長
    private static let maxWordLength = 12
    /// ファジー一致を許す最小読み長
    private static let minFuzzyLength = 3
    /// ファジー一致で許容する最大調音距離
    private static let maxFuzzyDistance: Float = 0.6
    /// 調音距離 1 あたりのコスト (対数確率に加算する負の値)
    private static let fuzzyDistanceCost: Float = -4.0
    /// 辞書に無いかなを 1 文字そのまま通すときの対数確率
    private static let unknownCharLogProb: Float = -12.0

    /// 直近の decode で選ばれた区間の内訳 (診断用)
    public private(set) var lastTrace: [TraceSegment] = []
}
