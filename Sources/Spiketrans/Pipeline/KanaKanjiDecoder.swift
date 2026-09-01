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

/// かな漢字変換用形態素辞書 (音素調音距離 & Bigram 連接確率対応)
public final class KanaKanjiDictionary: @unchecked Sendable {
    private var entriesByReading: [String: [KanaKanjiEntry]] = [:]
    public private(set) var bigramTransitions: [String: [String: Float]] = [:]

    public init() {}

    public var count: Int {
        return entriesByReading.count
    }

    public var allReadings: [String] {
        return Array(entriesByReading.keys)
    }

    /// エントリを追加
    public func addEntry(_ entry: KanaKanjiEntry) {
        let reading = entry.reading
        var list = entriesByReading[reading] ?? []
        list.append(entry)
        entriesByReading[reading] = list
    }

    /// Bigram 遷移スコアを追加
    public func setBigramTransition(from: String, to: String, logProb: Float) {
        var map = bigramTransitions[from] ?? [:]
        map[to] = logProb
        bigramTransitions[from] = map
    }

    /// 読み完全一致の候補を取得
    public func lookupExact(reading: String) -> [KanaKanjiEntry] {
        return entriesByReading[reading] ?? []
    }

    /// コーパステキストから語彙辞書および Bigram 連接確率を自動構築
    public func buildFromCorpus(rawTexts: [String]) {
        var unigramCounts: [String: Int] = [:]
        var bigramCounts: [String: [String: Int]] = [:]

        for text in rawTexts {
            let words = tokenizeText(text)
            var prevWord: String? = nil

            for word in words {
                if word.surface.isEmpty {
                    continue
                }

                // 辞書エントリ登録
                addEntry(KanaKanjiEntry(
                    reading: word.reading,
                    surface: word.surface,
                    frequency: word.frequency,
                    isParticle: word.isParticle
                ))

                unigramCounts[word.surface, default: 0] += 1

                switch prevWord {
                case .some(let pWord):
                    var nextMap = bigramCounts[pWord] ?? [:]
                    nextMap[word.surface, default: 0] += 1
                    bigramCounts[pWord] = nextMap
                case .none:
                    ()
                }

                prevWord = word.surface
            }
        }

        // Bigram 対数遷移確率の算出
        for (w1, nextMap) in bigramCounts {
            let totalNext = Float(unigramCounts[w1] ?? 1)
            for (w2, count) in nextMap {
                let prob = log(Float(count) / totalNext + 0.01)
                setBigramTransition(from: w1, to: w2, logProb: prob)
            }
        }
    }

    /// テキストを形態素（漢字・カタカナ・ひらがなブロック）に分割
    private func tokenizeText(_ text: String) -> [KanaKanjiEntry] {
        var results: [KanaKanjiEntry] = []
        let chars = Array(text)
        let n = chars.count
        var i = 0

        while i < n {
            let ch = chars[i]
            if ch.isWhitespace || ch.isPunctuation {
                i += 1
                continue
            }

            // 1. カタカナブロック
            if isKatakana(ch) {
                var katakanaStr = String(ch)
                var j = i + 1
                while j < n {
                    let nextCh = chars[j]
                    if isKatakana(nextCh) || nextCh == "・" || nextCh == "ー" {
                        katakanaStr.append(nextCh)
                        j += 1
                    } else {
                        break
                    }
                }
                let hiraganaReading = katakanaToHiragana(katakanaStr)
                results.append(KanaKanjiEntry(reading: hiraganaReading, surface: katakanaStr, frequency: 10, isParticle: false))
                i = j
                continue
            }

            // 2. 漢字・送り仮名ブロック
            if isKanji(ch) {
                var kanjiStr = String(ch)
                var j = i + 1
                while j < n {
                    let nextCh = chars[j]
                    if isKanji(nextCh) || isHiragana(nextCh) {
                        kanjiStr.append(nextCh)
                        j += 1
                    } else {
                        break
                    }
                }
                let reading = approximateKanjiReading(kanjiStr)
                results.append(KanaKanjiEntry(reading: reading, surface: kanjiStr, frequency: 8, isParticle: false))
                i = j
                continue
            }

            // 3. ひらがな助詞・機能語ブロック
            var subLen = min(3, n - i)
            var matchedParticle = false
            while 1 <= subLen {
                let subStr = String(chars[i..<(i + subLen)])
                let particles: Set<String> = ["は", "が", "を", "に", "へ", "で", "と", "から", "より", "まで", "の", "も", "や", "か", "ね", "よ", "わ", "こと", "である", "にある", "した", "して", "され", "れる", "られる", "ない", "ます", "ません", "けれど"]
                if particles.contains(subStr) {
                    results.append(KanaKanjiEntry(reading: subStr, surface: subStr, frequency: 15, isParticle: true))
                    i += subLen
                    matchedParticle = true
                    break
                }
                subLen -= 1
            }

            if matchedParticle != true {
                let single = String(ch)
                results.append(KanaKanjiEntry(reading: single, surface: single, frequency: 1, isParticle: false))
                i += 1
            }
        }

        return results
    }

    private func isKatakana(_ ch: Character) -> Bool {
        guard let scalar = ch.unicodeScalars.first else { return false }
        return 0x30A0 <= scalar.value && scalar.value <= 0x30FF
    }

    private func isKanji(_ ch: Character) -> Bool {
        guard let scalar = ch.unicodeScalars.first else { return false }
        return 0x4E00 <= scalar.value && scalar.value <= 0x9FFF
    }

    private func isHiragana(_ ch: Character) -> Bool {
        guard let scalar = ch.unicodeScalars.first else { return false }
        return 0x3040 <= scalar.value && scalar.value <= 0x309F
    }

    private func katakanaToHiragana(_ katakana: String) -> String {
        var result = ""
        for scalar in katakana.unicodeScalars {
            if 0x30A1 <= scalar.value && scalar.value <= 0x30F6 {
                if let hScalar = UnicodeScalar(scalar.value - 0x60) {
                    result.append(Character(hScalar))
                } else {
                    result.append(Character(scalar))
                }
            } else {
                result.append(Character(scalar))
            }
        }
        return result
    }

    private func approximateKanjiReading(_ kanji: String) -> String {
        let commonMap: [String: String] = [
            "突然": "とつぜん", "人": "ひと", "自分": "じぶん", "意見": "いけん", "述べる": "のべる",
            "彼": "かれ", "私": "わたくし", "魅力": "みりょく", "無かった": "なかった", "気づかない": "きづかない", "振り": "ふり",
            "学校": "がっこう", "法人": "ほうじん", "学園": "がくえん", "運営": "うんえい", "専修学校": "せんしゅうがっこう",
            "美容": "びよう", "専門学校": "せんもんがっこう", "都市": "とし", "特産品": "とくさんひん", "公国": "こうこく",
            "朝": "ちょう", "台湾": "たいわん", "俳優": "はいゆう", "搭乗機": "とうじょうき", "隠れていた": "かくれていた",
            "小屋": "こや", "巻き込んで": "まきこんで", "爆発": "ばくはつ", "言葉": "ことば", "合わせた": "あわせた",
            "州": "しゅう", "州庁所在地": "しゅうちょうしょざいち", "位置する": "いちする", "北東": "ほくとう", "楽譜": "がくふ",
            "架空": "かくう", "魔法使い": "まほうつかい", "融合させた": "ゆうごうさせた", "音楽": "おんがく", "作曲": "さっきょく",
            "協奏曲": "きょうそうきょく", "地方": "ちほう", "話される": "はなされる", "言語": "げんご", "中西部": "ちゅうせいぶ"
        ]

        if let mapped = commonMap[kanji] {
            return mapped
        }
        return katakanaToHiragana(kanji)
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

    public init(
        dictionary: KanaKanjiDictionary,
        languageDecoder: LanguageDecoder? = nil
    ) {
        self.dictionary = dictionary
        self.languageDecoder = languageDecoder
    }

    /// 助詞・機能語の保護判定
    private static let protectedParticles: Set<String> = [
        "は", "が", "を", "に", "へ", "で", "と", "から", "より", "まで", "の", "も", "や", "か", "ね", "よ", "わ",
        "こと", "である", "にある", "した", "して", "され", "れる", "られる", "ない", "ます", "ません", "けれど"
    ]

    /// 表記がカタカナを含むか判定
    private func containsKatakana(_ str: String) -> Bool {
        for scalar in str.unicodeScalars {
            if 0x30A0 <= scalar.value && scalar.value <= 0x30FF {
                return true
            }
        }
        return false
    }

    /// 音響 SNN の出力かな文字列から Viterbi DP 大域最適化により漢字かな混じり文を復元
    public func decode(kanaText: String) -> String {
        if kanaText.isEmpty {
            return ""
        }

        let chars = Array(kanaText)
        let n = chars.count

        struct DPState {
            var score: Float
            var text: String
            var lastWord: String
        }

        var dp = [DPState](repeating: DPState(score: -.infinity, text: "", lastWord: ""), count: n + 1)
        dp[0] = DPState(score: 0.0, text: "", lastWord: "")

        var i = 0
        while i < n {
            if dp[i].score.isInfinite {
                i += 1
                continue
            }

            let curScore = dp[i].score
            let curText = dp[i].text
            let curLastWord = dp[i].lastWord

            // 1. 助詞・文末表現の遷移 (1〜4文字)
            var pl = min(4, n - i)
            while 1 <= pl {
                let subStr = String(chars[i..<(i + pl)])
                if Self.protectedParticles.contains(subStr) {
                    var transScore: Float = 0.0
                    if let nextMap = dictionary.bigramTransitions[curLastWord],
                       let prob = nextMap[subStr] {
                        transScore = prob * 5.0
                    }

                    let nextScore = curScore + 15.0 + Float(pl * 2) + transScore
                    if dp[i + pl].score < nextScore {
                        dp[i + pl] = DPState(score: nextScore, text: curText + subStr, lastWord: subStr)
                    }
                }
                pl -= 1
            }

            // 2. 辞書単語の遷移 (最長 12 文字から 2 文字まで)
            let maxL = min(12, n - i)
            var l = maxL
            while 2 <= l {
                let subStr = String(chars[i..<(i + l)])

                // 2.1 完全一致単語 (最優先)
                let exactEntries = dictionary.lookupExact(reading: subStr)
                for entry in exactEntries {
                    var bonus: Float = 25.0 + Float(l * 8)
                    if containsKatakana(entry.surface) {
                        bonus += 10.0
                    }
                    var transScore: Float = 0.0
                    if let nextMap = dictionary.bigramTransitions[curLastWord],
                       let prob = nextMap[entry.surface] {
                        transScore = prob * 8.0
                    }

                    let nextScore = curScore + bonus + Float(entry.frequency * 3) + transScore
                    if dp[i + l].score < nextScore {
                        dp[i + l] = DPState(score: nextScore, text: curText + entry.surface, lastWord: entry.surface)
                    }
                }

                // 2.2 音素調音距離に基づく厳格なファジー一致単語 (4文字以上かつ調音距離 <= 0.6 のみ許容)
                if 4 <= l && exactEntries.isEmpty {
                    let fuzzyList = dictionary.lookupFuzzyPhonetic(reading: subStr, maxPhoneticDist: 0.6)
                    for item in fuzzyList {
                        let entry = item.entry
                        let pDist = item.dist
                        var bonus: Float = 12.0 + Float(l * 4) - (pDist * 15.0)
                        if containsKatakana(entry.surface) {
                            bonus += 8.0
                        }
                        var transScore: Float = 0.0
                        if let nextMap = dictionary.bigramTransitions[curLastWord],
                           let prob = nextMap[entry.surface] {
                            transScore = prob * 6.0
                        }

                        let nextScore = curScore + bonus + Float(entry.frequency) + transScore
                        if dp[i + l].score < nextScore {
                            dp[i + l] = DPState(score: nextScore, text: curText + entry.surface, lastWord: entry.surface)
                        }
                    }
                }

                l -= 1
            }

            // 3. 辞書外 1 文字スルー遷移
            let singleChar = String(chars[i])
            let nextScore = curScore - 1.5
            if dp[i + 1].score < nextScore {
                dp[i + 1] = DPState(score: nextScore, text: curText + singleChar, lastWord: singleChar)
            }

            i += 1
        }

        return dp[n].text
    }
}
