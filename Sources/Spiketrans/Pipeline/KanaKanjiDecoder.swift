import Foundation

/// かな・漢字辞書エントリ
public struct KanaKanjiEntry: Sendable {
    public let reading: String      // かな読み (例: "とつぜん")
    public let surface: String      // 漢字表記 (例: "突然")
    public let frequency: Int       // 出現頻度

    public init(reading: String, surface: String, frequency: Int = 1) {
        self.reading = reading
        self.surface = surface
        self.frequency = frequency
    }
}

/// コーパスから動的構築されるかな漢字変換辞書
public final class KanaKanjiDictionary: @unchecked Sendable {
    public private(set) var entriesByReading: [String: [KanaKanjiEntry]] = [:]
    public private(set) var maxReadingLength: Int = 0

    public init() {}

    /// 漢字混じり文コーパスから形態素・単語エントリを動的学習・登録
    public func buildFromCorpus(rawTexts: [String]) {
        let converter = KanjiConverter()
        let loc = Locale(identifier: "ja_JP") as CFLocale

        for text in rawTexts {
            if text.isEmpty {
                continue
            }

            let nsText = text as NSString
            let tokenizer = CFStringTokenizerCreate(
                nil,
                text as CFString,
                CFRangeMake(0, nsText.length),
                kCFStringTokenizerUnitWordBoundary,
                loc
            )

            while CFStringTokenizerAdvanceToNextToken(tokenizer) != [] {
                let range = CFStringTokenizerGetCurrentTokenRange(tokenizer)
                let surface = nsText.substring(with: NSRange(location: range.location, length: range.length))
                let reading = converter.convertToHiragana(surface)

                if reading.isEmpty != true && surface.isEmpty != true {
                    addEntry(reading: reading, surface: surface)
                }
            }
        }
    }

    /// 単語エントリの追加
    public func addEntry(reading: String, surface: String) {
        if maxReadingLength < reading.count {
            maxReadingLength = reading.count
        }

        var list = entriesByReading[reading] ?? []
        var found = false
        var i = 0
        while i < list.count {
            if list[i].surface == surface {
                list[i] = KanaKanjiEntry(
                    reading: reading,
                    surface: surface,
                    frequency: list[i].frequency + 1
                )
                found = true
                break
            }
            i += 1
        }

        if found != true {
            list.append(KanaKanjiEntry(reading: reading, surface: surface, frequency: 1))
        }

        entriesByReading[reading] = list
    }

    /// 読み完全一致の候補を取得
    public func lookupExact(reading: String) -> [KanaKanjiEntry] {
        return entriesByReading[reading] ?? []
    }

    /// 音素揺らぎを許容するファジー検索 (編集距離 <= 1)
    public func lookupFuzzy(reading: String, maxDistance: Int = 1) -> [KanaKanjiEntry] {
        var results: [KanaKanjiEntry] = []
        let rLen = reading.count

        for (candReading, candEntries) in entriesByReading {
            // 文字数差が maxDistance を超える場合はスキップ
            let diff = abs(candReading.count - rLen)
            if maxDistance < diff {
                continue
            }

            let dist = levenshteinDistance(reading, candReading)
            if dist <= maxDistance {
                results.append(contentsOf: candEntries)
            }
        }

        return results
    }

    private func levenshteinDistance(_ s1: String, _ s2: String) -> Int {
        let a1 = Array(s1)
        let a2 = Array(s2)
        let m = a1.count
        let n = a2.count
        if m == 0 { return n }
        if n == 0 { return m }

        var dp = [[Int]](repeating: [Int](repeating: 0, count: n + 1), count: m + 1)
        var i = 0
        while i <= m {
            dp[i][0] = i
            i += 1
        }
        var j = 0
        while j <= n {
            dp[0][j] = j
            j += 1
        }

        i = 1
        while i <= m {
            j = 1
            while j <= n {
                var cost = 1
                if a1[i - 1] == a2[j - 1] {
                    cost = 0
                }
                let d1 = dp[i - 1][j] + 1
                let d2 = dp[i][j - 1] + 1
                let d3 = dp[i - 1][j - 1] + cost
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
}

/// 第2段 堅牢なかな漢字ビーム探索統合デコーダ
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
        "こと", "である", "にある", "した", "して", "され", "れる", "られる", "ない", "ます", "ません", "けれど", "けれど"
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
        }

        var dp = [DPState](repeating: DPState(score: -.infinity, text: ""), count: n + 1)
        dp[0] = DPState(score: 0.0, text: "")

        var i = 0
        while i < n {
            if dp[i].score.isInfinite {
                i += 1
                continue
            }

            let curScore = dp[i].score
            let curText = dp[i].text

            // 1. 助詞・文末表現の遷移 (1〜4文字)
            var pl = min(4, n - i)
            while 1 <= pl {
                let subStr = String(chars[i..<(i + pl)])
                if Self.protectedParticles.contains(subStr) {
                    let nextScore = curScore + 12.0 + Float(pl * 2)
                    if dp[i + pl].score < nextScore {
                        dp[i + pl] = DPState(score: nextScore, text: curText + subStr)
                    }
                }
                pl -= 1
            }

            // 2. 辞書単語の遷移 (最長 10 文字から 2 文字まで)
            let maxL = min(10, n - i)
            var l = maxL
            while 2 <= l {
                let subStr = String(chars[i..<(i + l)])

                // 2.1 完全一致単語
                let exactEntries = dictionary.lookupExact(reading: subStr)
                for entry in exactEntries {
                    var bonus: Float = 15.0 + Float(l * 5)
                    if containsKatakana(entry.surface) {
                        bonus += 10.0
                    }
                    let nextScore = curScore + bonus + Float(entry.frequency)
                    if dp[i + l].score < nextScore {
                        dp[i + l] = DPState(score: nextScore, text: curText + entry.surface)
                    }
                }

                // 2.2 ファジー一致単語 (3文字以上で編集距離 1 を許容)
                if 3 <= l {
                    let fuzzyEntries = dictionary.lookupFuzzy(reading: subStr, maxDistance: 1)
                    for entry in fuzzyEntries {
                        var bonus: Float = 8.0 + Float(l * 3)
                        if containsKatakana(entry.surface) {
                            bonus += 8.0
                        }
                        let nextScore = curScore + bonus + Float(entry.frequency)
                        if dp[i + l].score < nextScore {
                            dp[i + l] = DPState(score: nextScore, text: curText + entry.surface)
                        }
                    }
                }

                l -= 1
            }

            // 3. 辞書外 1 文字スルー遷移
            let singleChar = String(chars[i])
            let nextScore = curScore - 1.0
            if dp[i + 1].score < nextScore {
                dp[i + 1] = DPState(score: nextScore, text: curText + singleChar)
            }

            i += 1
        }

        return dp[n].text
    }
}
