import Foundation

/// 漢字かな混じり文を形態素解析し、ひらがな読みへ正規化する。
public struct KanjiConverter: Sendable {
    public let vocabulary: PhonemeVocabulary

    public init(vocabulary: PhonemeVocabulary = PhonemeVocabulary()) {
        self.vocabulary = vocabulary
    }

    /// 形態素単位の表層と読みの組
    public struct Token: Sendable {
        public let surface: String
        public let reading: String

        public init(surface: String, reading: String) {
            self.surface = surface
            self.reading = reading
        }
    }

    /// 数字 1 文字のかな読み
    static let digitKana = ["ぜろ", "いち", "に", "さん", "よん", "ご", "ろく", "なな", "はち", "きゅう"]

    /// 全角数字を 0-9 の配列にする。数字以外が混ざっていたら nil
    static func digitValues(_ text: String) -> [Int]? {
        var values: [Int] = []
        for c in text.unicodeScalars {
            switch c.value {
            case 0x30...0x39:
                values.append(Int(c.value) - 0x30)
            case 0xFF10...0xFF19:
                values.append(Int(c.value) - 0xFF10)
            default:
                return nil
            }
        }
        if values.isEmpty {
            return nil
        }
        return values
    }

    /// 4 桁 (0-9999) の位取り読み。音便 (さんびゃく・ろっぴゃく・はっせん等) を含む
    static func fourDigitReading(_ value: Int) -> String {
        var result = ""
        let thousands = value / 1000
        let hundreds = (value / 100) % 10
        let tens = (value / 10) % 10
        let ones = value % 10
        switch thousands {
        case 0:
            break
        case 1:
            result += "せん"
        case 3:
            result += "さんぜん"
        case 8:
            result += "はっせん"
        default:
            result += digitKana[thousands] + "せん"
        }
        switch hundreds {
        case 0:
            break
        case 1:
            result += "ひゃく"
        case 3:
            result += "さんびゃく"
        case 6:
            result += "ろっぴゃく"
        case 8:
            result += "はっぴゃく"
        default:
            result += digitKana[hundreds] + "ひゃく"
        }
        switch tens {
        case 0:
            break
        case 1:
            result += "じゅう"
        default:
            result += digitKana[tens] + "じゅう"
        }
        if 0 < ones {
            result += digitKana[ones]
        }
        return result
    }

    /// 数字列のかな読み。位取り (万進法) で読み、
    /// 先頭ゼロの列 (電話番号等) と 17 桁以上は 1 桁ずつ読む
    static func numberReading(_ text: String) -> String? {
        guard let values = digitValues(text) else {
            return nil
        }
        let digitWise = { () -> String in
            var r = ""
            for v in values {
                r += digitKana[v]
            }
            return r
        }
        if values[0] == 0 || 16 < values.count {
            if values.count == 1 {
                return "ぜろ"
            }
            return digitWise()
        }
        var number = 0
        for v in values {
            number = number * 10 + v
        }
        let groupUnits = ["", "まん", "おく", "ちょう"]
        var groups: [Int] = []
        var rest = number
        while 0 < rest {
            groups.append(rest % 10000)
            rest /= 10000
        }
        var result = ""
        var gi = groups.count - 1
        while 0 <= gi {
            let g = groups[gi]
            if 0 < g {
                // 1 万・1 億は「いちまん」「いちおく」と読む
                if g == 1 && 0 < gi {
                    result += "いち"
                } else {
                    result += fourDigitReading(g)
                }
                result += groupUnits[gi]
            }
            gi -= 1
        }
        return result
    }

    /// 長音符・波ダッシュ・引き伸ばし記号（半角長音符、ホリゾンタルバー、ダッシュ含む）か判定
    static func isProlongedMark(_ val: UInt32) -> Bool {
        switch val {
        case 0x301C, // WAVE DASH (〜)
             0xFF5E, // FULLWIDTH TILDE (～)
             0x3030, // WAVY DASH (〰)
             0x223C, // TILDE OPERATOR (∼)
             0xFF70, // HALFWIDTH KATAKANA-HIRAGANA PROLONGED SOUND MARK (ｰ)
             0x2015, // HORIZONTAL BAR (―)
             0x2014, // EM DASH (—)
             0x2013, // EN DASH (–)
             0x2500, // BOX DRAWINGS LIGHT HORIZONTAL (─)
             0xFF0D: // FULLWIDTH HYPHEN-MINUS (－)
            return true
        default:
            return false
        }
    }

    /// 波ダッシュ等の引き伸ばし記号を長音符「ー」に正規化
    static func normalizeProlongedMarks(_ text: String) -> String {
        var result = ""
        result.reserveCapacity(text.count)
        for scalar in text.unicodeScalars {
            if isProlongedMark(scalar.value) {
                result.append("ー")
            } else {
                result.unicodeScalars.append(scalar)
            }
        }
        return result
    }

    /// 括弧 (半角・全角) の中身に「笑」を含む注記 (笑・爆笑・冷笑) を取り除く。
    /// 笑い声の種類を示す書き起こしの記号で「わらい」とは言っていない。中身は 6 文字までとし、
    /// 台詞の引用のような長い括弧書きは残す。単独の「w」は除かない (「だぶりゅー」と発音されている)
    static func removeLaughAnnotations(_ text: String) -> String {
        let chars = Array(text)
        var out: [Character] = []
        out.reserveCapacity(chars.count)
        var i = 0
        while i < chars.count {
            let c = chars[i]
            if c == "(" || c == "（" {
                var j = i + 1
                var hasLaugh = false
                while j < chars.count && j - i <= 7 && chars[j] != ")" && chars[j] != "）" {
                    if chars[j] == "笑" {
                        hasLaugh = true
                    }
                    j += 1
                }
                if hasLaugh && j < chars.count && (chars[j] == ")" || chars[j] == "）") {
                    i = j + 1
                    continue
                }
            }
            out.append(c)
            i += 1
        }
        return String(out)
    }

    /// ハイフン・ダッシュまたは引き伸ばし記号のスパンか判定
    static func isHyphenSpan(_ surface: String) -> Bool {
        if surface.count != 1 {
            return false
        }
        guard let val = surface.unicodeScalars.first?.value else {
            return false
        }
        switch val {
        case 0x2D,   // HYPHEN-MINUS (-)
             0xFF0D, // FULLWIDTH HYPHEN-MINUS (－)
             0x2010, // HYPHEN (‐)
             0x2011, // NON-BREAKING HYPHEN (‑)
             0x2012, // FIGURE DASH (‒)
             0x2013, // EN DASH (–)
             0x2014, // EM DASH (—)
             0x2015, // HORIZONTAL BAR (―)
             0x30FC: // PROLONGED SOUND MARK (ー)
            return true
        default:
            return false
        }
    }

    /// 英字・アルファベットトークンか判定 (半角および全角英字、ハイフン区切り英語複合語)
    static func isAlphabetToken(_ text: String) -> Bool {
        if text.isEmpty {
            return false
        }
        var letterCount = 0
        for scalar in text.unicodeScalars {
            let v = scalar.value
            switch v {
            case 0x41...0x5A, 0x61...0x7A, 0xFF21...0xFF3A, 0xFF41...0xFF5A:
                letterCount += 1
            case 0x2D, 0xFF0D, 0x2010...0x2015:
                ()
            default:
                return false
            }
        }
        return 0 < letterCount
    }

    /// 英文・アルファベットトークンの読みを取得
    ///
    /// 組み込み辞書 (SeedVocabulary) を優先し、大文字頭字語は単文字読みに展開する。
    /// 未知の英単語はローマ字規則による異常変換 (「いぷほね」「あっぷれ」等) を防ぐため空文字で保護する。
    static func alphabetReading(_ surface: String) -> String? {
        // 1. シード語彙テーブルから既知語を検索
        if let reading = SeedVocabulary.readingForAlphabetSurface(surface) {
            return reading
        }

        // 2. 単語が純粋な英字・頭字語か検査
        if isAlphabetToken(surface) {
            // 大文字のみの頭字語 (例: "GPU", "API", "SDK", "PC", "CPU", "USB") は各文字を展開
            var isAllUpper = true
            var letterCount = 0
            for scalar in surface.unicodeScalars {
                let v = scalar.value
                switch v {
                case 0x41...0x5A, 0xFF21...0xFF3A:
                    letterCount += 1
                case 0x2D, 0xFF0D, 0x2010...0x2015:
                    ()
                default:
                    isAllUpper = false
                    break
                }
                if isAllUpper != true {
                    break
                }
            }

            if isAllUpper && 0 < letterCount {
                var spelled = ""
                for scalar in surface.unicodeScalars {
                    let letter = String(scalar)
                    if let r = SeedVocabulary.readingForLetter(letter) {
                        spelled.append(r)
                    }
                }
                if spelled.isEmpty != true {
                    return spelled
                }
            }

            // 未知の英単語はローマ字読みへの異常崩れを防ぐため空文字とする
            return ""
        }

        return nil
    }

    /// テキストを形態素に分割し、表層と読みを同時に取得する。
    ///
    /// 読みは形態素解析器が文脈から決めたものを使う。漢字単体から読みを引く
    /// 静的な対応表では「人」が「ひと」「にん」「じん」のどれになるか決められない。
    public func tokenize(_ text: String) -> [Token] {
        if text.isEmpty {
            return []
        }

        // 引き伸ばし記号（〜等）の正規化と、発音されない笑いの注記の除去
        let cleanText = Self.removeLaughAnnotations(Self.normalizeProlongedMarks(text))

        let loc = Locale(identifier: "ja_JP") as CFLocale
        let nsText = cleanText as NSString
        let tokenizer = CFStringTokenizerCreate(
            nil,
            cleanText as CFString,
            CFRangeMake(0, nsText.length),
            kCFStringTokenizerUnitWordBoundary,
            loc
        )

        struct RawTokenSpan {
            let surface: String
            let range: CFRange
            let latinAttr: NSString?
        }

        var rawSpans: [RawTokenSpan] = []
        while CFStringTokenizerAdvanceToNextToken(tokenizer) != [] {
            let range = CFStringTokenizerGetCurrentTokenRange(tokenizer)
            let surface = nsText.substring(with: NSRange(location: range.location, length: range.length))
            let attr = CFStringTokenizerCopyCurrentTokenAttribute(tokenizer, kCFStringTokenizerAttributeLatinTranscription) as? NSString
            rawSpans.append(RawTokenSpan(surface: surface, range: range, latinAttr: attr))
        }

        // ハイフン結合英単語 (例: "Wi" + "-" + "Fi" -> "Wi-Fi") の連続トークンを連結
        var mergedSpans: [RawTokenSpan] = []
        var spanIdx = 0
        while spanIdx < rawSpans.count {
            var currentSpan = rawSpans[spanIdx]
            spanIdx += 1

            // 次がハイフンで、さらにその次がアルファベットトークンであり、かつ空白を挟まず連続している場合は結合
            while spanIdx + 1 < rawSpans.count {
                let hyphenSpan = rawSpans[spanIdx]
                let nextSpan = rawSpans[spanIdx + 1]
                let isContiguousHyphen = Self.isHyphenSpan(hyphenSpan.surface) &&
                    currentSpan.range.location + currentSpan.range.length == hyphenSpan.range.location &&
                    hyphenSpan.range.location + hyphenSpan.range.length == nextSpan.range.location
                if isContiguousHyphen && Self.isAlphabetToken(currentSpan.surface) && Self.isAlphabetToken(nextSpan.surface) {
                    let combinedSurface = "\(currentSpan.surface)-\(nextSpan.surface)"
                    let combinedRange = CFRangeMake(
                        currentSpan.range.location,
                        currentSpan.range.length + hyphenSpan.range.length + nextSpan.range.length
                    )
                    currentSpan = RawTokenSpan(surface: combinedSurface, range: combinedRange, latinAttr: nil)
                    spanIdx += 2
                } else {
                    break
                }
            }
            mergedSpans.append(currentSpan)
        }

        var tokens: [Token] = []
        var mIdx = 0
        while mIdx < mergedSpans.count {
            let span = mergedSpans[mIdx]
            mIdx += 1
            let surface = span.surface

            // 数字トークンの読みは位取りのかな読みを直接生成する
            if let numReading = Self.numberReading(surface) {
                tokens.append(Token(surface: surface, reading: Self.pronunciation(surface: surface, reading: numReading)))
                continue
            }

            // 英文・アルファベットトークン（Wi-Fi, iPhone等）の保護およびフォールバック処理
            if let alphaReading = Self.alphabetReading(surface) {
                tokens.append(Token(surface: surface, reading: kanaOnly(alphaReading)))
                continue
            }

            var reading = ""
            switch span.latinAttr {
            case .some(let latin):
                // NFD 分解で合成マクロン (\u{0304}) に統一し、CFStringTransform の前に長音符「ー」に置換
                let decomposed = (latin as String).decomposedStringWithCanonicalMapping
                let withProlongedMark = decomposed.replacingOccurrences(of: "\u{0304}", with: "ー")
                let ms = NSMutableString(string: withProlongedMark)
                CFStringTransform(ms as CFMutableString, nil, kCFStringTransformLatinHiragana, false)
                reading = normalizeKana(ms as String)
            case .none:
                reading = normalizeKana(surface)
            }

            tokens.append(Token(surface: surface, reading: kanaOnly(Self.pronunciation(surface: surface, reading: reading))))
        }

        return Self.applyReadingOverrides(tokens)
    }

    /// 形態素解析器の読みを話し言葉の読みへ置き換える表層 (形態素 1 つ)。値は表記の読みで `pronunciation` を通す
    static let readingOverrides: [String: String] = [
        "私": "わたし",
        "日本": "にほん",
        "明日": "あした",
        "皆": "みんな",
        "言う": "ゆー",
        "いう": "ゆー",
    ]

    /// 話し言葉で複数の読みがある表層の、`readingOverrides` 以外の読み。
    /// 教師かなは 1 つに決めるしかないので (話者が「あたし」と言ったかは文脈では分からない)、
    /// 第2段の辞書にだけ登録し、音響モデルがどの読みを出しても同じ表層に戻せるようにする
    public static let readingVariants: [String: [String]] = [
        "私": ["あたし", "わたくし"],
        "皆": ["みな"],
    ]

    /// 連続する形態素の表層を連結して照合し、1 つの形態素に併合する表。
    /// 形態素解析器が「お/母/さん」「木曜/日」のように切ると各片の読みが単独語の読みになるため
    static let phraseOverrides: [String: String] = [
        "お母さん": "おかあさん",
        "お父さん": "おとうさん",
        "お兄さん": "おにいさん",
        "お姉さん": "おねえさん",
        "皆さん": "みなさん",
        "皆様": "みなさま",
        "一昨日": "おととい",
        "一昨年": "おととし",
        "二日": "ふつか",
        "四人": "よにん",
        "世界中": "せかいじゅう",
        "日本中": "にほんじゅう",
        "一日中": "いちにちじゅう",
    ]

    static let maxPhraseTokens = 3

    /// 直前の数によらず同じ読みが返る助数詞の基本読み (「本/ぽん」等を揃える)
    static let counterBaseReadings: [String: String] = [
        "本": "ほん", "杯": "はい", "匹": "ひき", "分": "ふん", "歩": "ほ",
        "発": "はつ", "泊": "はく", "品": "ひん", "敗": "はい", "拍": "はく",
    ]

    /// ん で終わる数 (さん・せん・まん・なん) の後で濁音になる助数詞。表内の他の助数詞は半濁音 (さんぷん)
    static let voicedAfterNasal: Set<String> = ["本", "杯", "匹"]

    /// 数 + 助数詞で数ごとに読みが変わるもの (日付・月・時・人・つ)。値は表記の読みで `pronunciation` を通す。
    /// 日付は表にない日も「N にち」(形態素解析器は 11 日を「じゅういち か」と読む)
    static let numeralCounterReadings: [String: [Int: String]] = [
        "日": [1: "いちにち", 2: "ふつか", 3: "みっか", 4: "よっか", 5: "いつか", 6: "むいか", 7: "なのか", 8: "ようか",
               9: "ここのか", 10: "とおか", 14: "じゅうよっか", 20: "はつか", 24: "にじゅうよっか"],
        "月": [4: "しがつ", 7: "しちがつ", 9: "くがつ"],
        "時": [4: "よじ", 7: "しちじ", 9: "くじ"],
        "人": [1: "ひとり", 2: "ふたり", 4: "よにん"],
        "つ": [1: "ひとつ", 2: "ふたつ", 3: "みっつ", 4: "よっつ", 5: "いつつ", 6: "むっつ", 7: "ななつ", 8: "やっつ",
               9: "ここのつ", 10: "とお"],
    ]

    static let kanjiNumeralValues: [String: Int] = [
        "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
        "十四": 14, "二十": 20, "二十四": 24,
    ]

    /// 数詞の表層の値。算用数字と 24 までの漢数字
    static func numeralValue(_ surface: String) -> Int? {
        if let values = digitValues(surface) {
            if 4 < values.count {
                return nil
            }
            var v = 0
            for d in values {
                v = v * 10 + d
            }
            return v
        }
        return kanjiNumeralValues[surface]
    }

    /// 数と助数詞の組の読み。「1人/れん」のように解析器が併合したトークンも表層を分けて扱う
    static func numeralCounterReading(numeral: String, numeralReading: String, counter: String) -> String? {
        guard let table = numeralCounterReadings[counter], let value = numeralValue(numeral) else {
            return nil
        }
        if let reading = table[value] {
            return reading
        }
        if counter == "日" {
            return numeralReading + "にち"
        }
        return nil
    }

    /// 末尾 1 文字が助数詞表の見出しで、その前が数詞の表層 (「1人」「10日」) なら分ける
    static func splitNumeralCounter(_ surface: String) -> (String, String)? {
        guard 1 < surface.count, let last = surface.last, numeralCounterReadings[String(last)] != nil else {
            return nil
        }
        let numeral = String(surface.dropLast())
        if numeralValue(numeral) == nil {
            return nil
        }
        return (numeral, String(last))
    }

    /// 数詞の表層か (算用数字・漢数字・何・数・幾)
    static func isNumeralSurface(_ surface: String) -> Bool {
        if surface.isEmpty {
            return false
        }
        for scalar in surface.unicodeScalars {
            switch scalar.value {
            case 0x30...0x39, 0xFF10...0xFF19:
                continue
            default:
                break
            }
            if "〇一二三四五六七八九十百千万億兆何数幾".unicodeScalars.contains(scalar) != true {
                return false
            }
        }
        return true
    }

    static func isKanji(_ scalar: Unicode.Scalar) -> Bool {
        switch scalar.value {
        case 0x4E00...0x9FFF, 0x3400...0x4DBF:
            return true
        default:
            return false
        }
    }

    static func isKatakana(_ scalar: Unicode.Scalar) -> Bool {
        switch scalar.value {
        case 0x30A1...0x30FA:
            return true
        default:
            return false
        }
    }

    /// 表層の先頭が漢字・カタカナか
    static func startsWithKanjiOrKatakana(_ surface: String) -> Bool {
        guard let first = surface.unicodeScalars.first else {
            return false
        }
        return isKanji(first) || isKatakana(first)
    }

    /// か・さ・た・は・ぱ行で始まる読みか (数詞の促音化が起きる子音)
    static func startsWithVoicelessObstruent(_ reading: String) -> Bool {
        guard let first = reading.first else {
            return false
        }
        return "かきくけこさしすせそたちつてとはひふへほぱぴぷぺぽ".contains(first)
    }

    /// は行 → ぱ行 / ば行 の置き換え
    static func replacingHRow(_ reading: String, with row: String) -> String {
        guard let first = reading.first, let idx = "はひふへほ".firstIndex(of: first) else {
            return reading
        }
        let offset = "はひふへほ".distance(from: "はひふへほ".startIndex, to: idx)
        let rowChars = Array(row)
        return String(rowChars[offset]) + String(reading.dropFirst())
    }

    /// 数詞 + 助数詞の連声。数詞の末尾の促音化 (いち → いっ) と助数詞頭の半濁音・濁音化 (ほん → ぽん・ぼん)
    static func counterSandhi(numeral: Token, counter: Token) -> (String, String)? {
        if isNumeralSurface(numeral.surface) != true || startsWithKanjiOrKatakana(counter.surface) != true {
            if counter.surface.hasPrefix("か月") != true {
                return nil
            }
        }
        var base = counter.reading
        if let known = counterBaseReadings[counter.surface] {
            base = known
        }
        // 外来語の単位は促音化しない (いちへくたーる)
        if let first = counter.surface.unicodeScalars.first, isKatakana(first), let head = base.first, "はひふへほ".contains(head) {
            return nil
        }
        let n = numeral.reading
        let geminating: [(String, String, Bool)] = [
            ("いち", "いっ", true), ("はち", "はっ", true), ("じゅー", "じゅっ", true),
            ("ろく", "ろっ", false), ("ひゃく", "ひゃっ", false),
        ]
        for (tail, geminated, allRows) in geminating {
            if n.hasSuffix(tail) != true || startsWithVoicelessObstruent(base) != true {
                continue
            }
            let isKOrH = "かきくけこはひふへほ".contains(base.first!)
            if allRows != true && isKOrH != true {
                return nil
            }
            return (String(n.dropLast(tail.count)) + geminated, replacingHRow(base, with: "ぱぴぷぺぽ"))
        }
        if counterBaseReadings[counter.surface] == nil {
            return nil
        }
        for tail in ["さん", "せん", "まん", "なん"] {
            if n.hasSuffix(tail) != true {
                continue
            }
            if voicedAfterNasal.contains(counter.surface) {
                return (n, replacingHRow(base, with: "ばびぶべぼ"))
            }
            return (n, replacingHRow(base, with: "ぱぴぷぺぽ"))
        }
        if n.hasSuffix("よん") && voicedAfterNasal.contains(counter.surface) != true {
            return (n, replacingHRow(base, with: "ぱぴぷぺぽ"))
        }
        return (n, base)
    }

    /// 形態素解析器の読みのうち、話し言葉として一定でないものを上書きする。
    ///
    /// 解析器は「私/わたくし」「皆/みな」「日本/にっぽん」を返し、複合語を「お/母/さん」
    /// 「木曜/日」に切って片ごとの読みを付け、助数詞の連声 (いっぽん・さんぼん) を扱わない。
    /// 音響モデルの教師は実際に発音された音でなければならないので、ここで揃える
    static func applyReadingOverrides(_ tokens: [Token]) -> [Token] {
        var result: [Token] = []
        var i = 0
        while i < tokens.count {
            // 1. 連結表層の照合 (長い並びから)
            var span = min(maxPhraseTokens, tokens.count - i)
            var merged = false
            while 1 < span {
                var surface = ""
                var k = 0
                while k < span {
                    surface += tokens[i + k].surface
                    k += 1
                }
                if let reading = phraseOverrides[surface] {
                    result.append(Token(surface: surface, reading: pronunciation(surface: surface, reading: reading)))
                    i += span
                    merged = true
                    break
                }
                span -= 1
            }
            if merged {
                continue
            }
            var token = tokens[i]
            // 2. 単一形態素の置き換え
            if let reading = readingOverrides[token.surface] {
                token = Token(surface: token.surface, reading: pronunciation(surface: token.surface, reading: reading))
            }
            // 3. 曜日: 「木曜/もくよう」+「日/ひ」→ もくようび
            if i + 1 < tokens.count && token.surface.hasSuffix("曜") && tokens[i + 1].surface == "日" {
                result.append(Token(surface: token.surface + "日", reading: token.reading + "び"))
                i += 2
                continue
            }
            // 4. 数ごとに読みが変わる助数詞 (ついたち・しがつ・よじ・ひとり・みっつ)
            if i + 1 < tokens.count, isNumeralSurface(token.surface),
               let reading = numeralCounterReading(numeral: token.surface, numeralReading: token.reading, counter: tokens[i + 1].surface) {
                let surface = token.surface + tokens[i + 1].surface
                result.append(Token(surface: surface, reading: pronunciation(surface: surface, reading: reading)))
                i += 2
                continue
            }
            if let (numeral, counter) = splitNumeralCounter(token.surface),
               let reading = numeralCounterReading(numeral: numeral, numeralReading: pronunciation(surface: numeral, reading: numberReading(numeral) ?? ""), counter: counter) {
                result.append(Token(surface: token.surface, reading: pronunciation(surface: token.surface, reading: reading)))
                i += 1
                continue
            }
            // 5. 数詞 + 助数詞の連声
            if i + 1 < tokens.count, let (numeral, counter) = counterSandhi(numeral: token, counter: tokens[i + 1]) {
                result.append(Token(surface: token.surface, reading: numeral))
                result.append(Token(surface: tokens[i + 1].surface, reading: counter))
                i += 2
                continue
            }
            // 6. 国名・集団名 + 人 → じん (数詞の後の「三人/さんにん」は除く)
            if token.surface == "人" && token.reading == "にん", let prev = result.last,
               startsWithKanjiOrKatakana(prev.surface) && isNumeralSurface(prev.surface) != true {
                token = Token(surface: token.surface, reading: "じん")
            }
            // 7. 何: 助詞・動詞が続くときは「なに」(何が・何を・何して)、助数詞や「何で」は「なん」のまま
            if token.surface == "何" && i + 1 < tokens.count, let next = tokens[i + 1].surface.first {
                if "がをもよしや".contains(next) {
                    token = Token(surface: token.surface, reading: "なに")
                }
            }
            result.append(token)
            i += 1
        }
        return result
    }

    /// かなの母音 (あ/い/う/え/お)。ん・っ・ー・記号は nil
    static func vowel(of kana: Character) -> Character? {
        switch kana {
        case "あ", "か", "が", "さ", "ざ", "た", "だ", "な", "は", "ば", "ぱ", "ま", "や", "ら", "わ", "ぁ", "ゃ", "ゎ":
            return "あ"
        case "い", "き", "ぎ", "し", "じ", "ち", "ぢ", "に", "ひ", "び", "ぴ", "み", "り", "ゐ", "ぃ":
            return "い"
        case "う", "く", "ぐ", "す", "ず", "つ", "づ", "ぬ", "ふ", "ぶ", "ぷ", "む", "ゆ", "る", "ぅ", "ゅ", "ゔ":
            return "う"
        case "え", "け", "げ", "せ", "ぜ", "て", "で", "ね", "へ", "べ", "ぺ", "め", "れ", "ゑ", "ぇ":
            return "え"
        case "お", "こ", "ご", "そ", "ぞ", "と", "ど", "の", "ほ", "ぼ", "ぽ", "も", "よ", "ろ", "を", "ぉ", "ょ":
            return "お"
        default:
            return nil
        }
    }

    /// 表記の読みを発音のかなに揃える。
    ///
    /// 音響モデルの教師は音に対応していなければならない。表記のままだと同じ音 [oː] が
    /// 「おう」「おお」「ー」と 3 通りに書かれ、助詞の「は」「を」は「わ」「お」と読む。
    /// 表記から音を当てる負担を SNN に負わせないよう、ここで 1 通りに寄せる。
    /// 第2段のかな漢字辞書も同じ読みで作るので、経路全体で一貫する。
    ///   - 助詞の「は」「へ」(単独トークン) → わ・え、「を」→ お
    ///   - ぢ・づ → じ・ず
    ///   - 同じ母音の連なり (おう・おお・えい・ええ・ああ・いい・うう) の 2 文字目 → ー。
    ///     ただし終止形が「う」で終わる動詞 (思う・追う・食う) の語末は残す。
    ///     意向形 (行こう) やかな表記の「そう・もう・ありがとう」は長音にする
    /// 語末の「う」を長音にしない動詞 (終止形が お段・う段 + う)。
    /// 意向形 (行こう・見よう) やかな表記の「そう・もう・ありがとう」は長音なので、表にあるものだけ残す
    static let uEndingVerbs: Set<String> = [
        "思う", "追う", "問う", "酔う", "沿う", "食う", "吸う", "縫う", "乞う", "負う", "覆う", "請う",
        "装う", "添う", "集う", "揃う", "争う", "救う", "狂う", "通う", "見舞う", "住まう", "買う", "会う",
    ]

    static func pronunciation(surface: String, reading: String) -> String {
        switch surface {
        case "は":
            return "わ"
        case "へ":
            return "え"
        default:
            break
        }
        let chars = Array(reading)
        var result: [Character] = []
        result.reserveCapacity(chars.count)
        let keepsFinalU = Self.uEndingVerbs.contains(surface)
        var i = 0
        while i < chars.count {
            var c = chars[i]
            switch c {
            case "を":
                c = "お"
            case "ぢ":
                c = "じ"
            case "づ":
                c = "ず"
            default:
                break
            }
            if 0 < i, let prevVowel = Self.vowel(of: result[result.count - 1]) {
                let isLast = (i == chars.count - 1)
                var lengthens = false
                switch c {
                case "う":
                    lengthens = (prevVowel == "お" || prevVowel == "う") && (isLast && keepsFinalU) != true
                case "い":
                    lengthens = (prevVowel == "え" || prevVowel == "い")
                case "え":
                    lengthens = (prevVowel == "え")
                case "お":
                    lengthens = (prevVowel == "お")
                case "あ":
                    lengthens = (prevVowel == "あ")
                default:
                    break
                }
                if lengthens {
                    c = "ー"
                }
            }
            result.append(c)
            i += 1
        }
        return String(result)
    }

    /// ひらがなと長音のみを残す。
    ///
    /// 句読点は発音として音声に存在しないため、音響 SNN の教師には含めない。
    /// 句読点は第2段で語の連接統計から復元する。
    public func kanaOnly(_ text: String) -> String {
        // 合成済みの形に揃えてから 1 スカラーずつ選ぶ。
        // 書記素クラスタ単位で判定すると「を + 濁点」のように、かなに結合文字が
        // 付いたものが 1 文字として通ってしまい、語彙が際限なく増える
        var result = ""
        for scalar in text.precomposedStringWithCanonicalMapping.unicodeScalars {
            let val = scalar.value
            switch val {
            case 0x3041...0x3096, 0x30FC:
                result.unicodeScalars.append(scalar)
            case _ where Self.isProlongedMark(val):
                result.append("ー")
            default:
                break
            }
        }
        return result
    }

    /// 形態素の読みをひらがなに正規化する。句読点は `kanaOnly` で落とす。
    public func convertToHiragana(_ text: String) -> String {
        if text.isEmpty {
            return ""
        }

        var rawHira = ""
        for token in tokenize(text) {
            rawHira.append(token.reading)
        }

        return kanaOnly(rawHira)
    }

    /// カタカナをひらがなに正規化
    public func normalizeKana(_ text: String) -> String {
        if text.isEmpty {
            return ""
        }

        var normalized = ""
        for scalar in text.unicodeScalars {
            let val = scalar.value
            switch val {
            case 0x30A1...0x30F6:
                // カタカナ -> ひらがな
                switch UnicodeScalar(val - 0x60) {
                case .some(let hScalar):
                    normalized.append(Character(hScalar))
                case .none:
                    normalized.append(Character(scalar))
                }
            case _ where Self.isProlongedMark(val):
                normalized.append("ー")
            default:
                normalized.append(Character(scalar))
            }
        }

        return normalized
    }

    /// テキスト中の全発音から音素トークン ID 列を抽出
    public func toPhonemeTokenIds(_ text: String) -> [Int] {
        let hira = convertToHiragana(text)
        return vocabulary.textToTokens(hira)
    }

    /// テキスト中の全発音から音素文字列配列を抽出
    public func toPhonemes(_ text: String) -> [String] {
        let hira = convertToHiragana(text)
        return vocabulary.kanaToPhonemes(hira)
    }
}
