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

        // 引き伸ばし記号（〜等）の正規化
        let cleanText = Self.normalizeProlongedMarks(text)

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
                tokens.append(Token(surface: surface, reading: numReading))
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

            tokens.append(Token(surface: surface, reading: kanaOnly(reading)))
        }

        return tokens
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
