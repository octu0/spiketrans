import Foundation

/// かなテキストの正規化、ひらがな発音変換および音素抽出コンバータ
public struct KanjiConverter: Sendable {
    public let vocabulary: PhonemeVocabulary

    public init(vocabulary: PhonemeVocabulary = PhonemeVocabulary()) {
        self.vocabulary = vocabulary
    }

    /// 漢字混じり文をクリーンなひらがな発音文字列に変換 (ひらがな・長音・促音・読点・句点のみ抽出)
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

    /// テキストを形態素に分割し、表層と読みを同時に取得する。
    ///
    /// 読みは形態素解析器が文脈から決めたものを使う。漢字単体から読みを引く
    /// 静的な対応表では「人」が「ひと」「にん」「じん」のどれになるか決められない。
    public func tokenize(_ text: String) -> [Token] {
        if text.isEmpty {
            return []
        }

        let loc = Locale(identifier: "ja_JP") as CFLocale
        let nsText = text as NSString
        let tokenizer = CFStringTokenizerCreate(
            nil,
            text as CFString,
            CFRangeMake(0, nsText.length),
            kCFStringTokenizerUnitWordBoundary,
            loc
        )

        var tokens: [Token] = []
        while CFStringTokenizerAdvanceToNextToken(tokenizer) != [] {
            let range = CFStringTokenizerGetCurrentTokenRange(tokenizer)
            let surface = nsText.substring(with: NSRange(location: range.location, length: range.length))

            // 数字トークンの読みはラテン翻字が ASCII 数字のまま返り
            // かな抽出で脱落するため、位取りのかな読みを直接生成する
            if let numReading = Self.numberReading(surface) {
                tokens.append(Token(surface: surface, reading: numReading))
                continue
            }

            var reading = ""
            switch CFStringTokenizerCopyCurrentTokenAttribute(tokenizer, kCFStringTokenizerAttributeLatinTranscription) {
            case .some(let attr):
                let latin = attr as! NSString
                let ms = NSMutableString(string: latin)
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
            switch true {
            case 0x3041 <= val && val <= 0x3096:
                result.unicodeScalars.append(scalar)
            case val == 0x30FC:
                result.unicodeScalars.append(scalar)
            default:
                break
            }
        }
        return result
    }

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
            switch true {
            case 0x30A1 <= val && val <= 0x30F6:
                // カタカナ -> ひらがな
                switch UnicodeScalar(val - 0x60) {
                case .some(let hScalar):
                    normalized.append(Character(hScalar))
                case .none:
                    normalized.append(Character(scalar))
                }
            default:
                normalized.append(Character(scalar))
            }
        }

        return normalized
    }

    /// テキスト中の全発音から音素トークン ID 列を抽出
    public func toPhonemeTokenIds(_ text: String) -> [Int] {
        let hira = convertToHiragana(text)
        var pureKana = ""
        for c in hira {
            let scalarVal = c.unicodeScalars.first?.value ?? 0
            if 0x3041 <= scalarVal && scalarVal <= 0x3096 || c == "ー" {
                pureKana.append(c)
            }
        }
        return vocabulary.textToTokens(pureKana)
    }

    /// テキスト中の全発音から音素文字列配列を抽出
    public func toPhonemes(_ text: String) -> [String] {
        let hira = convertToHiragana(text)
        var pureKana = ""
        for c in hira {
            let scalarVal = c.unicodeScalars.first?.value ?? 0
            if 0x3041 <= scalarVal && scalarVal <= 0x3096 || c == "ー" {
                pureKana.append(c)
            }
        }
        return vocabulary.kanaToPhonemes(pureKana)
    }
}
