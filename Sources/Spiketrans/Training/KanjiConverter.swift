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
        var result = ""
        for c in text {
            let val = c.unicodeScalars.first?.value ?? 0
            switch true {
            case 0x3041 <= val && val <= 0x3096:
                result.append(c)
            case c == "ー":
                result.append(c)
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
