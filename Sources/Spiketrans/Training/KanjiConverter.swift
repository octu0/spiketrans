import Foundation

/// かなテキストの正規化、ひらがな発音変換および音素抽出コンバータ
public struct KanjiConverter: Sendable {
    public let vocabulary: PhonemeVocabulary

    public init(vocabulary: PhonemeVocabulary = PhonemeVocabulary()) {
        self.vocabulary = vocabulary
    }

    /// 漢字混じり文をクリーンなひらがな発音文字列に変換 (ひらがな・長音・促音・読点・句点のみ抽出)
    public func convertToHiragana(_ text: String) -> String {
        if text.isEmpty {
            return ""
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

        var rawHira = ""
        while CFStringTokenizerAdvanceToNextToken(tokenizer) != [] {
            let range = CFStringTokenizerGetCurrentTokenRange(tokenizer)
            let sub = nsText.substring(with: NSRange(location: range.location, length: range.length))
            if let attr = CFStringTokenizerCopyCurrentTokenAttribute(tokenizer, kCFStringTokenizerAttributeLatinTranscription) {
                let latin = attr as! NSString
                let ms = NSMutableString(string: latin)
                CFStringTransform(ms as CFMutableString, nil, kCFStringTransformLatinHiragana, false)
                let hira = normalizeKana(ms as String)
                rawHira.append(hira)
            } else {
                let hira = normalizeKana(sub)
                rawHira.append(hira)
            }
        }

        // ひらがな・長音・促音・読点・句点のみを抽出 (数字やラテン記号の完全除去)
        var cleanResult = ""
        for c in rawHira {
            let val = c.unicodeScalars.first?.value ?? 0
            switch true {
            case 0x3041 <= val && val <= 0x3096:
                cleanResult.append(c) // ひらがな
            case c == "ー", c == "、", c == "。":
                cleanResult.append(c)
            default:
                break
            }
        }

        return cleanResult
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
