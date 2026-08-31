import Foundation

/// かなテキストの正規化およびひらがな・カタカナ音素抽出コンバータ (ピンイン・辞書読み変換不使用)
public struct KanjiConverter: Sendable {
    public let vocabulary: PhonemeVocabulary

    public init(vocabulary: PhonemeVocabulary = PhonemeVocabulary()) {
        self.vocabulary = vocabulary
    }

    /// カタカナをひらがなに正規化 (漢字はそのまま保持、ピンイン変換は一切行わない)
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

    /// テキスト中のひらがな・カタカナ部分のみから音素トークン ID 列を抽出（漢字部分は音素化せずスキップ）
    public func toPhonemeTokenIds(_ text: String) -> [Int] {
        let normalized = normalizeKana(text)
        // ひらがな・長音のみを抽出
        var pureKana = ""
        for c in normalized {
            let scalarVal = c.unicodeScalars.first?.value ?? 0
            if 0x3041 <= scalarVal && scalarVal <= 0x3096 || c == "ー" {
                pureKana.append(c)
            }
        }
        return vocabulary.textToTokens(pureKana)
    }

    /// テキスト中のひらがな・カタカナ部分のみから音素文字列配列を抽出
    public func toPhonemes(_ text: String) -> [String] {
        let normalized = normalizeKana(text)
        var pureKana = ""
        for c in normalized {
            let scalarVal = c.unicodeScalars.first?.value ?? 0
            if 0x3041 <= scalarVal && scalarVal <= 0x3096 || c == "ー" {
                pureKana.append(c)
            }
        }
        return vocabulary.kanaToPhonemes(pureKana)
    }
}
