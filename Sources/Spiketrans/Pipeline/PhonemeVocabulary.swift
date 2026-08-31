import Foundation

/// 音素トークン定義
public struct PhonemeToken: Sendable, Hashable, Equatable {
    public let id: Int
    public let name: String
    public let isSpecial: Bool

    public init(id: Int, name: String, isSpecial: Bool) {
        self.id = id
        self.name = name
        self.isSpecial = isSpecial
    }
}

/// 日本語音素・かな語彙テーブルおよび双方向変換
public struct PhonemeVocabulary: Sendable {
    public static let padId = 0
    public static let silId = 1
    public static let unkId = 2
    public static let sosId = 3
    public static let eosId = 4

    public let size: Int
    private let tokenList: [String]
    private let tokenToIdTable: [String: Int]

    public init() {
        var list: [String] = []
        // 特殊トークン (0..4)
        list.append("<pad>") // 0
        list.append("<sil>") // 1
        list.append("<unk>") // 2
        list.append("<sos>") // 3
        list.append("<eos>") // 4

        // 日本語母音 (5..9)
        list.append("a") // 5
        list.append("i") // 6
        list.append("u") // 7
        list.append("e") // 8
        list.append("o") // 9

        // 日本語子音・音素 (10..25)
        list.append("k") // 10
        list.append("s") // 11
        list.append("t") // 12
        list.append("n") // 13
        list.append("h") // 14
        list.append("m") // 15
        list.append("y") // 16
        list.append("r") // 17
        list.append("w") // 18
        list.append("g") // 19
        list.append("z") // 20
        list.append("d") // 21
        list.append("b") // 22
        list.append("p") // 23
        list.append("N") // 24 (ん)
        list.append("Q") // 25 (っ)

        // 拡張・複合音素 (26..38)
        list.append("_")  // 26 (ー)
        list.append("sh") // 27
        list.append("ch") // 28
        list.append("ts") // 29
        list.append("ky") // 30
        list.append("ny") // 31
        list.append("hy") // 32
        list.append("my") // 33
        list.append("ry") // 34
        list.append("gy") // 35
        list.append("j")  // 36
        list.append("by") // 37
        list.append("py") // 38

        // 予約トークン (39..63) 計64語彙
        var rIdx = 39
        while rIdx < 64 {
            list.append("<res\(rIdx)>")
            rIdx += 1
        }

        self.size = list.count
        self.tokenList = list

        var table: [String: Int] = [:]
        var i = 0
        while i < list.count {
            table[list[i]] = i
            i += 1
        }
        self.tokenToIdTable = table
    }

    /// トークン文字列から ID を取得
    public func id(for token: String) -> Int {
        switch tokenToIdTable[token] {
        case .some(let val):
            return val
        case .none:
            return Self.unkId
        }
    }

    /// ID からトークン文字列を取得
    public func token(for id: Int) -> String {
        switch true {
        case id < 0:
            return "<unk>"
        case tokenList.count <= id:
            return "<unk>"
        default:
            return tokenList[id]
        }
    }

    /// 無音または Blank トークンであるか判定
    public func isSilence(_ id: Int) -> Bool {
        switch id {
        case Self.silId, Self.padId:
            return true
        default:
            return false
        }
    }

    /// カタカナをひらがなに正規化
    private func normalizeKatakanaToHiragana(_ text: String) -> String {
        var result = ""
        for scalar in text.unicodeScalars {
            let val = scalar.value
            switch true {
            case 0x30A1 <= val && val <= 0x30F6:
                if let hiraganaScalar = UnicodeScalar(val - 0x60) {
                    result.append(Character(hiraganaScalar))
                } else {
                    result.append(Character(scalar))
                }
            default:
                result.append(Character(scalar))
            }
        }
        return result
    }

    /// 日本語ひらがな/カタカナ文字列を音素列へ分解
    public func kanaToPhonemes(_ text: String) -> [String] {
        let normalized = normalizeKatakanaToHiragana(text)
        let chars = Array(normalized)
        var phonemes: [String] = []
        var i = 0
        let count = chars.count

        while i < count {
            let c = chars[i]

            // 2文字の拗音・複合文字判定
            if (i + 1) < count {
                let nextC = chars[i + 1]
                let pair = String([c, nextC])
                var matched = true
                switch pair {
                case "きゃ": phonemes.append(contentsOf: ["ky", "a"])
                case "きゅ": phonemes.append(contentsOf: ["ky", "u"])
                case "きょ": phonemes.append(contentsOf: ["ky", "o"])
                case "しゃ": phonemes.append(contentsOf: ["sh", "a"])
                case "しゅ": phonemes.append(contentsOf: ["sh", "u"])
                case "しょ": phonemes.append(contentsOf: ["sh", "o"])
                case "しぇ": phonemes.append(contentsOf: ["sh", "e"])
                case "ちゃ": phonemes.append(contentsOf: ["ch", "a"])
                case "ちゅ": phonemes.append(contentsOf: ["ch", "u"])
                case "ちょ": phonemes.append(contentsOf: ["ch", "o"])
                case "ちぇ": phonemes.append(contentsOf: ["ch", "e"])
                case "にゃ": phonemes.append(contentsOf: ["ny", "a"])
                case "にゅ": phonemes.append(contentsOf: ["ny", "u"])
                case "にょ": phonemes.append(contentsOf: ["ny", "o"])
                case "ひゃ": phonemes.append(contentsOf: ["hy", "a"])
                case "ひゅ": phonemes.append(contentsOf: ["hy", "u"])
                case "ひょ": phonemes.append(contentsOf: ["hy", "o"])
                case "みゃ": phonemes.append(contentsOf: ["my", "a"])
                case "みゅ": phonemes.append(contentsOf: ["my", "u"])
                case "みょ": phonemes.append(contentsOf: ["my", "o"])
                case "りゃ": phonemes.append(contentsOf: ["ry", "a"])
                case "りゅ": phonemes.append(contentsOf: ["ry", "u"])
                case "りょ": phonemes.append(contentsOf: ["ry", "o"])
                case "ぎゃ": phonemes.append(contentsOf: ["gy", "a"])
                case "ぎゅ": phonemes.append(contentsOf: ["gy", "u"])
                case "ぎょ": phonemes.append(contentsOf: ["gy", "o"])
                case "じゃ": phonemes.append(contentsOf: ["j", "a"])
                case "じゅ": phonemes.append(contentsOf: ["j", "u"])
                case "じょ": phonemes.append(contentsOf: ["j", "o"])
                case "じぇ": phonemes.append(contentsOf: ["j", "e"])
                case "びゃ": phonemes.append(contentsOf: ["by", "a"])
                case "びゅ": phonemes.append(contentsOf: ["by", "u"])
                case "びょ": phonemes.append(contentsOf: ["by", "o"])
                case "ぴゃ": phonemes.append(contentsOf: ["py", "a"])
                case "ぴゅ": phonemes.append(contentsOf: ["py", "u"])
                case "ぴょ": phonemes.append(contentsOf: ["py", "o"])
                case "つぁ": phonemes.append(contentsOf: ["ts", "a"])
                case "つぃ": phonemes.append(contentsOf: ["ts", "i"])
                case "つぇ": phonemes.append(contentsOf: ["ts", "e"])
                case "つぉ": phonemes.append(contentsOf: ["ts", "o"])
                case "ふぁ": phonemes.append(contentsOf: ["h", "a"])
                case "ふぃ": phonemes.append(contentsOf: ["h", "i"])
                case "ふぇ": phonemes.append(contentsOf: ["h", "e"])
                case "ふぉ": phonemes.append(contentsOf: ["h", "o"])
                default:
                    matched = false
                }

                if matched {
                    i += 2
                    continue
                }
            }

            // 1文字の分解
            switch c {
            case "あ", "ぁ": phonemes.append("a")
            case "い", "ぃ": phonemes.append("i")
            case "う", "ぅ": phonemes.append("u")
            case "え", "ぇ": phonemes.append("e")
            case "お", "ぉ": phonemes.append("o")
            case "か": phonemes.append(contentsOf: ["k", "a"])
            case "き": phonemes.append(contentsOf: ["k", "i"])
            case "く": phonemes.append(contentsOf: ["k", "u"])
            case "け": phonemes.append(contentsOf: ["k", "e"])
            case "こ": phonemes.append(contentsOf: ["k", "o"])
            case "さ": phonemes.append(contentsOf: ["s", "a"])
            case "し": phonemes.append(contentsOf: ["sh", "i"])
            case "す": phonemes.append(contentsOf: ["s", "u"])
            case "せ": phonemes.append(contentsOf: ["s", "e"])
            case "そ": phonemes.append(contentsOf: ["s", "o"])
            case "た": phonemes.append(contentsOf: ["t", "a"])
            case "ち": phonemes.append(contentsOf: ["ch", "i"])
            case "つ": phonemes.append(contentsOf: ["ts", "u"])
            case "て": phonemes.append(contentsOf: ["t", "e"])
            case "と": phonemes.append(contentsOf: ["t", "o"])
            case "な": phonemes.append(contentsOf: ["n", "a"])
            case "に": phonemes.append(contentsOf: ["n", "i"])
            case "ぬ": phonemes.append(contentsOf: ["n", "u"])
            case "ね": phonemes.append(contentsOf: ["n", "e"])
            case "の": phonemes.append(contentsOf: ["n", "o"])
            case "は": phonemes.append(contentsOf: ["h", "a"])
            case "ひ": phonemes.append(contentsOf: ["h", "i"])
            case "ふ": phonemes.append(contentsOf: ["h", "u"])
            case "へ": phonemes.append(contentsOf: ["h", "e"])
            case "ほ": phonemes.append(contentsOf: ["h", "o"])
            case "ま": phonemes.append(contentsOf: ["m", "a"])
            case "み": phonemes.append(contentsOf: ["m", "i"])
            case "む": phonemes.append(contentsOf: ["m", "u"])
            case "め": phonemes.append(contentsOf: ["m", "e"])
            case "も": phonemes.append(contentsOf: ["m", "o"])
            case "や", "ゃ": phonemes.append(contentsOf: ["y", "a"])
            case "ゆ", "ゅ": phonemes.append(contentsOf: ["y", "u"])
            case "よ", "ょ": phonemes.append(contentsOf: ["y", "o"])
            case "ら": phonemes.append(contentsOf: ["r", "a"])
            case "り": phonemes.append(contentsOf: ["r", "i"])
            case "る": phonemes.append(contentsOf: ["r", "u"])
            case "れ": phonemes.append(contentsOf: ["r", "e"])
            case "ろ": phonemes.append(contentsOf: ["r", "o"])
            case "わ": phonemes.append(contentsOf: ["w", "a"])
            case "を": phonemes.append("o")
            case "ん": phonemes.append("N")
            case "っ": phonemes.append("Q")
            case "ー": phonemes.append("_")
            case "が": phonemes.append(contentsOf: ["g", "a"])
            case "ぎ": phonemes.append(contentsOf: ["gy", "i"]) // or g i
            case "ぐ": phonemes.append(contentsOf: ["g", "u"])
            case "げ": phonemes.append(contentsOf: ["g", "e"])
            case "ご": phonemes.append(contentsOf: ["g", "o"])
            case "ざ": phonemes.append(contentsOf: ["z", "a"])
            case "じ": phonemes.append(contentsOf: ["j", "i"])
            case "ず": phonemes.append(contentsOf: ["z", "u"])
            case "ぜ": phonemes.append(contentsOf: ["z", "e"])
            case "ぞ": phonemes.append(contentsOf: ["z", "o"])
            case "だ": phonemes.append(contentsOf: ["d", "a"])
            case "ぢ": phonemes.append(contentsOf: ["j", "i"])
            case "づ": phonemes.append(contentsOf: ["z", "u"])
            case "で": phonemes.append(contentsOf: ["d", "e"])
            case "ど": phonemes.append(contentsOf: ["d", "o"])
            case "ば": phonemes.append(contentsOf: ["b", "a"])
            case "び": phonemes.append(contentsOf: ["b", "i"])
            case "ぶ": phonemes.append(contentsOf: ["b", "u"])
            case "べ": phonemes.append(contentsOf: ["b", "e"])
            case "ぼ": phonemes.append(contentsOf: ["b", "o"])
            case "ぱ": phonemes.append(contentsOf: ["p", "a"])
            case "ぴ": phonemes.append(contentsOf: ["p", "i"])
            case "ぷ": phonemes.append(contentsOf: ["p", "u"])
            case "ぺ": phonemes.append(contentsOf: ["p", "e"])
            case "ぽ": phonemes.append(contentsOf: ["p", "o"])
            default:
                break
            }
            i += 1
        }
        return phonemes
    }

    /// 音素列から日本語ひらがな文字列への復元
    public func phonemesToKana(_ phonemes: [String]) -> String {
        // 特殊トークンを除去
        var filtered: [String] = []
        var pIdx = 0
        while pIdx < phonemes.count {
            let p = phonemes[pIdx]
            switch p {
            case "<pad>", "<sil>", "<sos>", "<eos>", "<unk>":
                break
            default:
                if p.hasPrefix("<res") != true {
                    filtered.append(p)
                }
            }
            pIdx += 1
        }

        var result = ""
        var i = 0
        let count = filtered.count

        while i < count {
            let p = filtered[i]

            // 2音素の結合判定 (子音/拗音 + 母音)
            if (i + 1) < count {
                let nextP = filtered[i + 1]
                var combined: String? = nil

                switch p {
                case "k":
                    switch nextP {
                    case "a": combined = "か"
                    case "i": combined = "き"
                    case "u": combined = "く"
                    case "e": combined = "け"
                    case "o": combined = "こ"
                    default: break
                    }
                case "s":
                    switch nextP {
                    case "a": combined = "さ"
                    case "i": combined = "し"
                    case "u": combined = "す"
                    case "e": combined = "せ"
                    case "o": combined = "そ"
                    default: break
                    }
                case "sh":
                    switch nextP {
                    case "a": combined = "しゃ"
                    case "i": combined = "し"
                    case "u": combined = "しゅ"
                    case "e": combined = "しぇ"
                    case "o": combined = "しょ"
                    default: break
                    }
                case "t":
                    switch nextP {
                    case "a": combined = "た"
                    case "i": combined = "ち"
                    case "u": combined = "つ"
                    case "e": combined = "て"
                    case "o": combined = "と"
                    default: break
                    }
                case "ch":
                    switch nextP {
                    case "a": combined = "ちゃ"
                    case "i": combined = "ち"
                    case "u": combined = "ちゅ"
                    case "e": combined = "ちぇ"
                    case "o": combined = "ちょ"
                    default: break
                    }
                case "ts":
                    switch nextP {
                    case "a": combined = "つぁ"
                    case "i": combined = "つぃ"
                    case "u": combined = "つ"
                    case "e": combined = "つぇ"
                    case "o": combined = "つぉ"
                    default: break
                    }
                case "n":
                    switch nextP {
                    case "a": combined = "な"
                    case "i": combined = "に"
                    case "u": combined = "ぬ"
                    case "e": combined = "ね"
                    case "o": combined = "の"
                    default: break
                    }
                case "h":
                    switch nextP {
                    case "a": combined = "は"
                    case "i": combined = "ひ"
                    case "u": combined = "ふ"
                    case "e": combined = "へ"
                    case "o": combined = "ほ"
                    default: break
                    }
                case "m":
                    switch nextP {
                    case "a": combined = "ま"
                    case "i": combined = "み"
                    case "u": combined = "む"
                    case "e": combined = "め"
                    case "o": combined = "も"
                    default: break
                    }
                case "y":
                    switch nextP {
                    case "a": combined = "や"
                    case "u": combined = "ゆ"
                    case "o": combined = "よ"
                    default: break
                    }
                case "r":
                    switch nextP {
                    case "a": combined = "ら"
                    case "i": combined = "り"
                    case "u": combined = "る"
                    case "e": combined = "れ"
                    case "o": combined = "ろ"
                    default: break
                    }
                case "w":
                    switch nextP {
                    case "a": combined = "わ"
                    case "o": combined = "を"
                    default: break
                    }
                case "g":
                    switch nextP {
                    case "a": combined = "が"
                    case "i": combined = "ぎ"
                    case "u": combined = "ぐ"
                    case "e": combined = "げ"
                    case "o": combined = "ご"
                    default: break
                    }
                case "z":
                    switch nextP {
                    case "a": combined = "ざ"
                    case "i": combined = "じ"
                    case "u": combined = "ず"
                    case "e": combined = "ぜ"
                    case "o": combined = "ぞ"
                    default: break
                    }
                case "d":
                    switch nextP {
                    case "a": combined = "だ"
                    case "i": combined = "ぢ"
                    case "u": combined = "づ"
                    case "e": combined = "で"
                    case "o": combined = "ど"
                    default: break
                    }
                case "b":
                    switch nextP {
                    case "a": combined = "ば"
                    case "i": combined = "び"
                    case "u": combined = "ぶ"
                    case "e": combined = "べ"
                    case "o": combined = "ぼ"
                    default: break
                    }
                case "p":
                    switch nextP {
                    case "a": combined = "ぱ"
                    case "i": combined = "ぴ"
                    case "u": combined = "ぷ"
                    case "e": combined = "ぺ"
                    case "o": combined = "ぽ"
                    default: break
                    }
                case "ky":
                    switch nextP {
                    case "a": combined = "きゃ"
                    case "i": combined = "き"
                    case "u": combined = "きゅ"
                    case "e": combined = "きぇ"
                    case "o": combined = "きょ"
                    default: break
                    }
                case "ny":
                    switch nextP {
                    case "a": combined = "にゃ"
                    case "i": combined = "に"
                    case "u": combined = "にゅ"
                    case "e": combined = "にぇ"
                    case "o": combined = "にょ"
                    default: break
                    }
                case "hy":
                    switch nextP {
                    case "a": combined = "ひゃ"
                    case "i": combined = "ひ"
                    case "u": combined = "ひゅ"
                    case "e": combined = "ひぇ"
                    case "o": combined = "ひょ"
                    default: break
                    }
                case "my":
                    switch nextP {
                    case "a": combined = "みゃ"
                    case "i": combined = "み"
                    case "u": combined = "みゅ"
                    case "e": combined = "みぇ"
                    case "o": combined = "みょ"
                    default: break
                    }
                case "ry":
                    switch nextP {
                    case "a": combined = "りゃ"
                    case "i": combined = "り"
                    case "u": combined = "りゅ"
                    case "e": combined = "りぇ"
                    case "o": combined = "りょ"
                    default: break
                    }
                case "gy":
                    switch nextP {
                    case "a": combined = "ぎゃ"
                    case "i": combined = "ぎ"
                    case "u": combined = "ぎゅ"
                    case "e": combined = "ぎぇ"
                    case "o": combined = "ぎょ"
                    default: break
                    }
                case "j":
                    switch nextP {
                    case "a": combined = "じゃ"
                    case "i": combined = "じ"
                    case "u": combined = "じゅ"
                    case "e": combined = "じぇ"
                    case "o": combined = "じょ"
                    default: break
                    }
                case "by":
                    switch nextP {
                    case "a": combined = "びゃ"
                    case "i": combined = "び"
                    case "u": combined = "びゅ"
                    case "e": combined = "びぇ"
                    case "o": combined = "びょ"
                    default: break
                    }
                case "py":
                    switch nextP {
                    case "a": combined = "ぴゃ"
                    case "i": combined = "ぴ"
                    case "u": combined = "ぴゅ"
                    case "e": combined = "ぴぇ"
                    case "o": combined = "ぴょ"
                    default: break
                    }
                default:
                    break
                }

                if let text = combined {
                    result.append(text)
                    i += 2
                    continue
                }
            }

            // 単一音素の復元
            switch p {
            case "a": result.append("あ")
            case "i": result.append("い")
            case "u": result.append("う")
            case "e": result.append("え")
            case "o": result.append("お")
            case "N": result.append("ん")
            case "Q": result.append("っ")
            case "_": result.append("ー")
            case "k": result.append("く")
            case "s": result.append("す")
            case "sh": result.append("し")
            case "t": result.append("と")
            case "ch": result.append("ち")
            case "ts": result.append("つ")
            case "n": result.append("ん")
            case "h": result.append("ふ")
            case "m": result.append("む")
            case "y": result.append("い")
            case "r": result.append("る")
            case "w": result.append("う")
            case "g": result.append("ぐ")
            case "z": result.append("ず")
            case "j": result.append("じ")
            case "d": result.append("ど")
            case "b": result.append("ぶ")
            case "p": result.append("ぷ")
            default:
                break
            }
            i += 1
        }
        return result
    }

    /// トークン ID 系列から日本語ひらがな文字列へ変換
    public func tokensToText(_ tokenIds: [Int]) -> String {
        var phonemes: [String] = []
        var i = 0
        while i < tokenIds.count {
            let id = tokenIds[i]
            if isSilence(id) != true {
                phonemes.append(token(for: id))
            }
            i += 1
        }
        return phonemesToKana(phonemes)
    }

    /// 日本語文字列からトークン ID 系列へ変換
    public func textToTokens(_ text: String) -> [Int] {
        let phonemes = kanaToPhonemes(text)
        var ids = [Int](repeating: 0, count: phonemes.count)
        var i = 0
        while i < phonemes.count {
            ids[i] = id(for: phonemes[i])
            i += 1
        }
        return ids
    }
}
