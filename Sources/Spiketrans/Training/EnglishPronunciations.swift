import Foundation

/// 英語の発音辞書 (CMU 形式: `word PH PH ...`、音素は ARPAbet) から、英単語のカタカナ英語の読みを作る。
///
/// 配信の書き起こしには英語がそのまま残る (歌詞の "I wanna be a pop star"、"YouTube")。
/// 音響モデルの教師はかななので、英語の音素列を日本語の音韻に写して読みを付ける。
/// 表層は英語のまま第2段辞書に入るので、推論では英語で出せる。
public final class EnglishPronunciations: @unchecked Sendable {
    private let phones: [String: [String]]

    /// CMU 形式の辞書ファイルを読む
    public convenience init(contentsOfFile path: String) throws {
        let content = try String(contentsOfFile: path, encoding: .utf8)
        self.init(lines: content.split(separator: "\n").map(String.init))
    }

    public var count: Int {
        return phones.count
    }

    /// `word PH PH ...` の行から作る。`word(2)` の別読みは飛ばし、`#` 以降のコメントは捨てる
    public init(lines: [String]) {
        var table: [String: [String]] = [:]
        table.reserveCapacity(lines.count)
        for rawLine in lines {
            var line = rawLine
            if let hash = line.firstIndex(of: "#") {
                line = String(line[..<hash])
            }
            let parts = line.split(separator: " ", omittingEmptySubsequences: true)
            if parts.count < 2 {
                continue
            }
            let word = String(parts[0]).lowercased()
            if word.hasSuffix(")") {
                continue
            }
            var stripped: [String] = []
            stripped.reserveCapacity(parts.count - 1)
            var i = 1
            while i < parts.count {
                stripped.append(String(parts[i].filter { $0.isNumber != true }))
                i += 1
            }
            if table[word] == nil {
                table[word] = stripped
            }
        }
        self.phones = table
    }

    /// 英単語のかな読み。辞書に無ければ nil
    public func reading(of word: String) -> String? {
        guard let ph = phones[word.lowercased()] else {
            return nil
        }
        return Self.kana(fromPhones: ph)
    }

    // MARK: - ARPAbet → かな

    /// 母音: 核となる母音 (a/i/u/e/o) と、そのあとに続く尾 (長音・二重母音の 2 音目)
    static let vowels: [String: (Character, String)] = [
        "AA": ("a", ""), "AE": ("a", ""), "AH": ("a", ""), "AO": ("o", ""), "AW": ("a", "う"),
        "AY": ("a", "い"), "EH": ("e", ""), "ER": ("a", "ー"), "EY": ("e", "ー"), "IH": ("i", ""),
        "IY": ("i", "ー"), "OW": ("o", "ー"), "OY": ("o", "い"), "UH": ("u", ""), "UW": ("u", "ー"),
    ]

    /// 短母音 (直後の破裂音を促音にする)
    static let shortVowels: Set<String> = ["AA", "AE", "AH", "AO", "EH", "IH", "UH"]

    /// 子音行: a i u e o の順
    static let rows: [String: [String]] = [
        "B": ["ば", "び", "ぶ", "べ", "ぼ"], "CH": ["ちゃ", "ち", "ちゅ", "ちぇ", "ちょ"],
        "D": ["だ", "でぃ", "どぅ", "で", "ど"], "DH": ["ざ", "じ", "ず", "ぜ", "ぞ"],
        "F": ["ふぁ", "ふぃ", "ふ", "ふぇ", "ふぉ"], "G": ["が", "ぎ", "ぐ", "げ", "ご"],
        "HH": ["は", "ひ", "ふ", "へ", "ほ"], "JH": ["じゃ", "じ", "じゅ", "じぇ", "じょ"],
        "K": ["か", "き", "く", "け", "こ"], "L": ["ら", "り", "る", "れ", "ろ"],
        "M": ["ま", "み", "む", "め", "も"], "N": ["な", "に", "ぬ", "ね", "の"],
        "NG": ["んが", "んぎ", "んぐ", "んげ", "んご"], "P": ["ぱ", "ぴ", "ぷ", "ぺ", "ぽ"],
        "R": ["ら", "り", "る", "れ", "ろ"], "S": ["さ", "し", "す", "せ", "そ"],
        "SH": ["しゃ", "し", "しゅ", "しぇ", "しょ"], "T": ["た", "てぃ", "とぅ", "て", "と"],
        "TH": ["さ", "し", "す", "せ", "そ"], "V": ["ゔぁ", "ゔぃ", "ゔ", "ゔぇ", "ゔぉ"],
        "W": ["わ", "うぃ", "う", "うぇ", "うぉ"], "Y": ["や", "い", "ゆ", "いぇ", "よ"],
        "Z": ["ざ", "じ", "ず", "ぜ", "ぞ"], "ZH": ["じゃ", "じ", "じゅ", "じぇ", "じょ"],
    ]

    /// 子音 + Y + 母音 (tube = T Y UW) の拗音。い段のかなに小書きを付ける
    static let palatalTails: [Character: String] = ["a": "ゃ", "i": "", "u": "ゅ", "e": "ぇ", "o": "ょ"]

    /// 母音が続かない子音に補う母音 (語末や子音連続)
    static func epenthetic(_ consonant: String) -> String {
        switch consonant {
        case "T":
            return "と"
        case "D":
            return "ど"
        case "N":
            return "ん"
        case "M":
            return "む"
        case "NG":
            return "んぐ"
        case "L":
            return "る"
        case "R":
            return "ー"
        case "CH":
            return "ち"
        case "JH":
            return "じ"
        case "SH":
            return "しゅ"
        case "ZH":
            return "じゅ"
        case "W", "Y":
            return ""
        default:
            if let row = rows[consonant] {
                return row[2]
            }
            return ""
        }
    }

    static func vowelIndex(_ v: Character) -> Int {
        switch v {
        case "a":
            return 0
        case "i":
            return 1
        case "u":
            return 2
        case "e":
            return 3
        default:
            return 4
        }
    }

    /// AA (pot, star の母音) は日本語では お に写す (pop → ぽっぷ、hot → ほっと) が、
    /// R が続くときは あ (star → すたー)、W のあとも あ (wanna → わな)
    static func vowelNucleus(_ vowel: String, next: String?, consonant: String?, fallback: Character) -> Character {
        if vowel != "AA" {
            return fallback
        }
        if next == "R" || consonant == "W" {
            return "a"
        }
        return "o"
    }

    /// 母音単独のかな
    static func plainVowel(_ v: Character) -> String {
        return ["あ", "い", "う", "え", "お"][vowelIndex(v)]
    }

    /// ARPAbet の音素列をかなに写す。
    /// 子音は次の母音と結んで 1 かなにし、母音が続かない子音には母音を補う。
    /// 短母音の直後で語末または子音が続く破裂音は促音 (pop → ぽっぷ)。
    /// 母音のあとの R は長音 (star → すたー)。
    public static func kana(fromPhones phones: [String]) -> String {
        var out = ""
        var i = 0
        var prevVowel: String? = nil
        while i < phones.count {
            let p = phones[i]
            if let (nucleus, tail) = vowels[p] {
                out += plainVowel(vowelNucleus(p, next: i + 1 < phones.count ? phones[i + 1] : nil, consonant: nil, fallback: nucleus))
                appendTail(&out, tail)
                prevVowel = p
                i += 1
                continue
            }
            // NG の直後の K/G は ん に吸収する (thank → さんく)
            if p == "NG" && i + 1 < phones.count && (phones[i + 1] == "K" || phones[i + 1] == "G") {
                out += "ん"
                prevVowel = nil
                i += 1
                continue
            }
            // 子音。次に母音が来れば結ぶ
            var j = i + 1
            var palatal = false
            if j < phones.count && phones[j] == "Y" && j + 1 < phones.count && vowels[phones[j + 1]] != nil && p != "Y" {
                palatal = true
                j += 1
            }
            if j < phones.count, let (rawNucleus, tail) = vowels[phones[j]] {
                let nucleus = vowelNucleus(phones[j], next: j + 1 < phones.count ? phones[j + 1] : nil, consonant: p, fallback: rawNucleus)
                if palatal {
                    // 拗音の頭は い段のかな。T/D はてぃ・でぃではなく ち・じ (tube → ちゅー)
                    let base: String
                    switch p {
                    case "T":
                        base = "ち"
                    case "D":
                        base = "じ"
                    default:
                        base = rows[p]?[1] ?? ""
                    }
                    out += base + (palatalTails[nucleus] ?? "")
                } else if (p == "K" || p == "G") && phones[j] == "AE" {
                    // cat → きゃっと、gap → ぎゃっぷ。K/G の後の AE はカタカナでは拗音になる
                    out += (rows[p]?[1] ?? "") + "ゃ"
                } else if let row = rows[p] {
                    out += row[vowelIndex(nucleus)]
                } else {
                    out += plainVowel(nucleus)
                }
                appendTail(&out, tail)
                prevVowel = phones[j]
                i = j + 1
                continue
            }
            // 母音が続かない子音
            if p == "R" {
                if prevVowel != nil {
                    appendTail(&out, "ー")
                } else {
                    out += "る"
                }
            } else {
                let isStop = ["P", "T", "K", "B", "D", "G", "CH"].contains(p)
                if isStop, let pv = prevVowel, shortVowels.contains(pv) {
                    out += "っ"
                }
                out += epenthetic(p)
            }
            prevVowel = nil
            i += 1
        }
        return out
    }

    /// 長音の重なりを避けて尾を足す
    static func appendTail(_ out: inout String, _ tail: String) {
        if tail == "ー" && out.hasSuffix("ー") {
            return
        }
        out += tail
    }
}
