import Foundation

/// 漢字・かな・記号を含む言語モデル用テキスト語彙辞書
public struct TextVocabulary: Sendable {
    public static let padId = 0
    public static let unkId = 1
    public static let sosId = 2
    public static let eosId = 3

    public let size: Int
    private let tokenList: [Character]
    private let charToIdTable: [Character: Int]

    /// コーパステキスト配列から語彙辞書を動的構築
    public init(corpus: [String] = []) {
        var set: Set<Character> = []
        var cIdx = 0
        while cIdx < corpus.count {
            let line = corpus[cIdx]
            for c in line {
                set.insert(c)
            }
            cIdx += 1
        }

        // 基本特殊トークン: pad(0), unk(1), sos(2), eos(3)
        // Character として一意なプレースホルダー
        let specialChars: [Character] = ["\u{0000}", "\u{0001}", "\u{0002}", "\u{0003}"]
        
        let sortedChars = Array(set).sorted { (a, b) -> Bool in
            a < b
        }

        var list = specialChars
        var sIdx = 0
        while sIdx < sortedChars.count {
            let c = sortedChars[sIdx]
            if specialChars.contains(c) != true {
                list.append(c)
            }
            sIdx += 1
        }

        self.size = list.count
        self.tokenList = list

        var table: [Character: Int] = [:]
        var i = 0
        while i < list.count {
            table[list[i]] = i
            i += 1
        }
        self.charToIdTable = table
    }

    /// 事前定義文字リストから直接構築
    public init(characters: [Character]) {
        let specialChars: [Character] = ["\u{0000}", "\u{0001}", "\u{0002}", "\u{0003}"]
        var list = specialChars
        var sIdx = 0
        while sIdx < characters.count {
            let c = characters[sIdx]
            if specialChars.contains(c) != true {
                list.append(c)
            }
            sIdx += 1
        }
        self.size = list.count
        self.tokenList = list

        var table: [Character: Int] = [:]
        var i = 0
        while i < list.count {
            table[list[i]] = i
            i += 1
        }
        self.charToIdTable = table
    }

    /// 特殊トークンを除いた文字列。重みファイルへ同梱して ID 割当を再現するために使う。
    /// これを `init(characters:)` に渡せば同じ語彙が復元できる
    public var serializedCharacters: String {
        let specialChars: Set<Character> = ["\u{0000}", "\u{0001}", "\u{0002}", "\u{0003}"]
        var result = ""
        var i = 0
        while i < tokenList.count {
            let c = tokenList[i]
            if specialChars.contains(c) != true {
                result.append(c)
            }
            i += 1
        }
        return result
    }

    /// `serializedCharacters` から語彙を復元する
    public init(serializedCharacters: String) {
        self.init(characters: Array(serializedCharacters))
    }

    /// 文字からトークン ID を取得
    public func id(for char: Character) -> Int {
        switch charToIdTable[char] {
        case .some(let val):
            return val
        case .none:
            return Self.unkId
        }
    }

    /// トークン ID から文字を取得
    public func char(for id: Int) -> Character {
        switch true {
        case id < 0:
            return "?"
        case tokenList.count <= id:
            return "?"
        case id == Self.padId:
            return " "
        case id == Self.unkId:
            return "?"
        case id == Self.sosId:
            return "^"
        case id == Self.eosId:
            return "$"
        default:
            return tokenList[id]
        }
    }

    /// テキスト文字列からトークン ID 列へ変換
    public func textToIds(_ text: String) -> [Int] {
        var ids = [Int](repeating: 0, count: text.count)
        var i = 0
        for c in text {
            ids[i] = id(for: c)
            i += 1
        }
        return ids
    }

    /// トークン ID 列からテキスト文字列へ復元
    public func idsToText(_ ids: [Int]) -> String {
        var result = ""
        var i = 0
        while i < ids.count {
            let idVal = ids[i]
            switch idVal {
            case Self.padId, Self.sosId, Self.eosId:
                break
            case Self.unkId:
                result.append("?")
            default:
                if 0 <= idVal && idVal < tokenList.count {
                    result.append(tokenList[idVal])
                }
            }
            i += 1
        }
        return result
    }
}
