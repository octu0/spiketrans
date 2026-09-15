import XCTest
@testable import Spiketrans

final class EnglishPronunciationsTests: XCTestCase {
    private let lines = [
        "youtube Y UW1 T Y UW2 B",
        "wanna W AA1 N AH0",
        "pop P AA1 P",
        "star S T AA1 R",
        "goodbye G UH2 D B AY1",
        "thank TH AE1 NG K",
        "sweet S W IY1 T",
        "dream D R IY1 M",
        "heart HH AA1 R T",
        "hello HH AH0 L OW1",
        "cat K AE1 T",
        "water W AO1 T ER0",
        "coin K OY1 N",
        "sing S IH1 NG",
        "i AY1",
        "a AH0",
        "a(2) EY1",
        "bus B AH1 S # comment",
    ]

    func testKanaFromPhones() {
        let dict = EnglishPronunciations(lines: lines)
        XCTAssertEqual(dict.count, 17)
        let expected: [String: String] = [
            "youtube": "ゆーちゅーぶ", "wanna": "わな", "pop": "ぽっぷ", "star": "すたー",
            "goodbye": "ぐっどばい", "thank": "さんく", "sweet": "すうぃーと", "dream": "どりーむ",
            "heart": "はーと", "hello": "はろー", "cat": "きゃっと", "water": "うぉたー",
            "coin": "こいん", "sing": "しんぐ", "i": "あい", "a": "あ", "bus": "ばす",
        ]
        for (word, kana) in expected {
            XCTAssertEqual(dict.reading(of: word), kana, word)
        }
        XCTAssertEqual(dict.reading(of: "YouTube"), "ゆーちゅーぶ", "大文字小文字を区別しない")
        XCTAssertNil(dict.reading(of: "nyanta"), "辞書に無い語は nil")
    }

    /// 発音辞書を渡した変換器は英文をカタカナ英語で読む
    func testConverterUsesDictionary() {
        let dict = EnglishPronunciations(lines: lines + ["be B IY1", "pop P AA1 P", "star S T AA1 R", "wanna W AA1 N AH0"])
        let converter = KanjiConverter(english: dict)
        XCTAssertEqual(converter.convertToHiragana("I wanna be a pop star"), "あいわなびーあぽっぷすたー")
        XCTAssertEqual(converter.convertToHiragana("YouTubeを見てる"), "ゆーちゅーぶおみてる")
        // 辞書なしなら英単語は空読み
        XCTAssertEqual(KanjiConverter().convertToHiragana("wanna be"), "")
    }
}
