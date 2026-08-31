import XCTest
@testable import Spiketrans

final class TextVocabularyTests: XCTestCase {
    func testTextVocabularyFromCorpus() {
        let corpus = ["今日は晴天です", "明日も良い天気", "よろしくお願いします"]
        let vocab = TextVocabulary(corpus: corpus)

        XCTAssertTrue(4 <= vocab.size)

        let kyoId = vocab.id(for: "今")
        XCTAssertTrue(0 <= kyoId)
        XCTAssertEqual(vocab.char(for: kyoId), "今")

        let unkId = vocab.id(for: "🔥")
        XCTAssertEqual(unkId, TextVocabulary.unkId)
    }

    func testBidirectionalTextConversion() {
        let corpus = ["あいうえお", "漢字テスト"]
        let vocab = TextVocabulary(corpus: corpus)

        let original = "漢字テスト"
        let ids = vocab.textToIds(original)
        XCTAssertEqual(ids.count, 5)

        let recovered = vocab.idsToText(ids)
        XCTAssertEqual(recovered, original)
    }

    func testSpecialTokens() {
        let vocab = TextVocabulary(corpus: ["テスト"])
        XCTAssertEqual(vocab.id(for: "\u{0000}"), TextVocabulary.padId)
        XCTAssertEqual(vocab.id(for: "\u{0001}"), TextVocabulary.unkId)
        XCTAssertEqual(vocab.id(for: "\u{0002}"), TextVocabulary.sosId)
        XCTAssertEqual(vocab.id(for: "\u{0003}"), TextVocabulary.eosId)
    }
}
