import XCTest
@testable import Spiketrans

final class KanjiConverterTests: XCTestCase {
    func testKanaNormalization() {
        let converter = KanjiConverter()
        
        let text = "カタカナとひらがな"
        let normalized = converter.normalizeKana(text)
        XCTAssertEqual(normalized, "かたかなとひらがな")
        
        // 漢字はそのまま保持されること（ピンインに変換されないこと）
        let kanjiText = "日本語"
        let kanjiRes = converter.normalizeKana(kanjiText)
        XCTAssertEqual(kanjiRes, "日本語")
    }

    func testKanaToPhonemeTokens() {
        let converter = KanjiConverter()
        
        let text = "おはよう"
        let tokens = converter.toPhonemeTokenIds(text)
        XCTAssertFalse(tokens.isEmpty)
        
        let phonemes = converter.toPhonemes(text)
        XCTAssertFalse(phonemes.isEmpty)
        XCTAssertTrue(phonemes.contains("o"))
    }

    func testEmptyAndSpecialCharacters() {
        let converter = KanjiConverter()
        
        let empty = converter.normalizeKana("")
        XCTAssertEqual(empty, "")
        
        let emptyTokens = converter.toPhonemeTokenIds("")
        XCTAssertTrue(emptyTokens.isEmpty)
    }
}
