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

    func testNumberReadingPositional() {
        XCTAssertEqual(KanjiConverter.numberReading("0"), "ぜろ")
        XCTAssertEqual(KanjiConverter.numberReading("7"), "なな")
        XCTAssertEqual(KanjiConverter.numberReading("20"), "にじゅう")
        XCTAssertEqual(KanjiConverter.numberReading("213"), "にひゃくじゅうさん")
        XCTAssertEqual(KanjiConverter.numberReading("226"), "にひゃくにじゅうろく")
        XCTAssertEqual(KanjiConverter.numberReading("300"), "さんびゃく")
        XCTAssertEqual(KanjiConverter.numberReading("600"), "ろっぴゃく")
        XCTAssertEqual(KanjiConverter.numberReading("800"), "はっぴゃく")
        XCTAssertEqual(KanjiConverter.numberReading("3000"), "さんぜん")
        XCTAssertEqual(KanjiConverter.numberReading("8000"), "はっせん")
        XCTAssertEqual(KanjiConverter.numberReading("1473"), "せんよんひゃくななじゅうさん")
        XCTAssertEqual(KanjiConverter.numberReading("10000"), "いちまん")
        XCTAssertEqual(KanjiConverter.numberReading("100000000"), "いちおく")
        XCTAssertEqual(KanjiConverter.numberReading("12345678"), "せんにひゃくさんじゅうよんまんごせんろっぴゃくななじゅうはち")
        XCTAssertEqual(KanjiConverter.numberReading("１４７３"), "せんよんひゃくななじゅうさん")
    }

    func testNumberReadingDigitWise() {
        // 先頭ゼロ (電話番号等) は 1 桁ずつ読む
        XCTAssertEqual(KanjiConverter.numberReading("0120"), "ぜろいちにぜろ")
        // 数字以外が混ざったら nil
        XCTAssertNil(KanjiConverter.numberReading("１２c"))
        XCTAssertNil(KanjiConverter.numberReading("abc"))
        XCTAssertNil(KanjiConverter.numberReading(""))
    }

    func testTokenizeIncludesNumberReadings() {
        let converter = KanjiConverter()
        XCTAssertEqual(converter.convertToHiragana("１４７３年"), "せんよんひゃくななじゅうさんねん")
        XCTAssertEqual(converter.convertToHiragana("２０億円"), "にじゅうおくえん")
    }
}
