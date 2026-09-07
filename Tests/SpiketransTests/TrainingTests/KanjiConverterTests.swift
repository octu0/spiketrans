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

    // MARK: - 長音符「ー」復元と正規化テスト

    func testCombiningMacronRestoration() {
        let converter = KanjiConverter()

        // カタカナ外来語の長音符が正しく復元されること
        XCTAssertEqual(converter.convertToHiragana("マレーシア"), "まれーしあ")
        XCTAssertEqual(converter.convertToHiragana("データ"), "でーた")
        XCTAssertEqual(converter.convertToHiragana("あーあ"), "あーあ")
        XCTAssertEqual(converter.convertToHiragana("コンピューター"), "こんぴゅーたー")
        XCTAssertEqual(converter.convertToHiragana("パーティー"), "ぱーてぃー")
        XCTAssertEqual(converter.convertToHiragana("コーヒー"), "こーひー")
        XCTAssertEqual(converter.convertToHiragana("サーバー"), "さーばー")
        XCTAssertEqual(converter.convertToHiragana("ビール"), "びーる")
    }

    func testWaveDashAndProlongedNormalization() {
        let converter = KanjiConverter()

        // 波ダッシュ「〜」および全角チルダ「～」の長音符正規化
        XCTAssertEqual(converter.convertToHiragana("ら〜めん"), "らーめん")
        XCTAssertEqual(converter.convertToHiragana("ら～めん"), "らーめん")
        XCTAssertEqual(converter.convertToHiragana("う〜ん"), "うーん")
        XCTAssertEqual(converter.convertToHiragana("あ〜〜"), "あーー")

        // 半角カタカナ長音符「ｰ」(U+FF70)、ホリゾンタルバー「―」(U+2015)、EMダッシュ「—」(U+2014) の正規化
        XCTAssertEqual(converter.convertToHiragana("らｰめん"), "らーめん")
        XCTAssertEqual(converter.convertToHiragana("あ――"), "あーー")
        XCTAssertEqual(converter.convertToHiragana("え——"), "えーー")

        // normalizeKana / kanaOnly 単体での正規化
        XCTAssertEqual(converter.normalizeKana("ら〜めん"), "らーめん")
        XCTAssertEqual(converter.normalizeKana("う～ん"), "うーん")
        XCTAssertEqual(converter.normalizeKana("らｰめん"), "らーめん")
        XCTAssertEqual(converter.kanaOnly("あ〜い～う"), "あーいーう")
        XCTAssertEqual(converter.kanaOnly("らｰめん"), "らーめん")
        XCTAssertEqual(converter.kanaOnly("あ――"), "あーー")
    }

    func testWagoKangoNotProlonged() {
        let converter = KanjiConverter()

        // 和語・漢語はマクロンが出力されないため「ー」化しないことを保証
        XCTAssertEqual(converter.convertToHiragana("東京"), "とうきょう")
        XCTAssertEqual(converter.convertToHiragana("とうきょう"), "とうきょう")
        XCTAssertEqual(converter.convertToHiragana("京都"), "きょうと")
        XCTAssertEqual(converter.convertToHiragana("学校"), "がっこう")
        XCTAssertEqual(converter.convertToHiragana("大きい"), "おおきい")
        XCTAssertEqual(converter.convertToHiragana("おおきい"), "おおきい")
        XCTAssertEqual(converter.convertToHiragana("おじいさん"), "おじいさん")
        XCTAssertEqual(converter.convertToHiragana("妹"), "いもうと")
        XCTAssertEqual(converter.convertToHiragana("弟"), "おとうと")
        XCTAssertEqual(converter.convertToHiragana("氷"), "こおり")
    }

    func testAlphabetTokenProtectionAndBilingualSeed() {
        let converter = KanjiConverter()

        // 主要テック用語・外来語が正しいかな読みになること（「いぷほね」「あっぷれ」に崩れない）
        let tokens = converter.tokenize("iPhone")
        XCTAssertEqual(tokens.count, 1)
        XCTAssertEqual(tokens.first?.surface, "iPhone")
        XCTAssertEqual(tokens.first?.reading, "あいふぉーん")

        let appleTokens = converter.tokenize("Apple")
        XCTAssertEqual(appleTokens.count, 1)
        XCTAssertEqual(appleTokens.first?.surface, "Apple")
        XCTAssertEqual(appleTokens.first?.reading, "あっぷる")

        let googleTokens = converter.tokenize("Google")
        XCTAssertEqual(googleTokens.first?.reading, "ぐーぐる")

        let aiTokens = converter.tokenize("AI")
        XCTAssertEqual(aiTokens.first?.reading, "えーあい")

        let pcTokens = converter.tokenize("PC")
        XCTAssertEqual(pcTokens.first?.reading, "ぴーしー")

        let wifiTokens = converter.tokenize("WiFi")
        XCTAssertEqual(wifiTokens.first?.reading, "わいふぁい")

        let wifiHyphenTokens = converter.tokenize("Wi-Fi")
        XCTAssertEqual(wifiHyphenTokens.first?.surface, "Wi-Fi")
        XCTAssertEqual(wifiHyphenTokens.first?.reading, "わいふぁい")
        XCTAssertEqual(converter.convertToHiragana("Wi-Fi"), "わいふぁい")

        let openAiTokens = converter.tokenize("OpenAI")
        XCTAssertEqual(openAiTokens.first?.reading, "おーぷんえーあい")

        let airbnbTokens = converter.tokenize("AirBnB")
        XCTAssertEqual(airbnbTokens.first?.reading, "えあびー")

        // 大文字頭字語 (Acronym) の単文字読み展開
        let gpuTokens = converter.tokenize("GPU")
        XCTAssertEqual(gpuTokens.first?.reading, "じーぴーゆー")

        let sdkTokens = converter.tokenize("SDK")
        XCTAssertEqual(sdkTokens.first?.reading, "えすでぃーけー")

        // 未知の英単語はローマ字読みへの異常崩れを防ぐため空文字で保護
        let unknownTokens = converter.tokenize("unknownfoobar")
        XCTAssertEqual(unknownTokens.count, 1)
        XCTAssertEqual(unknownTokens.first?.reading, "")

        // 混在文のひらがな変換
        let mixed = converter.convertToHiragana("iPhoneとMacBookを使って作業する")
        XCTAssertEqual(mixed, "あいふぉーんとまっくぶっくをつかってさぎょうする")
    }

    // MARK: - フェーズ2: 境界値・悪意のある入力・不整合の検証テスト

    func testPhase2EdgeCasesAndExtremeInputs() {
        let converter = KanjiConverter()

        // 1. 空文字入力
        XCTAssertTrue(converter.tokenize("").isEmpty)
        XCTAssertEqual(converter.normalizeKana(""), "")
        XCTAssertEqual(converter.convertToHiragana(""), "")

        // 2. 波ダッシュ・引き伸ばし記号のみの連続
        XCTAssertEqual(converter.convertToHiragana("〜〜〜"), "ーーー")
        XCTAssertEqual(converter.convertToHiragana("あ〜〜〜〜"), "あーーーー")

        // 3. 全角英字 (Fullwidth Alphabet) の認識
        let fullwidthTokens = converter.tokenize("ｉＰｈｏｎｅ")
        XCTAssertEqual(fullwidthTokens.first?.reading, "あいふぉーん")

        // 4. 大文字・小文字のバリエーション
        let upperIPhone = converter.tokenize("IPHONE")
        XCTAssertEqual(upperIPhone.first?.reading, "あいふぉーん")

        let lowerApple = converter.tokenize("apple")
        XCTAssertEqual(lowerApple.first?.reading, "あっぷる")

        // 5. 単一アルファベット文字
        let singleA = converter.tokenize("A")
        XCTAssertEqual(singleA.first?.reading, "えー")
        let singleZ = converter.tokenize("Z")
        XCTAssertEqual(singleZ.first?.reading, "ぜっと")

        // 6. 未知の大文字頭字語 (Acronym)
        let unknownAcronym = converter.tokenize("XYZ")
        XCTAssertEqual(unknownAcronym.first?.reading, "えっくすわいぜっと")

        // 7. 数字・英語・長音・波ダッシュが複合した文章
        let complexText = "iPhone 15の価格は１４万円〜１５万円です。"
        let hira = converter.convertToHiragana(complexText)
        XCTAssertTrue(hira.contains("あいふぉーん"))
        XCTAssertTrue(hira.contains("じゅうご"))
        XCTAssertTrue(hira.contains("ー"))

        // 8. 全角ハイフン・各種ダッシュ結合英単語 (Wi-Fiバリエーション)
        let fullwidthWifi = converter.tokenize("Ｗｉ－Ｆｉ")
        XCTAssertEqual(fullwidthWifi.first?.reading, "わいふぁい")
        XCTAssertEqual(converter.convertToHiragana("Ｗｉ－Ｆｉ"), "わいふぁい")

        let enDashWifi = converter.tokenize("Wi–Fi")
        XCTAssertEqual(enDashWifi.first?.reading, "わいふぁい")
        XCTAssertEqual(converter.convertToHiragana("Wi–Fi"), "わいふぁい")

        let emDashWifi = converter.tokenize("Wi—Fi")
        XCTAssertEqual(emDashWifi.first?.reading, "わいふぁい")
        XCTAssertEqual(converter.convertToHiragana("Wi—Fi"), "わいふぁい")

        let barWifi = converter.tokenize("Wi―Fi")
        XCTAssertEqual(barWifi.first?.reading, "わいふぁい")
        XCTAssertEqual(converter.convertToHiragana("Wi―Fi"), "わいふぁい")

        // 9. Mirrativ (シード辞書からの動的自動導出)
        let mirrativTokens = converter.tokenize("Mirrativ")
        XCTAssertEqual(mirrativTokens.first?.reading, "みらてぃぶ")
        XCTAssertEqual(converter.convertToHiragana("Mirrativ"), "みらてぃぶ")

        // 10. B-to-B および複合ハイフン頭字語
        let btobTokens = converter.tokenize("B-to-B")
        XCTAssertEqual(btobTokens.first?.reading, "びーとぅーびー")
        XCTAssertEqual(converter.convertToHiragana("B-to-B"), "びーとぅーびー")

        let hyphenAcronym = converter.tokenize("Ｘ－Ｙ－Ｚ")
        XCTAssertEqual(hyphenAcronym.first?.reading, "えっくすわいぜっと")

        // 11. toPhonemeTokenIds / toPhonemes 簡素化パスの検証
        let phoneIds = converter.toPhonemeTokenIds("Wi-Fi")
        XCTAssertFalse(phoneIds.isEmpty)
        let phonemes = converter.toPhonemes("Wi-Fi")
        XCTAssertEqual(phonemes, ["w", "a", "i", "h", "a", "i"])
    }
}
