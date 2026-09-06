import XCTest
@testable import Spiketrans

final class KanaKanjiDecoderTests: XCTestCase {

    // MARK: - 1. シード辞書テーブル (SeedVocabulary) の構成検証

    func testSeedVocabularyContents() {
        let entries = SeedVocabulary.entries
        // 50件以上の十分な語彙が存在すること
        XCTAssertLessThanOrEqual(50, entries.count)

        // 略称・通称エントリの存在確認
        let subaraEntries = entries.filter { $0.reading == "すたば" && $0.surface == "スターバックス" }
        XCTAssertFalse(subaraEntries.isEmpty)

        let macdoEntries = entries.filter { $0.reading == "まくど" && $0.surface == "マクドナルド" }
        XCTAssertFalse(macdoEntries.isEmpty)

        let mercariEntries = entries.filter { $0.reading == "めすか" && $0.surface == "メルカリ" }
        XCTAssertFalse(mercariEntries.isEmpty)

        let airbnbEntries = entries.filter { $0.reading == "えあびー" && $0.surface == "AirBnB" }
        XCTAssertFalse(airbnbEntries.isEmpty)

        // 和英併記エントリの存在確認
        let iphoneEntries = entries.filter { $0.reading == "あいふぉーん" && $0.surface == "iPhone" }
        XCTAssertFalse(iphoneEntries.isEmpty)

        let appleEntriesEn = entries.filter { $0.reading == "あっぷる" && $0.surface == "Apple" }
        let appleEntriesJa = entries.filter { $0.reading == "あっぷる" && $0.surface == "アップル" }
        XCTAssertFalse(appleEntriesEn.isEmpty)
        XCTAssertFalse(appleEntriesJa.isEmpty)

        let googleEntries = entries.filter { $0.reading == "ぐーぐる" && $0.surface == "Google" }
        XCTAssertFalse(googleEntries.isEmpty)

        let openAiEntries = entries.filter { $0.reading == "おーぷんえーあい" && $0.surface == "OpenAI" }
        XCTAssertFalse(openAiEntries.isEmpty)

        let tiktokEntries = entries.filter { $0.reading == "てぃっくとっく" && $0.surface == "TikTok" }
        XCTAssertFalse(tiktokEntries.isEmpty)

        let netflixEntries = entries.filter { $0.reading == "ねっとふりっくす" && $0.surface == "Netflix" }
        XCTAssertFalse(netflixEntries.isEmpty)

        let youtubeEntries = entries.filter { $0.reading == "ゆーちゅーぶ" && $0.surface == "YouTube" }
        XCTAssertFalse(youtubeEntries.isEmpty)
    }

    // MARK: - 2. KanaKanjiDictionary への自動統合検証

    func testKanaKanjiDictionarySeedIntegration() {
        // デフォルト初期化でシード語彙が自動ロードされること
        let dict = KanaKanjiDictionary()
        XCTAssertLessThanOrEqual(50, dict.count)

        // シード語彙の完全一致検索
        let iphoneHits = dict.lookupExact(reading: "あいふぉーん")
        XCTAssertFalse(iphoneHits.isEmpty)
        XCTAssertEqual(iphoneHits.first?.surface, "iPhone")

        let starbucksHits = dict.lookupExact(reading: "すたば")
        XCTAssertFalse(starbucksHits.isEmpty)
        XCTAssertEqual(starbucksHits.first?.surface, "スターバックス")

        let macdoHits = dict.lookupExact(reading: "まくど")
        XCTAssertFalse(macdoHits.isEmpty)
        XCTAssertEqual(macdoHits.first?.surface, "マクドナルド")

        let mercariHits = dict.lookupExact(reading: "めすか")
        XCTAssertFalse(mercariHits.isEmpty)
        XCTAssertEqual(mercariHits.first?.surface, "メルカリ")

        let airbnbHits = dict.lookupExact(reading: "えあびー")
        XCTAssertFalse(airbnbHits.isEmpty)
        XCTAssertEqual(airbnbHits.first?.surface, "AirBnB")

        let openAiHits = dict.lookupExact(reading: "おーぷんえーあい")
        XCTAssertFalse(openAiHits.isEmpty)
        XCTAssertEqual(openAiHits.first?.surface, "OpenAI")

        // includeSeed: false の場合は空で初期化されること
        let emptyDict = KanaKanjiDictionary(includeSeed: false)
        XCTAssertEqual(emptyDict.count, 0)
    }

    // MARK: - 3. かな漢字デコーダでのシード語彙復元検証

    func testKanaKanjiDecoderSeedDecoding() {
        let dict = KanaKanjiDictionary()
        let decoder = KanaKanjiDecoder(dictionary: dict, languageBonus: 0.0)

        // 略称から正式名称・ブランド表記への変換
        let resStarbucks = decoder.decode(kanaText: "すたば")
        XCTAssertEqual(resStarbucks, "スターバックス")

        let resMacdo = decoder.decode(kanaText: "まくど")
        XCTAssertEqual(resMacdo, "マクドナルド")

        let resMercari = decoder.decode(kanaText: "めすか")
        XCTAssertEqual(resMercari, "メルカリ")

        let resAirBnB = decoder.decode(kanaText: "えあびー")
        XCTAssertEqual(resAirBnB, "AirBnB")

        // 英語テック固有名詞への変換
        let resIPhone = decoder.decode(kanaText: "あいふぉーん")
        XCTAssertEqual(resIPhone, "iPhone")

        let resOpenAI = decoder.decode(kanaText: "おーぷんえーあい")
        XCTAssertEqual(resOpenAI, "OpenAI")

        let resTikTok = decoder.decode(kanaText: "てぃっくとっく")
        XCTAssertTrue(resTikTok == "TikTok" || resTikTok == "ティックトック")

        let resYouTube = decoder.decode(kanaText: "ゆーちゅーぶ")
        XCTAssertTrue(resYouTube == "YouTube" || resYouTube == "ユーチューブ")

        let resNetflix = decoder.decode(kanaText: "ねっとふりっくす")
        XCTAssertTrue(resNetflix == "Netflix" || resNetflix == "ネットフリックス")
    }

    // MARK: - 4. コーパス学習時におけるシード辞書の保持検証

    func testBuildFromCorpusRetainsSeedVocabulary() {
        let dict = KanaKanjiDictionary()
        let corpus = [
            "人工知能の技術が進歩している。",
            "今日は良い天気ですね。"
        ]
        dict.buildFromCorpus(rawTexts: corpus)

        // コーパス由来の語が引けること (形態素「人工」「天気」等)
        let jinkouHits = dict.lookupExact(reading: "じんこう")
        XCTAssertFalse(jinkouHits.isEmpty)
        XCTAssertEqual(jinkouHits.first?.surface, "人工")

        let tenkiHits = dict.lookupExact(reading: "てんき")
        XCTAssertFalse(tenkiHits.isEmpty)
        XCTAssertEqual(tenkiHits.first?.surface, "天気")

        // コーパス学習後もシード語彙が保持されていること
        let iphoneHits = dict.lookupExact(reading: "あいふぉーん")
        XCTAssertFalse(iphoneHits.isEmpty)
        XCTAssertEqual(iphoneHits.first?.surface, "iPhone")

        let starbucksHits = dict.lookupExact(reading: "すたば")
        XCTAssertFalse(starbucksHits.isEmpty)
        XCTAssertEqual(starbucksHits.first?.surface, "スターバックス")
    }

    // MARK: - 5. 音素調音ファジー検索によるシード辞書引き検証

    func testFuzzyPhoneticSearchWithSeedVocabulary() {
        let dict = KanaKanjiDictionary()

        // 濁音・清音のゆらぎや撥音の調音類似でシードエントリがヒットすること
        // "まくと" (「まくど」の清音ゆらぎ) -> "マクドナルド"
        let fuzzyMacdo = dict.lookupFuzzyPhonetic(reading: "まくと", maxPhoneticDist: 0.5)
        let foundMacdo = fuzzyMacdo.contains { $0.entry.surface == "マクドナルド" }
        XCTAssertTrue(foundMacdo)

        // "すたぱ" (「すたば」の半濁音ゆらぎ) -> "スターバックス"
        let fuzzyStarbucks = dict.lookupFuzzyPhonetic(reading: "すたぱ", maxPhoneticDist: 0.5)
        let foundStarbucks = fuzzyStarbucks.contains { $0.entry.surface == "スターバックス" }
        XCTAssertTrue(foundStarbucks)
    }

    // MARK: - 6. フェーズ2: デコーダ境界値・未知入力・空文字検証

    func testPhase2DecoderEdgeCases() {
        let dict = KanaKanjiDictionary()
        let decoder = KanaKanjiDecoder(dictionary: dict, languageBonus: 0.0)

        // 空文字デコード
        XCTAssertEqual(decoder.decode(kanaText: ""), "")
        XCTAssertTrue(decoder.decodeNBest(kanaText: "", beamWidth: 3).isEmpty)

        // 辞書に存在しない未知かな列の 1 文字スルー
        let unkResult = decoder.decode(kanaText: "んんん")
        XCTAssertEqual(unkResult, "んんん")

        // シード辞書を含まない空の辞書でのデコード
        let emptyDict = KanaKanjiDictionary(includeSeed: false)
        let emptyDecoder = KanaKanjiDecoder(dictionary: emptyDict, languageBonus: 0.0)
        let emptyRes = emptyDecoder.decode(kanaText: "あいうえお")
        XCTAssertEqual(emptyRes, "あいうえお")

        // SeedVocabulary 大文字・小文字・全角のフォールディング検索検証
        XCTAssertEqual(SeedVocabulary.readingForAlphabetSurface("iphone"), "あいふぉーん")
        XCTAssertEqual(SeedVocabulary.readingForAlphabetSurface("IPHONE"), "あいふぉーん")
        XCTAssertEqual(SeedVocabulary.readingForAlphabetSurface("ｉＰｈｏｎｅ"), "あいふぉーん")
        XCTAssertEqual(SeedVocabulary.readingForAlphabetSurface("mirrativ"), "みらてぃぶ")
        XCTAssertEqual(SeedVocabulary.readingForAlphabetSurface("MIRRATIV"), "みらてぃぶ")
        XCTAssertEqual(SeedVocabulary.readingForLetter("a"), "えー")
        XCTAssertEqual(SeedVocabulary.readingForLetter("Ａ"), "えー")
        XCTAssertNil(SeedVocabulary.readingForAlphabetSurface(""))
        XCTAssertNil(SeedVocabulary.readingForLetter(""))

        // 単体および連続する長音記号「ー」のフォールバック耐久性
        XCTAssertEqual(decoder.decode(kanaText: "ー"), "ー")
        XCTAssertEqual(decoder.decode(kanaText: "ーーーーー"), "ーーーーー")
        XCTAssertEqual(decoder.decode(kanaText: "あああーーーーー"), "あああーーーーー")

        // 固有名詞・和英混在ストリームのデコード
        let mixedStream = decoder.decode(kanaText: "えあびーでとまったあとめるかりでうった")
        XCTAssertTrue(mixedStream.contains("AirBnB"))
        XCTAssertTrue(mixedStream.contains("メルカリ"))

        // BtoB デコード
        XCTAssertEqual(decoder.decode(kanaText: "びーとぅーびー"), "BtoB")
    }
}
