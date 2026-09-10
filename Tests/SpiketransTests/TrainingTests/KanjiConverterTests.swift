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
        XCTAssertEqual(converter.convertToHiragana("１４７３年"), "せんよんひゃくななじゅーさんねん")
        XCTAssertEqual(converter.convertToHiragana("２０億円"), "にじゅーおくえん")
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

    func testPronunciationNormalization() {
        let converter = KanjiConverter()

        // 同じ母音の連なりは長音に寄せる (表記ではなく音に対応させる)
        XCTAssertEqual(converter.convertToHiragana("東京"), "とーきょー")
        XCTAssertEqual(converter.convertToHiragana("先生"), "せんせー")
        XCTAssertEqual(converter.convertToHiragana("映画"), "えーが")
        XCTAssertEqual(converter.convertToHiragana("大きい"), "おーきー")
        XCTAssertEqual(converter.convertToHiragana("おじいさん"), "おじーさん")
        XCTAssertEqual(converter.convertToHiragana("おばあさん"), "おばーさん")
        XCTAssertEqual(converter.convertToHiragana("氷"), "こーり")
        XCTAssertEqual(converter.convertToHiragana("空気"), "くーき")
        // 動詞語尾の「う」は母音として残す
        XCTAssertEqual(converter.convertToHiragana("思う"), "おもう")
        XCTAssertEqual(converter.convertToHiragana("買う"), "かう")
        // 助詞と旧仮名
        XCTAssertEqual(converter.convertToHiragana("これは水を飲む"), "これわみずおのむ")
        XCTAssertEqual(converter.convertToHiragana("東京へ行く"), "とーきょーえいく")
        XCTAssertEqual(converter.convertToHiragana("続く"), "つずく")
        // 語彙にも「を」「ぢ」「づ」は現れない
        XCTAssertFalse(converter.convertToHiragana("鼻血を出す").contains("ぢ"))
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
        XCTAssertEqual(mixed, "あいふぉーんとまっくぶっくおつかってさぎょーする")
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
        XCTAssertTrue(hira.contains("じゅーご"))
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

    func testReadingOverrides() {
        let converter = KanjiConverter()
        // 形態素解析器の読みを話し言葉に揃える
        XCTAssertEqual(converter.convertToHiragana("私は皆と日本へ行く"), "わたしわみんなとにほんえいく")
        XCTAssertEqual(converter.convertToHiragana("皆さん、明日と一昨日"), "みなさんあしたとおととい")
        XCTAssertEqual(converter.convertToHiragana("そういうことだと言う"), "そーゆーことだとゆー")
        XCTAssertEqual(converter.convertToHiragana("言いました"), "いーました")
        // 複合語の併合
        XCTAssertEqual(converter.convertToHiragana("お母さんとお父さんとお兄さんとお姉さん"), "おかーさんとおとーさんとおにーさんとおねーさん")
        XCTAssertEqual(converter.convertToHiragana("木曜日と日曜日"), "もくよーびとにちよーび")
        XCTAssertEqual(converter.convertToHiragana("世界中"), "せかいじゅー")
        XCTAssertEqual(converter.tokenize("お母さん").count, 1)
        // 国名 + 人は「じん」、数詞 + 人は「にん」
        XCTAssertEqual(converter.convertToHiragana("日本人とフランス人と外国人"), "にほんじんとふらんすじんとがいこくじん")
        XCTAssertEqual(converter.convertToHiragana("3人と何人"), "さんにんとなんにん")
        // 何: 助詞・動詞が続けば「なに」、助数詞や「で」は「なん」
        XCTAssertEqual(converter.convertToHiragana("何が何を何してる"), "なにがなにおなにしてる")
        XCTAssertEqual(converter.convertToHiragana("何で何回"), "なんでなんかい")
    }

    func testCounterSandhi() {
        let converter = KanjiConverter()
        // 促音化と半濁音化 (算用数字も漢数字も同じ)
        XCTAssertEqual(converter.convertToHiragana("1本10本100回1週間"), "いっぽんじゅっぽんひゃっかいいっしゅーかん")
        XCTAssertEqual(converter.convertToHiragana("一杯六杯八匹十分"), "いっぱいろっぱいはっぴきじゅっぷん")
        XCTAssertEqual(converter.convertToHiragana("1か月1キロ10ページ"), "いっかげついっきろじゅっぺーじ")
        // ん の後は濁音 (本・杯・匹) と半濁音 (分・歩・発)
        XCTAssertEqual(converter.convertToHiragana("3本三杯何本千本3分三歩"), "さんぼんさんばいなんぼんせんぼんさんぷんさんぽ")
        // 四・七・九・二・五は変化しない (解析器が「四本/よんぽん」と返す癖も直す)
        XCTAssertEqual(converter.convertToHiragana("4本7本9本2分4分"), "よんほんななほんきゅーほんにふんよんぷん")
        // 六・百はか行・は行だけ促音化
        XCTAssertEqual(converter.convertToHiragana("六冊百冊六回"), "ろくさつひゃくさつろっかい")
        // 外来語の単位の は行 は促音化しない
        XCTAssertEqual(converter.convertToHiragana("1ヘクタール1パーセント"), "いちへくたーるいっぱーせんと")
        // 数によって読みが変わる助数詞
        XCTAssertEqual(converter.convertToHiragana("4月4日4時"), "しがつよっかよじ")
        XCTAssertEqual(converter.convertToHiragana("9月10日7時"), "くがつとーかしちじ")
        XCTAssertEqual(converter.convertToHiragana("11日20日24日"), "じゅーいちにちはつかにじゅーよっか")
        XCTAssertEqual(converter.convertToHiragana("1人2人4人3人"), "ひとりふたりよにんさんにん")
        XCTAssertEqual(converter.convertToHiragana("1つ3つ四つ八つ"), "ひとつみっつよっつやっつ")
        // 数詞ではない「位置」は促音化しない
        XCTAssertEqual(converter.convertToHiragana("位置確認"), "いちかくにん")
    }

    func testTranscriptionMarkersAreSilent() {
        let converter = KanjiConverter()
        // 「(笑)」「(爆笑)」「(冷笑)」は笑いの注記で発音されない
        XCTAssertEqual(converter.convertToHiragana("(笑)ん?撮ってない?"), "んとってない")
        XCTAssertEqual(converter.convertToHiragana("やられた(爆笑)駆除した（冷笑）ね"), "やられたくじょしたね")
        // 笑を含まない括弧書きは残る
        XCTAssertEqual(converter.convertToHiragana("彼(兄)が来た"), "かれあにがきた")
        // 単独の w は読み上げで「だぶりゅー」と発音されるので残す
        XCTAssertEqual(converter.convertToHiragana("それなw"), "それなだぶりゅー")
        XCTAssertEqual(converter.convertToHiragana("Wi-Fiとweb"), "わいふぁいとうぇぶ")
    }
}
