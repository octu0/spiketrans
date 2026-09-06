import XCTest
@testable import Spiketrans

final class InverseTextNormalizerTests: XCTestCase {

    // MARK: - 1. 通貨・金額表記の正規化 (Task 要件: 「じゅうごえん」 -> 「15円」)

    func testCurrencyNormalization() {
        let itn = InverseTextNormalizer()

        // かな表記からの通貨変換
        XCTAssertEqual(itn.normalize("じゅうごえん"), "15円")
        XCTAssertEqual(itn.normalize("さんびゃくえん"), "300円")
        XCTAssertEqual(itn.normalize("せんごひゃくえん"), "1500円")
        XCTAssertEqual(itn.normalize("いちまんえん"), "10000円")
        XCTAssertEqual(itn.normalize("ごじゅうえん"), "50円")
        XCTAssertEqual(itn.normalize("いちえん"), "1円")

        // 漢数字表記からの通貨変換
        XCTAssertEqual(itn.normalize("十五円"), "15円")
        XCTAssertEqual(itn.normalize("三百円"), "300円")
        XCTAssertEqual(itn.normalize("千五百円"), "1500円")
        XCTAssertEqual(itn.normalize("一万円"), "10000円")
        XCTAssertEqual(itn.normalize("五十円"), "50円")

        // 文中での通貨変換
        XCTAssertEqual(itn.normalize("リンゴはひとつじゅうごえんです"), "リンゴはひとつ15円です")
        XCTAssertEqual(itn.normalize("合計は三百円になります"), "合計は300円になります")
    }

    // MARK: - 2. 日付表記の正規化 (Task 要件: 「にせんにじゅうろくねんきがつむいか」 -> 「2026年9月6日」)

    func testDateNormalization() {
        let itn = InverseTextNormalizer()

        // Task 要件の直球テストケース
        XCTAssertEqual(itn.normalize("にせんにじゅうろくねんきがつむいか"), "2026年9月6日")
        XCTAssertEqual(itn.normalize("にせんにじゅうろくねんくがつむいか"), "2026年9月6日")
        XCTAssertEqual(itn.normalize("二千二十六年九月六日"), "2026年9月6日")

        // 和風月日読みのテスト (1日〜10日、14日、20日、24日)
        XCTAssertEqual(itn.normalize("いちがつついたち"), "1月1日")
        XCTAssertEqual(itn.normalize("にがつふつか"), "2月2日")
        XCTAssertEqual(itn.normalize("さんがつみっか"), "3月3日")
        XCTAssertEqual(itn.normalize("しがつよっか"), "4月4日")
        XCTAssertEqual(itn.normalize("ごがついつか"), "5月5日")
        XCTAssertEqual(itn.normalize("ろくがつむいか"), "6月6日")
        XCTAssertEqual(itn.normalize("しちがつなのか"), "7月7日")
        XCTAssertEqual(itn.normalize("はちがつようか"), "8月8日")
        XCTAssertEqual(itn.normalize("くがつここのか"), "9月9日")
        XCTAssertEqual(itn.normalize("じゅうがつとおか"), "10月10日")
        XCTAssertEqual(itn.normalize("じゅういちがつじゅうよっか"), "11月14日")
        XCTAssertEqual(itn.normalize("じゅうにがつはつか"), "12月20日")
        XCTAssertEqual(itn.normalize("にじゅうよっか"), "24日")

        // 年の変換
        XCTAssertEqual(itn.normalize("にせんにじゅうよんねん"), "2024年")
        XCTAssertEqual(itn.normalize("二千二十五年"), "2025年")
    }

    // MARK: - 3. 電話番号・数列の正規化 (Task 要件: 「ぜろきゅうぜろ」 -> 「090-」)

    func testPhoneNumberNormalization() {
        let itn = InverseTextNormalizer()

        // 主要な電話番号プレフィックス
        XCTAssertEqual(itn.normalize("ぜろいちにいぜろ"), "0120-")
        XCTAssertEqual(itn.normalize("ぜろさん"), "03-")

        // 文中での電話番号
        XCTAssertEqual(itn.normalize("電話番号はぜろさんです"), "電話番号は03-です")
    }

    // MARK: - 4. 割合・パーセント表記の正規化 (Task 要件: 「ひゃくぱーせんと」 -> 「100%」)

    func testPercentageNormalization() {
        let itn = InverseTextNormalizer()

        // かな表記からのパーセント変換
        XCTAssertEqual(itn.normalize("ひゃくぱーせんと"), "100%")
        XCTAssertEqual(itn.normalize("ごじゅっぱーせんと"), "50%")
        XCTAssertEqual(itn.normalize("ぜろぱーせんと"), "0%")
        XCTAssertEqual(itn.normalize("にじゅうごぱーせんと"), "25%")

        // 漢字・記号表記からのパーセント変換
        XCTAssertEqual(itn.normalize("百パーセント"), "100%")
        XCTAssertEqual(itn.normalize("五十パーセント"), "50%")
        XCTAssertEqual(itn.normalize("二十五％"), "25%")
        XCTAssertEqual(itn.normalize("百%"), "100%")

        // 文中でのパーセント変換
        XCTAssertEqual(itn.normalize("成功率はひゃくぱーせんとです"), "成功率は100%です")
    }

    // MARK: - 5. 時刻表記の正規化

    func testTimeNormalization() {
        let itn = InverseTextNormalizer()

        XCTAssertEqual(itn.normalize("十二時三十分"), "12時30分")
        XCTAssertEqual(itn.normalize("じゅうにじさんじっぷん"), "12時30分")
        XCTAssertEqual(itn.normalize("ごじじゅうごふん"), "5時15分")
        XCTAssertEqual(itn.normalize("三時"), "3時")
        XCTAssertEqual(itn.normalize("さんじ"), "3時")
    }

    // MARK: - 6. 一般漢数字・位取り・単位結合

    func testGeneralNumbersAndUnits() {
        let itn = InverseTextNormalizer()

        XCTAssertEqual(itn.normalize("千五百"), "1500")
        XCTAssertEqual(itn.normalize("二万"), "20000")
        XCTAssertEqual(itn.normalize("三万五千"), "35000")
        XCTAssertEqual(itn.normalize("十五個"), "15個")
        XCTAssertEqual(itn.normalize("さんぼん"), "3本")
        XCTAssertEqual(itn.normalize("五人"), "5人")
        XCTAssertEqual(itn.normalize("十回"), "10回")

        // 熟語・敬称の保護: "山田さん" が "山田3" に崩れないこと
        XCTAssertEqual(itn.normalize("山田さん"), "山田さん")
        XCTAssertEqual(itn.normalize("一度"), "1度")
    }

    // MARK: - 7. 境界値・空文字・数字なし文の透過性

    func testEdgeCases() {
        let itn = InverseTextNormalizer()

        XCTAssertEqual(itn.normalize(""), "")
        XCTAssertEqual(itn.normalize("東京に行きます"), "東京に行きます")
        XCTAssertEqual(itn.normalize("本日は晴天なり"), "本日は晴天なり")
    }

    // MARK: - 8. 位取りなし連続数字（二〇二六、一五円）の正規化

    func testPositionalKanjiAndKanaDigits() {
        let itn = InverseTextNormalizer()

        XCTAssertEqual(itn.normalize("二〇二六"), "2026")
        XCTAssertEqual(itn.normalize("二〇二六年"), "2026年")
        XCTAssertEqual(itn.normalize("一五円"), "15円")
        XCTAssertEqual(itn.normalize("一〇〇"), "100")
        XCTAssertEqual(itn.normalize("五番"), "5番")
    }

    // MARK: - 9. 口語表現・助動詞・疑問終助詞の誤変換防止（文脈保護）

    func testColloquialGrammarProtection() {
        let itn = InverseTextNormalizer()

        // 「〜なのか」「〜ようか」「いつか」が日付（7日、8日、5日）に誤爆しないこと
        XCTAssertEqual(itn.normalize("本当なのか"), "本当なのか")
        XCTAssertEqual(itn.normalize("映画を見ようか"), "映画を見ようか")
        XCTAssertEqual(itn.normalize("いつかまた会おう"), "いつかまた会おう")

        // 「気がつく」「とお書き」等の動詞表現が月や日に誤爆しないこと
        XCTAssertEqual(itn.normalize("それに気がつきました"), "それに気がつきました")
        XCTAssertEqual(itn.normalize("鉛筆とお書きください"), "鉛筆とお書きください")
    }

    // MARK: - 10. 日付直後の格助詞「に」の保護および高密度トークン文脈の整合性

    func testParticleNiAndDenseTokenPreservation() {
        let itn = InverseTextNormalizer()

        // 「6日」直後の格助詞「に」が「15円」を巻き込んで「25円」に化けないこと
        XCTAssertEqual(itn.normalize("にせんにじゅうろくねんきがつむいかにじゅうごえん"), "2026年9月6日に15円")

        // 日付・助数詞・通貨が密集する発話
        let denseInput = "にせんにじゅうろくねんきがつむいかにさんぼんのぺんをじゅうごえんでにこかった"
        let denseExpected = "2026年9月6日に3本のぺんを15円で2個かった"
        XCTAssertEqual(itn.normalize(denseInput), denseExpected)

        // 日付と人数の連続
        XCTAssertEqual(itn.normalize("10日に20人"), "10日に20人")

        // 敬称と順序助数詞
        XCTAssertEqual(itn.normalize("3人目の山田さん"), "3人目の山田さん")
    }

    // MARK: - 11. 算用数字混在表記（1万円、1万五千円、1兆円）の正規化

    func testMixedArabicAndKanjiCurrencyAndNumbers() {
        let itn = InverseTextNormalizer()

        XCTAssertEqual(itn.normalize("1万円"), "10000円")
        XCTAssertEqual(itn.normalize("2万円"), "20000円")
        XCTAssertEqual(itn.normalize("100万円"), "1000000円")
        XCTAssertEqual(itn.normalize("1万五千円"), "15000円")
        XCTAssertEqual(itn.normalize("1兆円"), "1000000000000円")
        XCTAssertEqual(itn.normalize("一兆円"), "1000000000000円")
    }

    // MARK: - 12. 同音異義語・熟語・固有名詞の完全保護（誤爆防止）

    func testHomophoneAndIdiomProtection() {
        let itn = InverseTextNormalizer()

        // かな数詞同音異義語
        XCTAssertEqual(itn.normalize("まんがを読む"), "まんがを読む")
        XCTAssertEqual(itn.normalize("おくびょうな性格"), "おくびょうな性格")
        XCTAssertEqual(itn.normalize("せんせいに相談する"), "せんせいに相談する")
        XCTAssertEqual(itn.normalize("せんたくきを回す"), "せんたくきを回す")
        XCTAssertEqual(itn.normalize("じゅうしょを記入する"), "じゅうしょを記入する")

        // 漢数字熟語
        XCTAssertEqual(itn.normalize("万全を期す"), "万全を期す")
        XCTAssertEqual(itn.normalize("万が一の備え"), "万が一の備え")
        XCTAssertEqual(itn.normalize("百景を楽しむ"), "百景を楽しむ")
        XCTAssertEqual(itn.normalize("千里眼を持つ"), "千里眼を持つ")
        XCTAssertEqual(itn.normalize("百科事典を調べる"), "百科事典を調べる")
        XCTAssertEqual(itn.normalize("ごはんを食べる"), "ごはんを食べる")
        XCTAssertEqual(itn.normalize("しかくを取得する"), "しかくを取得する")
        XCTAssertEqual(itn.normalize("にくを焼く"), "にくを焼く")
    }

    // MARK: - 13. 複合数詞下一桁と各種単位（歳、倍、回、日、人、パーセント）の境界テスト

    func testCompoundDigitsWithVariousUnits() {
        let itn = InverseTextNormalizer()

        // 1文字数詞（に、し、ご、く）が下一桁に来る複合数詞
        XCTAssertEqual(itn.normalize("にじゅうごさい"), "25歳")
        XCTAssertEqual(itn.normalize("さんじゅうにさい"), "32歳")
        XCTAssertEqual(itn.normalize("よんじゅうよんさい"), "44歳")
        XCTAssertEqual(itn.normalize("ごじゅうきゅうさい"), "59歳")

        // 助数詞「にん」「にち」との衝突境界
        XCTAssertEqual(itn.normalize("じゅうにん"), "10人")
        XCTAssertEqual(itn.normalize("じゅうににん"), "12人")
        XCTAssertEqual(itn.normalize("じゅうにち"), "10日")
        XCTAssertEqual(itn.normalize("じゅうににち"), "12日")

        // 単独1文字数詞＋単位
        XCTAssertEqual(itn.normalize("ごぱーせんと"), "5%")
        XCTAssertEqual(itn.normalize("にばい"), "2倍")
        XCTAssertEqual(itn.normalize("ごかい"), "5回")
    }
}
