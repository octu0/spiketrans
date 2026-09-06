import XCTest
@testable import Spiketrans

final class FillerWordFilterTests: XCTestCase {

    // MARK: - 1. 単独フィラーの検出・除去検証

    func testStandaloneFillerRemoval() {
        let filter = FillerWordFilter(mode: .remove)

        // 代表的なフィラー単体
        XCTAssertEqual(filter.filter("えー"), "")
        XCTAssertEqual(filter.filter("あー"), "")
        XCTAssertEqual(filter.filter("えっと"), "")
        XCTAssertEqual(filter.filter("あのー"), "")
        XCTAssertEqual(filter.filter("そのー"), "")
        XCTAssertEqual(filter.filter("まあ"), "")
        XCTAssertEqual(filter.filter("なんか"), "")
        XCTAssertEqual(filter.filter("うーん"), "")
        XCTAssertEqual(filter.filter("ええと"), "")
        XCTAssertEqual(filter.filter("えーと"), "")
        XCTAssertEqual(filter.filter("えーっと"), "")
        XCTAssertEqual(filter.filter("なんというか"), "")
    }

    // MARK: - 2. 長音・波ダッシュ引き伸ばし揺れの検証

    func testProlongedMarksVariations() {
        let filter = FillerWordFilter(mode: .remove)

        XCTAssertEqual(filter.filter("えーー"), "")
        XCTAssertEqual(filter.filter("えーーー"), "")
        XCTAssertEqual(filter.filter("あ〜"), "")
        XCTAssertEqual(filter.filter("あ〜〜"), "")
        XCTAssertEqual(filter.filter("う〜ん"), "")
        XCTAssertEqual(filter.filter("その〜"), "")
        XCTAssertEqual(filter.filter("あの〜"), "")
        XCTAssertEqual(filter.filter("まーっ"), "")
    }

    // MARK: - 3. 文頭・文中・文末フィラーの除去と読点クリーンアップ

    func testSententialFillerRemoval() {
        let filter = FillerWordFilter(mode: .remove)

        // 文頭フィラー
        XCTAssertEqual(filter.filter("えー本日はお日柄もよく"), "本日はお日柄もよく")
        XCTAssertEqual(filter.filter("えー、本日はお日柄もよく"), "本日はお日柄もよく")
        XCTAssertEqual(filter.filter("あーえっと、明日は雨です"), "明日は雨です")
        XCTAssertEqual(filter.filter("そのー、15円です"), "15円です")

        // 文中フィラー
        XCTAssertEqual(filter.filter("明日の天気は、えー、晴れです"), "明日の天気は晴れです")
        XCTAssertEqual(filter.filter("これは、うーん、難しい問題です"), "これは難しい問題です")

        // 複数フィラーの連続
        XCTAssertEqual(filter.filter("えー、あのー、そのー、会議を始めます"), "会議を始めます")

        // フィラーのみで構成された文
        XCTAssertEqual(filter.filter("えー、あのー、えっと"), "")
    }

    // MARK: - 4. 指示詞「あの」「その」の保護（誤消去防止）

    func testDemonstrativeProtection() {
        let filter = FillerWordFilter(mode: .remove)

        // 「あの人」「その本」はフィラーではなく指示詞（連体詞）なので消さない
        XCTAssertEqual(filter.filter("あの人を見かけました"), "あの人を見かけました")
        XCTAssertEqual(filter.filter("その本を読みました"), "その本を読みました")
        XCTAssertEqual(filter.filter("あの店に行きましょう"), "あの店に行きましょう")

        // 直後に読点やポーズがある「あの、」「その、」はフィラーとして除去されること
        XCTAssertEqual(filter.filter("あの、ちょっといいですか"), "ちょっといいですか")
        XCTAssertEqual(filter.filter("その、実はですね"), "実はですね")
        XCTAssertEqual(filter.filter("あのー、あの人は誰ですか"), "あの人は誰ですか")
    }

    // MARK: - 5. モード切替の検証

    func testFilterModes() {
        let removeFilter = FillerWordFilter(mode: .remove)
        let markFilter = FillerWordFilter(mode: .mark)
        let disabledFilter = FillerWordFilter(mode: .disabled)

        let text = "えー、明日は晴れです"

        XCTAssertEqual(removeFilter.filter(text), "明日は晴れです")
        XCTAssertEqual(markFilter.filter(text), "(えー)明日は晴れです")
        XCTAssertEqual(disabledFilter.filter(text), "えー、明日は晴れです")
    }

    // MARK: - 6. 境界値・空文字・通常文の完全透過性

    func testEdgeCases() {
        let filter = FillerWordFilter(mode: .remove)

        XCTAssertEqual(filter.filter(""), "")
        XCTAssertEqual(filter.filter("、"), "")
        XCTAssertEqual(filter.filter("   "), "")
        XCTAssertEqual(filter.filter("こんにちは、世界！"), "こんにちは、世界！")
        XCTAssertEqual(filter.filter("AIエンジンの高速化を達成しました"), "AIエンジンの高速化を達成しました")
    }

    // MARK: - 7. 外来語・一般名詞・不定代名詞の保護（誤消去防止）

    func testLoanwordAndGrammarProtection() {
        let filter = FillerWordFilter(mode: .remove)

        // カタカナ借用語（ひらがな表記）の語頭音保護
        XCTAssertEqual(filter.filter("あーとを鑑賞する"), "あーとを鑑賞する")
        XCTAssertEqual(filter.filter("まーけっとで買い物"), "まーけっとで買い物")
        XCTAssertEqual(filter.filter("えーすを狙う"), "えーすを狙う")

        // 「まあまあ」「何かが」等の誤消去防止
        XCTAssertEqual(filter.filter("まあまあです"), "まあまあです")
        XCTAssertEqual(filter.filter("なんかが起こった"), "なんかが起こった")
        XCTAssertEqual(filter.filter("何かありましたか"), "何かありましたか")

        // 漢字直前のフィラーは確実に除去されること
        XCTAssertEqual(filter.filter("あー本日は晴天です"), "本日は晴天です")
        XCTAssertEqual(filter.filter("えー田中です"), "田中です")
        XCTAssertEqual(filter.filter("まあ良いでしょう"), "良いでしょう")
    }

    // MARK: - 8. 括弧境界・連続フィラー・単独フィラー句点クリーンアップ

    func testBracketAndConsecutiveFillers() {
        let filter = FillerWordFilter(mode: .remove)

        // 括弧内での指示詞フィラー
        XCTAssertEqual(filter.filter("「あの、ちょっといいですか」"), "「ちょっといいですか」")
        XCTAssertEqual(filter.filter("「あの」"), "「」")

        // 読点なしの連続フィラー
        XCTAssertEqual(filter.filter("あのえーっと、本日は晴天です"), "本日は晴天です")
        XCTAssertEqual(filter.filter("そのうーん、難しいです"), "難しいです")

        // フィラー直後の句点サニタイズ
        XCTAssertEqual(filter.filter("えー。"), "")
        XCTAssertEqual(filter.filter("えー。本日は晴天です"), "本日は晴天です")
    }
}
