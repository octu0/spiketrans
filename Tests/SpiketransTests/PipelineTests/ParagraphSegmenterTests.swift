import XCTest
@testable import Spiketrans

final class ParagraphSegmenterTests: XCTestCase {

    // MARK: - 1. タイムスタンプフォーマットの検証 (Task 要件: [00:01.20 -> 00:05.80])

    func testTimestampFormatting() {
        // 単独秒数フォーマット
        XCTAssertEqual(ParagraphSegmenter.formatTimestamp(seconds: 1.20), "00:01.20")
        XCTAssertEqual(ParagraphSegmenter.formatTimestamp(seconds: 5.80), "00:05.80")
        XCTAssertEqual(ParagraphSegmenter.formatTimestamp(seconds: 65.45), "01:05.45")
        XCTAssertEqual(ParagraphSegmenter.formatTimestamp(seconds: 3661.05), "01:01:01.05")

        // 範囲フォーマット [00:01.20 -> 00:05.80]
        let rangeStr = ParagraphSegmenter.formatTimestampRange(start: 1.20, end: 5.80)
        XCTAssertEqual(rangeStr, "[00:01.20 -> 00:05.80]")

        // 負数やゼロの保護
        XCTAssertEqual(ParagraphSegmenter.formatTimestamp(seconds: -2.0), "00:00.00")
        XCTAssertEqual(ParagraphSegmenter.formatTimestamp(seconds: 0.0), "00:00.00")
    }

    // MARK: - 2. ポーズ時間（沈黙）による段落自動分割

    func testPauseBasedSegmentation() {
        let segmenter = ParagraphSegmenter(pauseThresholdSeconds: 1.2)

        // 第1セグメント [0.0 -> 2.0]: 初回段落の開始 (完了段落は nil)
        let p1 = segmenter.appendSegment(text: "本日は晴天です", startSeconds: 0.0, endSeconds: 2.0)
        XCTAssertNil(p1)

        // 第2セグメント [2.5 -> 4.0]: ポーズ 0.5 秒 (1.2 秒未満) -> 同一段落内に結合
        let p2 = segmenter.appendSegment(text: "気温も過ごしやすいです", startSeconds: 2.5, endSeconds: 4.0)
        XCTAssertNil(p2)

        // 第3セグメント [6.0 -> 8.5]: ポーズ 2.0 秒 (1.2 秒以上) -> 前の段落が分割確定！
        let completed = segmenter.appendSegment(text: "次の話題に移ります", startSeconds: 6.0, endSeconds: 8.5)
        XCTAssertNotNil(completed)

        if let comp = completed {
            XCTAssertEqual(comp.startTimeSeconds, 0.0)
            XCTAssertEqual(comp.endTimeSeconds, 4.0)
            XCTAssertEqual(comp.text, "本日は晴天です気温も過ごしやすいです")
            XCTAssertEqual(comp.timestampFormatted, "[00:00.00 -> 00:04.00]")
            XCTAssertEqual(comp.formattedLine, "[00:00.00 -> 00:04.00] 本日は晴天です気温も過ごしやすいです")
        }

        // 残存段落の flush
        let finalP = segmenter.flush()
        XCTAssertNotNil(finalP)
        if let fp = finalP {
            XCTAssertEqual(fp.startTimeSeconds, 6.0)
            XCTAssertEqual(fp.endTimeSeconds, 8.5)
            XCTAssertEqual(fp.text, "次の話題に移ります")
            XCTAssertEqual(fp.timestampFormatted, "[00:06.00 -> 00:08.50]")
        }
    }

    // MARK: - 3. 句点（。）とポーズによる文区切り分割

    func testSentenceDelimiterSegmentation() {
        let segmenter = ParagraphSegmenter(pauseThresholdSeconds: 1.5)

        // 句点付き発話 [1.0 -> 3.0]
        let p1 = segmenter.appendSegment(text: "よろしくお願いします。", startSeconds: 1.0, endSeconds: 3.0)
        XCTAssertNil(p1)

        // ポーズ 0.9 秒 (0.8 秒以上かつ句点後) -> 分割が発生
        let completed = segmenter.appendSegment(text: "それでは始めます。", startSeconds: 3.9, endSeconds: 5.5)
        XCTAssertNotNil(completed)

        if let comp = completed {
            XCTAssertEqual(comp.text, "よろしくお願いします。")
            XCTAssertEqual(comp.timestampFormatted, "[00:01.00 -> 00:03.00]")
        }
    }

    // MARK: - 4. バッチ一括分割ヘルパーの検証

    func testBatchSegmentation() {
        let r1 = TranscriptionResult(
            text: "こんにちは。",
            phonemes: [],
            tokenIds: [],
            startTimeSeconds: 1.20,
            endTimeSeconds: 3.50,
            confidence: 0.9,
            isFinal: true
        )
        let r2 = TranscriptionResult(
            text: "Spiketransの紹介です。",
            phonemes: [],
            tokenIds: [],
            startTimeSeconds: 3.80,
            endTimeSeconds: 5.80,
            confidence: 0.9,
            isFinal: true
        )
        let r3 = TranscriptionResult(
            text: "第2段落の内容です。",
            phonemes: [],
            tokenIds: [],
            startTimeSeconds: 8.00,
            endTimeSeconds: 10.50,
            confidence: 0.9,
            isFinal: true
        )

        let paragraphs = ParagraphSegmenter.segment(results: [r1, r2, r3], pauseThresholdSeconds: 1.2)
        XCTAssertEqual(paragraphs.count, 2)

        XCTAssertEqual(paragraphs[0].timestampFormatted, "[00:01.20 -> 00:05.80]")
        XCTAssertEqual(paragraphs[0].text, "こんにちは。Spiketransの紹介です。")
        XCTAssertEqual(paragraphs[0].formattedLine, "[00:01.20 -> 00:05.80] こんにちは。Spiketransの紹介です。")

        XCTAssertEqual(paragraphs[1].timestampFormatted, "[00:08.00 -> 00:10.50]")
        XCTAssertEqual(paragraphs[1].text, "第2段落の内容です。")
    }

    // MARK: - 5. 空文字・リセットの検証

    func testEmptyAndReset() {
        let segmenter = ParagraphSegmenter()

        XCTAssertNil(segmenter.appendSegment(text: "", startSeconds: 0.0, endSeconds: 1.0))
        XCTAssertNil(segmenter.appendSegment(text: "   ", startSeconds: 0.0, endSeconds: 1.0))
        XCTAssertNil(segmenter.flush())

        let p1 = segmenter.appendSegment(text: "テスト", startSeconds: 0.0, endSeconds: 1.0)
        XCTAssertNil(p1)
        segmenter.reset()
        XCTAssertNil(segmenter.flush())
    }
}
