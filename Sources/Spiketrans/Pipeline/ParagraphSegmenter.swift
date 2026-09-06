import Foundation

/// 段落データモデル (開始〜終了タイムスタンプ付き)
public struct ParagraphSegment: Sendable, Equatable {
    public let startTimeSeconds: Float
    public let endTimeSeconds: Float
    public let text: String

    public init(startTimeSeconds: Float, endTimeSeconds: Float, text: String) {
        self.startTimeSeconds = startTimeSeconds
        self.endTimeSeconds = endTimeSeconds
        self.text = text
    }

    /// タイムスタンプフォーマット: [00:01.20 -> 00:05.80]
    public var timestampFormatted: String {
        ParagraphSegmenter.formatTimestampRange(start: startTimeSeconds, end: endTimeSeconds)
    }

    /// タイムスタンプ付き段落テキスト (例: "[00:01.20 -> 00:05.80] 本日はお日柄もよく...")
    public var formattedLine: String {
        if text.isEmpty {
            return timestampFormatted
        }
        return "\(timestampFormatted) \(text)"
    }
}

/// 沈黙時間および文区切りに基づく改行段落分け＆タイムスタンプ処理エンジン
///
/// 一定以上の無音・沈黙時間（ポーズ閾値: 既定 1.2 秒）や文の区切りによって発話を段落に分割し、
/// 段落ごとに開始〜終了時刻のタイムスタンプ（例: `[00:01.20 -> 00:05.80]`）を出力する。
/// O(1) メモリ管理により、ストリーミング推論でもメモリ肥大化を起こさない。
public final class ParagraphSegmenter: @unchecked Sendable {
    public let pauseThresholdSeconds: Float

    private var currentStart: Float = 0.0
    private var currentEnd: Float = 0.0
    private var currentTextParts: [String] = []
    private var hasActiveParagraph: Bool = false

    public init(pauseThresholdSeconds: Float = 1.2) {
        self.pauseThresholdSeconds = pauseThresholdSeconds
    }

    /// 秒数を mm:ss.xx (または hh:mm:ss.xx) に整形
    public static func formatTimestamp(seconds: Float) -> String {
        var sec = seconds
        if sec < 0.0 {
            sec = 0.0
        }
        let totalCentis = Int((sec * 100.0).rounded())
        let centis = totalCentis % 100
        let totalSecs = totalCentis / 100
        let s = totalSecs % 60
        let totalMins = totalSecs / 60
        let m = totalMins % 60
        let h = totalMins / 60

        if 0 < h {
            return String(format: "%02d:%02d:%02d.%02d", h, m, s, centis)
        }
        return String(format: "%02d:%02d.%02d", m, s, centis)
    }

    /// 開始〜終了時刻を [00:01.20 -> 00:05.80] 形式に整形
    public static func formatTimestampRange(start: Float, end: Float) -> String {
        let sStr = formatTimestamp(seconds: start)
        let eStr = formatTimestamp(seconds: end)
        return "[\(sStr) -> \(eStr)]"
    }

    /// 文末区切り文字か判定
    private static func isSentenceDelimiter(_ ch: Character) -> Bool {
        switch ch {
        case "。", "！", "？", ".", "!", "?":
            return true
        default:
            return false
        }
    }

    /// 新しい発話セグメントを追加し、段落の分割が発生した場合は完了した段落を返す
    public func appendSegment(text: String, startSeconds: Float, endSeconds: Float) -> ParagraphSegment? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty {
            return nil
        }

        var completedParagraph: ParagraphSegment? = nil

        if hasActiveParagraph {
            // 直前の発話終了から今回の発話開始までの無音時間 (ポーズ)
            let pauseDuration = startSeconds - currentEnd

            var shouldSplit = false
            if pauseThresholdSeconds <= pauseDuration {
                // ポーズ閾値を超えている場合は無音による段落分割
                shouldSplit = true
            } else {
                // 直前の段落テキスト末尾が句点で、かつ一定ポーズ (0.8秒以上) がある場合も分割
                if let lastPart = currentTextParts.last, let lastChar = lastPart.last {
                    if Self.isSentenceDelimiter(lastChar) && 0.8 <= pauseDuration {
                        shouldSplit = true
                    }
                }
            }

            if shouldSplit {
                completedParagraph = buildCurrentParagraph()
                currentStart = startSeconds
                currentEnd = endSeconds
                currentTextParts = [trimmed]
                hasActiveParagraph = true
                return completedParagraph
            }

            // 同一段落内に結合
            currentEnd = endSeconds
            currentTextParts.append(trimmed)
        } else {
            // 初回段落の開始
            currentStart = startSeconds
            currentEnd = endSeconds
            currentTextParts = [trimmed]
            hasActiveParagraph = true
        }

        return nil
    }

    /// 現在蓄積されている段落を ParagraphSegment として確定構築
    private func buildCurrentParagraph() -> ParagraphSegment {
        let combinedText = currentTextParts.joined()
        return ParagraphSegment(
            startTimeSeconds: currentStart,
            endTimeSeconds: currentEnd,
            text: combinedText
        )
    }

    /// 残存している現在の段落を確定して出力
    public func flush() -> ParagraphSegment? {
        if hasActiveParagraph != true || currentTextParts.isEmpty {
            return nil
        }
        let paragraph = buildCurrentParagraph()
        reset()
        return paragraph
    }

    /// 内部状態の全リセット
    public func reset() {
        currentStart = 0.0
        currentEnd = 0.0
        currentTextParts.removeAll(keepingCapacity: true)
        hasActiveParagraph = false
    }

    /// 文字起こし結果シーケンスを段落に一括分割するバッチヘルパー
    public static func segment(
        results: [TranscriptionResult],
        pauseThresholdSeconds: Float = 1.2
    ) -> [ParagraphSegment] {
        let segmenter = ParagraphSegmenter(pauseThresholdSeconds: pauseThresholdSeconds)
        var paragraphs: [ParagraphSegment] = []

        var i = 0
        while i < results.count {
            let res = results[i]
            if let completed = segmenter.appendSegment(
                text: res.text,
                startSeconds: res.startTimeSeconds,
                endSeconds: res.endTimeSeconds
            ) {
                paragraphs.append(completed)
            }
            i += 1
        }

        if let finalParagraph = segmenter.flush() {
            paragraphs.append(finalParagraph)
        }

        return paragraphs
    }
}
