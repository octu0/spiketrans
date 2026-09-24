import Foundation

/// 第2段のかな漢字辞書だけに使う、音声の無いテキストの読み込み。
///
/// 辞書は学習セットの本文から作るが、`KanaKanjiDictionary.buildFromCorpus` は生テキストを受け取るだけなので、
/// 音声の付いていないテキスト (配信 SRT の本文、ドメインの語彙が多い文章) でも太らせられる。
/// 受け付けるのは UTF-8 の `.txt` (1 行 1 文) と `.srt` (キュー本文だけを取る)、およびそれらを含むディレクトリ (再帰)。
public enum DictionaryTextCorpus {
    /// パス (ファイルまたはディレクトリ) からテキスト行を集める。空行は捨てる。読めないファイルは飛ばす
    public static func load(path: String) -> [String] {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory) else {
            return []
        }
        if isDirectory.boolValue {
            return loadDirectory(path)
        }
        return loadFile(path)
    }

    static func loadDirectory(_ dir: String) -> [String] {
        guard let enumerator = FileManager.default.enumerator(atPath: dir) else {
            return []
        }
        var files: [String] = []
        while let relative = enumerator.nextObject() as? String {
            let lower = relative.lowercased()
            if lower.hasSuffix(".txt") || lower.hasSuffix(".srt") {
                files.append((dir as NSString).appendingPathComponent(relative))
            }
        }
        files.sort()
        var lines: [String] = []
        for file in files {
            lines.append(contentsOf: loadFile(file))
        }
        return lines
    }

    static func loadFile(_ path: String) -> [String] {
        guard let raw = try? String(contentsOfFile: path, encoding: .utf8) else {
            return []
        }
        let content = normalizedNewlines(raw)
        if path.lowercased().hasSuffix(".srt") {
            return srtCueTexts(content)
        }
        var lines: [String] = []
        for line in content.split(separator: "\n", omittingEmptySubsequences: true) {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty != true {
                lines.append(trimmed)
            }
        }
        return lines
    }

    /// CRLF (Windows 改行) と BOM を揃える
    static func normalizedNewlines(_ text: String) -> String {
        return text
            .replacingOccurrences(of: "\u{FEFF}", with: "")
            .replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "\r", with: "\n")
    }

    /// SRT のキュー本文だけを 1 キュー 1 行で返す (番号行と時刻行は捨てる。複数行の本文は空白で結ぶ)
    static func srtCueTexts(_ content: String) -> [String] {
        var cues: [String] = []
        var body: [String] = []
        var inCue = false
        func closeCue() {
            if inCue && body.isEmpty != true {
                cues.append(body.joined(separator: " "))
            }
            inCue = false
            body = []
        }
        for rawLine in content.split(separator: "\n", omittingEmptySubsequences: false) {
            let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
            if line.isEmpty {
                closeCue()
                continue
            }
            if line.contains("-->") {
                closeCue()
                inCue = true
                continue
            }
            if inCue {
                body.append(line)
            }
        }
        closeCue()
        return cues
    }
}
