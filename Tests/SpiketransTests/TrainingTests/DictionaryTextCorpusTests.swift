import XCTest
@testable import Spiketrans

final class DictionaryTextCorpusTests: XCTestCase {
    private var tempDir = ""

    override func setUp() {
        super.setUp()
        tempDir = (NSTemporaryDirectory() as NSString).appendingPathComponent("spiketrans_dicttext_\(UUID().uuidString)")
        try? FileManager.default.createDirectory(atPath: (tempDir as NSString).appendingPathComponent("sub"), withIntermediateDirectories: true)
    }

    override func tearDown() {
        try? FileManager.default.removeItem(atPath: tempDir)
        super.tearDown()
    }

    private func write(_ text: String, name: String) -> String {
        let path = (tempDir as NSString).appendingPathComponent(name)
        try? text.write(toFile: path, atomically: true, encoding: .utf8)
        return path
    }

    func testPlainTextLinesSkipBlankLines() {
        let path = write("一つ、二つ。\n\n  三つ  \n", name: "a.txt")
        XCTAssertEqual(DictionaryTextCorpus.load(path: path), ["一つ、二つ。", "三つ"])
    }

    func testSRTKeepsOnlyCueBodiesAndHandlesCRLF() {
        let srt = "\u{FEFF}1\r\n00:00:01,000 --> 00:00:02,000\r\n通知は全部オフにして\r\n二行目\r\n\r\n2\r\n00:00:02,000 --> 00:00:03,000\r\nこのまま深く潜っていこう\r\n"
        let path = write(srt, name: "b.srt")
        XCTAssertEqual(DictionaryTextCorpus.load(path: path), ["通知は全部オフにして 二行目", "このまま深く潜っていこう"])
    }

    func testDirectoryCollectsTextAndSRTRecursively() {
        _ = write("甲板\n", name: "z.txt")
        _ = write("1\n00:00:00,000 --> 00:00:01,000\n銀杏\n", name: "sub/a.srt")
        _ = write("無視される", name: "sub/c.md")
        XCTAssertEqual(DictionaryTextCorpus.load(path: tempDir), ["銀杏", "甲板"])
    }

    func testMissingPathIsEmpty() {
        XCTAssertEqual(DictionaryTextCorpus.load(path: (tempDir as NSString).appendingPathComponent("nothing.txt")), [])
    }
}
