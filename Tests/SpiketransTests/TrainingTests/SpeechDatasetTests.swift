import XCTest
@testable import Spiketrans

final class SpeechDatasetTests: XCTestCase {
    func testSpeechDatasetFromPCMPairs() {
        let corpus = ["こんにちは", "ありがとう"]
        let textVocab = TextVocabulary(corpus: corpus)

        // 16kHz, 16000サンプル (1秒分) の合成音声
        var dummyPCM = [Float](repeating: 0.0, count: 16000)
        var i = 0
        while i < 16000 {
            let t = Float(i) / 16000.0
            dummyPCM[i] = sin(2.0 * Float.pi * 440.0 * t) * 0.5
            i += 1
        }

        let pairs: [(pcmData: [Float], text: String)] = [
            (dummyPCM, "こんにちは"),
            (dummyPCM, "ありがとう")
        ]

        let dataset = SpeechDataset.fromPCMPairs(
            pairs: pairs,
            textVocabulary: textVocab
        )

        XCTAssertEqual(dataset.count, 2)
        XCTAssertEqual(dataset[0].rawText, "こんにちは")
        XCTAssertFalse(dataset[0].acousticFeatures.isEmpty)
        XCTAssertEqual(dataset[0].acousticFeatures[0].count, 128)
        XCTAssertFalse(dataset[0].textIds.isEmpty)
    }
}
