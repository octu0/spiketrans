import XCTest
@testable import Spiketrans

final class TrainerTests: XCTestCase {

    func testParallelTrainingWithWorkers() {
        let corpus = ["あ", "い", "う", "え", "お"]
        let textVocab = TextVocabulary(corpus: corpus)

        var dummyPCM = [Float](repeating: 0.0, count: 1600)
        var i = 0
        while i < 1600 {
            dummyPCM[i] = sin(2.0 * Float.pi * 400.0 * Float(i) / 16000.0) * 0.2
            i += 1
        }

        let dataset = SpeechDataset.fromPCMPairs(
            pairs: [
                (dummyPCM, "あ"),
                (dummyPCM, "い"),
                (dummyPCM, "う"),
                (dummyPCM, "え")
            ],
            textVocabulary: textVocab
        )

        let trainer = Trainer.makeDefault(
            textVocabulary: textVocab,
            config: TrainingConfig(epochs: 2, learningRate: 0.01)
        )

        let summary = trainer.fit(dataset: dataset, numWorkers: 4)
        XCTAssertEqual(summary.acousticEpochs.count, 2)
        XCTAssertEqual(summary.languageEpochs.count, 2)
        XCTAssertFalse(summary.finalAcousticLoss.isNaN)
        XCTAssertFalse(summary.finalLanguageLoss.isNaN)
    }
}
