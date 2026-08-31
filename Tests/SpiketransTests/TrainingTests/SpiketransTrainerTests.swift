import XCTest
@testable import Spiketrans

final class SpiketransTrainerTests: XCTestCase {
    func testAcousticAndLanguageTrainerStep() {
        let corpus = ["おはよう", "ありがとう"]
        let textVocab = TextVocabulary(corpus: corpus)

        var dummyPCM = [Float](repeating: 0.0, count: 3200) // 0.2秒
        var i = 0
        while i < 3200 {
            let t = Float(i) / 16000.0
            dummyPCM[i] = sin(2.0 * Float.pi * 300.0 * t) * 0.3
            i += 1
        }

        let dataset = SpeechDataset.fromPCMPairs(
            pairs: [(dummyPCM, "おはよう")],
            textVocabulary: textVocab
        )

        let trainer = SpiketransTrainer.makeDefault(
            textVocabulary: textVocab,
            config: TrainingConfig(epochs: 2, learningRate: 0.01)
        )

        let summary = trainer.fit(dataset: dataset)
        XCTAssertEqual(summary.acousticEpochs.count, 2)
        XCTAssertEqual(summary.languageEpochs.count, 2)
        XCTAssertFalse(summary.finalAcousticLoss.isNaN)
        XCTAssertFalse(summary.finalLanguageLoss.isNaN)

        // Base スライス抽出の検証
        let baseAcWeights = trainer.exportBaseAcousticWeights()
        XCTAssertEqual(baseAcWeights.hiddenDim, 128)

        let baseLmWeights = trainer.exportBaseLanguageWeights()
        XCTAssertEqual(baseLmWeights.hiddenDim, 128)

        // 直接文字起こし推論の検証
        let transcribed = trainer.transcribe(pcmData: dummyPCM, slice: .base)
        XCTAssertNotNil(transcribed)
    }

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

        let trainer = SpiketransTrainer.makeDefault(
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
