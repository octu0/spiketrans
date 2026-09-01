import Foundation
import XCTest
@testable import Spiketrans

final class CTCLossTests: XCTestCase {

    /// CTC 損失計算の数値安定性と非負性の検証
    func testCTCLossComputationAndGradients() {
        let calc = CTCLossCalculator(blankId: 0)
        let vCount = 10
        let tCount = 20

        // 一様な対数確率系列
        let logP = -log(Float(vCount))
        let logProbs = [[Float]](repeating: [Float](repeating: logP, count: vCount), count: tCount)
        let targets = [1, 2, 3, 4]

        let result = calc.computeLossAndGradients(logProbs: logProbs, targets: targets)
        XCTAssertFalse(result.loss.isNaN, "CTC Loss should not be NaN")
        XCTAssertFalse(result.loss.isInfinite, "CTC Loss should not be infinite")
        XCTAssertLessThan(0.0, result.loss, "CTC Loss should be strictly positive")
        XCTAssertEqual(result.gradients.count, tCount)
        XCTAssertEqual(result.gradients[0].count, vCount)

        // 勾配の総和がフレームごとにゼロに極めて近いことを検証 (Softmax 勾配特性)
        for t in 0..<tCount {
            var sumGrad: Float = 0.0
            for g in result.gradients[t] {
                sumGrad += g
            }
            XCTAssertLessThanOrEqual(abs(sumGrad), 1e-4, "Softmax gradient sum must be close to 0")
        }
    }

    /// CTC Prefix Beam Search デコーダの基本動作検証
    func testCTCBeamDecoderGreedyAndBeamSearch() {
        let vocab = TextVocabulary(corpus: ["あいうえお", "かきくけこ"])
        let decoder = CTCBeamDecoder(vocabulary: vocab, blankId: 0, beamWidth: 8)

        let tCount = 10
        let vCount = vocab.size

        // ターゲット "あ"(ID: 4) をフレーム 2..4 で強く発火させる対数確率
        var logProbs = [[Float]](repeating: [Float](repeating: -10.0, count: vCount), count: tCount)
        for t in 0..<tCount {
            logProbs[t][0] = 0.0 // Blank
        }
        logProbs[2][4] = 5.0
        logProbs[3][4] = 5.0
        logProbs[4][4] = 5.0

        // Softmax 正規化
        for t in 0..<tCount {
            var maxL: Float = -.infinity
            for v in 0..<vCount {
                if maxL < logProbs[t][v] { maxL = logProbs[t][v] }
            }
            var sumExp: Float = 0.0
            for v in 0..<vCount {
                sumExp += exp(logProbs[t][v] - maxL)
            }
            let lse = maxL + log(sumExp)
            for v in 0..<vCount {
                logProbs[t][v] -= lse
            }
        }

        let greedyRes = decoder.decodeGreedy(logProbs: logProbs)
        XCTAssertEqual(greedyRes.tokens, [4], "Greedy CTC should collapse repeated tokens")

        let beamRes = decoder.decode(logProbs: logProbs)
        XCTAssertEqual(beamRes.tokens, [4], "Beam search CTC should decode identical token")
    }
}
