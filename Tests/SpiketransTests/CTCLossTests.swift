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

    /// フレーム単位に流し込んだ結果が、一括デコードと一致すること
    func testCTCStreamingMatchesBatch() {
        let vocab = TextVocabulary(characters: ["あ", "い", "う", "え", "お"])
        let decoder = CTCBeamDecoder(vocabulary: vocab, blankId: 0, beamWidth: 8)

        // 適当だが決定的な対数確率系列を作る
        var logProbs: [[Float]] = []
        var t = 0
        while t < 40 {
            var frame = [Float](repeating: -8.0, count: vocab.size)
            let peak = (t / 5) % vocab.size
            frame[peak] = -0.2
            frame[(peak + 1) % vocab.size] = -1.5
            if t % 3 == 0 {
                frame[0] = -0.5
            } else {
                frame[0] = -3.0
            }
            logProbs.append(frame)
            t += 1
        }

        let batch = decoder.decode(logProbs: logProbs)

        let streaming = decoder.makeStreamingDecoder()
        var i = 0
        while i < logProbs.count {
            streaming.push(frame: logProbs[i])
            i += 1
        }
        let incremental = streaming.best

        XCTAssertEqual(streaming.frameCount, logProbs.count)
        XCTAssertEqual(incremental.text, batch.text)
        XCTAssertEqual(incremental.tokens, batch.tokens)
        XCTAssertEqual(incremental.score, batch.score, accuracy: 1e-5)
    }

    /// 途中結果が伸びていくこと、reset で初期状態へ戻ること
    func testCTCStreamingPartialAndReset() {
        let vocab = TextVocabulary(characters: ["あ", "い", "う"])
        let decoder = CTCBeamDecoder(vocabulary: vocab, blankId: 0, beamWidth: 4)
        let streaming = decoder.makeStreamingDecoder()

        XCTAssertEqual(streaming.best.text, "")

        // 「あ」を強く出すフレームを積むと、途中結果に文字が現れる
        var frame = [Float](repeating: -9.0, count: vocab.size)
        frame[vocab.id(for: "あ")] = -0.05
        streaming.push(frame: frame)
        streaming.push(frame: frame)
        let partial = streaming.best.text
        XCTAssertFalse(partial.isEmpty)

        streaming.reset()
        XCTAssertEqual(streaming.frameCount, 0)
        XCTAssertEqual(streaming.best.text, "")
    }
}
