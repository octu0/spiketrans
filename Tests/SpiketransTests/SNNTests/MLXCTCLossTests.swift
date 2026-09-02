import XCTest
import MLX
@testable import Spiketrans

final class MLXCTCLossTests: XCTestCase {
    /// MLX 演算で組んだ CTC 損失が Pure Swift 実装と一致すること
    func testMatchesPureSwiftImplementation() {
        let blankId = 0
        let vocab = 6
        let cases: [(frames: Int, labels: [Int])] = [
            (8, [1, 2, 3]),
            (12, [2, 2, 3]),      // 同一ラベル連続 (スキップ遷移が禁止される)
            (6, [4]),
            (10, [1, 2, 1, 2]),
        ]

        for (frames, labels) in cases {
            // 決定論的なロジットを生成
            var flat = [Float](repeating: 0.0, count: frames * vocab)
            var t = 0
            while t < frames {
                var v = 0
                while v < vocab {
                    flat[(t * vocab) + v] = Float(((t * 7) + (v * 13)) % 11) * 0.21 - 1.0
                    v += 1
                }
                t += 1
            }

            // Pure Swift 実装
            var logProbs = [[Float]](repeating: [Float](repeating: 0.0, count: vocab), count: frames)
            t = 0
            while t < frames {
                var maxLogit = -Float.greatestFiniteMagnitude
                var v = 0
                while v < vocab {
                    let x = flat[(t * vocab) + v]
                    if maxLogit < x { maxLogit = x }
                    v += 1
                }
                var sumExp: Float = 0.0
                v = 0
                while v < vocab {
                    sumExp += exp(flat[(t * vocab) + v] - maxLogit)
                    v += 1
                }
                let logSumExp = maxLogit + log(sumExp)
                v = 0
                while v < vocab {
                    logProbs[t][v] = flat[(t * vocab) + v] - logSumExp
                    v += 1
                }
                t += 1
            }
            let expected = CTCLossCalculator(blankId: blankId)
                .computeLossAndGradients(logProbs: logProbs, targets: labels).loss

            // MLX 実装
            let logits = MLXArray(flat, [1, frames, vocab])
            let ext = MLXCTCLoss.ExtendedTargets(
                targetsBatch: [labels],
                frameCounts: [frames],
                blankId: blankId
            )
            let actual = MLXCTCLoss.loss(logits: logits, targets: ext).item(Float.self)

            XCTAssertEqual(actual, expected, accuracy: 1e-3,
                           "frames=\(frames) labels=\(labels) で不一致")
        }
    }

    /// バッチ内で長さが異なるサンプルでも各サンプル独立に計算されること
    func testBatchMatchesIndividualLosses() {
        let blankId = 0
        let vocab = 5
        let maxFrames = 10
        let samples: [(frames: Int, labels: [Int])] = [(10, [1, 2]), (6, [3, 4, 1])]

        var flat = [Float](repeating: 0.0, count: samples.count * maxFrames * vocab)
        var b = 0
        while b < samples.count {
            var t = 0
            while t < maxFrames {
                var v = 0
                while v < vocab {
                    flat[((b * maxFrames + t) * vocab) + v] = Float(((b * 5) + (t * 3) + v) % 7) * 0.3 - 1.0
                    v += 1
                }
                t += 1
            }
            b += 1
        }

        let logits = MLXArray(flat, [samples.count, maxFrames, vocab])
        let batchExt = MLXCTCLoss.ExtendedTargets(
            targetsBatch: samples.map { $0.labels },
            frameCounts: samples.map { $0.frames },
            blankId: blankId
        )
        let batchLoss = MLXCTCLoss.loss(logits: logits, targets: batchExt).item(Float.self)

        // 個別に計算した損失の平均と一致すること
        var sum: Float = 0.0
        b = 0
        while b < samples.count {
            let single = logits[b..<(b + 1), 0..., 0...]
            let ext = MLXCTCLoss.ExtendedTargets(
                targetsBatch: [samples[b].labels],
                frameCounts: [samples[b].frames],
                blankId: blankId
            )
            sum += MLXCTCLoss.loss(logits: single, targets: ext).item(Float.self)
            b += 1
        }
        XCTAssertEqual(batchLoss, sum / Float(samples.count), accuracy: 1e-3)
    }
}
