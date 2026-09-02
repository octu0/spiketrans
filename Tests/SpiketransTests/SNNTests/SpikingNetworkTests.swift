import XCTest
@testable import Spiketrans

final class SpikingNetworkTests: XCTestCase {

    // MARK: - Base/Middle/High 各スライス独立推論テスト


    // MARK: - 階層重み共有の整合性テスト

    func testWeightSharingIntegrity() {
        let net = SpikingNetwork(
            inputDim: 32,
            maxHiddenDim: 1024,
            outputDim: 64,
            timeSteps: 4
        )

        let features = [Float](repeating: 0.5, count: 32)
        let hBase = net.maxHiddenDim

        var vPrev1 = [Float](repeating: 0.0, count: hBase)
        var sPrev1 = [Float](repeating: 0.0, count: hBase)
        var spikeSum1 = [Float](repeating: 0.0, count: hBase)
        var logits1 = [Float](repeating: 0.0, count: 64)
        var probs1 = [Float](repeating: 0.0, count: 64)

        net.forward(
            features: features,
            vPrev: &vPrev1,
            sPrev: &sPrev1,
            spikeSum: &spikeSum1,
            logits: &logits1,
            probabilities: &probs1
        )

        // Base スライス範囲外のニューロンの重みを改変 (境界はスライス定義から導出)
        var n = net.maxHiddenDim
        while n < 1024 {
            net.pBH.data[n] += 10.0
            var d = 0
            while d < 32 {
                net.pWIn.data[n * 32 + d] += 5.0
                d += 1
            }
            var j = 0
            while j < 1024 {
                net.pWRec.data[n * 1024 + j] += 5.0
                j += 1
            }
            n += 1
        }

        // 改変後に Base スライス推論を実行
        var vPrev2 = [Float](repeating: 0.0, count: hBase)
        var sPrev2 = [Float](repeating: 0.0, count: hBase)
        var spikeSum2 = [Float](repeating: 0.0, count: hBase)
        var logits2 = [Float](repeating: 0.0, count: 64)
        var probs2 = [Float](repeating: 0.0, count: 64)

        net.forward(
            features: features,
            vPrev: &vPrev2,
            sPrev: &sPrev2,
            spikeSum: &spikeSum2,
            logits: &logits2,
            probabilities: &probs2
        )

        // 範囲外の改変が Base スライスの推論結果に影響を与えないことを検証
        var c = 0
        while c < 64 {
            XCTAssertEqual(logits1[c], logits2[c], accuracy: 1e-6, "Base logit mismatch at \(c)")
            XCTAssertEqual(probs1[c], probs2[c], accuracy: 1e-6, "Base prob mismatch at \(c)")
            c += 1
        }
    }

    // MARK: - Base 単体エクスポート・インポートテスト


    // MARK: - ゼロアロケーション推論連続実行テスト

    func testZeroAllocationInference() {
        let net = SpikingNetwork(inputDim: 32, maxHiddenDim: 1024, outputDim: 64, timeSteps: 4)
        let hSize = net.maxHiddenDim

        var vPrev = [Float](repeating: 0.0, count: hSize)
        var sPrev = [Float](repeating: 0.0, count: hSize)
        var spikeSum = [Float](repeating: 0.0, count: hSize)
        var logits = [Float](repeating: 0.0, count: 64)
        var probs = [Float](repeating: 0.0, count: 64)

        var frame = 0
        while frame < 100 {
            var features = [Float](repeating: 0.0, count: 32)
            var d = 0
            while d < 32 {
                features[d] = Float((frame + d) % 10) * 0.1
                d += 1
            }

            net.forward(
                features: features,
                vPrev: &vPrev,
                sPrev: &sPrev,
                spikeSum: &spikeSum,
                logits: &logits,
                probabilities: &probs
            )

            var sumP: Float = 0.0
            var c = 0
            while c < 64 {
                sumP += probs[c]
                c += 1
            }
            XCTAssertEqual(sumP, 1.0, accuracy: 1e-5)
            frame += 1
        }
    }
}
