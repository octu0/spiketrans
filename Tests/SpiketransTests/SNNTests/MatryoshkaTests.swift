import XCTest
@testable import Spiketrans

final class MatryoshkaTests: XCTestCase {

    // MARK: - Base/Middle/High 各スライス独立推論テスト

    func testSliceInference() {
        let net = MatryoshkaNetwork(
            inputDim: 32,
            maxHiddenDim: 1024,
            outputDim: 64,
            timeSteps: 4
        )

        var features = [Float](repeating: 0.0, count: 32)
        var i = 0
        while i < 32 {
            features[i] = Float(i) * 0.03
            i += 1
        }

        let slices: [MatryoshkaSlice] = [.base, .middle, .high]
        var sIdx = 0
        while sIdx < slices.count {
            let slice = slices[sIdx]
            let hSize = slice.rawValue

            var vPrev = [Float](repeating: 0.0, count: hSize)
            var sPrev = [Float](repeating: 0.0, count: hSize)
            var spikeSum = [Float](repeating: 0.0, count: hSize)
            var logits = [Float](repeating: 0.0, count: 64)
            var probs = [Float](repeating: 0.0, count: 64)

            net.forwardSlice(
                features: features,
                slice: slice,
                vPrev: &vPrev,
                sPrev: &sPrev,
                spikeSum: &spikeSum,
                logits: &logits,
                probabilities: &probs
            )

            // 確率の総和が 1.0 であること
            var sumProb: Float = 0.0
            var c = 0
            while c < 64 {
                XCTAssertLessThanOrEqual(0.0, probs[c])
                XCTAssertLessThanOrEqual(probs[c], 1.0)
                sumProb += probs[c]
                c += 1
            }
            XCTAssertEqual(sumProb, 1.0, accuracy: 1e-5, "Probability sum for slice \(slice) must be 1.0")

            sIdx += 1
        }
    }

    // MARK: - 階層重み共有の整合性テスト

    func testWeightSharingIntegrity() {
        let net = MatryoshkaNetwork(
            inputDim: 32,
            maxHiddenDim: 1024,
            outputDim: 64,
            timeSteps: 4
        )

        let features = [Float](repeating: 0.5, count: 32)
        let hBase = MatryoshkaSlice.base.rawValue

        var vPrev1 = [Float](repeating: 0.0, count: hBase)
        var sPrev1 = [Float](repeating: 0.0, count: hBase)
        var spikeSum1 = [Float](repeating: 0.0, count: hBase)
        var logits1 = [Float](repeating: 0.0, count: 64)
        var probs1 = [Float](repeating: 0.0, count: 64)

        net.forwardSlice(
            features: features,
            slice: .base,
            vPrev: &vPrev1,
            sPrev: &sPrev1,
            spikeSum: &spikeSum1,
            logits: &logits1,
            probabilities: &probs1
        )

        // Base スライス範囲外 (ニューロン 128〜1023) の重みを改変
        var n = 128
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

        net.forwardSlice(
            features: features,
            slice: .base,
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

    func testBaseExportImport() throws {
        let originalNet = MatryoshkaNetwork(
            inputDim: 32,
            maxHiddenDim: 1024,
            outputDim: 64,
            timeSteps: 4
        )

        // 1. エクスポート
        let baseWeights = originalNet.exportBaseWeights()
        XCTAssertEqual(baseWeights.inputDim, 32)
        XCTAssertEqual(baseWeights.hiddenDim, 128)
        XCTAssertEqual(baseWeights.outputDim, 64)
        XCTAssertEqual(baseWeights.wIn.count, 128 * 32)
        XCTAssertEqual(baseWeights.wRec.count, 128 * 128)
        XCTAssertEqual(baseWeights.bH.count, 128)
        XCTAssertEqual(baseWeights.wOut.count, 64 * 128)
        XCTAssertEqual(baseWeights.bOut.count, 64)

        // 2. JSON シリアライズ / デシリアライズ検証
        let encoder = JSONEncoder()
        let data = try encoder.encode(baseWeights)
        let decoder = JSONDecoder()
        let decodedWeights = try decoder.decode(BaseSNNWeights.self, from: data)
        XCTAssertEqual(baseWeights, decodedWeights)

        // 3. 別モデルへのインポート
        let targetNet = MatryoshkaNetwork(
            inputDim: 32,
            maxHiddenDim: 1024,
            outputDim: 64,
            timeSteps: 4
        )
        targetNet.importBaseWeights(decodedWeights)

        // 4. 推論結果の一致検証
        var features = [Float](repeating: 0.0, count: 32)
        var i = 0
        while i < 32 {
            features[i] = sin(Float(i) * 0.2) * 0.5 + 0.5
            i += 1
        }

        let hBase = MatryoshkaSlice.base.rawValue
        var vPrevOrig = [Float](repeating: 0.0, count: hBase)
        var sPrevOrig = [Float](repeating: 0.0, count: hBase)
        var spikeSumOrig = [Float](repeating: 0.0, count: hBase)
        var logitsOrig = [Float](repeating: 0.0, count: 64)
        var probsOrig = [Float](repeating: 0.0, count: 64)

        originalNet.forwardSlice(
            features: features,
            slice: .base,
            vPrev: &vPrevOrig,
            sPrev: &sPrevOrig,
            spikeSum: &spikeSumOrig,
            logits: &logitsOrig,
            probabilities: &probsOrig
        )

        var vPrevTgt = [Float](repeating: 0.0, count: hBase)
        var sPrevTgt = [Float](repeating: 0.0, count: hBase)
        var spikeSumTgt = [Float](repeating: 0.0, count: hBase)
        var logitsTgt = [Float](repeating: 0.0, count: 64)
        var probsTgt = [Float](repeating: 0.0, count: 64)

        targetNet.forwardSlice(
            features: features,
            slice: .base,
            vPrev: &vPrevTgt,
            sPrev: &sPrevTgt,
            spikeSum: &spikeSumTgt,
            logits: &logitsTgt,
            probabilities: &probsTgt
        )

        var c = 0
        while c < 64 {
            XCTAssertEqual(logitsOrig[c], logitsTgt[c], accuracy: 1e-6, "Imported base logit mismatch at \(c)")
            XCTAssertEqual(probsOrig[c], probsTgt[c], accuracy: 1e-6, "Imported base prob mismatch at \(c)")
            c += 1
        }
    }

    // MARK: - ゼロアロケーション推論連続実行テスト

    func testZeroAllocationInference() {
        let net = MatryoshkaNetwork(inputDim: 32, maxHiddenDim: 1024, outputDim: 64, timeSteps: 4)
        let hSize = MatryoshkaSlice.high.rawValue

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

            net.forwardSlice(
                features: features,
                slice: .high,
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
