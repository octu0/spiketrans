import XCTest
@testable import Spiketrans

final class QuantizationTests: XCTestCase {

    // MARK: - Float32 -> Int32/Int16 量子化正確性テスト

    func testQuantizationAccuracy() {
        let net = MatryoshkaNetwork(inputDim: 32, maxHiddenDim: 256, outputDim: 64, timeSteps: 4)

        // 1. Int32 量子化
        let qConfig32 = QuantizedConfig.int32Config()
        let qWeights32 = QuantizedEngine.quantize(network: net, config: qConfig32)

        XCTAssertEqual(qWeights32.inputDim, 32)
        XCTAssertEqual(qWeights32.maxHiddenDim, 256)
        XCTAssertEqual(qWeights32.outputDim, 64)
        XCTAssertEqual(qWeights32.wIn.count, 256 * 32)
        XCTAssertEqual(qWeights32.wRec.count, 256 * 256)
        XCTAssertEqual(qWeights32.bH.count, 256)
        XCTAssertEqual(qWeights32.wOut.count, 64 * 256)
        XCTAssertEqual(qWeights32.bOut.count, 64)

        // 2. Int16 量子化
        let qConfig16 = QuantizedConfig.int16Config()
        let qWeights16 = QuantizedEngine.quantize(network: net, config: qConfig16)

        XCTAssertEqual(qWeights16.inputDim, 32)
        XCTAssertEqual(qWeights16.maxHiddenDim, 256)
        XCTAssertEqual(qWeights16.outputDim, 64)

        // スケール値の検証
        var i = 0
        while i < 256 {
            let floatBH = net.pBH.data[i]
            let int32BH = qWeights32.bH[i]
            let expected32 = Int32(round(floatBH * 65536.0))
            XCTAssertEqual(int32BH, expected32)

            let int16BH = qWeights16.bH[i]
            let expected16 = Int32(round(floatBH * 2048.0))
            XCTAssertEqual(int16BH, expected16)
            i += 1
        }
    }

    // MARK: - ビットシフト減衰の精度検証テスト

    func testBitShiftDecayPrecision() {
        let beta: Float = 0.8

        // 1. Int32 精度検証 (Scale: 65536, DecayNum: 52429, DecayBits: 16)
        let scale32: Float = 65536.0
        let decayNum32: Int64 = 52429
        let decayBits32: Int64 = 16

        // 2. Int16 精度検証 (Scale: 2048, DecayNum: 3277, DecayBits: 12)
        let scale16: Float = 2048.0
        let decayNum16: Int64 = 3277
        let decayBits16: Int64 = 12

        var testV: Float = 0.1
        while testV <= 1.0 {
            let trueDecay = testV * beta

            // Int32 シフト減衰
            let vInt32 = Int64(round(testV * scale32))
            let decayedInt32 = (vInt32 * decayNum32) >> decayBits32
            let floatDecayed32 = Float(decayedInt32) / scale32
            let err32 = abs(floatDecayed32 - trueDecay) / trueDecay
            XCTAssertLessThan(err32, 0.0001, "Int32 decay error too high for V=\(testV)")

            // Int16 シフト減衰
            let vInt16 = Int64(round(testV * scale16))
            let decayedInt16 = (vInt16 * decayNum16) >> decayBits16
            let floatDecayed16 = Float(decayedInt16) / scale16
            let err16 = abs(floatDecayed16 - trueDecay) / trueDecay
            XCTAssertLessThan(err16, 0.005, "Int16 decay error too high for V=\(testV)")

            testV += 0.1
        }
    }

    // MARK: - スパースリカレント加算の等価性テスト

    func testSparseRecurrentAddition() {
        let hSize = 64
        var wRec = [Int32](repeating: 0, count: hSize * hSize)
        var sPrev = [Int32](repeating: 0, count: hSize)

        var i = 0
        while i < hSize {
            var j = 0
            while j < hSize {
                wRec[i * hSize + j] = Int32((i + j) % 17) - 8
                j += 1
            }
            if i % 4 == 0 {
                sPrev[i] = 1
            }
            i += 1
        }

        // スパースリカレント加算 (乗算器フリー)
        var sparseResult = [Int32](repeating: 0, count: hSize)
        i = 0
        while i < hSize {
            var current: Int32 = 0
            let recOffset = i * hSize
            var j = 0
            while j < hSize {
                if sPrev[j] != 0 {
                    current += wRec[recOffset + j]
                }
                j += 1
            }
            sparseResult[i] = current
            i += 1
        }

        // 密行列ベクトル積 (基準値)
        var denseResult = [Int32](repeating: 0, count: hSize)
        i = 0
        while i < hSize {
            var current: Int32 = 0
            let recOffset = i * hSize
            var j = 0
            while j < hSize {
                current += wRec[recOffset + j] * sPrev[j]
                j += 1
            }
            denseResult[i] = current
            i += 1
        }

        i = 0
        while i < hSize {
            XCTAssertEqual(sparseResult[i], denseResult[i], "Sparse recurrent sum mismatch at index \(i)")
            i += 1
        }
    }

    // MARK: - Top-1 予測一致度テスト

    func testQuantizedInferenceTop1Match() {
        let net = MatryoshkaNetwork(inputDim: 32, maxHiddenDim: 256, outputDim: 64, timeSteps: 4)
        let qConfig32 = QuantizedConfig.int32Config()
        let qWeights32 = QuantizedEngine.quantize(network: net, config: qConfig32)
        let engine32 = QuantizedEngine(weights: qWeights32, timeSteps: 4)
        let workspace32 = QuantizedWorkspace(maxHiddenDim: 256, inputDim: 32, outputDim: 64)

        var features = [Float](repeating: 0.0, count: 32)
        var d = 0
        while d < 32 {
            features[d] = sin(Float(d) * 0.4) * 0.4 + 0.5
            d += 1
        }

        let slices: [MatryoshkaSlice] = [.base, .middle, .high]
        var sIdx = 0
        while sIdx < slices.count {
            let slice = slices[sIdx]
            let hSize = slice.rawValue

            // 1. Float32 推論
            var vPrev = [Float](repeating: 0.0, count: hSize)
            var sPrev = [Float](repeating: 0.0, count: hSize)
            var spikeSum = [Float](repeating: 0.0, count: hSize)
            var logitsFloat = [Float](repeating: 0.0, count: 64)
            var probsFloat = [Float](repeating: 0.0, count: 64)

            net.forwardSlice(
                features: features,
                slice: slice,
                vPrev: &vPrev,
                sPrev: &sPrev,
                spikeSum: &spikeSum,
                logits: &logitsFloat,
                probabilities: &probsFloat
            )

            // 2. Int32 量子化推論
            var probsInt32 = [Float](repeating: 0.0, count: 64)
            engine32.predictSlice(
                features: features,
                slice: slice,
                workspace: workspace32,
                outputProbs: &probsInt32
            )

            // 確率総和検証
            var sumP: Float = 0.0
            var c = 0
            while c < 64 {
                sumP += probsInt32[c]
                c += 1
            }
            XCTAssertEqual(sumP, 1.0, accuracy: 1e-4)

            // Top-1 インデックスの一致判定
            var maxFloat: Float = -1.0
            var top1Float = -1
            var maxInt32: Float = -1.0
            var top1Int32 = -1

            c = 0
            while c < 64 {
                if maxFloat < probsFloat[c] {
                    maxFloat = probsFloat[c]
                    top1Float = c
                }
                if maxInt32 < probsInt32[c] {
                    maxInt32 = probsInt32[c]
                    top1Int32 = c
                }
                c += 1
            }

            XCTAssertEqual(top1Float, top1Int32, "Top-1 prediction must match between Float32 and Int32 for slice \(slice)")
            sIdx += 1
        }
    }
}
