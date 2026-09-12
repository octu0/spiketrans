import XCTest
@testable import Spiketrans

final class QuantizationTests: XCTestCase {

    // MARK: - Float32 -> Int32/Int16 量子化正確性テスト

    func testQuantizationAccuracy() {
        let net = SpikingNetwork(inputDim: 32, maxHiddenDim: 256, outputDim: 64, timeSteps: 4)

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

    // MARK: - Float16

    func testFloat16QuantizationStoresHalfPrecision() {
        let net = SpikingNetwork(inputDim: 32, maxHiddenDim: 64, outputDim: 16, timeSteps: 4)
        let qWeights = QuantizedEngine.quantize(network: net, config: .float16Config())
        XCTAssertEqual(qWeights.config.format, QuantizedFormat.float16)
        XCTAssertEqual(qWeights.wIn.count, 0)
        XCTAssertEqual(qWeights.wInF16.count, net.pWIn.count)
        XCTAssertEqual(qWeights.wRecF16.count, net.pWRec.count)
        XCTAssertEqual(qWeights.bHF16.count, net.pBH.count)
        XCTAssertEqual(qWeights.wOutF16.count, net.pWOut.count)
        XCTAssertEqual(qWeights.bOutF16.count, net.pBOut.count)
        XCTAssertEqual(qWeights.config.beta, net.lifConfig.beta)
        XCTAssertEqual(qWeights.config.vTh, net.lifConfig.vTh)

        var i = 0
        while i < net.pBH.count {
            XCTAssertEqual(Float(qWeights.bHF16[i]), net.pBH.data[i], accuracy: 1e-3)
            i += 1
        }
    }

    func testFloat16PredictSoftmaxAndFinite() {
        let net = SpikingNetwork(inputDim: 32, maxHiddenDim: 64, outputDim: 16, timeSteps: 4)
        let engine = QuantizedEngine(
            weights: QuantizedEngine.quantize(network: net, config: .float16Config()),
            timeSteps: 4
        )
        let workspace = QuantizedWorkspace(maxHiddenDim: 64, inputDim: 32, outputDim: 16)
        var probs = [Float](repeating: 0.0, count: 16)
        engine.predict(
            features: [Float](repeating: 0.5, count: 32),
            workspace: workspace,
            outputProbs: &probs
        )
        var sumP: Float = 0.0
        var c = 0
        while c < 16 {
            XCTAssertFalse(probs[c].isNaN)
            XCTAssertFalse(probs[c].isInfinite)
            sumP += probs[c]
            c += 1
        }
        XCTAssertEqual(sumP, 1.0, accuracy: 1e-4)
    }

    /// 閾値下 0.4 は Float16 でも残り、飽和 20 は 1.0 に落ちる (生の 20 ではない)。
    func testFloat16ReadoutKeepsSubthresholdAndClipsSaturation() {
        XCTAssertEqual(Float(Float16(0.4)), 0.4, accuracy: 1e-3)
        let net = SpikingNetwork(
            numLayers: 1,
            inputDim: 1,
            maxHiddenDim: 2,
            outputDim: 1,
            timeSteps: 1,
            lifConfig: LIFConfig(beta: 0.8, vTh: 1.0, vReset: 0.0, alpha: 2.0)
        )
        var i = 0
        while i < net.pWIn.data.count {
            net.pWIn.data[i] = 0.0
            i += 1
        }
        i = 0
        while i < net.pWRec.data.count {
            net.pWRec.data[i] = 0.0
            i += 1
        }
        net.pBH.data[0] = 20.0
        net.pBH.data[1] = 0.4
        net.pWOut.data[0] = 1.0
        net.pWOut.data[1] = 1.0
        net.pBOut.data[0] = 0.0
        net.rebuildInferenceLayout()

        let engine = QuantizedEngine(
            weights: QuantizedEngine.quantize(network: net, config: .float16Config()),
            timeSteps: 1
        )
        let workspace = QuantizedWorkspace(maxHiddenDim: 2, inputDim: 1, outputDim: 1)
        var probs = [Float](repeating: 0.0, count: 1)
        engine.predict(features: [0.0], workspace: workspace, outputProbs: &probs)
        XCTAssertEqual(probs[0], 1.0, accuracy: 1e-5)

        XCTAssertEqual(workspace.logitsFloat[0], 1.4, accuracy: 1e-2)
        XCTAssertGreaterThan(abs(workspace.logitsFloat[0] - 1.0), 0.2)
        XCTAssertGreaterThan(abs(workspace.logitsFloat[0] - 20.4), 10.0)
    }

    func testFloat16Top1MatchesFloat32() {
        let net = SpikingNetwork(inputDim: 32, maxHiddenDim: 64, outputDim: 32, timeSteps: 4)
        let engine = QuantizedEngine(
            weights: QuantizedEngine.quantize(network: net, config: .float16Config()),
            timeSteps: 4
        )
        let workspace = QuantizedWorkspace(maxHiddenDim: 64, inputDim: 32, outputDim: 32)
        var floatProbs = [Float](repeating: 0.0, count: 32)
        var quantProbs = [Float](repeating: 0.0, count: 32)
        var spikeSum = [Float](repeating: 0.0, count: 64)
        var logits = [Float](repeating: 0.0, count: 32)
        var matches = 0
        var total = 0
        while total < 100 {
            var feat = [Float](repeating: 0.0, count: 32)
            var d = 0
            while d < 32 {
                feat[d] = Float((total * 29 + d * 13) % 100) / 100.0
                d += 1
            }
            var vp = [Float](repeating: 0.0, count: 64)
            var sp = [Float](repeating: 0.0, count: 64)
            net.forward(
                features: feat, vPrev: &vp, sPrev: &sp, readoutSum: &spikeSum,
                logits: &logits, probabilities: &floatProbs
            )
            engine.predict(features: feat, workspace: workspace, outputProbs: &quantProbs)
            var topF = 0
            var maxF: Float = -1.0
            var topQ = 0
            var maxQ: Float = -1.0
            var c = 0
            while c < 32 {
                if maxF < floatProbs[c] {
                    maxF = floatProbs[c]
                    topF = c
                }
                if maxQ < quantProbs[c] {
                    maxQ = quantProbs[c]
                    topQ = c
                }
                c += 1
            }
            if topF == topQ {
                matches += 1
            }
            total += 1
        }
        XCTAssertLessThanOrEqual(0.85, Float(matches) / Float(total))
    }

    func testFloat16HugeInputStaysFinite() {
        let net = SpikingNetwork(inputDim: 32, maxHiddenDim: 64, outputDim: 16, timeSteps: 4)
        let engine = QuantizedEngine(
            weights: QuantizedEngine.quantize(network: net, config: .float16Config()),
            timeSteps: 4
        )
        let workspace = QuantizedWorkspace(maxHiddenDim: 64, inputDim: 32, outputDim: 16)
        var probs = [Float](repeating: 0.0, count: 16)
        engine.predict(
            features: [Float](repeating: 100.0, count: 32),
            workspace: workspace,
            outputProbs: &probs
        )
        var sumP: Float = 0.0
        var c = 0
        while c < 16 {
            XCTAssertFalse(probs[c].isNaN)
            XCTAssertFalse(probs[c].isInfinite)
            sumP += probs[c]
            c += 1
        }
        XCTAssertEqual(sumP, 1.0, accuracy: 1e-4)
    }

    // MARK: - Top-1 予測一致度テスト

}
