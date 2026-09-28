import XCTest
@testable import Spiketrans

/// Milestone M2 (SNN Core & SIMD/Quantization Engine) 敵対的検証・極限ストレステストスイート
final class M2ChallengerStressTests: XCTestCase {

    // MARK: - 1. LIF ニューロン極限入力・数値安定性テスト (LIF Dynamics Extremes)

    /// 極大入力電流 (I = 1e6, 1e12) に対する有限値挙動、発火判定、次ステップリセットの検証
    func testLIFNeuronExtremePositiveInputCurrent() {
        let config = LIFConfig(beta: 0.8, vTh: 1.0, vReset: 0.0, alpha: 2.0)
        let state = LIFState(size: 8)

        // 1. 極大入力 (I = 1e6) 注入
        let vPrev: Float = 0.0
        let sPrev: Float = 0.0
        let step1 = LIFNeuronEngine.stepScalar(config: config, vPrev: vPrev, sPrev: sPrev, inputCurrent: 1e6)
        // 膜電位は学習側 (MLX) と同じ範囲に飽和する
        XCTAssertEqual(step1.vNext, LIFNeuronEngine.vClampMax, accuracy: 1e-1)
        XCTAssertEqual(step1.sNext, 1.0)
        XCTAssertFalse(step1.vNext.isNaN)
        XCTAssertFalse(step1.vNext.isInfinite)

        // 2. 次ステップ: sPrev = 1.0 により減衰項 beta * vPrev * (1 - sPrev) が厳密に 0.0 になること
        let step2 = LIFNeuronEngine.stepScalar(config: config, vPrev: step1.vNext, sPrev: step1.sNext, inputCurrent: 0.5)
        XCTAssertEqual(step2.vNext, 0.5, accuracy: 1e-6)
        XCTAssertEqual(step2.sNext, 0.0)

        // 3. 超極大入力 (I = 1e12)
        let stepHuge = LIFNeuronEngine.stepScalar(config: config, vPrev: 0.0, sPrev: 0.0, inputCurrent: 1e12)
        XCTAssertEqual(stepHuge.sNext, 1.0)
        let stepHugePost = LIFNeuronEngine.stepScalar(config: config, vPrev: stepHuge.vNext, sPrev: stepHuge.sNext, inputCurrent: 0.0)
        XCTAssertEqual(stepHugePost.vNext, 0.0, accuracy: 1e-6)
        XCTAssertEqual(stepHugePost.sNext, 0.0)

        // 4. SIMD8 での極大入力一括処理
        let inputSIMD = [Float](repeating: 1e6, count: 8)
        var vOutSIMD = [Float](repeating: 0.0, count: 8)
        var sOutSIMD = [Float](repeating: 0.0, count: 8)

        state.v.withUnsafeBufferPointer { vPtr in
            state.s.withUnsafeBufferPointer { sPtr in
                inputSIMD.withUnsafeBufferPointer { iPtr in
                    vOutSIMD.withUnsafeMutableBufferPointer { voPtr in
                        sOutSIMD.withUnsafeMutableBufferPointer { soPtr in
                            LIFNeuronEngine.stepSIMD8(
                                config: config,
                                vPrevPtr: vPtr.baseAddress!,
                                sPrevPtr: sPtr.baseAddress!,
                                inputPtr: iPtr.baseAddress!,
                                vNextPtr: voPtr.baseAddress!,
                                sNextPtr: soPtr.baseAddress!,
                                count: 8
                            )
                        }
                    }
                }
            }
        }

        var i = 0
        while i < 8 {
            XCTAssertEqual(vOutSIMD[i], LIFNeuronEngine.vClampMax, accuracy: 1e-1)
            XCTAssertEqual(sOutSIMD[i], 1.0)
            i += 1
        }
    }

    /// 負の極大入力電流 (I = -1e6, -100.0) に対する非発火および有限値減衰テスト
    func testLIFNeuronExtremeNegativeInputCurrent() {
        let config = LIFConfig(beta: 0.8, vTh: 1.0, vReset: 0.0, alpha: 2.0)

        // 負の極大電流注入 -> 発火しないこと (sNext = 0.0)
        let stepNeg = LIFNeuronEngine.stepScalar(config: config, vPrev: 0.0, sPrev: 0.0, inputCurrent: -1e6)
        XCTAssertEqual(stepNeg.vNext, LIFNeuronEngine.vClampMin, accuracy: 1e-1)
        XCTAssertEqual(stepNeg.sNext, 0.0)

        // 次ステップで負の電位が beta (0.8) 倍で減衰すること
        let stepNegDecay = LIFNeuronEngine.stepScalar(config: config, vPrev: stepNeg.vNext, sPrev: stepNeg.sNext, inputCurrent: 0.0)
        // 飽和値 -20.0 から beta (0.8) 倍で減衰する
        XCTAssertEqual(stepNegDecay.vNext, LIFNeuronEngine.vClampMin * 0.8, accuracy: 1e-1)
        XCTAssertEqual(stepNegDecay.sNext, 0.0)

        // 負電位からの復帰: 十分に大きな正電流を注入した際の発火動作
        let stepRecover = LIFNeuronEngine.stepScalar(config: config, vPrev: -10.0, sPrev: 0.0, inputCurrent: 15.0)
        // vNext = 0.8 * (-10.0) + 15.0 = -8.0 + 15.0 = 7.0 >= 1.0 -> 発火
        XCTAssertEqual(stepRecover.vNext, 7.0, accuracy: 1e-5)
        XCTAssertEqual(stepRecover.sNext, 1.0)
    }

    /// 全ゼロ入力電流での指数関数的減衰とゼロ発火の確認
    func testLIFNeuronZeroCurrentDecayToSilence() {
        let config = LIFConfig(beta: 0.5, vTh: 1.0, vReset: 0.0, alpha: 2.0)
        var v: Float = 0.99 // 閾値直下
        var s: Float = 0.0

        var step = 0
        while step < 20 {
            let res = LIFNeuronEngine.stepScalar(config: config, vPrev: v, sPrev: s, inputCurrent: 0.0)
            XCTAssertEqual(res.sNext, 0.0, "Zero input should never cause spike")
            XCTAssertLessThan(res.vNext, v, "Membrane potential must strictly decrease")
            v = res.vNext
            s = res.sNext
            step += 1
        }
        XCTAssertLessThan(v, 1e-6, "Potential must converge to near-zero")
    }

    /// 連続スパイク発火 (All Spike Burst) 時の安定性
    func testLIFNeuronContinuousSpikeBurst() {
        let config = LIFConfig(beta: 0.8, vTh: 1.0, vReset: 0.0, alpha: 2.0)
        var v: Float = 0.0
        var s: Float = 0.0

        var step = 0
        while step < 50 {
            // 毎ステップ閾値超過の電流 (1.5) を注入
            let res = LIFNeuronEngine.stepScalar(config: config, vPrev: v, sPrev: s, inputCurrent: 1.5)
            XCTAssertEqual(res.sNext, 1.0, "Must spike on step \(step)")
            XCTAssertEqual(res.vNext, 1.5, accuracy: 1e-6, "vDecayed is 0 due to sPrev=1, so vNext must equal 1.5")
            v = res.vNext
            s = res.sNext
            step += 1
        }
    }

    /// 劣閾値定常電流による定常膜電位収束 (ゼロ発火) テスト
    func testLIFNeuronSubthresholdSteadyState() {
        let beta: Float = 0.8
        let vTh: Float = 1.0
        let config = LIFConfig(beta: beta, vTh: vTh, vReset: 0.0, alpha: 2.0)

        // I = 0.1 の場合、理論上の定常電位は I / (1 - beta) = 0.1 / 0.2 = 0.5 < 1.0
        let inputCurrent: Float = 0.1
        let expectedSteady: Float = inputCurrent / (1.0 - beta)

        var v: Float = 0.0
        var s: Float = 0.0

        var step = 0
        while step < 100 {
            let res = LIFNeuronEngine.stepScalar(config: config, vPrev: v, sPrev: s, inputCurrent: inputCurrent)
            XCTAssertEqual(res.sNext, 0.0)
            v = res.vNext
            s = res.sNext
            step += 1
        }
        XCTAssertEqual(v, expectedSteady, accuracy: 1e-5)
    }

    /// SIMD8 とスカラーエンジンの様々な境界長 (0〜256) とランダム・極小極大値での完全一致検証
    func testLIFNeuronSIMD8BoundaryEquivalenceExhaustive() {
        let config = LIFConfig(beta: 0.75, vTh: 1.0, vReset: 0.0, alpha: 2.0)
        let testCounts = [0, 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 63, 64, 127, 128, 255, 256]

        var tIdx = 0
        while tIdx < testCounts.count {
            let count = testCounts[tIdx]
            var vPrev = [Float](repeating: 0.0, count: count)
            var sPrev = [Float](repeating: 0.0, count: count)
            var inputCurrent = [Float](repeating: 0.0, count: count)

            var vScalar = [Float](repeating: 0.0, count: count)
            var sScalar = [Float](repeating: 0.0, count: count)
            var vSIMD = [Float](repeating: 0.0, count: count)
            var sSIMD = [Float](repeating: 0.0, count: count)

            var i = 0
            while i < count {
                vPrev[i] = Float((i * 37) % 200 - 100) * 0.02
                if i % 2 == 0 {
                    sPrev[i] = 1.0
                }
                if i % 2 != 0 {
                    sPrev[i] = 0.0
                }
                inputCurrent[i] = Float((i * 53) % 300 - 100) * 0.01
                i += 1
            }

            // スカラー
            i = 0
            while i < count {
                let res = LIFNeuronEngine.stepScalar(config: config, vPrev: vPrev[i], sPrev: sPrev[i], inputCurrent: inputCurrent[i])
                vScalar[i] = res.vNext
                sScalar[i] = res.sNext
                i += 1
            }

            // SIMD8
            if 0 < count {
                vPrev.withUnsafeBufferPointer { vp in
                    sPrev.withUnsafeBufferPointer { sp in
                        inputCurrent.withUnsafeBufferPointer { ip in
                            vSIMD.withUnsafeMutableBufferPointer { vop in
                                sSIMD.withUnsafeMutableBufferPointer { sop in
                                    LIFNeuronEngine.stepSIMD8(
                                        config: config,
                                        vPrevPtr: vp.baseAddress!,
                                        sPrevPtr: sp.baseAddress!,
                                        inputPtr: ip.baseAddress!,
                                        vNextPtr: vop.baseAddress!,
                                        sNextPtr: sop.baseAddress!,
                                        count: count
                                    )
                                }
                            }
                        }
                    }
                }
            }

            // 一致検証
            i = 0
            while i < count {
                XCTAssertEqual(vSIMD[i], vScalar[i], accuracy: 1e-5, "VNext mismatch at count=\(count), idx=\(i)")
                XCTAssertEqual(sSIMD[i], sScalar[i], "SNext mismatch at count=\(count), idx=\(i)")
                i += 1
            }

            tIdx += 1
        }
    }

    // MARK: - 2. Fast Sigmoid 代理勾配および数値安定性テスト (Surrogate Gradient)

    // MARK: - 3. BPTT 学習・勾配爆発/消失・異常データ注入テスト (BPTTTrainer & Adam)

    // MARK: - 4. SNN 動的スライス切り替え・状態整合性テスト (SpikingNetwork)

    /// Softmax の極大・極小ロジットに対する数値安定性 (NaN / ゼロ除算の回避)
    func testSoftmaxNumericalStability() {
        let net = SpikingNetwork(inputDim: 32, maxHiddenDim: 256, outputDim: 64, timeSteps: 4)

        // 重みを極大化 (ロジットが 1e5 を超えるケース)
        var i = 0
        while i < net.pBOut.count {
            net.pBOut.data[i] = 1e5
            i += 1
        }
        net.pBOut.data[0] = 1e5 + 10.0 // インデックス 0 が最大

        let features = [Float](repeating: 0.5, count: 32)
        let hBase = net.maxHiddenDim
        var v = [Float](repeating: 0.0, count: hBase)
        var s = [Float](repeating: 0.0, count: hBase)
        var sp = [Float](repeating: 0.0, count: hBase)
        var logit = [Float](repeating: 0.0, count: 64)
        var prob = [Float](repeating: 0.0, count: 64)

        net.forward(
            features: features,
            vPrev: &v,
            sPrev: &s,
            readoutSum: &sp,
            logits: &logit,
            probabilities: &prob
        )

        var sumP: Float = 0.0
        var c = 0
        while c < 64 {
            XCTAssertFalse(prob[c].isNaN, "Softmax produced NaN")
            XCTAssertFalse(prob[c].isInfinite, "Softmax produced Inf")
            XCTAssertLessThanOrEqual(0.0, prob[c])
            XCTAssertLessThanOrEqual(prob[c], 1.0)
            sumP += prob[c]
            c += 1
        }
        XCTAssertEqual(sumP, 1.0, accuracy: 1e-5)
    }

    // MARK: - 5. 固定小数点量子化エンジン極限ストレステスト (QuantizedEngine)

    /// 固定小数点推論における極大入力電流・オーバーフロー耐性テスト
    func testQuantizedEngineExtremeInputStability() {
        let net = SpikingNetwork(inputDim: 32, maxHiddenDim: 256, outputDim: 64, timeSteps: 4)
        let qConfig32 = QuantizedConfig.int32Config()
        let qWeights32 = QuantizedEngine.quantize(network: net, config: qConfig32)
        let engine32 = QuantizedEngine(weights: qWeights32, timeSteps: 4)
        let workspace32 = QuantizedWorkspace(maxHiddenDim: 256, inputDim: 32, outputDim: 64)

        // 通常の範囲 [0, 1] を遥かに超える極大入力 (100.0)
        let hugeFeatures = [Float](repeating: 100.0, count: 32)
        var probs = [Float](repeating: 0.0, count: 64)

        engine32.predict(
            features: hugeFeatures,
            workspace: workspace32,
            outputProbs: &probs
        )

        var sumP: Float = 0.0
        var c = 0
        while c < 64 {
            XCTAssertFalse(probs[c].isNaN, "Quantized Softmax produced NaN on huge input")
            XCTAssertFalse(probs[c].isInfinite, "Quantized Softmax produced Inf on huge input")
            XCTAssertLessThanOrEqual(0.0, probs[c])
            XCTAssertLessThanOrEqual(probs[c], 1.0)
            sumP += probs[c]
            c += 1
        }
        XCTAssertEqual(sumP, 1.0, accuracy: 1e-4)
    }

    /// 固定小数点推論における全ゼロ入力時の確率分布およびゼロ除算耐性
    func testQuantizedEngineZeroInputStability() {
        let net = SpikingNetwork(inputDim: 32, maxHiddenDim: 256, outputDim: 64, timeSteps: 4)
        let qConfig32 = QuantizedConfig.int32Config()
        let qWeights32 = QuantizedEngine.quantize(network: net, config: qConfig32)
        let engine32 = QuantizedEngine(weights: qWeights32, timeSteps: 4)
        let workspace32 = QuantizedWorkspace(maxHiddenDim: 256, inputDim: 32, outputDim: 64)

        let zeroFeatures = [Float](repeating: 0.0, count: 32)
        var probs = [Float](repeating: 0.0, count: 64)

        engine32.predict(
            features: zeroFeatures,
            workspace: workspace32,
            outputProbs: &probs
        )

        var sumP: Float = 0.0
        var c = 0
        while c < 64 {
            XCTAssertFalse(probs[c].isNaN)
            sumP += probs[c]
            c += 1
        }
        XCTAssertEqual(sumP, 1.0, accuracy: 1e-4)
    }

    // MARK: - 6. NaN / Inf 入力および空シーケンス耐性テスト (Anomaly Data Robustness)

    /// LIF ニューロンに NaN / Inf / -Inf が入力された場合の非クラッシュ・SIMD8一致テスト
    func testLIFNeuronNaNAndInfInputs() {
        let config = LIFConfig(beta: 0.8, vTh: 1.0, vReset: 0.0, alpha: 2.0)

        // 1. スカラーでの NaN / Inf 注入
        let resNaN = LIFNeuronEngine.stepScalar(config: config, vPrev: 0.0, sPrev: 0.0, inputCurrent: Float.nan)
        XCTAssertTrue(resNaN.vNext.isNaN)
        // NaN に対して 1.0 <= vNext は false となり sNext = 0.0
        XCTAssertEqual(resNaN.sNext, 0.0)

        // Inf は飽和範囲に丸められ、無限大が伝播しない
        let resInf = LIFNeuronEngine.stepScalar(config: config, vPrev: 0.0, sPrev: 0.0, inputCurrent: Float.infinity)
        XCTAssertEqual(resInf.vNext, LIFNeuronEngine.vClampMax)
        XCTAssertEqual(resInf.sNext, 1.0)

        let resNegInf = LIFNeuronEngine.stepScalar(config: config, vPrev: 0.0, sPrev: 0.0, inputCurrent: -Float.infinity)
        XCTAssertEqual(resNegInf.vNext, LIFNeuronEngine.vClampMin)
        XCTAssertEqual(resNegInf.sNext, 0.0)

        // 2. SIMD8 での NaN / Inf 混在処理
        let vPrevArr: [Float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        let sPrevArr: [Float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        let inputArr: [Float] = [Float.nan, Float.infinity, -Float.infinity, 0.5, 1.5, Float.nan, 2.0, -1.0]

        var vOutArr = [Float](repeating: 0.0, count: 8)
        var sOutArr = [Float](repeating: 0.0, count: 8)

        vPrevArr.withUnsafeBufferPointer { vp in
            sPrevArr.withUnsafeBufferPointer { sp in
                inputArr.withUnsafeBufferPointer { ip in
                    vOutArr.withUnsafeMutableBufferPointer { vop in
                        sOutArr.withUnsafeMutableBufferPointer { sop in
                            LIFNeuronEngine.stepSIMD8(
                                config: config,
                                vPrevPtr: vp.baseAddress!,
                                sPrevPtr: sp.baseAddress!,
                                inputPtr: ip.baseAddress!,
                                vNextPtr: vop.baseAddress!,
                                sNextPtr: sop.baseAddress!,
                                count: 8
                            )
                        }
                    }
                }
            }
        }

        // SIMD8 とスカラーの一致検証
        var i = 0
        while i < 8 {
            let scRes = LIFNeuronEngine.stepScalar(config: config, vPrev: vPrevArr[i], sPrev: sPrevArr[i], inputCurrent: inputArr[i])
            if scRes.vNext.isNaN {
                XCTAssertTrue(vOutArr[i].isNaN)
            }
            if scRes.vNext.isNaN != true {
                XCTAssertEqual(vOutArr[i], scRes.vNext)
            }
            XCTAssertEqual(sOutArr[i], scRes.sNext)
            i += 1
        }
    }

    /// Int16 量子化と Int32 量子化のビットシフト減衰比較・ダイナミックレンジ検証
    func testQuantizedInt16VsInt32RangeAndDecay() {
        let config16 = QuantizedConfig.int16Config()
        let config32 = QuantizedConfig.int32Config()

        XCTAssertEqual(config16.scale, 2048.0)
        XCTAssertEqual(config16.scaleBits, 11)
        XCTAssertEqual(config16.vThInt, 2048)

        XCTAssertEqual(config32.scale, 65536.0)
        XCTAssertEqual(config32.scaleBits, 16)
        XCTAssertEqual(config32.vThInt, 65536)

        // 減衰比率の検証: decayNum / (2^decayBits) ~= 0.8
        let decayRatio16 = Float(config16.decayNum) / Float(1 << config16.decayBits)
        let decayRatio32 = Float(config32.decayNum) / Float(1 << config32.decayBits)

        XCTAssertLessThan(abs(decayRatio16 - 0.8), 0.001)
        XCTAssertLessThan(abs(decayRatio32 - 0.8), 0.0001)
    }
}
