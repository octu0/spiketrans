import XCTest
@testable import Spiketrans

final class LIFNeuronTests: XCTestCase {

    // MARK: - 膜電位減衰特性テスト

    func testDecayDynamics() {
        let config = LIFConfig(beta: 0.8, vTh: 1.0, vReset: 0.0, alpha: 2.0)
        var v: Float = 0.5
        var s: Float = 0.0

        // ステップ 1: 入力電流 0.0
        let step1 = LIFNeuronEngine.stepScalar(config: config, vPrev: v, sPrev: s, inputCurrent: 0.0)
        XCTAssertEqual(step1.vNext, 0.4, accuracy: 1e-6)
        XCTAssertEqual(step1.sNext, 0.0)
        v = step1.vNext
        s = step1.sNext

        // ステップ 2: 入力電流 0.0
        let step2 = LIFNeuronEngine.stepScalar(config: config, vPrev: v, sPrev: s, inputCurrent: 0.0)
        XCTAssertEqual(step2.vNext, 0.32, accuracy: 1e-6)
        XCTAssertEqual(step2.sNext, 0.0)
        v = step2.vNext
        s = step2.sNext

        // ステップ 3: 入力電流 0.0
        let step3 = LIFNeuronEngine.stepScalar(config: config, vPrev: v, sPrev: s, inputCurrent: 0.0)
        XCTAssertEqual(step3.vNext, 0.256, accuracy: 1e-6)
        XCTAssertEqual(step3.sNext, 0.0)
    }

    // MARK: - 発火閾値・リセット機構テスト

    func testThresholdAndSpike() {
        let config = LIFConfig(beta: 0.8, vTh: 1.0, vReset: 0.0, alpha: 2.0)

        // 1. 閾値未満 (0.9 < 1.0) -> 発火なし
        let subTh = LIFNeuronEngine.stepScalar(config: config, vPrev: 0.0, sPrev: 0.0, inputCurrent: 0.9)
        XCTAssertEqual(subTh.vNext, 0.9, accuracy: 1e-6)
        XCTAssertEqual(subTh.sNext, 0.0)

        // 2. 閾値以上 (1.0 <= 1.2) -> 発火
        let supraTh = LIFNeuronEngine.stepScalar(config: config, vPrev: 0.0, sPrev: 0.0, inputCurrent: 1.2)
        XCTAssertEqual(supraTh.vNext, 1.2, accuracy: 1e-6)
        XCTAssertEqual(supraTh.sNext, 1.0)

        // 3. 発火後の次ステップリセット (sPrev = 1.0 により減衰項が 0 にリセット)
        let postSpike = LIFNeuronEngine.stepScalar(config: config, vPrev: supraTh.vNext, sPrev: supraTh.sNext, inputCurrent: 0.3)
        XCTAssertEqual(postSpike.vNext, 0.3, accuracy: 1e-6)
        XCTAssertEqual(postSpike.sNext, 0.0)
    }

    // MARK: - SIMD8 vs スカラー完全一致テスト

    func testSIMD8Equivalence() {
        let testCounts = [0, 1, 7, 8, 9, 15, 16, 17, 32, 64, 128, 256]
        let config = LIFConfig(beta: 0.8, vTh: 1.0, vReset: 0.0, alpha: 2.0)

        var tIdx = 0
        while tIdx < testCounts.count {
            let count = testCounts[tIdx]
            var vPrev = [Float](repeating: 0.0, count: count)
            var sPrev = [Float](repeating: 0.0, count: count)
            var inputCurrent = [Float](repeating: 0.0, count: count)

            var vNextScalar = [Float](repeating: 0.0, count: count)
            var sNextScalar = [Float](repeating: 0.0, count: count)
            var vNextSIMD = [Float](repeating: 0.0, count: count)
            var sNextSIMD = [Float](repeating: 0.0, count: count)

            var i = 0
            while i < count {
                vPrev[i] = sin(Float(i) * 0.5) * 1.2
                if i % 3 == 0 {
                    sPrev[i] = 1.0
                } else {
                    sPrev[i] = 0.0
                }
                inputCurrent[i] = cos(Float(i) * 0.7) * 1.5
                i += 1
            }

            // スカラー計算
            i = 0
            while i < count {
                let res = LIFNeuronEngine.stepScalar(
                    config: config,
                    vPrev: vPrev[i],
                    sPrev: sPrev[i],
                    inputCurrent: inputCurrent[i]
                )
                vNextScalar[i] = res.vNext
                sNextScalar[i] = res.sNext
                i += 1
            }

            // SIMD8 計算
            if 0 < count {
                vPrev.withUnsafeBufferPointer { vPtr in
                    sPrev.withUnsafeBufferPointer { sPtr in
                        inputCurrent.withUnsafeBufferPointer { iPtr in
                            vNextSIMD.withUnsafeMutableBufferPointer { vOutPtr in
                                sNextSIMD.withUnsafeMutableBufferPointer { sOutPtr in
                                    LIFNeuronEngine.stepSIMD8(
                                        config: config,
                                        vPrevPtr: vPtr.baseAddress!,
                                        sPrevPtr: sPtr.baseAddress!,
                                        inputPtr: iPtr.baseAddress!,
                                        vNextPtr: vOutPtr.baseAddress!,
                                        sNextPtr: sOutPtr.baseAddress!,
                                        count: count
                                    )
                                }
                            }
                        }
                    }
                }
            }

            // 比較検証
            i = 0
            while i < count {
                let diffV = abs(vNextSIMD[i] - vNextScalar[i])
                XCTAssertLessThan(diffV, 1e-6, "VNext mismatch at index \(i) for count \(count)")
                XCTAssertEqual(sNextSIMD[i], sNextScalar[i], "SNext mismatch at index \(i) for count \(count)")
                i += 1
            }

            tIdx += 1
        }
    }

    // MARK: - Direct Input Current 注入テスト

    func testDirectInputCurrentInjection() {
        let config = LIFConfig(beta: 0.8, vTh: 1.0, vReset: 0.0, alpha: 2.0)
        let state = LIFState(size: 8)

        var input = [Float](repeating: 0.0, count: 8)
        var i = 0
        while i < 8 {
            input[i] = Float(i) * 0.2 // 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4
            i += 1
        }

        var vOut = [Float](repeating: 0.0, count: 8)
        var sOut = [Float](repeating: 0.0, count: 8)

        state.v.withUnsafeBufferPointer { vPtr in
            state.s.withUnsafeBufferPointer { sPtr in
                input.withUnsafeBufferPointer { iPtr in
                    vOut.withUnsafeMutableBufferPointer { voPtr in
                        sOut.withUnsafeMutableBufferPointer { soPtr in
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

        // i = 0..4 (0.0 .. 0.8) -> 発火なし
        i = 0
        while i < 5 {
            XCTAssertEqual(sOut[i], 0.0)
            XCTAssertEqual(vOut[i], Float(i) * 0.2, accuracy: 1e-6)
            i += 1
        }

        // i = 5..7 (1.0 .. 1.4) -> 発火
        while i < 8 {
            XCTAssertEqual(sOut[i], 1.0)
            XCTAssertEqual(vOut[i], Float(i) * 0.2, accuracy: 1e-6)
            i += 1
        }

        state.reset()
        i = 0
        while i < 8 {
            XCTAssertEqual(state.v[i], 0.0)
            XCTAssertEqual(state.s[i], 0.0)
            i += 1
        }
    }
}
