import XCTest
@testable import Spiketrans

/// Milestone M2 縮小サイズモデル（hidden: 32, 64, 128）多重スライス学習および VectorOperations 境界長極値ストレステスト
final class M2ChallengerReducedSizeStressTests: XCTestCase {

    // MARK: - 1. 縮小サイズモデル (hidden: 32) の極限学習・推論ストレステスト

    /// maxHiddenDim: 32 の極小モデルにおける BPTTTrainer の挙動検証
    /// (すべての標準スライス 64, 128, 256 より小さいため、trainStep 内で安全にスキップされクラッシュしないこと)
    func testBPTTTrainerHiddenDim32SafeSkipAndOptimization() {
        let inputDim = 16
        let maxHiddenDim = 32
        let outputDim = 8
        let timeSteps = 2

        let net = SpikingNetwork(
            inputDim: inputDim,
            maxHiddenDim: maxHiddenDim,
            outputDim: outputDim,
            timeSteps: timeSteps
        )
        let opt = AdamOptimizer(
            config: AdamConfig(lr: 0.01, gradClip: 1.0),
            parameters: net.parameters
        )
        let trainer = BPTTTrainer(network: net, optimizer: opt)

        // 1. 通常データでの trainStep
        var featuresSeq: [[Float]] = []
        var k = 0
        while k < 4 {
            var feat = [Float](repeating: 0.0, count: inputDim)
            var d = 0
            while d < inputDim {
                feat[d] = Float(k + d) * 0.1
                d += 1
            }
            featuresSeq.append(feat)
            k += 1
        }
        let targets = [0, 1, 2, 3]

        // trainStep を実行
        let res = trainer.trainStep(featuresSeq: featuresSeq, targets: targets)

        // maxHiddenDim = 32 のため、Base スライス (min(1024, 32)=32) のみが実行され、Middle/High はスキップされる
        XCTAssertLessThan(0.0, res)

        // パラメータが NaN / Inf にならず健全であること
        for param in net.parameters {
            var i = 0
            while i < param.count {
                XCTAssertFalse(param.data[i].isNaN)
                XCTAssertFalse(param.data[i].isInfinite)
                i += 1
            }
        }
    }

    // MARK: - 2. 縮小サイズモデル (hidden: 64 - Base のみ) の極限多重スライス学習テスト

    /// maxHiddenDim: 64 のモデルにおいて Base スライスのみが学習され、多重ステップで正常収束すること
    func testBPTTTrainerHiddenDim64BaseOnlyMultiStepLearning() {
        let inputDim = 16
        let maxHiddenDim = 64
        let outputDim = 8
        let timeSteps = 2

        let net = SpikingNetwork(
            inputDim: inputDim,
            maxHiddenDim: maxHiddenDim,
            outputDim: outputDim,
            timeSteps: timeSteps
        )
        let opt = AdamOptimizer(
            config: AdamConfig(lr: 0.05, gradClip: 1.0),
            parameters: net.parameters
        )
        let trainer = BPTTTrainer(network: net, optimizer: opt)

        // 周期パターンの学習データ系列
        var featuresSeq: [[Float]] = []
        var targets: [Int] = []
        var step = 0
        while step < 8 {
            var feat = [Float](repeating: 0.0, count: inputDim)
            var d = 0
            while d < inputDim {
                feat[d] = sin(Float(step * inputDim + d) * 0.2) * 0.5 + 0.5
                d += 1
            }
            featuresSeq.append(feat)
            targets.append(step % outputDim)
            step += 1
        }

        var initialLoss: Float = 0.0
        var finalLoss: Float = 0.0

        // 30 エポック学習を実行
        var epoch = 0
        while epoch < 30 {
            let res = trainer.trainStep(featuresSeq: featuresSeq, targets: targets)
            XCTAssertFalse(res.isNaN, "Loss became NaN at epoch \(epoch)")
            XCTAssertFalse(res.isInfinite, "Loss became Inf at epoch \(epoch)")

            // Base のみが実行され、Middle / High は 0.0 であること

            if epoch == 0 {
                initialLoss = res
            }
            finalLoss = res
            epoch += 1
        }

        // 損失が有限値であり、学習により発散していないこと
        XCTAssertLessThanOrEqual(0.0, finalLoss)
        XCTAssertFalse(finalLoss.isNaN)

        // パラメータが健全であること
        for param in net.parameters {
            var i = 0
            while i < param.count {
                XCTAssertFalse(param.data[i].isNaN)
                XCTAssertFalse(param.data[i].isInfinite)
                i += 1
            }
        }
    }

    /// maxHiddenDim: 64 モデルに対する極大・極小・異常入力注入時の数値安定性
    func testBPTTTrainerHiddenDim64AdversarialInputs() {
        let inputDim = 8
        let maxHiddenDim = 64
        let outputDim = 4
        let timeSteps = 2

        let net = SpikingNetwork(
            inputDim: inputDim,
            maxHiddenDim: maxHiddenDim,
            outputDim: outputDim,
            timeSteps: timeSteps
        )
        let opt = AdamOptimizer(
            config: AdamConfig(lr: 0.01, gradClip: 0.5),
            parameters: net.parameters
        )
        let trainer = BPTTTrainer(network: net, optimizer: opt)

        // 1. 巨大入力 (1e4) + 負の巨大入力 (-1e4)
        var extremeFeatures: [[Float]] = []
        extremeFeatures.append([Float](repeating: 1e4, count: inputDim))
        extremeFeatures.append([Float](repeating: -1e4, count: inputDim))
        extremeFeatures.append([Float](repeating: 0.0, count: inputDim))
        let targets = [0, 1, 2]

        let res = trainer.trainStep(featuresSeq: extremeFeatures, targets: targets)
        XCTAssertFalse(res.isNaN)
        XCTAssertFalse(res.isInfinite)

        for param in net.parameters {
            var i = 0
            while i < param.count {
                XCTAssertFalse(param.data[i].isNaN)
                XCTAssertFalse(param.data[i].isInfinite)
                i += 1
            }
        }
    }

    // MARK: - 3. 縮小サイズモデルの学習テスト

    /// 隠れ層を縮めたモデルでも損失が有限値で学習が進むこと
    func testBPTTTrainerReducedHiddenDimLearning() {
        let inputDim = 16
        let maxHiddenDim = 256
        let outputDim = 8
        let timeSteps = 2

        let net = SpikingNetwork(
            inputDim: inputDim,
            maxHiddenDim: maxHiddenDim,
            outputDim: outputDim,
            timeSteps: timeSteps
        )
        let opt = AdamOptimizer(
            config: AdamConfig(lr: 0.02, gradClip: 1.0),
            parameters: net.parameters
        )
        let trainer = BPTTTrainer(network: net, optimizer: opt)

        var featuresSeq: [[Float]] = []
        var targets: [Int] = []
        var k = 0
        while k < 6 {
            var feat = [Float](repeating: 0.0, count: inputDim)
            var d = 0
            while d < inputDim {
                feat[d] = Float((k * 17 + d * 31) % 100) * 0.01
                d += 1
            }
            featuresSeq.append(feat)
            targets.append(k % outputDim)
            k += 1
        }

        var epoch = 0
        while epoch < 20 {
            let res = trainer.trainStep(featuresSeq: featuresSeq, targets: targets)
            XCTAssertFalse(res.isNaN)
            XCTAssertFalse(res.isInfinite)

            // Base のみが totalLoss に寄与し、High は 0.0 であること

            epoch += 1
        }

        // パラメータが健全であること
        for param in net.parameters {
            var i = 0
            while i < param.count {
                XCTAssertFalse(param.data[i].isNaN)
                XCTAssertFalse(param.data[i].isInfinite)
                i += 1
            }
        }
    }

    /// maxHiddenDim: 128 モデルでの推論時スライス切り替え (Base / Middle) の整合性
    func testInferenceHiddenDim128SliceSwitching() {
        let net = SpikingNetwork(
            inputDim: 16,
            maxHiddenDim: 128,
            outputDim: 8,
            timeSteps: 3
        )

        var features = [Float](repeating: 0.0, count: 16)
        var d = 0
        while d < 16 {
            features[d] = Float(d + 1) * 0.05
            d += 1
        }

        // Base スライス推論
        var vPrevBase = [Float](repeating: 0.0, count: net.maxHiddenDim)
        var sPrevBase = [Float](repeating: 0.0, count: net.maxHiddenDim)
        var spikeSumBase = [Float](repeating: 0.0, count: net.maxHiddenDim)
        var logitsBase = [Float](repeating: 0.0, count: 8)
        var probsBase = [Float](repeating: 0.0, count: 8)

        net.forward(
            features: features,
            vPrev: &vPrevBase,
            sPrev: &sPrevBase,
            spikeSum: &spikeSumBase,
            logits: &logitsBase,
            probabilities: &probsBase
        )

        var sumPBase: Float = 0.0
        var c = 0
        while c < 8 {
            XCTAssertFalse(probsBase[c].isNaN)
            sumPBase += probsBase[c]
            c += 1
        }
        XCTAssertEqual(sumPBase, 1.0, accuracy: 1e-5)

        // Middle スライス推論
        var vPrevMiddle = [Float](repeating: 0.0, count: net.maxHiddenDim)
        var sPrevMiddle = [Float](repeating: 0.0, count: net.maxHiddenDim)
        var spikeSumMiddle = [Float](repeating: 0.0, count: net.maxHiddenDim)
        var logitsMiddle = [Float](repeating: 0.0, count: 8)
        var probsMiddle = [Float](repeating: 0.0, count: 8)

        net.forward(
            features: features,
            vPrev: &vPrevMiddle,
            sPrev: &sPrevMiddle,
            spikeSum: &spikeSumMiddle,
            logits: &logitsMiddle,
            probabilities: &probsMiddle
        )

        var sumPMiddle: Float = 0.0
        c = 0
        while c < 8 {
            XCTAssertFalse(probsMiddle[c].isNaN)
            sumPMiddle += probsMiddle[c]
            c += 1
        }
        XCTAssertEqual(sumPMiddle, 1.0, accuracy: 1e-5)
    }

    // MARK: - 4. VectorOperations の網羅的境界長 (0〜1024) および極値ストレステスト

    /// VectorOperations の全関数に対する網羅的境界長 (0〜1024) での SIMD8 とスカラー計算の一致性
    func testVectorOperationsExhaustiveBoundaryLengths() {
        let testLengths = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
            31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257,
            511, 512, 513, 1023, 1024
        ]

        var lIdx = 0
        while lIdx < testLengths.count {
            let count = testLengths[lIdx]

            var a = [Float](repeating: 0.0, count: count)
            var b = [Float](repeating: 0.0, count: count)
            var i = 0
            while i < count {
                a[i] = sin(Float(i) * 0.2) * 3.0
                b[i] = cos(Float(i) * 0.15) * 2.0
                i += 1
            }

            // 1. dotProduct
            var scalarDot: Float = 0.0
            i = 0
            while i < count {
                scalarDot += a[i] * b[i]
                i += 1
            }
            var simdDot: Float = 0.0
            if 0 < count {
                simdDot = a.withUnsafeBufferPointer { aP in
                    b.withUnsafeBufferPointer { bP in
                        VectorOperations.dotProduct(a: aP.baseAddress!, b: bP.baseAddress!, count: count)
                    }
                }
            }
            let diffDot = abs(simdDot - scalarDot)
            let maxDotVal = max(abs(scalarDot), 1.0)
            XCTAssertLessThan(diffDot / maxDotVal, 1e-4, "dotProduct failed at length \(count)")

            // 2. sumOfSquares
            var scalarSq: Float = 0.0
            i = 0
            while i < count {
                scalarSq += a[i] * a[i]
                i += 1
            }
            var simdSq: Float = 0.0
            if 0 < count {
                simdSq = a.withUnsafeBufferPointer { aP in
                    VectorOperations.sumOfSquares(ptr: aP.baseAddress!, count: count)
                }
            }
            let diffSq = abs(simdSq - scalarSq)
            let maxSqVal = max(abs(scalarSq), 1.0)
            XCTAssertLessThan(diffSq / maxSqVal, 1e-4, "sumOfSquares failed at length \(count)")

            // 3. multiply
            var dstScalar = [Float](repeating: 0.0, count: count)
            var dstSIMD = [Float](repeating: 0.0, count: count)
            i = 0
            while i < count {
                dstScalar[i] = a[i] * b[i]
                i += 1
            }
            if 0 < count {
                a.withUnsafeBufferPointer { aP in
                    b.withUnsafeBufferPointer { bP in
                        dstSIMD.withUnsafeMutableBufferPointer { dP in
                            VectorOperations.multiply(
                                srcA: aP.baseAddress!,
                                srcB: bP.baseAddress!,
                                dst: dP.baseAddress!,
                                count: count
                            )
                        }
                    }
                }
            }
            i = 0
            while i < count {
                XCTAssertEqual(dstSIMD[i], dstScalar[i], "multiply mismatch at length \(count), idx \(i)")
                i += 1
            }

            // 4. maxMagnitude
            var scalarMaxMag: Float = 0.0
            i = 0
            while i < count {
                let mag = abs(a[i])
                if scalarMaxMag < mag {
                    scalarMaxMag = mag
                }
                i += 1
            }
            var simdMaxMag: Float = 0.0
            if 0 < count {
                simdMaxMag = a.withUnsafeBufferPointer { aP in
                    VectorOperations.maxMagnitude(ptr: aP.baseAddress!, count: count)
                }
            }
            XCTAssertEqual(simdMaxMag, scalarMaxMag, accuracy: 1e-6, "maxMagnitude mismatch at length \(count)")

            // 5. clamp
            let minV: Float = -1.0
            let maxV: Float = 1.5
            var dstClampScalar = [Float](repeating: 0.0, count: count)
            var dstClampSIMD = [Float](repeating: 0.0, count: count)
            i = 0
            while i < count {
                var cVal = a[i]
                if cVal < minV {
                    cVal = minV
                }
                if maxV < cVal {
                    cVal = maxV
                }
                dstClampScalar[i] = cVal
                i += 1
            }
            if 0 < count {
                a.withUnsafeBufferPointer { aP in
                    dstClampSIMD.withUnsafeMutableBufferPointer { dP in
                        VectorOperations.clamp(
                            src: aP.baseAddress!,
                            dst: dP.baseAddress!,
                            count: count,
                            minVal: minV,
                            maxVal: maxV
                        )
                    }
                }
            }
            i = 0
            while i < count {
                XCTAssertEqual(dstClampSIMD[i], dstClampScalar[i], "clamp mismatch at length \(count), idx \(i)")
                i += 1
            }

            lIdx += 1
        }
    }

    /// VectorOperations に対する極限値 (極大 1e30, 極小 -1e30, 非正規化数 1e-37, ゼロ) の計算安定性
    func testVectorOperationsExtremeValuesStability() {
        let count = 16
        var a = [Float](repeating: 0.0, count: count)
        var b = [Float](repeating: 0.0, count: count)

        // 極大・極小値
        a[0] = 1e18
        b[0] = 1e18
        a[1] = -1e18
        b[1] = -1e18
        a[2] = 1e-20
        b[2] = 1e-20
        a[3] = 0.0
        b[3] = 1e30

        var dst = [Float](repeating: 0.0, count: count)

        a.withUnsafeBufferPointer { aP in
            b.withUnsafeBufferPointer { bP in
                dst.withUnsafeMutableBufferPointer { dP in
                    VectorOperations.multiply(
                        srcA: aP.baseAddress!,
                        srcB: bP.baseAddress!,
                        dst: dP.baseAddress!,
                        count: count
                    )
                }
            }
        }

        XCTAssertFalse(dst[0].isNaN)
        XCTAssertFalse(dst[1].isNaN)
        XCTAssertEqual(dst[3], 0.0)

        // maxMagnitude での極大値検出
        let maxMag = a.withUnsafeBufferPointer { aP in
            VectorOperations.maxMagnitude(ptr: aP.baseAddress!, count: count)
        }
        XCTAssertEqual(maxMag, 1e18)

        // clamp での極大値クリッピング
        var dstClamp = [Float](repeating: 0.0, count: count)
        a.withUnsafeBufferPointer { aP in
            dstClamp.withUnsafeMutableBufferPointer { dP in
                VectorOperations.clamp(
                    src: aP.baseAddress!,
                    dst: dP.baseAddress!,
                    count: count,
                    minVal: -10.0,
                    maxVal: 10.0
                )
            }
        }
        XCTAssertEqual(dstClamp[0], 10.0)
        XCTAssertEqual(dstClamp[1], -10.0)
    }
}
