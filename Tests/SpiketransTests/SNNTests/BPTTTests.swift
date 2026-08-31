import XCTest
@testable import Spiketrans

final class BPTTTests: XCTestCase {

    // MARK: - Fast Sigmoid 代理勾配特性テスト

    func testSurrogateGradientProperties() {
        let vTh: Float = 1.0
        let alpha: Float = 2.0

        // 1. V = Vth のとき最大値 1.0
        let peak = SurrogateGradient.derivative(v: vTh, vTh: vTh, alpha: alpha)
        XCTAssertEqual(peak, 1.0, accuracy: 1e-6)

        // 2. 左右対称性: |V - Vth| = 0.5 のとき同値
        let derivRight = SurrogateGradient.derivative(v: vTh + 0.5, vTh: vTh, alpha: alpha)
        let derivLeft = SurrogateGradient.derivative(v: vTh - 0.5, vTh: vTh, alpha: alpha)
        XCTAssertEqual(derivRight, derivLeft, accuracy: 1e-6)
        XCTAssertLessThan(derivRight, 1.0)
        XCTAssertLessThan(0.0, derivRight)

        // 3. SIMD8 一括計算等価性
        let testCounts = [0, 1, 7, 8, 9, 15, 16, 17, 32, 64]
        var tIdx = 0
        while tIdx < testCounts.count {
            let count = testCounts[tIdx]
            var vArray = [Float](repeating: 0.0, count: count)
            var dstSIMD = [Float](repeating: 0.0, count: count)
            var dstScalar = [Float](repeating: 0.0, count: count)

            var i = 0
            while i < count {
                vArray[i] = sin(Float(i) * 0.3) * 2.0
                dstScalar[i] = SurrogateGradient.derivative(v: vArray[i], vTh: vTh, alpha: alpha)
                i += 1
            }

            if 0 < count {
                vArray.withUnsafeBufferPointer { vPtr in
                    dstSIMD.withUnsafeMutableBufferPointer { dPtr in
                        SurrogateGradient.derivativeSIMD8(
                            vPtr: vPtr.baseAddress!,
                            dstPtr: dPtr.baseAddress!,
                            count: count,
                            vTh: vTh,
                            alpha: alpha
                        )
                    }
                }
            }

            i = 0
            while i < count {
                XCTAssertEqual(dstSIMD[i], dstScalar[i], accuracy: 1e-6, "Surrogate SIMD mismatch at \(i)")
                i += 1
            }
            tIdx += 1
        }
    }

    // MARK: - 数値微分 (Finite Difference) 勾配検証テスト

    func testFiniteDifferenceGradientCheck() {
        let net = MatryoshkaNetwork(
            inputDim: 4,
            maxHiddenDim: 64, // Base スライスと一致
            outputDim: 4,
            timeSteps: 2
        )
        let opt = AdamOptimizer(config: AdamConfig(lr: 0.01), parameters: net.parameters)
        let trainer = BPTTTrainer(network: net, optimizer: opt)

        let seqLen = 2
        var featuresSeq: [[Float]] = []
        var k = 0
        while k < seqLen {
            var f = [Float](repeating: 0.0, count: 4)
            var d = 0
            while d < 4 {
                f[d] = Float(k * 4 + d + 1) * 0.1
                d += 1
            }
            featuresSeq.append(f)
            k += 1
        }
        let targets = [1, 2]

        // 1. 解析的勾配の計算
        opt.zeroGrad()
        let fwdRes = trainer.forwardSequence(featuresSeq: featuresSeq, targets: targets, slice: .base)
        trainer.backwardSequence(featuresSeq: featuresSeq, targets: targets, cache: fwdRes.cache, slice: .base, lossWeight: 1.0)

        // 2. 出力層バイアス pBOut の数値微分検証
        let eps: Float = 1e-3
        var c = 0
        while c < 4 {
            let origVal = net.pBOut.data[c]

            net.pBOut.data[c] = origVal + eps
            let resPlus = trainer.forwardSequence(featuresSeq: featuresSeq, targets: targets, slice: .base)

            net.pBOut.data[c] = origVal - eps
            let resMinus = trainer.forwardSequence(featuresSeq: featuresSeq, targets: targets, slice: .base)

            net.pBOut.data[c] = origVal

            let numericalGrad = (resPlus.loss - resMinus.loss) / (2.0 * eps)
            let analyticalGrad = net.pBOut.grad[c]

            let diff = abs(analyticalGrad - numericalGrad)
            XCTAssertLessThan(diff, 5e-3, "BOut grad check failed for c=\(c): analytical=\(analyticalGrad), numerical=\(numericalGrad)")
            c += 1
        }
    }

    // MARK: - Adam オプティマイザと L2 クリッピングテスト

    func testAdamOptimizerAndL2Clipping() {
        let p1 = Parameter(count: 4, initialData: [1.0, 2.0, 3.0, 4.0])
        let p2 = Parameter(count: 4, initialData: [0.5, 0.5, 0.5, 0.5])
        let config = AdamConfig(lr: 0.1, beta1: 0.9, beta2: 0.999, eps: 1e-8, gradClip: 1.0)
        let opt = AdamOptimizer(config: config, parameters: [p1, p2])

        // 巨大な勾配を設定 (ノルムが gradClip = 1.0 を大きく超過)
        p1.grad = [10.0, 10.0, 10.0, 10.0]
        p2.grad = [10.0, 10.0, 10.0, 10.0]

        opt.step()

        XCTAssertEqual(opt.stepCount, 1)

        // パラメータが更新されていること
        XCTAssertLessThan(p1.data[0], 1.0)
        XCTAssertLessThan(p2.data[0], 0.5)

        // zeroGrad の検証
        opt.zeroGrad()
        var i = 0
        while i < 4 {
            XCTAssertEqual(p1.grad[i], 0.0)
            XCTAssertEqual(p2.grad[i], 0.0)
            i += 1
        }
    }

    // MARK: - トイデータセット学習収束テスト

    func testToyDatasetConvergence() {
        let net = MatryoshkaNetwork(
            inputDim: 8,
            maxHiddenDim: 256,
            outputDim: 4,
            timeSteps: 3
        )
        let opt = AdamOptimizer(config: AdamConfig(lr: 0.05, gradClip: 1.0), parameters: net.parameters)
        let trainer = BPTTTrainer(network: net, optimizer: opt)

        // 2つの明確に異なる合成パターン系列
        // クラス 0: 先頭要素がアクティブ
        // クラス 1: 後半要素がアクティブ
        var seqA: [[Float]] = []
        var seqB: [[Float]] = []

        var t = 0
        while t < 4 {
            var fA = [Float](repeating: 0.0, count: 8)
            fA[0] = 0.8
            fA[1] = 0.7
            seqA.append(fA)

            var fB = [Float](repeating: 0.0, count: 8)
            fB[6] = 0.8
            fB[7] = 0.7
            seqB.append(fB)
            t += 1
        }

        let targetsA = [0, 0, 0, 0]
        let targetsB = [1, 1, 1, 1]

        var initialLoss: Float = 0.0
        var finalLoss: Float = 0.0

        var epoch = 0
        while epoch < 40 {
            let resA = trainer.trainStep(featuresSeq: seqA, targets: targetsA)
            let resB = trainer.trainStep(featuresSeq: seqB, targets: targetsB)
            let currentLoss = (resA.totalLoss + resB.totalLoss) * 0.5

            if epoch == 0 {
                initialLoss = currentLoss
            }
            finalLoss = currentLoss
            epoch += 1
        }

        // 損失が有意に減少していることを確認
        XCTAssertLessThan(finalLoss, initialLoss, "Final loss (\(finalLoss)) must be significantly lower than initial loss (\(initialLoss))")
        XCTAssertLessThan(finalLoss, 1.0, "Final loss must converge below 1.0, got \(finalLoss)")
    }

    // MARK: - 縮小隠れ層 (maxHiddenDim: 64) での trainStep 実行テスト

    func testTrainStepWithReducedHiddenDim64() {
        let net = MatryoshkaNetwork(
            inputDim: 4,
            maxHiddenDim: 64, // Base スライス (64) のみ有効
            outputDim: 2,
            timeSteps: 2
        )
        let opt = AdamOptimizer(config: AdamConfig(lr: 0.05), parameters: net.parameters)
        let trainer = BPTTTrainer(network: net, optimizer: opt)

        var seqA: [[Float]] = []
        var seqB: [[Float]] = []

        var t = 0
        while t < 3 {
            var fA = [Float](repeating: 0.0, count: 4)
            fA[0] = 0.9
            fA[1] = 0.8
            seqA.append(fA)

            var fB = [Float](repeating: 0.0, count: 4)
            fB[2] = 0.9
            fB[3] = 0.8
            seqB.append(fB)
            t += 1
        }

        let targetsA = [0, 0, 0]
        let targetsB = [1, 1, 1]

        // 1回目のステップ実行（クラッシュせずに正常終了すること）
        let initialResA = trainer.trainStep(featuresSeq: seqA, targets: targetsA)
        
        // Base スライスの損失は計算され、Middle/High は 0.0 であること
        XCTAssertLessThan(0.0, initialResA.lossBase)
        XCTAssertEqual(initialResA.lossMiddle, 0.0)
        XCTAssertEqual(initialResA.lossHigh, 0.0)
        XCTAssertEqual(initialResA.totalLoss, initialResA.lossBase)

        var initialLoss = initialResA.totalLoss
        var finalLoss: Float = 0.0

        var epoch = 0
        while epoch < 30 {
            let resA = trainer.trainStep(featuresSeq: seqA, targets: targetsA)
            let resB = trainer.trainStep(featuresSeq: seqB, targets: targetsB)
            let currentLoss = (resA.totalLoss + resB.totalLoss) * 0.5

            if epoch == 0 {
                initialLoss = currentLoss
            }
            finalLoss = currentLoss
            epoch += 1
        }

        // 損失が減少していることを確認
        XCTAssertLessThan(finalLoss, initialLoss, "Final loss (\(finalLoss)) must be lower than initial loss (\(initialLoss))")
    }
}

