import XCTest
import MLX
import MLXNN
import MLXOptimizers
@testable import Spiketrans

final class MLXCompileBPTTTests: XCTestCase {
    override func setUp() {
        super.setUp()
        MLXRandom.seed(42)
    }

    /// 決定的なテスト用合成特徴量バッチ [B, T, D]
    private func makeSyntheticBatch(
        batchSize: Int,
        frames: Int,
        dim: Int
    ) -> [[[Float]]] {
        var batch: [[[Float]]] = []
        var b = 0
        while b < batchSize {
            var seq: [[Float]] = []
            var t = 0
            while t < frames {
                var vec = [Float](repeating: 0.0, count: dim)
                var d = 0
                while d < dim {
                    vec[d] = 0.5 + 0.5 * sin(Float((b * 100) + (t * 7) + (d * 13)) * 0.37)
                    d += 1
                }
                seq.append(vec)
                t += 1
            }
            batch.append(seq)
            b += 1
        }
        return batch
    }

    /// 4.1 & 4.7 (1): 2 層における eager vs compile の logitsBatch 一致 (誤差 <= 1e-4)
    func testCompileVsEagerLogitsMatchTwoLayers() {
        let inputDim = 16
        let hidden = 32
        let outputDim = 8
        let frames = 32
        let batchSize = 2

        let net = MLXSpikingNetwork(
            numLayers: 2,
            inputDim: inputDim,
            maxHiddenDim: hidden,
            outputDim: outputDim,
            timeSteps: 4
        )
        let trainer = MLXBPTTTrainer(network: net, bpttWindow: 4)

        let batch = makeSyntheticBatch(batchSize: batchSize, frames: frames, dim: inputDim)
        var flatFeat: [Float] = []
        var b = 0
        while b < batchSize {
            var t = 0
            while t < frames {
                flatFeat.append(contentsOf: batch[b][t])
                t += 1
            }
            b += 1
        }
        let featArray = MLXArray(flatFeat, [batchSize, frames, inputDim])

        let logitsEager = trainer.logitsBatch(network: net, features: featArray, compiled: false)
        eval(logitsEager)

        let logitsCompiled = trainer.logitsBatch(network: net, features: featArray, compiled: true)
        eval(logitsCompiled)

        let diff = abs(logitsEager - logitsCompiled)
        eval(diff)
        let maxDiff = max(diff).item(Float.self)
        print("Logits max absolute diff (eager vs compiled): \(maxDiff)")
        XCTAssertLessThanOrEqual(maxDiff, 1e-4)
    }

    /// 4.2 & 4.7 (2): 同一特徴量でラベル列のみ変えた 2 バッチで損失が変わること (定数キャプチャ防止)
    func testLabelsChangeLossWithSameFeatures() {
        let inputDim = 16
        let hidden = 32
        let outputDim = 8
        let frames = 32

        let net = MLXSpikingNetwork(
            numLayers: 2,
            inputDim: inputDim,
            maxHiddenDim: hidden,
            outputDim: outputDim,
            timeSteps: 4
        )
        let trainer = MLXBPTTTrainer(network: net, bpttWindow: 4)

        let batch = makeSyntheticBatch(batchSize: 2, frames: frames, dim: inputDim)
        let targets1 = [[1, 2, 3], [4, 5, 2]]
        let targets2 = [[6, 7, 1], [3, 2, 5]]

        let loss1 = trainer.trainBatchCTC(featuresBatch: batch, targetsBatch: targets1, compiled: true)
        let loss2 = trainer.trainBatchCTC(featuresBatch: batch, targetsBatch: targets2, compiled: true)

        print("Label sensitivity loss1=\(loss1), loss2=\(loss2)")
        XCTAssertFalse(loss1.isNaN)
        XCTAssertFalse(loss2.isNaN)
        XCTAssertNotEqual(loss1, loss2)
    }

    /// 4.3 & 4.7 (3): 系列長バケット (T=32 と T=64 の混在) で NaN にならず、2 回目の T=32 で cache hit すること
    func testBucketMixingAndCacheHit() {
        let inputDim = 16
        let hidden = 32
        let outputDim = 8

        let net = MLXSpikingNetwork(
            numLayers: 2,
            inputDim: inputDim,
            maxHiddenDim: hidden,
            outputDim: outputDim,
            timeSteps: 4
        )
        let trainer = MLXBPTTTrainer(network: net, bpttWindow: 4)

        let batch32 = makeSyntheticBatch(batchSize: 2, frames: 32, dim: inputDim)
        let batch64 = makeSyntheticBatch(batchSize: 2, frames: 64, dim: inputDim)
        let targets = [[1, 2, 3], [4, 5]]

        // 1 回目: T=32 (compile count: 1, hit count: 0)
        let loss32First = trainer.trainBatchCTC(featuresBatch: batch32, targetsBatch: targets, compiled: true)
        XCTAssertFalse(loss32First.isNaN)
        XCTAssertFalse(loss32First.isInfinite)
        XCTAssertEqual(trainer.ctcCompileCount, 1)
        XCTAssertEqual(trainer.ctcCacheHitCount, 0)

        // 2 回目: T=64 (compile count: 2, hit count: 0)
        let loss64 = trainer.trainBatchCTC(featuresBatch: batch64, targetsBatch: targets, compiled: true)
        XCTAssertFalse(loss64.isNaN)
        XCTAssertFalse(loss64.isInfinite)
        XCTAssertEqual(trainer.ctcCompileCount, 2)
        XCTAssertEqual(trainer.ctcCacheHitCount, 0)

        // 3 回目: 再び T=32 (compile count: 2, hit count: 1 -> cache hit 利用)
        let loss32Second = trainer.trainBatchCTC(featuresBatch: batch32, targetsBatch: targets, compiled: true)
        XCTAssertFalse(loss32Second.isNaN)
        XCTAssertFalse(loss32Second.isInfinite)
        XCTAssertEqual(trainer.ctcCompileCount, 2)
        XCTAssertEqual(trainer.ctcCacheHitCount, 1)
        print("Bucket test passed: hits=\(trainer.ctcCacheHitCount), compiles=\(trainer.ctcCompileCount)")
    }

    /// 4.7 (4): 1 層でも 2 層でも CTC 1 ステップが有限損失を返すこと
    func testSingleAndMultiLayerCTCFiniteLoss() {
        let inputDim = 16
        let hidden = 32
        let outputDim = 8
        let frames = 32

        // 1 層 SNN
        let netL1 = MLXSpikingNetwork(
            numLayers: 1,
            inputDim: inputDim,
            maxHiddenDim: hidden,
            outputDim: outputDim,
            timeSteps: 4
        )
        let trainerL1 = MLXBPTTTrainer(network: netL1, bpttWindow: 4)
        let batchL1 = makeSyntheticBatch(batchSize: 2, frames: frames, dim: inputDim)
        let targetsL1 = [[1, 2], [3, 4]]
        let lossL1 = trainerL1.trainBatchCTC(featuresBatch: batchL1, targetsBatch: targetsL1, compiled: true)
        XCTAssertFalse(lossL1.isNaN)
        XCTAssertFalse(lossL1.isInfinite)
        XCTAssertLessThan(0.0, lossL1)

        // 2 層 SNN
        let netL2 = MLXSpikingNetwork(
            numLayers: 2,
            inputDim: inputDim,
            maxHiddenDim: hidden,
            outputDim: outputDim,
            timeSteps: 4
        )
        let trainerL2 = MLXBPTTTrainer(network: netL2, bpttWindow: 4)
        let batchL2 = makeSyntheticBatch(batchSize: 2, frames: frames, dim: inputDim)
        let targetsL2 = [[1, 2], [3, 4]]
        let lossL2 = trainerL2.trainBatchCTC(featuresBatch: batchL2, targetsBatch: targetsL2, compiled: true)
        XCTAssertFalse(lossL2.isNaN)
        XCTAssertFalse(lossL2.isInfinite)
        XCTAssertLessThan(0.0, lossL2)

        print("Finite loss: L1=\(lossL1), L2=\(lossL2)")
    }

    /// 4.4: 連続 20 ステップの学習と clearCache() の安全性
    func testContinuousTwentyStepsAndClearCache() {
        let inputDim = 16
        let hidden = 64
        let outputDim = 12
        let frames = 64
        let batchSize = 8

        let net = MLXSpikingNetwork(
            numLayers: 2,
            inputDim: inputDim,
            maxHiddenDim: hidden,
            outputDim: outputDim,
            timeSteps: 4
        )
        let trainer = MLXBPTTTrainer(
            network: net,
            config: TrainingConfig(learningRate: 0.005),
            bpttWindow: 4
        )

        let batch = makeSyntheticBatch(batchSize: batchSize, frames: frames, dim: inputDim)
        var targets: [[Int]] = []
        var b = 0
        while b < batchSize {
            targets.append([1, 2, 3, 4])
            b += 1
        }

        var initialLoss: Float = 0.0
        var currentLoss: Float = 0.0
        var step = 0
        while step < 20 {
            currentLoss = trainer.trainBatchCTC(featuresBatch: batch, targetsBatch: targets, compiled: true)
            XCTAssertFalse(currentLoss.isNaN)
            XCTAssertFalse(currentLoss.isInfinite)

            if step == 0 {
                initialLoss = currentLoss
            }
            if step == 10 {
                // 10 ステップ目でキャッシュクリアを呼んでも落ちないこと
                Memory.clearCache()
            }
            step += 1
        }

        print("20 steps result: initial=\(initialLoss), final=\(currentLoss)")
        XCTAssertLessThan(currentLoss, initialLoss)
    }

    /// 4.1: 5 ステップ連続学習での損失列および wRec の数値一致
    func testEagerVsCompiledFiveStepsNumerics() {
        let inputDim = 16
        let hidden = 32
        let outputDim = 8
        let frames = 32
        let bpttWindow = 4
        let timeSteps = 4
        let numLayers = 2

        let netEager = MLXSpikingNetwork(
            numLayers: numLayers,
            inputDim: inputDim,
            maxHiddenDim: hidden,
            outputDim: outputDim,
            timeSteps: timeSteps
        )
        let weights = netEager.exportWeights()
        let netCompiled = MLXSpikingNetwork(weights: weights)

        let trainerEager = MLXBPTTTrainer(
            network: netEager,
            config: TrainingConfig(learningRate: 0.01),
            bpttWindow: bpttWindow
        )
        let trainerCompiled = MLXBPTTTrainer(
            network: netCompiled,
            config: TrainingConfig(learningRate: 0.01),
            bpttWindow: bpttWindow
        )

        let batch = makeSyntheticBatch(batchSize: 2, frames: frames, dim: inputDim)
        let targets = [[1, 2, 3], [4, 5, 2]]

        var step = 0
        while step < 5 {
            let lossEager = trainerEager.trainBatchCTC(featuresBatch: batch, targetsBatch: targets, compiled: false)
            let lossCompiled = trainerCompiled.trainBatchCTC(featuresBatch: batch, targetsBatch: targets, compiled: true)

            let diffLoss = abs(lossEager - lossCompiled)
            print("Step \(step): eager=\(lossEager), compiled=\(lossCompiled), diff=\(diffLoss)")
            XCTAssertLessThanOrEqual(diffLoss, 1e-3)
            step += 1
        }

        let diffWRec = abs(netEager.wRec - netCompiled.wRec)
        eval(diffWRec)
        let maxDiffWRec = max(diffWRec).item(Float.self)
        print("Max absolute difference in wRec after 5 steps: \(maxDiffWRec)")
        XCTAssertLessThanOrEqual(maxDiffWRec, 1e-4)
    }

    /// 4.5: GPU 上での Release 速度ベンチマーク (10% 以上高速化の検証)
    func testGPUSpeedupBenchmark() {
        Device.setDefault(device: .gpu)

        let bSize = 8
        let inDim = 512
        let hidden = 1024
        let outDim = 85
        let frames = 128
        let timeSteps = 4
        let bpttWindow = 4
        let numLayers = 2

        let netEager = MLXSpikingNetwork(
            numLayers: numLayers,
            inputDim: inDim,
            maxHiddenDim: hidden,
            outputDim: outDim,
            timeSteps: timeSteps
        )
        let weights = netEager.exportWeights()
        let netCompiled = MLXSpikingNetwork(weights: weights)

        let trainerEager = MLXBPTTTrainer(
            network: netEager,
            config: TrainingConfig(learningRate: 0.001),
            bpttWindow: bpttWindow
        )
        let trainerCompiled = MLXBPTTTrainer(
            network: netCompiled,
            config: TrainingConfig(learningRate: 0.001),
            bpttWindow: bpttWindow
        )

        let batch = makeSyntheticBatch(batchSize: bSize, frames: frames, dim: inDim)
        var targets: [[Int]] = []
        var b = 0
        while b < bSize {
            var tgt: [Int] = []
            var l = 0
            while l < 20 {
                tgt.append(((b + l) % (outDim - 1)) + 1)
                l += 1
            }
            targets.append(tgt)
            b += 1
        }

        // Warmup 2 回
        var w = 0
        while w < 2 {
            _ = trainerEager.trainBatchCTC(featuresBatch: batch, targetsBatch: targets, compiled: false)
            _ = trainerCompiled.trainBatchCTC(featuresBatch: batch, targetsBatch: targets, compiled: true)
            w += 1
        }

        // 5 回計測 (eager)
        var eagerTimes: [Double] = []
        var i = 0
        while i < 5 {
            let t0 = CFAbsoluteTimeGetCurrent()
            _ = trainerEager.trainBatchCTC(featuresBatch: batch, targetsBatch: targets, compiled: false)
            let t1 = CFAbsoluteTimeGetCurrent()
            eagerTimes.append((t1 - t0) * 1000.0)
            i += 1
        }

        // 5 回計測 (compiled)
        var compiledTimes: [Double] = []
        i = 0
        while i < 5 {
            let t0 = CFAbsoluteTimeGetCurrent()
            _ = trainerCompiled.trainBatchCTC(featuresBatch: batch, targetsBatch: targets, compiled: true)
            let t1 = CFAbsoluteTimeGetCurrent()
            compiledTimes.append((t1 - t0) * 1000.0)
            i += 1
        }

        let eagerAvg = eagerTimes.reduce(0.0, +) / Double(eagerTimes.count)
        let compiledAvg = compiledTimes.reduce(0.0, +) / Double(compiledTimes.count)
        let speedup = eagerAvg / compiledAvg

        let benchReport = """
eager_ms=\(String(format: "%.2f", eagerAvg))
compiled_ms=\(String(format: "%.2f", compiledAvg))
speedup=\(String(format: "%.3f", speedup))x
eager_times_ms=\(eagerTimes.map { String(format: "%.2f", $0) }.joined(separator: ", "))
compiled_times_ms=\(compiledTimes.map { String(format: "%.2f", $0) }.joined(separator: ", "))
"""
        print("=== Benchmark Result ===")
        print(benchReport)

        let benchPath = "/Users/octu0/workspace/spiketrans/.tmp/mlx_compile_bench.txt"
        try? benchReport.write(toFile: benchPath, atomically: true, encoding: .utf8)

        // 10% 以上高速化 (compiledAvg <= eagerAvg * 0.90)
        XCTAssertLessThanOrEqual(compiledAvg, eagerAvg * 0.90)
    }
}
