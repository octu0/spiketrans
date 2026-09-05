import XCTest
import MLX
import MLXNN
import MLXOptimizers
@testable import Spiketrans

final class MLXCompileAuditVerificationTests: XCTestCase {
    override func setUp() {
        super.setUp()
        MLXRandom.seed(12345)
    }

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

    /// Phase B.5 & 疑い 1: コンパイルキャッシュが学習後の重み更新を次バッチ / logitsBatch に反映すること
    func testWeightUpdateReflectedInLogitsAndNextBatch() {
        let inDim = 16
        let hidden = 32
        let outDim = 8
        let frames = 32

        let net = MLXSpikingNetwork(
            numLayers: 2,
            inputDim: inDim,
            maxHiddenDim: hidden,
            outputDim: outDim,
            timeSteps: 4
        )
        let trainer = MLXBPTTTrainer(
            network: net,
            config: TrainingConfig(learningRate: 0.05),
            bpttWindow: 4
        )

        let batch = makeSyntheticBatch(batchSize: 2, frames: frames, dim: inDim)
        var flatFeat: [Float] = []
        var b = 0
        while b < 2 {
            var t = 0
            while t < frames {
                flatFeat.append(contentsOf: batch[b][t])
                t += 1
            }
            b += 1
        }
        let featArray = MLXArray(flatFeat, [2, frames, inDim])

        // 1. logitsBatch の事前計算 (compiled)
        let logitsBefore = trainer.logitsBatch(network: net, features: featArray, compiled: true)
        eval(logitsBefore)

        // 2. 1 ステップ学習を実行して重みを更新
        let targets = [[1, 2, 3], [4, 5, 2]]
        let loss1 = trainer.trainBatchCTC(featuresBatch: batch, targetsBatch: targets, compiled: true)
        eval(net.wRec)

        // 3. 学習後の logitsBatch を compiled (キャッシュヒット) と eager で計算
        let logitsAfterCompiled = trainer.logitsBatch(network: net, features: featArray, compiled: true)
        let logitsAfterEager = trainer.logitsBatch(network: net, features: featArray, compiled: false)
        eval(logitsAfterCompiled)
        eval(logitsAfterEager)

        // 重み更新前後のロジット差: 0 ではなく変化していること
        let diffBeforeAfter = abs(logitsBefore - logitsAfterCompiled)
        eval(diffBeforeAfter)
        let maxDiffBeforeAfter = max(diffBeforeAfter).item(Float.self)
        print("Logits diff before vs after update: \(maxDiffBeforeAfter)")
        XCTAssertLessThan(0.001, maxDiffBeforeAfter)

        // compiled (hit) と eager の一致: <= 1e-4
        let diffCompiledEager = abs(logitsAfterCompiled - logitsAfterEager)
        eval(diffCompiledEager)
        let maxDiffCompiledEager = max(diffCompiledEager).item(Float.self)
        print("Logits diff compiled vs eager after update: \(maxDiffCompiledEager)")
        XCTAssertLessThanOrEqual(maxDiffCompiledEager, 1e-4)

        // 4. trainBatchCTC の次ステップ (同一系列長 T=32 でキャッシュヒット) で損失が更新後重みを反映しているか
        let loss2 = trainer.trainBatchCTC(featuresBatch: batch, targetsBatch: targets, compiled: true)
        print("Loss step 1: \(loss1), Loss step 2: \(loss2)")
        XCTAssertFalse(loss2.isNaN)
        XCTAssertFalse(loss2.isInfinite)
        XCTAssertNotEqual(loss1, loss2)
    }

    /// Phase C.1: B が減る最終バッチ相当 (B=8 のあと B=1)、空ラベル除外、maxT パディング境界
    func testAdversarialBatchSizeChangeAndBoundaries() {
        let inDim = 16
        let hidden = 32
        let outDim = 8

        let net = MLXSpikingNetwork(
            numLayers: 2,
            inputDim: inDim,
            maxHiddenDim: hidden,
            outputDim: outDim,
            timeSteps: 4
        )
        let trainer = MLXBPTTTrainer(
            network: net,
            config: TrainingConfig(learningRate: 0.01),
            bpttWindow: 4
        )

        // 1. B=8 のバッチ (T=32)
        let batch8 = makeSyntheticBatch(batchSize: 8, frames: 32, dim: inDim)
        var targets8: [[Int]] = []
        var i = 0
        while i < 8 {
            targets8.append([1, 2, 3])
            i += 1
        }
        let lossB8 = trainer.trainBatchCTC(featuresBatch: batch8, targetsBatch: targets8, compiled: true)
        XCTAssertFalse(lossB8.isNaN)
        XCTAssertLessThan(0.0, lossB8)

        // 2. 直後に B=1 (最終バッチ相当、同一系列長 T=32)
        let batch1 = makeSyntheticBatch(batchSize: 1, frames: 32, dim: inDim)
        let targets1 = [[2, 3, 4]]
        let lossB1 = trainer.trainBatchCTC(featuresBatch: batch1, targetsBatch: targets1, compiled: true)
        XCTAssertFalse(lossB1.isNaN)
        XCTAssertLessThan(0.0, lossB1)

        // 3. 空ラベルが含まれるバッチ (2 サンプルのうち 1 つが空ラベル)
        let batchMixed = makeSyntheticBatch(batchSize: 2, frames: 32, dim: inDim)
        let targetsMixed = [[1, 2], []]
        let lossMixed = trainer.trainBatchCTC(featuresBatch: batchMixed, targetsBatch: targetsMixed, compiled: true)
        XCTAssertFalse(lossMixed.isNaN)
        XCTAssertLessThan(0.0, lossMixed)

        // 全て空ラベルのバッチ -> 0.0 が返ること
        let targetsEmpty: [[Int]] = [[], []]
        let lossEmpty = trainer.trainBatchCTC(featuresBatch: batchMixed, targetsBatch: targetsEmpty, compiled: true)
        XCTAssertEqual(lossEmpty, 0.0)

        // 4. maxT パディング境界: T=31 -> 32, T=32 -> 32, T=33 -> 64, T=64 -> 64
        let batch31 = makeSyntheticBatch(batchSize: 2, frames: 31, dim: inDim)
        let batch32 = makeSyntheticBatch(batchSize: 2, frames: 32, dim: inDim)
        let batch33 = makeSyntheticBatch(batchSize: 2, frames: 33, dim: inDim)
        let batch64 = makeSyntheticBatch(batchSize: 2, frames: 64, dim: inDim)
        let targets2 = [[1, 2], [3, 4]]

        let loss31 = trainer.trainBatchCTC(featuresBatch: batch31, targetsBatch: targets2, compiled: true)
        let loss32 = trainer.trainBatchCTC(featuresBatch: batch32, targetsBatch: targets2, compiled: true)
        let loss33 = trainer.trainBatchCTC(featuresBatch: batch33, targetsBatch: targets2, compiled: true)
        let loss64 = trainer.trainBatchCTC(featuresBatch: batch64, targetsBatch: targets2, compiled: true)

        XCTAssertFalse(loss31.isNaN)
        XCTAssertFalse(loss32.isNaN)
        XCTAssertFalse(loss33.isNaN)
        XCTAssertFalse(loss64.isNaN)
        print("Boundaries loss: T31=\(loss31), T32=\(loss32), T33=\(loss33), T64=\(loss64)")
    }

    /// Phase C.4: 独立 GPU ベンチマーク再測定
    func testIndependentGPUAuditBenchmark() {
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

        // 計測 5 回 (eager)
        var eagerTimes: [Double] = []
        var i = 0
        while i < 5 {
            let t0 = CFAbsoluteTimeGetCurrent()
            _ = trainerEager.trainBatchCTC(featuresBatch: batch, targetsBatch: targets, compiled: false)
            let t1 = CFAbsoluteTimeGetCurrent()
            eagerTimes.append((t1 - t0) * 1000.0)
            i += 1
        }

        // 計測 5 回 (compiled)
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

        let auditBenchReport = """
eager_ms=\(String(format: "%.2f", eagerAvg))
compiled_ms=\(String(format: "%.2f", compiledAvg))
speedup=\(String(format: "%.3f", speedup))x
eager_times_ms=\(eagerTimes.map { String(format: "%.2f", $0) }.joined(separator: ", "))
compiled_times_ms=\(compiledTimes.map { String(format: "%.2f", $0) }.joined(separator: ", "))
"""
        print("=== Audit Benchmark Result ===")
        print(auditBenchReport)

        let auditBenchPath = "/Users/octu0/workspace/spiketrans/.tmp/mlx_compile_audit_bench.txt"
        try? auditBenchReport.write(toFile: auditBenchPath, atomically: true, encoding: .utf8)

        // 要求: speedup >= 1.10 (compiledAvg <= eagerAvg / 1.10)
        XCTAssertLessThanOrEqual(compiledAvg, eagerAvg / 1.10)
    }
}
