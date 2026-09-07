import XCTest
import MLX
@testable import Spiketrans

/// compile() 済みの学習ステップが学習率の変更に追従することを確かめる。
/// 学習率が Float のままだとトレース時の値が焼き込まれ、lr=0 にしても重みが動く
final class MLXCompiledLearningRateTests: XCTestCase {
    private func makeBatch(inputDim: Int, frames: Int) -> ([[[Float]]], [[Int]]) {
        var seq: [[Float]] = []
        var t = 0
        while t < frames {
            var frame = [Float](repeating: 0.0, count: inputDim)
            var d = 0
            while d < inputDim {
                frame[d] = 0.5 + 0.5 * sin(Float(t * 7 + d * 13) * 0.37)
                d += 1
            }
            seq.append(frame)
            t += 1
        }
        return ([seq, seq], [[1, 2, 3, 2], [4, 5, 6]])
    }

    private func maxAbsDiff(_ a: [Float], _ b: [Float]) -> Float {
        var m: Float = 0.0
        var i = 0
        while i < min(a.count, b.count) {
            m = max(m, abs(a[i] - b[i]))
            i += 1
        }
        return m
    }

    /// lr=0 のステップで重みが動かず、lr を戻すと再び動く
    private func assertFollowsLearningRate(compiled: Bool) {
        let inputDim = 16
        let (feats, targets) = makeBatch(inputDim: inputDim, frames: 32)
        let net = MLXSpikingNetwork(numLayers: 2, inputDim: inputDim, maxHiddenDim: 64, outputDim: 8)
        let trainer = MLXBPTTTrainer(network: net, config: TrainingConfig(learningRate: 0.01), bpttWindow: 4)
        trainer.trainBatchCTC(featuresBatch: feats, targetsBatch: targets, compiled: compiled)
        trainer.trainBatchCTC(featuresBatch: feats, targetsBatch: targets, compiled: compiled)

        trainer.updateLearningRate(0.0)
        let frozen = net.exportWeights()
        trainer.trainBatchCTC(featuresBatch: feats, targetsBatch: targets, compiled: compiled)
        let afterZero = net.exportWeights()
        XCTAssertEqual(maxAbsDiff(frozen.wIn, afterZero.wIn), 0.0, "compiled=\(compiled): 学習率 0 で重みが動いた")

        trainer.updateLearningRate(0.01)
        trainer.trainBatchCTC(featuresBatch: feats, targetsBatch: targets, compiled: compiled)
        let afterRestore = net.exportWeights()
        XCTAssertLessThan(0.0, maxAbsDiff(afterZero.wIn, afterRestore.wIn), "compiled=\(compiled): 学習率を戻しても重みが動かない")
    }

    func testEagerStepFollowsLearningRate() {
        assertFollowsLearningRate(compiled: false)
    }

    func testCompiledStepFollowsLearningRate() {
        assertFollowsLearningRate(compiled: true)
    }

    /// compile の上限を超える長い系列は eager に落ちる (コンパイル回数が増えない)
    func testLongSequenceFallsBackToEager() {
        let inputDim = 16
        let (feats, targets) = makeBatch(inputDim: inputDim, frames: compiledMaxFrames + 1)
        let net = MLXSpikingNetwork(numLayers: 1, inputDim: inputDim, maxHiddenDim: 32, outputDim: 8)
        let trainer = MLXBPTTTrainer(network: net, config: TrainingConfig(learningRate: 0.01), bpttWindow: 4)
        let loss = trainer.trainBatchCTC(featuresBatch: feats, targetsBatch: targets, compiled: true)
        XCTAssertFalse(loss.isNaN)
        XCTAssertEqual(trainer.ctcCompileCount, 0)
    }
}
