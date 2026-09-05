import XCTest
import MLX
@testable import Spiketrans

/// 多層 SNN (層 0 再帰 + 層 1 以降フィードフォワード) の学習側 MLX と推論側 Pure Swift の整合
final class MultiLayerSNNTests: XCTestCase {
    override func setUp() {
        super.setUp()
        Device.setDefault(device: .cpu)
        MLXRandom.seed(42)
    }

    /// 決定的なテスト用特徴量系列 (0〜1)
    private func makeFeatures(frames: Int, dim: Int) -> [[Float]] {
        var seq: [[Float]] = []
        var t = 0
        while t < frames {
            var frame = [Float](repeating: 0.0, count: dim)
            var d = 0
            while d < dim {
                frame[d] = 0.5 + 0.5 * sin(Float(t * 7 + d * 13) * 0.37)
                d += 1
            }
            seq.append(frame)
            t += 1
        }
        return seq
    }

    func testThreeLayerWeightsRoundTripThroughJSON() throws {
        let mlxNet = MLXSpikingNetwork(numLayers: 3, inputDim: 16, maxHiddenDim: 32, outputDim: 10)
        let weights = mlxNet.exportWeights()
        XCTAssertEqual(weights.numLayers, 3)
        XCTAssertEqual(weights.wLayers.count, 2)
        XCTAssertEqual(weights.wLayers[0].count, 32 * 32)
        XCTAssertEqual(weights.gammaRMS[1], [Float](repeating: 1.0, count: 32))

        let data = try JSONEncoder().encode(weights)
        let decoded = try JSONDecoder().decode(SpikingNetworkWeights.self, from: data)
        XCTAssertEqual(decoded, weights)

        let cpuNet = SpikingNetwork(weights: decoded)
        XCTAssertEqual(cpuNet.numLayers, 3)
        XCTAssertEqual(cpuNet.exportWeights(), weights)

        // MLX 側へ戻しても同じ
        let reloaded = MLXSpikingNetwork(weights: decoded)
        XCTAssertEqual(reloaded.exportWeights(), weights)
    }

    func testTwoLayerForwardMatchesBetweenMLXAndPureSwift() {
        let inputDim = 16
        let hidden = 64
        let outputDim = 12
        let frames = 8
        let mlxNet = MLXSpikingNetwork(
            numLayers: 2, inputDim: inputDim, maxHiddenDim: hidden, outputDim: outputDim, timeSteps: 4
        )
        // 層 1 の重みを初期値より大きくして上位層が実際に発火する状態で比べる
        mlxNet.wLayers[0] = mlxNet.wLayers[0] * 8.0
        mlxNet.wIn = mlxNet.wIn * 4.0
        let weights = mlxNet.exportWeights()
        let cpuNet = SpikingNetwork(weights: weights)

        let features = makeFeatures(frames: frames, dim: inputDim)
        var flat: [Float] = []
        for frame in features {
            flat.append(contentsOf: frame)
        }
        let trainer = MLXBPTTTrainer(network: mlxNet, bpttWindow: 4)
        let mlxLogits = trainer.logitsBatch(network: mlxNet, features: MLXArray(flat, [1, frames, inputDim]))
        eval(mlxLogits)
        let mlxFlat = mlxLogits.asArray(Float.self)

        var vPrev = [Float](repeating: 0.0, count: 2 * hidden)
        var sPrev = [Float](repeating: 0.0, count: 2 * hidden)
        var aPrev = [Float](repeating: 0.0, count: 2 * hidden)
        var spikeSum = [Float](repeating: 0.0, count: hidden)
        var logits = [Float](repeating: 0.0, count: outputDim)
        var probs = [Float](repeating: 0.0, count: outputDim)
        let scratch = ForwardScratch(maxHiddenDim: hidden)

        var upperLayerSpikes: Float = 0.0
        var t = 0
        while t < frames {
            cpuNet.forward(
                features: features[t], vPrev: &vPrev, sPrev: &sPrev, aPrev: &aPrev,
                spikeSum: &spikeSum, logits: &logits, probabilities: &probs, scratch: scratch
            )
            var k = 0
            while k < hidden {
                upperLayerSpikes += spikeSum[k]
                k += 1
            }
            var c = 0
            while c < outputDim {
                XCTAssertEqual(logits[c], mlxFlat[t * outputDim + c], accuracy: 1e-3, "frame \(t) class \(c)")
                c += 1
            }
            t += 1
        }
        // 上位層が沈黙していれば一致は自明なので、発火があることも確かめる
        XCTAssertLessThan(0.0, upperLayerSpikes)
    }

    func testTwoLayerCTCTrainingReducesLoss() {
        let inputDim = 16
        let outputDim = 8
        let mlxNet = MLXSpikingNetwork(numLayers: 2, inputDim: inputDim, maxHiddenDim: 64, outputDim: outputDim)
        let trainer = MLXBPTTTrainer(network: mlxNet, config: TrainingConfig(learningRate: 0.01), bpttWindow: 4)

        let batch: [[[Float]]] = [makeFeatures(frames: 24, dim: inputDim), makeFeatures(frames: 20, dim: inputDim)]
        let targets: [[Int]] = [[1, 2, 3, 2], [4, 5, 6]]

        let first = trainer.trainBatchCTC(featuresBatch: batch, targetsBatch: targets)
        XCTAssertFalse(first.isNaN)
        var last = first
        var step = 0
        while step < 30 {
            last = trainer.trainBatchCTC(featuresBatch: batch, targetsBatch: targets)
            step += 1
        }
        XCTAssertFalse(last.isNaN)
        XCTAssertLessThan(last, first)

        // 上位層のパラメータも更新されている
        let after = mlxNet.exportWeights()
        XCTAssertNotEqual(after.gammaRMS[0], [Float](repeating: 1.0, count: 64))
    }
}
