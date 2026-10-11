import XCTest
import MLX
@testable import Spiketrans

/// Hawk 型音響モデル (SNN との比較用) の学習側 MLX と推論側 CPU の整合。
/// 線形の再帰は Metal カーネルなので GPU で動かす
final class HawkTests: XCTestCase {
    override func setUp() {
        super.setUp()
        Device.setDefault(device: .gpu)
        MLXRandom.seed(7)
    }

    private func makeFeatures(frames: Int, dim: Int, phase: Int = 0) -> [[Float]] {
        var seq: [[Float]] = []
        var t = 0
        while t < frames {
            var frame = [Float](repeating: 0.0, count: dim)
            var d = 0
            while d < dim {
                frame[d] = sin(Float((t + phase) * 7 + d * 13) * 0.37)
                d += 1
            }
            seq.append(frame)
            t += 1
        }
        return seq
    }

    private func smallNetwork() -> MLXHawkNetwork {
        return MLXHawkNetwork(inputDim: 12, modelDim: 16, rnnDim: 24, mlpDim: 32, heads: 3, convWidth: 4, numLayers: 2, outputDim: 7)
    }

    /// 時間ループで解いた参照 (計算グラフに展開するので自動微分できる)
    private func sequentialScan(_ a: MLXArray, _ x: MLXArray) -> MLXArray {
        var h = MLXArray.zeros([a.shape[0], a.shape[2]])
        var outs: [MLXArray] = []
        var t = 0
        while t < a.shape[1] {
            h = a[0..., t, 0...] * h + x[0..., t, 0...]
            outs.append(h)
            t += 1
        }
        return stacked(outs, axis: 1)
    }

    private func assertClose(_ a: MLXArray, _ b: MLXArray, _ tolerance: Float, file: StaticString = #filePath, line: UInt = #line) {
        eval(a, b)
        let x = a.asArray(Float.self)
        let y = b.asArray(Float.self)
        XCTAssertEqual(x.count, y.count, file: file, line: line)
        var worst: Float = 0.0
        var i = 0
        while i < x.count {
            worst = max(worst, abs(x[i] - y[i]))
            i += 1
        }
        XCTAssertLessThan(worst, tolerance, file: file, line: line)
    }

    func testLinearScanMatchesSequentialForwardAndGradient() {
        let a = MLXRandom.uniform(low: 0.2, high: 0.99, [3, 41, 10])
        let x = MLXRandom.normal([3, 41, 10])
        let weight = MLXRandom.normal([3, 41, 10])
        assertClose(MLXLinearScan.scan(a, x), sequentialScan(a, x), 1e-4)

        let kernelGrad = grad({ (arrays: [MLXArray]) -> MLXArray in
            sum(MLXLinearScan.scan(arrays[0], arrays[1]) * weight)
        })([a, x])
        let referenceGrad = grad({ (arrays: [MLXArray]) -> MLXArray in
            sum(self.sequentialScan(arrays[0], arrays[1]) * weight)
        })([a, x])
        assertClose(kernelGrad[0], referenceGrad[0], 1e-3)
        assertClose(kernelGrad[1], referenceGrad[1], 1e-3)
    }

    /// CPU 推論 (チャンクをまたいで状態を引き継ぐ) が MLX の一括の前向きと一致する
    func testForwardMatchesBetweenMLXAndCPU() {
        let net = smallNetwork()
        // 学習で動く値を初期値から離しておく (ゲートの偏り・畳み込みの偏り・正規化のゲイン)
        let start = net.exportWeights()
        let shifted = HawkWeights(
            inputDim: start.inputDim, modelDim: start.modelDim, rnnDim: start.rnnDim, mlpDim: start.mlpDim,
            heads: start.heads, convWidth: start.convWidth, outputDim: start.outputDim, wIn: start.wIn, bIn: start.bIn,
            layers: start.layers.map { l in
                HawkLayerWeights(
                    norm1: l.norm1.map { $0 * 1.3 }, wGate: l.wGate, wX: l.wX, convW: l.convW, convB: l.convB.map { _ in 0.1 },
                    wA: l.wA, bA: l.bA.map { _ in 0.4 }, wI: l.wI, bI: l.bI.map { _ in -0.3 }, lambda: l.lambda, wO: l.wO,
                    norm2: l.norm2, wMLPGate: l.wMLPGate, wMLPUp: l.wMLPUp, wMLPDown: l.wMLPDown
                )
            },
            finalNorm: start.finalNorm, wOut: start.wOut, bOut: start.bOut.map { _ in 0.05 }
        )
        net.importWeights(from: shifted)
        let frames = 37
        let features = makeFeatures(frames: frames, dim: 12)
        let logits = net(MLXArray(features.flatMap { $0 }, [1, frames, 12]))
        let expected = logits - logSumExp(logits, axis: -1, keepDims: true)

        let cpu = HawkNetwork(weights: net.exportWeights())
        let got = cpu.logProbs(features: features, chunkFrames: 5)
        XCTAssertEqual(got.count, frames)
        assertClose(MLXArray(got.flatMap { $0 }, [1, frames, 7]), expected, 2e-4)
    }

    func testWeightsRoundTripThroughFile() throws {
        let net = smallNetwork()
        let weights = net.exportWeights()
        XCTAssertEqual(weights.parameterCount, net.parameters().flattened().reduce(0) { $0 + $1.1.size })
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("hawk_roundtrip_\(UUID().uuidString).json")
        try weights.save(to: url)
        let loaded = try HawkWeights.load(from: url)
        XCTAssertEqual(loaded, weights)
        let other = MLXHawkNetwork(weights: loaded)
        XCTAssertEqual(other.exportWeights(), weights)
    }

    /// 無音 (特徴量がすべて 0) のフレームが続いても、8 層を通した勾配があふれない
    func testGradientStaysFiniteThroughSilentFrames() {
        let net = MLXHawkNetwork(inputDim: 12, modelDim: 32, rnnDim: 32, mlpDim: 48, heads: 4, convWidth: 4, numLayers: 8, outputDim: 7)
        var features = makeFeatures(frames: 60, dim: 12)
        var t = 20
        while t < 40 {
            features[t] = [Float](repeating: 0.0, count: 12)
            t += 1
        }
        let input = MLXArray(features.flatMap { $0 }, [1, 60, 12])
        let g = grad({ (f: MLXArray) -> MLXArray in
            let logits = net(f)
            return sum(logits * logits)
        })(input)
        eval(g)
        XCTAssertFalse(any(isNaN(g) .|| isInf(g)).item(Bool.self))
        XCTAssertLessThan(abs(g).max().item(Float.self), 1e6)
    }

    func testCTCTrainingReducesLoss() {
        let net = smallNetwork()
        let before = net.exportWeights()
        let trainer = MLXHawkTrainer(network: net, learningRate: 0.01)
        let batch: [[[Float]]] = [makeFeatures(frames: 24, dim: 12), makeFeatures(frames: 20, dim: 12, phase: 5)]
        let targets: [[Int]] = [[1, 2, 3, 2], [4, 5, 6]]
        let first = trainer.trainBatchCTC(featuresBatch: batch, targetsBatch: targets)
        var last = first
        var step = 0
        while step < 40 {
            last = trainer.trainBatchCTC(featuresBatch: batch, targetsBatch: targets)
            step += 1
        }
        XCTAssertFalse(last.isNaN)
        XCTAssertLessThan(last, first * 0.5)
        let after = net.exportWeights()
        XCTAssertNotEqual(after.layers[1].lambda, before.layers[1].lambda)
        XCTAssertNotEqual(after.layers[0].wA, before.layers[0].wA)
    }
}
