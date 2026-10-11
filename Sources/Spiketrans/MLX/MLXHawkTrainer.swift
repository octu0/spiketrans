import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// Hawk 型音響モデルの CTC 学習。RG-LRU は線形の再帰なので、SNN と違って系列全体を切り詰めずに逆伝播する
public final class MLXHawkTrainer: @unchecked Sendable {
    public let network: MLXHawkNetwork
    public let optimizer: ArrayLRAdam

    /// 系列長 (32 の倍数に切り上げたフレーム数) をキーとするコンパイル済みの学習ステップ
    /// 直近のバッチの勾配ノルム (クリップ前)
    public private(set) var lastGradientNorm: Float = 0.0

    private var compiledSteps: [Int: ([MLXArray]) -> [MLXArray]] = [:]

    public init(network: MLXHawkNetwork, learningRate: Float) {
        self.network = network
        self.optimizer = ArrayLRAdam(learningRate: learningRate)
        MLX.GPU.set(cacheLimit: MLXBPTTTrainer.cacheLimitBytes)
    }

    public func updateLearningRate(_ lr: Float) {
        optimizer.learningRate = MLXArray(lr)
    }

    /// 重みを保存済みの状態へ巻き戻し、Adam のモーメントも捨てる
    public func rollback(to weights: HawkWeights) {
        network.importWeights(from: weights)
        optimizer.resetState()
        compiledSteps.removeAll()
        eval(network, optimizer)
    }

    /// 推論用のロジット [B, T, outputDim]
    public func logits(features: MLXArray) -> MLXArray {
        return network(features)
    }

    /// 1 バッチの CTC 学習。戻り値は損失
    public func trainBatchCTC(
        featuresBatch: [[[Float]]],
        targetsBatch: [[Int]],
        blankId: Int = 0,
        compiled: Bool = true
    ) -> Float {
        // ラベルが空のサンプルは CTC を定義できないので除外する
        var validFeatures: [[[Float]]] = []
        var validTargets: [[Int]] = []
        var frameCounts: [Int] = []
        var maxT = 0
        var b = 0
        while b < featuresBatch.count {
            let frames = featuresBatch[b].count
            if 0 < frames && targetsBatch[b].isEmpty != true {
                validFeatures.append(featuresBatch[b])
                validTargets.append(targetsBatch[b])
                frameCounts.append(frames)
                maxT = max(maxT, frames)
            }
            b += 1
        }
        if validFeatures.isEmpty {
            return 0.0
        }
        // 系列長を 32 の倍数へ切り上げ、形の種類を減らす (増えたフレームは inputLengths で凍結される)
        maxT = ((maxT + 31) / 32) * 32
        let count = validFeatures.count
        let inDim = network.inputDim
        var flat = [Float](repeating: 0.0, count: count * maxT * inDim)
        b = 0
        while b < count {
            var t = 0
            while t < validFeatures[b].count {
                let offset = (b * maxT + t) * inDim
                let row = validFeatures[b][t]
                var d = 0
                while d < inDim {
                    flat[offset + d] = row[d]
                    d += 1
                }
                t += 1
            }
            b += 1
        }
        let features = MLXArray(flat, [count, maxT, inDim])
        let targets = MLXCTCLoss.ExtendedTargets(targetsBatch: validTargets, frameCounts: frameCounts, blankId: blankId)

        let lossFn = { (model: MLXHawkNetwork, arrays: [MLXArray]) -> [MLXArray] in
            let ext = MLXCTCLoss.ExtendedTargets(
                extTargets: arrays[1],
                skipMask: arrays[2],
                validMask: arrays[3],
                finalIndex1: arrays[4],
                finalIndex2: arrays[5],
                hasSecondFinal: arrays[6],
                inputLengths: arrays[7]
            )
            return [MLXCTCLoss.loss(logits: model(arrays[0]), targets: ext)]
        }
        let lg = valueAndGrad(model: network, lossFn)
        // 学習率は最後の入力配列。トレース時に差し替え、以後の呼び出しでは渡した値が更新式に使われる
        func step(arrays: [MLXArray]) -> [MLXArray] {
            let (lossValues, grads) = lg(self.network, Array(arrays[0..<8]))
            let (clipped, norm) = clipGradNorm(gradients: grads, maxNorm: 5.0)
            let outer = self.optimizer.learningRate
            self.optimizer.learningRate = arrays[8]
            self.optimizer.update(model: self.network, gradients: clipped)
            self.optimizer.learningRate = outer
            return lossValues + [norm]
        }
        var stepFn: ([MLXArray]) -> [MLXArray] = step
        if compiled {
            switch compiledSteps[maxT] {
            case .some(let cached):
                stepFn = cached
            case .none:
                let fn = compile(inputs: [network, optimizer], outputs: [network, optimizer], step)
                compiledSteps[maxT] = fn
                stepFn = fn
            }
        }
        let lossValues = stepFn([features] + targets.toArrays() + [optimizer.learningRate])
        eval(network, optimizer, lossValues)
        lastGradientNorm = lossValues[1].item(Float.self)
        return lossValues[0].item(Float.self)
    }
}
