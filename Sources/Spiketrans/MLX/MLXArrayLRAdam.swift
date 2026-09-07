import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// Adam の 1 次・2 次モーメント
public struct ArrayLRAdamState: Updatable {
    public let m: MLXArray
    public let v: MLXArray

    public func innerState() -> [MLXArray] {
        return [m, v]
    }
}

/// 学習率を MLXArray で持つ Adam (バイアス補正なし。更新式は mlx-swift の Adam と同じ)。
///
/// compile() した学習ステップに学習率を入力配列として渡すためのもの。mlx-swift の
/// Adam は学習率が Float で、compile のトレース時の値がグラフに焼き込まれるため、
/// キャッシュした関数を使い続けるとコサインスケジュールが効かなくなる。
/// OptimizerBase は他モジュールから継承できない (初期化子が internal) ので Optimizer を直接実装する
public final class ArrayLRAdam: Optimizer {
    public var learningRate: MLXArray
    public let betas: (Float, Float)
    public let eps: Float

    private var stateStorage = NestedDictionary<String, ArrayLRAdamState>()

    public init(learningRate: Float, betas: (Float, Float) = (0.9, 0.999), eps: Float = 1e-8) {
        self.learningRate = MLXArray(learningRate)
        self.betas = betas
        self.eps = eps
    }

    public func innerState() -> [MLXArray] {
        return stateStorage.flattenedValues().flatMap { $0.innerState() }
    }

    public func update(model: Module, gradients: ModuleParameters) {
        let (b1, b2) = betas
        let lr = learningRate
        let epsValue = eps
        let (newParameters, newStates) = gradients.mapValues(model.parameters(), stateStorage) {
            (gradient: MLXArray, parameter: MLXArray?, state: ArrayLRAdamState?) -> (MLXArray, ArrayLRAdamState?) in
            let p = parameter!
            let mPrev: MLXArray
            let vPrev: MLXArray
            switch state {
            case .some(let s):
                mPrev = s.m
                vPrev = s.v
            case .none:
                mPrev = MLXArray.zeros(like: p)
                vPrev = MLXArray.zeros(like: p)
            }
            let m = b1 * mPrev + (1 - b1) * gradient
            let v = b2 * vPrev + (1 - b2) * square(gradient)
            let update = lr * m / (sqrt(v) + epsValue)
            return (p - update, ArrayLRAdamState(m: m, v: v))
        }
        stateStorage = newStates
        model.update(parameters: newParameters)
    }
}
