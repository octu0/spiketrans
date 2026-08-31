import Foundation

/// Adam オプティマイザ設定
public struct AdamConfig: Sendable, Equatable {
    public let lr: Float
    public let beta1: Float
    public let beta2: Float
    public let eps: Float
    public let gradClip: Float

    public init(
        lr: Float = 0.01,
        beta1: Float = 0.9,
        beta2: Float = 0.999,
        eps: Float = 1e-8,
        gradClip: Float = 1.0
    ) {
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.gradClip = gradClip
    }
}

/// 学習対象パラメータとその勾配・モーメントバッファ
public final class Parameter: @unchecked Sendable {
    public var data: [Float]
    public var grad: [Float]
    public var m: [Float]
    public var v: [Float]
    public let count: Int

    public init(count: Int, initialData: [Float]? = nil) {
        self.count = count
        if let initD = initialData {
            self.data = initD
        } else {
            self.data = [Float](repeating: 0.0, count: count)
        }
        self.grad = [Float](repeating: 0.0, count: count)
        self.m = [Float](repeating: 0.0, count: count)
        self.v = [Float](repeating: 0.0, count: count)
    }

    @inline(__always)
    public func zeroGrad() {
        var i = 0
        while i < count {
            grad[i] = 0.0
            i += 1
        }
    }
}

/// Adam オプティマイザ本体
public final class AdamOptimizer: @unchecked Sendable {
    public let config: AdamConfig
    public let parameters: [Parameter]
    public private(set) var stepCount: Int

    public init(config: AdamConfig, parameters: [Parameter]) {
        self.config = config
        self.parameters = parameters
        self.stepCount = 0
    }

    public func step() {
        stepCount += 1
        let t = Float(stepCount)

        // 1. パラメータごとの L2 ノルムクリッピング
        if 0.0 < config.gradClip {
            for param in parameters {
                if 0 < param.count {
                    var normSq: Float = 0.0
                    param.grad.withUnsafeBufferPointer { ptr in
                        normSq = VectorOperations.sumOfSquares(ptr: ptr.baseAddress!, count: param.count)
                    }
                    let norm = sqrt(normSq)
                    if config.gradClip < norm {
                        let scale = config.gradClip / (norm + 1e-6)
                        var i = 0
                        while i < param.count {
                            param.grad[i] *= scale
                            i += 1
                        }
                    }
                }
            }
        }

        // 2. バイアス補正係数
        let beta1Correction = 1.0 - pow(config.beta1, t)
        let beta2Correction = 1.0 - pow(config.beta2, t)

        // 3. モーメント更新およびパラメータ更新
        for param in parameters {
            var i = 0
            let n = param.count
            while i < n {
                let g = param.grad[i]
                param.m[i] = config.beta1 * param.m[i] + (1.0 - config.beta1) * g
                param.v[i] = config.beta2 * param.v[i] + (1.0 - config.beta2) * g * g

                let mHat = param.m[i] / beta1Correction
                let vHat = param.v[i] / beta2Correction

                let update = (config.lr * mHat) / (sqrt(vHat) + config.eps)
                param.data[i] -= update
                i += 1
            }
        }
    }

    public func zeroGrad() {
        for param in parameters {
            param.zeroGrad()
        }
    }
}
