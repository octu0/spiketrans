import Foundation

/// Cosine Annealing 学習率スケジューラ
public struct CosineLRScheduler: Sendable {
    public let lrMax: Float
    public let lrMin: Float
    public let totalEpochs: Int
    public let warmupEpochs: Int

    public init(
        lrMax: Float = 0.025,
        lrMin: Float = 0.002,
        totalEpochs: Int = 60,
        warmupEpochs: Int = 3
    ) {
        self.lrMax = lrMax
        self.lrMin = lrMin
        self.totalEpochs = totalEpochs
        self.warmupEpochs = warmupEpochs
    }

    /// 指定エポック（1-indexed）の学習率を計算
    public func learningRate(forEpoch epoch: Int) -> Float {
        if epoch <= warmupEpochs {
            let warmupRatio = Float(epoch) / Float(max(1, warmupEpochs))
            return lrMin + (lrMax - lrMin) * warmupRatio
        }

        let decayEpochs = totalEpochs - warmupEpochs
        let currentDecay = epoch - warmupEpochs
        let progress = Float(currentDecay) / Float(max(1, decayEpochs))
        let clampedProgress = min(1.0, max(0.0, progress))
        let cosDecay = 0.5 * (1.0 + cos(clampedProgress * Float.pi))
        return lrMin + (lrMax - lrMin) * cosDecay
    }
}
