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
        return learningRate(step: epoch, totalSteps: totalEpochs, warmupSteps: warmupEpochs)
    }

    /// 指定ステップ（1-indexed）の学習率を計算。
    ///
    /// データ量が増えて 1 エポックのバッチ数が数万になると、暖機をエポック単位で
    /// 刻むと学習全体に対する割合が大きすぎる (6 エポック中 4 エポックが暖機など)。
    /// バッチ単位で刻めばエポック数に関係なく同じ形の曲線になる。
    public func learningRate(step: Int, totalSteps: Int, warmupSteps: Int) -> Float {
        let warmup = max(1, warmupSteps)
        if step <= warmup {
            let warmupRatio = Float(step) / Float(warmup)
            return lrMin + (lrMax - lrMin) * warmupRatio
        }

        let decaySteps = totalSteps - warmup
        let currentDecay = step - warmup
        let progress = Float(currentDecay) / Float(max(1, decaySteps))
        let clampedProgress = min(1.0, max(0.0, progress))
        let cosDecay = 0.5 * (1.0 + cos(clampedProgress * Float.pi))
        return lrMin + (lrMax - lrMin) * cosDecay
    }
}
