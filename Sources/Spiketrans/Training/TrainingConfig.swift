import Foundation

/// 学習設定パラメータ
public struct TrainingConfig: Sendable {
    public let epochs: Int
    public let learningRate: Float
    public let logInterval: Int
    public let clipNorm: Float

    public init(
        epochs: Int = 50,
        learningRate: Float = 0.005,
        logInterval: Int = 10,
        clipNorm: Float = 5.0
    ) {
        self.epochs = epochs
        self.learningRate = learningRate
        self.logInterval = logInterval
        self.clipNorm = clipNorm
    }
}
