import Foundation
import MLX
import MLXNN

/// MLX (Apple Silicon GPU) による軽量 2D-Convolutional フロントエンド
public final class MLXConv2DSubsampling: Module, @unchecked Sendable {
    public let inChannels: Int
    public let outChannels1: Int
    public let outChannels2: Int
    public let melChannels: Int
    public let outputDim: Int

    public var conv1Weight: MLXArray  // [outChannels1, 3, 3, inChannels]
    public var conv1Bias: MLXArray    // [outChannels1]
    public var conv2Weight: MLXArray  // [outChannels2, 3, 3, outChannels1]
    public var conv2Bias: MLXArray    // [outChannels2]
    public var projWeight: MLXArray   // [outputDim, outChannels2 * 16]
    public var projBias: MLXArray     // [outputDim]

    public init(
        inChannels: Int = 1,
        outChannels1: Int = 16,
        outChannels2: Int = 16,
        melChannels: Int = 64,
        outputDim: Int = 128
    ) {
        self.inChannels = inChannels
        self.outChannels1 = outChannels1
        self.outChannels2 = outChannels2
        self.melChannels = melChannels
        self.outputDim = outputDim

        let scale1 = sqrt(2.0 / Float(inChannels * 3 * 3))
        self.conv1Weight = MLXRandom.uniform(
            low: -scale1, high: scale1,
            [outChannels1, 3, 3, inChannels]
        )
        self.conv1Bias = MLXArray.zeros([outChannels1])

        let scale2 = sqrt(2.0 / Float(outChannels1 * 3 * 3))
        self.conv2Weight = MLXRandom.uniform(
            low: -scale2, high: scale2,
            [outChannels2, 3, 3, outChannels1]
        )
        self.conv2Bias = MLXArray.zeros([outChannels2])

        let flatDim = outChannels2 * 16
        let scaleProj = sqrt(2.0 / Float(flatDim))
        self.projWeight = MLXRandom.uniform(
            low: -scaleProj, high: scaleProj,
            [outputDim, flatDim]
        )
        self.projBias = MLXArray.zeros([outputDim])

        super.init()
    }

    public init(weights: Conv2DSubsamplingWeights) {
        self.inChannels = weights.inChannels
        self.outChannels1 = weights.outChannels1
        self.outChannels2 = weights.outChannels2
        self.melChannels = weights.melChannels
        self.outputDim = weights.outputDim

        self.conv1Weight = MLXArray(weights.conv1Weight, [weights.outChannels1, 3, 3, weights.inChannels])
        self.conv1Bias = MLXArray(weights.conv1Bias, [weights.outChannels1])
        self.conv2Weight = MLXArray(weights.conv2Weight, [weights.outChannels2, 3, 3, weights.outChannels1])
        self.conv2Bias = MLXArray(weights.conv2Bias, [weights.outChannels2])
        self.projWeight = MLXArray(weights.projWeight, [weights.outputDim, weights.outChannels2 * 16])
        self.projBias = MLXArray(weights.projBias, [weights.outputDim])

        super.init()
    }

    /// [B, T, melChannels] または [B, T, melChannels, 1] を入力とし、
    /// 時間軸 1/4 圧縮された [B, T/4, outputDim] を返す
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let x4: MLXArray
        switch x.ndim {
        case 3:
            x4 = x.reshaped([x.shape[0], x.shape[1], x.shape[2], 1])
        default:
            x4 = x
        }

        // Conv1: 時間軸は過去2フレームのみ参照する因果的パディング (2, 0)、周波数軸は (1, 1)
        let xPad = padded(x4, widths: [IntOrPair((0, 0)), IntOrPair((2, 0)), IntOrPair((1, 1)), IntOrPair((0, 0))])
        var y1 = conv2d(xPad, conv1Weight, stride: .init((2, 2)), padding: .init((0, 0)))
        y1 = y1 + conv1Bias
        y1 = maximum(y1, MLXArray(0.0))

        // Conv2: 時間軸因果的パディング (2, 0)、周波数軸 (1, 1)
        let y1Pad = padded(y1, widths: [IntOrPair((0, 0)), IntOrPair((2, 0)), IntOrPair((1, 1)), IntOrPair((0, 0))])
        var y2 = conv2d(y1Pad, conv2Weight, stride: .init((2, 2)), padding: .init((0, 0)))
        y2 = y2 + conv2Bias
        y2 = maximum(y2, MLXArray(0.0))

        // 周波数とチャネルを平坦化: [B, T/4, 16, 16] -> [B, T/4, 256]
        let b = y2.shape[0]
        let t = y2.shape[1]
        let flatDim = y2.shape[2] * y2.shape[3]
        let yFlat = y2.reshaped([b, t, flatDim])

        // 線形射影層: [B, T/4, 256] @ [256, outputDim] + projBias
        return matmul(yFlat, projWeight.T) + projBias
    }

    public func exportWeights() -> Conv2DSubsamplingWeights {
        eval([conv1Weight, conv1Bias, conv2Weight, conv2Bias, projWeight, projBias])
        return Conv2DSubsamplingWeights(
            inChannels: inChannels,
            outChannels1: outChannels1,
            outChannels2: outChannels2,
            melChannels: melChannels,
            outputDim: outputDim,
            conv1Weight: conv1Weight.asArray(Float.self),
            conv1Bias: conv1Bias.asArray(Float.self),
            conv2Weight: conv2Weight.asArray(Float.self),
            conv2Bias: conv2Bias.asArray(Float.self),
            projWeight: projWeight.asArray(Float.self),
            projBias: projBias.asArray(Float.self)
        )
    }

    public func importWeights(from weights: Conv2DSubsamplingWeights) {
        self.conv1Weight = MLXArray(weights.conv1Weight, [weights.outChannels1, 3, 3, weights.inChannels])
        self.conv1Bias = MLXArray(weights.conv1Bias, [weights.outChannels1])
        self.conv2Weight = MLXArray(weights.conv2Weight, [weights.outChannels2, 3, 3, weights.outChannels1])
        self.conv2Bias = MLXArray(weights.conv2Bias, [weights.outChannels2])
        self.projWeight = MLXArray(weights.projWeight, [weights.outputDim, weights.outChannels2 * 16])
        self.projBias = MLXArray(weights.projBias, [weights.outputDim])
        eval([conv1Weight, conv1Bias, conv2Weight, conv2Bias, projWeight, projBias])
    }
}
