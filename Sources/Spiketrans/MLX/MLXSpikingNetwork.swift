import Foundation
import MLX
import MLXNN

/// MLX (Apple Silicon GPU) によるSNN ネットワーク
public final class MLXSpikingNetwork: Module, @unchecked Sendable {
    public let inputDim: Int
    public let maxHiddenDim: Int
    public let outputDim: Int
    public let timeSteps: Int
    public private(set) var lifConfig: LIFConfig
    public private(set) var betaArray: MLXArray    // [1, maxHiddenDim]

    public var wIn: MLXArray       // [inputDim, maxHiddenDim]
    public var wRec: MLXArray      // [maxHiddenDim, maxHiddenDim]
    public var bH: MLXArray        // [maxHiddenDim]
    public var wOut: MLXArray      // [maxHiddenDim, outputDim] (全スライスで共有)
    public var bOut: MLXArray      // [outputDim]

    public init(
        inputDim: Int = 128,
        maxHiddenDim: Int = 1024,
        outputDim: Int = 523,
        timeSteps: Int = 4,
        lifConfig: LIFConfig = LIFConfig()
    ) {
        self.inputDim = inputDim
        self.maxHiddenDim = maxHiddenDim
        self.outputDim = outputDim
        self.timeSteps = timeSteps
        self.lifConfig = lifConfig

        let betaVec = lifConfig.betaVector(count: maxHiddenDim)
        self.betaArray = MLXArray(betaVec, [1, maxHiddenDim])

        let scaleIn = sqrt(2.0 / Float(inputDim))
        let scaleRec = 0.1 / sqrt(Float(maxHiddenDim))
        let scaleOut = sqrt(2.0 / Float(maxHiddenDim))

        self.wIn = MLXRandom.uniform(low: -scaleIn, high: scaleIn, [inputDim, maxHiddenDim])
        self.wRec = MLXRandom.uniform(low: -scaleRec, high: scaleRec, [maxHiddenDim, maxHiddenDim])
        self.bH = MLXArray.zeros([maxHiddenDim])
        self.wOut = MLXRandom.uniform(low: -scaleOut, high: scaleOut, [maxHiddenDim, outputDim])
        self.bOut = MLXArray.zeros([outputDim])

        super.init()
    }

    /// Pure Swift の SpikingNetworkWeights から重みをインポート
    public func importWeights(from data: SpikingNetworkWeights) {
        self.lifConfig = data.lifConfig
        let betaVec = self.lifConfig.betaVector(count: maxHiddenDim)
        self.betaArray = MLXArray(betaVec, [1, maxHiddenDim])

        // wIn: Pure Swift は [maxHiddenDim, inputDim] なので転置して [inputDim, maxHiddenDim] に変換
        let wInArr = MLXArray(data.wIn, [data.maxHiddenDim, data.inputDim]).transposed()
        let wRecArr = MLXArray(data.wRec, [data.maxHiddenDim, data.maxHiddenDim]).transposed()
        let bHArr = MLXArray(data.bH, [data.maxHiddenDim])
        // wOut: Pure Swift は [outputDim, maxHiddenDim] なので転置して [maxHiddenDim, outputDim] に変換
        let wOutArr = MLXArray(data.wOut, [data.outputDim, data.maxHiddenDim]).transposed()
        let bOutArr = MLXArray(data.bOut, [data.outputDim])

        self.wIn = wInArr
        self.wRec = wRecArr
        self.bH = bHArr
        self.wOut = wOutArr
        self.bOut = bOutArr
        eval(self.wIn, self.wRec, self.bH, self.wOut, self.bOut, self.betaArray)
    }

    /// Pure Swift の SpikingNetworkWeights へ重みをエクスポート
    public func exportWeights() -> SpikingNetworkWeights {
        eval(self.wIn, self.wRec, self.bH, self.wOut, self.bOut)

        // [inputDim, maxHiddenDim] -> transposed [maxHiddenDim, inputDim] -> [Float]
        let wInFlat = self.wIn.transposed().asArray(Float.self)
        let wRecFlat = self.wRec.transposed().asArray(Float.self)
        let bHFlat = self.bH.asArray(Float.self)
        let wOutFlat = self.wOut.transposed().asArray(Float.self)
        let bOutFlat = self.bOut.asArray(Float.self)

        return SpikingNetworkWeights(
            inputDim: inputDim,
            maxHiddenDim: maxHiddenDim,
            outputDim: outputDim,
            timeSteps: timeSteps,
            lifConfig: lifConfig,
            wIn: wInFlat,
            wRec: wRecFlat,
            bH: bHFlat,
            wOut: wOutFlat,
            bOut: bOutFlat
        )
    }
}