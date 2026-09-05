import Foundation
import MLX
import MLXNN

/// MLX (Apple Silicon GPU) によるSNN ネットワーク。
///
/// 層 0 は再帰 LIF 層、層 1 以降は前層のスパイクを受けるフィードフォワード LIF 層。
/// 上位層への入力電流は RMSNorm で単位スケールに揃え、前層の入力電流を加算する
/// (電流空間の残差接続)。疎な発火率をそのまま重み付けすると上位層が沈黙するため
public final class MLXSpikingNetwork: Module, @unchecked Sendable {
    public let numLayers: Int
    public let inputDim: Int
    public let maxHiddenDim: Int
    public let outputDim: Int
    public let timeSteps: Int
    public let lifConfig: LIFConfig

    // 層 0
    public var wIn: MLXArray       // [inputDim, maxHiddenDim]
    public var wRec: MLXArray      // [maxHiddenDim, maxHiddenDim]
    public var bH: MLXArray        // [maxHiddenDim]

    // 層 1 以降
    public var wLayers: [MLXArray]   // 各 [maxHiddenDim, maxHiddenDim]
    public var bHLayers: [MLXArray]  // 各 [maxHiddenDim]
    public var gammaRMS: [MLXArray]  // 各 [maxHiddenDim]

    // リードアウト
    public var wOut: MLXArray      // [maxHiddenDim, outputDim]
    public var bOut: MLXArray      // [outputDim]

    public init(
        numLayers: Int = 1,
        inputDim: Int = 128,
        maxHiddenDim: Int = 1024,
        outputDim: Int = 523,
        timeSteps: Int = 4,
        lifConfig: LIFConfig = LIFConfig()
    ) {
        self.numLayers = max(1, numLayers)
        self.inputDim = inputDim
        self.maxHiddenDim = maxHiddenDim
        self.outputDim = outputDim
        self.timeSteps = timeSteps
        self.lifConfig = lifConfig

        let scaleIn = sqrt(2.0 / Float(inputDim))
        let scaleRec = 0.1 / sqrt(Float(maxHiddenDim))
        let scaleOut = sqrt(2.0 / Float(maxHiddenDim))
        let scaleLayer = sqrt(2.0 / Float(maxHiddenDim))

        self.wIn = MLXRandom.uniform(low: -scaleIn, high: scaleIn, [inputDim, maxHiddenDim])
        self.wRec = MLXRandom.uniform(low: -scaleRec, high: scaleRec, [maxHiddenDim, maxHiddenDim])
        self.bH = MLXArray.zeros([maxHiddenDim])

        var wl: [MLXArray] = []
        var bl: [MLXArray] = []
        var gl: [MLXArray] = []
        var l = 1
        while l < self.numLayers {
            wl.append(MLXRandom.uniform(low: -scaleLayer, high: scaleLayer, [maxHiddenDim, maxHiddenDim]))
            bl.append(MLXArray.zeros([maxHiddenDim]))
            gl.append(MLXArray.ones([maxHiddenDim]))
            l += 1
        }
        self.wLayers = wl
        self.bHLayers = bl
        self.gammaRMS = gl

        self.wOut = MLXRandom.uniform(low: -scaleOut, high: scaleOut, [maxHiddenDim, outputDim])
        self.bOut = MLXArray.zeros([outputDim])

        super.init()
    }

    /// 保存済み重みから構成ごと復元する
    public convenience init(weights: SpikingNetworkWeights) {
        self.init(
            numLayers: weights.numLayers,
            inputDim: weights.inputDim,
            maxHiddenDim: weights.maxHiddenDim,
            outputDim: weights.outputDim,
            timeSteps: weights.timeSteps,
            lifConfig: weights.lifConfig
        )
        self.importWeights(from: weights)
    }

    /// Pure Swift の SpikingNetworkWeights から重みをインポート。
    /// Pure Swift 側は [出力, 入力] の行優先なので転置して持つ
    public func importWeights(from data: SpikingNetworkWeights) {
        let hSize = data.maxHiddenDim
        self.wIn = MLXArray(data.wIn, [hSize, data.inputDim]).transposed()
        self.wRec = MLXArray(data.wRec, [hSize, hSize]).transposed()
        self.bH = MLXArray(data.bH, [hSize])
        self.wOut = MLXArray(data.wOut, [data.outputDim, hSize]).transposed()
        self.bOut = MLXArray(data.bOut, [data.outputDim])

        var arraysToEval: [MLXArray] = [self.wIn, self.wRec, self.bH, self.wOut, self.bOut]
        var l = 0
        while l < min(self.wLayers.count, data.wLayers.count) {
            self.wLayers[l] = MLXArray(data.wLayers[l], [hSize, hSize]).transposed()
            self.bHLayers[l] = MLXArray(data.bHLayers[l], [hSize])
            self.gammaRMS[l] = MLXArray(data.gammaRMS[l], [hSize])
            arraysToEval.append(self.wLayers[l])
            arraysToEval.append(self.bHLayers[l])
            arraysToEval.append(self.gammaRMS[l])
            l += 1
        }
        eval(arraysToEval)
    }

    /// Pure Swift の SpikingNetworkWeights へ重みをエクスポート
    public func exportWeights(vocabulary: TextVocabulary? = nil) -> SpikingNetworkWeights {
        var arraysToEval: [MLXArray] = [self.wIn, self.wRec, self.bH, self.wOut, self.bOut]
        var l = 0
        while l < wLayers.count {
            arraysToEval.append(wLayers[l])
            arraysToEval.append(bHLayers[l])
            arraysToEval.append(gammaRMS[l])
            l += 1
        }
        eval(arraysToEval)

        var wl: [[Float]] = []
        var bl: [[Float]] = []
        var gl: [[Float]] = []
        l = 0
        while l < wLayers.count {
            wl.append(self.wLayers[l].transposed().asArray(Float.self))
            bl.append(self.bHLayers[l].asArray(Float.self))
            gl.append(self.gammaRMS[l].asArray(Float.self))
            l += 1
        }

        return SpikingNetworkWeights(
            inputDim: inputDim,
            maxHiddenDim: maxHiddenDim,
            outputDim: outputDim,
            timeSteps: timeSteps,
            lifConfig: lifConfig,
            wIn: self.wIn.transposed().asArray(Float.self),
            wRec: self.wRec.transposed().asArray(Float.self),
            bH: self.bH.asArray(Float.self),
            wLayers: wl,
            bHLayers: bl,
            gammaRMS: gl,
            wOut: self.wOut.transposed().asArray(Float.self),
            bOut: self.bOut.asArray(Float.self),
            vocabularyCharacters: vocabulary?.serializedCharacters
        )
    }
}
