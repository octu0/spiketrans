import Foundation
import MLX
import MLXNN

/// MLX (Apple Silicon GPU) によるSNN ネットワーク
public final class MLXSpikingNetwork: Module, @unchecked Sendable {
    public let numLayers: Int
    public let inputDim: Int
    public let maxHiddenDim: Int
    public let outputDim: Int
    public let timeSteps: Int
    public let lifConfig: LIFConfig

    /// 2D-Conv Subsampling フロントエンド (オプション)
    public var convSubsampling: MLXConv2DSubsampling?

    // 第0層パラメータ
    public var wIn: MLXArray       // [inputDim, maxHiddenDim]
    public var wRec: MLXArray      // [maxHiddenDim, maxHiddenDim]
    public var bH: MLXArray        // [maxHiddenDim]

    // 上位層パラメータ
    public var wLayers: [MLXArray]   // 各 [maxHiddenDim, maxHiddenDim]
    public var bHLayers: [MLXArray]  // 各 [maxHiddenDim]
    public var gammaRMS: [MLXArray]  // 各 [maxHiddenDim]

    // リードアウト層パラメータ
    public var wOut: MLXArray      // [maxHiddenDim, outputDim]
    public var bOut: MLXArray      // [outputDim]

    public init(
        numLayers: Int = 2,
        inputDim: Int = 128,
        maxHiddenDim: Int = 1024,
        outputDim: Int = 523,
        timeSteps: Int = 4,
        lifConfig: LIFConfig = LIFConfig(),
        convSubsampling: MLXConv2DSubsampling? = nil
    ) {
        self.numLayers = max(1, numLayers)
        self.inputDim = inputDim
        self.maxHiddenDim = maxHiddenDim
        self.outputDim = outputDim
        self.timeSteps = timeSteps
        self.lifConfig = lifConfig
        self.convSubsampling = convSubsampling

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

    /// 保存済み重みから完全に復元して初期化
    public convenience init(weights: SpikingNetworkWeights) {
        let subsampler: MLXConv2DSubsampling?
        switch weights.convSubsampling {
        case .some(let cWeights):
            subsampler = MLXConv2DSubsampling(weights: cWeights)
        case .none:
            subsampler = nil
        }
        self.init(
            numLayers: weights.numLayers,
            inputDim: weights.inputDim,
            maxHiddenDim: weights.maxHiddenDim,
            outputDim: weights.outputDim,
            timeSteps: weights.timeSteps,
            lifConfig: weights.lifConfig,
            convSubsampling: subsampler
        )
        self.importWeights(from: weights)
    }

    /// Pure Swift の SpikingNetworkWeights から重みをインポート
    public func importWeights(from data: SpikingNetworkWeights) {
        let wInArr = MLXArray(data.wIn, [data.maxHiddenDim, data.inputDim]).transposed()
        let wRecArr = MLXArray(data.wRec, [data.maxHiddenDim, data.maxHiddenDim]).transposed()
        let bHArr = MLXArray(data.bH, [data.maxHiddenDim])
        let wOutArr = MLXArray(data.wOut, [data.outputDim, data.maxHiddenDim]).transposed()
        let bOutArr = MLXArray(data.bOut, [data.outputDim])

        self.wIn = wInArr
        self.wRec = wRecArr
        self.bH = bHArr
        self.wOut = wOutArr
        self.bOut = bOutArr

        switch data.convSubsampling {
        case .some(let cWeights):
            self.convSubsampling = MLXConv2DSubsampling(weights: cWeights)
        case .none:
            self.convSubsampling = nil
        }

        var arraysToEval: [MLXArray] = [self.wIn, self.wRec, self.bH, self.wOut, self.bOut]

        switch data.wLayers {
        case .some(let wl):
            var l = 0
            while l < min(self.wLayers.count, wl.count) {
                let arr = MLXArray(wl[l], [data.maxHiddenDim, data.maxHiddenDim]).transposed()
                self.wLayers[l] = arr
                arraysToEval.append(arr)
                l += 1
            }
        case .none:
            break
        }
        switch data.bHLayers {
        case .some(let bl):
            var l = 0
            while l < min(self.bHLayers.count, bl.count) {
                let arr = MLXArray(bl[l], [data.maxHiddenDim])
                self.bHLayers[l] = arr
                arraysToEval.append(arr)
                l += 1
            }
        case .none:
            break
        }
        switch data.gammaRMS {
        case .some(let gl):
            var l = 0
            while l < min(self.gammaRMS.count, gl.count) {
                let arr = MLXArray(gl[l], [data.maxHiddenDim])
                self.gammaRMS[l] = arr
                arraysToEval.append(arr)
                l += 1
            }
        case .none:
            break
        }

        eval(arraysToEval)
    }

    /// Pure Swift の SpikingNetworkWeights へ重みをエクスポート
    public func exportWeights(vocabulary: TextVocabulary? = nil) -> SpikingNetworkWeights {
        var arraysToEval: [MLXArray] = [self.wIn, self.wRec, self.bH, self.wOut, self.bOut]
        if 1 < numLayers {
            var l = 0
            while l < wLayers.count {
                arraysToEval.append(wLayers[l])
                arraysToEval.append(bHLayers[l])
                arraysToEval.append(gammaRMS[l])
                l += 1
            }
        }
        eval(arraysToEval)

        let wInFlat = self.wIn.transposed().asArray(Float.self)
        let wRecFlat = self.wRec.transposed().asArray(Float.self)
        let bHFlat = self.bH.asArray(Float.self)
        let wOutFlat = self.wOut.transposed().asArray(Float.self)
        let bOutFlat = self.bOut.asArray(Float.self)

        var wlFlat: [[Float]]? = nil
        var blFlat: [[Float]]? = nil
        var glFlat: [[Float]]? = nil

        if 1 < numLayers {
            var wl: [[Float]] = []
            var bl: [[Float]] = []
            var gl: [[Float]] = []
            var l = 0
            while l < wLayers.count {
                wl.append(self.wLayers[l].transposed().asArray(Float.self))
                bl.append(self.bHLayers[l].asArray(Float.self))
                gl.append(self.gammaRMS[l].asArray(Float.self))
                l += 1
            }
            wlFlat = wl
            blFlat = bl
            glFlat = gl
        }

        return SpikingNetworkWeights(
            numLayers: numLayers,
            inputDim: inputDim,
            maxHiddenDim: maxHiddenDim,
            outputDim: outputDim,
            timeSteps: timeSteps,
            lifConfig: lifConfig,
            wIn: wInFlat,
            wRec: wRecFlat,
            bH: bHFlat,
            wLayers: wlFlat,
            bHLayers: blFlat,
            gammaRMS: glFlat,
            wOut: wOutFlat,
            bOut: bOutFlat,
            vocabularyCharacters: vocabulary?.serializedCharacters,
            convSubsampling: self.convSubsampling?.exportWeights()
        )
    }
}