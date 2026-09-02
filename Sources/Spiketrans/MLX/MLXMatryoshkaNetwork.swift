import Foundation
import MLX
import MLXNN

/// MLX (Apple Silicon GPU) によるマトリョーシカ SNN ネットワーク
public final class MLXMatryoshkaNetwork: Module, @unchecked Sendable {
    public let inputDim: Int
    public let maxHiddenDim: Int
    public let outputDim: Int
    public let timeSteps: Int
    public let lifConfig: LIFConfig

    public var wIn: MLXArray       // [inputDim, maxHiddenDim]
    public var wRec: MLXArray      // [maxHiddenDim, maxHiddenDim]
    public var bH: MLXArray        // [maxHiddenDim]
    public var wOut: MLXArray      // [maxHiddenDim, outputDim]
    public var bOut: MLXArray      // [outputDim] (High スライス用)
    // スライスごとに blank の閾値を独立に較正するための出力バイアス。
    // sliceNorm は重み付き和にしか掛からないため、バイアスを共有すると
    // 小さいスライスほど非 blank 側が相対的に持ち上がり挿入過多になる。
    public var bOutBase: MLXArray  // [outputDim]
    public var bOutMid: MLXArray   // [outputDim]

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

        let scaleIn = sqrt(2.0 / Float(inputDim))
        let scaleRec = 0.1 / sqrt(Float(maxHiddenDim))
        let scaleOut = sqrt(2.0 / Float(maxHiddenDim))

        self.wIn = MLXRandom.uniform(low: -scaleIn, high: scaleIn, [inputDim, maxHiddenDim])
        self.wRec = MLXRandom.uniform(low: -scaleRec, high: scaleRec, [maxHiddenDim, maxHiddenDim])
        self.bH = MLXArray.zeros([maxHiddenDim])
        self.wOut = MLXRandom.uniform(low: -scaleOut, high: scaleOut, [maxHiddenDim, outputDim])
        self.bOut = MLXArray.zeros([outputDim])
        self.bOutBase = MLXArray.zeros([outputDim])
        self.bOutMid = MLXArray.zeros([outputDim])

        super.init()
    }

    /// Pure Swift の MatryoshkaWeightsData から重みをインポート
    public func importWeights(from data: MatryoshkaWeightsData) {
        // wIn: Pure Swift は [maxHiddenDim, inputDim] なので転置して [inputDim, maxHiddenDim] に変換
        let wInArr = MLXArray(data.wIn, [data.maxHiddenDim, data.inputDim]).transposed()
        let wRecArr = MLXArray(data.wRec, [data.maxHiddenDim, data.maxHiddenDim]).transposed()
        let bHArr = MLXArray(data.bH, [data.maxHiddenDim])
        // wOut: Pure Swift は [outputDim, maxHiddenDim] なので転置して [maxHiddenDim, outputDim] に変換
        let wOutArr = MLXArray(data.wOut, [data.outputDim, data.maxHiddenDim]).transposed()
        let bOutArr = MLXArray(data.bOut, [data.outputDim])
        let bOutBaseArr = MLXArray(data.bOutBase, [data.outputDim])
        let bOutMidArr = MLXArray(data.bOutMiddle, [data.outputDim])

        self.wIn = wInArr
        self.wRec = wRecArr
        self.bH = bHArr
        self.wOut = wOutArr
        self.bOut = bOutArr
        self.bOutBase = bOutBaseArr
        self.bOutMid = bOutMidArr
        eval(self.wIn, self.wRec, self.bH, self.wOut, self.bOut, self.bOutBase, self.bOutMid)
    }

    /// Pure Swift の MatryoshkaWeightsData へ重みをエクスポート
    public func exportWeights() -> MatryoshkaWeightsData {
        eval(self.wIn, self.wRec, self.bH, self.wOut, self.bOut, self.bOutBase, self.bOutMid)

        // [inputDim, maxHiddenDim] -> transposed [maxHiddenDim, inputDim] -> [Float]
        let wInFlat = self.wIn.transposed().asArray(Float.self)
        let wRecFlat = self.wRec.transposed().asArray(Float.self)
        let bHFlat = self.bH.asArray(Float.self)
        let wOutFlat = self.wOut.transposed().asArray(Float.self)
        let bOutFlat = self.bOut.asArray(Float.self)
        let bOutBaseFlat = self.bOutBase.asArray(Float.self)
        let bOutMidFlat = self.bOutMid.asArray(Float.self)

        return MatryoshkaWeightsData(
            inputDim: inputDim,
            maxHiddenDim: maxHiddenDim,
            outputDim: outputDim,
            timeSteps: timeSteps,
            lifConfig: lifConfig,
            wIn: wInFlat,
            wRec: wRecFlat,
            bH: bHFlat,
            wOut: wOutFlat,
            bOut: bOutFlat,
            bOutBase: bOutBaseFlat,
            bOutMiddle: bOutMidFlat
        )
    }
}