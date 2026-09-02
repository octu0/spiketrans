import Foundation

/// 固定小数点量子化設定
public struct QuantizedConfig: Sendable, Equatable {
    public let vThInt: Int32
    public let decayNum: Int32
    public let decayBits: Int
    public let scale: Float
    public let scaleBits: Int

    public init(
        vThInt: Int32,
        decayNum: Int32,
        decayBits: Int,
        scale: Float,
        scaleBits: Int
    ) {
        self.vThInt = vThInt
        self.decayNum = decayNum
        self.decayBits = decayBits
        self.scale = scale
        self.scaleBits = scaleBits
    }

    public static func int16Config() -> QuantizedConfig {
        return QuantizedConfig(
            vThInt: 2048,
            decayNum: 3277,
            decayBits: 12,
            scale: 2048.0,
            scaleBits: 11
        )
    }

    public static func int32Config() -> QuantizedConfig {
        return QuantizedConfig(
            vThInt: 65536,
            decayNum: 52429,
            decayBits: 16,
            scale: 65536.0,
            scaleBits: 16
        )
    }
}

/// 固定小数点量子化重みコンテナ
public struct QuantizedWeights: Sendable, Equatable {
    public let inputDim: Int
    public let maxHiddenDim: Int
    public let outputDim: Int
    public let config: QuantizedConfig
    public let wIn: [Int32]
    public let wRec: [Int32]
    public let bH: [Int32]
    public let wOut: [Int32]        // 全スライスで共有
    public let bOut: [Int32]

    public init(
        inputDim: Int,
        maxHiddenDim: Int,
        outputDim: Int,
        config: QuantizedConfig,
        wIn: [Int32],
        wRec: [Int32],
        bH: [Int32],
        wOut: [Int32],
        bOut: [Int32]
    ) {
        self.inputDim = inputDim
        self.maxHiddenDim = maxHiddenDim
        self.outputDim = outputDim
        self.config = config
        self.wIn = wIn
        self.wRec = wRec
        self.bH = bH
        self.wOut = wOut
        self.bOut = bOut

    }
}

/// 固定小数点推論用事前確保ワークスペース (0 アロケーション)
public final class QuantizedWorkspace: @unchecked Sendable {
    public var vPrev: ContiguousArray<Int32>
    public var sPrev: ContiguousArray<Int32>
    public var vNext: ContiguousArray<Int32>
    public var sNext: ContiguousArray<Int32>
    public var spikeSum: ContiguousArray<Int32>
    public var inputInt: ContiguousArray<Int32>
    public var logitsInt: ContiguousArray<Int64>

    public init(maxHiddenDim: Int, inputDim: Int, outputDim: Int) {
        self.vPrev = ContiguousArray<Int32>(repeating: 0, count: maxHiddenDim)
        self.sPrev = ContiguousArray<Int32>(repeating: 0, count: maxHiddenDim)
        self.vNext = ContiguousArray<Int32>(repeating: 0, count: maxHiddenDim)
        self.sNext = ContiguousArray<Int32>(repeating: 0, count: maxHiddenDim)
        self.spikeSum = ContiguousArray<Int32>(repeating: 0, count: maxHiddenDim)
        self.inputInt = ContiguousArray<Int32>(repeating: 0, count: inputDim)
        self.logitsInt = ContiguousArray<Int64>(repeating: 0, count: outputDim)
    }

    @inline(__always)
    public func reset() {
        var i = 0
        while i < vPrev.count {
            vPrev[i] = 0
            sPrev[i] = 0
            vNext[i] = 0
            sNext[i] = 0
            spikeSum[i] = 0
            i += 1
        }
    }
}

/// Int32 / Int16 固定小数点推論エンジン
public final class QuantizedEngine: @unchecked Sendable {
    public let weights: QuantizedWeights
    public let timeSteps: Int

    public init(weights: QuantizedWeights, timeSteps: Int = 4) {
        self.weights = weights
        self.timeSteps = timeSteps
    }

    /// Float32 モデルから Int32 / Int16 量子化モデルを生成するコンバータ
    public static func quantize(
        network: SpikingNetwork,
        config: QuantizedConfig,
        minVal: Int64 = -2147483648,
        maxVal: Int64 = 2147483647
    ) -> QuantizedWeights {
        let scale = config.scale

        let quantizeArray = { (src: [Float]) -> [Int32] in
            var res = [Int32](repeating: 0, count: src.count)
            let minD = Double(minVal)
            let maxD = Double(maxVal)
            var i = 0
            while i < src.count {
                let v = src[i]
                if v.isNaN || v.isInfinite {
                    res[i] = 0
                } else {
                    var vD = Double(v) * Double(scale)
                    if vD < minD {
                        vD = minD
                    }
                    if maxD < vD {
                        vD = maxD
                    }
                    res[i] = Int32(round(vD))
                }
                i += 1
            }
            return res
        }

        return QuantizedWeights(
            inputDim: network.inputDim,
            maxHiddenDim: network.maxHiddenDim,
            outputDim: network.outputDim,
            config: config,
            wIn: quantizeArray(network.pWIn.data),
            wRec: quantizeArray(network.pWRec.data),
            bH: quantizeArray(network.pBH.data),
            wOut: quantizeArray(network.pWOut.data),
            bOut: quantizeArray(network.pBOut.data)
        )
    }

    /// 固定小数点推論（乗算フリー・スパース加算・ビットシフト減衰）
    public func predict(
        features: [Float],
        workspace: QuantizedWorkspace,
        outputProbs: inout [Float]
    ) {
        let hSize = weights.maxHiddenDim
        let scale = weights.config.scale
        let scaleBits = Int64(weights.config.scaleBits)
        let vThInt = weights.config.vThInt
        let decayNum = Int64(weights.config.decayNum)
        let decayBits = Int64(weights.config.decayBits)

        // 1. 入力特徴量の整数化
        var d = 0
        while d < weights.inputDim {
            let fVal = features[d] * scale
            workspace.inputInt[d] = Int32(round(Double(fVal)))
            d += 1
        }

        workspace.reset()

        // 2. 時間ステップループ
        var t = 0
        while t < timeSteps {
            var i = 0
            while i < hSize {
                // 入力電流: bH + WIn * inputInt
                var current = weights.bH[i]
                let inOffset = i * weights.inputDim
                var inD = 0
                while inD < weights.inputDim {
                    current += Int32((Int64(weights.wIn[inOffset + inD]) * Int64(workspace.inputInt[inD])) >> scaleBits)
                    inD += 1
                }

                // スパースリカレント加算 (乗算器フリー: 前ステップでスパイクした重みのみ加算)
                let recOffset = i * weights.maxHiddenDim
                var j = 0
                while j < hSize {
                    if workspace.sPrev[j] != 0 {
                        current += weights.wRec[recOffset + j]
                    }
                    j += 1
                }

                // ビットシフト減衰
                var vEffective: Int64 = Int64(workspace.vPrev[i])
                if workspace.sPrev[i] != 0 {
                    vEffective = 0
                }
                let vDecayed = Int32((vEffective * decayNum) >> decayBits)
                let vNext = vDecayed + current

                var sNext: Int32 = 0
                if vThInt <= vNext {
                    sNext = 1
                }

                workspace.vNext[i] = vNext
                workspace.sNext[i] = sNext
                workspace.spikeSum[i] += sNext
                i += 1
            }

            // ステップ終了時に状態を更新
            i = 0
            while i < hSize {
                workspace.vPrev[i] = workspace.vNext[i]
                workspace.sPrev[i] = workspace.sNext[i]
                i += 1
            }

            t += 1
        }

        // 3. リードアウト層整数計算

        var maxLogit: Int64 = -1 << 60
        var c = 0
        while c < weights.outputDim {
            var sum = Int64(weights.bOut[c]) * Int64(timeSteps)
            let wOffset = c * weights.maxHiddenDim
            var i = 0
            while i < hSize {
                let sCount = workspace.spikeSum[i]
                if 0 < sCount {
                    sum += Int64(weights.wOut[wOffset + i]) * Int64(sCount)
                }
                i += 1
            }
            workspace.logitsInt[c] = sum
            if maxLogit < sum {
                maxLogit = sum
            }
            c += 1
        }

        // 4. スケーリング Softmax
        let normFactor = 1.0 / (scale * Float(timeSteps))
        var sumExp: Float = 0.0
        c = 0
        while c < weights.outputDim {
            var diff = Float(workspace.logitsInt[c] - maxLogit) * normFactor
            if diff < -50.0 {
                diff = -50.0
            }
            let expVal = exp(diff)
            outputProbs[c] = expVal
            sumExp += expVal
            c += 1
        }

        let invSum = 1.0 / sumExp
        c = 0
        while c < weights.outputDim {
            outputProbs[c] *= invSum
            c += 1
        }
    }
}
