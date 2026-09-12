import Foundation

public enum QuantizedFormat: Sendable, Equatable {
    case int16
    case int32
    case float16
}

/// 量子化設定。int16 / int32 は固定小数点、float16 は IEEE 半精度。
public struct QuantizedConfig: Sendable, Equatable {
    public let format: QuantizedFormat
    public let vThInt: Int32
    public let decayNum: Int32
    public let decayBits: Int
    public let scale: Float
    public let scaleBits: Int
    /// float16 経路だけ使う。固定小数点では無視する
    public let beta: Float
    public let vTh: Float

    public init(
        vThInt: Int32,
        decayNum: Int32,
        decayBits: Int,
        scale: Float,
        scaleBits: Int,
        format: QuantizedFormat = .int32,
        beta: Float = 0.8,
        vTh: Float = 1.0
    ) {
        self.format = format
        self.vThInt = vThInt
        self.decayNum = decayNum
        self.decayBits = decayBits
        self.scale = scale
        self.scaleBits = scaleBits
        self.beta = beta
        self.vTh = vTh
    }

    public static func int16Config() -> QuantizedConfig {
        return QuantizedConfig(
            vThInt: 2048,
            decayNum: 3277,
            decayBits: 12,
            scale: 2048.0,
            scaleBits: 11,
            format: .int16
        )
    }

    public static func int32Config() -> QuantizedConfig {
        return QuantizedConfig(
            vThInt: 65536,
            decayNum: 52429,
            decayBits: 16,
            scale: 65536.0,
            scaleBits: 16,
            format: .int32
        )
    }

    public static func float16Config(beta: Float = 0.8, vTh: Float = 1.0) -> QuantizedConfig {
        return QuantizedConfig(
            vThInt: 0,
            decayNum: 0,
            decayBits: 0,
            scale: 1.0,
            scaleBits: 0,
            format: .float16,
            beta: beta,
            vTh: vTh
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
    public let wOut: [Int32]
    public let bOut: [Int32]
    public let wInF16: [Float16]
    public let wRecF16: [Float16]
    public let bHF16: [Float16]
    public let wOutF16: [Float16]
    public let bOutF16: [Float16]

    public init(
        inputDim: Int,
        maxHiddenDim: Int,
        outputDim: Int,
        config: QuantizedConfig,
        wIn: [Int32],
        wRec: [Int32],
        bH: [Int32],
        wOut: [Int32],
        bOut: [Int32],
        wInF16: [Float16] = [],
        wRecF16: [Float16] = [],
        bHF16: [Float16] = [],
        wOutF16: [Float16] = [],
        bOutF16: [Float16] = []
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
        self.wInF16 = wInF16
        self.wRecF16 = wRecF16
        self.bHF16 = bHF16
        self.wOutF16 = wOutF16
        self.bOutF16 = bOutF16
    }
}

/// 固定小数点フォワードの膜電位・スパイク・入力・ロジット。`predict` が毎回確保しない。
public final class QuantizedWorkspace: @unchecked Sendable {
    public var vPrev: ContiguousArray<Int32>
    public var sPrev: ContiguousArray<Int32>
    public var vNext: ContiguousArray<Int32>
    public var sNext: ContiguousArray<Int32>
    public var inputInt: ContiguousArray<Int32>
    public var vPrevF16: ContiguousArray<Float16>
    public var vNextF16: ContiguousArray<Float16>
    public var inputF16: ContiguousArray<Float16>
    public var readoutSum: ContiguousArray<Float>
    public var logitsFloat: ContiguousArray<Float>

    public init(maxHiddenDim: Int, inputDim: Int, outputDim: Int) {
        self.vPrev = ContiguousArray<Int32>(repeating: 0, count: maxHiddenDim)
        self.sPrev = ContiguousArray<Int32>(repeating: 0, count: maxHiddenDim)
        self.vNext = ContiguousArray<Int32>(repeating: 0, count: maxHiddenDim)
        self.sNext = ContiguousArray<Int32>(repeating: 0, count: maxHiddenDim)
        self.inputInt = ContiguousArray<Int32>(repeating: 0, count: inputDim)
        self.vPrevF16 = ContiguousArray<Float16>(repeating: 0, count: maxHiddenDim)
        self.vNextF16 = ContiguousArray<Float16>(repeating: 0, count: maxHiddenDim)
        self.inputF16 = ContiguousArray<Float16>(repeating: 0, count: inputDim)
        self.readoutSum = ContiguousArray<Float>(repeating: 0.0, count: maxHiddenDim)
        self.logitsFloat = ContiguousArray<Float>(repeating: 0.0, count: outputDim)
    }

    @inline(__always)
    public func reset() {
        var i = 0
        while i < vPrev.count {
            vPrev[i] = 0
            sPrev[i] = 0
            vNext[i] = 0
            sNext[i] = 0
            vPrevF16[i] = 0
            vNextF16[i] = 0
            readoutSum[i] = 0.0
            i += 1
        }
    }
}

/// Int32 / Int16 固定小数点、および Float16 のフォワード。層 0 のみ。
public final class QuantizedEngine: @unchecked Sendable {
    public let weights: QuantizedWeights
    public let timeSteps: Int

    public init(weights: QuantizedWeights, timeSteps: Int = 4) {
        self.weights = weights
        self.timeSteps = timeSteps
    }

    /// 層 0 の重みを量子化する。上位層・適応閾値は持たない。
    public static func quantize(
        network: SpikingNetwork,
        config: QuantizedConfig,
        minVal: Int64 = -2147483648,
        maxVal: Int64 = 2147483647
    ) -> QuantizedWeights {
        if config.format == .float16 {
            return quantizeFloat16(network: network)
        }
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

    private static func quantizeFloat16(network: SpikingNetwork) -> QuantizedWeights {
        let cfg = QuantizedConfig.float16Config(
            beta: network.lifConfig.beta,
            vTh: network.lifConfig.vTh
        )
        return QuantizedWeights(
            inputDim: network.inputDim,
            maxHiddenDim: network.maxHiddenDim,
            outputDim: network.outputDim,
            config: cfg,
            wIn: [],
            wRec: [],
            bH: [],
            wOut: [],
            bOut: [],
            wInF16: toFloat16(network.pWIn.data),
            wRecF16: toFloat16(network.pWRec.data),
            bHF16: toFloat16(network.pBH.data),
            wOutF16: toFloat16(network.pWOut.data),
            bOutF16: toFloat16(network.pBOut.data)
        )
    }

    private static func toFloat16(_ src: [Float]) -> [Float16] {
        let maxF16 = Float(Float16.greatestFiniteMagnitude)
        var res = [Float16](repeating: 0, count: src.count)
        var i = 0
        while i < src.count {
            let v = src[i]
            if v.isFinite != true {
                res[i] = 0
            } else if maxF16 < v {
                res[i] = Float16.greatestFiniteMagnitude
            } else if v < -maxF16 {
                res[i] = -Float16.greatestFiniteMagnitude
            } else {
                res[i] = Float16(v)
            }
            i += 1
        }
        return res
    }

    /// 固定小数点推論。入力は整数 MAC、再帰は発火ニューロンの重み加算、減衰はビットシフト。
    public func predict(
        features: [Float],
        workspace: QuantizedWorkspace,
        outputProbs: inout [Float]
    ) {
        if weights.config.format == .float16 {
            predictFloat16(features: features, workspace: workspace, outputProbs: &outputProbs)
            return
        }
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

                let vDecayed = Int32((Int64(workspace.vPrev[i]) * decayNum) >> decayBits)
                let vIntegrated = vDecayed + current
                var sNext: Int32 = 0
                if vThInt <= vIntegrated {
                    sNext = 1
                }
                workspace.readoutSum[i] += LIFNeuronEngine.scaleReadout(
                    Float(vIntegrated),
                    vTh: Float(vThInt)
                )
                var vNext = vIntegrated
                if sNext != 0 {
                    vNext -= vThInt
                }

                workspace.vNext[i] = vNext
                workspace.sNext[i] = sNext
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

        let invT = 1.0 / Float(timeSteps)
        let invScale = 1.0 / scale

        var maxLogit: Float = -Float.greatestFiniteMagnitude
        var c = 0
        while c < weights.outputDim {
            var sum = Float(weights.bOut[c]) * invScale
            let wOffset = c * weights.maxHiddenDim
            var i = 0
            while i < hSize {
                sum += Float(weights.wOut[wOffset + i]) * invScale * workspace.readoutSum[i] * invT
                i += 1
            }
            workspace.logitsFloat[c] = sum
            if maxLogit < sum {
                maxLogit = sum
            }
            c += 1
        }
        writeSoftmax(maxLogit: maxLogit, workspace: workspace, outputProbs: &outputProbs)
    }

    /// Float16 演算。最終層と同じくリーク → 閾値単位読み出し → subtractive reset。
    private func predictFloat16(
        features: [Float],
        workspace: QuantizedWorkspace,
        outputProbs: inout [Float]
    ) {
        let hSize = weights.maxHiddenDim
        let beta = Float16(weights.config.beta)
        let vTh = Float16(weights.config.vTh)
        let vMin = Float16(LIFNeuronEngine.vClampMin)
        let vMax = Float16(LIFNeuronEngine.vClampMax)
        let vThFloat = weights.config.vTh

        var d = 0
        while d < weights.inputDim {
            let fVal = features[d]
            if fVal.isFinite {
                workspace.inputF16[d] = Float16(fVal)
            } else {
                workspace.inputF16[d] = 0
            }
            d += 1
        }

        workspace.reset()

        var t = 0
        while t < timeSteps {
            var i = 0
            while i < hSize {
                var current = weights.bHF16[i]
                let inOffset = i * weights.inputDim
                var inD = 0
                while inD < weights.inputDim {
                    current += weights.wInF16[inOffset + inD] * workspace.inputF16[inD]
                    inD += 1
                }
                let recOffset = i * weights.maxHiddenDim
                var j = 0
                while j < hSize {
                    if workspace.sPrev[j] != 0 {
                        current += weights.wRecF16[recOffset + j]
                    }
                    j += 1
                }

                var vIntegrated = workspace.vPrevF16[i] * beta + current
                if vIntegrated < vMin {
                    vIntegrated = vMin
                }
                if vMax < vIntegrated {
                    vIntegrated = vMax
                }

                var sNext: Int32 = 0
                if vTh <= vIntegrated {
                    sNext = 1
                }
                workspace.readoutSum[i] += LIFNeuronEngine.scaleReadout(
                    Float(vIntegrated),
                    vTh: vThFloat
                )
                var vNext = vIntegrated
                if sNext != 0 {
                    vNext -= vTh
                }
                if vNext < vMin {
                    vNext = vMin
                }
                if vMax < vNext {
                    vNext = vMax
                }

                workspace.vNextF16[i] = vNext
                workspace.sNext[i] = sNext
                i += 1
            }

            i = 0
            while i < hSize {
                workspace.vPrevF16[i] = workspace.vNextF16[i]
                workspace.sPrev[i] = workspace.sNext[i]
                i += 1
            }
            t += 1
        }

        let invT = 1.0 / Float(timeSteps)
        var maxLogit: Float = -Float.greatestFiniteMagnitude
        var c = 0
        while c < weights.outputDim {
            var sum = Float(weights.bOutF16[c])
            let wOffset = c * weights.maxHiddenDim
            var i = 0
            while i < hSize {
                sum += Float(weights.wOutF16[wOffset + i]) * workspace.readoutSum[i] * invT
                i += 1
            }
            workspace.logitsFloat[c] = sum
            if maxLogit < sum {
                maxLogit = sum
            }
            c += 1
        }
        writeSoftmax(maxLogit: maxLogit, workspace: workspace, outputProbs: &outputProbs)
    }

    private func writeSoftmax(
        maxLogit: Float,
        workspace: QuantizedWorkspace,
        outputProbs: inout [Float]
    ) {
        var sumExp: Float = 0.0
        var c = 0
        while c < weights.outputDim {
            var diff = workspace.logitsFloat[c] - maxLogit
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
