import Foundation

/// マトリョーシカスライス粒度
public enum MatryoshkaSlice: Int, Sendable, CaseIterable {
    case base   = 128
    case middle = 512
    case high   = 1024
}

/// SNN 全体重みコンテナ
public struct MatryoshkaWeights: Sendable, Equatable {
    public let inputDim: Int
    public let hiddenDim: Int     // 1024 (High)
    public let outputDim: Int
    public var wIn: [Float]       // [hiddenDim, inputDim]
    public var wRec: [Float]      // [hiddenDim, hiddenDim]
    public var bH: [Float]        // [hiddenDim]
    public var wOut: [Float]      // [outputDim, hiddenDim]
    public var bOut: [Float]      // [outputDim]

    public init(
        inputDim: Int,
        hiddenDim: Int,
        outputDim: Int,
        wIn: [Float],
        wRec: [Float],
        bH: [Float],
        wOut: [Float],
        bOut: [Float]
    ) {
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.outputDim = outputDim
        self.wIn = wIn
        self.wRec = wRec
        self.bH = bH
        self.wOut = wOut
        self.bOut = bOut
    }
}

/// Base 単体エクスポート用モデル重みコンテナ
public struct BaseSNNWeights: Codable, Sendable, Equatable {
    public let inputDim: Int
    public let hiddenDim: Int     // 1024
    public let outputDim: Int
    public let wIn: [Float]       // 1024 * inputDim
    public let wRec: [Float]      // 1024 * 1024
    public let bH: [Float]        // 1024
    public let wOut: [Float]      // outputDim * 1024
    public let bOut: [Float]      // outputDim

    public init(
        inputDim: Int,
        hiddenDim: Int,
        outputDim: Int,
        wIn: [Float],
        wRec: [Float],
        bH: [Float],
        wOut: [Float],
        bOut: [Float]
    ) {
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.outputDim = outputDim
        self.wIn = wIn
        self.wRec = wRec
        self.bH = bH
        self.wOut = wOut
        self.bOut = bOut
    }
}

/// マトリョーシカ SNN ネットワーク本体 (Float32)
public final class MatryoshkaNetwork: @unchecked Sendable {
    public let inputDim: Int
    public let maxHiddenDim: Int  // 4096
    public let outputDim: Int
    public let timeSteps: Int
    public let lifConfig: LIFConfig

    public let pWIn: Parameter
    public let pWRec: Parameter
    public let pBH: Parameter
    public let pWOut: Parameter
    public let pBOut: Parameter

    public init(
        inputDim: Int = 64,
        maxHiddenDim: Int = 4096,
        outputDim: Int = 64,
        timeSteps: Int = 4,
        lifConfig: LIFConfig = LIFConfig()
    ) {
        self.inputDim = inputDim
        self.maxHiddenDim = maxHiddenDim
        self.outputDim = outputDim
        self.timeSteps = timeSteps
        self.lifConfig = lifConfig

        // Direct Input Current に最適化した重み初期化
        let scaleIn = sqrt(2.0 / Float(inputDim))
        var initWIn = [Float](repeating: 0.0, count: maxHiddenDim * inputDim)
        var i = 0
        while i < maxHiddenDim * inputDim {
            initWIn[i] = Float.random(in: -scaleIn...scaleIn)
            i += 1
        }

        let scaleRec = sqrt(2.0 / Float(maxHiddenDim)) * 0.2
        var initWRec = [Float](repeating: 0.0, count: maxHiddenDim * maxHiddenDim)
        i = 0
        while i < maxHiddenDim * maxHiddenDim {
            initWRec[i] = Float.random(in: -scaleRec...scaleRec)
            i += 1
        }

        let initBH = [Float](repeating: 0.35, count: maxHiddenDim)
        
        let scaleOut = sqrt(2.0 / Float(maxHiddenDim))
        var initWOut = [Float](repeating: 0.0, count: outputDim * maxHiddenDim)
        i = 0
        while i < outputDim * maxHiddenDim {
            initWOut[i] = Float.random(in: -scaleOut...scaleOut)
            i += 1
        }

        let initBOut = [Float](repeating: 0.0, count: outputDim)

        self.pWIn = Parameter(count: maxHiddenDim * inputDim, initialData: initWIn)
        self.pWRec = Parameter(count: maxHiddenDim * maxHiddenDim, initialData: initWRec)
        self.pBH = Parameter(count: maxHiddenDim, initialData: initBH)
        self.pWOut = Parameter(count: outputDim * maxHiddenDim, initialData: initWOut)
        self.pBOut = Parameter(count: outputDim, initialData: initBOut)
    }

    public var parameters: [Parameter] {
        return [pWIn, pWRec, pBH, pWOut, pBOut]
    }

    /// Base 単体モデルのエクスポート
    public func exportBaseWeights() -> BaseSNNWeights {
        let hBase = min(MatryoshkaSlice.base.rawValue, maxHiddenDim)
        var baseWIn = [Float](repeating: 0.0, count: hBase * inputDim)
        var baseWRec = [Float](repeating: 0.0, count: hBase * hBase)
        var baseBH = [Float](repeating: 0.0, count: hBase)
        var baseWOut = [Float](repeating: 0.0, count: outputDim * hBase)
        var baseBOut = [Float](repeating: 0.0, count: outputDim)

        // WIn
        var r = 0
        while r < hBase {
            var c = 0
            while c < inputDim {
                baseWIn[r * inputDim + c] = pWIn.data[r * inputDim + c]
                c += 1
            }
            r += 1
        }

        // WRec
        r = 0
        while r < hBase {
            var c = 0
            while c < hBase {
                baseWRec[r * hBase + c] = pWRec.data[r * maxHiddenDim + c]
                c += 1
            }
            r += 1
        }

        // BH
        r = 0
        while r < hBase {
            baseBH[r] = pBH.data[r]
            r += 1
        }

        // WOut
        var o = 0
        while o < outputDim {
            var c = 0
            while c < hBase {
                baseWOut[o * hBase + c] = pWOut.data[o * maxHiddenDim + c]
                c += 1
            }
            o += 1
        }

        // BOut
        o = 0
        while o < outputDim {
            baseBOut[o] = pBOut.data[o]
            o += 1
        }

        return BaseSNNWeights(
            inputDim: inputDim,
            hiddenDim: hBase,
            outputDim: outputDim,
            wIn: baseWIn,
            wRec: baseWRec,
            bH: baseBH,
            wOut: baseWOut,
            bOut: baseBOut
        )
    }

    /// Base 単体モデルのインポート
    public func importBaseWeights(_ base: BaseSNNWeights) {
        let hBase = min(MatryoshkaSlice.base.rawValue, maxHiddenDim)

        // WIn: 先頭 hBase 行
        var r = 0
        while r < hBase {
            var c = 0
            while c < inputDim {
                pWIn.data[r * inputDim + c] = base.wIn[r * inputDim + c]
                c += 1
            }
            r += 1
        }

        // WRec: 左上 hBase x hBase
        r = 0
        while r < hBase {
            var c = 0
            while c < hBase {
                pWRec.data[r * maxHiddenDim + c] = base.wRec[r * hBase + c]
                c += 1
            }
            r += 1
        }

        // BH: 先頭 hBase
        r = 0
        while r < hBase {
            pBH.data[r] = base.bH[r]
            r += 1
        }

        // WOut: outputDim 行 x 先頭 hBase 列
        var o = 0
        while o < outputDim {
            var c = 0
            while c < hBase {
                pWOut.data[o * maxHiddenDim + c] = base.wOut[o * hBase + c]
                c += 1
            }
            o += 1
        }

        // BOut
        o = 0
        while o < outputDim {
            pBOut.data[o] = base.bOut[o]
            o += 1
        }
    }

    /// 全体重みのエクスポート
    public func exportWeights() -> MatryoshkaWeightsData {
        return MatryoshkaWeightsData(
            inputDim: inputDim,
            maxHiddenDim: maxHiddenDim,
            outputDim: outputDim,
            timeSteps: timeSteps,
            beta: lifConfig.beta,
            vTh: lifConfig.vTh,
            vReset: lifConfig.vReset,
            alpha: lifConfig.alpha,
            wIn: pWIn.data,
            wRec: pWRec.data,
            bH: pBH.data,
            wOut: pWOut.data,
            bOut: pBOut.data
        )
    }

    /// 全体重みのインポート
    public func importWeights(from weightsData: MatryoshkaWeightsData) {
        if weightsData.wIn.count == pWIn.data.count {
            pWIn.data = weightsData.wIn
        }
        if weightsData.wRec.count == pWRec.data.count {
            pWRec.data = weightsData.wRec
        }
        if weightsData.bH.count == pBH.data.count {
            pBH.data = weightsData.bH
        }
        if weightsData.wOut.count == pWOut.data.count {
            pWOut.data = weightsData.wOut
        }
        if weightsData.bOut.count == pBOut.data.count {
            pBOut.data = weightsData.bOut
        }
    }

    /// 指定スライスでの推論（Hot Path ゼロアロケーション）
    public func forwardSlice(
        features: [Float],
        slice: MatryoshkaSlice,
        vPrev: inout [Float],
        sPrev: inout [Float],
        spikeSum: inout [Float],
        logits: inout [Float],
        probabilities: inout [Float]
    ) {
        if features.count < inputDim {
            return
        }
        let hSize = min(slice.rawValue, maxHiddenDim)
        
        // 1. 各ニューロンのスパイク積算リセット
        var i = 0
        while i < hSize {
            spikeSum[i] = 0.0
            i += 1
        }

        var vNext = [Float](repeating: 0.0, count: hSize)
        var sNext = [Float](repeating: 0.0, count: hSize)

        // 2. 時間ステップループ
        var t = 0
        while t < timeSteps {
            var n = 0
            while n < hSize {
                // 入力電流: bH + WIn * features
                var current = pBH.data[n]
                let inOffset = n * inputDim
                var d = 0
                while d < inputDim {
                    current += pWIn.data[inOffset + d] * features[d]
                    d += 1
                }

                // リカレント電流: WRec * sPrev (スパース加算)
                let recOffset = n * maxHiddenDim
                var j = 0
                while j < hSize {
                    if sPrev[j] != 0.0 {
                        current += pWRec.data[recOffset + j]
                    }
                    j += 1
                }

                // LIF ステップ
                let stepRes = LIFNeuronEngine.stepScalar(
                    config: lifConfig,
                    vPrev: vPrev[n],
                    sPrev: sPrev[n],
                    inputCurrent: current
                )
                vNext[n] = stepRes.vNext
                sNext[n] = stepRes.sNext
                spikeSum[n] += stepRes.sNext
                n += 1
            }

            // ステップ終了時に状態を更新
            n = 0
            while n < hSize {
                vPrev[n] = vNext[n]
                sPrev[n] = sNext[n]
                n += 1
            }

            t += 1
        }

        // 3. リードアウト層計算 (スライスサイズに応じたスケール正規化)
        let invT = 1.0 / Float(timeSteps)
        let sliceNorm = sqrt(Float(maxHiddenDim) / Float(hSize))
        var maxLogit: Float = -Float.greatestFiniteMagnitude
        var c = 0
        while c < outputDim {
            var sumW: Float = 0.0
            let wOffset = c * maxHiddenDim
            var k = 0
            while k < hSize {
                let spikeRate = spikeSum[k] * invT
                sumW += pWOut.data[wOffset + k] * spikeRate
                k += 1
            }
            let logit = pBOut.data[c] + (sumW * sliceNorm)
            logits[c] = logit
            if maxLogit < logit {
                maxLogit = logit
            }
            c += 1
        }

        // 4. Softmax
        var sumExp: Float = 0.0
        c = 0
        while c < outputDim {
            let expVal = exp(logits[c] - maxLogit)
            probabilities[c] = expVal
            sumExp += expVal
            c += 1
        }
        let invSum = 1.0 / sumExp
        c = 0
        while c < outputDim {
            probabilities[c] *= invSum
            c += 1
        }
    }
}
