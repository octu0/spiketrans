import Foundation

/// スパイキングニューラルネットワーク本体 (Float32)
public final class SpikingNetwork: @unchecked Sendable {
    public let inputDim: Int
    public let maxHiddenDim: Int  // 4096
    public let outputDim: Int
    public let timeSteps: Int
    /// LIF / ALIF 設定 (重みインポート時に保存済みの値へ追従するため var)
    public private(set) var lifConfig: LIFConfig
    /// ニューロンごとの膜電位減衰率ベクトル (多重時定数対応)
    public private(set) var betaVector: [Float]

    public let pWIn: Parameter
    public let pWRec: Parameter
    public let pBH: Parameter
    public let pWOut: Parameter        // 全スライスで共有 [outputDim, maxHiddenDim]
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
        self.betaVector = lifConfig.betaVector(count: maxHiddenDim)

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

    /// 全体重みのエクスポート
    public func exportWeights() -> SpikingNetworkWeights {
        return SpikingNetworkWeights(
            inputDim: inputDim,
            maxHiddenDim: maxHiddenDim,
            outputDim: outputDim,
            timeSteps: timeSteps,
            lifConfig: lifConfig,
            wIn: pWIn.data,
            wRec: pWRec.data,
            bH: pBH.data,
            wOut: pWOut.data,
            bOut: pBOut.data
        )
    }

    /// 全体重みのインポート (学習時の LIF / ALIF パラメータも同時に復元)
    public func importWeights(from weightsData: SpikingNetworkWeights) {
        // 学習時と推論時で発火ダイナミクスがずれないよう ALIF パラメータごと引き継ぐ
        self.lifConfig = weightsData.lifConfig
        self.betaVector = self.lifConfig.betaVector(count: maxHiddenDim)
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

    /// 1 フレームの推論（ALIF 適応型発火閾値 & Event-driven 疎スパイク加算 & ゼロアロケーション）
    ///
    /// Hot Path でのヒープ再アロケーションを避けるため、中間バッファは呼び出し側が
    /// `ForwardScratch` として事前確保して渡す。
    public func forward(
        features: [Float],
        vPrev: inout [Float],
        sPrev: inout [Float],
        aPrev: inout [Float],
        spikeSum: inout [Float],
        logits: inout [Float],
        probabilities: inout [Float],
        scratch: ForwardScratch
    ) {
        if features.count < inputDim {
            return
        }
        let hSize = maxHiddenDim

        // 1. 各ニューロンのスパイク積算リセット
        var i = 0
        while i < hSize {
            spikeSum[i] = 0.0
            i += 1
        }

        // 2. 入力電流の事前計算 (全タイムステップで不変のためループ外へ Hoist)
        var n = 0
        while n < hSize {
            var curr = pBH.data[n]
            let inOffset = n * inputDim
            var d = 0
            while d < inputDim {
                curr += pWIn.data[inOffset + d] * features[d]
                d += 1
            }
            scratch.inputCurrents[n] = curr
            n += 1
        }

        // 3. 時間ステップループ (ALIF 適応型閾値 & Event-driven 疎スパイク加算)
        var t = 0
        while t < timeSteps {
            // 3.1 直前ステップで発火したニューロンのインデックスのみを収集
            var activeCount = 0
            var j = 0
            while j < hSize {
                if sPrev[j] != 0.0 {
                    scratch.activeSpikes[activeCount] = j
                    activeCount += 1
                }
                j += 1
            }

            n = 0
            while n < hSize {
                var current = scratch.inputCurrents[n]
                let recOffset = n * maxHiddenDim

                // 発火ニューロンの重みのみを加算 (ゼロ乗算を完全スキップ)
                var a = 0
                while a < activeCount {
                    let spikeIdx = scratch.activeSpikes[a]
                    current += pWRec.data[recOffset + spikeIdx]
                    a += 1
                }

                // ALIF 適応型ステップ (多重時定数対応)
                let stepRes = LIFNeuronEngine.stepScalarAdaptive(
                    config: lifConfig,
                    beta: betaVector[n],
                    vPrev: vPrev[n],
                    sPrev: sPrev[n],
                    aPrev: aPrev[n],
                    inputCurrent: current
                )
                scratch.vNext[n] = stepRes.vNext
                scratch.sNext[n] = stepRes.sNext
                scratch.aNext[n] = stepRes.aNext
                spikeSum[n] += stepRes.sNext
                n += 1
            }

            // ステップ終了時に状態を更新
            n = 0
            while n < hSize {
                vPrev[n] = scratch.vNext[n]
                sPrev[n] = scratch.sNext[n]
                aPrev[n] = scratch.aNext[n]
                n += 1
            }

            t += 1
        }

        // 4. リードアウト層計算 (Event-driven 疎加算 & スケール正規化)
        let invT = 1.0 / Float(timeSteps)
        let biasData = pBOut.data

        var activeOutCount = 0
        var k = 0
        while k < hSize {
            let sRate = spikeSum[k] * invT
            if sRate != 0.0 {
                scratch.activeReadoutIndices[activeOutCount] = k
                scratch.activeRates[activeOutCount] = sRate
                activeOutCount += 1
            }
            k += 1
        }

        var maxLogit: Float = -Float.greatestFiniteMagnitude
        var c = 0
        while c < outputDim {
            var sumW: Float = 0.0
            let wOffset = c * maxHiddenDim
            var a = 0
            while a < activeOutCount {
                let idx = scratch.activeReadoutIndices[a]
                sumW += pWOut.data[wOffset + idx] * scratch.activeRates[a]
                a += 1
            }
            let logit = biasData[c] + sumW
            logits[c] = logit
            if maxLogit < logit {
                maxLogit = logit
            }
            c += 1
        }

        // 5. Softmax
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

    /// 中間バッファと適応閾値状態を呼び出し側で保持しない簡便版 (Hot Path 以外)
    ///
    /// 呼び出しごとに ALIF の適応状態がゼロから始まるため、フレームを跨いだ
    /// 神経順応は再現されない。連続フレーム推論では aPrev を保持する版を使うこと。
    public func forward(
        features: [Float],
        vPrev: inout [Float],
        sPrev: inout [Float],
        spikeSum: inout [Float],
        logits: inout [Float],
        probabilities: inout [Float]
    ) {
        var aPrev = [Float](repeating: 0.0, count: maxHiddenDim)
        let scratch = ForwardScratch(maxHiddenDim: maxHiddenDim)
        forward(
            features: features,
            vPrev: &vPrev,
            sPrev: &sPrev,
            aPrev: &aPrev,
            spikeSum: &spikeSum,
            logits: &logits,
            probabilities: &probabilities,
            scratch: scratch
        )
    }
}

/// forward の Hot Path 用事前確保中間バッファ (ゼロアロケーション維持)
///
/// スレッドセーフではないため、並列推論では推論スレッドごとに 1 つ確保すること。
public final class ForwardScratch: @unchecked Sendable {
    public var inputCurrents: [Float]
    public var vNext: [Float]
    public var sNext: [Float]
    public var aNext: [Float]
    public var activeSpikes: [Int]
    public var activeReadoutIndices: [Int]
    public var activeRates: [Float]

    public init(maxHiddenDim: Int) {
        let size = max(1, maxHiddenDim)
        self.inputCurrents = [Float](repeating: 0.0, count: size)
        self.vNext = [Float](repeating: 0.0, count: size)
        self.sNext = [Float](repeating: 0.0, count: size)
        self.aNext = [Float](repeating: 0.0, count: size)
        self.activeSpikes = [Int](repeating: 0, count: size)
        self.activeReadoutIndices = [Int](repeating: 0, count: size)
        self.activeRates = [Float](repeating: 0.0, count: size)
    }
}
