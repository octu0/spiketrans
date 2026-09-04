import Foundation

/// スパイキングニューラルネットワーク本体 (Float32)
public final class SpikingNetwork: @unchecked Sendable {
    public let inputDim: Int
    public let maxHiddenDim: Int  // 4096
    public let outputDim: Int
    public let timeSteps: Int
    /// LIF / ALIF 設定 (重みインポート時に保存済みの値へ追従するため var)
    public private(set) var lifConfig: LIFConfig

    /// 推論用の転置コピー。wRecT[j * maxHiddenDim + n] = wRec[n][j] で、
    /// 発火ニューロン j の流出重みが連続に並ぶため Event-driven 加算を SIMD 化できる
    public private(set) var wRecT: [Float] = []
    /// 推論用の転置コピー。wOutT[k * outputDim + c] = wOut[c][k]
    public private(set) var wOutT: [Float] = []

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

        rebuildInferenceLayout()
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
        rebuildInferenceLayout()
    }

    /// 1 フレームの推論（ALIF 適応型発火閾値 & Event-driven 疎スパイク加算 & ゼロアロケーション）
    ///
    /// Hot Path でのヒープ再アロケーションを避けるため、中間バッファは呼び出し側が
    /// `ForwardScratch` として事前確保して渡す。
    /// 推論用の転置レイアウトを再構築する。
    /// CPU 学習などで pWRec / pWOut を直接更新した後は必ず呼ぶこと
    /// (init と importWeights では自動で呼ばれる)
    public func rebuildInferenceLayout() {
        let hSize = maxHiddenDim
        if wRecT.count != hSize * hSize {
            wRecT = [Float](repeating: 0.0, count: hSize * hSize)
        }
        pWRec.data.withUnsafeBufferPointer { src in
            wRecT.withUnsafeMutableBufferPointer { dst in
                var n = 0
                while n < hSize {
                    let rowOffset = n * hSize
                    var j = 0
                    while j < hSize {
                        dst[j * hSize + n] = src[rowOffset + j]
                        j += 1
                    }
                    n += 1
                }
            }
        }
        if wOutT.count != hSize * outputDim {
            wOutT = [Float](repeating: 0.0, count: hSize * outputDim)
        }
        pWOut.data.withUnsafeBufferPointer { src in
            wOutT.withUnsafeMutableBufferPointer { dst in
                var c = 0
                while c < outputDim {
                    let rowOffset = c * hSize
                    var k = 0
                    while k < hSize {
                        dst[k * outputDim + c] = src[rowOffset + k]
                        k += 1
                    }
                    c += 1
                }
            }
        }
    }

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

        // 2. 入力電流の事前計算 (全タイムステップで不変のためループ外へ Hoist)。
        //    レーン = ニューロンの SIMD8 化。各レーンは独立した行内積なので
        //    スカラー版と加算順序が変わらず、結果はビット一致する
        let hLimit = hSize - (hSize % 8)
        pWIn.data.withUnsafeBufferPointer { wInBuf in
            features.withUnsafeBufferPointer { featBuf in
                pBH.data.withUnsafeBufferPointer { bhBuf in
                    scratch.inputCurrents.withUnsafeMutableBufferPointer { curBuf in
                        let wIn = wInBuf.baseAddress!
                        let feat = featBuf.baseAddress!
                        let bh = bhBuf.baseAddress!
                        let cur = curBuf.baseAddress!
                        var n = 0
                        while n < hLimit {
                            var acc = SIMD8<Float>(
                                bh[n+0], bh[n+1], bh[n+2], bh[n+3],
                                bh[n+4], bh[n+5], bh[n+6], bh[n+7]
                            )
                            let r0 = (n+0) * inputDim
                            let r1 = (n+1) * inputDim
                            let r2 = (n+2) * inputDim
                            let r3 = (n+3) * inputDim
                            let r4 = (n+4) * inputDim
                            let r5 = (n+5) * inputDim
                            let r6 = (n+6) * inputDim
                            let r7 = (n+7) * inputDim
                            var d = 0
                            while d < inputDim {
                                let w = SIMD8<Float>(
                                    wIn[r0+d], wIn[r1+d], wIn[r2+d], wIn[r3+d],
                                    wIn[r4+d], wIn[r5+d], wIn[r6+d], wIn[r7+d]
                                )
                                acc = acc + (w * SIMD8<Float>(repeating: feat[d]))
                                d += 1
                            }
                            cur[n+0] = acc[0]
                            cur[n+1] = acc[1]
                            cur[n+2] = acc[2]
                            cur[n+3] = acc[3]
                            cur[n+4] = acc[4]
                            cur[n+5] = acc[5]
                            cur[n+6] = acc[6]
                            cur[n+7] = acc[7]
                            n += 8
                        }
                        while n < hSize {
                            var curr = bh[n]
                            let inOffset = n * inputDim
                            var d = 0
                            while d < inputDim {
                                curr += wIn[inOffset + d] * feat[d]
                                d += 1
                            }
                            cur[n] = curr
                            n += 1
                        }
                    }
                }
            }
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

            // 3.2 ステップ電流 = 入力電流 + 発火ニューロンの流出重み。
            //     転置レイアウト wRecT により発火 j の行が連続 → SIMD 加算。
            //     加算順序 (j 昇順) はスカラー版と同じでビット一致する
            scratch.stepCurrents.withUnsafeMutableBufferPointer { stepBuf in
                let step = stepBuf.baseAddress!
                scratch.inputCurrents.withUnsafeBufferPointer { inBuf in
                    step.update(from: inBuf.baseAddress!, count: hSize)
                }
                wRecT.withUnsafeBufferPointer { recBuf in
                    let recT = recBuf.baseAddress!
                    var a = 0
                    while a < activeCount {
                        let rowOffset = scratch.activeSpikes[a] * hSize
                        var n = 0
                        while n < hLimit {
                            let acc = SIMD8<Float>(
                                step[n+0], step[n+1], step[n+2], step[n+3],
                                step[n+4], step[n+5], step[n+6], step[n+7]
                            )
                            let w = SIMD8<Float>(
                                recT[rowOffset+n+0], recT[rowOffset+n+1],
                                recT[rowOffset+n+2], recT[rowOffset+n+3],
                                recT[rowOffset+n+4], recT[rowOffset+n+5],
                                recT[rowOffset+n+6], recT[rowOffset+n+7]
                            )
                            let sum = acc + w
                            step[n+0] = sum[0]
                            step[n+1] = sum[1]
                            step[n+2] = sum[2]
                            step[n+3] = sum[3]
                            step[n+4] = sum[4]
                            step[n+5] = sum[5]
                            step[n+6] = sum[6]
                            step[n+7] = sum[7]
                            n += 8
                        }
                        while n < hSize {
                            step[n] += recT[rowOffset + n]
                            n += 1
                        }
                        a += 1
                    }
                }
            }

            // 3.3 ALIF 更新の SIMD8 一括処理 (状態は直接更新し vNext コピーを省く)
            LIFNeuronEngine.stepAdaptiveSIMD8Array(
                config: lifConfig,
                currents: scratch.stepCurrents,
                vState: &vPrev,
                sState: &sPrev,
                aState: &aPrev,
                spikeSum: &spikeSum,
                count: hSize
            )

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

        // 転置レイアウト wOutT により、有効ニューロン k の重み行 (出力次元方向) が
        // 連続に並ぶ。集計は k ごとに同じ順序で加算するためスカラー版とビット一致する
        if scratch.readoutSums.count < outputDim {
            scratch.readoutSums = [Float](repeating: 0.0, count: outputDim)
        }
        let outLimit = outputDim - (outputDim % 8)
        scratch.readoutSums.withUnsafeMutableBufferPointer { sumBuf in
            let sums = sumBuf.baseAddress!
            var c = 0
            while c < outputDim {
                sums[c] = 0.0
                c += 1
            }
            wOutT.withUnsafeBufferPointer { outBuf in
                let outT = outBuf.baseAddress!
                var a = 0
                while a < activeOutCount {
                    let rowOffset = scratch.activeReadoutIndices[a] * outputDim
                    let rate = SIMD8<Float>(repeating: scratch.activeRates[a])
                    var c = 0
                    while c < outLimit {
                        let acc = SIMD8<Float>(
                            sums[c+0], sums[c+1], sums[c+2], sums[c+3],
                            sums[c+4], sums[c+5], sums[c+6], sums[c+7]
                        )
                        let w = SIMD8<Float>(
                            outT[rowOffset+c+0], outT[rowOffset+c+1],
                            outT[rowOffset+c+2], outT[rowOffset+c+3],
                            outT[rowOffset+c+4], outT[rowOffset+c+5],
                            outT[rowOffset+c+6], outT[rowOffset+c+7]
                        )
                        let sum = acc + (w * rate)
                        sums[c+0] = sum[0]
                        sums[c+1] = sum[1]
                        sums[c+2] = sum[2]
                        sums[c+3] = sum[3]
                        sums[c+4] = sum[4]
                        sums[c+5] = sum[5]
                        sums[c+6] = sum[6]
                        sums[c+7] = sum[7]
                        c += 8
                    }
                    while c < outputDim {
                        sums[c] += outT[rowOffset + c] * scratch.activeRates[a]
                        c += 1
                    }
                    a += 1
                }
            }
        }

        var maxLogit: Float = -Float.greatestFiniteMagnitude
        var c = 0
        while c < outputDim {
            let logit = biasData[c] + scratch.readoutSums[c]
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
    public var stepCurrents: [Float]
    public var vNext: [Float]
    public var sNext: [Float]
    public var aNext: [Float]
    public var activeSpikes: [Int]
    public var activeReadoutIndices: [Int]
    public var activeRates: [Float]
    public var readoutSums: [Float]

    public init(maxHiddenDim: Int) {
        let size = max(1, maxHiddenDim)
        self.inputCurrents = [Float](repeating: 0.0, count: size)
        self.stepCurrents = [Float](repeating: 0.0, count: size)
        self.vNext = [Float](repeating: 0.0, count: size)
        self.sNext = [Float](repeating: 0.0, count: size)
        self.aNext = [Float](repeating: 0.0, count: size)
        self.activeSpikes = [Int](repeating: 0, count: size)
        self.activeReadoutIndices = [Int](repeating: 0, count: size)
        self.activeRates = [Float](repeating: 0.0, count: size)
        self.readoutSums = []
    }
}
