import Foundation

/// スパイキングニューラルネットワーク本体 (Float32)
public final class SpikingNetwork: @unchecked Sendable {
    public let numLayers: Int
    public let inputDim: Int
    public let maxHiddenDim: Int  // 4096
    public let outputDim: Int
    public let timeSteps: Int
    /// LIF / ALIF 設定 (重みインポート時に保存済みの値へ追従するため var)
    public private(set) var lifConfig: LIFConfig
    /// 2D-Conv Subsampling フロントエンド (オプション)
    public var convSubsampling: Conv2DSubsampling?

    /// 推論用の転置コピー。wRecT[j * maxHiddenDim + n] = wRec[n][j] で、
    /// 発火ニューロン j の流出重みが連続に並ぶため Event-driven 加算を SIMD 化できる
    public private(set) var wRecT: [Float] = []
    /// 上位層（Layer 1 以降）の順伝播用転置重み
    public private(set) var wLayersT: [[Float]] = []
    /// 推論用の転置コピー。wOutT[k * outputDim + c] = wOut[c][k]
    public private(set) var wOutT: [Float] = []

    // 第0層パラメータ
    public let pWIn: Parameter
    public let pWRec: Parameter
    public let pBH: Parameter

    // 上位層（Layer 1 以降）パラメータ
    public let pWLayers: [Parameter]
    public let pBHLayers: [Parameter]
    public let pGammaRMS: [Parameter]

    // リードアウト層パラメータ
    public let pWOut: Parameter        // [outputDim, maxHiddenDim]
    public let pBOut: Parameter

    public init(
        numLayers: Int = 1,
        inputDim: Int = 64,
        maxHiddenDim: Int = 4096,
        outputDim: Int = 64,
        timeSteps: Int = 4,
        lifConfig: LIFConfig = LIFConfig(),
        convSubsampling: Conv2DSubsampling? = nil
    ) {
        self.numLayers = max(1, numLayers)
        self.inputDim = inputDim
        self.maxHiddenDim = maxHiddenDim
        self.outputDim = outputDim
        self.timeSteps = timeSteps
        self.lifConfig = lifConfig
        self.convSubsampling = convSubsampling

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

        // 上位層パラメータの初期化
        var wLayersList: [Parameter] = []
        var bhLayersList: [Parameter] = []
        var gammaRMSList: [Parameter] = []
        let scaleLayer = sqrt(2.0 / Float(maxHiddenDim))

        var l = 1
        while l < self.numLayers {
            var initWLayer = [Float](repeating: 0.0, count: maxHiddenDim * maxHiddenDim)
            var idx = 0
            while idx < maxHiddenDim * maxHiddenDim {
                initWLayer[idx] = Float.random(in: -scaleLayer...scaleLayer)
                idx += 1
            }
            let initBHL = [Float](repeating: 0.0, count: maxHiddenDim)
            let initGamma = [Float](repeating: 1.0, count: maxHiddenDim)

            wLayersList.append(Parameter(count: maxHiddenDim * maxHiddenDim, initialData: initWLayer))
            bhLayersList.append(Parameter(count: maxHiddenDim, initialData: initBHL))
            gammaRMSList.append(Parameter(count: maxHiddenDim, initialData: initGamma))
            l += 1
        }

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
        self.pWLayers = wLayersList
        self.pBHLayers = bhLayersList
        self.pGammaRMS = gammaRMSList
        self.pWOut = Parameter(count: outputDim * maxHiddenDim, initialData: initWOut)
        self.pBOut = Parameter(count: outputDim, initialData: initBOut)

        rebuildInferenceLayout()
    }

    /// 保存済み重みから完全に復元して初期化
    public convenience init(weights: SpikingNetworkWeights) {
        let subsampler: Conv2DSubsampling?
        switch weights.convSubsampling {
        case .some(let cWeights):
            subsampler = Conv2DSubsampling(weights: cWeights)
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

    public var parameters: [Parameter] {
        var params: [Parameter] = [pWIn, pWRec, pBH]
        var l = 0
        while l < pWLayers.count {
            params.append(pWLayers[l])
            params.append(pBHLayers[l])
            params.append(pGammaRMS[l])
            l += 1
        }
        params.append(pWOut)
        params.append(pBOut)
        return params
    }

    /// 全体重みのエクスポート
    public func exportWeights(vocabulary: TextVocabulary? = nil) -> SpikingNetworkWeights {
        var exportedWLayers: [[Float]]? = nil
        var exportedBHLayers: [[Float]]? = nil
        var exportedGammaRMS: [[Float]]? = nil
        if 1 < numLayers {
            var wl: [[Float]] = []
            var bl: [[Float]] = []
            var gl: [[Float]] = []
            var l = 0
            while l < pWLayers.count {
                wl.append(pWLayers[l].data)
                bl.append(pBHLayers[l].data)
                gl.append(pGammaRMS[l].data)
                l += 1
            }
            exportedWLayers = wl
            exportedBHLayers = bl
            exportedGammaRMS = gl
        }
        return SpikingNetworkWeights(
            numLayers: numLayers,
            inputDim: inputDim,
            maxHiddenDim: maxHiddenDim,
            outputDim: outputDim,
            timeSteps: timeSteps,
            lifConfig: lifConfig,
            wIn: pWIn.data,
            wRec: pWRec.data,
            bH: pBH.data,
            wLayers: exportedWLayers,
            bHLayers: exportedBHLayers,
            gammaRMS: exportedGammaRMS,
            wOut: pWOut.data,
            bOut: pBOut.data,
            vocabularyCharacters: vocabulary?.serializedCharacters,
            convSubsampling: convSubsampling?.exportWeights()
        )
    }

    /// 全体重みのインポート (学習時の LIF / ALIF パラメータも同時に復元)
    public func importWeights(from weightsData: SpikingNetworkWeights) {
        self.lifConfig = weightsData.lifConfig
        switch weightsData.convSubsampling {
        case .some(let cWeights):
            self.convSubsampling = Conv2DSubsampling(weights: cWeights)
        case .none:
            self.convSubsampling = nil
        }
        if weightsData.wIn.count == pWIn.data.count {
            pWIn.data = weightsData.wIn
        }
        if weightsData.wRec.count == pWRec.data.count {
            pWRec.data = weightsData.wRec
        }
        if weightsData.bH.count == pBH.data.count {
            pBH.data = weightsData.bH
        }
        switch weightsData.wLayers {
        case .some(let wl):
            var l = 0
            while l < min(pWLayers.count, wl.count) {
                if wl[l].count == pWLayers[l].data.count {
                    pWLayers[l].data = wl[l]
                }
                l += 1
            }
        case .none:
            break
        }
        switch weightsData.bHLayers {
        case .some(let bl):
            var l = 0
            while l < min(pBHLayers.count, bl.count) {
                if bl[l].count == pBHLayers[l].data.count {
                    pBHLayers[l].data = bl[l]
                }
                l += 1
            }
        case .none:
            break
        }
        switch weightsData.gammaRMS {
        case .some(let gl):
            var l = 0
            while l < min(pGammaRMS.count, gl.count) {
                if gl[l].count == pGammaRMS[l].data.count {
                    pGammaRMS[l].data = gl[l]
                }
                l += 1
            }
        case .none:
            break
        }
        if weightsData.wOut.count == pWOut.data.count {
            pWOut.data = weightsData.wOut
        }
        if weightsData.bOut.count == pBOut.data.count {
            pBOut.data = weightsData.bOut
        }
        rebuildInferenceLayout()
    }

    /// 推論用の転置レイアウトを再構築する。
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
        var l = 0
        while l < pWLayers.count {
            if wLayersT.count <= l {
                wLayersT.append([Float](repeating: 0.0, count: hSize * hSize))
            }
            if wLayersT[l].count != hSize * hSize {
                wLayersT[l] = [Float](repeating: 0.0, count: hSize * hSize)
            }
            let srcLayer = pWLayers[l].data
            srcLayer.withUnsafeBufferPointer { src in
                wLayersT[l].withUnsafeMutableBufferPointer { dst in
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
            l += 1
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
        let totalStateSize = numLayers * hSize
        if vPrev.count < totalStateSize {
            vPrev = [Float](repeating: 0.0, count: totalStateSize)
            sPrev = [Float](repeating: 0.0, count: totalStateSize)
            aPrev = [Float](repeating: 0.0, count: totalStateSize)
        }

        // 1. 各ニューロンのスパイク積算リセット
        var i = 0
        while i < hSize {
            spikeSum[i] = 0.0
            i += 1
        }

        // 2. 入力電流の事前計算 (全タイムステップで不変のためループ外へ Hoist)。
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
            // 3.1 第0層 (時系列文脈・共調音を保持する再帰層)
            var activeCount0 = 0
            var j = 0
            while j < hSize {
                if sPrev[j] != 0.0 {
                    scratch.activeSpikes[activeCount0] = j
                    activeCount0 += 1
                }
                j += 1
            }

            scratch.stepCurrents.withUnsafeMutableBufferPointer { stepBuf in
                let step = stepBuf.baseAddress!
                scratch.inputCurrents.withUnsafeBufferPointer { inBuf in
                    step.update(from: inBuf.baseAddress!, count: hSize)
                }
                wRecT.withUnsafeBufferPointer { recBuf in
                    let recT = recBuf.baseAddress!
                    var a = 0
                    while a < activeCount0 {
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

            if 1 < numLayers {
                scratch.stepCurrentsPrev.withUnsafeMutableBufferPointer { prevBuf in
                    scratch.stepCurrents.withUnsafeBufferPointer { curBuf in
                        prevBuf.baseAddress!.update(from: curBuf.baseAddress!, count: hSize)
                    }
                }
            }

            vPrev.withUnsafeMutableBufferPointer { vBuf in
                sPrev.withUnsafeMutableBufferPointer { sBuf in
                    aPrev.withUnsafeMutableBufferPointer { aBuf in
                        scratch.stepCurrents.withUnsafeBufferPointer { curBuf in
                            spikeSum.withUnsafeMutableBufferPointer { sumBuf in
                                var sumTarget: UnsafeMutablePointer<Float>? = nil
                                if numLayers == 1 {
                                    sumTarget = sumBuf.baseAddress!
                                }
                                LIFNeuronEngine.stepAdaptiveSIMD8(
                                    config: lifConfig,
                                    vPtr: vBuf.baseAddress!,
                                    sPtr: sBuf.baseAddress!,
                                    aPtr: aBuf.baseAddress!,
                                    curPtr: curBuf.baseAddress!,
                                    spikeSumPtr: sumTarget,
                                    count: hSize
                                )
                            }
                        }
                    }
                }
            }

            // 3.2 上位層 (Layer 1 以降: 電流RMSNorm & Membrane-Shortcut)
            var layerIdx = 1
            while layerIdx < numLayers {
                let prevLayerOffset = (layerIdx - 1) * hSize
                let thisLayerOffset = layerIdx * hSize
                let upperIdx = layerIdx - 1

                // 前層のスパイク発火インデックスを収集
                var activeLayerCount = 0
                var kj = 0
                while kj < hSize {
                    if sPrev[prevLayerOffset + kj] != 0.0 {
                        scratch.activeLayerSpikes[activeLayerCount] = kj
                        activeLayerCount += 1
                    }
                    kj += 1
                }

                // 結合電流 = bHLayers + wLayersT * activeSpikes
                let bData = pBHLayers[upperIdx].data
                let wLayerTData = wLayersT[upperIdx]
                scratch.stepCurrents.withUnsafeMutableBufferPointer { stepBuf in
                    let step = stepBuf.baseAddress!
                    bData.withUnsafeBufferPointer { bBuf in
                        step.update(from: bBuf.baseAddress!, count: hSize)
                    }
                    wLayerTData.withUnsafeBufferPointer { wBuf in
                        let wT = wBuf.baseAddress!
                        var a = 0
                        while a < activeLayerCount {
                            let rowOffset = scratch.activeLayerSpikes[a] * hSize
                            var n = 0
                            while n < hLimit {
                                let acc = SIMD8<Float>(
                                    step[n+0], step[n+1], step[n+2], step[n+3],
                                    step[n+4], step[n+5], step[n+6], step[n+7]
                                )
                                let w = SIMD8<Float>(
                                    wT[rowOffset+n+0], wT[rowOffset+n+1],
                                    wT[rowOffset+n+2], wT[rowOffset+n+3],
                                    wT[rowOffset+n+4], wT[rowOffset+n+5],
                                    wT[rowOffset+n+6], wT[rowOffset+n+7]
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
                                step[n] += wT[rowOffset + n]
                                n += 1
                            }
                            a += 1
                        }
                    }
                }

                // 電流 RMSNorm & Membrane-Shortcut (電流スキップ接続)
                let gammaData = pGammaRMS[upperIdx].data
                scratch.stepCurrents.withUnsafeMutableBufferPointer { stepBuf in
                    scratch.stepCurrentsPrev.withUnsafeMutableBufferPointer { prevBuf in
                        gammaData.withUnsafeBufferPointer { gammaBuf in
                            let step = stepBuf.baseAddress!
                            let prev = prevBuf.baseAddress!
                            let gamma = gammaBuf.baseAddress!

                            // RMS 計算
                            var sumSqVec = SIMD8<Float>(repeating: 0.0)
                            var n = 0
                            while n < hLimit {
                                let v = SIMD8<Float>(
                                    step[n+0], step[n+1], step[n+2], step[n+3],
                                    step[n+4], step[n+5], step[n+6], step[n+7]
                                )
                                sumSqVec = sumSqVec + (v * v)
                                n += 8
                            }
                            var totalSq = (sumSqVec[0] + sumSqVec[1] + sumSqVec[2] + sumSqVec[3]) +
                                          (sumSqVec[4] + sumSqVec[5] + sumSqVec[6] + sumSqVec[7])
                            while n < hSize {
                                totalSq += step[n] * step[n]
                                n += 1
                            }
                            let meanSq = totalSq / Float(hSize)
                            let rms = sqrt(meanSq + 1e-5)
                            let invRms = 1.0 / rms
                            let invRmsVec = SIMD8<Float>(repeating: invRms)

                            // RMSNorm 適用 + Membrane-Shortcut (前層電流 prev との直和)
                            n = 0
                            while n < hLimit {
                                let raw = SIMD8<Float>(
                                    step[n+0], step[n+1], step[n+2], step[n+3],
                                    step[n+4], step[n+5], step[n+6], step[n+7]
                                )
                                let g = SIMD8<Float>(
                                    gamma[n+0], gamma[n+1], gamma[n+2], gamma[n+3],
                                    gamma[n+4], gamma[n+5], gamma[n+6], gamma[n+7]
                                )
                                let p = SIMD8<Float>(
                                    prev[n+0], prev[n+1], prev[n+2], prev[n+3],
                                    prev[n+4], prev[n+5], prev[n+6], prev[n+7]
                                )
                                let norm = (raw * invRmsVec) * g
                                let totalCur = norm + p
                                step[n+0] = totalCur[0]
                                step[n+1] = totalCur[1]
                                step[n+2] = totalCur[2]
                                step[n+3] = totalCur[3]
                                step[n+4] = totalCur[4]
                                step[n+5] = totalCur[5]
                                step[n+6] = totalCur[6]
                                step[n+7] = totalCur[7]
                                n += 8
                            }
                            while n < hSize {
                                let norm = (step[n] * invRms) * gamma[n]
                                step[n] = norm + prev[n]
                                n += 1
                            }

                            // 次の上位層があればショートカット用に保存
                            if (layerIdx + 1) < numLayers {
                                prev.update(from: step, count: hSize)
                            }
                        }
                    }
                }

                // 上位層 LIF 更新
                let isFinalLayer = (layerIdx + 1) == numLayers
                vPrev.withUnsafeMutableBufferPointer { vBuf in
                    sPrev.withUnsafeMutableBufferPointer { sBuf in
                        aPrev.withUnsafeMutableBufferPointer { aBuf in
                            scratch.stepCurrents.withUnsafeBufferPointer { curBuf in
                                spikeSum.withUnsafeMutableBufferPointer { sumBuf in
                                    var sumTarget: UnsafeMutablePointer<Float>? = nil
                                    if isFinalLayer {
                                        sumTarget = sumBuf.baseAddress!
                                    }
                                    LIFNeuronEngine.stepAdaptiveSIMD8(
                                        config: lifConfig,
                                        vPtr: vBuf.baseAddress!.advanced(by: thisLayerOffset),
                                        sPtr: sBuf.baseAddress!.advanced(by: thisLayerOffset),
                                        aPtr: aBuf.baseAddress!.advanced(by: thisLayerOffset),
                                        curPtr: curBuf.baseAddress!,
                                        spikeSumPtr: sumTarget,
                                        count: hSize
                                    )
                                }
                            }
                        }
                    }
                }

                layerIdx += 1
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
    public func forward(
        features: [Float],
        vPrev: inout [Float],
        sPrev: inout [Float],
        spikeSum: inout [Float],
        logits: inout [Float],
        probabilities: inout [Float]
    ) {
        var aPrev = [Float](repeating: 0.0, count: numLayers * maxHiddenDim)
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
public final class ForwardScratch: @unchecked Sendable {
    public var inputCurrents: [Float]
    public var stepCurrents: [Float]
    public var stepCurrentsPrev: [Float]
    public var vNext: [Float]
    public var sNext: [Float]
    public var aNext: [Float]
    public var activeSpikes: [Int]
    public var activeLayerSpikes: [Int]
    public var activeReadoutIndices: [Int]
    public var activeRates: [Float]
    public var readoutSums: [Float]

    public init(maxHiddenDim: Int) {
        let size = max(1, maxHiddenDim)
        self.inputCurrents = [Float](repeating: 0.0, count: size)
        self.stepCurrents = [Float](repeating: 0.0, count: size)
        self.stepCurrentsPrev = [Float](repeating: 0.0, count: size)
        self.vNext = [Float](repeating: 0.0, count: size)
        self.sNext = [Float](repeating: 0.0, count: size)
        self.aNext = [Float](repeating: 0.0, count: size)
        self.activeSpikes = [Int](repeating: 0, count: size)
        self.activeLayerSpikes = [Int](repeating: 0, count: size)
        self.activeReadoutIndices = [Int](repeating: 0, count: size)
        self.activeRates = [Float](repeating: 0.0, count: size)
        self.readoutSums = []
    }
}
