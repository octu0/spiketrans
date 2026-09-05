import Foundation

/// BPTT 計算用の中間キャッシュ
public final class ForwardCache: @unchecked Sendable {
    public let seqLen: Int
    public let timeSteps: Int
    public let hiddenDim: Int
    public let outputDim: Int
    public let numLayers: Int

    public var vStates: [[[Float]]]  // [seqLen][timeSteps][hiddenDim] (第0層)
    public var sStates: [[[Float]]]  // [seqLen][timeSteps][hiddenDim] (第0層)
    public var aStates: [[[Float]]]  // [seqLen][timeSteps][hiddenDim] (第0層 ALIF)
    public var vStatesLayers: [[[[Float]]]] // [numLayers-1][seqLen][timeSteps][hiddenDim]
    public var sStatesLayers: [[[[Float]]]] // [numLayers-1][seqLen][timeSteps][hiddenDim]
    public var aStatesLayers: [[[[Float]]]] // [numLayers-1][seqLen][timeSteps][hiddenDim]
    public var denseCurLayers: [[[[Float]]]] // [numLayers-1][seqLen][timeSteps][hiddenDim]
    public var rmsLayers: [[[Float]]]        // [numLayers-1][seqLen][timeSteps]
    public var spikeAvg: [[Float]]   // [seqLen][hiddenDim] (最終層)
    public var logits: [[Float]]     // [seqLen][outputDim]
    public var probs: [[Float]]      // [seqLen][outputDim]

    public init(seqLen: Int, timeSteps: Int, hiddenDim: Int, outputDim: Int, numLayers: Int = 1) {
        self.seqLen = seqLen
        self.timeSteps = timeSteps
        self.hiddenDim = hiddenDim
        self.outputDim = outputDim
        self.numLayers = numLayers

        self.vStates = [[[Float]]](
            repeating: [[Float]](repeating: [Float](repeating: 0.0, count: hiddenDim), count: timeSteps),
            count: seqLen
        )
        self.sStates = [[[Float]]](
            repeating: [[Float]](repeating: [Float](repeating: 0.0, count: hiddenDim), count: timeSteps),
            count: seqLen
        )
        self.aStates = [[[Float]]](
            repeating: [[Float]](repeating: [Float](repeating: 0.0, count: hiddenDim), count: timeSteps),
            count: seqLen
        )

        if 1 < numLayers {
            let upperCount = numLayers - 1
            self.vStatesLayers = [[[[Float]]]](
                repeating: [[[Float]]](
                    repeating: [[Float]](repeating: [Float](repeating: 0.0, count: hiddenDim), count: timeSteps),
                    count: seqLen
                ),
                count: upperCount
            )
            self.sStatesLayers = [[[[Float]]]](
                repeating: [[[Float]]](
                    repeating: [[Float]](repeating: [Float](repeating: 0.0, count: hiddenDim), count: timeSteps),
                    count: seqLen
                ),
                count: upperCount
            )
            self.aStatesLayers = [[[[Float]]]](
                repeating: [[[Float]]](
                    repeating: [[Float]](repeating: [Float](repeating: 0.0, count: hiddenDim), count: timeSteps),
                    count: seqLen
                ),
                count: upperCount
            )
            self.denseCurLayers = [[[[Float]]]](
                repeating: [[[Float]]](
                    repeating: [[Float]](repeating: [Float](repeating: 0.0, count: hiddenDim), count: timeSteps),
                    count: seqLen
                ),
                count: upperCount
            )
            self.rmsLayers = [[[Float]]](
                repeating: [[Float]](repeating: [Float](repeating: 1.0, count: timeSteps), count: seqLen),
                count: upperCount
            )
        } else {
            self.vStatesLayers = []
            self.sStatesLayers = []
            self.aStatesLayers = []
            self.denseCurLayers = []
            self.rmsLayers = []
        }

        self.spikeAvg = [[Float]](
            repeating: [Float](repeating: 0.0, count: hiddenDim),
            count: seqLen
        )
        self.logits = [[Float]](
            repeating: [Float](repeating: 0.0, count: outputDim),
            count: seqLen
        )
        self.probs = [[Float]](
            repeating: [Float](repeating: 0.0, count: outputDim),
            count: seqLen
        )
    }
}

/// スレッドセーフな独立勾配バッファ構造体 (並列学習用)
public struct NetworkGradients: @unchecked Sendable {
    public var gradWIn: [Float]
    public var gradWRec: [Float]
    public var gradWOut: [Float]        // 全スライス共有
    public var gradBOut: [Float]
    public var gradWLayers: [[Float]]
    public var gradBHLayers: [[Float]]
    public var gradGammaRMS: [[Float]]

    public init(inputDim: Int, maxHiddenDim: Int, outputDim: Int, numLayers: Int = 1) {
        self.gradWIn = [Float](repeating: 0.0, count: maxHiddenDim * inputDim)
        self.gradWRec = [Float](repeating: 0.0, count: maxHiddenDim * maxHiddenDim)
        self.gradWOut = [Float](repeating: 0.0, count: outputDim * maxHiddenDim)
        self.gradBOut = [Float](repeating: 0.0, count: outputDim)
        if 1 < numLayers {
            let upperCount = numLayers - 1
            self.gradWLayers = [[Float]](repeating: [Float](repeating: 0.0, count: maxHiddenDim * maxHiddenDim), count: upperCount)
            self.gradBHLayers = [[Float]](repeating: [Float](repeating: 0.0, count: maxHiddenDim), count: upperCount)
            self.gradGammaRMS = [[Float]](repeating: [Float](repeating: 0.0, count: maxHiddenDim), count: upperCount)
        } else {
            self.gradWLayers = []
            self.gradBHLayers = []
            self.gradGammaRMS = []
        }
    }

    public mutating func zeroGrad() {
        var i = 0
        while i < gradWIn.count { gradWIn[i] = 0.0; i += 1 }
        i = 0
        while i < gradWRec.count { gradWRec[i] = 0.0; i += 1 }
        i = 0
        while i < gradWOut.count { gradWOut[i] = 0.0; i += 1 }
        i = 0
        while i < gradBOut.count { gradBOut[i] = 0.0; i += 1 }
        var l = 0
        while l < gradWLayers.count {
            i = 0
            while i < gradWLayers[l].count { gradWLayers[l][i] = 0.0; i += 1 }
            i = 0
            while i < gradBHLayers[l].count { gradBHLayers[l][i] = 0.0; i += 1 }
            i = 0
            while i < gradGammaRMS[l].count { gradGammaRMS[l][i] = 0.0; i += 1 }
            l += 1
        }
    }

    public mutating func accumulate(from other: NetworkGradients) {
        var i = 0
        while i < gradWIn.count { gradWIn[i] += other.gradWIn[i]; i += 1 }
        i = 0
        while i < gradWRec.count { gradWRec[i] += other.gradWRec[i]; i += 1 }
        i = 0
        while i < gradWOut.count { gradWOut[i] += other.gradWOut[i]; i += 1 }
        i = 0
        while i < gradBOut.count { gradBOut[i] += other.gradBOut[i]; i += 1 }
        var l = 0
        while l < gradWLayers.count {
            if l < other.gradWLayers.count {
                i = 0
                while i < gradWLayers[l].count { gradWLayers[l][i] += other.gradWLayers[l][i]; i += 1 }
                i = 0
                while i < gradBHLayers[l].count { gradBHLayers[l][i] += other.gradBHLayers[l][i]; i += 1 }
                i = 0
                while i < gradGammaRMS[l].count { gradGammaRMS[l][i] += other.gradGammaRMS[l][i]; i += 1 }
            }
            l += 1
        }
    }
}

/// BPTT 学習トレーナー
public final class BPTTTrainer: @unchecked Sendable {
    public let network: SpikingNetwork
    public let optimizer: AdamOptimizer

    public init(network: SpikingNetwork, optimizer: AdamOptimizer) {
        self.network = network
        self.optimizer = optimizer
    }

    /// 新規勾配バッファの割り当て
    public func makeGradients() -> NetworkGradients {
        return NetworkGradients(
            inputDim: network.inputDim,
            maxHiddenDim: network.maxHiddenDim,
            outputDim: network.outputDim,
            numLayers: network.numLayers
        )
    }

    /// 単一スライスのシーケンス順伝播と損失計算
    public func forwardSequence(
        featuresSeq: [[Float]],
        targets: [Int]
    ) -> (cache: ForwardCache, loss: Float) {
        let seqLen = featuresSeq.count
        let hSize = network.maxHiddenDim
        let tSteps = network.timeSteps
        let outDim = network.outputDim

        let cache = ForwardCache(seqLen: seqLen, timeSteps: tSteps, hiddenDim: hSize, outputDim: outDim, numLayers: network.numLayers)
        if seqLen <= 0 {
            return (cache, 0.0)
        }

        let sliceBiasData = network.pBOut.data

        var vPrev0 = [Float](repeating: 0.0, count: hSize)
        var sPrev0 = [Float](repeating: 0.0, count: hSize)
        var aPrev0 = [Float](repeating: 0.0, count: hSize)

        var vPrevLayers: [[Float]] = []
        var sPrevLayers: [[Float]] = []
        var aPrevLayers: [[Float]] = []
        if 1 < network.numLayers {
            let upperCount = network.numLayers - 1
            vPrevLayers = [[Float]](repeating: [Float](repeating: 0.0, count: hSize), count: upperCount)
            sPrevLayers = [[Float]](repeating: [Float](repeating: 0.0, count: hSize), count: upperCount)
            aPrevLayers = [[Float]](repeating: [Float](repeating: 0.0, count: hSize), count: upperCount)
        }

        var totalLoss: Float = 0.0
        var numTargets: Float = 0.0

        var k = 0
        while k < seqLen {
            let feat = featuresSeq[k]
            var spikeSum = [Float](repeating: 0.0, count: hSize)

            var t = 0
            while t < tSteps {
                // 第0層 (時系列文脈・共調音を担う再帰 LIF 層)
                var vNew0 = [Float](repeating: 0.0, count: hSize)
                var sNew0 = [Float](repeating: 0.0, count: hSize)
                var aNew0 = [Float](repeating: 0.0, count: hSize)
                var totalCurrent0 = [Float](repeating: 0.0, count: hSize)

                var i = 0
                while i < hSize {
                    let wInOffset = i * network.inputDim
                    var current: Float = network.pBH.data[i]
                    var d = 0
                    while d < network.inputDim {
                        current += network.pWIn.data[wInOffset + d] * feat[d]
                        d += 1
                    }

                    let wRecOffset = i * network.maxHiddenDim
                    var j = 0
                    while j < hSize {
                        current += network.pWRec.data[wRecOffset + j] * sPrev0[j]
                        j += 1
                    }
                    totalCurrent0[i] = current

                    let vDecayed = network.lifConfig.beta * vPrev0[i] * (1.0 - sPrev0[i])
                    let vUpdated = LIFNeuronEngine.clampMembrane(vDecayed + current)
                    vNew0[i] = vUpdated

                    let aUpdated = (network.lifConfig.rho * aPrev0[i]) + (network.lifConfig.gamma * sPrev0[i])
                    aNew0[i] = aUpdated
                    let dynVTh = network.lifConfig.vTh + aUpdated

                    if dynVTh <= vUpdated {
                        sNew0[i] = 1.0
                    }
                    if vUpdated < dynVTh {
                        sNew0[i] = 0.0
                    }

                    if network.numLayers == 1 {
                        spikeSum[i] += sNew0[i]
                    }
                    cache.vStates[k][t][i] = vNew0[i]
                    cache.sStates[k][t][i] = sNew0[i]
                    cache.aStates[k][t][i] = aUpdated
                    i += 1
                }

                vPrev0 = vNew0
                sPrev0 = sNew0
                aPrev0 = aNew0

                // 上位層 (Layer 1 以降: 電流 RMSNorm & Membrane-Shortcut)
                if 1 < network.numLayers {
                    var prevCurrent = totalCurrent0
                    var l = 1
                    while l < network.numLayers {
                        let upperIdx = l - 1
                        let sPrevLayer: [Float]
                        switch l {
                        case 1:
                            sPrevLayer = sNew0
                        default:
                            sPrevLayer = cache.sStatesLayers[upperIdx - 1][k][t]
                        }

                        let bHL = network.pBHLayers[upperIdx].data
                        let wL = network.pWLayers[upperIdx].data
                        var denseCur = [Float](repeating: 0.0, count: hSize)
                        var sumSq: Float = 0.0

                        i = 0
                        while i < hSize {
                            var cVal = bHL[i]
                            let rowOffset = i * hSize
                            var j = 0
                            while j < hSize {
                                cVal += wL[rowOffset + j] * sPrevLayer[j]
                                j += 1
                            }
                            denseCur[i] = cVal
                            sumSq += cVal * cVal
                            i += 1
                        }

                        let rms = sqrt((sumSq / Float(hSize)) + 1e-5)
                        let invRms = 1.0 / rms
                        cache.rmsLayers[upperIdx][k][t] = rms
                        cache.denseCurLayers[upperIdx][k][t] = denseCur

                        let gamma = network.pGammaRMS[upperIdx].data
                        var totalCurrent_l = [Float](repeating: 0.0, count: hSize)
                        i = 0
                        while i < hSize {
                            let normCur = (denseCur[i] * invRms) * gamma[i]
                            totalCurrent_l[i] = normCur + prevCurrent[i]
                            i += 1
                        }
                        prevCurrent = totalCurrent_l

                        // 上位層 LIF 膜電位・スパイク更新
                        var vNew_l = [Float](repeating: 0.0, count: hSize)
                        var sNew_l = [Float](repeating: 0.0, count: hSize)
                        var aNew_l = [Float](repeating: 0.0, count: hSize)
                        let vPrev_l = vPrevLayers[upperIdx]
                        let sPrev_l = sPrevLayers[upperIdx]
                        let aPrev_l = aPrevLayers[upperIdx]
                        let isFinal = (l + 1) == network.numLayers

                        i = 0
                        while i < hSize {
                            let vDecayed = network.lifConfig.beta * vPrev_l[i] * (1.0 - sPrev_l[i])
                            let vUpdated = LIFNeuronEngine.clampMembrane(vDecayed + totalCurrent_l[i])
                            vNew_l[i] = vUpdated

                            let aUpdated = (network.lifConfig.rho * aPrev_l[i]) + (network.lifConfig.gamma * sPrev_l[i])
                            aNew_l[i] = aUpdated
                            let dynVTh = network.lifConfig.vTh + aUpdated

                            if dynVTh <= vUpdated {
                                sNew_l[i] = 1.0
                            }
                            if vUpdated < dynVTh {
                                sNew_l[i] = 0.0
                            }

                            if isFinal {
                                spikeSum[i] += sNew_l[i]
                            }
                            i += 1
                        }

                        cache.vStatesLayers[upperIdx][k][t] = vNew_l
                        cache.sStatesLayers[upperIdx][k][t] = sNew_l
                        cache.aStatesLayers[upperIdx][k][t] = aNew_l

                        vPrevLayers[upperIdx] = vNew_l
                        sPrevLayers[upperIdx] = sNew_l
                        aPrevLayers[upperIdx] = aNew_l
                        l += 1
                    }
                }

                t += 1
            }

            // スパイク平均
            let invT = 1.0 / Float(tSteps)
            var i = 0
            while i < hSize {
                let sAvg = spikeSum[i] * invT
                cache.spikeAvg[k][i] = sAvg
                i += 1
            }

            // 出力ロジット: WOut * sAvg + BOut
            var maxLogit: Float = -Float.greatestFiniteMagnitude
            var c = 0
            while c < outDim {
                let wOutOffset = c * network.maxHiddenDim
                var sumW: Float = 0.0
                var j = 0
                while j < hSize {
                    sumW += network.pWOut.data[wOutOffset + j] * cache.spikeAvg[k][j]
                    j += 1
                }
                let logit = sliceBiasData[c] + sumW
                cache.logits[k][c] = logit
                if maxLogit < logit {
                    maxLogit = logit
                }
                c += 1
            }

            // Softmax & Cross-Entropy 損失
            var sumExp: Float = 0.0
            c = 0
            while c < outDim {
                let expV = exp(cache.logits[k][c] - maxLogit)
                cache.probs[k][c] = expV
                sumExp += expV
                c += 1
            }
            let invSum = 1.0 / sumExp
            c = 0
            while c < outDim {
                cache.probs[k][c] *= invSum
                c += 1
            }

            if k < targets.count {
                let tgt = targets[k]
                if 0 <= tgt {
                    if tgt < outDim {
                        let weight: Float
                        if tgt == TextVocabulary.padId {
                            weight = 0.3
                        } else {
                            weight = 1.0
                        }
                        var p = cache.probs[k][tgt]
                        if p < 1e-15 {
                            p = 1e-15
                        }
                        totalLoss += -log(p) * weight
                        numTargets += weight
                    }
                }
            }

            k += 1
        }

        var avgLoss: Float = 0.0
        if 0.0 < numTargets {
            avgLoss = totalLoss / numTargets
        }
        return (cache, avgLoss)
    }

    /// 単一スライスの時間逆伝播（メインネットワークの grad 配列に直接蓄積）
    public func backwardSequence(
        featuresSeq: [[Float]],
        targets: [Int],
        cache: ForwardCache,
        lossWeight: Float = 1.0
    ) {
        var grads = makeGradients()
        backwardSequence(
            featuresSeq: featuresSeq,
            targets: targets,
            cache: cache,
            lossWeight: lossWeight,
            grads: &grads
        )
        var i = 0
        while i < network.pWIn.grad.count {
            network.pWIn.grad[i] += grads.gradWIn[i]
            i += 1
        }
        i = 0
        while i < network.pWRec.grad.count {
            network.pWRec.grad[i] += grads.gradWRec[i]
            i += 1
        }
        i = 0
        while i < network.pWOut.grad.count {
            network.pWOut.grad[i] += grads.gradWOut[i]
            i += 1
        }
        i = 0
        while i < network.pBOut.grad.count {
            network.pBOut.grad[i] += grads.gradBOut[i]
            i += 1
        }
        if 1 < network.numLayers {
            var l = 0
            while l < network.pWLayers.count {
                if l < grads.gradWLayers.count {
                    var j = 0
                    while j < network.pWLayers[l].grad.count {
                        network.pWLayers[l].grad[j] += grads.gradWLayers[l][j]
                        j += 1
                    }
                    j = 0
                    while j < network.pBHLayers[l].grad.count {
                        network.pBHLayers[l].grad[j] += grads.gradBHLayers[l][j]
                        j += 1
                    }
                    j = 0
                    while j < network.pGammaRMS[l].grad.count {
                        network.pGammaRMS[l].grad[j] += grads.gradGammaRMS[l][j]
                        j += 1
                    }
                }
                l += 1
            }
        }
    }

    /// 単一スライスの時間逆伝播（指定された勾配バッファに蓄積）
    public func backwardSequence(
        featuresSeq: [[Float]],
        targets: [Int],
        cache: ForwardCache,
        lossWeight: Float = 1.0,
        grads: inout NetworkGradients
    ) {
        let seqLen = cache.seqLen
        let hSize = network.maxHiddenDim
        let tSteps = cache.timeSteps
        let outDim = cache.outputDim
        let numLayers = network.numLayers
        let upperCount = numLayers - 1

        var numTargets: Float = 0.0
        var k = 0
        while k < seqLen {
            if k < targets.count {
                let tgt = targets[k]
                if 0 <= tgt {
                    if tgt < outDim {
                        if tgt == TextVocabulary.padId {
                            numTargets += 0.3
                        } else {
                            numTargets += 1.0
                        }
                    }
                }
            }
            k += 1
        }
        if numTargets <= 0.0 {
            return
        }

        let lossScale = (1.0 / numTargets) * lossWeight

        var dVNextStep0 = [Float](repeating: 0.0, count: hSize)
        var dSNextStep0 = [Float](repeating: 0.0, count: hSize)

        var dVNextStepLayers: [[Float]] = []
        var dSNextStepLayers: [[Float]] = []
        if 1 < numLayers {
            dVNextStepLayers = [[Float]](repeating: [Float](repeating: 0.0, count: hSize), count: upperCount)
            dSNextStepLayers = [[Float]](repeating: [Float](repeating: 0.0, count: hSize), count: upperCount)
        }

        let invT = 1.0 / Float(tSteps)

        k = seqLen - 1
        while 0 <= k {
            let feat = featuresSeq[k]
            let spikeAvg = cache.spikeAvg[k]
            let probs = cache.probs[k]

            var dLogits = [Float](repeating: 0.0, count: outDim)
            if k < targets.count {
                let tgt = targets[k]
                if 0 <= tgt {
                    if tgt < outDim {
                        let weight: Float
                        if tgt == TextVocabulary.padId {
                            weight = 0.3
                        } else {
                            weight = 1.0
                        }
                        var c = 0
                        while c < outDim {
                            var grad = probs[c]
                            if c == tgt {
                                grad -= 1.0
                            }
                            dLogits[c] = grad * weight * lossScale
                            c += 1
                        }
                    }
                }
            }

            var dSpikeAvg = [Float](repeating: 0.0, count: hSize)
            var c = 0
            while c < outDim {
                let dL = dLogits[c]
                grads.gradBOut[c] += dL
                let wOffset = c * network.maxHiddenDim
                var i = 0
                while i < hSize {
                    grads.gradWOut[wOffset + i] += dL * spikeAvg[i]
                    dSpikeAvg[i] += network.pWOut.data[wOffset + i] * dL
                    i += 1
                }
                c += 1
            }

            var dVTime0 = dVNextStep0
            var dSTime0 = dSNextStep0
            var dVTimeLayers = dVNextStepLayers
            var dSTimeLayers = dSNextStepLayers

            var t = tSteps - 1
            while 0 <= t {
                var dSTimeStep0 = dSTime0
                var dSTimeStepLayers = dSTimeLayers

                if numLayers == 1 {
                    var i = 0
                    while i < hSize {
                        dSTimeStep0[i] += dSpikeAvg[i] * invT
                        i += 1
                    }
                } else {
                    let lastUpperIdx = upperCount - 1
                    var i = 0
                    while i < hSize {
                        dSTimeStepLayers[lastUpperIdx][i] += dSpikeAvg[i] * invT
                        i += 1
                    }
                }

                var dShortcutToPrev = [Float](repeating: 0.0, count: hSize)
                if 1 < numLayers {
                    var l = numLayers - 1
                    while 1 <= l {
                        let upperIdx = l - 1
                        let vCurr_l = cache.vStatesLayers[upperIdx][k][t]
                        let aCurr_l = cache.aStatesLayers[upperIdx][k][t]
                        let denseCur = cache.denseCurLayers[upperIdx][k][t]
                        let rms = cache.rmsLayers[upperIdx][k][t]
                        let invRms = 1.0 / rms
                        let gamma = network.pGammaRMS[upperIdx].data
                        let wL = network.pWLayers[upperIdx].data

                        var dVList_l = [Float](repeating: 0.0, count: hSize)
                        var dTotalCur_l = [Float](repeating: 0.0, count: hSize)

                        var i = 0
                        while i < hSize {
                            let dS_total = dSTimeStepLayers[upperIdx][i]
                            let surrogateGrad = SurrogateGradient.derivative(
                                v: vCurr_l[i],
                                vTh: network.lifConfig.vTh + aCurr_l[i],
                                alpha: network.lifConfig.alpha
                            )
                            let dV_i = dVTimeLayers[upperIdx][i] + (dS_total * surrogateGrad)
                            dVList_l[i] = dV_i
                            dTotalCur_l[i] = dV_i + dShortcutToPrev[i]
                            i += 1
                        }

                        dShortcutToPrev = dTotalCur_l

                        var dotG: Float = 0.0
                        i = 0
                        while i < hSize {
                            grads.gradGammaRMS[upperIdx][i] += dTotalCur_l[i] * (denseCur[i] * invRms)
                            dotG += (dTotalCur_l[i] * gamma[i]) * denseCur[i]
                            i += 1
                        }

                        let S = dotG / (Float(hSize) * rms * rms)
                        var dDenseCur = [Float](repeating: 0.0, count: hSize)
                        i = 0
                        while i < hSize {
                            dDenseCur[i] = ((dTotalCur_l[i] * gamma[i]) - (denseCur[i] * S)) * invRms
                            i += 1
                        }

                        let s_prev: [Float]
                        switch l {
                        case 1:
                            s_prev = cache.sStates[k][t]
                        default:
                            s_prev = cache.sStatesLayers[upperIdx - 1][k][t]
                        }

                        i = 0
                        while i < hSize {
                            let dD = dDenseCur[i]
                            grads.gradBHLayers[upperIdx][i] += dD
                            let rowOffset = i * hSize
                            var j = 0
                            while j < hSize {
                                grads.gradWLayers[upperIdx][rowOffset + j] += dD * s_prev[j]
                                let dS_contrib = wL[rowOffset + j] * dD
                                switch l {
                                case 1:
                                    dSTimeStep0[j] += dS_contrib
                                default:
                                    dSTimeStepLayers[upperIdx - 1][j] += dS_contrib
                                }
                                j += 1
                            }
                            i += 1
                        }

                        let vPrevT_l: [Float]
                        let sPrevT_l: [Float]
                        if t == 0 {
                            if 0 < k {
                                vPrevT_l = cache.vStatesLayers[upperIdx][k - 1][tSteps - 1]
                                sPrevT_l = cache.sStatesLayers[upperIdx][k - 1][tSteps - 1]
                            } else {
                                vPrevT_l = [Float](repeating: 0.0, count: hSize)
                                sPrevT_l = [Float](repeating: 0.0, count: hSize)
                            }
                        } else {
                            vPrevT_l = cache.vStatesLayers[upperIdx][k][t - 1]
                            sPrevT_l = cache.sStatesLayers[upperIdx][k][t - 1]
                        }

                        var j = 0
                        while j < hSize {
                            let decayFactor = network.lifConfig.beta * (1.0 - sPrevT_l[j])
                            dVTimeLayers[upperIdx][j] = dVList_l[j] * decayFactor
                            dSTimeLayers[upperIdx][j] = -network.lifConfig.beta * vPrevT_l[j] * dVList_l[j]
                            j += 1
                        }

                        l -= 1
                    }
                }

                // 第0層の逆伝播 (再帰 LIF 層 + ショートカット受流し)
                let vCurr0 = cache.vStates[k][t]
                let aCurr0 = cache.aStates[k][t]
                let vPrevT0: [Float]
                let sPrevT0: [Float]
                if t == 0 {
                    if 0 < k {
                        vPrevT0 = cache.vStates[k - 1][tSteps - 1]
                        sPrevT0 = cache.sStates[k - 1][tSteps - 1]
                    } else {
                        vPrevT0 = [Float](repeating: 0.0, count: hSize)
                        sPrevT0 = [Float](repeating: 0.0, count: hSize)
                    }
                } else {
                    vPrevT0 = cache.vStates[k][t - 1]
                    sPrevT0 = cache.sStates[k][t - 1]
                }

                var dVList0 = [Float](repeating: 0.0, count: hSize)
                var i = 0
                while i < hSize {
                    let dS_total = dSTimeStep0[i]
                    let surrogateGrad = SurrogateGradient.derivative(
                        v: vCurr0[i],
                        vTh: network.lifConfig.vTh + aCurr0[i],
                        alpha: network.lifConfig.alpha
                    )
                    let dV_i = dVTime0[i] + (dS_total * surrogateGrad)
                    dVList0[i] = dV_i

                    var dInput_i = dV_i
                    if 1 < numLayers {
                        dInput_i += dShortcutToPrev[i]
                    }

                    // 入力重み勾配 WIn
                    let inOffset = i * network.inputDim
                    var d = 0
                    while d < network.inputDim {
                        grads.gradWIn[inOffset + d] += dInput_i * feat[d]
                        d += 1
                    }

                    // 再帰重み勾配 WRec
                    let recOffset = i * network.maxHiddenDim
                    var j = 0
                    while j < hSize {
                        grads.gradWRec[recOffset + j] += dInput_i * sPrevT0[j]
                        j += 1
                    }
                    i += 1
                }

                var j = 0
                while j < hSize {
                    let decayFactor = network.lifConfig.beta * (1.0 - sPrevT0[j])
                    dVTime0[j] = dVList0[j] * decayFactor

                    var dS_j = -network.lifConfig.beta * vPrevT0[j] * dVList0[j]
                    var iRec = 0
                    while iRec < hSize {
                        var dIn = dVList0[iRec]
                        if 1 < numLayers {
                            dIn += dShortcutToPrev[iRec]
                        }
                        dS_j += network.pWRec.data[iRec * network.maxHiddenDim + j] * dIn
                        iRec += 1
                    }
                    dSTime0[j] = dS_j
                    j += 1
                }

                t -= 1
            }

            dVNextStep0 = dVTime0
            dSNextStep0 = dSTime0
            dVNextStepLayers = dVTimeLayers
            dSNextStepLayers = dSTimeLayers
            k -= 1
        }
    }

    /// 1 サンプルの多重スライス勾配計算 (並列ワーカー用)
    public func computeSampleGradients(
        featuresSeq: [[Float]],
        targets: [Int],
        grads: inout NetworkGradients
    ) -> Float {
        let cache = forwardSequence(featuresSeq: featuresSeq, targets: targets)
        backwardSequence(
            featuresSeq: featuresSeq,
            targets: targets,
            cache: cache.cache,
            lossWeight: 1.0,
            grads: &grads
        )
        return cache.loss
    }

    /// 合算された勾配をメインネットワークに適用して最適化ステップを実行
    public func applyGradientsAndStep(grads: NetworkGradients, sampleCount: Int) {
        optimizer.zeroGrad()
        let scale = 1.0 / Float(max(1, sampleCount))

        var i = 0
        while i < network.pWIn.grad.count {
            network.pWIn.grad[i] = grads.gradWIn[i] * scale
            i += 1
        }
        i = 0
        while i < network.pWRec.grad.count {
            network.pWRec.grad[i] = grads.gradWRec[i] * scale
            i += 1
        }
        i = 0
        while i < network.pWOut.grad.count {
            network.pWOut.grad[i] = grads.gradWOut[i] * scale * 2.0
            i += 1
        }
        i = 0
        while i < network.pBOut.grad.count {
            // 出力層バイアスの特定文字張り付きを抑え、重み主導の分類を促進
            network.pBOut.grad[i] = grads.gradBOut[i] * scale * 0.05
            i += 1
        }
        if 1 < network.numLayers {
            var l = 0
            while l < network.pWLayers.count {
                if l < grads.gradWLayers.count {
                    var j = 0
                    while j < network.pWLayers[l].grad.count {
                        network.pWLayers[l].grad[j] = grads.gradWLayers[l][j] * scale
                        j += 1
                    }
                    j = 0
                    while j < network.pBHLayers[l].grad.count {
                        network.pBHLayers[l].grad[j] = grads.gradBHLayers[l][j] * scale
                        j += 1
                    }
                    j = 0
                    while j < network.pGammaRMS[l].grad.count {
                        network.pGammaRMS[l].grad[j] = grads.gradGammaRMS[l][j] * scale
                        j += 1
                    }
                }
                l += 1
            }
        }

        optimizer.step()
        // 重みが変わったので推論用の転置レイアウトを作り直す
        network.rebuildInferenceLayout()
    }

    /// 単一サンプルの学習ステップ (直列用)
    public func trainStep(
        featuresSeq: [[Float]],
        targets: [Int]
    ) -> Float {
        var grads = makeGradients()
        let result = computeSampleGradients(featuresSeq: featuresSeq, targets: targets, grads: &grads)
        applyGradientsAndStep(grads: grads, sampleCount: 1)
        return result
    }

    /// 単一スライスのシーケンス順伝播と対数確率系列の生成
    public func forwardSequenceLogProbs(
        featuresSeq: [[Float]]
    ) -> (cache: ForwardCache, logProbs: [[Float]]) {
        let seqLen = featuresSeq.count
        let hSize = network.maxHiddenDim
        let tSteps = network.timeSteps
        let outDim = network.outputDim

        let cache = ForwardCache(seqLen: seqLen, timeSteps: tSteps, hiddenDim: hSize, outputDim: outDim)
        var logProbs = [[Float]](repeating: [Float](repeating: 0.0, count: outDim), count: seqLen)
        if seqLen <= 0 {
            return (cache, logProbs)
        }

        let sliceBiasData = network.pBOut.data

        var vPrev = [Float](repeating: 0.0, count: hSize)
        var sPrev = [Float](repeating: 0.0, count: hSize)
        var aPrev = [Float](repeating: 0.0, count: hSize)

        var k = 0
        while k < seqLen {
            let feat = featuresSeq[k]
            var spikeSum = [Float](repeating: 0.0, count: hSize)

            var t = 0
            while t < tSteps {
                var vNew = [Float](repeating: 0.0, count: hSize)
                var sNew = [Float](repeating: 0.0, count: hSize)
                var aNew = [Float](repeating: 0.0, count: hSize)

                var i = 0
                while i < hSize {
                    let wInOffset = i * network.inputDim
                    var current: Float = network.pBH.data[i]
                    var d = 0
                    while d < network.inputDim {
                        current += network.pWIn.data[wInOffset + d] * feat[d]
                        d += 1
                    }

                    let wRecOffset = i * network.maxHiddenDim
                    var j = 0
                    while j < hSize {
                        current += network.pWRec.data[wRecOffset + j] * sPrev[j]
                        j += 1
                    }

                    let vDecayed = network.lifConfig.beta * vPrev[i] * (1.0 - sPrev[i])
                    let vUpdated = LIFNeuronEngine.clampMembrane(vDecayed + current)
                    vNew[i] = vUpdated

                    // ALIF 適応閾値の更新 (gamma = 0.0 のとき固定閾値 LIF と等価)
                    let aUpdated = (network.lifConfig.rho * aPrev[i]) + (network.lifConfig.gamma * sPrev[i])
                    aNew[i] = aUpdated
                    let dynVTh = network.lifConfig.vTh + aUpdated

                    if dynVTh <= vUpdated {
                        sNew[i] = 1.0
                    }
                    if vUpdated < dynVTh {
                        sNew[i] = 0.0
                    }

                    spikeSum[i] += sNew[i]
                    cache.vStates[k][t][i] = vNew[i]
                    cache.sStates[k][t][i] = sNew[i]
                    cache.aStates[k][t][i] = aUpdated
                    i += 1
                }

                vPrev = vNew
                sPrev = sNew
                aPrev = aNew
                t += 1
            }

            let invT = 1.0 / Float(tSteps)
            var i = 0
            while i < hSize {
                let sAvg = spikeSum[i] * invT
                cache.spikeAvg[k][i] = sAvg
                i += 1
            }

            // 出力層ロジット計算
            var maxL: Float = -.infinity
            var c = 0
            while c < outDim {
                var sum: Float = sliceBiasData[c]
                let wOutOffset = c * network.maxHiddenDim
                var h = 0
                while h < hSize {
                    sum += network.pWOut.data[wOutOffset + h] * cache.spikeAvg[k][h]
                    h += 1
                }
                cache.logits[k][c] = sum
                if maxL < sum {
                    maxL = sum
                }
                c += 1
            }

            // Log-Softmax
            var sumExp: Float = 0.0
            c = 0
            while c < outDim {
                let e = exp(cache.logits[k][c] - maxL)
                cache.probs[k][c] = e
                sumExp += e
                c += 1
            }
            let logSumExp = maxL + log(max(1e-12, sumExp))
            c = 0
            while c < outDim {
                logProbs[k][c] = cache.logits[k][c] - logSumExp
                cache.probs[k][c] /= max(1e-12, sumExp)
                c += 1
            }

            k += 1
        }

        return (cache, logProbs)
    }

    /// CTC 損失から供給された勾配 (dLogits: T x V) による時間逆伝播
    public func backwardCTCSequence(
        featuresSeq: [[Float]],
        dLogits: [[Float]],
        cache: ForwardCache,
        lossWeight: Float = 1.0,
        grads: inout NetworkGradients
    ) {
        let seqLen = cache.seqLen
        let hSize = network.maxHiddenDim
        let tSteps = cache.timeSteps
        let outDim = cache.outputDim

        if seqLen <= 0 {
            return
        }


        var dVNextStep = [Float](repeating: 0.0, count: hSize)
        var dSNextStep = [Float](repeating: 0.0, count: hSize)
        let invT = 1.0 / Float(tSteps)

        var k = seqLen - 1
        while 0 <= k {
            let feat = featuresSeq[k]
            let spikeAvg = cache.spikeAvg[k]

            // 1. 出力層の勾配計算
            var dSpikeAvg = [Float](repeating: 0.0, count: hSize)
            var c = 0
            while c < outDim {
                let dL = dLogits[k][c] * lossWeight
                grads.gradBOut[c] += dL

                let wOutOffset = c * network.maxHiddenDim
                var h = 0
                while h < hSize {
                    grads.gradWOut[wOutOffset + h] += dL * spikeAvg[h]
                    dSpikeAvg[h] += network.pWOut.data[wOutOffset + h] * dL
                    h += 1
                }
                c += 1
            }

            // 2. 隠れ層タイムステップ逆伝播
            var t = tSteps - 1
            while 0 <= t {
                let vCurr = cache.vStates[k][t]
                let aCurr = cache.aStates[k][t]
                let sPrev: [Float]

                if 0 < t {
                    sPrev = cache.sStates[k][t - 1]
                } else {
                    if 0 < k {
                        sPrev = cache.sStates[k - 1][tSteps - 1]
                    } else {
                        sPrev = [Float](repeating: 0.0, count: hSize)
                    }
                }

                var dVThisStep = [Float](repeating: 0.0, count: hSize)
                var dSThisStep = [Float](repeating: 0.0, count: hSize)

                var i = 0
                while i < hSize {
                    let dS = (dSpikeAvg[i] * invT) + dSNextStep[i]
                    // 代理勾配は順伝播で実際に用いた動的閾値で評価する
                    // (適応状態 a は定数扱い = detached adaptation 近似)
                    let surrogate = SurrogateGradient.derivative(
                        v: vCurr[i],
                        vTh: network.lifConfig.vTh + aCurr[i],
                        alpha: network.lifConfig.alpha
                    )
                    let dV = dVNextStep[i] + (dS * surrogate)

                    // 入力重み勾配
                    let wInOffset = i * network.inputDim
                    var d = 0
                    while d < network.inputDim {
                        grads.gradWIn[wInOffset + d] += dV * feat[d]
                        d += 1
                    }

                    // 再帰重み勾配
                    let wRecOffset = i * network.maxHiddenDim
                    var j = 0
                    while j < hSize {
                        grads.gradWRec[wRecOffset + j] += dV * sPrev[j]
                        dSThisStep[j] += network.pWRec.data[wRecOffset + j] * dV
                        j += 1
                    }

                    dVThisStep[i] = dV * network.lifConfig.beta * (1.0 - sPrev[i])
                    i += 1
                }

                dVNextStep = dVThisStep
                dSNextStep = dSThisStep
                t -= 1
            }

            k -= 1
        }
    }

    /// CTC 損失による 多重スライス勾配計算
    public func computeSampleCTCGradients(
        featuresSeq: [[Float]],
        targets: [Int],
        grads: inout NetworkGradients,
        ctcLossCalc: CTCLossCalculator = CTCLossCalculator(blankId: 0)
    ) -> Float {
        let fwd = forwardSequenceLogProbs(featuresSeq: featuresSeq)
        let ctcResult = ctcLossCalc.computeLossAndGradients(logProbs: fwd.logProbs, targets: targets)

        backwardCTCSequence(
            featuresSeq: featuresSeq,
            dLogits: ctcResult.gradients,
            cache: fwd.cache,
            lossWeight: 1.0,
            grads: &grads
        )

        return ctcResult.loss
    }
}
