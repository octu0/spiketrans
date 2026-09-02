import Foundation

/// BPTT 計算用の中間キャッシュ
public final class ForwardCache: @unchecked Sendable {
    public let seqLen: Int
    public let timeSteps: Int
    public let hiddenDim: Int
    public let outputDim: Int

    public var vStates: [[[Float]]]  // [seqLen][timeSteps][hiddenDim]
    public var sStates: [[[Float]]]  // [seqLen][timeSteps][hiddenDim]
    public var aStates: [[[Float]]]  // [seqLen][timeSteps][hiddenDim] (ALIF 適応閾値状態)
    public var spikeAvg: [[Float]]   // [seqLen][hiddenDim]
    public var logits: [[Float]]     // [seqLen][outputDim]
    public var probs: [[Float]]      // [seqLen][outputDim]

    public init(seqLen: Int, timeSteps: Int, hiddenDim: Int, outputDim: Int) {
        self.seqLen = seqLen
        self.timeSteps = timeSteps
        self.hiddenDim = hiddenDim
        self.outputDim = outputDim

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
    public var gradWOut: [Float]
    public var gradBOut: [Float]        // High スライス用
    public var gradBOutBase: [Float]
    public var gradBOutMiddle: [Float]

    public init(inputDim: Int, maxHiddenDim: Int, outputDim: Int) {
        self.gradWIn = [Float](repeating: 0.0, count: maxHiddenDim * inputDim)
        self.gradWRec = [Float](repeating: 0.0, count: maxHiddenDim * maxHiddenDim)
        self.gradWOut = [Float](repeating: 0.0, count: outputDim * maxHiddenDim)
        self.gradBOut = [Float](repeating: 0.0, count: outputDim)
        self.gradBOutBase = [Float](repeating: 0.0, count: outputDim)
        self.gradBOutMiddle = [Float](repeating: 0.0, count: outputDim)
    }

    /// スライスに対応するバイアス勾配へのアクセス
    public mutating func addBiasGradient(slice: MatryoshkaSlice, index: Int, value: Float) {
        switch slice {
        case .base:
            gradBOutBase[index] += value
        case .middle:
            gradBOutMiddle[index] += value
        case .high:
            gradBOut[index] += value
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
        i = 0
        while i < gradBOutBase.count { gradBOutBase[i] = 0.0; i += 1 }
        i = 0
        while i < gradBOutMiddle.count { gradBOutMiddle[i] = 0.0; i += 1 }
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
        i = 0
        while i < gradBOutBase.count { gradBOutBase[i] += other.gradBOutBase[i]; i += 1 }
        i = 0
        while i < gradBOutMiddle.count { gradBOutMiddle[i] += other.gradBOutMiddle[i]; i += 1 }
    }
}

/// BPTT 学習トレーナー
public final class BPTTTrainer: @unchecked Sendable {
    public let network: MatryoshkaNetwork
    public let optimizer: AdamOptimizer

    public init(network: MatryoshkaNetwork, optimizer: AdamOptimizer) {
        self.network = network
        self.optimizer = optimizer
    }

    /// 新規勾配バッファの割り当て
    public func makeGradients() -> NetworkGradients {
        return NetworkGradients(
            inputDim: network.inputDim,
            maxHiddenDim: network.maxHiddenDim,
            outputDim: network.outputDim
        )
    }

    /// 単一スライスのシーケンス順伝播と損失計算
    public func forwardSequence(
        featuresSeq: [[Float]],
        targets: [Int],
        slice: MatryoshkaSlice
    ) -> (cache: ForwardCache, loss: Float) {
        let seqLen = featuresSeq.count
        let hSize = min(slice.rawValue, network.maxHiddenDim)
        let tSteps = network.timeSteps
        let outDim = network.outputDim

        let cache = ForwardCache(seqLen: seqLen, timeSteps: tSteps, hiddenDim: hSize, outputDim: outDim)
        if seqLen <= 0 {
            return (cache, 0.0)
        }

        let sliceBiasData = network.outputBias(for: slice).data

        var vPrev = [Float](repeating: 0.0, count: hSize)
        var sPrev = [Float](repeating: 0.0, count: hSize)
        var aPrev = [Float](repeating: 0.0, count: hSize)

        var totalLoss: Float = 0.0
        var numTargets: Float = 0.0

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
                    // 入力電流: bH + WIn * feat
                    let wInOffset = i * network.inputDim
                    var current: Float = network.pBH.data[i]
                    var d = 0
                    while d < network.inputDim {
                        current += network.pWIn.data[wInOffset + d] * feat[d]
                        d += 1
                    }

                    // 再帰電流: WRec * sPrev
                    let wRecOffset = i * network.maxHiddenDim
                    var j = 0
                    while j < hSize {
                        current += network.pWRec.data[wRecOffset + j] * sPrev[j]
                        j += 1
                    }

                    // LIF 膜電位更新
                    let vDecayed = network.lifConfig.beta * vPrev[i] * (1.0 - sPrev[i])
                    let vUpdated = vDecayed + current
                    vNew[i] = vUpdated

                    // ALIF 適応閾値の更新 (gamma = 0.0 のとき固定閾値 LIF と等価)
                    let aUpdated = (network.lifConfig.rho * aPrev[i]) + (network.lifConfig.gamma * sPrev[i])
                    aNew[i] = aUpdated
                    let dynVTh = network.lifConfig.vTh + aUpdated

                    // スパイク発火
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

            // スパイク平均
            let invT = 1.0 / Float(tSteps)
            var i = 0
            while i < hSize {
                let sAvg = spikeSum[i] * invT
                cache.spikeAvg[k][i] = sAvg
                i += 1
            }

            // 出力ロジット: sliceNorm * WOut * sAvg + BOut
            let sliceNorm = sqrt(Float(network.maxHiddenDim) / Float(hSize))
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
                let logit = sliceBiasData[c] + (sumW * sliceNorm)
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
        slice: MatryoshkaSlice,
        lossWeight: Float = 1.0
    ) {
        var grads = makeGradients()
        backwardSequence(
            featuresSeq: featuresSeq,
            targets: targets,
            cache: cache,
            slice: slice,
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
            network.pBOutBase.grad[i] += grads.gradBOutBase[i]
            network.pBOutMiddle.grad[i] += grads.gradBOutMiddle[i]
            i += 1
        }
    }

    /// 単一スライスの時間逆伝播（指定された勾配バッファに蓄積）
    public func backwardSequence(
        featuresSeq: [[Float]],
        targets: [Int],
        cache: ForwardCache,
        slice: MatryoshkaSlice,
        lossWeight: Float = 1.0,
        grads: inout NetworkGradients
    ) {
        let seqLen = cache.seqLen
        let hSize = min(slice.rawValue, network.maxHiddenDim)
        let tSteps = cache.timeSteps
        let outDim = cache.outputDim

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
        var dVNextStep = [Float](repeating: 0.0, count: hSize)
        var dSNextStep = [Float](repeating: 0.0, count: hSize)
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

            let sliceNorm = sqrt(Float(network.maxHiddenDim) / Float(hSize))
            var dSpikeAvg = [Float](repeating: 0.0, count: hSize)
            var c = 0
            while c < outDim {
                let dL = dLogits[c]
                grads.addBiasGradient(slice: slice, index: c, value: dL)
                let wOffset = c * network.maxHiddenDim
                var i = 0
                while i < hSize {
                    grads.gradWOut[wOffset + i] += dL * spikeAvg[i] * sliceNorm
                    dSpikeAvg[i] += network.pWOut.data[wOffset + i] * dL * sliceNorm
                    i += 1
                }
                c += 1
            }

            var dVTime = dVNextStep
            var dSTime = dSNextStep

            var t = tSteps - 1
            while 0 <= t {
                let vCurr = cache.vStates[k][t]
                let vPrevT: [Float]
                let sPrevT: [Float]

                if t == 0 {
                    if 0 < k {
                        vPrevT = cache.vStates[k - 1][tSteps - 1]
                        sPrevT = cache.sStates[k - 1][tSteps - 1]
                    } else {
                        vPrevT = [Float](repeating: 0.0, count: hSize)
                        sPrevT = [Float](repeating: 0.0, count: hSize)
                    }
                } else {
                    vPrevT = cache.vStates[k][t - 1]
                    sPrevT = cache.sStates[k][t - 1]
                }

                let aCurr = cache.aStates[k][t]

                var dVList = [Float](repeating: 0.0, count: hSize)
                var i = 0
                while i < hSize {
                    let dS_total = (dSpikeAvg[i] * invT) + dSTime[i]
                    // 代理勾配は順伝播で実際に用いた動的閾値で評価する
                    // (適応状態 a は定数扱い = detached adaptation 近似)
                    let surrogateGrad = SurrogateGradient.derivative(
                        v: vCurr[i],
                        vTh: network.lifConfig.vTh + aCurr[i],
                        alpha: network.lifConfig.alpha
                    )
                    let dV_i = dVTime[i] + (dS_total * surrogateGrad)
                    dVList[i] = dV_i

                    let dInput_i = dV_i

                    // 入力重み勾配
                    let inOffset = i * network.inputDim
                    var d = 0
                    while d < network.inputDim {
                        grads.gradWIn[inOffset + d] += dInput_i * feat[d]
                        d += 1
                    }

                    // 再帰重み勾配
                    let recOffset = i * network.maxHiddenDim
                    var j = 0
                    while j < hSize {
                        grads.gradWRec[recOffset + j] += dInput_i * sPrevT[j]
                        j += 1
                    }
                    i += 1
                }

                var newDVTime = [Float](repeating: 0.0, count: hSize)
                var newDSTime = [Float](repeating: 0.0, count: hSize)

                var j = 0
                while j < hSize {
                    let decayFactor = network.lifConfig.beta * (1.0 - sPrevT[j])
                    newDVTime[j] = dVList[j] * decayFactor

                    var dS_j = -network.lifConfig.beta * vPrevT[j] * dVList[j]
                    var iRec = 0
                    while iRec < hSize {
                        dS_j += network.pWRec.data[iRec * network.maxHiddenDim + j] * dVList[iRec]
                        iRec += 1
                    }
                    newDSTime[j] = dS_j
                    j += 1
                }

                dVTime = newDVTime
                dSTime = newDSTime
                t -= 1
            }

            dVNextStep = dVTime
            dSNextStep = dSTime
            k -= 1
        }
    }

    /// 1 サンプルの多重スライス勾配計算 (並列ワーカー用)
    public func computeSampleGradients(
        featuresSeq: [[Float]],
        targets: [Int],
        grads: inout NetworkGradients
    ) -> (totalLoss: Float, lossBase: Float, lossMiddle: Float, lossHigh: Float) {
        var lossBase: Float = 0.0
        var lossMiddle: Float = 0.0
        var lossHigh: Float = 0.0
        var totalLoss: Float = 0.0

        var activeSlices: [MatryoshkaSlice] = []
        for slice in MatryoshkaSlice.allCases {
            if slice == .base || slice.rawValue <= network.maxHiddenDim {
                activeSlices.append(slice)
            }
        }

        for slice in activeSlices {
            let cache = forwardSequence(featuresSeq: featuresSeq, targets: targets, slice: slice)
            switch slice {
            case .base:
                lossBase = cache.loss
            case .middle:
                lossMiddle = cache.loss
            case .high:
                lossHigh = cache.loss
            }
            totalLoss += cache.loss
            let lossWeight: Float = 1.0
            backwardSequence(
                featuresSeq: featuresSeq,
                targets: targets,
                cache: cache.cache,
                slice: slice,
                lossWeight: lossWeight,
                grads: &grads
            )
        }

        return (totalLoss, lossBase, lossMiddle, lossHigh)
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
            network.pBOutBase.grad[i] = grads.gradBOutBase[i] * scale * 0.05
            network.pBOutMiddle.grad[i] = grads.gradBOutMiddle[i] * scale * 0.05
            i += 1
        }

        optimizer.step()
    }

    /// Matryoshka 多重スライス同時学習ステップ (単一サンプル直列用)
    public func trainStep(
        featuresSeq: [[Float]],
        targets: [Int]
    ) -> (totalLoss: Float, lossBase: Float, lossMiddle: Float, lossHigh: Float) {
        var grads = makeGradients()
        let result = computeSampleGradients(featuresSeq: featuresSeq, targets: targets, grads: &grads)
        applyGradientsAndStep(grads: grads, sampleCount: 1)
        return result
    }

    /// 単一スライスのシーケンス順伝播と対数確率系列の生成
    public func forwardSequenceLogProbs(
        featuresSeq: [[Float]],
        slice: MatryoshkaSlice
    ) -> (cache: ForwardCache, logProbs: [[Float]]) {
        let seqLen = featuresSeq.count
        let hSize = min(slice.rawValue, network.maxHiddenDim)
        let tSteps = network.timeSteps
        let outDim = network.outputDim

        let cache = ForwardCache(seqLen: seqLen, timeSteps: tSteps, hiddenDim: hSize, outputDim: outDim)
        var logProbs = [[Float]](repeating: [Float](repeating: 0.0, count: outDim), count: seqLen)
        if seqLen <= 0 {
            return (cache, logProbs)
        }

        let sliceBiasData = network.outputBias(for: slice).data

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
                    let vUpdated = vDecayed + current
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
        slice: MatryoshkaSlice,
        lossWeight: Float = 1.0,
        grads: inout NetworkGradients
    ) {
        let seqLen = cache.seqLen
        let hSize = min(slice.rawValue, network.maxHiddenDim)
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
                grads.addBiasGradient(slice: slice, index: c, value: dL)

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

    /// CTC 損失による Matryoshka 多重スライス勾配計算
    public func computeSampleCTCGradients(
        featuresSeq: [[Float]],
        targets: [Int],
        grads: inout NetworkGradients,
        ctcLossCalc: CTCLossCalculator = CTCLossCalculator(blankId: 0)
    ) -> (totalLoss: Float, lossBase: Float, lossMiddle: Float, lossHigh: Float) {
        var totalLoss: Float = 0.0
        var lossBase: Float = 0.0
        var lossMiddle: Float = 0.0
        var lossHigh: Float = 0.0

        for slice in MatryoshkaSlice.allCases {
            let lossWeight: Float
            switch slice {
            case .base:
                lossWeight = 0.2
            case .middle:
                lossWeight = 0.3
            case .high:
                lossWeight = 0.5
            }

            let fwd = forwardSequenceLogProbs(featuresSeq: featuresSeq, slice: slice)
            let ctcResult = ctcLossCalc.computeLossAndGradients(logProbs: fwd.logProbs, targets: targets)

            switch slice {
            case .base:
                lossBase = ctcResult.loss
            case .middle:
                lossMiddle = ctcResult.loss
            case .high:
                lossHigh = ctcResult.loss
            }
            totalLoss += ctcResult.loss * lossWeight

            backwardCTCSequence(
                featuresSeq: featuresSeq,
                dLogits: ctcResult.gradients,
                cache: fwd.cache,
                slice: slice,
                lossWeight: lossWeight,
                grads: &grads
            )
        }

        return (totalLoss, lossBase, lossMiddle, lossHigh)
    }
}
