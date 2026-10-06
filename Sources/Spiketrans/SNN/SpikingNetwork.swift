import Foundation

/// Float32 の SNN。層 0 は再帰 LIF、層 1 以降は FF LIF (`upperRecurrent` なら層 1 以降も再帰)。
/// 上位層は結合電流を RMSNorm してから前層電流を足す。
public final class SpikingNetwork: @unchecked Sendable {
    public let numLayers: Int
    public let inputDim: Int
    public let maxHiddenDim: Int
    public let outputDim: Int
    public let timeSteps: Int
    /// LIF / ALIF 設定 (重みインポート時に保存済みの値へ追従するため var)
    public private(set) var lifConfig: LIFConfig

    /// 推論用の転置コピー。wRecT[j * maxHiddenDim + n] = wRec[n][j] で、
    /// 発火ニューロン j の流出重みが連続に並ぶため Event-driven 加算を SIMD 化できる
    public private(set) var wRecT: [Float] = []
    /// 層 1 以降の結合重みの転置コピー (wRecT と同じ並び)
    public private(set) var wLayersT: [[Float]] = []
    /// 層 1 以降の再帰結合の転置コピー (wRecT と同じ並び)。再帰なしの構成では空
    public private(set) var wRecLayersT: [[Float]] = []
    /// 推論用の転置コピー。wOutT[k * outputDim + c] = wOut[c][k]
    public private(set) var wOutT: [Float] = []

    // 層 0 (再帰 LIF 層)
    public let pWIn: Parameter
    public let pWRec: Parameter
    public let pBH: Parameter

    // 層 1 以降 (前層スパイクを受けるフィードフォワード LIF 層)
    public let pWLayers: [Parameter]
    public let pBHLayers: [Parameter]
    public let pGammaRMS: [Parameter]
    /// 層 1 以降の再帰結合 (同じ層の直前サブステップのスパイクから)。再帰なしの構成では空
    public let pWRecLayers: [Parameter]
    /// ニューロンごとの膜電位減衰率 (層 0 から numLayers 本)。共通 beta の構成では空
    public let pBetaLayers: [Parameter]
    /// 各層の LIF に入る電流全体を RMSNorm するときのゲイン (層 0 から numLayers 本)。正規化しない構成では空
    public let pInputNormGains: [Parameter]
    /// ゲート付き記憶の係数 (`SpikingNetworkWeights.gateLayers`)。記憶の状態は aPrev に持つ (適応閾値とは排他)
    public private(set) var gateLayers: [[Float]]? {
        didSet {
            refreshGateCache()
        }
    }
    /// 層ごとの log sigmoid(Λ) (推論のたびに計算しない)
    private var gateLogDecay: [[Float]] = []

    // リードアウト
    public let pWOut: Parameter        // [outputDim, maxHiddenDim]
    public let pBOut: Parameter

    /// 音響本線の `inputDim` は `StreamingFeatureFrontEnd.acousticInputDim()` (既定 512)。
    public init(
        numLayers: Int = 1,
        inputDim: Int = 64,
        maxHiddenDim: Int = 4096,
        outputDim: Int = 64,
        timeSteps: Int = 4,
        lifConfig: LIFConfig = LIFConfig(),
        upperRecurrent: Bool = false,
        learnedBeta: Bool = false,
        inputNorm: Bool = false,
        inputNormGainInit: Float = 1.0,
        gatedMemory: Bool = false
    ) {
        self.numLayers = max(1, numLayers)
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

        // 上位層パラメータの初期化
        var wLayersList: [Parameter] = []
        var bhLayersList: [Parameter] = []
        var gammaRMSList: [Parameter] = []
        var wRecLayersList: [Parameter] = []
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
            if upperRecurrent {
                var initWRecLayer = [Float](repeating: 0.0, count: maxHiddenDim * maxHiddenDim)
                idx = 0
                while idx < maxHiddenDim * maxHiddenDim {
                    initWRecLayer[idx] = Float.random(in: -scaleRec...scaleRec)
                    idx += 1
                }
                wRecLayersList.append(Parameter(count: maxHiddenDim * maxHiddenDim, initialData: initWRecLayer))
            }
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
        self.pWRecLayers = wRecLayersList
        var betaList: [Parameter] = []
        if learnedBeta {
            let lo = SpikingNetworkWeights.learnedBetaRange.lowerBound
            let hi = SpikingNetworkWeights.learnedBetaRange.upperBound
            var values = [Float](repeating: 0.0, count: maxHiddenDim)
            var n = 0
            while n < maxHiddenDim {
                var frac: Float = 0.0
                if 1 < maxHiddenDim {
                    frac = Float(n) / Float(maxHiddenDim - 1)
                }
                values[n] = lo + (hi - lo) * frac
                n += 1
            }
            var li = 0
            while li < self.numLayers {
                betaList.append(Parameter(count: maxHiddenDim, initialData: values))
                li += 1
            }
        }
        self.pBetaLayers = betaList
        var normList: [Parameter] = []
        if inputNorm {
            var li = 0
            while li < self.numLayers {
                normList.append(Parameter(count: maxHiddenDim, initialData: [Float](repeating: inputNormGainInit, count: maxHiddenDim)))
                li += 1
            }
        }
        self.pInputNormGains = normList
        self.pWOut = Parameter(count: outputDim * maxHiddenDim, initialData: initWOut)
        self.pBOut = Parameter(count: outputDim, initialData: initBOut)

        rebuildInferenceLayout()
        if gatedMemory {
            precondition(lifConfig.gamma == 0.0, "ゲート付き記憶は適応閾値の状態を記憶に使うので、適応閾値 (gamma > 0) とは併用できない")
            self.gateLayers = SpikingNetworkWeights.initialGateLayers(numLayers: self.numLayers, hidden: maxHiddenDim)
            refreshGateCache()
        }
    }

    /// 保存済み重みから構成ごと復元する
    public convenience init(weights: SpikingNetworkWeights) {
        self.init(
            numLayers: weights.numLayers,
            inputDim: weights.inputDim,
            maxHiddenDim: weights.maxHiddenDim,
            outputDim: weights.outputDim,
            timeSteps: weights.timeSteps,
            lifConfig: weights.lifConfig,
            upperRecurrent: weights.hasUpperRecurrence,
            learnedBeta: weights.hasLearnedBeta,
            inputNorm: weights.hasInputNorm,
            gatedMemory: weights.hasGatedMemory
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
        params.append(contentsOf: pWRecLayers)
        params.append(contentsOf: pBetaLayers)
        params.append(contentsOf: pInputNormGains)
        params.append(pWOut)
        params.append(pBOut)
        return params
    }

    /// 全体重みのエクスポート
    public func exportWeights(vocabulary: TextVocabulary? = nil) -> SpikingNetworkWeights {
        return SpikingNetworkWeights(
            inputDim: inputDim,
            maxHiddenDim: maxHiddenDim,
            outputDim: outputDim,
            timeSteps: timeSteps,
            lifConfig: lifConfig,
            wIn: pWIn.data,
            wRec: pWRec.data,
            bH: pBH.data,
            wLayers: pWLayers.map { $0.data },
            bHLayers: pBHLayers.map { $0.data },
            gammaRMS: pGammaRMS.map { $0.data },
            wRecLayers: exportedRecLayers(),
            betaLayers: exportedBetaLayers(),
            inputNormGains: exportedNormGains(),
            gateLayers: gateLayers,
            wOut: pWOut.data,
            bOut: pBOut.data,
            vocabularyCharacters: vocabulary?.serializedCharacters
        )
    }

    private func exportedNormGains() -> [[Float]]? {
        if pInputNormGains.isEmpty {
            return nil
        }
        return pInputNormGains.map { $0.data }
    }

    private func exportedBetaLayers() -> [[Float]]? {
        if pBetaLayers.isEmpty {
            return nil
        }
        return pBetaLayers.map { $0.data }
    }

    private func exportedRecLayers() -> [[Float]]? {
        if pWRecLayers.isEmpty {
            return nil
        }
        return pWRecLayers.map { $0.data }
    }

    /// 全体重みのインポート (学習時の LIF / ALIF パラメータも同時に復元)
    public func importWeights(from weightsData: SpikingNetworkWeights) {
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
        var l = 0
        while l < min(pWLayers.count, weightsData.wLayers.count) {
            if weightsData.wLayers[l].count == pWLayers[l].data.count {
                pWLayers[l].data = weightsData.wLayers[l]
            }
            if weightsData.bHLayers[l].count == pBHLayers[l].data.count {
                pBHLayers[l].data = weightsData.bHLayers[l]
            }
            if weightsData.gammaRMS[l].count == pGammaRMS[l].data.count {
                pGammaRMS[l].data = weightsData.gammaRMS[l]
            }
            l += 1
        }
        if let rec = weightsData.wRecLayers {
            var r = 0
            while r < min(pWRecLayers.count, rec.count) {
                if rec[r].count == pWRecLayers[r].data.count {
                    pWRecLayers[r].data = rec[r]
                }
                r += 1
            }
        }
        if let betas = weightsData.betaLayers {
            var bi = 0
            while bi < min(pBetaLayers.count, betas.count) {
                if betas[bi].count == pBetaLayers[bi].data.count {
                    pBetaLayers[bi].data = betas[bi]
                }
                bi += 1
            }
        }
        if let gains = weightsData.inputNormGains {
            var gi = 0
            while gi < min(pInputNormGains.count, gains.count) {
                if gains[gi].count == pInputNormGains[gi].data.count {
                    pInputNormGains[gi].data = gains[gi]
                }
                gi += 1
            }
        }
        if let gates = weightsData.gateLayers, let current = gateLayers, gates.count == current.count {
            gateLayers = gates
        }
        if weightsData.wOut.count == pWOut.data.count {
            pWOut.data = weightsData.wOut
        }
        if weightsData.bOut.count == pBOut.data.count {
            pBOut.data = weightsData.bOut
        }
        rebuildInferenceLayout()
    }

    private func refreshGateCache() {
        guard let gates = gateLayers else {
            gateLogDecay = []
            return
        }
        let hidden = maxHiddenDim
        gateLogDecay = gates.map { g in
            var out = [Float](repeating: 0.0, count: hidden)
            var n = 0
            while n < hidden {
                // log sigmoid(x) = -log(1 + exp(-x))
                out[n] = -log1pf(expf(-g[n]))
                n += 1
            }
            return out
        }
    }

    /// ゲート付き記憶を 1 層ぶん更新し (1 フレームに 1 回)、記憶の出力 m = w_o ⊙ h を out に書く。
    /// own はその層が残差に足す電流 (学習側 `MLXBPTTTrainer.substep` の own)
    @inline(__always)
    private func updateGatedMemory(
        layer: Int,
        own: UnsafePointer<Float>,
        h: UnsafeMutablePointer<Float>,
        out: UnsafeMutablePointer<Float>,
        count: Int
    ) {
        guard let gates = gateLayers, layer < gates.count else {
            return
        }
        let hidden = maxHiddenDim
        let c = SpikingNetworkWeights.gateDecayExponent
        gates[layer].withUnsafeBufferPointer { gBuf in
            gateLogDecay[layer].withUnsafeBufferPointer { lBuf in
                let g = gBuf.baseAddress!
                let logDecay = lBuf.baseAddress!
                let limit = count - (count % 8)
                let cVec = SIMD8<Float>(repeating: c)
                let one = SIMD8<Float>(repeating: 1.0)
                let eps = SIMD8<Float>(repeating: 1e-6)
                let zero = SIMD8<Float>(repeating: 0.0)
                var n = 0
                while n < limit {
                    let u = UnsafeRawPointer(own.advanced(by: n)).loadUnaligned(as: SIMD8<Float>.self)
                    let wr = UnsafeRawPointer(g.advanced(by: hidden + n)).loadUnaligned(as: SIMD8<Float>.self)
                    let br = UnsafeRawPointer(g.advanced(by: 2 * hidden + n)).loadUnaligned(as: SIMD8<Float>.self)
                    let wi = UnsafeRawPointer(g.advanced(by: 3 * hidden + n)).loadUnaligned(as: SIMD8<Float>.self)
                    let bi = UnsafeRawPointer(g.advanced(by: 4 * hidden + n)).loadUnaligned(as: SIMD8<Float>.self)
                    let wo = UnsafeRawPointer(g.advanced(by: 5 * hidden + n)).loadUnaligned(as: SIMD8<Float>.self)
                    let ld = UnsafeRawPointer(logDecay.advanced(by: n)).loadUnaligned(as: SIMD8<Float>.self)
                    let hPrev = UnsafeRawPointer(h.advanced(by: n)).loadUnaligned(as: SIMD8<Float>.self)
                    let keep = SIMDMath.sigmoid(wr * u + br)
                    let admit = SIMDMath.sigmoid(wi * u + bi)
                    let decay = SIMDMath.exp((cVec * keep) * ld)
                    var gain = one - decay * decay + eps
                    gain = gain.replacing(with: zero, where: gain .< zero)
                    let hNext = decay * hPrev + gain.squareRoot() * (admit * u)
                    UnsafeMutableRawPointer(h.advanced(by: n)).storeBytes(of: hNext, as: SIMD8<Float>.self)
                    UnsafeMutableRawPointer(out.advanced(by: n)).storeBytes(of: wo * hNext, as: SIMD8<Float>.self)
                    n += 8
                }
                while n < count {
                    let u = own[n]
                    let keep = 1.0 / (1.0 + expf(-(g[hidden + n] * u + g[2 * hidden + n])))
                    let admit = 1.0 / (1.0 + expf(-(g[3 * hidden + n] * u + g[4 * hidden + n])))
                    let decay = expf((c * keep) * logDecay[n])
                    h[n] = decay * h[n] + sqrtf(max(0.0, 1.0 - decay * decay + 1e-6)) * (admit * u)
                    out[n] = g[5 * hidden + n] * h[n]
                    n += 1
                }
            }
        }
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
        var r = 0
        while r < pWRecLayers.count {
            if wRecLayersT.count <= r {
                wRecLayersT.append([Float](repeating: 0.0, count: hSize * hSize))
            }
            if wRecLayersT[r].count != hSize * hSize {
                wRecLayersT[r] = [Float](repeating: 0.0, count: hSize * hSize)
            }
            let srcRec = pWRecLayers[r].data
            srcRec.withUnsafeBufferPointer { src in
                wRecLayersT[r].withUnsafeMutableBufferPointer { dst in
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
            r += 1
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
        readoutSum: inout [Float],
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
        if gateLayers != nil && scratch.memoryOut.count < totalStateSize {
            scratch.memoryOut = [Float](repeating: 0.0, count: totalStateSize)
        }

        // 1. 読み出し用の膜電位積算をリセット
        var i = 0
        while i < hSize {
            readoutSum[i] = 0.0
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
            // 3.1 層 0: 直前ステップで発火したニューロンの流出重みを足す (再帰結合)。
            //     転置レイアウト wRecT により発火 j の行が連続 → SIMD 加算
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

            let gated = gateLayers != nil
            if gated {
                // 記憶の出力は 1 フレームに 1 回 (最初のサブステップで) 更新し、毎サブステップ電流に足す
                scratch.stepCurrents.withUnsafeMutableBufferPointer { stepBuf in
                    scratch.memoryOut.withUnsafeMutableBufferPointer { memBuf in
                        if t == 0 {
                            aPrev.withUnsafeMutableBufferPointer { aBuf in
                                updateGatedMemory(
                                    layer: 0, own: UnsafePointer(stepBuf.baseAddress!), h: aBuf.baseAddress!,
                                    out: memBuf.baseAddress!, count: hSize
                                )
                            }
                        }
                        Self.addRow(UnsafePointer(memBuf.baseAddress!), to: stepBuf.baseAddress!, count: hSize)
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
                            readoutSum.withUnsafeMutableBufferPointer { sumBuf in
                                stepLayer(
                                    isLast: numLayers == 1,
                                    layer: 0,
                                    vPtr: vBuf.baseAddress!,
                                    sPtr: sBuf.baseAddress!,
                                    aPtr: adaptPointer(aBuf.baseAddress!, gated: gated, scratch: scratch),
                                    curPtr: curBuf.baseAddress!,
                                    readoutSumPtr: sumBuf.baseAddress!,
                                    count: hSize,
                                    scratch: scratch
                                )
                            }
                        }
                    }
                }
            }

            // 3.2 層 1 以降: 前層スパイクの結合電流を RMSNorm し、前層の入力電流を加算
            //     (学習側 MLXBPTTTrainer.logitsBatch と同じ式)
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

                let gammaData = pGammaRMS[upperIdx].data
                scratch.stepCurrents.withUnsafeMutableBufferPointer { stepBuf in
                    scratch.stepCurrentsPrev.withUnsafeMutableBufferPointer { prevBuf in
                        scratch.ownCurrents.withUnsafeMutableBufferPointer { ownBuf in
                        gammaData.withUnsafeBufferPointer { gammaBuf in
                            let step = stepBuf.baseAddress!
                            let prev = prevBuf.baseAddress!
                            let own = ownBuf.baseAddress!
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
                            let rms = sqrt(meanSq + rmsNormEpsilon)
                            let invRms = 1.0 / rms
                            let invRmsVec = SIMD8<Float>(repeating: invRms)

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
                                UnsafeMutableRawPointer(own.advanced(by: n)).storeBytes(of: norm, as: SIMD8<Float>.self)
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
                                own[n] = norm
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
                }

                // 再帰構成: 同じ層の直前サブステップのスパイクによる再帰電流を足す。
                // 次の上位層へ渡す残差 (stepCurrentsPrev) にも同じ電流を含める (学習側と同じ式)
                if upperIdx < wRecLayersT.count {
                    var activeRecCount = 0
                    var rj = 0
                    while rj < hSize {
                        if sPrev[thisLayerOffset + rj] != 0.0 {
                            scratch.activeLayerSpikes[activeRecCount] = rj
                            activeRecCount += 1
                        }
                        rj += 1
                    }
                    let hasNext = (layerIdx + 1) < numLayers
                    wRecLayersT[upperIdx].withUnsafeBufferPointer { wBuf in
                        scratch.stepCurrents.withUnsafeMutableBufferPointer { stepBuf in
                            scratch.stepCurrentsPrev.withUnsafeMutableBufferPointer { prevBuf in
                                let wT = wBuf.baseAddress!
                                var a = 0
                                while a < activeRecCount {
                                    let row = wT.advanced(by: scratch.activeLayerSpikes[a] * hSize)
                                    Self.addRow(row, to: stepBuf.baseAddress!, count: hSize)
                                    if hasNext {
                                        Self.addRow(row, to: prevBuf.baseAddress!, count: hSize)
                                    }
                                    if gated {
                                        scratch.ownCurrents.withUnsafeMutableBufferPointer { ownBuf in
                                            Self.addRow(row, to: ownBuf.baseAddress!, count: hSize)
                                        }
                                    }
                                    a += 1
                                }
                            }
                        }
                    }
                }

                if gated {
                    let hasNext = (layerIdx + 1) < numLayers
                    scratch.stepCurrents.withUnsafeMutableBufferPointer { stepBuf in
                        scratch.stepCurrentsPrev.withUnsafeMutableBufferPointer { prevBuf in
                            scratch.memoryOut.withUnsafeMutableBufferPointer { memBuf in
                                let mem = memBuf.baseAddress!.advanced(by: thisLayerOffset)
                                if t == 0 {
                                    aPrev.withUnsafeMutableBufferPointer { aBuf in
                                        scratch.ownCurrents.withUnsafeBufferPointer { ownBuf in
                                            updateGatedMemory(
                                                layer: layerIdx, own: ownBuf.baseAddress!,
                                                h: aBuf.baseAddress!.advanced(by: thisLayerOffset),
                                                out: mem, count: hSize
                                            )
                                        }
                                    }
                                }
                                Self.addRow(UnsafePointer(mem), to: stepBuf.baseAddress!, count: hSize)
                                if hasNext {
                                    Self.addRow(UnsafePointer(mem), to: prevBuf.baseAddress!, count: hSize)
                                }
                            }
                        }
                    }
                }

                vPrev.withUnsafeMutableBufferPointer { vBuf in
                    sPrev.withUnsafeMutableBufferPointer { sBuf in
                        aPrev.withUnsafeMutableBufferPointer { aBuf in
                            scratch.stepCurrents.withUnsafeBufferPointer { curBuf in
                                readoutSum.withUnsafeMutableBufferPointer { sumBuf in
                                    stepLayer(
                                        isLast: (layerIdx + 1) == numLayers,
                                        layer: layerIdx,
                                        vPtr: vBuf.baseAddress!.advanced(by: thisLayerOffset),
                                        sPtr: sBuf.baseAddress!.advanced(by: thisLayerOffset),
                                        aPtr: adaptPointer(aBuf.baseAddress!.advanced(by: thisLayerOffset), gated: gated, scratch: scratch),
                                        curPtr: curBuf.baseAddress!,
                                        readoutSumPtr: sumBuf.baseAddress!,
                                        count: hSize,
                                        scratch: scratch
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

        // 4. 線形層。最終層で積んだ閾値単位膜電位の平均 (学習側 logitsBatch と同じ)
        let invT = 1.0 / Float(timeSteps)
        let biasData = pBOut.data

        var activeOutCount = 0
        var k = 0
        while k < hSize {
            let rate = readoutSum[k] * invT
            if rate != 0.0 {
                scratch.activeReadoutIndices[activeOutCount] = k
                scratch.activeRates[activeOutCount] = rate
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

    /// `aPrev` と `ForwardScratch` を都度確保する版。ストリーミングでは使わない。
    /// 適応閾値・ゲート付き記憶の状態はフレームをまたいで残らないので、それらの構成の音響モデルには使えない
    public func forward(
        features: [Float],
        vPrev: inout [Float],
        sPrev: inout [Float],
        readoutSum: inout [Float],
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
            readoutSum: &readoutSum,
            logits: &logits,
            probabilities: &probabilities,
            scratch: scratch
        )
    }

    /// LIF に渡す適応閾値の状態。ゲート付き記憶の構成では a を記憶に使うので、常に 0 の作業領域を渡して適応閾値を無効にする
    @inline(__always)
    private func adaptPointer(_ a: UnsafeMutablePointer<Float>, gated: Bool, scratch: ForwardScratch) -> UnsafeMutablePointer<Float> {
        if gated {
            return scratch.adaptZeros
        }
        return a
    }

    /// dst[0..<count] に row を足す
    @inline(__always)
    private static func addRow(_ row: UnsafePointer<Float>, to dst: UnsafeMutablePointer<Float>, count: Int) {
        let limit = count - (count % 8)
        var n = 0
        while n < limit {
            let acc = UnsafeRawPointer(dst.advanced(by: n)).loadUnaligned(as: SIMD8<Float>.self)
            let w = UnsafeRawPointer(row.advanced(by: n)).loadUnaligned(as: SIMD8<Float>.self)
            UnsafeMutableRawPointer(dst.advanced(by: n)).storeBytes(of: acc + w, as: SIMD8<Float>.self)
            n += 8
        }
        while n < count {
            dst[n] += row[n]
            n += 1
        }
    }

    /// 隠れ層はハードリセット、最終層は閾値単位の膜電位を読んで余りを残す。
    /// 入力正規化の構成では、残差電流 (curPtr) は変えずに RMSNorm·ゲインを scratch.normCurrents に作って LIF へ渡す
    @inline(__always)
    private func stepLayer(
        isLast: Bool,
        layer: Int,
        vPtr: UnsafeMutablePointer<Float>,
        sPtr: UnsafeMutablePointer<Float>,
        aPtr: UnsafeMutablePointer<Float>,
        curPtr: UnsafePointer<Float>,
        readoutSumPtr: UnsafeMutablePointer<Float>,
        count: Int,
        scratch: ForwardScratch
    ) {
        if layer < pInputNormGains.count {
            var sumSq: Float = 0.0
            var n = 0
            while n < count {
                sumSq += curPtr[n] * curPtr[n]
                n += 1
            }
            let invRms = 1.0 / sqrt(sumSq / Float(max(1, count)) + inputNormEpsilon)
            let gains = pInputNormGains[layer].data
            scratch.normCurrents.withUnsafeMutableBufferPointer { normBuf in
                let norm = normBuf.baseAddress!
                var k = 0
                while k < count {
                    norm[k] = (curPtr[k] * invRms) * gains[k]
                    k += 1
                }
            }
            scratch.normCurrents.withUnsafeBufferPointer { normBuf in
                stepLayerScaled(
                    isLast: isLast, layer: layer, vPtr: vPtr, sPtr: sPtr, aPtr: aPtr,
                    curPtr: normBuf.baseAddress!, readoutSumPtr: readoutSumPtr, count: count
                )
            }
            return
        }
        stepLayerScaled(
            isLast: isLast, layer: layer, vPtr: vPtr, sPtr: sPtr, aPtr: aPtr,
            curPtr: curPtr, readoutSumPtr: readoutSumPtr, count: count
        )
    }

    @inline(__always)
    private func stepLayerScaled(
        isLast: Bool,
        layer: Int,
        vPtr: UnsafeMutablePointer<Float>,
        sPtr: UnsafeMutablePointer<Float>,
        aPtr: UnsafeMutablePointer<Float>,
        curPtr: UnsafePointer<Float>,
        readoutSumPtr: UnsafeMutablePointer<Float>,
        count: Int
    ) {
        if layer < pBetaLayers.count {
            pBetaLayers[layer].data.withUnsafeBufferPointer { bBuf in
                stepLayerWithBeta(
                    isLast: isLast, vPtr: vPtr, sPtr: sPtr, aPtr: aPtr, curPtr: curPtr,
                    readoutSumPtr: readoutSumPtr, count: count, betaPtr: bBuf.baseAddress
                )
            }
            return
        }
        stepLayerWithBeta(
            isLast: isLast, vPtr: vPtr, sPtr: sPtr, aPtr: aPtr, curPtr: curPtr,
            readoutSumPtr: readoutSumPtr, count: count, betaPtr: nil
        )
    }

    @inline(__always)
    private func stepLayerWithBeta(
        isLast: Bool,
        vPtr: UnsafeMutablePointer<Float>,
        sPtr: UnsafeMutablePointer<Float>,
        aPtr: UnsafeMutablePointer<Float>,
        curPtr: UnsafePointer<Float>,
        readoutSumPtr: UnsafeMutablePointer<Float>,
        count: Int,
        betaPtr: UnsafePointer<Float>?
    ) {
        if isLast {
            LIFNeuronEngine.stepReadoutAdaptiveSIMD8(
                config: lifConfig,
                vPtr: vPtr,
                sPtr: sPtr,
                aPtr: aPtr,
                curPtr: curPtr,
                readoutSumPtr: readoutSumPtr,
                count: count,
                betaPtr: betaPtr
            )
        } else {
            LIFNeuronEngine.stepAdaptiveSIMD8(
                config: lifConfig,
                vPtr: vPtr,
                sPtr: sPtr,
                aPtr: aPtr,
                curPtr: curPtr,
                count: count,
                betaPtr: betaPtr
            )
        }
    }
}

/// `SpikingNetwork.forward` の層間電流と次状態。呼び出し側が保持する。
public final class ForwardScratch: @unchecked Sendable {
    public var inputCurrents: [Float]
    public var stepCurrents: [Float]
    public var stepCurrentsPrev: [Float]
    /// 入力正規化の構成で LIF に渡す正規化済み電流
    public var normCurrents: [Float]
    public var activeSpikes: [Int]
    public var activeLayerSpikes: [Int]
    public var activeReadoutIndices: [Int]
    public var activeRates: [Float]
    public var readoutSums: [Float]
    /// ゲート付き記憶の構成で、各層が残差に足す電流
    public var ownCurrents: [Float]
    /// ゲート付き記憶の構成で、各層の記憶の出力 (1 フレームに 1 回更新) [層数 × 幅]
    public var memoryOut: [Float]
    /// ゲート付き記憶の構成で LIF に渡す適応閾値の状態 (gamma = 0 なので常に 0 のまま)
    public let adaptZeros: UnsafeMutablePointer<Float>

    deinit {
        adaptZeros.deallocate()
    }

    public init(maxHiddenDim: Int) {
        let size = max(1, maxHiddenDim)
        self.inputCurrents = [Float](repeating: 0.0, count: size)
        self.stepCurrents = [Float](repeating: 0.0, count: size)
        self.stepCurrentsPrev = [Float](repeating: 0.0, count: size)
        self.normCurrents = [Float](repeating: 0.0, count: size)
        self.activeSpikes = [Int](repeating: 0, count: size)
        self.activeLayerSpikes = [Int](repeating: 0, count: size)
        self.activeReadoutIndices = [Int](repeating: 0, count: size)
        self.activeRates = [Float](repeating: 0.0, count: size)
        self.readoutSums = []
        self.ownCurrents = [Float](repeating: 0.0, count: size)
        self.memoryOut = []
        self.adaptZeros = UnsafeMutablePointer<Float>.allocate(capacity: size)
        self.adaptZeros.initialize(repeating: 0.0, count: size)
    }
}
