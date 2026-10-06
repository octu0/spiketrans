import Foundation
import MLX
import MLXNN

/// MLX 上の SNN。層 0 は再帰 LIF、層 1 以降は FF LIF (`upperRecurrent` なら層 1 以降も再帰)。
/// 上位層は結合電流を RMSNorm し、前層の入力電流を足す。疎スパイクをそのまま
/// 重み付けすると上位層が沈黙するため。
public final class MLXSpikingNetwork: Module, @unchecked Sendable {
    public let numLayers: Int
    public let inputDim: Int
    public let maxHiddenDim: Int
    public let outputDim: Int
    public let timeSteps: Int
    public let lifConfig: LIFConfig

    // 層 0
    public var wIn: MLXArray       // [inputDim, maxHiddenDim]
    public var wRec: MLXArray      // [maxHiddenDim, maxHiddenDim]
    public var bH: MLXArray        // [maxHiddenDim]

    // 層 1 以降
    public var wLayers: [MLXArray]   // 各 [maxHiddenDim, maxHiddenDim]
    public var bHLayers: [MLXArray]  // 各 [maxHiddenDim]
    public var gammaRMS: [MLXArray]  // 各 [maxHiddenDim]
    /// 層 1 以降の再帰結合 (同じ層の直前サブステップのスパイクから)。再帰なしの構成では空
    public var wRecLayers: [MLXArray]  // 各 [maxHiddenDim, maxHiddenDim]
    /// ニューロンごとの減衰率の logit (減衰率 = sigmoid)。共通 beta の構成では空
    public var betaLogits: [MLXArray]  // 各 [maxHiddenDim]、層 0 から numLayers 本
    /// 各層の LIF に入る電流全体 (残差込み) を RMSNorm するときのゲイン。正規化しない構成では空
    public var inputNormGains: [MLXArray]  // 各 [maxHiddenDim]、層 0 から numLayers 本
    /// ゲート付き記憶 (RG-LRU 型) の係数。行の意味は `SpikingNetworkWeights.gateLayers`。無い構成では空。
    /// 記憶の状態は適応閾値の状態 a に持つので、適応閾値とは併用しない
    public var gateParams: [MLXArray]  // 各 [gateRows, maxHiddenDim]、層 0 から numLayers 本
    /// 声の種類の補助ヘッド [wVoice [maxHiddenDim, voiceClasses], bVoice [voiceClasses]]。無い構成では空
    public var voiceHead: [MLXArray]
    /// 事前学習の予測ヘッド [wPred [maxHiddenDim, n * inputDim], bPred [n * inputDim]] (n = 予測先の数)。無い構成では空
    public var predictionHead: [MLXArray]

    /// 補助ヘッドが読む層 (最終層の 1 つ下。層数が足りなければ層 0)
    public var voiceLayer: Int {
        return max(0, numLayers - 2)
    }

    /// 発火 (0/1) の代わりに、閾値の前後だけ 0〜1 の連続値になる出力を次の層へ渡す対照実験用の切り替え。
    /// 重みには保存しない。CPU 推論は発火したニューロンだけを足す作りなので、この構成は MLX でしか推論できない
    public var continuousSpikes: Bool = false

    // リードアウト
    public var wOut: MLXArray      // [maxHiddenDim, outputDim]
    public var bOut: MLXArray      // [outputDim]

    public init(
        numLayers: Int = 1,
        inputDim: Int = 128,
        maxHiddenDim: Int = 1024,
        outputDim: Int = 523,
        timeSteps: Int = 4,
        lifConfig: LIFConfig = LIFConfig(),
        upperRecurrent: Bool = false,
        learnedBeta: Bool = false,
        inputNorm: Bool = false,
        inputNormGainInit: Float = 1.0,
        voiceHead: Bool = false,
        gatedMemory: Bool = false,
        predictionHead: Bool = false,
        predictionTargets: Int = 1,
        predictionClasses: Int = 0
    ) {
        self.numLayers = max(1, numLayers)
        self.inputDim = inputDim
        self.maxHiddenDim = maxHiddenDim
        self.outputDim = outputDim
        self.timeSteps = timeSteps
        self.lifConfig = lifConfig

        let scaleIn = sqrt(2.0 / Float(inputDim))
        let scaleRec = 0.1 / sqrt(Float(maxHiddenDim))
        let scaleOut = sqrt(2.0 / Float(maxHiddenDim))
        let scaleLayer = sqrt(2.0 / Float(maxHiddenDim))

        self.wIn = MLXRandom.uniform(low: -scaleIn, high: scaleIn, [inputDim, maxHiddenDim])
        self.wRec = MLXRandom.uniform(low: -scaleRec, high: scaleRec, [maxHiddenDim, maxHiddenDim])
        self.bH = MLXArray.zeros([maxHiddenDim])

        var wl: [MLXArray] = []
        var bl: [MLXArray] = []
        var gl: [MLXArray] = []
        var rl: [MLXArray] = []
        var l = 1
        while l < self.numLayers {
            wl.append(MLXRandom.uniform(low: -scaleLayer, high: scaleLayer, [maxHiddenDim, maxHiddenDim]))
            bl.append(MLXArray.zeros([maxHiddenDim]))
            gl.append(MLXArray.ones([maxHiddenDim]))
            if upperRecurrent {
                rl.append(MLXRandom.uniform(low: -scaleRec, high: scaleRec, [maxHiddenDim, maxHiddenDim]))
            }
            l += 1
        }
        self.wLayers = wl
        self.bHLayers = bl
        self.gammaRMS = gl
        self.wRecLayers = rl
        var bLogits: [MLXArray] = []
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
                let b = lo + (hi - lo) * frac
                values[n] = logf(b / (1.0 - b))
                n += 1
            }
            var li = 0
            while li < self.numLayers {
                bLogits.append(MLXArray(values, [maxHiddenDim]))
                li += 1
            }
        }
        self.betaLogits = bLogits
        var normGains: [MLXArray] = []
        if inputNorm {
            var li = 0
            while li < self.numLayers {
                normGains.append(MLXArray.ones([maxHiddenDim]) * inputNormGainInit)
                li += 1
            }
        }
        self.inputNormGains = normGains
        var gates: [MLXArray] = []
        if gatedMemory {
            for layer in SpikingNetworkWeights.initialGateLayers(numLayers: self.numLayers, hidden: maxHiddenDim) {
                gates.append(MLXArray(layer, [SpikingNetworkWeights.gateRows, maxHiddenDim]))
            }
        }
        self.gateParams = gates
        var head: [MLXArray] = []
        if voiceHead {
            let classes = SpikingNetworkWeights.voiceClasses
            head.append(MLXRandom.uniform(low: -scaleOut, high: scaleOut, [maxHiddenDim, classes]))
            head.append(MLXArray.zeros([classes]))
        }
        self.voiceHead = head
        var pred: [MLXArray] = []
        if predictionHead {
            var outDim = inputDim * max(1, predictionTargets)
            if 0 < predictionClasses {
                outDim = predictionClasses
            }
            pred.append(MLXRandom.uniform(low: -scaleOut, high: scaleOut, [maxHiddenDim, outDim]))
            pred.append(MLXArray.zeros([outDim]))
        }
        self.predictionHead = pred

        self.wOut = MLXRandom.uniform(low: -scaleOut, high: scaleOut, [maxHiddenDim, outputDim])
        self.bOut = MLXArray.zeros([outputDim])

        super.init()
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
            voiceHead: weights.hasVoiceHead,
            gatedMemory: weights.hasGatedMemory,
            predictionHead: weights.hasPredictionHead,
            predictionTargets: max(1, weights.predictionTargets)
        )
        self.importWeights(from: weights)
    }

    /// SpikingNetworkWeights から重みをインポート。
    /// [出力, 入力] の行優先なので転置して持つ
    public func importWeights(from data: SpikingNetworkWeights) {
        let hSize = data.maxHiddenDim
        // Module は Mirror で見つけたパラメータ配列をキャッシュする。プロパティに新しい MLXArray を
        // 代入するとキャッシュ側の古い配列が optimizer.update の対象のまま残り、学習が止まる。
        // update(parameters:) はキャッシュ済みの配列へ in-place に書き込むので、こちらを使う
        var params = ModuleParameters()
        params["wIn"] = .value(MLXArray(data.wIn, [hSize, data.inputDim]).transposed())
        params["wRec"] = .value(MLXArray(data.wRec, [hSize, hSize]).transposed())
        params["bH"] = .value(MLXArray(data.bH, [hSize]))
        params["wOut"] = .value(MLXArray(data.wOut, [data.outputDim, hSize]).transposed())
        params["bOut"] = .value(MLXArray(data.bOut, [data.outputDim]))

        var layerW: [NestedItem<String, MLXArray>] = []
        var layerB: [NestedItem<String, MLXArray>] = []
        var layerG: [NestedItem<String, MLXArray>] = []
        var l = 0
        while l < min(self.wLayers.count, data.wLayers.count) {
            layerW.append(.value(MLXArray(data.wLayers[l], [hSize, hSize]).transposed()))
            layerB.append(.value(MLXArray(data.bHLayers[l], [hSize])))
            layerG.append(.value(MLXArray(data.gammaRMS[l], [hSize])))
            l += 1
        }
        if 0 < layerW.count {
            params["wLayers"] = .array(layerW)
            params["bHLayers"] = .array(layerB)
            params["gammaRMS"] = .array(layerG)
        }
        if let rec = data.wRecLayers {
            var layerR: [NestedItem<String, MLXArray>] = []
            var r = 0
            while r < min(self.wRecLayers.count, rec.count) {
                layerR.append(.value(MLXArray(rec[r], [hSize, hSize]).transposed()))
                r += 1
            }
            if 0 < layerR.count {
                params["wRecLayers"] = .array(layerR)
            }
        }
        if let betas = data.betaLayers {
            var layerBeta: [NestedItem<String, MLXArray>] = []
            var bi = 0
            while bi < min(self.betaLogits.count, betas.count) {
                let logits = betas[bi].map { b -> Float in
                    let c = min(max(b, 1e-4), 1.0 - 1e-4)
                    return logf(c / (1.0 - c))
                }
                layerBeta.append(.value(MLXArray(logits, [hSize])))
                bi += 1
            }
            if 0 < layerBeta.count {
                params["betaLogits"] = .array(layerBeta)
            }
        }
        if let gains = data.inputNormGains {
            var layerGain: [NestedItem<String, MLXArray>] = []
            var gi = 0
            while gi < min(self.inputNormGains.count, gains.count) {
                layerGain.append(.value(MLXArray(gains[gi], [hSize])))
                gi += 1
            }
            if 0 < layerGain.count {
                params["inputNormGains"] = .array(layerGain)
            }
        }
        if let gates = data.gateLayers, gates.count == self.gateParams.count {
            params["gateParams"] = .array(gates.map { .value(MLXArray($0, [SpikingNetworkWeights.gateRows, hSize])) })
        }
        if let w = data.wPred, let b = data.bPred, self.predictionHead.count == 2 {
            params["predictionHead"] = .array([
                .value(MLXArray(w, [b.count, hSize]).transposed()),
                .value(MLXArray(b, [b.count]))
            ])
        }
        if let w = data.wVoice, let b = data.bVoice, self.voiceHead.count == 2 {
            let classes = SpikingNetworkWeights.voiceClasses
            params["voiceHead"] = .array([
                .value(MLXArray(w, [classes, hSize]).transposed()),
                .value(MLXArray(b, [classes]))
            ])
        }
        self.update(parameters: params)
        eval(self)
    }

    /// SpikingNetworkWeights へ重みをエクスポート
    public func exportWeights(vocabulary: TextVocabulary? = nil) -> SpikingNetworkWeights {
        var arraysToEval: [MLXArray] = [self.wIn, self.wRec, self.bH, self.wOut, self.bOut]
        var l = 0
        while l < wLayers.count {
            arraysToEval.append(wLayers[l])
            arraysToEval.append(bHLayers[l])
            arraysToEval.append(gammaRMS[l])
            l += 1
        }
        arraysToEval.append(contentsOf: wRecLayers)
        arraysToEval.append(contentsOf: betaLogits)
        arraysToEval.append(contentsOf: inputNormGains)
        arraysToEval.append(contentsOf: gateParams)
        arraysToEval.append(contentsOf: voiceHead)
        arraysToEval.append(contentsOf: predictionHead)
        eval(arraysToEval)

        var wl: [[Float]] = []
        var bl: [[Float]] = []
        var gl: [[Float]] = []
        l = 0
        while l < wLayers.count {
            wl.append(self.wLayers[l].transposed().asArray(Float.self))
            bl.append(self.bHLayers[l].asArray(Float.self))
            gl.append(self.gammaRMS[l].asArray(Float.self))
            l += 1
        }
        var rl: [[Float]]? = nil
        if wRecLayers.isEmpty != true {
            rl = wRecLayers.map { $0.transposed().asArray(Float.self) }
        }
        var betas: [[Float]]? = nil
        if betaLogits.isEmpty != true {
            betas = betaLogits.map { sigmoid($0).asArray(Float.self) }
        }
        var normGains: [[Float]]? = nil
        if inputNormGains.isEmpty != true {
            normGains = inputNormGains.map { $0.asArray(Float.self) }
        }
        var gates: [[Float]]? = nil
        if gateParams.isEmpty != true {
            gates = gateParams.map { $0.asArray(Float.self) }
        }
        var wPred: [Float]? = nil
        var bPred: [Float]? = nil
        if predictionHead.count == 2 {
            wPred = predictionHead[0].transposed().asArray(Float.self)
            bPred = predictionHead[1].asArray(Float.self)
        }
        var wVoice: [Float]? = nil
        var bVoice: [Float]? = nil
        if voiceHead.count == 2 {
            wVoice = voiceHead[0].transposed().asArray(Float.self)
            bVoice = voiceHead[1].asArray(Float.self)
        }

        return SpikingNetworkWeights(
            inputDim: inputDim,
            maxHiddenDim: maxHiddenDim,
            outputDim: outputDim,
            timeSteps: timeSteps,
            lifConfig: lifConfig,
            wIn: self.wIn.transposed().asArray(Float.self),
            wRec: self.wRec.transposed().asArray(Float.self),
            bH: self.bH.asArray(Float.self),
            wLayers: wl,
            bHLayers: bl,
            gammaRMS: gl,
            wRecLayers: rl,
            betaLayers: betas,
            inputNormGains: normGains,
            gateLayers: gates,
            wVoice: wVoice,
            bVoice: bVoice,
            wPred: wPred,
            bPred: bPred,
            wOut: self.wOut.transposed().asArray(Float.self),
            bOut: self.bOut.asArray(Float.self),
            vocabularyCharacters: vocabulary?.serializedCharacters
        )
    }
}
