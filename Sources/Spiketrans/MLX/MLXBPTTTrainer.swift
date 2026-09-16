import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// 上位層の電流 RMSNorm で分散 0 のときの除算を避ける微小値 (推論側と共有)
internal let rmsNormEpsilon: Float = 1e-5

/// compile() する系列長の上限 (32 の倍数に切り上げた後のフレーム数)。
/// compile は時間ループを静的に展開するため、実測で T=256 が 13 GB、T=500 が 46〜72 GB、
/// T=1000 で OOM。これを超える長い系列は eager で流す
public let compiledMaxFrames = 256

/// MLX 上の切り詰め BPTT。系列が `compiledMaxFrames` を超えると eager。
public final class MLXBPTTTrainer: @unchecked Sendable {
    public let network: MLXSpikingNetwork
    public let optimizer: ArrayLRAdam
    public let config: TrainingConfig
    /// 切り詰め BPTT の窓幅 (フレーム単位)。
    ///
    /// この本数ごとに膜電位・スパイク・適応閾値の勾配を切り離す。1 にすると
    /// フレームをまたぐ信用割り当てが完全に消え、第1段は実質フレーム独立の
    /// 分類器になる。大きくすると時間文脈を学習できる一方、計算グラフが深くなる。
    public let bpttWindow: Int

    /// 系列長 (padded maxT) をキーとするコンパイル済み CTC 学習ステップのキャッシュ
    private var compiledCTCSteps: [Int: ([MLXArray]) -> [MLXArray]] = [:]

    /// 系列長 (T) をキーとするコンパイル済み logitsBatch のキャッシュ
    private var compiledLogitsSteps: [Int: ([MLXArray]) -> [MLXArray]] = [:]

    /// compiledMaxFrames を超える系列をチャンク分割で学習するときの 1 チャンクのフレーム数
    public let longSequenceChunkFrames = compiledMaxFrames / 2
    /// false にすると長い系列も一括で微分する (比較用)
    public var chunkLongSequences = true

    /// CTC 学習ステップのコンパイル（キャッシュミス）回数
    public private(set) var ctcCompileCount: Int = 0

    /// CTC 学習ステップのキャッシュヒット回数
    public private(set) var ctcCacheHitCount: Int = 0

    /// MLX が使い回しのために抱えるバッファキャッシュの上限。
    /// 系列長バケットごとに別サイズのバッファが要るため、無制限だと約 50 GB まで膨らんで実メモリ 64 GB を使い切る。
    /// 逆に 1 ステップの作業領域 (B=64・T=256 で 15〜27 GB) より小さいと、同じ形が続いても毎ステップ確保し直して
    /// 2 倍遅くなる (256 フレーム: 0.8 → 1.6 秒)。その間に置く
    public static let cacheLimitBytes = 24 << 30

    public init(
        network: MLXSpikingNetwork,
        config: TrainingConfig = TrainingConfig(learningRate: 0.015),
        bpttWindow: Int = 16
    ) {
        self.network = network
        self.config = config
        self.optimizer = ArrayLRAdam(learningRate: config.learningRate)
        self.bpttWindow = max(1, bpttWindow)
        MLX.GPU.set(cacheLimit: Self.cacheLimitBytes)
    }

    /// 学習率を更新。配列で持つので compile 済みステップにも入力として渡る
    public func updateLearningRate(_ lr: Float) {
        self.optimizer.learningRate = MLXArray(lr)
    }

    /// 重みを保存済みの状態へ巻き戻す。Adam のモーメントも捨て、
    /// 古い重みを参照している compile 済みステップも作り直す
    public func rollback(to weights: SpikingNetworkWeights) {
        network.importWeights(from: weights)
        optimizer.resetState()
        compiledCTCSteps.removeAll()
        compiledLogitsSteps.removeAll()
        eval(network, optimizer)
    }

    /// 1 サブステップ分の全層 LIF 更新。
    /// 入力 [層 0 の入力電流, v_0...v_L-1, s_0...s_L-1, a_0...a_L-1]、出力 [v..., s..., a..., 最終層の読み出し]。
    /// 層 0 は再帰、層 1 以降は前層スパイクの RMSNorm 電流 + 前層電流の残差。
    /// 最終層だけハードリセットせず、閾値単位の膜電位を読んでから余りを残す
    func substep(network: MLXSpikingNetwork, arrays: [MLXArray]) -> [MLXArray] {
        let numLayers = network.numLayers
        let beta = network.lifConfig.beta
        let vTh = network.lifConfig.vTh
        let alpha = network.lifConfig.alpha
        let rho = network.lifConfig.rho
        let gamma = network.lifConfig.gamma
        let vMin = LIFNeuronEngine.vClampMin
        let vMax = LIFNeuronEngine.vClampMax
        let readoutK = LIFNeuronEngine.readoutClipInThresholdUnits

        let current0 = arrays[0]
        var v = Array(arrays[1..<(1 + numLayers)])
        var s = Array(arrays[(1 + numLayers)..<(1 + 2 * numLayers)])
        var a = Array(arrays[(1 + 2 * numLayers)..<(1 + 3 * numLayers)])
        var readout = MLXArray.zeros(like: v[0])

        var current = current0 + matmul(s[0], network.wRec)
        var l = 0
        while l < numLayers {
            if 0 < l {
                let upperIdx = l - 1
                let denseCur = matmul(s[l - 1], network.wLayers[upperIdx]) + network.bHLayers[upperIdx]
                // MLXFast.rmsNorm は valueAndGrad + compile で Metal 生存バッファ上限を超える
                let meanSq = mean(denseCur * denseCur, axis: -1, keepDims: true)
                let rms = sqrt(meanSq + rmsNormEpsilon)
                current = (denseCur / rms) * network.gammaRMS[upperIdx] + current
            }

            let isLast = (l + 1) == numLayers
            if isLast {
                v[l] = clip(v[l] * beta + current, min: vMin, max: vMax)
            } else {
                v[l] = clip((v[l] * beta) * (1.0 - s[l]) + current, min: vMin, max: vMax)
            }

            a[l] = (a[l] * rho) + (s[l] * gamma)
            let dynVTh = vTh + a[l]
            let vRel = (v[l] - dynVTh) * alpha
            let sSurrogate = 0.5 * (vRel / (1.0 + abs(vRel)) + 1.0)
            let sHard = (dynVTh .<= v[l]).asType(.float32)
            s[l] = stopGradient(sHard - sSurrogate) + sSurrogate

            if isLast {
                // 読み出しは閾値単位でクリップするが、勾配はクリップ前の値を通す (straight-through)。
                // clip の勾配は |v| >= vTh で 0 になり、発火しているニューロンから勾配が流れず
                // 学習が 4 倍以上遅くなった (JSUT 20ep 未学習 16.7% → 43.4%)
                let scaled = v[l] / vTh
                let clipped = clip(scaled, min: -readoutK, max: readoutK)
                readout = stopGradient(clipped - scaled) + scaled
                v[l] = clip(v[l] - sHard * vTh, min: vMin, max: vMax)
            }
            l += 1
        }
        return v + s + a + [readout]
    }

    /// バッチ（複数発話）に対するフォワードとロジット系列 [B, T, outputDim] の計算
    public func logitsBatch(
        network: MLXSpikingNetwork,
        features: MLXArray,          // [B, T, inputDim]
        compiled: Bool = false
    ) -> MLXArray {
        if compiled {
            let seqLen = features.shape[1]
            if let cached = compiledLogitsSteps[seqLen] {
                return cached([features])[0]
            }
            let fn = compile(inputs: [network]) { arrays in
                return [self.logitsBatch(network: network, features: arrays[0], compiled: false)]
            }
            compiledLogitsSteps[seqLen] = fn
            return fn([features])[0]
        }
        let batchSize = features.shape[0]
        let seqLen = features.shape[1]
        let hMax = network.maxHiddenDim
        let numLayers = network.numLayers

        // 層 0 の入力電流系列: [B, T, hMax] = [B, T, inputDim] @ [inputDim, hMax] + bH
        let currentSeq0 = matmul(features, network.wIn) + network.bH
        var v = [MLXArray](repeating: MLXArray.zeros([batchSize, hMax]), count: numLayers)
        var s = [MLXArray](repeating: MLXArray.zeros([batchSize, hMax]), count: numLayers)
        var a = [MLXArray](repeating: MLXArray.zeros([batchSize, hMax]), count: numLayers)
        let readoutList = forwardFrames(network: network, currentSeq0: currentSeq0, from: 0, to: seqLen, v: &v, s: &s, a: &a)
        return matmul(stacked(readoutList, axis: 1), network.wOut) + network.bOut
    }

    /// フレーム [t0, t1) を前向きに進め、各フレームの読み出し (サブステップ平均、[B, hMax]) を返す。
    /// 状態 v, s, a は更新して返す。bpttWindow フレームごとに状態の勾配を切り離す (切り詰め BPTT)
    func forwardFrames(
        network: MLXSpikingNetwork,
        currentSeq0: MLXArray,
        from t0: Int,
        to t1: Int,
        v: inout [MLXArray],
        s: inout [MLXArray],
        a: inout [MLXArray]
    ) -> [MLXArray] {
        let tSteps = network.timeSteps
        let numLayers = network.numLayers
        var readoutList: [MLXArray] = []
        readoutList.reserveCapacity(t1 - t0)

        var t = t0
        while t < t1 {
            let current0_t = currentSeq0[0..., t, 0...]
            var readoutSum = MLXArray.zeros(like: v[0])

            if (t % bpttWindow) == 0 {
                var l = 0
                while l < numLayers {
                    v[l] = stopGradient(v[l])
                    s[l] = stopGradient(s[l])
                    a[l] = stopGradient(a[l])
                    l += 1
                }
            }

            var step = 0
            while step < tSteps {
                var inputs: [MLXArray] = [current0_t]
                inputs.append(contentsOf: v)
                inputs.append(contentsOf: s)
                inputs.append(contentsOf: a)
                let out = substep(network: network, arrays: inputs)
                v = Array(out[0..<numLayers])
                s = Array(out[numLayers..<(2 * numLayers)])
                a = Array(out[(2 * numLayers)..<(3 * numLayers)])
                readoutSum = readoutSum + out[3 * numLayers]
                step += 1
            }

            readoutList.append(readoutSum / Float(tSteps))
            t += 1
        }
        return readoutList
    }

    /// 長い系列の損失とパラメータ勾配をチャンク分割で求める。
    ///
    /// 全系列を一度に微分すると逆伝播用の中間値が系列長に比例して残り (64 件 × 256 フレームで
    /// 15 GB、768 フレームで 110 GB)、実メモリを超えると 1 バッチ 30 秒かかる。切り詰め BPTT は
    /// 窓の外へ勾配を流さないので、次の 3 段で同じ勾配を小さなメモリで得られる。
    ///   1. 勾配なしの前向きで、チャンク先頭の状態と全フレームのロジットを取る
    ///   2. CTC 損失のロジットについての勾配を vjp で求める
    ///   3. チャンクごとに前向きをやり直し、ロジットとその勾配の内積を逆伝播してパラメータ勾配を足す
    /// チャンク境界は窓境界に揃えるので、一括で微分した結果と一致する
    func chunkedLossAndGradients(
        features: MLXArray,
        extTargets: MLXCTCLoss.ExtendedTargets
    ) -> (loss: MLXArray, gradients: ModuleParameters) {
        let batchSize = features.shape[0]
        let seqLen = features.shape[1]
        let hMax = network.maxHiddenDim
        let numLayers = network.numLayers
        let chunkFrames = max(bpttWindow, (longSequenceChunkFrames / bpttWindow) * bpttWindow)

        // 1. 勾配なしの前向き
        var v = [MLXArray](repeating: MLXArray.zeros([batchSize, hMax]), count: numLayers)
        var s = [MLXArray](repeating: MLXArray.zeros([batchSize, hMax]), count: numLayers)
        var a = [MLXArray](repeating: MLXArray.zeros([batchSize, hMax]), count: numLayers)
        let currentSeq0 = stopGradient(matmul(features, network.wIn) + network.bH)
        var chunkStarts: [[MLXArray]] = []
        var logitsChunks: [MLXArray] = []
        var t0 = 0
        while t0 < seqLen {
            let t1 = min(seqLen, t0 + chunkFrames)
            chunkStarts.append(v + s + a)
            let readouts = forwardFrames(network: network, currentSeq0: currentSeq0, from: t0, to: t1, v: &v, s: &s, a: &a)
            let chunkLogits = matmul(stacked(readouts, axis: 1), network.wOut) + network.bOut
            eval(v + s + a + [chunkLogits])
            logitsChunks.append(chunkLogits)
            t0 = t1
        }
        let logits = concatenated(logitsChunks, axis: 1)

        // 2. CTC 損失のロジット勾配
        let (lossOut, cotangents) = vjp(
            { (arrays: [MLXArray]) -> [MLXArray] in
                return [MLXCTCLoss.loss(logits: arrays[0], targets: extTargets)]
            },
            primals: [logits],
            cotangents: [MLXArray(Float(1.0))]
        )
        let gradLogits = cotangents[0]
        eval(lossOut[0], gradLogits)

        // 3. チャンクごとの逆伝播
        var total: ModuleParameters? = nil
        var c = 0
        t0 = 0
        while t0 < seqLen {
            let t1 = min(seqLen, t0 + chunkFrames)
            let lg = valueAndGrad(model: network) { (model: MLXSpikingNetwork, arrays: [MLXArray]) -> [MLXArray] in
                let cur = matmul(arrays[0], model.wIn) + model.bH
                var cv = Array(arrays[2..<(2 + numLayers)])
                var cs = Array(arrays[(2 + numLayers)..<(2 + 2 * numLayers)])
                var ca = Array(arrays[(2 + 2 * numLayers)..<(2 + 3 * numLayers)])
                let readouts = self.forwardFrames(network: model, currentSeq0: cur, from: 0, to: t1 - t0, v: &cv, s: &cs, a: &ca)
                let chunkLogits = matmul(stacked(readouts, axis: 1), model.wOut) + model.bOut
                return [sum(chunkLogits * arrays[1])]
            }
            var inputs: [MLXArray] = [features[0..., t0..<t1, 0...], gradLogits[0..., t0..<t1, 0...]]
            inputs.append(contentsOf: chunkStarts[c])
            let (_, grads) = lg(network, inputs)
            switch total {
            case .some(let acc):
                total = acc.mapValues(grads) { (x: MLXArray, y: MLXArray?) -> MLXArray in
                    return x + (y ?? MLXArray.zeros(like: x))
                }
            case .none:
                total = grads
            }
            eval(total!.flattenedValues())
            c += 1
            t0 = t1
        }
        return (lossOut[0], total!)
    }

    /// バッチ（複数発話）に対するフレーム整列教師の交差エントロピー損失
    public func lossBatch(
        network: MLXSpikingNetwork,
        features: MLXArray,          // [B, T, inputDim]
        targets: MLXArray            // [B, T] (Int32, パディングは -1)
    ) -> MLXArray {
        let logits = logitsBatch(network: network, features: features)

        // 重み: targets == -1 (パディング) は 0.0, targets == 0 (padId) は 0.3, 0 < targets は 1.0
        let weights = which(targets .== -1, MLXArray(0.0), which(targets .== 0, MLXArray(0.3), MLXArray(1.0)))
        let totalWeight = sum(weights) + 1e-6
        let cleanTargets = clip(targets, min: 0, max: Float(network.outputDim - 1)).asType(.int32)

        let lossArray = crossEntropy(logits: logits, targets: cleanTargets, weights: weights, reduction: .sum)
        return lossArray / totalWeight
    }

    /// ミニバッチ学習ステップ
    public func trainBatch(
        featuresBatch: [[[Float]]],
        targetsBatch: [[Int]]
    ) -> Float {
        let bSize = featuresBatch.count
        if bSize == 0 { return 0.0 }

        var maxT = 0
        for f in featuresBatch {
            maxT = max(maxT, f.count)
        }
        if maxT == 0 { return 0.0 }

        let inDim = network.inputDim
        var flatFeat = [Float](repeating: 0.0, count: bSize * maxT * inDim)
        var flatTgt = [Int32](repeating: -1, count: bSize * maxT)

        var b = 0
        while b < bSize {
            let fSeq = featuresBatch[b]
            let tSeq = targetsBatch[b]
            let curT = fSeq.count
            var t = 0
            while t < curT {
                let fVec = fSeq[t]
                let offset = ((b * maxT) + t) * inDim
                var d = 0
                while d < inDim {
                    flatFeat[offset + d] = fVec[d]
                    d += 1
                }
                t += 1
            }

            var ot = 0
            while ot < min(curT, tSeq.count) {
                flatTgt[(b * maxT) + ot] = Int32(tSeq[ot])
                ot += 1
            }
            b += 1
        }

        let featArray = MLXArray(flatFeat, [bSize, maxT, inDim])
        let targetArray = MLXArray(flatTgt, [bSize, maxT])

        let lg = valueAndGrad(model: network) { model, fArr, tArr -> MLXArray in
            return self.lossBatch(network: model, features: fArr, targets: tArr)
        }

        let (lossVal, grads) = lg(network, featArray, targetArray)
        let (clippedGrads, _) = clipGradNorm(gradients: grads, maxNorm: 5.0)
        optimizer.update(model: network, gradients: clippedGrads)
        eval(network, lossVal)

        return lossVal.item(Float.self)
    }

    /// 単一サンプル学習ステップ
    public func trainStep(
        features: [[Float]],
        targets: [Int]
    ) -> Float {
        return trainBatch(featuresBatch: [features], targetsBatch: [targets])
    }

    /// CTC ミニバッチ学習ステップ (GPU)
    ///
    /// CTC の前向き再帰を MLX 演算で構成しているため、フォワードは 1 回で済み、
    /// ロジットを CPU へ取り出す必要もない。
    ///
    /// - Parameter targetsBatch: フレームに整列していないラベル列 (かな ID 列)。
    @discardableResult
    public func trainBatchCTC(
        featuresBatch: [[[Float]]],
        targetsBatch: [[Int]],
        blankId: Int = 0,
        compiled: Bool = true
    ) -> Float {
        let bSize = featuresBatch.count
        if bSize == 0 {
            return 0.0
        }

        // ラベルが空のサンプルは CTC を定義できないので除外する
        var validFeatures: [[[Float]]] = []
        var validTargets: [[Int]] = []
        var frameCounts: [Int] = []
        var maxT = 0
        var b = 0
        while b < bSize {
            let frames = featuresBatch[b].count
            if 0 < frames && targetsBatch[b].isEmpty != true {
                validFeatures.append(featuresBatch[b])
                validTargets.append(targetsBatch[b])
                frameCounts.append(frames)
                if maxT < frames {
                    maxT = frames
                }
            }
            b += 1
        }
        if validFeatures.isEmpty || maxT == 0 {
            return 0.0
        }

        // 系列長を 32 の倍数へ切り上げる。
        // MLX は解放したバッファを形ごとに使い回すため、毎バッチ長さが違うと
        // 使い回せないバッファが溜まり Metal のリソース上限に達する。
        // 増えたフレームは inputLengths で凍結されるので損失は変わらない
        maxT = ((maxT + 31) / 32) * 32

        let validCount = validFeatures.count
        let inDim = network.inputDim
        var flatFeat = [Float](repeating: 0.0, count: validCount * maxT * inDim)
        b = 0
        while b < validCount {
            let fSeq = validFeatures[b]
            var t = 0
            while t < fSeq.count {
                let fVec = fSeq[t]
                let offset = ((b * maxT) + t) * inDim
                var d = 0
                while d < inDim {
                    flatFeat[offset + d] = fVec[d]
                    d += 1
                }
                t += 1
            }
            b += 1
        }

        let featArray = MLXArray(flatFeat, [validCount, maxT, inDim])
        let extTargets = MLXCTCLoss.ExtendedTargets(
            targetsBatch: validTargets,
            frameCounts: frameCounts,
            blankId: blankId
        )

        if compiledMaxFrames < maxT && chunkLongSequences {
            let (loss, grads) = chunkedLossAndGradients(features: featArray, extTargets: extTargets)
            let (clippedGrads, _) = clipGradNorm(gradients: grads, maxNorm: 5.0)
            optimizer.update(model: network, gradients: clippedGrads)
            eval(network, optimizer, loss)
            return loss.item(Float.self)
        }

        if compiled != true || compiledMaxFrames < maxT {
            let lg = valueAndGrad(model: network) { (model: MLXSpikingNetwork, arrays: [MLXArray]) -> [MLXArray] in
                let logits = self.logitsBatch(network: model, features: arrays[0], compiled: false)
                return [MLXCTCLoss.loss(logits: logits, targets: extTargets)]
            }

            let (lossValues, grads) = lg(network, [featArray])
            let (clippedGrads, _) = clipGradNorm(gradients: grads, maxNorm: 5.0)
            optimizer.update(model: network, gradients: clippedGrads)
            eval(network, optimizer, lossValues)

            return lossValues[0].item(Float.self)
        }

        let stepFn: ([MLXArray]) -> [MLXArray]
        if let cached = compiledCTCSteps[maxT] {
            ctcCacheHitCount += 1
            stepFn = cached
        } else {
            ctcCompileCount += 1
            let computeLoss = { (model: MLXSpikingNetwork, arrays: [MLXArray]) -> [MLXArray] in
                let logits = self.logitsBatch(network: model, features: arrays[0], compiled: false)
                let targets = MLXCTCLoss.ExtendedTargets(
                    extTargets: arrays[1],
                    skipMask: arrays[2],
                    validMask: arrays[3],
                    finalIndex1: arrays[4],
                    finalIndex2: arrays[5],
                    hasSecondFinal: arrays[6],
                    inputLengths: arrays[7]
                )
                let loss = MLXCTCLoss.loss(logits: logits, targets: targets)
                return [loss]
            }
            let lg = valueAndGrad(model: self.network, computeLoss)
            // 学習率は最後の入力配列。トレース時にここで差し替えるので、以後の呼び出しでは
            // 入力に渡した値がそのまま更新式に使われる。トレース用のプレースホルダ配列は
            // 実体を持たないので、更新後は元の配列へ戻す
            func step(arrays: [MLXArray]) -> [MLXArray] {
                let (lossValues, grads) = lg(self.network, Array(arrays[0..<8]))
                let (clippedGrads, _) = clipGradNorm(gradients: grads, maxNorm: 5.0)
                let outerLearningRate = self.optimizer.learningRate
                self.optimizer.learningRate = arrays[8]
                self.optimizer.update(model: self.network, gradients: clippedGrads)
                self.optimizer.learningRate = outerLearningRate
                return lossValues
            }
            let newStep = compile(
                inputs: [self.network, self.optimizer],
                outputs: [self.network, self.optimizer],
                step
            )
            compiledCTCSteps[maxT] = newStep
            stepFn = newStep
        }

        var inputs: [MLXArray] = [featArray]
        inputs.append(contentsOf: extTargets.toArrays())
        inputs.append(optimizer.learningRate)

        let lossValues = stepFn(inputs)
        // オプティマイザの状態も評価する。network と損失だけ評価すると
        // Adam の m/v が遅延グラフとして積み上がり、生存バッファ数が
        // Metal のリソース上限 (約 50 万) に達して落ちる
        eval(network, optimizer, lossValues)

        return lossValues[0].item(Float.self)
    }
}
