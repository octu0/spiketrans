import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// 上位層の電流 RMSNorm で分散 0 のときの除算を避ける微小値 (推論側と共有)
internal let rmsNormEpsilon: Float = 1e-5

/// MLX GPU による SNN BPTT 学習エンジン (フレーム単位安定化 BPTT & ミニバッチ対応)
public final class MLXBPTTTrainer: @unchecked Sendable {
    public let network: MLXSpikingNetwork
    public let optimizer: Adam
    public let config: TrainingConfig
    /// 切り詰め BPTT の窓幅 (フレーム単位)。
    ///
    /// この本数ごとに膜電位・スパイク・適応閾値の勾配を切り離す。1 にすると
    /// フレームをまたぐ信用割り当てが完全に消え、第1段は実質フレーム独立の
    /// 分類器になる。大きくすると時間文脈を学習できる一方、計算グラフが深くなる。
    public let bpttWindow: Int
    public init(
        network: MLXSpikingNetwork,
        config: TrainingConfig = TrainingConfig(learningRate: 0.015),
        bpttWindow: Int = 16
    ) {
        self.network = network
        self.config = config
        self.optimizer = Adam(learningRate: config.learningRate)
        self.bpttWindow = max(1, bpttWindow)
    }

    /// 学習率を更新
    public func updateLearningRate(_ lr: Float) {
        self.optimizer.learningRate = lr
    }

    /// バッチ（複数発話）に対するフォワードとロジット系列 [B, T, outputDim] の計算
    public func logitsBatch(
        network: MLXSpikingNetwork,
        features: MLXArray           // [B, T, inputDim]
    ) -> MLXArray {
        let batchSize = features.shape[0]
        let seqLen = features.shape[1]
        let hMax = network.maxHiddenDim
        let tSteps = network.timeSteps
        let numLayers = network.numLayers
        let beta = network.lifConfig.beta
        let vTh = network.lifConfig.vTh
        let alpha = network.lifConfig.alpha
        let rho = network.lifConfig.rho
        let gamma = network.lifConfig.gamma

        // 層 0 の入力電流系列: [B, T, hMax] = [B, T, inputDim] @ [inputDim, hMax] + bH
        let currentSeq0 = matmul(features, network.wIn) + network.bH

        var v = [MLXArray](repeating: MLXArray.zeros([batchSize, hMax]), count: numLayers)
        var s = [MLXArray](repeating: MLXArray.zeros([batchSize, hMax]), count: numLayers)
        var a = [MLXArray](repeating: MLXArray.zeros([batchSize, hMax]), count: numLayers)

        var sAvgList: [MLXArray] = []
        sAvgList.reserveCapacity(seqLen)

        var t = 0
        while t < seqLen {
            let current0_t = currentSeq0[0..., t, 0...]
            var sSumFinal = MLXArray.zeros([batchSize, hMax])

            // 切り詰め BPTT: bpttWindow フレームごとに勾配を切り離す。
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
                // 層 0: 再帰 LIF 層 (時間文脈を担う)
                let rec0 = matmul(s[0], network.wRec)
                let totalCurrent0 = current0_t + rec0
                let vDecayed0 = (v[0] * beta) * (1.0 - s[0])
                v[0] = clip(vDecayed0 + totalCurrent0, min: -20.0, max: 20.0)

                a[0] = (a[0] * rho) + (s[0] * gamma)
                let dynVTh0 = vTh + a[0]

                let vRel0 = (v[0] - dynVTh0) * alpha
                let sSurrogate0 = 0.5 * (vRel0 / (1.0 + abs(vRel0)) + 1.0)
                let sHard0 = (dynVTh0 .<= v[0]).asType(.float32)
                s[0] = stopGradient(sHard0 - sSurrogate0) + sSurrogate0

                var prevCurrent = totalCurrent0

                // 層 1 以降: 前層スパイクを受けるフィードフォワード LIF 層。
                // 前層の発火は疎で電流の分散が小さすぎるため RMSNorm で単位スケールに揃え、
                // 前層の入力電流を加算する (電流空間の残差)。サロゲート勾配が閾値近傍の
                // 外で 0 になっても、この加算経路を通って下位層へ勾配が届く
                var l = 1
                while l < numLayers {
                    let upperIdx = l - 1
                    let denseCur = matmul(s[l - 1], network.wLayers[upperIdx]) + network.bHLayers[upperIdx]

                    let meanSq = mean(denseCur * denseCur, axis: -1, keepDims: true)
                    let rms = sqrt(meanSq + rmsNormEpsilon)
                    let normCur = (denseCur / rms) * network.gammaRMS[upperIdx]

                    let totalCurrent_l = normCur + prevCurrent
                    prevCurrent = totalCurrent_l

                    let vDecayed_l = (v[l] * beta) * (1.0 - s[l])
                    v[l] = clip(vDecayed_l + totalCurrent_l, min: -20.0, max: 20.0)

                    a[l] = (a[l] * rho) + (s[l] * gamma)
                    let dynVTh_l = vTh + a[l]

                    let vRel_l = (v[l] - dynVTh_l) * alpha
                    let sSurrogate_l = 0.5 * (vRel_l / (1.0 + abs(vRel_l)) + 1.0)
                    let sHard_l = (dynVTh_l .<= v[l]).asType(.float32)
                    s[l] = stopGradient(sHard_l - sSurrogate_l) + sSurrogate_l

                    l += 1
                }

                // 最終層のスパイクを積算
                sSumFinal = sSumFinal + s[numLayers - 1]
                step += 1
            }

            let sAvg_t = sSumFinal / Float(tSteps)
            sAvgList.append(sAvg_t)
            t += 1
        }

        // [B, T, hMax]
        let sAvgSeq = stacked(sAvgList, axis: 1)

        return matmul(sAvgSeq, network.wOut) + network.bOut
    }

    /// バッチ（複数発話）に対するフレーム整列教師の交差エントロピー損失
    public func lossBatch(
        network: MLXSpikingNetwork,
        features: MLXArray,          // [B, T, inputDim]
        targets: MLXArray            // [B, T] (Int32, パディングは -1)
    ) -> MLXArray {
        let logits = logitsBatch(network: network, features: features)

        // 重み: targets == -1 (パディング) は 0.0, targets == 0 (padId) は 0.3, targets > 0 は 1.0
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
    public func trainBatchCTC(
        featuresBatch: [[[Float]]],
        targetsBatch: [[Int]],
        blankId: Int = 0
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

        let lg = valueAndGrad(model: network) { (model: MLXSpikingNetwork, arrays: [MLXArray]) -> [MLXArray] in
            let logits = self.logitsBatch(network: model, features: arrays[0])
            return [MLXCTCLoss.loss(logits: logits, targets: extTargets)]
        }

        let (lossValues, grads) = lg(network, [featArray])
        let (clippedGrads, _) = clipGradNorm(gradients: grads, maxNorm: 5.0)
        optimizer.update(model: network, gradients: clippedGrads)
        // オプティマイザの状態も評価する。network と損失だけ評価すると
        // Adam の m/v が遅延グラフとして積み上がり、生存バッファ数が
        // Metal のリソース上限 (約 50 万) に達して落ちる
        eval(network, optimizer, lossValues)

        return lossValues[0].item(Float.self)
    }
}
