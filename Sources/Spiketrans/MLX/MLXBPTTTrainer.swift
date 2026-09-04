import Foundation
import MLX
import MLXNN
import MLXOptimizers

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
        let beta = network.lifConfig.beta
        let vTh = network.lifConfig.vTh
        let alpha = network.lifConfig.alpha
        let rho = network.lifConfig.rho
        let gamma = network.lifConfig.gamma

        // [B, T, hMax] = [B, T, inputDim] @ [inputDim, hMax] + bH
        let currentSeq = matmul(features, network.wIn) + network.bH

        var v = MLXArray.zeros([batchSize, hMax])
        var s = MLXArray.zeros([batchSize, hMax])
        var a = MLXArray.zeros([batchSize, hMax])

        var sAvgList: [MLXArray] = []
        sAvgList.reserveCapacity(seqLen)

        var t = 0
        while t < seqLen {
            let current_t = currentSeq[0..., t, 0...]
            var sSum = MLXArray.zeros([batchSize, hMax])

            // 切り詰め BPTT: bpttWindow フレームごとに勾配を切り離す。
            // 窓の内側ではフレームをまたいで勾配が流れるため、時間文脈を学習できる。
            if (t % bpttWindow) == 0 {
                v = stopGradient(v)
                s = stopGradient(s)
                a = stopGradient(a)
            }

            var step = 0
            while step < tSteps {
                let rec = matmul(s, network.wRec)
                let vDecayed = (v * beta) * (1.0 - s)
                v = clip(vDecayed + current_t + rec, min: -20.0, max: 20.0)

                // 適応型発火閾値 (ALIF: 生物の神経順応)
                a = (a * rho) + (s * gamma)
                let dynVTh = vTh + a

                // Fast Sigmoid Surrogate Gradient
                let vRel = (v - dynVTh) * alpha
                let sSurrogate = 0.5 * (vRel / (1.0 + abs(vRel)) + 1.0)
                let sHard = (v .>= dynVTh).asType(.float32)
                s = stopGradient(sHard - sSurrogate) + sSurrogate

                sSum = sSum + s
                step += 1
            }

            let sAvg_t = sSum / Float(tSteps)
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

        for b in 0..<bSize {
            let fSeq = featuresBatch[b]
            let tSeq = targetsBatch[b]
            let curT = fSeq.count
            for t in 0..<curT {
                let fVec = fSeq[t]
                let offset = (b * maxT + t) * inDim
                for d in 0..<inDim {
                    flatFeat[offset + d] = fVec[d]
                }
                flatTgt[b * maxT + t] = Int32(tSeq[t])
            }
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
