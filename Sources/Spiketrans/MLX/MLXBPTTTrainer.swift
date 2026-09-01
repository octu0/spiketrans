import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// MLX GPU による SNN BPTT 学習エンジン (フレーム単位安定化 BPTT & ミニバッチ対応)
public final class MLXBPTTTrainer: @unchecked Sendable {
    public let network: MLXMatryoshkaNetwork
    public let optimizer: Adam
    public let config: TrainingConfig

    public init(
        network: MLXMatryoshkaNetwork,
        config: TrainingConfig = TrainingConfig(learningRate: 0.015)
    ) {
        self.network = network
        self.config = config
        self.optimizer = Adam(learningRate: config.learningRate)
    }

    /// 学習率を更新
    public func updateLearningRate(_ lr: Float) {
        self.optimizer.learningRate = lr
    }

    /// バッチ（複数発話）に対するフォワード & 損失計算
    public func lossBatch(
        network: MLXMatryoshkaNetwork,
        features: MLXArray,          // [B, T, inputDim]
        targets: MLXArray            // [B, T] (Int32, パディングは -1)
    ) -> (loss: MLXArray, lossBase: MLXArray, lossMid: MLXArray, lossHigh: MLXArray) {
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

            // フレーム間の勾配切り離し (フレーム内 4-step BPTT による完全な数値安定性)
            v = stopGradient(v)
            s = stopGradient(s)
            a = stopGradient(a)

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

        // スライス別ロジット計算
        // 1. Base (128)
        let sBase = sAvgSeq[0..., 0..., 0..<128]
        let wOutBase = network.wOut[0..<128, 0...]
        let scaleBase = sqrt(Float(hMax) / 128.0)
        let logitsBase = (matmul(sBase, wOutBase) * scaleBase) + network.bOut

        // 2. Middle (512)
        let sMid = sAvgSeq[0..., 0..., 0..<512]
        let wOutMid = network.wOut[0..<512, 0...]
        let scaleMid = sqrt(Float(hMax) / 512.0)
        let logitsMid = (matmul(sMid, wOutMid) * scaleMid) + network.bOut

        // 3. High (1024)
        let logitsHigh = matmul(sAvgSeq, network.wOut) + network.bOut

        // 重み: targets == -1 (パディング) は 0.0, targets == 0 (padId) は 0.3, targets > 0 は 1.0
        let weights = which(targets .== -1, MLXArray(0.0), which(targets .== 0, MLXArray(0.3), MLXArray(1.0)))
        let totalWeight = sum(weights) + 1e-6
        let cleanTargets = clip(targets, min: 0, max: Float(network.outputDim - 1)).asType(.int32)

        func computeSliceLoss(logits: MLXArray) -> MLXArray {
            let lossArray = crossEntropy(logits: logits, targets: cleanTargets, weights: weights, reduction: .sum)
            return lossArray / totalWeight
        }

        let lossB = computeSliceLoss(logits: logitsBase)
        let lossM = computeSliceLoss(logits: logitsMid)
        let lossH = computeSliceLoss(logits: logitsHigh)
        let totalLoss = (lossB * 0.1) + (lossM * 0.2) + (lossH * 1.0)

        return (totalLoss, lossB, lossM, lossH)
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
            let res = self.lossBatch(network: model, features: fArr, targets: tArr)
            return res.loss
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
    ) -> (totalLoss: Float, lossBase: Float, lossMiddle: Float, lossHigh: Float) {
        let lossVal = trainBatch(featuresBatch: [features], targetsBatch: [targets])
        let subL = lossVal / 3.0
        return (lossVal, subL, subL, subL)
    }
}