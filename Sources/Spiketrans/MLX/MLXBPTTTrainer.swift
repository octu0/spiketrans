import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// MLX GPU による SNN BPTT 学習エンジン (フレーム単位安定化 BPTT & ミニバッチ対応)
public final class MLXBPTTTrainer: @unchecked Sendable {
    public let network: MLXMatryoshkaNetwork
    public let optimizer: Adam
    public let config: TrainingConfig
    /// 切り詰め BPTT の窓幅 (フレーム単位)。
    ///
    /// この本数ごとに膜電位・スパイク・適応閾値の勾配を切り離す。1 にすると
    /// フレームをまたぐ信用割り当てが完全に消え、第1段は実質フレーム独立の
    /// 分類器になる。大きくすると時間文脈を学習できる一方、計算グラフが深くなる。
    public let bpttWindow: Int
    /// マトリョーシカ各スライスの損失重み。
    ///
    /// Base は「速度優先だが文字起こしとして成立する」ことが要件のため、
    /// 用途に応じて High 偏重から均等寄りへ調整できるようにしている。
    public let sliceWeightBase: Float
    public let sliceWeightMiddle: Float
    public let sliceWeightHigh: Float

    public init(
        network: MLXMatryoshkaNetwork,
        config: TrainingConfig = TrainingConfig(learningRate: 0.015),
        bpttWindow: Int = 16,
        sliceWeightBase: Float = 0.1,
        sliceWeightMiddle: Float = 0.2,
        sliceWeightHigh: Float = 1.0
    ) {
        self.network = network
        self.config = config
        self.optimizer = Adam(learningRate: config.learningRate)
        self.bpttWindow = max(1, bpttWindow)
        self.sliceWeightBase = sliceWeightBase
        self.sliceWeightMiddle = sliceWeightMiddle
        self.sliceWeightHigh = sliceWeightHigh
    }

    /// 学習率を更新
    public func updateLearningRate(_ lr: Float) {
        self.optimizer.learningRate = lr
    }

    /// バッチ（複数発話）に対するフォワードとマトリョーシカ 3 スライスのロジット系列の計算
    ///
    /// 戻り値はいずれも [B, T, outputDim]。損失の種類 (交差エントロピー / CTC) に依存しないため
    /// フレーム整列教師と CTC の双方から共有する。
    public func sliceLogits(
        network: MLXMatryoshkaNetwork,
        features: MLXArray           // [B, T, inputDim]
    ) -> (base: MLXArray, mid: MLXArray, high: MLXArray) {
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

        // スライス別ロジット計算
        // 1. Base (128)
        let sBase = sAvgSeq[0..., 0..., 0..<128]
        let wOutBase = network.wOut[0..<128, 0...]
        let scaleBase = sqrt(Float(hMax) / 128.0)
        let logitsBase = (matmul(sBase, wOutBase) * scaleBase) + network.bOutBase

        // 2. Middle (512)
        let sMid = sAvgSeq[0..., 0..., 0..<512]
        let wOutMid = network.wOut[0..<512, 0...]
        let scaleMid = sqrt(Float(hMax) / 512.0)
        let logitsMid = (matmul(sMid, wOutMid) * scaleMid) + network.bOutMid

        // 3. High (1024)
        let logitsHigh = matmul(sAvgSeq, network.wOut) + network.bOut

        return (base: logitsBase, mid: logitsMid, high: logitsHigh)
    }

    /// バッチ（複数発話）に対するフレーム整列教師の交差エントロピー損失
    public func lossBatch(
        network: MLXMatryoshkaNetwork,
        features: MLXArray,          // [B, T, inputDim]
        targets: MLXArray            // [B, T] (Int32, パディングは -1)
    ) -> (loss: MLXArray, lossBase: MLXArray, lossMid: MLXArray, lossHigh: MLXArray) {
        let logits = sliceLogits(network: network, features: features)
        let logitsBase = logits.base
        let logitsMid = logits.mid
        let logitsHigh = logits.high

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
        let totalLoss = (lossB * sliceWeightBase) + (lossM * sliceWeightMiddle) + (lossH * sliceWeightHigh)

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

    /// ロジット系列 (フラット [B, T, V]) の 1 サンプル分を対数確率に変換
    private func logSoftmaxSample(
        flatLogits: [Float],
        batchIndex: Int,
        frameCount: Int,
        maxT: Int,
        vocabSize: Int
    ) -> [[Float]] {
        var logProbs = [[Float]](
            repeating: [Float](repeating: 0.0, count: vocabSize),
            count: frameCount
        )

        var t = 0
        while t < frameCount {
            let base = ((batchIndex * maxT) + t) * vocabSize

            var maxLogit = -Float.greatestFiniteMagnitude
            var v = 0
            while v < vocabSize {
                let x = flatLogits[base + v]
                if maxLogit < x {
                    maxLogit = x
                }
                v += 1
            }

            var sumExp: Float = 0.0
            v = 0
            while v < vocabSize {
                sumExp += exp(flatLogits[base + v] - maxLogit)
                v += 1
            }
            let logSumExp = maxLogit + log(sumExp)

            v = 0
            while v < vocabSize {
                logProbs[t][v] = flatLogits[base + v] - logSumExp
                v += 1
            }
            t += 1
        }

        return logProbs
    }

    /// CTC ミニバッチ学習ステップ (GPU)
    ///
    /// CTC の前向き後ろ向きは Pure Swift の `CTCLossCalculator` で計算し、得られた
    /// dL/dLogit を `sum(logits * stopGradient(grad))` というサロゲート損失として MLX に注入する。
    /// このサロゲートのロジットに関する勾配は CTC の真の勾配に一致するため、以降の
    /// 自動微分でネットワーク全体に正しく逆伝播される。
    ///
    /// - Parameter targetsBatch: フレームに整列していないラベル列 (かな ID 列)。
    public func trainBatchCTC(
        featuresBatch: [[[Float]]],
        targetsBatch: [[Int]],
        blankId: Int = 0
    ) -> (totalLoss: Float, lossBase: Float, lossMiddle: Float, lossHigh: Float) {
        let bSize = featuresBatch.count
        if bSize == 0 {
            return (0.0, 0.0, 0.0, 0.0)
        }

        var maxT = 0
        var frameCounts = [Int](repeating: 0, count: bSize)
        var b = 0
        while b < bSize {
            let c = featuresBatch[b].count
            frameCounts[b] = c
            if maxT < c {
                maxT = c
            }
            b += 1
        }
        if maxT == 0 {
            return (0.0, 0.0, 0.0, 0.0)
        }

        let inDim = network.inputDim
        let vocabSize = network.outputDim
        var flatFeat = [Float](repeating: 0.0, count: bSize * maxT * inDim)

        b = 0
        while b < bSize {
            let fSeq = featuresBatch[b]
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

        let featArray = MLXArray(flatFeat, [bSize, maxT, inDim])

        // 1. 勾配を取らないフォワードでロジットを求め、CPU 側で CTC の損失と勾配を計算する
        let probe = sliceLogits(network: network, features: featArray)
        eval(probe.base, probe.mid, probe.high)

        let flatBase = probe.base.asArray(Float.self)
        let flatMid = probe.mid.asArray(Float.self)
        let flatHigh = probe.high.asArray(Float.self)

        // CTC の前向き後ろ向きは CPU 律速のため、(サンプル × スライス) 単位で並列化する。
        // 各タスクは gradFlat の互いに素な区間だけを書くのでロックは不要。
        let sliceCount = 3
        let planeSize = bSize * maxT * vocabSize

        // 勾配プレーンは生ポインタで確保する。Swift の Array を複数スレッドから
        // withUnsafeMutableBufferPointer すると同一要素への排他アクセス違反で abort するため。
        final class CTCBuffer: @unchecked Sendable {
            let planes: [UnsafeMutablePointer<Float>]
            let planeSize: Int
            var losses: [Float]
            var valid: [Bool]

            init(sliceCount: Int, planeSize: Int, taskCount: Int) {
                self.planeSize = planeSize
                var allocated: [UnsafeMutablePointer<Float>] = []
                allocated.reserveCapacity(sliceCount)
                var i = 0
                while i < sliceCount {
                    let p = UnsafeMutablePointer<Float>.allocate(capacity: planeSize)
                    p.initialize(repeating: 0.0, count: planeSize)
                    allocated.append(p)
                    i += 1
                }
                self.planes = allocated
                self.losses = [Float](repeating: 0.0, count: taskCount)
                self.valid = [Bool](repeating: false, count: taskCount)
            }

            /// プレーンの内容を Array として取り出す
            func planeArray(_ index: Int) -> [Float] {
                return Array(UnsafeBufferPointer(start: planes[index], count: planeSize))
            }

            deinit {
                for p in planes {
                    p.deinitialize(count: planeSize)
                    p.deallocate()
                }
            }
        }

        let taskCount = bSize * sliceCount
        let buffer = CTCBuffer(sliceCount: sliceCount, planeSize: planeSize, taskCount: taskCount)
        let logitPlanes = [flatBase, flatMid, flatHigh]
        let capturedFrameCounts = frameCounts
        let capturedTargets = targetsBatch
        let capturedMaxT = maxT
        let capturedVocab = vocabSize

        DispatchQueue.concurrentPerform(iterations: taskCount) { taskIdx in
            let sIdx = taskIdx / bSize
            let bIdx = taskIdx % bSize
            let frameCount = capturedFrameCounts[bIdx]
            let labels = capturedTargets[bIdx]
            if frameCount <= 0 || labels.isEmpty {
                return
            }

            let logProbs = self.logSoftmaxSample(
                flatLogits: logitPlanes[sIdx],
                batchIndex: bIdx,
                frameCount: frameCount,
                maxT: capturedMaxT,
                vocabSize: capturedVocab
            )
            let ctcCalc = CTCLossCalculator(blankId: blankId)
            let res = ctcCalc.computeLossAndGradients(logProbs: logProbs, targets: labels)

            buffer.losses[taskIdx] = res.loss
            buffer.valid[taskIdx] = true

            if res.gradients.count == frameCount {
                let dst = buffer.planes[sIdx]
                var t = 0
                while t < frameCount {
                    let row = res.gradients[t]
                    let offset = ((bIdx * capturedMaxT) + t) * capturedVocab
                    var v = 0
                    while v < capturedVocab {
                        dst[offset + v] = row[v]
                        v += 1
                    }
                    t += 1
                }
            }
        }

        var sliceLossSum = [Float](repeating: 0.0, count: sliceCount)
        var validCount: Float = 0.0
        var taskIdx = 0
        while taskIdx < taskCount {
            if buffer.valid[taskIdx] {
                let sIdx = taskIdx / bSize
                sliceLossSum[sIdx] += buffer.losses[taskIdx]
                if sIdx == 0 {
                    validCount += 1.0
                }
            }
            taskIdx += 1
        }

        if validCount == 0.0 {
            return (0.0, 0.0, 0.0, 0.0)
        }

        let invValid = 1.0 / validCount
        let scaleBase: Float = sliceWeightBase * invValid
        let scaleMid: Float = sliceWeightMiddle * invValid
        let scaleHigh: Float = sliceWeightHigh * invValid
        let gradBase = MLXArray(buffer.planeArray(0), [bSize, maxT, vocabSize]) * scaleBase
        let gradMid = MLXArray(buffer.planeArray(1), [bSize, maxT, vocabSize]) * scaleMid
        let gradHigh = MLXArray(buffer.planeArray(2), [bSize, maxT, vocabSize]) * scaleHigh
        eval(gradBase, gradMid, gradHigh)

        // 2. サロゲート損失で自動微分し、ネットワーク重みを更新する
        let lg = valueAndGrad(model: network) { (model: MLXMatryoshkaNetwork, arrays: [MLXArray]) -> [MLXArray] in
            let logits = self.sliceLogits(network: model, features: arrays[0])
            let surrogate = sum(logits.base * gradBase)
                + sum(logits.mid * gradMid)
                + sum(logits.high * gradHigh)
            return [surrogate]
        }

        let (_, grads) = lg(network, [featArray])
        let (clippedGrads, _) = clipGradNorm(gradients: grads, maxNorm: 5.0)
        optimizer.update(model: network, gradients: clippedGrads)
        eval(network)

        let lossBase = sliceLossSum[0] * invValid
        let lossMid = sliceLossSum[1] * invValid
        let lossHigh = sliceLossSum[2] * invValid
        let total = (lossBase * sliceWeightBase) + (lossMid * sliceWeightMiddle) + (lossHigh * sliceWeightHigh)

        return (total, lossBase, lossMid, lossHigh)
    }
}