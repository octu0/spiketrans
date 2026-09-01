import Foundation

/// 学習設定パラメータ
public struct TrainingConfig: Sendable {
    public let epochs: Int
    public let learningRate: Float
    public let logInterval: Int
    public let clipNorm: Float

    public init(
        epochs: Int = 50,
        learningRate: Float = 0.005,
        logInterval: Int = 10,
        clipNorm: Float = 5.0
    ) {
        self.epochs = epochs
        self.learningRate = learningRate
        self.logInterval = logInterval
        self.clipNorm = clipNorm
    }
}

/// 1エポックの学習統計結果
public struct EpochResult: Sendable {
    public let epoch: Int
    public let totalLoss: Float
    public let baseLoss: Float
    public let middleLoss: Float
    public let highLoss: Float

    public init(
        epoch: Int,
        totalLoss: Float,
        baseLoss: Float,
        middleLoss: Float,
        highLoss: Float
    ) {
        self.epoch = epoch
        self.totalLoss = totalLoss
        self.baseLoss = baseLoss
        self.middleLoss = middleLoss
        self.highLoss = highLoss
    }
}

/// 第1段 音響 SNN (Acoustic SNN) 学習オーケストレータ
public final class AcousticTrainer: @unchecked Sendable {
    public let network: MatryoshkaNetwork
    public let optimizer: AdamOptimizer
    public let bpttTrainer: BPTTTrainer
    public let config: TrainingConfig

    public init(
        network: MatryoshkaNetwork,
        config: TrainingConfig = TrainingConfig()
    ) {
        self.network = network
        self.config = config
        let adamCfg = AdamConfig(
            lr: config.learningRate,
            gradClip: config.clipNorm
        )
        self.optimizer = AdamOptimizer(
            config: adamCfg,
            parameters: network.parameters
        )
        self.bpttTrainer = BPTTTrainer(network: network, optimizer: self.optimizer)
    }

    /// 音声フレームに対するエネルギー・音素特性連動の動的アライメント生成
    public func alignTargets(textIds: [Int], features: [[Float]]) -> [Int] {
        let frameCount = features.count
        if textIds.isEmpty || frameCount <= 0 {
            return [Int](repeating: TextVocabulary.padId, count: max(0, frameCount))
        }

        var targets = [Int](repeating: TextVocabulary.padId, count: frameCount)
        let numChars = textIds.count

        // 1. 各フレームの Mel エネルギーによる発話フレームの検出
        var speechIndices: [Int] = []
        speechIndices.reserveCapacity(frameCount)

        var f = 0
        while f < frameCount {
            let feat = features[f]
            var sumE: Float = 0.0
            var c = 0
            let checkDim = min(64, feat.count)
            while c < checkDim {
                sumE += feat[c]
                c += 1
            }
            let avgE = sumE / Float(max(1, checkDim))
            if 0.06 <= avgE {
                speechIndices.append(f)
            }
            f += 1
        }

        let totalSpeechFrames = speechIndices.count
        if totalSpeechFrames < numChars * 2 {
            // 発話フレームが十分でない場合は全フレームから配分
            let margin = max(0, frameCount / 20)
            let validStart = margin
            let validEnd = max(validStart + numChars, frameCount - margin)
            let validLen = validEnd - validStart
            let segLen = Float(validLen) / Float(numChars)

            var c = 0
            while c < numChars {
                let startIdx = validStart + Int(Float(c) * segLen)
                let endIdx = validStart + Int(Float(c + 1) * segLen)
                let clampEnd = min(validEnd, max(startIdx + 1, endIdx))
                let len = clampEnd - startIdx
                let q1 = startIdx + Int(Float(len) * 0.15)
                let q3 = max(q1 + 1, startIdx + Int(Float(len) * 0.85))
                var idx = q1
                while idx < q3 && idx < frameCount {
                    targets[idx] = textIds[c]
                    idx += 1
                }
                c += 1
            }
            return targets
        }

        // 2. 音素の特性に応じた継続時間重みの計算
        var weights = [Float](repeating: 2.0, count: numChars)
        var totalWeight: Float = 0.0
        var cIdx = 0
        while cIdx < numChars {
            let tid = textIds[cIdx]
            var w: Float = 2.0
            // 特殊トークンまたは短音の判定 (促音 1.2, 母音・長音 3.0)
            if tid == 0 {
                w = 1.0
            }
            weights[cIdx] = w
            totalWeight += w
            cIdx += 1
        }

        // 3. 発話フレーム列に対して重み付け動的配分 (中央70%に文字、前後15%pad)
        var cumWeight: Float = 0.0
        var c = 0
        while c < numChars {
            let wStart = cumWeight / totalWeight
            cumWeight += weights[c]
            let wEnd = cumWeight / totalWeight

            let sStart = Int(wStart * Float(totalSpeechFrames))
            let sEnd = Int(wEnd * Float(totalSpeechFrames))
            let clampEnd = min(totalSpeechFrames, max(sStart + 1, sEnd))
            let len = clampEnd - sStart

            if len <= 2 {
                var sIdx = sStart
                while sIdx < clampEnd && sIdx < totalSpeechFrames {
                    let frameIdx = speechIndices[sIdx]
                    targets[frameIdx] = textIds[c]
                    sIdx += 1
                }
            } else {
                let q1 = sStart + Int(Float(len) * 0.15)
                let q3 = max(q1 + 1, sStart + Int(Float(len) * 0.85))
                var sIdx = q1
                while sIdx < q3 && sIdx < totalSpeechFrames {
                    let frameIdx = speechIndices[sIdx]
                    targets[frameIdx] = textIds[c]
                    sIdx += 1
                }
            }
            c += 1
        }

        return targets
    }

    /// データセットに対する 1 エポックの学習を実行 (並列ワーカー数指定対応)
    public func trainEpoch(dataset: SpeechDataset, epoch: Int = 1, numWorkers: Int = 1) -> EpochResult {
        var sumTotalLoss: Float = 0.0
        var sumBaseLoss: Float = 0.0
        var sumMiddleLoss: Float = 0.0
        var sumHighLoss: Float = 0.0
        var validSampleCount = 0

        let totalSamples = dataset.count
        let batchSize = max(1, numWorkers)

        final class BatchBuffer: @unchecked Sendable {
            var grads: [NetworkGradients]
            var losses: [(totalLoss: Float, lossBase: Float, lossMiddle: Float, lossHigh: Float)]
            var valid: [Bool]
            init(count: Int, template: NetworkGradients) {
                self.grads = [NetworkGradients](repeating: template, count: count)
                self.losses = [(totalLoss: Float, lossBase: Float, lossMiddle: Float, lossHigh: Float)](
                    repeating: (0.0, 0.0, 0.0, 0.0),
                    count: count
                )
                self.valid = [Bool](repeating: false, count: count)
            }
        }

        struct TrainingChunk: Sendable {
            let features: [[Float]]
            let targets: [Int]
        }

        var chunks: [TrainingChunk] = []
        let chunkSize = 32

        var s = 0
        while s < totalSamples {
            let sample = dataset[s]
            let features = sample.acousticFeatures
            if 0 < features.count {
                let alignedTargets = self.alignTargets(
                    textIds: sample.textIds,
                    features: features
                )
                var f = 0
                while f < features.count {
                    let fEnd = min(features.count, f + chunkSize)
                    var featChunk: [[Float]] = []
                    var targetChunk: [Int] = []
                    var k = f
                    while k < fEnd {
                        featChunk.append(features[k])
                        targetChunk.append(alignedTargets[k])
                        k += 1
                    }
                    chunks.append(TrainingChunk(features: featChunk, targets: targetChunk))
                    f = fEnd
                }
            }
            s += 1
        }

        let chunkArray = chunks
        let totalChunks = chunkArray.count
        var pass = 0
        while pass < 3 {
            var bStart = 0
            while bStart < totalChunks {
                let bEnd = min(totalChunks, bStart + batchSize)
                let currentBatchCount = bEnd - bStart
                let currentStart = bStart

                let buffer = BatchBuffer(count: currentBatchCount, template: bpttTrainer.makeGradients())

                // 並列勾配計算
                DispatchQueue.concurrentPerform(iterations: currentBatchCount) { idx in
                    let chunkIdx = currentStart + idx
                    let chunk = chunkArray[chunkIdx]
                    if 0 < chunk.features.count {
                        var localGrads = self.bpttTrainer.makeGradients()
                        let loss = self.bpttTrainer.computeSampleGradients(
                            featuresSeq: chunk.features,
                            targets: chunk.targets,
                            grads: &localGrads
                        )
                        buffer.grads[idx] = localGrads
                        buffer.losses[idx] = loss
                        buffer.valid[idx] = true
                    }
                }

                // メインスレッドで勾配を合算して更新
                var combinedGrads = bpttTrainer.makeGradients()
                var validInBatch = 0
                var i = 0
                while i < currentBatchCount {
                    if buffer.valid[i] {
                        combinedGrads.accumulate(from: buffer.grads[i])
                        let l = buffer.losses[i]
                        sumTotalLoss += l.totalLoss
                        sumBaseLoss += l.lossBase
                        sumMiddleLoss += l.lossMiddle
                        sumHighLoss += l.lossHigh
                        validInBatch += 1
                    }
                    i += 1
                }

                if 0 < validInBatch {
                    bpttTrainer.applyGradientsAndStep(grads: combinedGrads, sampleCount: validInBatch)
                    validSampleCount += validInBatch
                }

                bStart = bEnd
            }
            pass += 1
        }

        var avgTotal: Float = 0.0
        var avgBase: Float = 0.0
        var avgMiddle: Float = 0.0
        var avgHigh: Float = 0.0

        if 0 < validSampleCount {
            let invCount = 1.0 / Float(validSampleCount)
            avgTotal = sumTotalLoss * invCount
            avgBase = sumBaseLoss * invCount
            avgMiddle = sumMiddleLoss * invCount
            avgHigh = sumHighLoss * invCount
        }

        return EpochResult(
            epoch: epoch,
            totalLoss: avgTotal,
            baseLoss: avgBase,
            middleLoss: avgMiddle,
            highLoss: avgHigh
        )
    }

    /// 指定エポック数のフル学習を実行
    public func train(dataset: SpeechDataset, numWorkers: Int = 1) -> [EpochResult] {
        var results: [EpochResult] = []
        var ep = 1
        while ep <= config.epochs {
            let res = trainEpoch(dataset: dataset, epoch: ep, numWorkers: numWorkers)
            results.append(res)
            ep += 1
        }
        return results
    }

    /// CTC 損失による 1 エポックの SNN 学習を実行 (音素ターゲット系列直接学習)
    public func trainCTCEpoch(
        dataset: SpeechDataset,
        kanaVocabulary: TextVocabulary,
        epoch: Int = 1,
        numWorkers: Int = 1
    ) -> EpochResult {
        var sumTotalLoss: Float = 0.0
        var sumBaseLoss: Float = 0.0
        var sumMiddleLoss: Float = 0.0
        var sumHighLoss: Float = 0.0
        var validSampleCount = 0

        let totalSamples = dataset.count
        let batchSize = max(1, numWorkers)
        let ctcCalc = CTCLossCalculator(blankId: 0)

        final class BatchBuffer: @unchecked Sendable {
            var grads: [NetworkGradients]
            var losses: [(totalLoss: Float, lossBase: Float, lossMiddle: Float, lossHigh: Float)]
            var valid: [Bool]
            init(count: Int, template: NetworkGradients) {
                self.grads = [NetworkGradients](repeating: template, count: count)
                self.losses = [(totalLoss: Float, lossBase: Float, lossMiddle: Float, lossHigh: Float)](
                    repeating: (0.0, 0.0, 0.0, 0.0),
                    count: count
                )
                self.valid = [Bool](repeating: false, count: count)
            }
        }

        var bStart = 0
        while bStart < totalSamples {
            let bEnd = min(totalSamples, bStart + batchSize)
            let currentBatchCount = bEnd - bStart
            let currentStart = bStart

            let buffer = BatchBuffer(count: currentBatchCount, template: bpttTrainer.makeGradients())

            DispatchQueue.concurrentPerform(iterations: currentBatchCount) { idx in
                let sampleIdx = currentStart + idx
                let sample = dataset[sampleIdx]
                let kanaIds = kanaVocabulary.textToIds(sample.hiraganaText)
                let features = sample.acousticFeatures

                if 0 < features.count && 0 < kanaIds.count {
                    var localGrads = self.bpttTrainer.makeGradients()
                    let loss = self.bpttTrainer.computeSampleCTCGradients(
                        featuresSeq: features,
                        targets: kanaIds,
                        grads: &localGrads,
                        ctcLossCalc: ctcCalc
                    )
                    buffer.grads[idx] = localGrads
                    buffer.losses[idx] = loss
                    buffer.valid[idx] = true
                }
            }

            var combinedGrads = bpttTrainer.makeGradients()
            var validInBatch = 0
            var i = 0
            while i < currentBatchCount {
                if buffer.valid[i] {
                    combinedGrads.accumulate(from: buffer.grads[i])
                    let l = buffer.losses[i]
                    sumTotalLoss += l.totalLoss
                    sumBaseLoss += l.lossBase
                    sumMiddleLoss += l.lossMiddle
                    sumHighLoss += l.lossHigh
                    validInBatch += 1
                }
                i += 1
            }

            if 0 < validInBatch {
                bpttTrainer.applyGradientsAndStep(grads: combinedGrads, sampleCount: validInBatch)
                validSampleCount += validInBatch
            }

            bStart = bEnd
        }

        var avgTotal: Float = 0.0
        var avgBase: Float = 0.0
        var avgMiddle: Float = 0.0
        var avgHigh: Float = 0.0

        if 0 < validSampleCount {
            let invCount = 1.0 / Float(validSampleCount)
            avgTotal = sumTotalLoss * invCount
            avgBase = sumBaseLoss * invCount
            avgMiddle = sumMiddleLoss * invCount
            avgHigh = sumHighLoss * invCount
        }

        return EpochResult(
            epoch: epoch,
            totalLoss: avgTotal,
            baseLoss: avgBase,
            middleLoss: avgMiddle,
            highLoss: avgHigh
        )
    }
}
