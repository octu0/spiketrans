import Foundation

/// 言語 SNN の CPU 学習。入力はトークン埋め込み。
public final class LanguageTrainer: @unchecked Sendable {
    public let network: SpikingNetwork
    public let optimizer: AdamOptimizer
    public let bpttTrainer: BPTTTrainer
    public let textVocabulary: TextVocabulary
    public let config: TrainingConfig

    public init(
        network: SpikingNetwork,
        textVocabulary: TextVocabulary,
        config: TrainingConfig = TrainingConfig()
    ) {
        self.network = network
        self.textVocabulary = textVocabulary
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

    /// 正弦埋め込み。`LanguageDecoder` のワンホットとは別物。`languageBonus == 0` のあいだは学習しない。
    public func buildTokenFeature(tokenId: Int) -> [Float] {
        let dim = network.inputDim
        var feat = [Float](repeating: 0.0, count: dim)
        var d = 0
        while d < dim {
            let angle = Float(tokenId * 17 + d * 11 + 3)
            feat[d] = (sin(angle) * 0.5) + 0.5
            d += 1
        }
        return feat
    }

    public func trainEpoch(dataset: SpeechDataset, epoch: Int = 1, numWorkers: Int = 1) -> EpochResult {
        var sumTotalLoss: Float = 0.0
        var validSampleCount = 0

        let totalSamples = dataset.count
        let batchSize = max(1, numWorkers)

        final class BatchBuffer: @unchecked Sendable {
            var grads: [NetworkGradients]
            var losses: [Float]
            var valid: [Bool]
            init(count: Int, template: NetworkGradients) {
                self.grads = [NetworkGradients](repeating: template, count: count)
                self.losses = [Float](repeating: 0.0, count: count)
                self.valid = [Bool](repeating: false, count: count)
            }
        }

        var bStart = 0
        while bStart < totalSamples {
            let bEnd = min(totalSamples, bStart + batchSize)
            let currentBatchCount = bEnd - bStart
            let currentStart = bStart

            let buffer = BatchBuffer(count: currentBatchCount, template: bpttTrainer.makeGradients())

            // 並列勾配計算
            DispatchQueue.concurrentPerform(iterations: currentBatchCount) { idx in
                let sampleIdx = currentStart + idx
                let tokens = [TextVocabulary.sosId] + dataset.textIds(at: sampleIdx) + [TextVocabulary.eosId]

                if 1 < tokens.count {
                    var featSeq: [[Float]] = []
                    var targetSeq: [Int] = []

                    var i = 0
                    while i < (tokens.count - 1) {
                        featSeq.append(self.buildTokenFeature(tokenId: tokens[i]))
                        targetSeq.append(tokens[i + 1])
                        i += 1
                    }

                    var localGrads = self.bpttTrainer.makeGradients()
                    let loss = self.bpttTrainer.computeSampleGradients(
                        featuresSeq: featSeq,
                        targets: targetSeq,
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
                        sumTotalLoss += l
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

        return EpochResult(
            epoch: epoch,
            totalLoss: sumTotalLoss / Float(max(1, validSampleCount))
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

    /// ひらがな・音素系列から漢字テキストへの変換自己回帰学習を 1 エポック実行
    public func trainKanaToKanjiEpoch(
        dataset: SpeechDataset,
        kanaVocabulary: TextVocabulary,
        epoch: Int = 1,
        numWorkers: Int = 1
    ) -> EpochResult {
        var sumTotalLoss: Float = 0.0
        var validSampleCount = 0

        let totalSamples = dataset.count
        let batchSize = max(1, numWorkers)

        final class BatchBuffer: @unchecked Sendable {
            var grads: [NetworkGradients]
            var losses: [Float]
            var valid: [Bool]
            init(count: Int, template: NetworkGradients) {
                self.grads = [NetworkGradients](repeating: template, count: count)
                self.losses = [Float](repeating: 0.0, count: count)
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
                let kanaIds = kanaVocabulary.textToIds(dataset.hiraganaText(at: sampleIdx))
                let kanjiIds = dataset.textIds(at: sampleIdx)

                if 0 < kanaIds.count && 0 < kanjiIds.count {
                    // 入力: かなトークン特徴量系列
                    var featSeq: [[Float]] = []
                    var targetSeq: [Int] = []

                    let stepCount = max(kanaIds.count, kanjiIds.count)
                    var s = 0
                    while s < stepCount {
                        var kId = TextVocabulary.padId
                        if s < kanaIds.count {
                            kId = kanaIds[s]
                        }
                        featSeq.append(self.buildTokenFeature(tokenId: kId))

                        var tId = TextVocabulary.padId
                        if s < kanjiIds.count {
                            tId = kanjiIds[s]
                        }
                        targetSeq.append(tId)
                        s += 1
                    }

                    var localGrads = self.bpttTrainer.makeGradients()
                    let loss = self.bpttTrainer.computeSampleGradients(
                        featuresSeq: featSeq,
                        targets: targetSeq,
                        grads: &localGrads
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
                        sumTotalLoss += l
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

        return EpochResult(
            epoch: epoch,
            totalLoss: sumTotalLoss / Float(max(1, validSampleCount))
        )
    }
}
