import Foundation

/// 第1段 音響推論用事前確保ワークスペース (0 アロケーション)
public final class AcousticWorkspace: @unchecked Sendable {
    public var vPrev: [Float]
    public var sPrev: [Float]
    public var aPrev: [Float]
    public var spikeSum: [Float]
    public var logits: [Float]
    public var probabilities: [Float]
    public var quantizedWorkspace: QuantizedWorkspace?
    /// forwardSlice の中間バッファ (Hot Path ゼロアロケーション用)
    public let scratch: MatryoshkaScratch

    public init(maxHiddenDim: Int = 4096, outputDim: Int = 523, inputDim: Int = 64) {
        self.scratch = MatryoshkaScratch(maxHiddenDim: maxHiddenDim)
        self.vPrev = [Float](repeating: 0.0, count: maxHiddenDim)
        self.sPrev = [Float](repeating: 0.0, count: maxHiddenDim)
        self.aPrev = [Float](repeating: 0.0, count: maxHiddenDim)
        self.spikeSum = [Float](repeating: 0.0, count: maxHiddenDim)
        self.logits = [Float](repeating: 0.0, count: outputDim)
        self.probabilities = [Float](repeating: 0.0, count: outputDim)
        self.quantizedWorkspace = QuantizedWorkspace(
            maxHiddenDim: maxHiddenDim,
            inputDim: inputDim,
            outputDim: outputDim
        )
    }

    @inline(__always)
    public func resetHiddenState() {
        var i = 0
        while i < vPrev.count {
            vPrev[i] = 0.0
            sPrev[i] = 0.0
            aPrev[i] = 0.0
            i += 1
        }
    }

    @inline(__always)
    public func reset() {
        var i = 0
        while i < vPrev.count {
            vPrev[i] = 0.0
            sPrev[i] = 0.0
            aPrev[i] = 0.0
            spikeSum[i] = 0.0
            i += 1
        }
        i = 0
        while i < logits.count {
            logits[i] = 0.0
            probabilities[i] = 0.0
            i += 1
        }
        quantizedWorkspace?.reset()
    }
}

/// 音響フレームごとの事後確率分布結果
public struct AcousticFrameProbabilities: Sendable, Equatable {
    public let frameIndex: Int
    public let topTokenId: Int
    public let topProbability: Float
    public let probabilities: [Float]

    public init(
        frameIndex: Int,
        topTokenId: Int,
        topProbability: Float,
        probabilities: [Float]
    ) {
        self.frameIndex = frameIndex
        self.topTokenId = topTokenId
        self.topProbability = topProbability
        self.probabilities = probabilities
    }
}

/// 第1段 音響 SNN デコーダ (64次元音響特徴量 -> 直接漢字かな文字事後確率分布)
public final class AcousticDecoder: @unchecked Sendable {
    public let network: MatryoshkaNetwork
    public let quantizedEngine: QuantizedEngine?
    public let vocabulary: TextVocabulary
    public let fallbackVocabulary: PhonemeVocabulary
    public let slice: MatryoshkaSlice
    public let silenceThreshold: Float

    public init(
        network: MatryoshkaNetwork,
        quantizedEngine: QuantizedEngine? = nil,
        vocabulary: TextVocabulary = TextVocabulary(),
        fallbackVocabulary: PhonemeVocabulary = PhonemeVocabulary(),
        slice: MatryoshkaSlice = .high,
        silenceThreshold: Float = 0.5
    ) {
        self.network = network
        self.quantizedEngine = quantizedEngine
        self.vocabulary = vocabulary
        self.fallbackVocabulary = fallbackVocabulary
        self.slice = slice
        self.silenceThreshold = silenceThreshold
    }

    /// 1フレームの音響特徴量から音素事後確率分布を推定
    @inline(__always)
    public func decodeFrame(
        features: [Float],
        workspace: AcousticWorkspace,
        frameIndex: Int = 0
    ) -> AcousticFrameProbabilities {
        switch quantizedEngine {
        case .some(let qEngine):
            let qWs: QuantizedWorkspace
            switch workspace.quantizedWorkspace {
            case .some(let ws):
                qWs = ws
            case .none:
                let newWs = QuantizedWorkspace(
                    maxHiddenDim: network.maxHiddenDim,
                    inputDim: network.inputDim,
                    outputDim: network.outputDim
                )
                workspace.quantizedWorkspace = newWs
                qWs = newWs
            }
            qEngine.predictSlice(
                features: features,
                slice: slice,
                workspace: qWs,
                outputProbs: &workspace.probabilities
            )
        case .none:
            network.forwardSlice(
                features: features,
                slice: slice,
                vPrev: &workspace.vPrev,
                sPrev: &workspace.sPrev,
                aPrev: &workspace.aPrev,
                spikeSum: &workspace.spikeSum,
                logits: &workspace.logits,
                probabilities: &workspace.probabilities,
                scratch: workspace.scratch
            )
        }

        // Argmax & Top Probability の探索
        var bestId = 0
        var maxP: Float = -1.0
        var c = 0
        let outDim = network.outputDim
        while c < outDim {
            let p = workspace.probabilities[c]
            if maxP < p {
                maxP = p
                bestId = c
            }
            c += 1
        }

        return AcousticFrameProbabilities(
            frameIndex: frameIndex,
            topTokenId: bestId,
            topProbability: maxP,
            probabilities: workspace.probabilities
        )
    }

    /// 特徴量シーケンスのデコード
    public func decodeSequence(
        featuresSeq: [[Float]],
        workspace: AcousticWorkspace,
        boundaries: [Int]? = nil
    ) -> [AcousticFrameProbabilities] {
        var results = [AcousticFrameProbabilities]()
        results.reserveCapacity(featuresSeq.count)
        var fIdx = 0
        while fIdx < featuresSeq.count {
            let probs = decodeFrame(
                features: featuresSeq[fIdx],
                workspace: workspace,
                frameIndex: fIdx
            )
            results.append(probs)
            fIdx += 1
        }
        return results
    }

    /// CTC 重複圧縮 (Collapse) による文字トークン系列の抽出
    public func collapseTokens(
        _ frameProbs: [AcousticFrameProbabilities],
        blankThreshold: Float = 0.4
    ) -> [Int] {
        var collapsed: [Int] = []
        var lastNonBlankToken: Int? = nil
        var lastTokenWasBlank = true

        var fIdx = 0
        while fIdx < frameProbs.count {
            let fp = frameProbs[fIdx]
            let topId = fp.topTokenId

            if topId == TextVocabulary.padId || topId < 4 {
                lastTokenWasBlank = true
            } else {
                var shouldAppend = false
                switch lastNonBlankToken {
                case .none:
                    shouldAppend = true
                case .some(let prev):
                    if prev != topId || lastTokenWasBlank {
                        shouldAppend = true
                    }
                }

                if shouldAppend {
                    collapsed.append(topId)
                    lastNonBlankToken = topId
                }
                lastTokenWasBlank = false
            }

            fIdx += 1
        }

        return collapsed
    }
}
