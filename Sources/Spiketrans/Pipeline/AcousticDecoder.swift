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
    public let scratch: ForwardScratch

    public init(maxHiddenDim: Int = 4096, outputDim: Int = 523, inputDim: Int = 64, numLayers: Int = 1) {
        let actualLayers = max(1, numLayers)
        let totalHidden = actualLayers * maxHiddenDim
        self.scratch = ForwardScratch(maxHiddenDim: maxHiddenDim)
        self.vPrev = [Float](repeating: 0.0, count: totalHidden)
        self.sPrev = [Float](repeating: 0.0, count: totalHidden)
        self.aPrev = [Float](repeating: 0.0, count: totalHidden)
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
            i += 1
        }
        i = 0
        while i < spikeSum.count {
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
    public let network: SpikingNetwork
    public let convSubsampling: Conv2DSubsampling?
    public let quantizedEngine: QuantizedEngine?
    public let vocabulary: TextVocabulary
    public let fallbackVocabulary: PhonemeVocabulary
    public let silenceThreshold: Float

    public init(
        network: SpikingNetwork,
        convSubsampling: Conv2DSubsampling? = nil,
        quantizedEngine: QuantizedEngine? = nil,
        vocabulary: TextVocabulary = TextVocabulary(),
        fallbackVocabulary: PhonemeVocabulary = PhonemeVocabulary(),
        silenceThreshold: Float = 0.5
    ) {
        self.network = network
        switch convSubsampling {
        case .some(let cs):
            self.convSubsampling = cs
        case .none:
            self.convSubsampling = network.convSubsampling
        }
        self.quantizedEngine = quantizedEngine
        self.vocabulary = vocabulary
        self.fallbackVocabulary = fallbackVocabulary
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
            qEngine.predict(
                features: features,
                workspace: qWs,
                outputProbs: &workspace.probabilities
            )
        case .none:
            network.forward(
                features: features,
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

    /// ストリーミング推論用: 1フレームの Mel 特徴量を供給し、Conv2D Subsampling 経由で
    /// 4フレーム蓄積ごとに 1 回 SNN 音響推論を実行 (因果的・未来フレーム非参照)
    public func decodeStreaming(
        melFrame: [Float],
        state: inout Conv2DStreamingState,
        workspace: AcousticWorkspace,
        frameIndex: Int = 0
    ) -> AcousticFrameProbabilities? {
        switch convSubsampling {
        case .some(let subsampler):
            let res = subsampler.forwardStreaming(melFrame: melFrame, state: &state)
            switch res {
            case .some(let subFeat):
                return decodeFrame(features: subFeat, workspace: workspace, frameIndex: frameIndex)
            case .none:
                return nil
            }
        case .none:
            return decodeFrame(features: melFrame, workspace: workspace, frameIndex: frameIndex)
        }
    }

    /// 特徴量シーケンスのデコード (Mel 特徴量時は convSubsampling を自動適用)
    public func decodeSequence(
        featuresSeq: [[Float]],
        workspace: AcousticWorkspace,
        boundaries: [Int]? = nil
    ) -> [AcousticFrameProbabilities] {
        let inputSeq: [[Float]]
        let effectiveBoundaries: [Int]?
        switch convSubsampling {
        case .some(let subsampler):
            if featuresSeq.first?.count == subsampler.melChannels {
                inputSeq = subsampler.forward(melSpectrogram: featuresSeq)
                switch boundaries {
                case .some(let b):
                    effectiveBoundaries = FormantSegmenter.subsampleBoundaries(boundaries: b, factor: 4)
                case .none:
                    effectiveBoundaries = nil
                }
            } else {
                inputSeq = featuresSeq
                effectiveBoundaries = boundaries
            }
        case .none:
            inputSeq = featuresSeq
            effectiveBoundaries = boundaries
        }

        var results = [AcousticFrameProbabilities]()
        results.reserveCapacity(inputSeq.count)
        var fIdx = 0
        while fIdx < inputSeq.count {
            let probs = decodeFrame(
                features: inputSeq[fIdx],
                workspace: workspace,
                frameIndex: fIdx
            )
            results.append(probs)
            fIdx += 1
        }
        return results
    }

    /// 64ch Mel スペクトログラムから 2D-Conv Subsampling を経由して直接デコード
    public func decodeMelSequence(
        melFeaturesSeq: [[Float]],
        workspace: AcousticWorkspace,
        boundaries: [Int]? = nil
    ) -> [AcousticFrameProbabilities] {
        let inputSeq: [[Float]]
        let effectiveBoundaries: [Int]?
        switch convSubsampling {
        case .some(let subsampler):
            inputSeq = subsampler.forward(melSpectrogram: melFeaturesSeq)
            switch boundaries {
            case .some(let b):
                effectiveBoundaries = FormantSegmenter.subsampleBoundaries(boundaries: b, factor: 4)
            case .none:
                effectiveBoundaries = nil
            }
        case .none:
            inputSeq = melFeaturesSeq
            effectiveBoundaries = boundaries
        }

        var results = [AcousticFrameProbabilities]()
        results.reserveCapacity(inputSeq.count)
        var fIdx = 0
        while fIdx < inputSeq.count {
            let probs = decodeFrame(
                features: inputSeq[fIdx],
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
