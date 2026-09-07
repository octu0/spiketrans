import Foundation

/// 音響フォワードの膜電位・スパイク・適応閾値と出力。層 l は `[l * maxHiddenDim, (l + 1) * maxHiddenDim)`。
public final class AcousticWorkspace: @unchecked Sendable {
    public var vPrev: [Float]
    public var sPrev: [Float]
    public var aPrev: [Float]
    public var spikeSum: [Float]
    public var logits: [Float]
    public var probabilities: [Float]
    public var quantizedWorkspace: QuantizedWorkspace?
    /// `forward` が毎フレーム確保しない中間バッファ
    public let scratch: ForwardScratch

    public init(maxHiddenDim: Int = 4096, outputDim: Int = 523, inputDim: Int = 64, numLayers: Int = 1) {
        // 膜電位・スパイク・適応閾値は層ごとに持つ (層 l は [l * maxHiddenDim, (l + 1) * maxHiddenDim))
        let stateSize = max(1, numLayers) * maxHiddenDim
        self.scratch = ForwardScratch(maxHiddenDim: maxHiddenDim)
        self.vPrev = [Float](repeating: 0.0, count: stateSize)
        self.sPrev = [Float](repeating: 0.0, count: stateSize)
        self.aPrev = [Float](repeating: 0.0, count: stateSize)
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
        resetHiddenState()
        var i = 0
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

/// 第1段 音響 SNN。1 フレームの特徴から文字事後確率を出す
public final class AcousticDecoder: @unchecked Sendable {
    public let network: SpikingNetwork
    public let quantizedEngine: QuantizedEngine?

    public init(
        network: SpikingNetwork,
        quantizedEngine: QuantizedEngine? = nil
    ) {
        self.network = network
        self.quantizedEngine = quantizedEngine
    }

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

    public func decodeSequence(
        featuresSeq: [[Float]],
        workspace: AcousticWorkspace
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

    /// 連続同一トークンと pad / 制御トークン (id < 4) を潰す。
    public func collapseTokens(
        _ frameProbs: [AcousticFrameProbabilities]
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
