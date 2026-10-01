import Foundation
import MLX

/// 層ごとの電流と発火の統計 (診断用)
public struct LayerStatistics: Sendable {
    /// LIF に入る電流全体 (残差込み) の RMS
    public let currentRMS: Float
    /// この層で足した正規化済み結合電流 (層 0 は入力電流 + 再帰電流) の RMS
    public let ownCurrentRMS: Float
    /// サブステップあたりの発火率
    public let spikeRate: Float
    /// 膜電位が上限にかかっている割合
    public let clippedRate: Float
}

extension MLXBPTTTrainer {
    /// 発話ごとの特徴量 [T, inputDim] を流し、層ごとの電流 RMS・発火率・上限クリップ率を返す。
    /// `substep` と同じ式で状態を進めるが、勾配は取らない
    public func layerStatistics(network: MLXSpikingNetwork, features: MLXArray) -> [LayerStatistics] {
        let numLayers = network.numLayers
        let beta = network.lifConfig.beta
        let vTh = network.lifConfig.vTh
        let rho = network.lifConfig.rho
        let gamma = network.lifConfig.gamma
        let vMin = LIFNeuronEngine.vClampMin
        let vMax = LIFNeuronEngine.vClampMax
        let hMax = network.maxHiddenDim
        let seqLen = features.shape[0]
        let tSteps = network.timeSteps

        let currentSeq0 = matmul(features, network.wIn) + network.bH
        var v = [MLXArray](repeating: MLXArray.zeros([1, hMax]), count: numLayers)
        var s = [MLXArray](repeating: MLXArray.zeros([1, hMax]), count: numLayers)
        var a = [MLXArray](repeating: MLXArray.zeros([1, hMax]), count: numLayers)
        var sumSq = [Double](repeating: 0.0, count: numLayers)
        var ownSumSq = [Double](repeating: 0.0, count: numLayers)
        var spikes = [Double](repeating: 0.0, count: numLayers)
        var clipped = [Double](repeating: 0.0, count: numLayers)
        var samples: Double = 0.0

        var t = 0
        while t < seqLen {
            let current0 = currentSeq0[t].reshaped([1, hMax])
            var step = 0
            while step < tSteps {
                var stream = current0 + matmul(s[0], network.wRec)
                var l = 0
                while l < numLayers {
                    var own = stream
                    if 0 < l {
                        let upperIdx = l - 1
                        let denseCur = matmul(s[l - 1], network.wLayers[upperIdx]) + network.bHLayers[upperIdx]
                        let meanSq = mean(denseCur * denseCur, axis: -1, keepDims: true)
                        let rms = sqrt(meanSq + rmsNormEpsilon)
                        own = (denseCur / rms) * network.gammaRMS[upperIdx]
                        if upperIdx < network.wRecLayers.count {
                            own = own + matmul(s[l], network.wRecLayers[upperIdx])
                        }
                        stream = own + stream
                    }
                    var current = stream
                    if l < network.inputNormGains.count {
                        let streamMeanSq = mean(stream * stream, axis: -1, keepDims: true)
                        current = (stream / sqrt(streamMeanSq + inputNormEpsilon)) * network.inputNormGains[l]
                    }
                    var decayed = v[l] * beta
                    if l < network.betaLogits.count {
                        decayed = v[l] * sigmoid(network.betaLogits[l])
                    }
                    let isLast = (l + 1) == numLayers
                    var vRaw: MLXArray
                    if isLast {
                        vRaw = decayed + current
                    } else {
                        vRaw = decayed * (1.0 - s[l]) + current
                    }
                    v[l] = clip(vRaw, min: vMin, max: vMax)
                    a[l] = (a[l] * rho) + (s[l] * gamma)
                    let dynVTh = vTh + a[l]
                    let sHard = (dynVTh .<= v[l]).asType(.float32)
                    s[l] = sHard
                    if isLast {
                        v[l] = clip(v[l] - sHard * vTh, min: vMin, max: vMax)
                    }
                    let stats = stacked([
                        mean(current * current),
                        mean(own * own),
                        mean(sHard),
                        mean((MLXArray(vMax) .<= vRaw).asType(.float32))
                    ])
                    eval(stats)
                    let values = stats.asArray(Float.self)
                    sumSq[l] += Double(values[0])
                    ownSumSq[l] += Double(values[1])
                    spikes[l] += Double(values[2])
                    clipped[l] += Double(values[3])
                    l += 1
                }
                samples += 1.0
                step += 1
            }
            t += 1
        }

        var out: [LayerStatistics] = []
        var l = 0
        while l < numLayers {
            let n = max(1.0, samples)
            out.append(LayerStatistics(
                currentRMS: Float((sumSq[l] / n).squareRoot()),
                ownCurrentRMS: Float((ownSumSq[l] / n).squareRoot()),
                spikeRate: Float(spikes[l] / n),
                clippedRate: Float(clipped[l] / n)
            ))
            l += 1
        }
        return out
    }
}
