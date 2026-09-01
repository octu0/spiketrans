import Foundation
import Numerics

/// Pure Swift / SIMD による対数空間安定 CTC (Connectionist Temporal Classification) 損失計算器
public struct CTCLossCalculator: Sendable {
    public let blankId: Int

    public init(blankId: Int = 0) {
        self.blankId = blankId
    }

    /// 対数加算 (Log-Sum-Exp: log(exp(a) + exp(b)))
    @inline(__always)
    public static func logAdd(_ a: Float, _ b: Float) -> Float {
        if a == -.infinity { return b }
        if b == -.infinity { return a }
        if a < b {
            return b + log(1.0 + exp(a - b))
        }
        return a + log(1.0 + exp(b - a))
    }

    /// 対数確率系列 (T x V) とターゲット系列 (U) から CTC 損失および Softmax 前勾配 (T x V) を計算
    public func computeLossAndGradients(
        logProbs: [[Float]],       // フレームごとの対数確率 (T x V)
        targets: [Int]             // ターゲット文字 ID 列 (U)
    ) -> (loss: Float, gradients: [[Float]]) {
        let tCount = logProbs.count
        let uCount = targets.count

        if tCount <= 0 || logProbs[0].isEmpty {
            return (loss: 0.0, gradients: [])
        }

        let vCount = logProbs[0].count
        if uCount <= 0 {
            // ターゲットが空の場合は全フレーム Blank
            var loss: Float = 0.0
            var grads = [[Float]](repeating: [Float](repeating: 0.0, count: vCount), count: tCount)
            var t = 0
            while t < tCount {
                let lp = logProbs[t]
                loss -= lp[blankId]
                var v = 0
                while v < vCount {
                    grads[t][v] = exp(lp[v])
                    v += 1
                }
                grads[t][blankId] -= 1.0
                t += 1
            }
            return (loss: loss, gradients: grads)
        }

        // 1. 拡張ターゲット系列 l' の作成 (長さ L = 2U + 1)
        let lLen = (uCount * 2) + 1
        var extendedTargets = [Int](repeating: blankId, count: lLen)
        var u = 0
        while u < uCount {
            extendedTargets[(u * 2) + 1] = targets[u]
            u += 1
        }

        if tCount < uCount {
            // フレーム数がターゲット文字数より短い場合はアライメント不能
            return (loss: Float(uCount * 10), gradients: [[Float]](repeating: [Float](repeating: 0.0, count: vCount), count: tCount))
        }

        // 2. 対数空間 Forward 確率 alpha (T x L)
        var alpha = [[Float]](repeating: [Float](repeating: -.infinity, count: lLen), count: tCount)
        alpha[0][0] = logProbs[0][blankId]
        if 1 < lLen {
            alpha[0][1] = logProbs[0][extendedTargets[1]]
        }

        var t = 1
        while t < tCount {
            let lp = logProbs[t]
            var s = 0
            while s < lLen {
                let targetId = extendedTargets[s]
                let logP = lp[targetId]

                var logSum = alpha[t - 1][s]
                if 0 < s {
                    logSum = Self.logAdd(logSum, alpha[t - 1][s - 1])
                }
                if 1 < s {
                    let prevTarget = extendedTargets[s - 2]
                    if targetId != blankId && targetId != prevTarget {
                        logSum = Self.logAdd(logSum, alpha[t - 1][s - 2])
                    }
                }

                if logSum != -.infinity {
                    alpha[t][s] = logSum + logP
                }
                s += 1
            }
            t += 1
        }

        // 全確率 P(Y|X) の対数
        var term2: Float = -.infinity
        if 1 < lLen {
            term2 = alpha[tCount - 1][lLen - 2]
        }
        let logTotalProb = Self.logAdd(alpha[tCount - 1][lLen - 1], term2)
        let loss = -logTotalProb

        if logTotalProb == -.infinity || loss.isNaN {
            return (loss: Float(uCount * 5), gradients: [[Float]](repeating: [Float](repeating: 0.0, count: vCount), count: tCount))
        }

        // 3. 対数空間 Backward 確率 beta (T x L)
        var beta = [[Float]](repeating: [Float](repeating: -.infinity, count: lLen), count: tCount)
        beta[tCount - 1][lLen - 1] = 0.0
        if 1 < lLen {
            beta[tCount - 1][lLen - 2] = 0.0
        }

        t = tCount - 2
        while 0 <= t {
            let nextLp = logProbs[t + 1]
            var s = 0
            while s < lLen {
                var logSum = beta[t + 1][s] + nextLp[extendedTargets[s]]
                if (s + 1) < lLen {
                    logSum = Self.logAdd(logSum, beta[t + 1][s + 1] + nextLp[extendedTargets[s + 1]])
                }
                if (s + 2) < lLen {
                    let currTarget = extendedTargets[s]
                    let nextNextTarget = extendedTargets[s + 2]
                    if currTarget != blankId && currTarget != nextNextTarget {
                        logSum = Self.logAdd(logSum, beta[t + 1][s + 2] + nextLp[nextNextTarget])
                    }
                }

                beta[t][s] = logSum
                t_loop_condition: if t < 0 { break }
                s += 1
            }
            t -= 1
        }

        // 4. Softmax 勾配の計算: dL/dLogit_t,k = exp(logProbs[t][k]) - gamma_t(k)
        var gradients = [[Float]](repeating: [Float](repeating: 0.0, count: vCount), count: tCount)
        t = 0
        while t < tCount {
            let lp = logProbs[t]
            // 各クラス k の事後確率 gamma_t(k)
            var logGamma = [Float](repeating: -.infinity, count: vCount)
            var s = 0
            while s < lLen {
                let k = extendedTargets[s]
                let ab = alpha[t][s] + beta[t][s]
                logGamma[k] = Self.logAdd(logGamma[k], ab)
                s += 1
            }

            var k = 0
            while k < vCount {
                let prob = exp(lp[k])
                var postProb: Float = 0.0
                if logGamma[k] != -.infinity {
                    postProb = exp(logGamma[k] - logTotalProb)
                }
                // CTC 損失の Softmax 勾配: P(k) - PostP(k)
                gradients[t][k] = prob - postProb
                k += 1
            }
            t += 1
        }

        return (loss: loss, gradients: gradients)
    }
}
