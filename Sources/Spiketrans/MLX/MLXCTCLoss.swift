import Foundation
import MLX
import MLXNN

/// MLX 演算で構成した CTC 損失。
///
/// 対数空間の前向き再帰 (alpha) だけを微分可能な演算で組み立てる。
/// 後ろ向き再帰は自動微分が導出するため実装しない。
/// これによりロジットを CPU へ取り出す必要がなくなり、
/// フォワードを 1 回に減らせる。
public enum MLXCTCLoss {
    /// -inf の代替。logAddExp で inf - inf の不定形を避けるため有限の大きな負値を使う
    static let negativeInfinity: Float = -1e30

    /// バッチのラベル列を CTC の拡張ラベル列に変換する
    ///
    /// 拡張ラベル列は blank と実ラベルを交互に並べたもので長さ 2U+1 となる。
    public struct ExtendedTargets {
        public let extTargets: MLXArray     // [B, Lmax] Int32
        public let skipMask: MLXArray       // [B, Lmax] 1.0 でスキップ遷移可
        public let validMask: MLXArray      // [B, Lmax] 1.0 で有効な位置
        public let finalIndex1: MLXArray    // [B] Int32  L_b - 1
        public let finalIndex2: MLXArray    // [B] Int32  L_b - 2 (下限 0)
        public let hasSecondFinal: MLXArray // [B] 1.0 で L_b >= 2
        public let inputLengths: MLXArray   // [B] Int32
        public let maxExtendedLength: Int

        public init(
            targetsBatch: [[Int]],
            frameCounts: [Int],
            blankId: Int
        ) {
            let batchSize = targetsBatch.count
            var extendedLengths = [Int](repeating: 0, count: batchSize)
            var maxLen = 1
            var b = 0
            while b < batchSize {
                let u = targetsBatch[b].count
                extendedLengths[b] = (u * 2) + 1
                if maxLen < extendedLengths[b] {
                    maxLen = extendedLengths[b]
                }
                b += 1
            }
            // 拡張ラベル長も 8 の倍数へ切り上げて形状の種類を減らす。
            // 超過分は validMask = 0 なので損失には影響しない
            maxLen = ((maxLen + 7) / 8) * 8
            self.maxExtendedLength = maxLen

            var flatExt = [Int32](repeating: Int32(blankId), count: batchSize * maxLen)
            var flatSkip = [Float](repeating: 0.0, count: batchSize * maxLen)
            var flatValid = [Float](repeating: 0.0, count: batchSize * maxLen)
            var idx1 = [Int32](repeating: 0, count: batchSize)
            var idx2 = [Int32](repeating: 0, count: batchSize)
            var hasSecond = [Float](repeating: 0.0, count: batchSize)
            var lengths = [Int32](repeating: 0, count: batchSize)

            b = 0
            while b < batchSize {
                let labels = targetsBatch[b]
                let lLen = extendedLengths[b]
                let base = b * maxLen

                // 拡張ラベル列: blank, l0, blank, l1, ... , blank
                var u = 0
                while u < labels.count {
                    flatExt[base + (u * 2) + 1] = Int32(labels[u])
                    u += 1
                }

                var sPos = 0
                while sPos < lLen {
                    flatValid[base + sPos] = 1.0
                    // s-2 からのスキップ遷移は、s が実ラベルで
                    // かつ s-2 の実ラベルと異なるときだけ許される
                    if 2 <= sPos {
                        let cur = flatExt[base + sPos]
                        let prev2 = flatExt[base + sPos - 2]
                        if cur != Int32(blankId) && cur != prev2 {
                            flatSkip[base + sPos] = 1.0
                        }
                    }
                    sPos += 1
                }

                idx1[b] = Int32(lLen - 1)
                if 2 <= lLen {
                    idx2[b] = Int32(lLen - 2)
                    hasSecond[b] = 1.0
                }
                lengths[b] = Int32(frameCounts[b])
                b += 1
            }

            self.extTargets = MLXArray(flatExt, [batchSize, maxLen])
            self.skipMask = MLXArray(flatSkip, [batchSize, maxLen])
            self.validMask = MLXArray(flatValid, [batchSize, maxLen])
            self.finalIndex1 = MLXArray(idx1, [batchSize])
            self.finalIndex2 = MLXArray(idx2, [batchSize])
            self.hasSecondFinal = MLXArray(hasSecond, [batchSize])
            self.inputLengths = MLXArray(lengths, [batchSize])
        }
    }

    /// alpha を s 方向に k だけずらす。空いた先頭は -inf 相当で埋める。
    static func shiftedRight(_ alpha: MLXArray, by k: Int) -> MLXArray {
        let batchSize = alpha.shape[0]
        let length = alpha.shape[1]
        if length <= k {
            return full([batchSize, length], values: MLXArray(negativeInfinity), type: Float.self)
        }
        let head = full([batchSize, k], values: MLXArray(negativeInfinity), type: Float.self)
        return concatenated([head, alpha[0..., 0..<(length - k)]], axis: 1)
    }

    /// バッチ平均した CTC 損失を返す
    ///
    /// - Parameters:
    ///   - logits: [B, T, V]
    ///   - targets: 拡張済みラベル情報
    public static func loss(
        logits: MLXArray,
        targets: ExtendedTargets
    ) -> MLXArray {
        let seqLen = logits.shape[1]
        let logProbs = logSoftmax(logits, axis: -1)

        // 各時刻の拡張ラベル位置における対数確率 [B, Lmax]
        func emission(at t: Int) -> MLXArray {
            let step = logProbs[0..., t, 0...]
            return takeAlong(step, targets.extTargets, axis: 1)
        }

        let invalid = targets.validMask .<= 0.0

        // t = 0: s = 0 と s = 1 のみ到達可能
        var alpha = emission(at: 0)
        let lengthRange = MLXArray(Array(0..<targets.maxExtendedLength).map { Int32($0) },
                                   [1, targets.maxExtendedLength])
        let startReachable = lengthRange .<= MLXArray(Int32(1))
        alpha = which(startReachable, alpha, MLXArray(negativeInfinity))
        alpha = which(invalid, MLXArray(negativeInfinity), alpha)

        var t = 1
        while t < seqLen {
            let stay = alpha
            let fromPrev = shiftedRight(alpha, by: 1)
            let skipRaw = shiftedRight(alpha, by: 2)
            let skip = which(targets.skipMask .> 0.0, skipRaw, MLXArray(negativeInfinity))

            var next = logAddExp(logAddExp(stay, fromPrev), skip) + emission(at: t)
            next = which(invalid, MLXArray(negativeInfinity), next)

            // 自身の系列長を超えたフレームでは alpha を凍結する
            let active = MLXArray(Int32(t)) .< targets.inputLengths
            alpha = which(active.expandedDimensions(axis: 1), next, alpha)
            t += 1
        }

        // 終端は最後の blank と最後の実ラベルの 2 経路
        let last1 = takeAlong(alpha, targets.finalIndex1.expandedDimensions(axis: 1), axis: 1)
            .squeezed(axis: 1)
        let last2Raw = takeAlong(alpha, targets.finalIndex2.expandedDimensions(axis: 1), axis: 1)
            .squeezed(axis: 1)
        let last2 = which(targets.hasSecondFinal .> 0.0, last2Raw, MLXArray(negativeInfinity))

        let logLikelihood = logAddExp(last1, last2)
        return -mean(logLikelihood)
    }
}
