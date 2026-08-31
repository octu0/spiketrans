import Foundation

/// Fast Sigmoid 代理勾配計算モジュール
public enum SurrogateGradient {
    /// スカラー代理勾配計算: dS/dV = 1 / (1 + alpha * |V - Vth|)^2
    @inline(__always)
    public static func derivative(v: Float, vTh: Float, alpha: Float) -> Float {
        let diff = abs(v - vTh)
        let denom = 1.0 + alpha * diff
        return 1.0 / (denom * denom)
    }

    /// SIMD8 による 8 レーン代理勾配一括計算
    @inline(__always)
    public static func derivativeSIMD8(
        vPtr: UnsafePointer<Float>,
        dstPtr: UnsafeMutablePointer<Float>,
        count: Int,
        vTh: Float,
        alpha: Float
    ) {
        let width = 8
        let limit = count - (count % width)
        let vThVec = SIMD8<Float>(repeating: vTh)
        let alphaVec = SIMD8<Float>(repeating: alpha)
        let oneVec = SIMD8<Float>(repeating: 1.0)
        let zeroVec = SIMD8<Float>(repeating: 0.0)
        var i = 0

        while i < limit {
            let v = SIMD8<Float>(
                vPtr[i+0], vPtr[i+1], vPtr[i+2], vPtr[i+3],
                vPtr[i+4], vPtr[i+5], vPtr[i+6], vPtr[i+7]
            )
            let diff = v - vThVec
            let absDiff = diff.replacing(with: -diff, where: diff .< zeroVec)
            let denom = oneVec + alphaVec * absDiff
            let deriv = oneVec / (denom * denom)

            dstPtr[i+0] = deriv[0]
            dstPtr[i+1] = deriv[1]
            dstPtr[i+2] = deriv[2]
            dstPtr[i+3] = deriv[3]
            dstPtr[i+4] = deriv[4]
            dstPtr[i+5] = deriv[5]
            dstPtr[i+6] = deriv[6]
            dstPtr[i+7] = deriv[7]

            i += width
        }

        while i < count {
            dstPtr[i] = derivative(v: vPtr[i], vTh: vTh, alpha: alpha)
            i += 1
        }
    }
}
