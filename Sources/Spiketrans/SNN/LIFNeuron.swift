import Foundation

/// LIF (Leaky Integrate-and-Fire) ニューロン設定パラメータ
public struct LIFConfig: Sendable, Equatable {
    public let beta: Float      // 膜電位減衰率 (0.0 < beta < 1.0)
    public let vTh: Float       // 発火閾値 (通常 1.0)
    public let vReset: Float    // リセット電位 (通常 0.0)
    public let alpha: Float     // Surrogate Gradient 鋭さパラメータ (通常 2.0)

    public init(
        beta: Float = 0.8,
        vTh: Float = 1.0,
        vReset: Float = 0.0,
        alpha: Float = 2.0
    ) {
        self.beta = beta
        self.vTh = vTh
        self.vReset = vReset
        self.alpha = alpha
    }
}

/// 膜電位とスパイク状態を保持するコンテナ (Hot Path ゼロアロケーション用)
public final class LIFState: @unchecked Sendable {
    public var v: ContiguousArray<Float>
    public var s: ContiguousArray<Float>
    public let size: Int

    public init(size: Int) {
        self.size = size
        self.v = ContiguousArray<Float>(repeating: 0.0, count: size)
        self.s = ContiguousArray<Float>(repeating: 0.0, count: size)
    }

    @inline(__always)
    public func reset() {
        var i = 0
        while i < size {
            v[i] = 0.0
            s[i] = 0.0
            i += 1
        }
    }
}

/// SIMD8 ベクトル化 LIF 膜電位更新エンジン
public enum LIFNeuronEngine {
    /// 1 ニューロンのスカラー更新ステップ
    @inline(__always)
    public static func stepScalar(
        config: LIFConfig,
        vPrev: Float,
        sPrev: Float,
        inputCurrent: Float
    ) -> (vNext: Float, sNext: Float) {
        let vDecayed = config.beta * vPrev * (1.0 - sPrev)
        let vNext = vDecayed + inputCurrent
        var sNext: Float = 0.0
        if config.vTh <= vNext {
            sNext = 1.0
        }
        return (vNext, sNext)
    }

    /// SIMD8 による 8 ニューロン一括更新ステップ
    @inline(__always)
    public static func stepSIMD8(
        config: LIFConfig,
        vPrevPtr: UnsafePointer<Float>,
        sPrevPtr: UnsafePointer<Float>,
        inputPtr: UnsafePointer<Float>,
        vNextPtr: UnsafeMutablePointer<Float>,
        sNextPtr: UnsafeMutablePointer<Float>,
        count: Int
    ) {
        let width = 8
        let limit = count - (count % width)
        let betaVec = SIMD8<Float>(repeating: config.beta)
        let oneVec = SIMD8<Float>(repeating: 1.0)
        let vThVec = SIMD8<Float>(repeating: config.vTh)
        var i = 0

        while i < limit {
            let va = SIMD8<Float>(
                vPrevPtr[i+0], vPrevPtr[i+1], vPrevPtr[i+2], vPrevPtr[i+3],
                vPrevPtr[i+4], vPrevPtr[i+5], vPrevPtr[i+6], vPrevPtr[i+7]
            )
            let sa = SIMD8<Float>(
                sPrevPtr[i+0], sPrevPtr[i+1], sPrevPtr[i+2], sPrevPtr[i+3],
                sPrevPtr[i+4], sPrevPtr[i+5], sPrevPtr[i+6], sPrevPtr[i+7]
            )
            let ia = SIMD8<Float>(
                inputPtr[i+0], inputPtr[i+1], inputPtr[i+2], inputPtr[i+3],
                inputPtr[i+4], inputPtr[i+5], inputPtr[i+6], inputPtr[i+7]
            )

            let vDecayed = betaVec * va * (oneVec - sa)
            let vNext = vDecayed + ia
            let mask = vThVec .<= vNext

            vNextPtr[i+0] = vNext[0]
            vNextPtr[i+1] = vNext[1]
            vNextPtr[i+2] = vNext[2]
            vNextPtr[i+3] = vNext[3]
            vNextPtr[i+4] = vNext[4]
            vNextPtr[i+5] = vNext[5]
            vNextPtr[i+6] = vNext[6]
            vNextPtr[i+7] = vNext[7]

            var s0: Float = 0.0
            var s1: Float = 0.0
            var s2: Float = 0.0
            var s3: Float = 0.0
            var s4: Float = 0.0
            var s5: Float = 0.0
            var s6: Float = 0.0
            var s7: Float = 0.0

            if mask[0] { s0 = 1.0 }
            if mask[1] { s1 = 1.0 }
            if mask[2] { s2 = 1.0 }
            if mask[3] { s3 = 1.0 }
            if mask[4] { s4 = 1.0 }
            if mask[5] { s5 = 1.0 }
            if mask[6] { s6 = 1.0 }
            if mask[7] { s7 = 1.0 }

            sNextPtr[i+0] = s0
            sNextPtr[i+1] = s1
            sNextPtr[i+2] = s2
            sNextPtr[i+3] = s3
            sNextPtr[i+4] = s4
            sNextPtr[i+5] = s5
            sNextPtr[i+6] = s6
            sNextPtr[i+7] = s7

            i += width
        }

        while i < count {
            let res = stepScalar(
                config: config,
                vPrev: vPrevPtr[i],
                sPrev: sPrevPtr[i],
                inputCurrent: inputPtr[i]
            )
            vNextPtr[i] = res.vNext
            sNextPtr[i] = res.sNext
            i += 1
        }
    }
}
