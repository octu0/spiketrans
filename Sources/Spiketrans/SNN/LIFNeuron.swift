import Foundation

/// LIF (Leaky Integrate-and-Fire) ニューロン設定パラメータ
public struct LIFConfig: Sendable, Equatable {
    public let beta: Float      // 膜電位減衰率 (0.0 < beta < 1.0)
    public let vTh: Float       // 基本発火閾値 (通常 1.0)
    public let vReset: Float    // リセット電位 (通常 0.0)
    public let alpha: Float     // Surrogate Gradient 鋭さパラメータ (通常 2.0)
    public let rho: Float       // 適応閾値減衰率 (通常 0.85)
    public let gamma: Float     // 発火時閾値上昇幅 (0.0 で標準固定閾値, >0.0 で適応型 ALIF)

    public init(
        beta: Float = 0.8,
        vTh: Float = 1.0,
        vReset: Float = 0.0,
        alpha: Float = 2.0,
        rho: Float = 0.85,
        gamma: Float = 0.0
    ) {
        self.beta = beta
        self.vTh = vTh
        self.vReset = vReset
        self.alpha = alpha
        self.rho = rho
        self.gamma = gamma
    }
}

/// 膜電位とスパイク状態を保持するコンテナ (Hot Path ゼロアロケーション用)
public final class LIFState: @unchecked Sendable {
    public var v: ContiguousArray<Float>
    public var s: ContiguousArray<Float>
    public var a: ContiguousArray<Float>
    public let size: Int

    public init(size: Int) {
        self.size = size
        self.v = ContiguousArray<Float>(repeating: 0.0, count: size)
        self.s = ContiguousArray<Float>(repeating: 0.0, count: size)
        self.a = ContiguousArray<Float>(repeating: 0.0, count: size)
    }

    @inline(__always)
    public func reset() {
        var i = 0
        while i < size {
            v[i] = 0.0
            s[i] = 0.0
            a[i] = 0.0
            i += 1
        }
    }
}

/// SIMD8 ベクトル化 LIF 膜電位更新エンジン
public enum LIFNeuronEngine {
    /// 膜電位の飽和範囲。学習側 (MLX) が同じ範囲でクリップしているため、
    /// 推論側でも揃えないと発火パターンが学習時と食い違う。
    public static let vClampMin: Float = -20.0
    public static let vClampMax: Float = 20.0

    @inline(__always)
    public static func clampMembrane(_ v: Float) -> Float {
        if v < vClampMin {
            return vClampMin
        }
        if vClampMax < v {
            return vClampMax
        }
        return v
    }
    /// 1 ニューロンのスカラー更新ステップ
    @inline(__always)
    public static func stepScalar(
        config: LIFConfig,
        vPrev: Float,
        sPrev: Float,
        inputCurrent: Float
    ) -> (vNext: Float, sNext: Float) {
        let vDecayed = config.beta * vPrev * (1.0 - sPrev)
        let vNext = clampMembrane(vDecayed + inputCurrent)
        var sNext: Float = 0.0
        if config.vTh <= vNext {
            sNext = 1.0
        }
        return (vNext: vNext, sNext: sNext)
    }

    /// 1 ニューロンの適応型発火閾値 (ALIF) スカラー更新ステップ
    @inline(__always)
    public static func stepScalarAdaptive(
        config: LIFConfig,
        vPrev: Float,
        sPrev: Float,
        aPrev: Float,
        inputCurrent: Float
    ) -> (vNext: Float, sNext: Float, aNext: Float) {
        let vDecayed = config.beta * vPrev * (1.0 - sPrev)
        let vNext = clampMembrane(vDecayed + inputCurrent)
        let aNext = (config.rho * aPrev) + (config.gamma * sPrev)
        let dynVTh = config.vTh + aNext
        var sNext: Float = 0.0
        if dynVTh <= vNext {
            sNext = 1.0
        }
        return (vNext: vNext, sNext: sNext, aNext: aNext)
    }

    /// ALIF 状態を SIMD8 で一括更新する。
    ///
    /// 状態配列 (v, s, a) をポインタ経由で直接更新する。層ごとの状態は 1 本の配列の
    /// 区間なので、呼び出し側が先頭ポインタをずらして層を選ぶ。spikeSumPtr が nil の
    /// 層は積算しない (リードアウトに使うのは最終層だけ)。
    /// 演算はニューロンごとに独立なので、スカラー版と結果はビット一致する。
    @inline(__always)
    public static func stepAdaptiveSIMD8(
        config: LIFConfig,
        vPtr: UnsafeMutablePointer<Float>,
        sPtr: UnsafeMutablePointer<Float>,
        aPtr: UnsafeMutablePointer<Float>,
        curPtr: UnsafePointer<Float>,
        spikeSumPtr: UnsafeMutablePointer<Float>?,
        count: Int
    ) {
        let limit = count - (count % 8)
        let betaVec = SIMD8<Float>(repeating: config.beta)
        let oneVec = SIMD8<Float>(repeating: 1.0)
        let rhoVec = SIMD8<Float>(repeating: config.rho)
        let gammaVec = SIMD8<Float>(repeating: config.gamma)
        let vThVec = SIMD8<Float>(repeating: config.vTh)
        let lowVec = SIMD8<Float>(repeating: vClampMin)
        let highVec = SIMD8<Float>(repeating: vClampMax)
        let zeroVec = SIMD8<Float>(repeating: 0.0)

        var i = 0
        while i < limit {
            let vPrev = SIMD8<Float>(
                vPtr[i+0], vPtr[i+1], vPtr[i+2], vPtr[i+3],
                vPtr[i+4], vPtr[i+5], vPtr[i+6], vPtr[i+7]
            )
            let sPrev = SIMD8<Float>(
                sPtr[i+0], sPtr[i+1], sPtr[i+2], sPtr[i+3],
                sPtr[i+4], sPtr[i+5], sPtr[i+6], sPtr[i+7]
            )
            let aPrev = SIMD8<Float>(
                aPtr[i+0], aPtr[i+1], aPtr[i+2], aPtr[i+3],
                aPtr[i+4], aPtr[i+5], aPtr[i+6], aPtr[i+7]
            )
            let inCur = SIMD8<Float>(
                curPtr[i+0], curPtr[i+1], curPtr[i+2], curPtr[i+3],
                curPtr[i+4], curPtr[i+5], curPtr[i+6], curPtr[i+7]
            )

            let vDecayed = betaVec * vPrev * (oneVec - sPrev)
            let vRaw = vDecayed + inCur
            var vNext = vRaw.replacing(with: lowVec, where: vRaw .< lowVec)
            vNext = vNext.replacing(with: highVec, where: highVec .< vNext)
            let aNext = (rhoVec * aPrev) + (gammaVec * sPrev)
            let dynVTh = vThVec + aNext
            let sNext = zeroVec.replacing(with: oneVec, where: dynVTh .<= vNext)

            switch spikeSumPtr {
            case .some(let sumPtr):
                let sumPrev = SIMD8<Float>(
                    sumPtr[i+0], sumPtr[i+1], sumPtr[i+2], sumPtr[i+3],
                    sumPtr[i+4], sumPtr[i+5], sumPtr[i+6], sumPtr[i+7]
                )
                let sumNext = sumPrev + sNext
                var lane = 0
                while lane < 8 {
                    sumPtr[i+lane] = sumNext[lane]
                    lane += 1
                }
            case .none:
                break
            }

            var lane = 0
            while lane < 8 {
                vPtr[i+lane] = vNext[lane]
                sPtr[i+lane] = sNext[lane]
                aPtr[i+lane] = aNext[lane]
                lane += 1
            }
            i += 8
        }
        while i < count {
            let res = stepScalarAdaptive(
                config: config,
                vPrev: vPtr[i],
                sPrev: sPtr[i],
                aPrev: aPtr[i],
                inputCurrent: curPtr[i]
            )
            vPtr[i] = res.vNext
            sPtr[i] = res.sNext
            aPtr[i] = res.aNext
            switch spikeSumPtr {
            case .some(let sumPtr):
                sumPtr[i] += res.sNext
            case .none:
                break
            }
            i += 1
        }
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
            // スカラー版と同じ飽和範囲に揃える
            let vRaw = vDecayed + ia
            // clamped() は NaN を境界値に潰してしまうため、比較で置換して
            // スカラー版と同じく NaN をそのまま通す
            let lowVec = SIMD8<Float>(repeating: vClampMin)
            let highVec = SIMD8<Float>(repeating: vClampMax)
            var vNext = vRaw.replacing(with: lowVec, where: vRaw .< lowVec)
            vNext = vNext.replacing(with: highVec, where: highVec .< vNext)
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
