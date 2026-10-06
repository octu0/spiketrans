import Foundation

/// SIMD8 の超越関数。推論のゲート計算で 1 要素ずつ expf を呼ばないためのもの
public enum SIMDMath {
    /// exp(x)。2^n · e^r に分け (|r| ≤ ln2/2)、e^r を 6 次の多項式で近似する。相対誤差は約 2e-7。
    /// x は [-87, 88] に丸める (float の範囲外で無限大・非正規化数にしない)
    @inline(__always)
    public static func exp(_ x: SIMD8<Float>) -> SIMD8<Float> {
        let lo = SIMD8<Float>(repeating: -87.0)
        let hi = SIMD8<Float>(repeating: 88.0)
        var v = x.replacing(with: lo, where: x .< lo)
        v = v.replacing(with: hi, where: hi .< v)
        let log2e = SIMD8<Float>(repeating: 1.442695041)
        // 1.5 × 2^23 を足すと、仮数部の下位に最も近い整数が入る (丸めと整数化を SIMD のまま行う。
        // SIMD8<Int32>(_:rounding:) はレーンごとの汎用変換になり、推論時間の大半を占めた)
        let magic = SIMD8<Float>(repeating: 12582912.0)
        let shifted = v * log2e + magic
        let n = shifted - magic
        // ln2 を上位と下位に分けて引き、丸め誤差を抑える
        let r = (v - n * SIMD8<Float>(repeating: 0.693359375)) - n * SIMD8<Float>(repeating: -2.12194440e-4)
        var p = SIMD8<Float>(repeating: 1.0 / 720.0)
        p = p * r + SIMD8<Float>(repeating: 1.0 / 120.0)
        p = p * r + SIMD8<Float>(repeating: 1.0 / 24.0)
        p = p * r + SIMD8<Float>(repeating: 1.0 / 6.0)
        p = p * r + SIMD8<Float>(repeating: 0.5)
        p = p * r + SIMD8<Float>(repeating: 1.0)
        p = p * r + SIMD8<Float>(repeating: 1.0)
        let integer = unsafeBitCast(shifted, to: SIMD8<Int32>.self) &- SIMD8<Int32>(repeating: 0x4B40_0000)
        let bits = (integer &+ SIMD8<Int32>(repeating: 127)) &<< SIMD8<Int32>(repeating: 23)
        return p * unsafeBitCast(bits, to: SIMD8<Float>.self)
    }

    /// 1 / (1 + exp(-x))
    @inline(__always)
    public static func sigmoid(_ x: SIMD8<Float>) -> SIMD8<Float> {
        let one = SIMD8<Float>(repeating: 1.0)
        return one / (one + exp(-x))
    }
}
