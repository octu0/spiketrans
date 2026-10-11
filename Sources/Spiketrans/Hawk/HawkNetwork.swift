import Accelerate
import Foundation

/// Hawk 型音響モデルの CPU 推論。チャンクごとに前向きに進め、状態 (RG-LRU の記憶と畳み込みの直近フレーム) を引き継ぐ。
/// 線形はチャンクのフレームをまとめた行列積 (Accelerate)、ゲートと再帰は SIMD8。
/// 幅 (modelDim・rnnDim・mlpDim) は 8 の倍数であること
public final class HawkNetwork: @unchecked Sendable {
    public let weights: HawkWeights
    /// 層ごとの c log σ(Λ) (log a_t = r_t × これ)
    private let logDecay: [[Float]]

    /// 1 発話ぶんの状態
    public final class State {
        /// 層ごとの RG-LRU の記憶 [rnnDim]
        public var memory: [[Float]]
        /// 層ごとの畳み込みに入る直近 convWidth - 1 フレーム [(convWidth - 1) × rnnDim]
        public var convTail: [[Float]]

        init(weights: HawkWeights) {
            memory = [[Float]](repeating: [Float](repeating: 0.0, count: weights.rnnDim), count: weights.numLayers)
            convTail = [[Float]](repeating: [Float](repeating: 0.0, count: (weights.convWidth - 1) * weights.rnnDim), count: weights.numLayers)
        }
    }

    public init(weights: HawkWeights) {
        precondition(weights.modelDim % 8 == 0 && weights.rnnDim % 8 == 0 && weights.mlpDim % 8 == 0, "幅は 8 の倍数")
        self.weights = weights
        self.logDecay = weights.layers.map { layer in
            layer.lambda.map { lambda -> Float in
                // log σ(Λ) を桁あふれなく求める
                var logSigmoid = -log1pf(expf(-lambda))
                if lambda < 0.0 {
                    logSigmoid = lambda - log1pf(expf(lambda))
                }
                return HawkWeights.decayExponent * logSigmoid
            }
        }
    }

    public func makeState() -> State {
        return State(weights: weights)
    }

    /// 発話全体の対数確率 [T][outputDim]
    public func logProbs(features: [[Float]], chunkFrames: Int = 128) -> [[Float]] {
        let state = makeState()
        let inDim = weights.inputDim
        let outDim = weights.outputDim
        var result: [[Float]] = []
        result.reserveCapacity(features.count)
        var t0 = 0
        while t0 < features.count {
            let t1 = min(features.count, t0 + max(1, chunkFrames))
            var chunk = [Float](repeating: 0.0, count: (t1 - t0) * inDim)
            var t = t0
            while t < t1 {
                let row = features[t]
                let base = (t - t0) * inDim
                var d = 0
                while d < min(inDim, row.count) {
                    chunk[base + d] = row[d]
                    d += 1
                }
                t += 1
            }
            let logits = forward(chunk: chunk, frames: t1 - t0, state: state)
            var f = 0
            while f < t1 - t0 {
                result.append(Self.logSoftmax(Array(logits[(f * outDim)..<((f + 1) * outDim)])))
                f += 1
            }
            t0 = t1
        }
        return result
    }

    /// チャンク [frames × inputDim] を前向きに進め、ロジット [frames × outputDim] を返す。状態は更新する
    public func forward(chunk: [Float], frames: Int, state: State) -> [Float] {
        let w = weights
        let d = w.modelDim
        let r = w.rnnDim
        let m = w.mlpDim
        let hd = w.headDim
        let k = w.convWidth
        var x = [Float](repeating: 0.0, count: frames * d)
        var f = 0
        while f < frames {
            var c = 0
            while c < d {
                x[f * d + c] = w.bIn[c]
                c += 1
            }
            f += 1
        }
        Self.gemm(chunk, w.wIn, &x, m: frames, n: d, k: w.inputDim, beta: 1.0)

        var normed = [Float](repeating: 0.0, count: frames * d)
        var branch = [Float](repeating: 0.0, count: frames * r)
        var extended = [Float](repeating: 0.0, count: (frames + k - 1) * r)
        var u = [Float](repeating: 0.0, count: frames * r)
        var keepPre = [Float](repeating: 0.0, count: frames * r)
        var admitPre = [Float](repeating: 0.0, count: frames * r)
        var mlpGate = [Float](repeating: 0.0, count: frames * m)
        var mlpUp = [Float](repeating: 0.0, count: frames * m)

        var l = 0
        while l < w.numLayers {
            let lw = w.layers[l]
            Self.rmsNorm(x, lw.norm1, &normed, frames: frames, width: d)
            Self.gemm(normed, lw.wGate, &branch, m: frames, n: r, k: d)
            Self.geluInPlace(&branch)
            // 畳み込みの入力: 前のチャンクの末尾 k - 1 フレームに、このチャンクの線形の出力を続ける
            extended.withUnsafeMutableBufferPointer { ext in
                state.convTail[l].withUnsafeBufferPointer { tail in
                    ext.baseAddress!.update(from: tail.baseAddress!, count: (k - 1) * r)
                }
            }
            extended.withUnsafeMutableBufferPointer { ext in
                normed.withUnsafeBufferPointer { a in
                    lw.wX.withUnsafeBufferPointer { b in
                        Self.gemmRaw(a.baseAddress!, d, b.baseAddress!, r, ext.baseAddress! + (k - 1) * r, r, m: frames, n: r, k: d, beta: 0.0)
                    }
                }
            }
            Self.causalConv(extended, lw.convW, lw.convB, &u, frames: frames, width: r, kernel: k)
            state.convTail[l] = Array(extended[(frames * r)..<((frames + k - 1) * r)])
            // ブロック対角のゲート
            var h = 0
            while h < w.heads {
                let offset = h * hd * hd
                u.withUnsafeBufferPointer { ub in
                    lw.wA.withUnsafeBufferPointer { wa in
                        keepPre.withUnsafeMutableBufferPointer { out in
                            Self.gemmRaw(ub.baseAddress! + h * hd, r, wa.baseAddress! + offset, hd, out.baseAddress! + h * hd, r, m: frames, n: hd, k: hd, beta: 0.0)
                        }
                    }
                    lw.wI.withUnsafeBufferPointer { wi in
                        admitPre.withUnsafeMutableBufferPointer { out in
                            Self.gemmRaw(ub.baseAddress! + h * hd, r, wi.baseAddress! + offset, hd, out.baseAddress! + h * hd, r, m: frames, n: hd, k: hd, beta: 0.0)
                        }
                    }
                }
                h += 1
            }
            scan(layer: l, frames: frames, u: u, keepPre: keepPre, admitPre: admitPre, branch: &branch, memory: &state.memory[l])
            Self.gemm(branch, lw.wO, &x, m: frames, n: d, k: r, beta: 1.0)

            Self.rmsNorm(x, lw.norm2, &normed, frames: frames, width: d)
            Self.gemm(normed, lw.wMLPGate, &mlpGate, m: frames, n: m, k: d)
            Self.gemm(normed, lw.wMLPUp, &mlpUp, m: frames, n: m, k: d)
            Self.geluInPlace(&mlpGate)
            Self.multiplyInPlace(&mlpGate, mlpUp)
            Self.gemm(mlpGate, lw.wMLPDown, &x, m: frames, n: d, k: m, beta: 1.0)
            l += 1
        }
        Self.rmsNorm(x, w.finalNorm, &normed, frames: frames, width: d)
        var logits = [Float](repeating: 0.0, count: frames * w.outputDim)
        f = 0
        while f < frames {
            var c = 0
            while c < w.outputDim {
                logits[f * w.outputDim + c] = w.bOut[c]
                c += 1
            }
            f += 1
        }
        Self.gemm(normed, w.wOut, &logits, m: frames, n: w.outputDim, k: d, beta: 1.0)
        return logits
    }

    /// RG-LRU: a = exp(r c log σ(Λ))、h = a h + √(1 - a²) (i u)。出力 h を GeLU の枝に掛けて branch に書く
    private func scan(layer: Int, frames: Int, u: [Float], keepPre: [Float], admitPre: [Float], branch: inout [Float], memory: inout [Float]) {
        let lw = weights.layers[layer]
        let r = weights.rnnDim
        let one = SIMD8<Float>(repeating: 1.0)
        let eps = SIMD8<Float>(repeating: 1e-6)
        logDecay[layer].withUnsafeBufferPointer { decay in
        lw.bA.withUnsafeBufferPointer { bA in
        lw.bI.withUnsafeBufferPointer { bI in
        u.withUnsafeBufferPointer { ub in
        keepPre.withUnsafeBufferPointer { kp in
        admitPre.withUnsafeBufferPointer { ap in
        branch.withUnsafeMutableBufferPointer { out in
        memory.withUnsafeMutableBufferPointer { mem in
            let dp = decay.baseAddress!
            let bap = bA.baseAddress!
            let bip = bI.baseAddress!
            let up = ub.baseAddress!
            let kpp = kp.baseAddress!
            let app = ap.baseAddress!
            let op = out.baseAddress!
            let mp = mem.baseAddress!
            var c = 0
            while c < r {
                let logA = Self.load(dp + c)
                let keepBias = Self.load(bap + c)
                let admitBias = Self.load(bip + c)
                var state = Self.load(mp + c)
                var t = 0
                while t < frames {
                    let idx = t * r + c
                    let keep = SIMDMath.sigmoid(Self.load(kpp + idx) + keepBias)
                    let admit = SIMDMath.sigmoid(Self.load(app + idx) + admitBias)
                    let a = SIMDMath.exp(keep * logA)
                    let scale = (one - a * a + eps).squareRoot()
                    state = a * state + scale * (admit * Self.load(up + idx))
                    Self.store(state * Self.load(op + idx), op + idx)
                    t += 1
                }
                Self.store(state, mp + c)
                c += 8
            }
        }}}}}}}}
    }

    // MARK: - 補助

    /// c = a · b (+ beta c)。行優先、a [m, k]・b [k, n]・c [m, n]
    static func gemm(_ a: [Float], _ b: [Float], _ c: inout [Float], m: Int, n: Int, k: Int, beta: Float = 0.0) {
        a.withUnsafeBufferPointer { ap in
            b.withUnsafeBufferPointer { bp in
                c.withUnsafeMutableBufferPointer { cp in
                    gemmRaw(ap.baseAddress!, k, bp.baseAddress!, n, cp.baseAddress!, n, m: m, n: n, k: k, beta: beta)
                }
            }
        }
    }

    static func gemmRaw(_ a: UnsafePointer<Float>, _ lda: Int, _ b: UnsafePointer<Float>, _ ldb: Int,
                        _ c: UnsafeMutablePointer<Float>, _ ldc: Int, m: Int, n: Int, k: Int, beta: Float) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(m), Int32(n), Int32(k),
                    1.0, a, Int32(lda), b, Int32(ldb), beta, c, Int32(ldc))
    }

    static func rmsNorm(_ x: [Float], _ gain: [Float], _ out: inout [Float], frames: Int, width: Int) {
        var f = 0
        while f < frames {
            let base = f * width
            var sumSq: Float = 0.0
            var c = 0
            while c < width {
                sumSq += x[base + c] * x[base + c]
                c += 1
            }
            let scale = 1.0 / sqrtf(sumSq / Float(width) + rmsNormEpsilon)
            c = 0
            while c < width {
                out[base + c] = x[base + c] * scale * gain[c]
                c += 1
            }
            f += 1
        }
    }

    /// x σ(1.702 x)
    static func geluInPlace(_ x: inout [Float]) {
        let k = SIMD8<Float>(repeating: 1.702)
        x.withUnsafeMutableBufferPointer { p in
            let base = p.baseAddress!
            var i = 0
            while i + 8 <= p.count {
                let v = load(base + i)
                store(v * SIMDMath.sigmoid(k * v), base + i)
                i += 8
            }
            while i < p.count {
                p[i] = p[i] / (1.0 + expf(-1.702 * p[i]))
                i += 1
            }
        }
    }

    @inline(__always)
    static func load(_ p: UnsafePointer<Float>) -> SIMD8<Float> {
        return UnsafeRawPointer(p).loadUnaligned(as: SIMD8<Float>.self)
    }

    @inline(__always)
    static func load(_ p: UnsafeMutablePointer<Float>) -> SIMD8<Float> {
        return UnsafeRawPointer(p).loadUnaligned(as: SIMD8<Float>.self)
    }

    @inline(__always)
    static func store(_ v: SIMD8<Float>, _ p: UnsafeMutablePointer<Float>) {
        UnsafeMutableRawPointer(p).storeBytes(of: v, as: SIMD8<Float>.self)
    }

    static func multiplyInPlace(_ x: inout [Float], _ y: [Float]) {
        var i = 0
        while i < x.count {
            x[i] *= y[i]
            i += 1
        }
    }

    /// 因果畳み込み。extended は [(frames + kernel - 1) × width] で、出力フレーム t は行 t〜t + kernel - 1 を見る
    static func causalConv(_ extended: [Float], _ convW: [Float], _ convB: [Float], _ out: inout [Float], frames: Int, width: Int, kernel: Int) {
        var t = 0
        while t < frames {
            var c = 0
            while c < width {
                var acc = convB[c]
                var j = 0
                while j < kernel {
                    acc += convW[j * width + c] * extended[(t + j) * width + c]
                    j += 1
                }
                out[t * width + c] = acc
                c += 1
            }
            t += 1
        }
    }

    static func logSoftmax(_ logits: [Float]) -> [Float] {
        var peak = -Float.greatestFiniteMagnitude
        for v in logits {
            if peak < v {
                peak = v
            }
        }
        var total: Float = 0.0
        for v in logits {
            total += expf(v - peak)
        }
        let norm = peak + logf(total)
        return logits.map { $0 - norm }
    }
}
