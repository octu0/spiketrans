import Foundation

/// 2D-Convolutional Subsampling 重みデータモデル (Codable & Sendable)
public struct Conv2DSubsamplingWeights: Sendable, Codable, Equatable {
    public let inChannels: Int
    public let outChannels1: Int
    public let outChannels2: Int
    public let melChannels: Int
    public let outputDim: Int

    /// Conv1 重み [outChannels1, 3, 3, inChannels] (平坦化: 16 * 3 * 3 * 1 = 144)
    public let conv1Weight: [Float]
    /// Conv1 バイアス [outChannels1] (16)
    public let conv1Bias: [Float]

    /// Conv2 重み [outChannels2, 3, 3, outChannels1] (平坦化: 16 * 3 * 3 * 16 = 2304)
    public let conv2Weight: [Float]
    /// Conv2 バイアス [outChannels2] (16)
    public let conv2Bias: [Float]

    /// 線形射影層 重み [outputDim, outChannels2 * 16] (平坦化: outputDim * 256)
    public let projWeight: [Float]
    /// 線形射影層 バイアス [outputDim]
    public let projBias: [Float]

    public init(
        inChannels: Int = 1,
        outChannels1: Int = 16,
        outChannels2: Int = 16,
        melChannels: Int = 64,
        outputDim: Int = 128,
        conv1Weight: [Float],
        conv1Bias: [Float],
        conv2Weight: [Float],
        conv2Bias: [Float],
        projWeight: [Float],
        projBias: [Float]
    ) {
        self.inChannels = inChannels
        self.outChannels1 = outChannels1
        self.outChannels2 = outChannels2
        self.melChannels = melChannels
        self.outputDim = outputDim
        self.conv1Weight = conv1Weight
        self.conv1Bias = conv1Bias
        self.conv2Weight = conv2Weight
        self.conv2Bias = conv2Bias
        self.projWeight = projWeight
        self.projBias = projBias
    }
}

/// ストリーミング推論用の因果バッファ状態
public struct Conv2DStreamingState: Sendable {
    public var pastMelFrames: [[Float]]
    public var totalMelFrames: Int
    public var pastConv1Frames: [[Float]]
    public var totalConv1Frames: Int

    public init() {
        self.pastMelFrames = []
        self.pastMelFrames.reserveCapacity(8)
        self.totalMelFrames = 0
        self.pastConv1Frames = []
        self.pastConv1Frames.reserveCapacity(8)
        self.totalConv1Frames = 0
    }

    public mutating func reset() {
        pastMelFrames.removeAll(keepingCapacity: true)
        totalMelFrames = 0
        pastConv1Frames.removeAll(keepingCapacity: true)
        totalConv1Frames = 0
    }
}

/// 軽量 2D-Convolutional フロントエンド (Pure Swift SIMD 実装)
///
/// 64ch Mel スペクトログラムを入力とし、時間軸を 1/4 に圧縮しつつ
/// 周波数の局所並進不変性（話者のピッチ・フォルマント周波数シフトの吸収）を抽出する。
/// 時間軸方向は未来フレームを一切参照しない因果的パディング (Causal Padding) を採用し、
/// ストリーミング文字起こしにおける遅延をゼロにする。
public final class Conv2DSubsampling: @unchecked Sendable {
    public let inChannels: Int
    public let outChannels1: Int
    public let outChannels2: Int
    public let melChannels: Int
    public let outputDim: Int

    public private(set) var conv1Weight: [Float]
    public private(set) var conv1Bias: [Float]
    public private(set) var conv2Weight: [Float]
    public private(set) var conv2Bias: [Float]
    public private(set) var projWeight: [Float]
    public private(set) var projBias: [Float]

    private var conv1SIMD0: [SIMD8<Float>] = []
    private var conv1SIMD1: [SIMD8<Float>] = []
    private var conv1BiasSIMD0: SIMD8<Float> = .zero
    private var conv1BiasSIMD1: SIMD8<Float> = .zero

    public init(
        inChannels: Int = 1,
        outChannels1: Int = 16,
        outChannels2: Int = 16,
        melChannels: Int = 64,
        outputDim: Int = 128
    ) {
        self.inChannels = inChannels
        self.outChannels1 = outChannels1
        self.outChannels2 = outChannels2
        self.melChannels = melChannels
        self.outputDim = outputDim

        // He (Kaiming) 一様分布による重み初期化
        let scale1 = sqrt(2.0 / Float(inChannels * 3 * 3))
        let count1 = outChannels1 * 3 * 3 * inChannels
        var w1 = [Float](repeating: 0.0, count: count1)
        var i = 0
        while i < count1 {
            w1[i] = Float.random(in: -scale1...scale1)
            i += 1
        }
        self.conv1Weight = w1
        self.conv1Bias = [Float](repeating: 0.0, count: outChannels1)

        let scale2 = sqrt(2.0 / Float(outChannels1 * 3 * 3))
        let count2 = outChannels2 * 3 * 3 * outChannels1
        var w2 = [Float](repeating: 0.0, count: count2)
        i = 0
        while i < count2 {
            w2[i] = Float.random(in: -scale2...scale2)
            i += 1
        }
        self.conv2Weight = w2
        self.conv2Bias = [Float](repeating: 0.0, count: outChannels2)

        let flatDim = outChannels2 * 16 // 16 * 16 = 256
        let scaleProj = sqrt(2.0 / Float(flatDim))
        let countProj = outputDim * flatDim
        var wp = [Float](repeating: 0.0, count: countProj)
        i = 0
        while i < countProj {
            wp[i] = Float.random(in: -scaleProj...scaleProj)
            i += 1
        }
        self.projWeight = wp
        self.projBias = [Float](repeating: 0.0, count: outputDim)
        rebuildSIMDWeights()
    }

    public init(weights: Conv2DSubsamplingWeights) {
        self.inChannels = weights.inChannels
        self.outChannels1 = weights.outChannels1
        self.outChannels2 = weights.outChannels2
        self.melChannels = weights.melChannels
        self.outputDim = weights.outputDim
        self.conv1Weight = weights.conv1Weight
        self.conv1Bias = weights.conv1Bias
        self.conv2Weight = weights.conv2Weight
        self.conv2Bias = weights.conv2Bias
        self.projWeight = weights.projWeight
        self.projBias = weights.projBias
        rebuildSIMDWeights()
    }

    public func exportWeights() -> Conv2DSubsamplingWeights {
        return Conv2DSubsamplingWeights(
            inChannels: inChannels,
            outChannels1: outChannels1,
            outChannels2: outChannels2,
            melChannels: melChannels,
            outputDim: outputDim,
            conv1Weight: conv1Weight,
            conv1Bias: conv1Bias,
            conv2Weight: conv2Weight,
            conv2Bias: conv2Bias,
            projWeight: projWeight,
            projBias: projBias
        )
    }

    public func importWeights(from weights: Conv2DSubsamplingWeights) {
        self.conv1Weight = weights.conv1Weight
        self.conv1Bias = weights.conv1Bias
        self.conv2Weight = weights.conv2Weight
        self.conv2Bias = weights.conv2Bias
        self.projWeight = weights.projWeight
        self.projBias = weights.projBias
        rebuildSIMDWeights()
    }

    private func rebuildSIMDWeights() {
        var s0 = [SIMD8<Float>](repeating: .zero, count: 9)
        var s1 = [SIMD8<Float>](repeating: .zero, count: 9)
        conv1Weight.withUnsafeBufferPointer { wBuf in
            let wPtr = wBuf.baseAddress!
            var k = 0
            while k < 9 {
                s0[k] = SIMD8<Float>(
                    wPtr[0 * 9 + k], wPtr[1 * 9 + k], wPtr[2 * 9 + k], wPtr[3 * 9 + k],
                    wPtr[4 * 9 + k], wPtr[5 * 9 + k], wPtr[6 * 9 + k], wPtr[7 * 9 + k]
                )
                s1[k] = SIMD8<Float>(
                    wPtr[8 * 9 + k], wPtr[9 * 9 + k], wPtr[10 * 9 + k], wPtr[11 * 9 + k],
                    wPtr[12 * 9 + k], wPtr[13 * 9 + k], wPtr[14 * 9 + k], wPtr[15 * 9 + k]
                )
                k += 1
            }
        }
        self.conv1SIMD0 = s0
        self.conv1SIMD1 = s1
        conv1Bias.withUnsafeBufferPointer { bBuf in
            let bPtr = bBuf.baseAddress!
            self.conv1BiasSIMD0 = SIMD8<Float>(bPtr[0], bPtr[1], bPtr[2], bPtr[3], bPtr[4], bPtr[5], bPtr[6], bPtr[7])
            self.conv1BiasSIMD1 = SIMD8<Float>(bPtr[8], bPtr[9], bPtr[10], bPtr[11], bPtr[12], bPtr[13], bPtr[14], bPtr[15])
        }
    }

    /// 64ch Mel スペクトログラム系列 [T][64] から 1/4 時間圧縮された特徴量系列 [T/4][outputDim] を抽出
    public func forward(melSpectrogram: [[Float]]) -> [[Float]] {
        let tTotal = melSpectrogram.count
        if tTotal <= 0 {
            return []
        }

        // 第1段 Conv2D (時間 1/2 圧縮, 周波数 64 -> 32)
        let t1 = (tTotal + 1) / 2
        let conv1Out = forwardConv1(melSpectrogram: melSpectrogram, tTotal: tTotal, t1: t1)

        // 第2段 Conv2D (時間 1/2 圧縮, 周波数 32 -> 16)
        let t2 = (t1 + 1) / 2
        let conv2Out = forwardConv2(conv1Out: conv1Out, t1: t1, t2: t2)

        // 第3段 線形射影 (256 -> outputDim)
        return forwardProjection(conv2Out: conv2Out, t2: t2)
    }

    // MARK: - Hot Path 内部処理: Conv1

    private func forwardConv1(
        melSpectrogram: [[Float]],
        tTotal: Int,
        t1: Int
    ) -> [Float] {
        let fOutCount = 32
        let cOutCount = 16
        var output = [Float](repeating: 0.0, count: t1 * fOutCount * cOutCount)

        // 入力 Mel を連続バッファにフラット展開 (メモリ局所性 & ポインタの安全性)
        var flatMel = [Float](repeating: 0.0, count: tTotal * 64)
        flatMel.withUnsafeMutableBufferPointer { flatBuf in
            var dst = flatBuf.baseAddress!
            var i = 0
            while i < tTotal {
                melSpectrogram[i].withUnsafeBufferPointer { srcBuf in
                    dst.update(from: srcBuf.baseAddress!, count: 64)
                }
                dst = dst.advanced(by: 64)
                i += 1
            }
        }

        let b0 = conv1BiasSIMD0
        let b1 = conv1BiasSIMD1
        let w0 = conv1SIMD0
        let w1 = conv1SIMD1

        flatMel.withUnsafeBufferPointer { inBuf in
            let inPtr = inBuf.baseAddress!
            output.withUnsafeMutableBufferPointer { outBuf in
                let outPtr = outBuf.baseAddress!

                // 1. 時間境界: t = 0 (tSrc0, tSrc1 はパディング、tSrc2 = 0 のみ有効)
                if 0 < t1 {
                    let ptr2 = inPtr
                    computeConv1CellT0Boundary(ptr2: ptr2, b0: b0, b1: b1, w0: w0, w1: w1, dst: outPtr)
                    var f = 1
                    while f < fOutCount {
                        computeConv1CellT0Inner(ptr2: ptr2, f: f, b0: b0, b1: b1, w0: w0, w1: w1, dst: outPtr.advanced(by: f * cOutCount))
                        f += 1
                    }
                }

                // 2. 時間内部: 1 <= t < t1 (ptr0, ptr1, ptr2 すべて非 nil)
                var t = 1
                while t < t1 {
                    let tSrc0 = 2 * t - 2
                    let tSrc1 = 2 * t - 1
                    let tSrc2 = 2 * t

                    let ptr0 = inPtr.advanced(by: tSrc0 * 64)
                    let ptr1 = inPtr.advanced(by: tSrc1 * 64)
                    let ptr2 = inPtr.advanced(by: tSrc2 * 64)
                    let tOutOffset = t * fOutCount * cOutCount

                    // f = 0 (周波数境界: fSrc0 = -1 でゼロパディング)
                    computeConv1CellTimeInnerBoundaryF0(
                        ptr0: ptr0,
                        ptr1: ptr1,
                        ptr2: ptr2,
                        b0: b0,
                        b1: b1,
                        w0: w0,
                        w1: w1,
                        dst: outPtr.advanced(by: tOutOffset)
                    )

                    // f = 1..<32 (周波数内部: 境界外なし、完全アンロール)
                    var f = 1
                    while f < fOutCount {
                        computeConv1CellTimeInnerPure(
                            ptr0: ptr0,
                            ptr1: ptr1,
                            ptr2: ptr2,
                            f: f,
                            b0: b0,
                            b1: b1,
                            w0: w0,
                            w1: w1,
                            dst: outPtr.advanced(by: tOutOffset + (f * cOutCount))
                        )
                        f += 1
                    }

                    t += 1
                }
            }
        }

        return output
    }

    @inline(__always)
    private func computeConv1CellTimeInnerPure(
        ptr0: UnsafePointer<Float>,
        ptr1: UnsafePointer<Float>,
        ptr2: UnsafePointer<Float>,
        f: Int,
        b0: SIMD8<Float>,
        b1: SIMD8<Float>,
        w0: [SIMD8<Float>],
        w1: [SIMD8<Float>],
        dst: UnsafeMutablePointer<Float>
    ) {
        let f0 = 2 * f - 1
        let f1 = 2 * f
        let f2 = 2 * f + 1

        let p0_0 = ptr0[f0]
        let p0_1 = ptr0[f1]
        let p0_2 = ptr0[f2]

        let p1_0 = ptr1[f0]
        let p1_1 = ptr1[f1]
        let p1_2 = ptr1[f2]

        let p2_0 = ptr2[f0]
        let p2_1 = ptr2[f1]
        let p2_2 = ptr2[f2]

        var acc0 = b0
        var acc1 = b1

        acc0 += w0[0] * p0_0 + w0[1] * p0_1 + w0[2] * p0_2
        acc0 += w0[3] * p1_0 + w0[4] * p1_1 + w0[5] * p1_2
        acc0 += w0[6] * p2_0 + w0[7] * p2_1 + w0[8] * p2_2

        acc1 += w1[0] * p0_0 + w1[1] * p0_1 + w1[2] * p0_2
        acc1 += w1[3] * p1_0 + w1[4] * p1_1 + w1[5] * p1_2
        acc1 += w1[6] * p2_0 + w1[7] * p2_1 + w1[8] * p2_2

        let zero = SIMD8<Float>(repeating: 0.0)
        let r0 = acc0.replacing(with: zero, where: acc0 .< zero)
        let r1 = acc1.replacing(with: zero, where: acc1 .< zero)

        dst[0] = r0[0]; dst[1] = r0[1]; dst[2] = r0[2]; dst[3] = r0[3]
        dst[4] = r0[4]; dst[5] = r0[5]; dst[6] = r0[6]; dst[7] = r0[7]
        dst[8] = r1[0]; dst[9] = r1[1]; dst[10] = r1[2]; dst[11] = r1[3]
        dst[12] = r1[4]; dst[13] = r1[5]; dst[14] = r1[6]; dst[15] = r1[7]
    }

    @inline(__always)
    private func computeConv1CellTimeInnerBoundaryF0(
        ptr0: UnsafePointer<Float>,
        ptr1: UnsafePointer<Float>,
        ptr2: UnsafePointer<Float>,
        b0: SIMD8<Float>,
        b1: SIMD8<Float>,
        w0: [SIMD8<Float>],
        w1: [SIMD8<Float>],
        dst: UnsafeMutablePointer<Float>
    ) {
        let p0_1 = ptr0[0]
        let p0_2 = ptr0[1]

        let p1_1 = ptr1[0]
        let p1_2 = ptr1[1]

        let p2_1 = ptr2[0]
        let p2_2 = ptr2[1]

        var acc0 = b0
        var acc1 = b1

        acc0 += w0[1] * p0_1 + w0[2] * p0_2
        acc0 += w0[4] * p1_1 + w0[5] * p1_2
        acc0 += w0[7] * p2_1 + w0[8] * p2_2

        acc1 += w1[1] * p0_1 + w1[2] * p0_2
        acc1 += w1[4] * p1_1 + w1[5] * p1_2
        acc1 += w1[7] * p2_1 + w1[8] * p2_2

        let zero = SIMD8<Float>(repeating: 0.0)
        let r0 = acc0.replacing(with: zero, where: acc0 .< zero)
        let r1 = acc1.replacing(with: zero, where: acc1 .< zero)

        dst[0] = r0[0]; dst[1] = r0[1]; dst[2] = r0[2]; dst[3] = r0[3]
        dst[4] = r0[4]; dst[5] = r0[5]; dst[6] = r0[6]; dst[7] = r0[7]
        dst[8] = r1[0]; dst[9] = r1[1]; dst[10] = r1[2]; dst[11] = r1[3]
        dst[12] = r1[4]; dst[13] = r1[5]; dst[14] = r1[6]; dst[15] = r1[7]
    }

    @inline(__always)
    private func computeConv1CellT0Inner(
        ptr2: UnsafePointer<Float>,
        f: Int,
        b0: SIMD8<Float>,
        b1: SIMD8<Float>,
        w0: [SIMD8<Float>],
        w1: [SIMD8<Float>],
        dst: UnsafeMutablePointer<Float>
    ) {
        let f0 = 2 * f - 1
        let f1 = 2 * f
        let f2 = 2 * f + 1

        let p2_0 = ptr2[f0]
        let p2_1 = ptr2[f1]
        let p2_2 = ptr2[f2]

        var acc0 = b0
        var acc1 = b1

        acc0 += w0[6] * p2_0 + w0[7] * p2_1 + w0[8] * p2_2
        acc1 += w1[6] * p2_0 + w1[7] * p2_1 + w1[8] * p2_2

        let zero = SIMD8<Float>(repeating: 0.0)
        let r0 = acc0.replacing(with: zero, where: acc0 .< zero)
        let r1 = acc1.replacing(with: zero, where: acc1 .< zero)

        dst[0] = r0[0]; dst[1] = r0[1]; dst[2] = r0[2]; dst[3] = r0[3]
        dst[4] = r0[4]; dst[5] = r0[5]; dst[6] = r0[6]; dst[7] = r0[7]
        dst[8] = r1[0]; dst[9] = r1[1]; dst[10] = r1[2]; dst[11] = r1[3]
        dst[12] = r1[4]; dst[13] = r1[5]; dst[14] = r1[6]; dst[15] = r1[7]
    }

    @inline(__always)
    private func computeConv1CellT0Boundary(
        ptr2: UnsafePointer<Float>,
        b0: SIMD8<Float>,
        b1: SIMD8<Float>,
        w0: [SIMD8<Float>],
        w1: [SIMD8<Float>],
        dst: UnsafeMutablePointer<Float>
    ) {
        let p2_1 = ptr2[0]
        let p2_2 = ptr2[1]

        var acc0 = b0
        var acc1 = b1

        acc0 += w0[7] * p2_1 + w0[8] * p2_2
        acc1 += w1[7] * p2_1 + w1[8] * p2_2

        let zero = SIMD8<Float>(repeating: 0.0)
        let r0 = acc0.replacing(with: zero, where: acc0 .< zero)
        let r1 = acc1.replacing(with: zero, where: acc1 .< zero)

        dst[0] = r0[0]; dst[1] = r0[1]; dst[2] = r0[2]; dst[3] = r0[3]
        dst[4] = r0[4]; dst[5] = r0[5]; dst[6] = r0[6]; dst[7] = r0[7]
        dst[8] = r1[0]; dst[9] = r1[1]; dst[10] = r1[2]; dst[11] = r1[3]
        dst[12] = r1[4]; dst[13] = r1[5]; dst[14] = r1[6]; dst[15] = r1[7]
    }

    // MARK: - Hot Path 内部処理: Conv2

    private func forwardConv2(
        conv1Out: [Float],
        t1: Int,
        t2: Int
    ) -> [Float] {
        let f1Count = 32
        let c1Count = 16
        let f2Count = 16
        let c2Count = 16
        var output = [Float](repeating: 0.0, count: t2 * f2Count * c2Count)

        conv2Weight.withUnsafeBufferPointer { wBuf in
            let wPtr = wBuf.baseAddress!
            conv2Bias.withUnsafeBufferPointer { bBuf in
                let bPtr = bBuf.baseAddress!
                conv1Out.withUnsafeBufferPointer { inBuf in
                    let inPtr = inBuf.baseAddress!
                    output.withUnsafeMutableBufferPointer { outBuf in
                        let outPtr = outBuf.baseAddress!

                        // 1. 時間境界: t = 0 (tSrc0, tSrc1 はパディング、tSrc2 = 0 のみ有効)
                        if 0 < t2 {
                            let ptr2 = inPtr
                            computeConv2CellT0Boundary(ptr2: ptr2, wPtr: wPtr, bPtr: bPtr, dst: outPtr)
                            var f = 1
                            while f < f2Count {
                                computeConv2CellT0Inner(ptr2: ptr2, f: f, wPtr: wPtr, bPtr: bPtr, dst: outPtr.advanced(by: f * c2Count))
                                f += 1
                            }
                        }

                        // 2. 時間内部: 1 <= t < t2 (ptr0, ptr1, ptr2 すべて非 nil)
                        var t = 1
                        while t < t2 {
                            let tSrc0 = 2 * t - 2
                            let tSrc1 = 2 * t - 1
                            let tSrc2 = 2 * t

                            let ptr0 = inPtr.advanced(by: tSrc0 * f1Count * c1Count)
                            let ptr1 = inPtr.advanced(by: tSrc1 * f1Count * c1Count)
                            let ptr2 = inPtr.advanced(by: tSrc2 * f1Count * c1Count)
                            let tOutOffset = t * f2Count * c2Count

                            // f = 0 (周波数境界: fSrc0 = -1 でゼロパディング)
                            computeConv2CellTimeInnerBoundaryF0(
                                ptr0: ptr0,
                                ptr1: ptr1,
                                ptr2: ptr2,
                                wPtr: wPtr,
                                bPtr: bPtr,
                                dst: outPtr.advanced(by: tOutOffset)
                            )

                            // f = 1..<16 (周波数内部: 境界外なし、完全アンロール)
                            var f = 1
                            while f < f2Count {
                                computeConv2CellTimeInnerPure(
                                    ptr0: ptr0,
                                    ptr1: ptr1,
                                    ptr2: ptr2,
                                    f: f,
                                    wPtr: wPtr,
                                    bPtr: bPtr,
                                    dst: outPtr.advanced(by: tOutOffset + (f * c2Count))
                                )
                                f += 1
                            }

                            t += 1
                        }
                    }
                }
            }
        }

        return output
    }

    @inline(__always)
    private func computeConv2CellTimeInnerPure(
        ptr0: UnsafePointer<Float>,
        ptr1: UnsafePointer<Float>,
        ptr2: UnsafePointer<Float>,
        f: Int,
        wPtr: UnsafePointer<Float>,
        bPtr: UnsafePointer<Float>,
        dst: UnsafeMutablePointer<Float>
    ) {
        let f0 = (2 * f - 1) * 16
        let f1 = (2 * f) * 16
        let f2 = (2 * f + 1) * 16

        let p0_0 = ptr0.advanced(by: f0)
        let p0_1 = ptr0.advanced(by: f1)
        let p0_2 = ptr0.advanced(by: f2)

        let p1_0 = ptr1.advanced(by: f0)
        let p1_1 = ptr1.advanced(by: f1)
        let p1_2 = ptr1.advanced(by: f2)

        let p2_0 = ptr2.advanced(by: f0)
        let p2_1 = ptr2.advanced(by: f1)
        let p2_2 = ptr2.advanced(by: f2)

        var cOut = 0
        while cOut < 16 {
            var sum = bPtr[cOut]
            let wBase = cOut * 144

            sum += dotProduct16(input: p0_0, weight: wPtr.advanced(by: wBase + 0))
            sum += dotProduct16(input: p0_1, weight: wPtr.advanced(by: wBase + 16))
            sum += dotProduct16(input: p0_2, weight: wPtr.advanced(by: wBase + 32))

            sum += dotProduct16(input: p1_0, weight: wPtr.advanced(by: wBase + 48))
            sum += dotProduct16(input: p1_1, weight: wPtr.advanced(by: wBase + 64))
            sum += dotProduct16(input: p1_2, weight: wPtr.advanced(by: wBase + 80))

            sum += dotProduct16(input: p2_0, weight: wPtr.advanced(by: wBase + 96))
            sum += dotProduct16(input: p2_1, weight: wPtr.advanced(by: wBase + 112))
            sum += dotProduct16(input: p2_2, weight: wPtr.advanced(by: wBase + 128))

            if sum < 0.0 {
                dst[cOut] = 0.0
            } else {
                dst[cOut] = sum
            }
            cOut += 1
        }
    }

    @inline(__always)
    private func computeConv2CellTimeInnerBoundaryF0(
        ptr0: UnsafePointer<Float>,
        ptr1: UnsafePointer<Float>,
        ptr2: UnsafePointer<Float>,
        wPtr: UnsafePointer<Float>,
        bPtr: UnsafePointer<Float>,
        dst: UnsafeMutablePointer<Float>
    ) {
        let p0_1 = ptr0
        let p0_2 = ptr0.advanced(by: 16)

        let p1_1 = ptr1
        let p1_2 = ptr1.advanced(by: 16)

        let p2_1 = ptr2
        let p2_2 = ptr2.advanced(by: 16)

        var cOut = 0
        while cOut < 16 {
            var sum = bPtr[cOut]
            let wBase = cOut * 144

            sum += dotProduct16(input: p0_1, weight: wPtr.advanced(by: wBase + 16))
            sum += dotProduct16(input: p0_2, weight: wPtr.advanced(by: wBase + 32))

            sum += dotProduct16(input: p1_1, weight: wPtr.advanced(by: wBase + 64))
            sum += dotProduct16(input: p1_2, weight: wPtr.advanced(by: wBase + 80))

            sum += dotProduct16(input: p2_1, weight: wPtr.advanced(by: wBase + 112))
            sum += dotProduct16(input: p2_2, weight: wPtr.advanced(by: wBase + 128))

            if sum < 0.0 {
                dst[cOut] = 0.0
            } else {
                dst[cOut] = sum
            }
            cOut += 1
        }
    }

    @inline(__always)
    private func computeConv2CellT0Inner(
        ptr2: UnsafePointer<Float>,
        f: Int,
        wPtr: UnsafePointer<Float>,
        bPtr: UnsafePointer<Float>,
        dst: UnsafeMutablePointer<Float>
    ) {
        let f0 = (2 * f - 1) * 16
        let f1 = (2 * f) * 16
        let f2 = (2 * f + 1) * 16

        let p2_0 = ptr2.advanced(by: f0)
        let p2_1 = ptr2.advanced(by: f1)
        let p2_2 = ptr2.advanced(by: f2)

        var cOut = 0
        while cOut < 16 {
            var sum = bPtr[cOut]
            let wBase = cOut * 144

            sum += dotProduct16(input: p2_0, weight: wPtr.advanced(by: wBase + 96))
            sum += dotProduct16(input: p2_1, weight: wPtr.advanced(by: wBase + 112))
            sum += dotProduct16(input: p2_2, weight: wPtr.advanced(by: wBase + 128))

            if sum < 0.0 {
                dst[cOut] = 0.0
            } else {
                dst[cOut] = sum
            }
            cOut += 1
        }
    }

    @inline(__always)
    private func computeConv2CellT0Boundary(
        ptr2: UnsafePointer<Float>,
        wPtr: UnsafePointer<Float>,
        bPtr: UnsafePointer<Float>,
        dst: UnsafeMutablePointer<Float>
    ) {
        let p2_1 = ptr2
        let p2_2 = ptr2.advanced(by: 16)

        var cOut = 0
        while cOut < 16 {
            var sum = bPtr[cOut]
            let wBase = cOut * 144

            sum += dotProduct16(input: p2_1, weight: wPtr.advanced(by: wBase + 112))
            sum += dotProduct16(input: p2_2, weight: wPtr.advanced(by: wBase + 128))

            if sum < 0.0 {
                dst[cOut] = 0.0
            } else {
                dst[cOut] = sum
            }
            cOut += 1
        }
    }

    @inline(__always)
    private func dotProduct16(input: UnsafePointer<Float>, weight: UnsafePointer<Float>) -> Float {
        let in0 = SIMD8<Float>(input[0], input[1], input[2], input[3], input[4], input[5], input[6], input[7])
        let in1 = SIMD8<Float>(input[8], input[9], input[10], input[11], input[12], input[13], input[14], input[15])
        let w0 = SIMD8<Float>(weight[0], weight[1], weight[2], weight[3], weight[4], weight[5], weight[6], weight[7])
        let w1 = SIMD8<Float>(weight[8], weight[9], weight[10], weight[11], weight[12], weight[13], weight[14], weight[15])
        return (in0 * w0).sum() + (in1 * w1).sum()
    }

    // MARK: - Hot Path 内部処理: Linear Projection

    private func forwardProjection(conv2Out: [Float], t2: Int) -> [[Float]] {
        let flatDim = 256
        var results = [[Float]](repeating: [Float](repeating: 0.0, count: outputDim), count: t2)

        projWeight.withUnsafeBufferPointer { wBuf in
            let wPtr = wBuf.baseAddress!
            projBias.withUnsafeBufferPointer { bBuf in
                let bPtr = bBuf.baseAddress!
                conv2Out.withUnsafeBufferPointer { inBuf in
                    let inBase = inBuf.baseAddress!

                    var t = 0
                    while t < t2 {
                        let tInPtr = inBase.advanced(by: t * flatDim)
                        var d = 0
                        while d < outputDim {
                            let wRow = wPtr.advanced(by: d * flatDim)
                            let dot = VectorOperations.dotProduct(a: tInPtr, b: wRow, count: flatDim)
                            results[t][d] = bPtr[d] + dot
                            d += 1
                        }
                        t += 1
                    }
                }
            }
        }

        return results
    }

    // MARK: - ストリーミング推論

    /// 1フレームずつの Mel 特徴量入力をストリーミング処理し、4フレーム蓄積ごとに
    /// 1つの 1/4 時間圧縮特徴量 [outputDim] を返す (因果的・未来参照ゼロ)
    public func forwardStreaming(
        melFrame: [Float],
        state: inout Conv2DStreamingState
    ) -> [Float]? {
        state.pastMelFrames.append(melFrame)
        state.totalMelFrames += 1

        let tMel = state.totalMelFrames
        // Conv1 は時間 stride = 2 で発火 (奇数番目フレーム到着時: 1, 3, 5, ...)
        if (tMel % 2) != 1 {
            return nil
        }

        // Conv1 出力を計算
        let t1Idx = state.totalConv1Frames
        let c1Out = computeStreamingConv1Frame(
            t1Idx: t1Idx,
            frames: state.pastMelFrames
        )

        state.pastConv1Frames.append(c1Out)
        state.totalConv1Frames += 1

        // メモリ O(1) 制限: 過去 6 フレームより古い Mel フレームは因果参照範囲外なので刈り込み
        if 8 < state.pastMelFrames.count {
            state.pastMelFrames.removeFirst(state.pastMelFrames.count - 4)
        }

        let tConv1 = state.totalConv1Frames
        // Conv2 は時間 stride = 2 で発火 (奇数番目 Conv1 フレーム到着時: 1, 3, 5, ...)
        if (tConv1 % 2) != 1 {
            return nil
        }

        // Conv2 出力を計算
        let t2Idx = tConv1 / 2
        let c2Out = computeStreamingConv2Frame(
            t2Idx: t2Idx,
            frames: state.pastConv1Frames
        )

        if 8 < state.pastConv1Frames.count {
            state.pastConv1Frames.removeFirst(state.pastConv1Frames.count - 4)
        }

        // 線形射影 (256 -> outputDim)
        var projOut = [Float](repeating: 0.0, count: outputDim)
        let flatDim = 256
        projWeight.withUnsafeBufferPointer { wBuf in
            let wPtr = wBuf.baseAddress!
            projBias.withUnsafeBufferPointer { bBuf in
                let bPtr = bBuf.baseAddress!
                c2Out.withUnsafeBufferPointer { inBuf in
                    let inPtr = inBuf.baseAddress!
                    var d = 0
                    while d < outputDim {
                        let wRow = wPtr.advanced(by: d * flatDim)
                        let dot = VectorOperations.dotProduct(a: inPtr, b: wRow, count: flatDim)
                        projOut[d] = bPtr[d] + dot
                        d += 1
                    }
                }
            }
        }

        return projOut
    }

    private func computeStreamingConv1Frame(
        t1Idx: Int,
        frames: [[Float]]
    ) -> [Float] {
        let fOutCount = 32
        let cOutCount = 16
        var output = [Float](repeating: 0.0, count: fOutCount * cOutCount)
        let count = frames.count

        let b0 = conv1BiasSIMD0
        let b1 = conv1BiasSIMD1
        let w0 = conv1SIMD0
        let w1 = conv1SIMD1

        output.withUnsafeMutableBufferPointer { outBuf in
            let outPtr = outBuf.baseAddress!

            if 0 == t1Idx {
                frames[count - 1].withUnsafeBufferPointer { p2Buf in
                    let ptr2 = p2Buf.baseAddress!
                    computeConv1CellT0Boundary(ptr2: ptr2, b0: b0, b1: b1, w0: w0, w1: w1, dst: outPtr)
                    var f = 1
                    while f < fOutCount {
                        computeConv1CellT0Inner(ptr2: ptr2, f: f, b0: b0, b1: b1, w0: w0, w1: w1, dst: outPtr.advanced(by: f * cOutCount))
                        f += 1
                    }
                }
            } else {
                frames[count - 3].withUnsafeBufferPointer { p0Buf in
                    let ptr0 = p0Buf.baseAddress!
                    frames[count - 2].withUnsafeBufferPointer { p1Buf in
                        let ptr1 = p1Buf.baseAddress!
                        frames[count - 1].withUnsafeBufferPointer { p2Buf in
                            let ptr2 = p2Buf.baseAddress!

                            computeConv1CellTimeInnerBoundaryF0(
                                ptr0: ptr0,
                                ptr1: ptr1,
                                ptr2: ptr2,
                                b0: b0,
                                b1: b1,
                                w0: w0,
                                w1: w1,
                                dst: outPtr
                            )

                            var f = 1
                            while f < fOutCount {
                                computeConv1CellTimeInnerPure(
                                    ptr0: ptr0,
                                    ptr1: ptr1,
                                    ptr2: ptr2,
                                    f: f,
                                    b0: b0,
                                    b1: b1,
                                    w0: w0,
                                    w1: w1,
                                    dst: outPtr.advanced(by: f * cOutCount)
                                )
                                f += 1
                            }
                        }
                    }
                }
            }
        }

        return output
    }

    private func computeStreamingConv2Frame(
        t2Idx: Int,
        frames: [[Float]]
    ) -> [Float] {
        let f2Count = 16
        let c2Count = 16
        var output = [Float](repeating: 0.0, count: f2Count * c2Count)
        let count = frames.count

        conv2Weight.withUnsafeBufferPointer { wBuf in
            let wPtr = wBuf.baseAddress!
            conv2Bias.withUnsafeBufferPointer { bBuf in
                let bPtr = bBuf.baseAddress!
                output.withUnsafeMutableBufferPointer { outBuf in
                    let outPtr = outBuf.baseAddress!

                    if 0 == t2Idx {
                        frames[count - 1].withUnsafeBufferPointer { p2Buf in
                            let ptr2 = p2Buf.baseAddress!
                            computeConv2CellT0Boundary(ptr2: ptr2, wPtr: wPtr, bPtr: bPtr, dst: outPtr)
                            var f = 1
                            while f < f2Count {
                                computeConv2CellT0Inner(ptr2: ptr2, f: f, wPtr: wPtr, bPtr: bPtr, dst: outPtr.advanced(by: f * c2Count))
                                f += 1
                            }
                        }
                    } else {
                        frames[count - 3].withUnsafeBufferPointer { p0Buf in
                            let ptr0 = p0Buf.baseAddress!
                            frames[count - 2].withUnsafeBufferPointer { p1Buf in
                                let ptr1 = p1Buf.baseAddress!
                                frames[count - 1].withUnsafeBufferPointer { p2Buf in
                                    let ptr2 = p2Buf.baseAddress!

                                    computeConv2CellTimeInnerBoundaryF0(
                                        ptr0: ptr0,
                                        ptr1: ptr1,
                                        ptr2: ptr2,
                                        wPtr: wPtr,
                                        bPtr: bPtr,
                                        dst: outPtr
                                    )

                                    var f = 1
                                    while f < f2Count {
                                        computeConv2CellTimeInnerPure(
                                            ptr0: ptr0,
                                            ptr1: ptr1,
                                            ptr2: ptr2,
                                            f: f,
                                            wPtr: wPtr,
                                            bPtr: bPtr,
                                            dst: outPtr.advanced(by: f * c2Count)
                                        )
                                        f += 1
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        return output
    }
}
