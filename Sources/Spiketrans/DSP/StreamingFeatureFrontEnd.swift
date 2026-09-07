import Foundation

/// ホップ単位の音響特徴抽出。SNN へ渡すベクトル (既定 512 次元) をここで作る。
///
/// 処理は RMS ゲイン → プリエンファシス → フォルマント EQ 付き 64ch Mel →
/// 時間 3-tap (128 次元) → `frameStack` 本の連結。オフライン (`extractFeaturesFromPCM`)
/// はクリップ全体の RMS を `setGain` してからホップを流す。ストリーミングは未来が
/// 見えないので発話開始以降の走行 RMS を使う。Mel / 3-tap / 束ねの実装はここだけ。
/// 以前は `StreamingTranscriber` が 64 次元 raw Mel を渡しており、学習済み
/// `inputDim=512` の重みと一致しなかった。
///
/// ハミング窓は `DSPWorkspace` 既定の 1024 点テーブルを使い、先頭 `frameSize`
/// 点を掛ける。`frameSize` 点で窓を作り直すと学習時と形状が変わる。
public final class StreamingFeatureFrontEnd: @unchecked Sendable {
    public static let melChannels = 64
    /// 平滑 64 + 時間差分 64
    public static let tapDim = 128
    /// 10 ms hop を 4 本束ねて 40 ms。CTC には 10 ms が細かすぎ、逐次ステップが学習時間に効く
    public static let defaultStack = 4
    /// JSUT 相当の入力電流レンジへ揃える目標 RMS
    public static let targetRMS: Float = 0.05
    /// ほぼ無音のクリップでノイズだけを増幅しない上限
    public static let maxGain: Float = 20.0

    /// 3-tap 128 次元を `stack` 本連結した SNN 入力次元。既定 512。
    public static func acousticInputDim(stack: Int = defaultStack) -> Int {
        return tapDim * max(1, stack)
    }

    /// 目標 RMS 0.05 へ揃えるゲイン。無音で発散しないよう 20 倍で切る。
    public static func gainForRMS(_ rms: Float) -> Float {
        if 1e-6 < rms {
            let g = targetRMS / rms
            if maxGain < g {
                return maxGain
            }
            return g
        }
        return 1.0
    }

    public let frameStack: Int
    public let stackedDim: Int
    public let hopSize: Int
    public let frameSize: Int

    private let filterbank: Filterbank
    private let workspace: DSPWorkspace
    private let preemphasisCoeff: Float

    private var prevMel: [Float]
    private var currMel: [Float]
    private var tapFrame: [Float]
    private var stackBuf: [Float]
    private var preemphBuf: [Float]

    private var hasPrevMel: Bool = false
    private var hasCurrMel: Bool = false
    private var stackFill: Int = 0
    private var emittedStackCount: Int = 0
    private var utteranceFirstFrame: Bool = true
    private var streamRawPrev: Float = 0.0
    private var gain: Float = 1.0
    private var gainFrozen: Bool = false
    private var rmsSumSquares: Float = 0.0
    private var rmsSampleCount: Int = 0

    public init(frameStack: Int = defaultStack, dspConfig: DSPConfig = DSPConfig()) {
        let stack = max(1, frameStack)
        self.frameStack = stack
        self.stackedDim = Self.tapDim * stack
        self.hopSize = dspConfig.hopSize
        self.frameSize = dspConfig.frameSize
        self.preemphasisCoeff = dspConfig.preemphasisCoeff
        self.filterbank = Filterbank(config: dspConfig)
        self.workspace = DSPWorkspace(
            lpcOrder: dspConfig.lpcOrder,
            melChannels: Self.melChannels,
            maxPitchLag: dspConfig.maxPitchLag
        )
        self.prevMel = [Float](repeating: 0.0, count: Self.melChannels)
        self.currMel = [Float](repeating: 0.0, count: Self.melChannels)
        self.tapFrame = [Float](repeating: 0.0, count: Self.tapDim)
        self.stackBuf = [Float](repeating: 0.0, count: Self.tapDim * stack)
        self.preemphBuf = [Float](repeating: 0.0, count: dspConfig.frameSize)
    }

    /// ゲインを固定する。クリップ全体の RMS が先に分かるオフライン抽出用。
    /// 呼ぶと以降の `pushRawFrame` は走行 RMS を更新しない。
    public func setGain(_ g: Float) {
        gain = g
        gainFrozen = true
    }

    /// VAD 発話の境界で呼ぶ。3-tap と束ねバッファを捨て、隣接発話の Mel が混ざらないようにする。
    /// `setGain` 済みならゲインは維持し、未固定なら走行 RMS をやり直す。
    public func beginUtterance() {
        hasPrevMel = false
        hasCurrMel = false
        stackFill = 0
        emittedStackCount = 0
        utteranceFirstFrame = true
        streamRawPrev = 0.0
        if gainFrozen != true {
            gain = 1.0
            rmsSumSquares = 0.0
            rmsSampleCount = 0
        }
        var i = 0
        while i < stackBuf.count {
            stackBuf[i] = 0.0
            i += 1
        }
    }

    /// 1 ホップ分の raw フレームを入れる。3-tap は次ホップが揃うまで出さず、
    /// `frameStack` 本たまると `stackedDim` のベクトルを返す。戻り値は内部バッファで、
    /// 次の push / flush で上書きされる。
    public func pushRawFrame(pcmPtr: UnsafePointer<Float>, count: Int) -> [Float]? {
        if count < frameSize {
            return nil
        }
        updateGain(pcmPtr: pcmPtr, count: count)
        applyPreemphasis(pcmPtr: pcmPtr, count: count)
        if ingestMelFromPreemph() != true {
            return nil
        }
        return pushTapIntoStack()
    }

    /// 発話末の 3-tap を、オフライン末尾と同じく next=curr で確定する。
    /// 束ねが 1 本も出ていなければゼロ埋めで 1 本出す。それ以外の端数は捨てる
    /// (`extractFeaturesFromPCM` の `count / frameStack` と同じ)。
    public func flush() -> [Float]? {
        if hasCurrMel != true {
            return nil
        }
        if hasPrevMel != true {
            writeThreeTap(prev: currMel, curr: currMel, next: currMel)
        } else {
            writeThreeTap(prev: prevMel, curr: currMel, next: currMel)
        }
        hasCurrMel = false
        hasPrevMel = false
        if let stacked = pushTapIntoStack() {
            return stacked
        }
        if 0 < stackFill && emittedStackCount == 0 {
            var i = stackFill * Self.tapDim
            while i < stackBuf.count {
                stackBuf[i] = 0.0
                i += 1
            }
            stackFill = 0
            emittedStackCount += 1
            return stackBuf
        }
        return nil
    }

    private func updateGain(pcmPtr: UnsafePointer<Float>, count: Int) {
        if gainFrozen {
            return
        }
        if utteranceFirstFrame {
            var i = 0
            while i < count {
                let x = pcmPtr[i]
                rmsSumSquares += x * x
                i += 1
            }
            rmsSampleCount += count
        } else {
            var addCount = hopSize
            if count < hopSize {
                addCount = count
            }
            let start = count - addCount
            var src = start
            while src < count {
                let x = pcmPtr[src]
                rmsSumSquares += x * x
                src += 1
            }
            rmsSampleCount += addCount
        }
        var rms: Float = 0.0
        if 0 < rmsSampleCount {
            rms = sqrtf(rmsSumSquares / Float(rmsSampleCount))
        }
        gain = Self.gainForRMS(rms)
    }

    private func applyPreemphasis(pcmPtr: UnsafePointer<Float>, count: Int) {
        let coeff = preemphasisCoeff
        if utteranceFirstFrame {
            preemphBuf[0] = pcmPtr[0] * gain
            var i = 1
            while i < count {
                preemphBuf[i] = (pcmPtr[i] - (coeff * pcmPtr[i - 1])) * gain
                i += 1
            }
            utteranceFirstFrame = false
        } else {
            preemphBuf[0] = (pcmPtr[0] - (coeff * streamRawPrev)) * gain
            var i = 1
            while i < count {
                preemphBuf[i] = (pcmPtr[i] - (coeff * pcmPtr[i - 1])) * gain
                i += 1
            }
        }
        streamRawPrev = pcmPtr[hopSize - 1]
    }

    private func ingestMelFromPreemph() -> Bool {
        let mel = preemphBuf.withUnsafeBufferPointer { buf in
            return filterbank.extractFeatures(
                pcmPtr: buf.baseAddress!,
                count: frameSize,
                workspace: workspace
            )
        }
        if hasCurrMel != true {
            currMel = mel
            hasCurrMel = true
            return false
        }
        if hasPrevMel != true {
            writeThreeTap(prev: currMel, curr: currMel, next: mel)
            prevMel = currMel
            currMel = mel
            hasPrevMel = true
            return true
        }
        writeThreeTap(prev: prevMel, curr: currMel, next: mel)
        prevMel = currMel
        currMel = mel
        return true
    }

    private func writeThreeTap(prev: [Float], curr: [Float], next: [Float]) {
        var c = 0
        while c < Self.melChannels {
            let xPrev = prev[c]
            let xCurr = curr[c]
            let xNext = next[c]
            tapFrame[c] = (0.25 * xPrev) + (0.5 * xCurr) + (0.25 * xNext)
            tapFrame[Self.melChannels + c] = 0.5 * (xNext - xPrev)
            c += 1
        }
    }

    private func pushTapIntoStack() -> [Float]? {
        let dim = Self.tapDim
        let offset = stackFill * dim
        var d = 0
        while d < dim {
            stackBuf[offset + d] = tapFrame[d]
            d += 1
        }
        stackFill += 1
        if stackFill < frameStack {
            return nil
        }
        stackFill = 0
        emittedStackCount += 1
        return stackBuf
    }
}
