import Foundation

/// ホップ単位の音響特徴抽出。SNN 入力ベクトルを作る。
///
/// オフラインはクリップ RMS を `setGain` してからホップを流す。ストリーミングは
/// 発話開始以降の走行 RMS を使う (未来のサンプルが見えない)。ハミング窓は
/// `DSPWorkspace` 既定の 1024 点テーブル。`frameSize` 点で作り直すと学習時と形状が変わる。
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
    /// 長い文脈 (背景音の要約) の次元。平滑メルの指数移動平均 64
    public static let contextDim = 64
    /// 背景音の要約の時定数 (秒)。1 ホップ 10 ms ごとに 1 - exp(-0.01 / τ) だけ追従する
    public static let contextTimeConstant: Float = 1.0

    @inline(__always)
    public static func acousticInputDim(stack: Int = defaultStack, longContext: Bool = false) -> Int {
        var dim = tapDim * max(1, stack)
        if longContext {
            dim += contextDim
        }
        return dim
    }

    /// ネットワークの入力次元から束ね数を求める (背景音の要約の有無によらない)
    public static func frameStack(forInputDim inputDim: Int) -> Int {
        return max(1, inputDim / tapDim)
    }

    /// ネットワークの入力次元が背景音の要約を含むか
    public static func hasLongContext(inputDim: Int) -> Bool {
        return (inputDim % tapDim) == contextDim
    }

    @inline(__always)
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
    public let longContext: Bool
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
    /// 平滑メルの指数移動平均 (背景音の要約)。longContext のときだけ使う
    private var contextMel: [Float]
    private var hasContext: Bool = false
    private let contextRate: Float

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

    public init(frameStack: Int = defaultStack, longContext: Bool = false, dspConfig: DSPConfig = DSPConfig()) {
        let stack = max(1, frameStack)
        self.frameStack = stack
        self.longContext = longContext
        self.stackedDim = Self.acousticInputDim(stack: stack, longContext: longContext)
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
        self.stackBuf = [Float](repeating: 0.0, count: Self.acousticInputDim(stack: stack, longContext: longContext))
        self.preemphBuf = [Float](repeating: 0.0, count: dspConfig.frameSize)
        self.contextMel = [Float](repeating: 0.0, count: Self.melChannels)
        let hopSeconds = Float(dspConfig.hopSize) / Float(dspConfig.sampleRate)
        self.contextRate = 1.0 - expf(-hopSeconds / Self.contextTimeConstant)
    }

    /// オフライン用。クリップ RMS が先に分かるとき、以降の走行 RMS 更新を止める。
    public func setGain(_ g: Float) {
        gain = g
        gainFrozen = true
    }

    /// 発話境界。3-tap / 束ねを捨てる。`setGain` 済みならゲインは維持する。
    public func beginUtterance() {
        hasPrevMel = false
        hasCurrMel = false
        stackFill = 0
        emittedStackCount = 0
        utteranceFirstFrame = true
        streamRawPrev = 0.0
        hasContext = false
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

    /// 1 ホップ入れる。`frameStack` 本そろうまで nil。戻り値は次の push / flush で上書きされる。
    @inline(__always)
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

    /// 末尾 3-tap を next=curr で確定する。未出力の束ねはゼロ埋め 1 本、それ以外の端数は捨てる。
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
            while i < Self.tapDim * frameStack {
                stackBuf[i] = 0.0
                i += 1
            }
            writeContext()
            stackFill = 0
            emittedStackCount += 1
            return stackBuf
        }
        return nil
    }

    @inline(__always)
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

    @inline(__always)
    private func applyPreemphasis(pcmPtr: UnsafePointer<Float>, count: Int) {
        let coeff = preemphasisCoeff
        if utteranceFirstFrame {
            preemphBuf[0] = pcmPtr[0] * gain
            utteranceFirstFrame = false
        } else {
            preemphBuf[0] = (pcmPtr[0] - (coeff * streamRawPrev)) * gain
        }
        var i = 1
        while i < count {
            preemphBuf[i] = (pcmPtr[i] - (coeff * pcmPtr[i - 1])) * gain
            i += 1
        }
        streamRawPrev = pcmPtr[hopSize - 1]
    }

    @inline(__always)
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

    @inline(__always)
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

    @inline(__always)
    private func pushTapIntoStack() -> [Float]? {
        let dim = Self.tapDim
        let offset = stackFill * dim
        var d = 0
        while d < dim {
            stackBuf[offset + d] = tapFrame[d]
            d += 1
        }
        if longContext {
            updateContext()
        }
        stackFill += 1
        if stackFill < frameStack {
            return nil
        }
        stackFill = 0
        emittedStackCount += 1
        writeContext()
        return stackBuf
    }

    /// 平滑メルを指数移動平均に取り込む。発話の最初のホップで初期化する
    @inline(__always)
    private func updateContext() {
        let rate = contextRate
        var c = 0
        if hasContext != true {
            while c < Self.melChannels {
                contextMel[c] = tapFrame[c]
                c += 1
            }
            hasContext = true
            return
        }
        while c < Self.melChannels {
            contextMel[c] += rate * (tapFrame[c] - contextMel[c])
            c += 1
        }
    }

    /// 束ねた末尾に背景音の要約を書く
    @inline(__always)
    private func writeContext() {
        if longContext != true {
            return
        }
        let offset = Self.tapDim * frameStack
        var c = 0
        while c < Self.melChannels {
            stackBuf[offset + c] = contextMel[c]
            c += 1
        }
    }
}
