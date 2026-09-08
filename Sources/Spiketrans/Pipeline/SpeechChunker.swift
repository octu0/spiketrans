import Foundation

/// 長い音声を発話単位の区間に切る。VAD の無音で区切り、短い間は結合し、
/// 長すぎる区間は内部でいちばん静かな窓で割る。`segment` (学習データ作成) と
/// `transcribe` (長時間文字起こし) が同じ切り方を使う。
///
/// 12 時間の音声でも入力配列のコピーを作らないよう、VAD にはフレーム単位で
/// ゲインを掛けたバッファを渡す (VAD は発話単位の RMS 正規化を前提にしている)
public struct SpeechChunker: Sendable {
    /// 区間を結合する無音の上限 (秒)。これより短い間は同じ発話とみなす
    public let mergeGapSeconds: Float
    /// 単独では短すぎて捨てる区間 (秒)
    public let minSegmentSeconds: Float
    /// 1 区間の上限 (秒)。超える区間は内部の最小エネルギー窓で割る
    public let maxSegmentSeconds: Float
    public let dspConfig: DSPConfig

    public struct Span: Sendable, Equatable {
        public var start: Int
        public var end: Int
        public init(start: Int, end: Int) {
            self.start = start
            self.end = end
        }
    }

    public init(
        mergeGapSeconds: Float = 0.35,
        minSegmentSeconds: Float = 0.6,
        maxSegmentSeconds: Float = 15.0,
        dspConfig: DSPConfig = DSPConfig()
    ) {
        self.mergeGapSeconds = mergeGapSeconds
        self.minSegmentSeconds = minSegmentSeconds
        self.maxSegmentSeconds = max(2.0, maxSegmentSeconds)
        self.dspConfig = dspConfig
    }

    /// 返す index は入力 `pcm` に対するもの
    public func chunk(pcm: [Float]) -> [Span] {
        let sampleRate = Float(dspConfig.sampleRate)
        let frameSize = dspConfig.frameSize
        let hopSize = dspConfig.hopSize
        if pcm.count < frameSize {
            return []
        }

        var sumSq: Float = 0.0
        var i = 0
        while i < pcm.count {
            sumSq += pcm[i] * pcm[i]
            i += 1
        }
        let gain = StreamingFeatureFrontEnd.gainForRMS(sqrt(sumSq / Float(pcm.count)))

        let vad = VAD(config: dspConfig)
        let workspace = DSPWorkspace(lpcOrder: dspConfig.lpcOrder, melChannels: dspConfig.melChannels)
        var frame = [Float](repeating: 0.0, count: frameSize)

        // VAD.segmentUtterances と同じ状態機械。フレームごとにゲインを掛けるため自前で回す
        let minSpeechFrames = 5
        let hangoverFrames = 20
        let rollSamples = Int(0.15 * sampleRate)
        var merged: [Span] = []
        var state = VAD.State.silence
        var speechStart = 0

        func close(at offset: Int) {
            let end = min(pcm.count, offset + frameSize + rollSamples)
            if let last = merged.last, Float(speechStart - last.end) / sampleRate < mergeGapSeconds {
                merged[merged.count - 1].end = max(last.end, end)
            } else {
                merged.append(Span(start: speechStart, end: end))
            }
        }

        pcm.withUnsafeBufferPointer { src in
            frame.withUnsafeMutableBufferPointer { dst in
                let base = src.baseAddress!
                let buf = dst.baseAddress!
                var offset = 0
                while (offset + frameSize) <= pcm.count {
                    var k = 0
                    while k < frameSize {
                        buf[k] = base[offset + k] * gain
                        k += 1
                    }
                    let isSpeech = vad.processFrame(ptr: buf, count: frameSize, workspace: workspace).isSpeech
                    switch state {
                    case .silence:
                        if isSpeech {
                            state = .speechTriggered(triggerCount: 1)
                            speechStart = max(0, offset - rollSamples)
                        }
                    case .speechTriggered(let count):
                        if isSpeech {
                            if minSpeechFrames <= (count + 1) {
                                state = .activeSpeech(speechCount: count + 1)
                            } else {
                                state = .speechTriggered(triggerCount: count + 1)
                            }
                        } else {
                            state = .silence
                        }
                    case .activeSpeech(let count):
                        if isSpeech {
                            state = .activeSpeech(speechCount: count + 1)
                        } else {
                            state = .hangover(hangoverCount: 1)
                        }
                    case .hangover(let count):
                        if isSpeech {
                            state = .activeSpeech(speechCount: count + 1)
                        } else {
                            if hangoverFrames <= (count + 1) {
                                close(at: offset)
                                state = .silence
                            } else {
                                state = .hangover(hangoverCount: count + 1)
                            }
                        }
                    }
                    offset += hopSize
                }
                switch state {
                case .activeSpeech, .hangover:
                    close(at: pcm.count)
                default:
                    break
                }
            }
        }

        var result: [Span] = []
        var queue = merged
        while queue.isEmpty != true {
            let span = queue.removeFirst()
            let seconds = Float(span.end - span.start) / sampleRate
            if maxSegmentSeconds < seconds {
                let cut = quietestSplit(span, in: pcm, sampleRate: sampleRate)
                if span.start < cut && cut < span.end {
                    queue.insert(Span(start: cut, end: span.end), at: 0)
                    queue.insert(Span(start: span.start, end: cut), at: 0)
                    continue
                }
            }
            if minSegmentSeconds <= seconds {
                result.append(span)
            }
        }
        return result
    }

    /// 端から 1 秒以内を除き、200 ms 窓のエネルギーが最小の位置。エネルギー比較なのでゲインは不要
    private func quietestSplit(_ span: Span, in samples: [Float], sampleRate: Float) -> Int {
        let window = Int(0.2 * sampleRate)
        let hop = Int(0.05 * sampleRate)
        let margin = Int(1.0 * sampleRate)
        var best = (span.start + span.end) / 2
        var bestEnergy = Float.greatestFiniteMagnitude
        var pos = span.start + margin
        while (pos + window) <= (span.end - margin) {
            var e: Float = 0.0
            var k = pos
            while k < pos + window {
                e += samples[k] * samples[k]
                k += 1
            }
            if e < bestEnergy {
                bestEnergy = e
                best = pos + window / 2
            }
            pos += hop
        }
        return best
    }
}
