import Foundation

public final class VAD: @unchecked Sendable {
    public let config: DSPConfig
    private var noiseFloorEnergy: Float = 1e-4
    private let noiseFloorAlpha: Float = 0.05
    
    // ステートマシン状態
    public enum State: Sendable {
        case silence
        case speechTriggered(triggerCount: Int)
        case activeSpeech(speechCount: Int)
        case hangover(hangoverCount: Int)
    }
    
    public init(config: DSPConfig = DSPConfig()) {
        self.config = config
    }
    
    /// フレームごとの多次元 VAD 判定
    @discardableResult
    @inline(__always)
    public func processFrame(
        ptr: UnsafePointer<Float>,
        count: Int,
        workspace: DSPWorkspace
    ) -> VADResult {
        if count < 8 {
            return VADResult(isSpeech: false, rms: 0.0, zcr: 0.0, voicingRatio: 0.0, noiseFloor: noiseFloorEnergy)
        }
        
        // 1. RMS の計算
        let sumSq = VectorOperations.sumOfSquares(ptr: ptr, count: count)
        let frameEnergy = sumSq / Float(count)
        let rms = sqrt(frameEnergy)
        
        // 2. ゼロ交差率 (ZCR)
        var zcrCount = 0
        var i = 1
        while i < count {
            let prev = ptr[i-1]
            let curr = ptr[i]
            if (prev < 0.0 && 0.0 <= curr) || (0.0 <= prev && curr < 0.0) {
                zcrCount += 1
            }
            i += 1
        }
        let zcr = Float(zcrCount) / Float(count - 1)
        
        // 3. 自己相関有声度 (Voicing Ratio)
        let calcLen = max(8, count - config.maxPitchLag)
        let e0 = VectorOperations.sumOfSquares(ptr: ptr, count: calcLen)
        var maxNormR: Float = 0.0
        
        if 1e-6 <= e0 {
            var lag = config.minPitchLag
            let maxLag = config.maxPitchLag
            while lag <= maxLag {
                if (lag + calcLen) <= count {
                    let lagPtr = ptr.advanced(by: lag)
                    let dot = VectorOperations.dotProduct(a: ptr, b: lagPtr, count: calcLen)
                    let eLag = VectorOperations.sumOfSquares(ptr: lagPtr, count: calcLen)
                    let denom = sqrt(e0 * eLag)
                    if 1e-6 <= denom {
                        let normR = dot / denom
                        if maxNormR < normR {
                            maxNormR = normR
                        }
                    }
                }
                lag += 1
            }
        }
        let voicingRatio = maxNormR
        
        // 4. 動的しきい値判定
        let energyThreshold = max(1e-4, noiseFloorEnergy * config.vadEnergyThresholdRatio)
        let isEnergyActive = energyThreshold <= frameEnergy
        let isVoiced = config.vadVoicingRatioThreshold <= voicingRatio
        let isFricative = (config.vadZcrThreshold <= zcr) && (0.005 <= rms)
        
        let isSpeech = isEnergyActive && (isVoiced || isFricative)
        
        // 5. 背景ノイズフロア EMA 更新 (非発話時)
        if isSpeech != true {
            noiseFloorEnergy = ((1.0 - noiseFloorAlpha) * noiseFloorEnergy) + (noiseFloorAlpha * frameEnergy)
        }
        
        return VADResult(
            isSpeech: isSpeech,
            rms: rms,
            zcr: zcr,
            voicingRatio: voicingRatio,
            noiseFloor: noiseFloorEnergy
        )
    }
    
    /// 長時間音声全体のセグメンテーション
    public func segmentUtterances(
        pcmData: [Float],
        workspace: DSPWorkspace
    ) -> [SpeechSegment] {
        let totalSamples = pcmData.count
        let frameSize = config.frameSize
        let hopSize = config.hopSize
        if totalSamples < frameSize {
            return []
        }
        
        var segments: [SpeechSegment] = []
        var state: State = .silence
        var speechStartIndex = 0
        var offset = 0
        
        let minSpeechFrames = 5 // 50ms
        let hangoverFrames = 20 // 200ms
        let preRollSamples = 2400 // 150ms (16kHz)
        let postRollSamples = 2400 // 150ms
        
        pcmData.withUnsafeBufferPointer { pcmPtr in
            let base = pcmPtr.baseAddress!
            
            while (offset + frameSize) <= totalSamples {
                let framePtr = base.advanced(by: offset)
                let result = processFrame(ptr: framePtr, count: frameSize, workspace: workspace)
                
                switch state {
                case .silence:
                    if result.isSpeech {
                        state = .speechTriggered(triggerCount: 1)
                        speechStartIndex = max(0, offset - preRollSamples)
                    }
                case .speechTriggered(let count):
                    if result.isSpeech {
                        if minSpeechFrames <= (count + 1) {
                            state = .activeSpeech(speechCount: count + 1)
                        } else {
                            state = .speechTriggered(triggerCount: count + 1)
                        }
                    } else {
                        state = .silence
                    }
                case .activeSpeech(let count):
                    if result.isSpeech {
                        state = .activeSpeech(speechCount: count + 1)
                    } else {
                        state = .hangover(hangoverCount: 1)
                    }
                case .hangover(let count):
                    if result.isSpeech {
                        state = .activeSpeech(speechCount: count + 1)
                    } else {
                        if hangoverFrames <= (count + 1) {
                            let endIndex = min(totalSamples, offset + frameSize + postRollSamples)
                            let duration = Float(endIndex - speechStartIndex) / Float(config.sampleRate)
                            segments.append(SpeechSegment(
                                startIndex: speechStartIndex,
                                endIndex: endIndex,
                                durationSeconds: duration
                            ))
                            state = .silence
                        } else {
                            state = .hangover(hangoverCount: count + 1)
                        }
                    }
                }
                
                offset += hopSize
            }
            
            // 終端処理
            switch state {
            case .activeSpeech, .hangover:
                let endIndex = totalSamples
                let duration = Float(endIndex - speechStartIndex) / Float(config.sampleRate)
                segments.append(SpeechSegment(
                    startIndex: speechStartIndex,
                    endIndex: endIndex,
                    durationSeconds: duration
                ))
            default:
                break
            }
        }
        
        return segments
    }
}
