import Foundation

public struct PitchDetector: Sendable {
    public let config: DSPConfig
    
    public init(config: DSPConfig = DSPConfig()) {
        self.config = config
    }
    
    /// フレームから F0 (Hz) および HNR (dB) を抽出
    @inline(__always)
    public func detectPitch(
        ptr: UnsafePointer<Float>,
        count: Int,
        workspace: DSPWorkspace
    ) -> PitchResult {
        if count < config.maxPitchLag + 8 {
            return PitchResult(f0: 0.0, hnr: 0.0, isVoiced: false)
        }
        
        let frameSize = count
        let clipPtr = workspace.clippedFrame.withUnsafeMutableBufferPointer { $0.baseAddress! }
        
        // 1. センタークリッピング (Sondhi 法: 生信号から算出)
        let maxAbs = VectorOperations.maxMagnitude(ptr: ptr, count: frameSize)
        if maxAbs < 1e-4 {
            return PitchResult(f0: 0.0, hnr: 0.0, isVoiced: false)
        }
        
        let cl = 0.3 * maxAbs
        let negCl = -1.0 * cl
        
        var i = 0
        let width = 8
        let limit = frameSize - (frameSize % width)
        let clVec = SIMD8<Float>(repeating: cl)
        let negClVec = SIMD8<Float>(repeating: negCl)
        
        while i < limit {
            let v = SIMD8<Float>(
                ptr[i+0], ptr[i+1], ptr[i+2], ptr[i+3],
                ptr[i+4], ptr[i+5], ptr[i+6], ptr[i+7]
            )
            var clipped = SIMD8<Float>(repeating: 0.0)
            let posMask = clVec .< v
            let negMask = v .< negClVec
            
            clipped = clipped.replacing(with: v - clVec, where: posMask)
            clipped = clipped.replacing(with: v + clVec, where: negMask)
            
            clipPtr[i+0] = clipped[0]
            clipPtr[i+1] = clipped[1]
            clipPtr[i+2] = clipped[2]
            clipPtr[i+3] = clipped[3]
            clipPtr[i+4] = clipped[4]
            clipPtr[i+5] = clipped[5]
            clipPtr[i+6] = clipped[6]
            clipPtr[i+7] = clipped[7]
            i += width
        }
        while i < frameSize {
            let val = ptr[i]
            var c: Float = 0.0
            if cl < val {
                c = val - cl
            }
            if val < negCl {
                c = val + cl
            }
            clipPtr[i] = c
            i += 1
        }
        
        // 2. 原信号エネルギーおよび正規化自己相関の計算
        let calcLength = frameSize - config.maxPitchLag
        let e0 = VectorOperations.sumOfSquares(ptr: ptr, count: calcLength)
        if e0 < 1e-6 {
            return PitchResult(f0: 0.0, hnr: 0.0, isVoiced: false)
        }
        
        var lag = config.minPitchLag
        let maxLag = config.maxPitchLag
        let rValues = workspace.pitchAutocorr.withUnsafeMutableBufferPointer { $0.baseAddress! }
        
        while lag <= maxLag {
            let rawLagPtr = ptr.advanced(by: lag)
            let dotRaw = VectorOperations.dotProduct(a: ptr, b: rawLagPtr, count: calcLength)
            let eLag = VectorOperations.sumOfSquares(ptr: rawLagPtr, count: calcLength)
            let denom = sqrt(e0 * eLag)
            var normR: Float = 0.0
            if 1e-6 <= denom {
                normR = dotRaw / denom
            }
            rValues[lag] = normR
            lag += 1
        }
        
        // 3. 極大値の抽出とオクターブエラー抑制 (workspace バッファ使用)
        var globalMaxR: Float = -Float.greatestFiniteMagnitude
        let peaksLag = workspace.pitchPeaksLag.withUnsafeMutableBufferPointer { $0.baseAddress! }
        let peaksR = workspace.pitchPeaksR.withUnsafeMutableBufferPointer { $0.baseAddress! }
        var peakCount = 0
        
        lag = config.minPitchLag + 1
        while lag < maxLag {
            let prevR = rValues[lag-1]
            let currR = rValues[lag]
            let nextR = rValues[lag+1]
            
            if prevR < currR && nextR < currR {
                peaksLag[peakCount] = lag
                peaksR[peakCount] = currR
                peakCount += 1
                if globalMaxR < currR {
                    globalMaxR = currR
                }
            }
            lag += 1
        }
        
        // 有声音の正規化自己相関しきい値 (0.65 以上)
        if peakCount == 0 || globalMaxR < 0.65 {
            return PitchResult(f0: 0.0, hnr: 0.0, isVoiced: false)
        }
        
        // 最大ピーク強度の 80% 以上の強度を持つ最小ラグを選択 (オクターブエラー抑制)
        var bestLag = -1
        var bestNormR: Float = 0.0
        let threshold = 0.8 * globalMaxR
        
        var pIdx = 0
        while pIdx < peakCount {
            let r = peaksR[pIdx]
            if threshold <= r {
                bestLag = peaksLag[pIdx]
                bestNormR = r
                break
            }
            pIdx += 1
        }
        
        if bestLag == -1 {
            return PitchResult(f0: 0.0, hnr: 0.0, isVoiced: false)
        }
        
        let f0 = Float(config.sampleRate) / Float(bestLag)
        var hnr: Float = 40.0
        if bestNormR < 0.999 {
            let noiseFrac = 1.0 - bestNormR
            if 1e-5 <= noiseFrac {
                let ratio = bestNormR / noiseFrac
                if 1e-5 <= ratio {
                    hnr = min(40.0, max(0.0, 10.0 * log10(ratio)))
                }
            }
        }
        
        return PitchResult(f0: f0, hnr: hnr, isVoiced: true)
    }
}
