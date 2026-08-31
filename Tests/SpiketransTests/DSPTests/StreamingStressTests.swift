import XCTest
@testable import Spiketrans
#if canImport(Darwin)
import Darwin
#endif

final class StreamingStressTests: XCTestCase {
    
    private func getResidentMemoryBytes() -> UInt64 {
        #if canImport(Darwin)
        var taskInfo = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
        let kerr: kern_return_t = withUnsafeMutablePointer(to: &taskInfo) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        if kerr == KERN_SUCCESS {
            return taskInfo.resident_size
        }
        return 0
        #else
        return 0
        #endif
    }
    
    /// 10,000 フレーム（約200秒分）の連続 PCM ストリーミング処理およびメモリリーク検証
    func testTenThousandFramesStreamingMemoryLeak() {
        let config = DSPConfig(sampleRate: 16000, frameSize: 320, hopSize: 160)
        let workspace = DSPWorkspace(maxFrameSize: 1024, lpcOrder: config.lpcOrder, melChannels: config.melChannels)
        let vad = VAD(config: config)
        let pitchDetector = PitchDetector(config: config)
        let lpc = LPC(config: config)
        let solver = DurandKernerSolver()
        let formantExtractor = FormantExtractor(sampleRate: Float(config.sampleRate))
        let filterbank = Filterbank(config: config)
        
        let frameSize = config.frameSize
        let totalFrames = 10000
        let totalAudioSeconds = Float(totalFrames * config.hopSize) / Float(config.sampleRate)
        
        // 信号生成バッファ (変化する周波数とノイズを伴う音声シミュレーション)
        var syntheticBuffer = [Float](repeating: 0.0, count: frameSize)
        
        var initialRss: UInt64 = 0
        var midRss: UInt64 = 0
        var finalRss: UInt64 = 0
        
        let startTime = CFAbsoluteTimeGetCurrent()
        
        var frameIdx = 0
        while frameIdx < totalFrames {
            // フレーム波形の生成 (サイン波 + 倍音 + 微小ノイズ)
            let f0: Float = 120.0 + (100.0 * sin(Float(frameIdx) * 0.01))
            let factor = (2.0 * Float.pi * f0) / 16000.0
            
            var s = 0
            while s < frameSize {
                let sampleTime = Float(s) * factor
                syntheticBuffer[s] = (0.4 * sin(sampleTime)) + (0.2 * sin(sampleTime * 2.0)) + (0.01 * sin(sampleTime * 5.0))
                s += 1
            }
            
            // DSP パイプライン実行
            syntheticBuffer.withUnsafeBufferPointer { pcmPtr in
                let framePtr = pcmPtr.baseAddress!
                
                vad.processFrame(ptr: framePtr, count: frameSize, workspace: workspace)
                let pitchRes = pitchDetector.detectPitch(ptr: framePtr, count: frameSize, workspace: workspace)
                let lpcSuccess = lpc.computeCoefficients(ptr: framePtr, count: frameSize, workspace: workspace)
                
                var formantRes = FormantResult(f1: 0.0, f2: 0.0, f3: 0.0, b1: 0.0, b2: 0.0, b3: 0.0, count: 0)
                if lpcSuccess {
                    workspace.lpcCoeffs.withUnsafeBufferPointer { cPtr in
                        let solverSuccess = solver.solve(coefficients: cPtr.baseAddress!, order: config.lpcOrder, workspace: workspace)
                        if solverSuccess {
                            workspace.durandKernerCurr.withUnsafeBufferPointer { rPtr in
                                formantRes = formantExtractor.extractFormants(roots: rPtr.baseAddress!, count: config.lpcOrder)
                            }
                        }
                    }
                }
                
                filterbank.extractFeatures(
                    pcmPtr: framePtr,
                    count: frameSize,
                    workspace: workspace
                )
            }
            
            // 2000フレーム時点（ウォームアップ完了後）のメモリ
            if frameIdx == 2000 {
                initialRss = getResidentMemoryBytes()
            }
            // 5000フレーム時点のメモリ
            if frameIdx == 5000 {
                midRss = getResidentMemoryBytes()
            }
            
            frameIdx += 1
        }
        
        finalRss = getResidentMemoryBytes()
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        let rtf = Float(elapsed) / totalAudioSeconds
        
        print("--- 10,000 Frames Streaming Test Result ---")
        print("Total frames: \(totalFrames)")
        print("Simulated audio: \(totalAudioSeconds) s")
        print("Elapsed time: \(String(format: "%.4f", elapsed)) s")
        print("Real-Time Factor (RTF): \(String(format: "%.6f", rtf)) xRT")
        print("RSS at frame 2,000: \(Double(initialRss) / (1024.0 * 1024.0)) MB")
        print("RSS at frame 5,000: \(Double(midRss) / (1024.0 * 1024.0)) MB")
        print("RSS at frame 10,000: \(Double(finalRss) / (1024.0 * 1024.0)) MB")
        
        // メモリリーク検証: 2000フレームから10000フレームの間でメモリ肥大化がないこと
        if 0 < initialRss {
            let growthBytes = Int64(finalRss) - Int64(initialRss)
            let growthMB = Double(growthBytes) / (1024.0 * 1024.0)
            print("Memory growth (frame 2000 to 10000): \(growthMB) MB")
            // 許容誤差: OSのページテーブル管理やスレッドスタックマッピングによる変動を考慮し 5.0MB 未満
            XCTAssertLessThan(growthMB, 5.0, "Excessive memory growth detected over 10,000 frames: \(growthMB) MB")
        }
        
        // RTF 検証: Debug ビルドでも一定以上のスループットがあること
        XCTAssertLessThan(rtf, 0.1, "RTF is too high: \(rtf)")
    }
    
    /// 多様な過酷シナリオ（無音、急峻クリッピング、チャープ周波数スイープ、高雑音、急激な有声/無声遷移）での安定性テスト
    func testDiverseAudioAdversarialStress() {
        let config = DSPConfig(sampleRate: 16000, frameSize: 320, hopSize: 160)
        let workspace = DSPWorkspace(maxFrameSize: 1024, lpcOrder: config.lpcOrder, melChannels: config.melChannels)
        let vad = VAD(config: config)
        let pitchDetector = PitchDetector(config: config)
        let lpc = LPC(config: config)
        let solver = DurandKernerSolver()
        let formantExtractor = FormantExtractor(sampleRate: Float(config.sampleRate))
        let filterbank = Filterbank(config: config)
        
        let frameSize = config.frameSize
        let patternCount = 5
        let framesPerPattern = 500
        
        var pattern = 0
        while pattern < patternCount {
            var f = 0
            while f < framesPerPattern {
                var frame = [Float](repeating: 0.0, count: frameSize)
                
                switch pattern {
                case 0:
                    // 完全な無音
                    break
                case 1:
                    // 最大振幅クリッピング波形 (±1.0 矩形波)
                    var i = 0
                    while i < frameSize {
                        var v: Float = -0.999
                        if i % 20 < 10 {
                            v = 0.999
                        }
                        frame[i] = v
                        i += 1
                    }
                case 2:
                    // チャープ信号 (50Hz -> 4000Hz 超高速周波数スイープ)
                    let startFreq: Float = 50.0
                    let endFreq: Float = 4000.0
                    let currentFreq = startFreq + ((endFreq - startFreq) * (Float(f) / Float(framesPerPattern)))
                    let factor = (2.0 * Float.pi * currentFreq) / 16000.0
                    var i = 0
                    while i < frameSize {
                        frame[i] = 0.7 * sin(Float(i) * factor)
                        i += 1
                    }
                case 3:
                    // 高エネルギー白色雑音
                    var rng: UInt64 = UInt64(pattern * 1000 + f + 1)
                    var i = 0
                    while i < frameSize {
                        rng ^= rng << 13
                        rng ^= rng >> 7
                        rng ^= rng << 17
                        let uVal = Float(rng & 0xFFFFFF) / Float(0xFFFFFF)
                        frame[i] = (uVal - 0.5) * 1.5
                        i += 1
                    }
                case 4:
                    // 極微小信号 (1e-6)
                    var i = 0
                    while i < frameSize {
                        frame[i] = 1e-6 * sin(Float(i) * 0.1)
                        i += 1
                    }
                default:
                    break
                }
                
                // パイプライン実行（クラッシュ・NaN・Infが発生しないこと）
                frame.withUnsafeBufferPointer { pcmPtr in
                    let framePtr = pcmPtr.baseAddress!
                    
                    let vadRes = vad.processFrame(ptr: framePtr, count: frameSize, workspace: workspace)
                    XCTAssertFalse(vadRes.rms.isNaN)
                    XCTAssertFalse(vadRes.zcr.isNaN)
                    
                    let pitchRes = pitchDetector.detectPitch(ptr: framePtr, count: frameSize, workspace: workspace)
                    XCTAssertFalse(pitchRes.f0.isNaN)
                    XCTAssertFalse(pitchRes.hnr.isNaN)
                    
                    let lpcSuccess = lpc.computeCoefficients(ptr: framePtr, count: frameSize, workspace: workspace)
                    var formantRes = FormantResult(f1: 0.0, f2: 0.0, f3: 0.0, b1: 0.0, b2: 0.0, b3: 0.0, count: 0)
                    if lpcSuccess {
                        workspace.lpcCoeffs.withUnsafeBufferPointer { cPtr in
                            let solverSuccess = solver.solve(coefficients: cPtr.baseAddress!, order: config.lpcOrder, workspace: workspace)
                            if solverSuccess {
                                workspace.durandKernerCurr.withUnsafeBufferPointer { rPtr in
                                    formantRes = formantExtractor.extractFormants(roots: rPtr.baseAddress!, count: config.lpcOrder)
                                }
                            }
                        }
                    }
                    
                    let features = filterbank.extractFeatures(
                        pcmPtr: framePtr,
                        count: frameSize,
                        workspace: workspace
                    )
                    
                    XCTAssertEqual(features.count, 64)
                    var featIdx = 0
                    while featIdx < 64 {
                        let val = features[featIdx]
                        XCTAssertFalse(val.isNaN, "Feature [\(featIdx)] is NaN in pattern \(pattern)")
                        XCTAssertFalse(val.isInfinite, "Feature [\(featIdx)] is Infinite in pattern \(pattern)")
                        XCTAssertLessThanOrEqual(0.0, val, "Feature [\(featIdx)] is negative: \(val)")
                        XCTAssertLessThanOrEqual(val, 1.0, "Feature [\(featIdx)] exceeds 1.0: \(val)")
                        featIdx += 1
                    }
                }
                
                f += 1
            }
            pattern += 1
        }
    }
}
