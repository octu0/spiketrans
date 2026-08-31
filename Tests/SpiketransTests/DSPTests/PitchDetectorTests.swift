import XCTest
@testable import Spiketrans

final class PitchDetectorTests: XCTestCase {
    
    func testPureTonePitch150Hz() {
        let config = DSPConfig(sampleRate: 16000, maxPitchLag: 320)
        let detector = PitchDetector(config: config)
        let workspace = DSPWorkspace(maxFrameSize: 1024, maxPitchLag: 320)
        
        let frameSize = 512
        var frame = [Float](repeating: 0.0, count: frameSize)
        let targetFreq: Float = 150.0
        let factor = (2.0 * Float.pi * targetFreq) / 16000.0
        var i = 0
        while i < frameSize {
            frame[i] = 0.8 * sin(Float(i) * factor)
            i += 1
        }
        
        let result = frame.withUnsafeBufferPointer { ptr in
            return detector.detectPitch(ptr: ptr.baseAddress!, count: frameSize, workspace: workspace)
        }
        
        XCTAssertTrue(result.isVoiced)
        XCTAssertLessThan(abs(result.f0 - targetFreq), 5.0)
        XCTAssertLessThan(15.0, result.hnr)
    }
    
    func testPureTonePitch250Hz() {
        let config = DSPConfig(sampleRate: 16000, maxPitchLag: 320)
        let detector = PitchDetector(config: config)
        let workspace = DSPWorkspace(maxFrameSize: 1024, maxPitchLag: 320)
        
        let frameSize = 512
        var frame = [Float](repeating: 0.0, count: frameSize)
        let targetFreq: Float = 250.0
        let factor = (2.0 * Float.pi * targetFreq) / 16000.0
        var i = 0
        while i < frameSize {
            frame[i] = 0.8 * sin(Float(i) * factor)
            i += 1
        }
        
        let result = frame.withUnsafeBufferPointer { ptr in
            return detector.detectPitch(ptr: ptr.baseAddress!, count: frameSize, workspace: workspace)
        }
        
        XCTAssertTrue(result.isVoiced)
        XCTAssertLessThan(abs(result.f0 - targetFreq), 5.0)
    }
    
    func testPureTonePitch400Hz() {
        let config = DSPConfig(sampleRate: 16000, maxPitchLag: 320)
        let detector = PitchDetector(config: config)
        let workspace = DSPWorkspace(maxFrameSize: 1024, maxPitchLag: 320)
        
        let frameSize = 512
        var frame = [Float](repeating: 0.0, count: frameSize)
        let targetFreq: Float = 400.0
        let factor = (2.0 * Float.pi * targetFreq) / 16000.0
        var i = 0
        while i < frameSize {
            frame[i] = 0.8 * sin(Float(i) * factor)
            i += 1
        }
        
        let result = frame.withUnsafeBufferPointer { ptr in
            return detector.detectPitch(ptr: ptr.baseAddress!, count: frameSize, workspace: workspace)
        }
        
        XCTAssertTrue(result.isVoiced)
        XCTAssertLessThan(abs(result.f0 - targetFreq), 5.0)
    }
    
    func testOctaveErrorSuppression() {
        let config = DSPConfig(sampleRate: 16000, maxPitchLag: 320)
        let detector = PitchDetector(config: config)
        let workspace = DSPWorkspace(maxFrameSize: 1024, maxPitchLag: 320)
        
        let frameSize = 512
        var frame = [Float](repeating: 0.0, count: frameSize)
        let f0: Float = 150.0
        let f1: Float = 300.0 // 2倍音
        let factor0 = (2.0 * Float.pi * f0) / 16000.0
        let factor1 = (2.0 * Float.pi * f1) / 16000.0
        
        var i = 0
        while i < frameSize {
            frame[i] = (0.5 * sin(Float(i) * factor0)) + (0.5 * sin(Float(i) * factor1))
            i += 1
        }
        
        let result = frame.withUnsafeBufferPointer { ptr in
            return detector.detectPitch(ptr: ptr.baseAddress!, count: frameSize, workspace: workspace)
        }
        
        XCTAssertTrue(result.isVoiced)
        // 2倍音 (300Hz) ではなく基本波 (150Hz) が選択されていることを検証
        XCTAssertLessThan(abs(result.f0 - f0), 5.0)
    }
    
    func testAllZeroSilencePitch() {
        let config = DSPConfig(sampleRate: 16000, maxPitchLag: 320)
        let detector = PitchDetector(config: config)
        let workspace = DSPWorkspace(maxFrameSize: 1024, maxPitchLag: 320)
        
        let frame = [Float](repeating: 0.0, count: 512)
        let result = frame.withUnsafeBufferPointer { ptr in
            return detector.detectPitch(ptr: ptr.baseAddress!, count: 512, workspace: workspace)
        }
        
        XCTAssertFalse(result.isVoiced)
        XCTAssertEqual(result.f0, 0.0)
        XCTAssertEqual(result.hnr, 0.0)
    }
    
    func testWhiteNoiseUnvoiced() {
        let config = DSPConfig(sampleRate: 16000, maxPitchLag: 320)
        let detector = PitchDetector(config: config)
        let workspace = DSPWorkspace(maxFrameSize: 1024, maxPitchLag: 320)
        
        let frameSize = 512
        var noise = [Float](repeating: 0.0, count: frameSize)
        var rngState: UInt64 = 88172645463325252
        var i = 0
        while i < frameSize {
            rngState ^= rngState << 13
            rngState ^= rngState >> 7
            rngState ^= rngState << 17
            let uVal = Float(rngState & 0xFFFFFF) / Float(0xFFFFFF)
            noise[i] = (uVal - 0.5) * 0.2
            i += 1
        }
        
        let result = noise.withUnsafeBufferPointer { ptr in
            return detector.detectPitch(ptr: ptr.baseAddress!, count: frameSize, workspace: workspace)
        }
        
        // 白色雑音では相関が低いため isVoiced != true
        XCTAssertFalse(result.isVoiced)
    }
}
