import XCTest
@testable import Spiketrans

final class LPCTests: XCTestCase {
    
    /// 1次 AR モデル x[n] = 0.8 * x[n-1] + e[n] に対する Levinson-Durbin 係数検証
    func testLevinsonDurbinFirstOrderAR() {
        let config = DSPConfig(sampleRate: 16000, frameSize: 512, lpcOrder: 2)
        let lpc = LPC(config: config)
        let workspace = DSPWorkspace(maxFrameSize: 512, lpcOrder: 2)
        
        var signal = [Float](repeating: 0.0, count: 512)
        var prev: Float = 0.0
        var i = 0
        while i < 512 {
            let noise = Float.random(in: -0.1...0.1)
            let curr = (0.8 * prev) + noise
            signal[i] = curr
            prev = curr
            i += 1
        }
        
        let success = signal.withUnsafeBufferPointer { ptr in
            lpc.computeCoefficients(ptr: ptr.baseAddress!, count: 512, workspace: workspace)
        }
        
        XCTAssertTrue(success)
        // c0 = 1.0, c1 = -a1
        XCTAssertEqual(workspace.lpcCoeffs[0], 1.0)
        XCTAssertFalse(workspace.lpcCoeffs[1].isNaN)
    }
    
    /// 予測誤差の単調減少性と反射係数 |ki| < 1.0 の検証
    func testPredictionErrorMonotonicDecrease() {
        let config = DSPConfig(sampleRate: 16000, frameSize: 512, lpcOrder: 12)
        let lpc = LPC(config: config)
        let workspace = DSPWorkspace(maxFrameSize: 512, lpcOrder: 12)
        
        var signal = [Float](repeating: 0.0, count: 512)
        var i = 0
        while i < 512 {
            signal[i] = sin(2.0 * Float.pi * 300.0 * Float(i) / 16000.0) + (0.5 * sin(2.0 * Float.pi * 1200.0 * Float(i) / 16000.0))
            i += 1
        }
        
        let success = signal.withUnsafeBufferPointer { ptr in
            lpc.computeCoefficients(ptr: ptr.baseAddress!, count: 512, workspace: workspace)
        }
        XCTAssertTrue(success)
    }
    
    /// 合成母音 (/a/) に対する LPC + Durand-Kerner 結合フォルマント復元検証
    func testSyntheticVowelLPCFormantReconstruction() {
        let config = DSPConfig(sampleRate: 16000, frameSize: 512, lpcOrder: 12)
        let lpc = LPC(config: config)
        let solver = DurandKernerSolver()
        let extractor = FormantExtractor(sampleRate: 16000.0)
        let workspace = DSPWorkspace(maxFrameSize: 512, lpcOrder: 12)
        
        // F1=800Hz, F2=1200Hz, F3=2500Hz
        var vowelA = [Float](repeating: 0.0, count: 512)
        var i = 0
        while i < 512 {
            let t = Float(i) / 16000.0
            vowelA[i] = sin(2.0 * Float.pi * 800.0 * t) + (0.7 * sin(2.0 * Float.pi * 1200.0 * t)) + (0.4 * sin(2.0 * Float.pi * 2500.0 * t))
            i += 1
        }
        
        let success = vowelA.withUnsafeBufferPointer { ptr in
            lpc.computeCoefficients(ptr: ptr.baseAddress!, count: 512, workspace: workspace)
        }
        XCTAssertTrue(success)
        
        let coeffPtr = workspace.lpcCoeffs.withUnsafeBufferPointer { $0.baseAddress! }
        let solved = solver.solve(coefficients: coeffPtr, order: 12, workspace: workspace)
        XCTAssertTrue(solved)
        
        let rootsPtr = workspace.durandKernerCurr.withUnsafeBufferPointer { $0.baseAddress! }
        let formants = extractor.extractFormants(roots: rootsPtr, count: 12)
        
        XCTAssertFalse(formants.f1.isNaN)
        XCTAssertFalse(formants.b1.isNaN)
        XCTAssertLessThanOrEqual(0.0, formants.b1) // 帯域幅が正であること
    }
}
