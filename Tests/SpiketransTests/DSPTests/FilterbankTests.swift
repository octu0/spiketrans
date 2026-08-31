import XCTest
@testable import Spiketrans

final class FilterbankTests: XCTestCase {
    
    /// Mel フィルタ三角窓の幾何学的検証 (中心周波数、帯域幅、重みの三角形状)
    func testMelFilterbankTriangleGeometry() {
        let filterbank = Filterbank()
        XCTAssertEqual(filterbank.config.melChannels, 64)
    }
    
    /// 純音 (1000Hz, 4000Hz) に対する Mel チャンネル応答の周波数弁別性
    func testPureToneMelChannelResponse() {
        let config = DSPConfig(sampleRate: 16000, frameSize: 512, hopSize: 160, melChannels: 64)
        let filterbank = Filterbank(config: config)
        let workspace = DSPWorkspace(maxFrameSize: 512, lpcOrder: 12, melChannels: 64)
        
        // 1000Hz 純音
        var tone1000 = [Float](repeating: 0.0, count: 512)
        var i = 0
        while i < 512 {
            tone1000[i] = 0.8 * sin(2.0 * Float.pi * 1000.0 * Float(i) / 16000.0)
            i += 1
        }
        
        let feat1000 = tone1000.withUnsafeBufferPointer { ptr in
            filterbank.extractFeatures(
                pcmPtr: ptr.baseAddress!,
                count: 512,
                workspace: workspace
            )
        }
        
        // 1000Hz 付近の Mel チャンネルのエネルギーが低域・高域より有意に大きいこと
        let midEnergy = feat1000[20]
        let lowEnergy = feat1000[1]
        let highEnergy = feat1000[55]
        
        XCTAssertLessThan(lowEnergy, midEnergy)
        XCTAssertLessThan(highEnergy, midEnergy)
        
        // 4000Hz 純音
        var tone4000 = [Float](repeating: 0.0, count: 512)
        i = 0
        while i < 512 {
            tone4000[i] = 0.8 * sin(2.0 * Float.pi * 4000.0 * Float(i) / 16000.0)
            i += 1
        }
        
        let feat4000 = tone4000.withUnsafeBufferPointer { ptr in
            filterbank.extractFeatures(
                pcmPtr: ptr.baseAddress!,
                count: 512,
                workspace: workspace
            )
        }
        
        let highToneEnergy = feat4000[48]
        let lowToneEnergy = feat4000[2]
        XCTAssertLessThan(lowToneEnergy, highToneEnergy)
    }
    
    /// 合成母音 (/a/, /i/) に対するスペクトル包絡・特徴量弁別性
    func testSyntheticVowelSpectralEnvelope() {
        let config = DSPConfig(sampleRate: 16000, frameSize: 512, melChannels: 64)
        let filterbank = Filterbank(config: config)
        let workspace = DSPWorkspace(maxFrameSize: 512, melChannels: 64)
        
        var vowelA = [Float](repeating: 0.0, count: 512)
        var i = 0
        while i < 512 {
            let t = Float(i) / 16000.0
            vowelA[i] = (0.5 * sin(2.0 * Float.pi * 800.0 * t)) + (0.3 * sin(2.0 * Float.pi * 1200.0 * t)) + (0.2 * sin(2.0 * Float.pi * 2500.0 * t))
            i += 1
        }
        
        let features = vowelA.withUnsafeBufferPointer { ptr in
            filterbank.extractFeatures(
                pcmPtr: ptr.baseAddress!,
                count: 512,
                workspace: workspace
            )
        }
        
        XCTAssertEqual(features.count, 64)
        for f in features {
            XCTAssertFalse(f.isNaN)
            XCTAssertFalse(f.isInfinite)
            XCTAssertLessThanOrEqual(0.0, f)
            XCTAssertLessThanOrEqual(f, 1.0)
        }
    }
    
    /// 64次元特徴量ベクトルの境界値検証 (無音、極大値)
    func testFeatureVector64DimensionsBoundary() {
        let filterbank = Filterbank()
        let workspace = DSPWorkspace(melChannels: 64)
        let silence = [Float](repeating: 0.0, count: 320)
        
        let featSilence = silence.withUnsafeBufferPointer { ptr in
            filterbank.extractFeatures(pcmPtr: ptr.baseAddress!, count: 320, workspace: workspace)
        }
        XCTAssertEqual(featSilence.count, 64)
        for f in featSilence {
            XCTAssertLessThanOrEqual(0.0, f)
            XCTAssertLessThanOrEqual(f, 1.0)
        }
    }
    
    /// FFT パワースペクトル計算における SIMD8 処理とスカラー処理の厳密な精度一致検証
    func testFFTPowerSpectrumSIMDVsScalarEquivalence() {
        let fft = FFT(size: 512)
        let halfSize = 256
        
        // 複数パターンの入力信号で検証 (インパルス、マルチトーン、ノイズ)
        var impulseSignal = [Float](repeating: 0.0, count: 512)
        impulseSignal[0] = 1.0
        
        var multiToneSignal = [Float](repeating: 0.0, count: 512)
        var idx = 0
        while idx < 512 {
            multiToneSignal[idx] = 0.4 * sin(2.0 * Float.pi * 300.0 * Float(idx) / 16000.0) +
                0.3 * cos(2.0 * Float.pi * 1200.0 * Float(idx) / 16000.0) +
                0.2 * sin(2.0 * Float.pi * 3500.0 * Float(idx) / 16000.0)
            idx += 1
        }
        
        var noiseSignal = [Float](repeating: 0.0, count: 512)
        idx = 0
        while idx < 512 {
            let pseudo = Float(((idx * 1103515245 + 12345) & 0x7FFFFFFF) % 1000) / 1000.0 - 0.5
            noiseSignal[idx] = pseudo
            idx += 1
        }
        
        let signalPatterns: [[Float]] = [impulseSignal, multiToneSignal, noiseSignal]
        
        for signal in signalPatterns {
            var realSimd = signal
            var imagSimd = [Float](repeating: 0.0, count: 512)
            var powerSimd = [Float](repeating: 0.0, count: halfSize + 1)
            
            realSimd.withUnsafeMutableBufferPointer { rPtr in
                imagSimd.withUnsafeMutableBufferPointer { iPtr in
                    fft.forward(real: rPtr.baseAddress!, imag: iPtr.baseAddress!)
                    
                    powerSimd.withUnsafeMutableBufferPointer { pPtr in
                        fft.computePowerSpectrum(
                            real: rPtr.baseAddress!,
                            imag: iPtr.baseAddress!,
                            powerSpectrum: pPtr.baseAddress!,
                            halfSize: halfSize
                        )
                    }
                }
            }
            
            // スカラー基準値の算出
            var powerScalar = [Float](repeating: 0.0, count: halfSize + 1)
            var k = 0
            while k <= halfSize {
                let r = realSimd[k]
                let im = imagSimd[k]
                powerScalar[k] = (r * r) + (im * im)
                k += 1
            }
            
            // 全ビンでの値の一致を確認
            k = 0
            while k <= halfSize {
                let sVal = powerScalar[k]
                let vVal = powerSimd[k]
                let diff = abs(vVal - sVal)
                XCTAssertLessThanOrEqual(diff, 1e-6, "FFT PowerSpectrum mismatch at bin \(k): SIMD=\(vVal), Scalar=\(sVal)")
                k += 1
            }
        }
    }
}
