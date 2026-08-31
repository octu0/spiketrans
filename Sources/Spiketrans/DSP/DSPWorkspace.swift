import Foundation

/// ホットパスにおける動的メモリアロケーション回数 0 を保証する事前確保ワークスペース
public final class DSPWorkspace: @unchecked Sendable {
    public let maxFrameSize: Int
    public let lpcOrder: Int
    public let melChannels: Int
    
    // バッファ配列
    public var rawFrame: [Float]
    public var windowedFrame: [Float]
    public var clippedFrame: [Float]
    public var preemphasizedFrame: [Float]
    public var hammingWindow: [Float]
    
    // FFT & スペクトル
    public var fftReal: [Float]
    public var fftImag: [Float]
    public var powerSpectrum: [Float]
    
    // LPC & ピッチ用 (VAD / 有声判定)
    public var pitchAutocorr: [Float]
    public var pitchPeaksLag: [Int]
    public var pitchPeaksR: [Float]
    public var lpcAutoCorr: [Float]
    public var lpcCoeffs: [Float]
    public var lpcTempA: [Float]
    public var durandKernerCurr: [Complex]
    public var durandKernerNext: [Complex]
    public var formantCandidatesFreq: [Float]
    public var formantCandidatesBw: [Float]
    
    // フィルタバンク & 64次元音響特徴量
    public var melEnergies: [Float]
    public var featureBuffer: [Float]
    
    public init(
        maxFrameSize: Int = 1024,
        lpcOrder: Int = 12,
        melChannels: Int = 64,
        maxPitchLag: Int = 320,
        fftSize: Int = 512
    ) {
        self.maxFrameSize = maxFrameSize
        self.lpcOrder = lpcOrder
        self.melChannels = melChannels
        
        self.rawFrame = [Float](repeating: 0.0, count: maxFrameSize)
        self.windowedFrame = [Float](repeating: 0.0, count: maxFrameSize)
        self.clippedFrame = [Float](repeating: 0.0, count: maxFrameSize)
        self.preemphasizedFrame = [Float](repeating: 0.0, count: maxFrameSize)
        self.hammingWindow = [Float](repeating: 0.0, count: maxFrameSize)
        
        self.fftReal = [Float](repeating: 0.0, count: fftSize)
        self.fftImag = [Float](repeating: 0.0, count: fftSize)
        self.powerSpectrum = [Float](repeating: 0.0, count: (fftSize / 2) + 1)
        
        self.pitchAutocorr = [Float](repeating: 0.0, count: maxPitchLag + 2)
        self.pitchPeaksLag = [Int](repeating: 0, count: maxPitchLag + 2)
        self.pitchPeaksR = [Float](repeating: 0.0, count: maxPitchLag + 2)
        
        self.lpcAutoCorr = [Float](repeating: 0.0, count: lpcOrder + 1)
        self.lpcCoeffs = [Float](repeating: 0.0, count: lpcOrder + 1)
        self.lpcTempA = [Float](repeating: 0.0, count: lpcOrder + 1)
        
        self.durandKernerCurr = [Complex](repeating: Complex(real: 0.0, imag: 0.0), count: lpcOrder)
        self.durandKernerNext = [Complex](repeating: Complex(real: 0.0, imag: 0.0), count: lpcOrder)
        self.formantCandidatesFreq = [Float](repeating: 0.0, count: lpcOrder)
        self.formantCandidatesBw = [Float](repeating: 0.0, count: lpcOrder)
        
        self.melEnergies = [Float](repeating: 0.0, count: melChannels)
        self.featureBuffer = [Float](repeating: 0.0, count: melChannels)
        
        // ハミング窓の事前計算
        let factor = (2.0 * Float.pi) / Float(maxFrameSize - 1)
        var i = 0
        while i < maxFrameSize {
            self.hammingWindow[i] = 0.54 - (0.46 * cos(Float(i) * factor))
            i += 1
        }
    }
}
