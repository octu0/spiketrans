import Foundation

public struct Filterbank: Sendable {
    public let config: DSPConfig
    private let fft: FFT
    private let vad: VAD
    private let lpc: LPC
    private let durandKerner: DurandKernerSolver
    private let formantExtractor: FormantExtractor
    
    // 事前計算された Mel フィルタバンクの中心周波数と重みテーブル
    private let melWeights: [[(bin: Int, weight: Float)]]
    
    public init(config: DSPConfig = DSPConfig()) {
        self.config = config
        self.fft = FFT(size: 512)
        self.vad = VAD(config: config)
        self.lpc = LPC(config: config)
        self.durandKerner = DurandKernerSolver()
        self.formantExtractor = FormantExtractor(sampleRate: Float(config.sampleRate))
        
        // 64ch Mel フィルタの構築
        var weights: [[(bin: Int, weight: Float)]] = []
        let numChannels = config.melChannels
        let fftSize = 512
        let halfFft = fftSize / 2
        let fMin: Float = 100.0
        let fMax: Float = Float(config.sampleRate) / 2.0
        
        let melMin = 2595.0 * log10(1.0 + (fMin / 700.0))
        let melMax = 2595.0 * log10(1.0 + (fMax / 700.0))
        let melStep = (melMax - melMin) / Float(numChannels + 1)
        
        var melPoints = [Float](repeating: 0.0, count: numChannels + 2)
        var i = 0
        while i <= (numChannels + 1) {
            let mel = melMin + (Float(i) * melStep)
            melPoints[i] = 700.0 * (pow(10.0, mel / 2595.0) - 1.0)
            i += 1
        }
        
        var ch = 1
        while ch <= numChannels {
            let leftFreq = melPoints[ch - 1]
            let centerFreq = melPoints[ch]
            let rightFreq = melPoints[ch + 1]
            
            var channelWeights: [(bin: Int, weight: Float)] = []
            var k = 0
            while k < halfFft {
                let freq = (Float(k) * Float(config.sampleRate)) / Float(fftSize)
                if leftFreq <= freq && freq <= centerFreq {
                    let w = (freq - leftFreq) / (centerFreq - leftFreq)
                    channelWeights.append((bin: k, weight: w))
                }
                if centerFreq < freq && freq <= rightFreq {
                    let w = (rightFreq - freq) / (rightFreq - centerFreq)
                    channelWeights.append((bin: k, weight: w))
                }
                k += 1
            }
            weights.append(channelWeights)
            ch += 1
        }
        self.melWeights = weights
    }
    
    /// FFT パワースペクトルから 64 次元 Mel 特徴量ベクトルを抽出 (Direct Input Current: 0.0〜1.0)
    @discardableResult
    @inline(__always)
    public func extractFeatures(
        pcmPtr: UnsafePointer<Float>,
        count: Int,
        workspace: DSPWorkspace
    ) -> [Float] {
        let fftReal = workspace.fftReal.withUnsafeMutableBufferPointer { $0.baseAddress! }
        let fftImag = workspace.fftImag.withUnsafeMutableBufferPointer { $0.baseAddress! }
        let powerSpec = workspace.powerSpectrum.withUnsafeMutableBufferPointer { $0.baseAddress! }
        let winTable = workspace.hammingWindow.withUnsafeBufferPointer { $0.baseAddress! }
        let melEnergies = workspace.melEnergies.withUnsafeMutableBufferPointer { $0.baseAddress! }
        let featureBuf = workspace.featureBuffer.withUnsafeMutableBufferPointer { $0.baseAddress! }
        
        let fftSize = 512
        var winCount = count
        if fftSize < winCount {
            winCount = fftSize
        }
        
        // 1. ハミング窓の適用 & ゼロパディング
        var i = 0
        while i < winCount {
            fftReal[i] = pcmPtr[i] * winTable[i]
            fftImag[i] = 0.0
            i += 1
        }
        while i < fftSize {
            fftReal[i] = 0.0
            fftImag[i] = 0.0
            i += 1
        }
        
        // 2. 512点 FFT 実行
        fft.forward(real: fftReal, imag: fftImag)
        
        // 3. パワースペクトルの算出 (256 ビン)
        fft.computePowerSpectrum(real: fftReal, imag: fftImag, powerSpectrum: powerSpec, halfSize: 256)
        
        // 3.5 フォルマント連動スペクトル適応イコライジング (VAD + LPC + Durand-Kerner)
        let vadRes = vad.processFrame(ptr: pcmPtr, count: winCount, workspace: workspace)
        if vadRes.isSpeech {
            let lpcSuccess = lpc.computeCoefficients(ptr: pcmPtr, count: winCount, workspace: workspace)
            var formantRes = FormantResult(f1: 0.0, f2: 0.0, f3: 0.0, b1: 0.0, b2: 0.0, b3: 0.0, count: 0)
            if lpcSuccess {
                workspace.lpcCoeffs.withUnsafeBufferPointer { cPtr in
                    if durandKerner.solve(coefficients: cPtr.baseAddress!, order: config.lpcOrder, workspace: workspace) {
                        workspace.durandKernerCurr.withUnsafeBufferPointer { rPtr in
                            formantRes = formantExtractor.extractFormants(roots: rPtr.baseAddress!, count: config.lpcOrder, workspace: workspace)
                        }
                    }
                }
            }

            let sRate = Float(config.sampleRate)
            var k = 0
            while k < 256 {
                let freq = (Float(k) * sRate) / 512.0
                var gain: Float = 1.0

                if freq < 200.0 || 4000.0 < freq {
                    gain = 0.2 // 非人声帯域 (低域ハミング・高域ノイズ) を減衰
                } else {
                    gain = 1.0 // 人声帯域 (200〜4000Hz) のベースライン
                    if 0 < formantRes.count {
                        // F1 ゲイン強調
                        if 0.0 < formantRes.f1 {
                            let w1 = max(80.0, formantRes.b1 * 0.5)
                            let diff1 = abs(freq - formantRes.f1)
                            if diff1 <= w1 {
                                let peakGain = 1.0 + (1.5 * (1.0 - (diff1 / w1)))
                                if gain < peakGain { gain = peakGain }
                            }
                        }
                        // F2 ゲイン強調
                        if 0.0 < formantRes.f2 {
                            let w2 = max(80.0, formantRes.b2 * 0.5)
                            let diff2 = abs(freq - formantRes.f2)
                            if diff2 <= w2 {
                                let peakGain = 1.0 + (1.5 * (1.0 - (diff2 / w2)))
                                if gain < peakGain { gain = peakGain }
                            }
                        }
                        // F3 ゲイン強調
                        if 0.0 < formantRes.f3 {
                            let w3 = max(80.0, formantRes.b3 * 0.5)
                            let diff3 = abs(freq - formantRes.f3)
                            if diff3 <= w3 {
                                let peakGain = 1.0 + (1.5 * (1.0 - (diff3 / w3)))
                                if gain < peakGain { gain = peakGain }
                            }
                        }
                    }
                }

                powerSpec[k] = max(1e-12, powerSpec[k] * gain)
                k += 1
            }
        }
        
        // 4. 64ch Mel エネルギーの積算
        var ch = 0
        while ch < config.melChannels {
            let wList = melWeights[ch]
            var sumEnergy: Float = 1e-6
            var idx = 0
            while idx < wList.count {
                let item = wList[idx]
                sumEnergy += powerSpec[item.bin] * item.weight
                idx += 1
            }
            melEnergies[ch] = log(sumEnergy)
            ch += 1
        }
        
        // 5. 64ch Mel 特徴量の Direct Input Current への正規化 ([0.0, 1.0])
        ch = 0
        while ch < config.melChannels {
            let rawE = melEnergies[ch]
            let normE = (rawE + 10.0) / 10.0
            var c = normE
            if normE < 0.0 {
                c = 0.0
            }
            if 1.0 < c {
                c = 1.0
            }
            featureBuf[ch] = c
            ch += 1
        }
        
        return Array(workspace.featureBuffer)
    }
}
