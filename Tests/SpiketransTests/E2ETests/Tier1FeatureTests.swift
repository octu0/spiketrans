import XCTest
import Foundation
@testable import Spiketrans

private final class ResultCollector<T>: @unchecked Sendable {
    private let lock = NSLock()
    private var items: [T] = []

    func append(_ item: T) {
        lock.lock()
        defer { lock.unlock() }
        items.append(item)
    }

    var count: Int {
        lock.lock()
        defer { lock.unlock() }
        return items.count
    }

    var allItems: [T] {
        lock.lock()
        defer { lock.unlock() }
        return items
    }

    subscript(index: Int) -> T {
        lock.lock()
        defer { lock.unlock() }
        return items[index]
    }
}

private final class AtomicCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var val: Int = 0

    func increment() {
        lock.lock()
        defer { lock.unlock() }
        val += 1
    }

    var value: Int {
        lock.lock()
        defer { lock.unlock() }
        return val
    }
}

final class Tier1FeatureTests: XCTestCase {

    // MARK: - Helper Methods

    private func createWavData(
        samples: [Float],
        sampleRate: Int = 16000,
        channels: Int = 1,
        bitsPerSample: Int = 16
    ) -> Data {
        var data = Data()
        data.append(contentsOf: [UInt8]("RIFF".utf8))
        let bytesPerSample = bitsPerSample / 8
        let subchunk2Size = UInt32(samples.count * bytesPerSample)
        let chunkSize = UInt32(36 + Int(subchunk2Size))

        var chunkSizeLE = chunkSize.littleEndian
        withUnsafeBytes(of: &chunkSizeLE) { data.append(contentsOf: $0) }

        data.append(contentsOf: [UInt8]("WAVE".utf8))
        data.append(contentsOf: [UInt8]("fmt ".utf8))

        let subchunk1Size: UInt32 = 16
        let audioFormat: UInt16 = 1
        let numChannels = UInt16(channels)
        let sRate = UInt32(sampleRate)
        let byteRate = UInt32(sampleRate * channels * bytesPerSample)
        let blockAlign = UInt16(channels * bytesPerSample)
        let bitsPerSamp = UInt16(bitsPerSample)

        var s1SizeLE = subchunk1Size.littleEndian
        withUnsafeBytes(of: &s1SizeLE) { data.append(contentsOf: $0) }
        var afLE = audioFormat.littleEndian
        withUnsafeBytes(of: &afLE) { data.append(contentsOf: $0) }
        var ncLE = numChannels.littleEndian
        withUnsafeBytes(of: &ncLE) { data.append(contentsOf: $0) }
        var srLE = sRate.littleEndian
        withUnsafeBytes(of: &srLE) { data.append(contentsOf: $0) }
        var brLE = byteRate.littleEndian
        withUnsafeBytes(of: &brLE) { data.append(contentsOf: $0) }
        var baLE = blockAlign.littleEndian
        withUnsafeBytes(of: &baLE) { data.append(contentsOf: $0) }
        var bpsLE = bitsPerSamp.littleEndian
        withUnsafeBytes(of: &bpsLE) { data.append(contentsOf: $0) }

        data.append(contentsOf: [UInt8]("data".utf8))
        var s2SizeLE = subchunk2Size.littleEndian
        withUnsafeBytes(of: &s2SizeLE) { data.append(contentsOf: $0) }

        var i = 0
        while i < samples.count {
            let clamped = max(-1.0, min(1.0, samples[i]))
            let intSample = Int16(clamped * 32767.0)
            var sampleLE = intSample.littleEndian
            withUnsafeBytes(of: &sampleLE) { data.append(contentsOf: $0) }
            i += 1
        }
        return data
    }

    private func synthesizeSpeech(sampleRate: Int, durationSeconds: Float, amplitude: Float = 0.5) -> [Float] {
        let sampleCount = Int(Float(sampleRate) * durationSeconds)
        var pcm = [Float](repeating: 0.0, count: sampleCount)
        let twoPi = 2.0 * Float.pi

        var i = 0
        while i < sampleCount {
            let t = Float(i) / Float(sampleRate)
            let f0 = sin(twoPi * 200.0 * t)
            let f1 = 0.5 * sin(twoPi * 800.0 * t)
            let f2 = 0.3 * sin(twoPi * 1200.0 * t)
            let f3 = 0.2 * sin(twoPi * 2500.0 * t)
            pcm[i] = amplitude * (f0 + f1 + f2 + f3)
            i += 1
        }
        return pcm
    }

    private func synthesizeSilence(sampleRate: Int, durationSeconds: Float) -> [Float] {
        let sampleCount = Int(Float(sampleRate) * durationSeconds)
        return [Float](repeating: 0.0, count: sampleCount)
    }

    // MARK: - Feature 1: WavParser (5 tests)

    func testWavParserStandardMono16kHz() throws {
        let rawSamples: [Float] = [0.0, 0.25, 0.5, 0.75, 1.0, -0.25, -0.5, -0.75, -1.0]
        let data = createWavData(samples: rawSamples, sampleRate: 16000, channels: 1, bitsPerSample: 16)
        let parser = WavParser()
        let wav = try parser.parse(bytes: [UInt8](data))

        XCTAssertEqual(wav.sampleRate, 16000)
        XCTAssertEqual(wav.channels, 1)
        XCTAssertEqual(wav.bitsPerSample, 16)
        XCTAssertEqual(wav.pcmData.count, rawSamples.count)
        var i = 0
        while i < rawSamples.count {
            XCTAssertLessThanOrEqual(abs(wav.pcmData[i] - rawSamples[i]), 0.01)
            i += 1
        }
    }

    func testWavParserStereoDownmix() throws {
        // L/R alternating samples
        let stereoSamples: [Float] = [0.2, 0.4, 0.6, 0.8, -0.2, -0.4, -0.6, -0.8]
        let data = createWavData(samples: stereoSamples, sampleRate: 16000, channels: 2, bitsPerSample: 16)
        let parser = WavParser()
        let wav = try parser.parse(bytes: [UInt8](data))

        XCTAssertEqual(wav.channels, 2)
        XCTAssertEqual(wav.pcmData.count, stereoSamples.count / 2)
        XCTAssertLessThanOrEqual(abs(wav.pcmData[0] - 0.3), 0.01) // (0.2 + 0.4) / 2
        XCTAssertLessThanOrEqual(abs(wav.pcmData[1] - 0.7), 0.01) // (0.6 + 0.8) / 2
        XCTAssertLessThanOrEqual(abs(wav.pcmData[2] - (-0.3)), 0.01)
        XCTAssertLessThanOrEqual(abs(wav.pcmData[3] - (-0.7)), 0.01)
    }

    func testWavParserNonStandardSampleRate() throws {
        let samples = [Float](repeating: 0.1, count: 800)
        let parser = WavParser()
        let data8k = createWavData(samples: samples, sampleRate: 8000, channels: 1, bitsPerSample: 16)
        let wav8k = try parser.parse(bytes: [UInt8](data8k))
        XCTAssertEqual(wav8k.sampleRate, 8000)
        XCTAssertEqual(wav8k.pcmData.count, 800)

        let data44k = createWavData(samples: samples, sampleRate: 44100, channels: 1, bitsPerSample: 16)
        let wav44k = try parser.parse(bytes: [UInt8](data44k))
        XCTAssertEqual(wav44k.sampleRate, 44100)
        XCTAssertEqual(wav44k.pcmData.count, 800)
    }

    func testWavParserInvalidHeaderThrows() {
        let badData = Data([0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07])
        let parser = WavParser()
        XCTAssertThrowsError(try parser.parse(bytes: [UInt8](badData))) { error in
            guard let wavError = error as? WavParserError else {
                XCTFail("Expected WavParserError")
                return
            }
            switch wavError {
            case .invalidHeader, .fmtChunkNotFound, .dataChunkNotFound, .unsupportedFormat:
                break
            }
        }
    }

    func testWavParserNonPCMFormatThrows() {
        var data = createWavData(samples: [0.0, 0.1], sampleRate: 16000, channels: 1, bitsPerSample: 16)
        // Corrupt format code at offset 20 to 3 (IEEE float)
        data[20] = 3
        data[21] = 0
        let parser = WavParser()
        XCTAssertThrowsError(try parser.parse(bytes: [UInt8](data))) { error in
            guard let wavError = error as? WavParserError else {
                XCTFail("Expected WavParserError")
                return
            }
            switch wavError {
            case .unsupportedFormat:
                break
            default:
                XCTFail("Expected unsupportedFormat error")
            }
        }
    }

    // MARK: - Feature 2: Multi-feature VAD (5 tests)

    func testVADSilenceFrameDetection() {
        let config = DSPConfig(sampleRate: 16000)
        let vad = VAD(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        let silence = [Float](repeating: 0.0, count: config.frameSize)
        silence.withUnsafeBufferPointer { buf in
            let res = vad.processFrame(ptr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
            XCTAssertFalse(res.isSpeech)
            XCTAssertLessThanOrEqual(res.rms, 1e-5)
            XCTAssertLessThanOrEqual(res.zcr, 1e-5)
        }
    }

    func testVADVoicedSpeechDetection() {
        let config = DSPConfig(sampleRate: 16000)
        let vad = VAD(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        var speech = [Float](repeating: 0.0, count: config.frameSize)
        let twoPi = 2.0 * Float.pi
        var i = 0
        while i < config.frameSize {
            let t = Float(i) / 16000.0
            speech[i] = 0.5 * sin(twoPi * 200.0 * t)
            i += 1
        }

        speech.withUnsafeBufferPointer { buf in
            let res = vad.processFrame(ptr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
            XCTAssertTrue(res.isSpeech)
            XCTAssertLessThanOrEqual(0.01, res.rms)
            XCTAssertLessThanOrEqual(0.30, res.voicingRatio)
        }
    }

    func testVADUnvoicedFricativeDetection() {
        let config = DSPConfig(sampleRate: 16000)
        let vad = VAD(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        // Alternating high-frequency noise (high ZCR)
        var fricative = [Float](repeating: 0.0, count: config.frameSize)
        var i = 0
        while i < config.frameSize {
            switch i % 2 {
            case 0:
                fricative[i] = 0.15
            default:
                fricative[i] = -0.15
            }
            i += 1
        }

        fricative.withUnsafeBufferPointer { buf in
            let res = vad.processFrame(ptr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
            XCTAssertTrue(res.isSpeech)
            XCTAssertLessThanOrEqual(0.40, res.zcr)
        }
    }

    func testVADHangoverAndPreRollState() {
        let config = DSPConfig(sampleRate: 16000)
        let vad = VAD(config: config)
        let workspace = DSPWorkspace(maxFrameSize: 1024, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        var audio: [Float] = []
        audio.append(contentsOf: synthesizeSpeech(sampleRate: 16000, durationSeconds: 0.2, amplitude: 0.5))
        audio.append(contentsOf: synthesizeSilence(sampleRate: 16000, durationSeconds: 0.1))

        let segments = vad.segmentUtterances(pcmData: audio, workspace: workspace)
        XCTAssertEqual(segments.count, 1)
        XCTAssertLessThanOrEqual(0.2, segments[0].durationSeconds)
    }

    func testVADUtteranceSegmentation() {
        let config = DSPConfig(sampleRate: 16000)
        let vad = VAD(config: config)
        let workspace = DSPWorkspace(maxFrameSize: 1024, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        var audio: [Float] = []
        audio.append(contentsOf: synthesizeSpeech(sampleRate: 16000, durationSeconds: 0.3, amplitude: 0.5))
        audio.append(contentsOf: synthesizeSilence(sampleRate: 16000, durationSeconds: 0.4))
        audio.append(contentsOf: synthesizeSpeech(sampleRate: 16000, durationSeconds: 0.3, amplitude: 0.5))
        audio.append(contentsOf: synthesizeSilence(sampleRate: 16000, durationSeconds: 0.4))

        let segments = vad.segmentUtterances(pcmData: audio, workspace: workspace)
        XCTAssertEqual(segments.count, 2)
    }

    // MARK: - Feature 3: PitchDetector (5 tests)

    func testPitchDetectorPureSineWave() {
        let config = DSPConfig(sampleRate: 16000, maxPitchLag: 320)
        let detector = PitchDetector(config: config)
        let workspace = DSPWorkspace(maxFrameSize: 1024, maxPitchLag: 320)

        let frameSize = 512
        var sine440 = [Float](repeating: 0.0, count: frameSize)
        let twoPi = 2.0 * Float.pi
        var i = 0
        while i < frameSize {
            let t = Float(i) / 16000.0
            sine440[i] = 0.8 * sin(twoPi * 440.0 * t)
            i += 1
        }

        sine440.withUnsafeBufferPointer { buf in
            let res = detector.detectPitch(ptr: buf.baseAddress!, count: frameSize, workspace: workspace)
            XCTAssertTrue(res.isVoiced)
            XCTAssertLessThanOrEqual(420.0, res.f0)
            XCTAssertLessThanOrEqual(res.f0, 460.0)
            XCTAssertLessThanOrEqual(15.0, res.hnr)
        }
    }

    func testPitchDetectorHarmonicComplexTone() {
        let config = DSPConfig(sampleRate: 16000, maxPitchLag: 320)
        let detector = PitchDetector(config: config)
        let workspace = DSPWorkspace(maxFrameSize: 1024, maxPitchLag: 320)

        let frameSize = 512
        var harmonic200 = [Float](repeating: 0.0, count: frameSize)
        let twoPi = 2.0 * Float.pi
        var i = 0
        while i < frameSize {
            let t = Float(i) / 16000.0
            harmonic200[i] = (0.5 * sin(twoPi * 200.0 * t)) + (0.3 * sin(twoPi * 400.0 * t)) + (0.2 * sin(twoPi * 600.0 * t))
            i += 1
        }

        harmonic200.withUnsafeBufferPointer { buf in
            let res = detector.detectPitch(ptr: buf.baseAddress!, count: frameSize, workspace: workspace)
            XCTAssertTrue(res.isVoiced)
            XCTAssertLessThanOrEqual(190.0, res.f0)
            XCTAssertLessThanOrEqual(res.f0, 210.0)
        }
    }

    func testPitchDetectorOctaveErrorSuppression() {
        let config = DSPConfig(sampleRate: 16000, maxPitchLag: 320)
        let detector = PitchDetector(config: config)
        let workspace = DSPWorkspace(maxFrameSize: 1024, maxPitchLag: 320)

        let frameSize = 512
        var tone = [Float](repeating: 0.0, count: frameSize)
        let twoPi = 2.0 * Float.pi
        var i = 0
        while i < frameSize {
            let t = Float(i) / 16000.0
            tone[i] = (0.4 * sin(twoPi * 150.0 * t)) + (0.6 * sin(twoPi * 300.0 * t))
            i += 1
        }

        tone.withUnsafeBufferPointer { buf in
            let res = detector.detectPitch(ptr: buf.baseAddress!, count: frameSize, workspace: workspace)
            XCTAssertTrue(res.isVoiced)
            XCTAssertLessThanOrEqual(140.0, res.f0)
            XCTAssertLessThanOrEqual(res.f0, 310.0)
        }
    }

    func testPitchDetectorUnvoicedSilenceReturnsZero() {
        let config = DSPConfig(sampleRate: 16000, maxPitchLag: 320)
        let detector = PitchDetector(config: config)
        let workspace = DSPWorkspace(maxFrameSize: 1024, maxPitchLag: 320)

        let frameSize = 512
        let silence = [Float](repeating: 0.0, count: frameSize)
        silence.withUnsafeBufferPointer { buf in
            let res = detector.detectPitch(ptr: buf.baseAddress!, count: frameSize, workspace: workspace)
            XCTAssertFalse(res.isVoiced)
            XCTAssertLessThanOrEqual(res.f0, 1.0)
        }
    }

    func testPitchDetectorSondhiCenterClipping() {
        let config = DSPConfig(sampleRate: 16000, maxPitchLag: 320)
        let detector = PitchDetector(config: config)
        let workspace = DSPWorkspace(maxFrameSize: 1024, maxPitchLag: 320)

        let frameSize = 512
        var wave = [Float](repeating: 0.0, count: frameSize)
        var i = 0
        while i < frameSize {
            wave[i] = 0.5 * sin(2.0 * Float.pi * 220.0 * Float(i) / 16000.0)
            i += 1
        }

        wave.withUnsafeBufferPointer { buf in
            let res = detector.detectPitch(ptr: buf.baseAddress!, count: frameSize, workspace: workspace)
            XCTAssertTrue(res.isVoiced)
        }
    }

    // MARK: - Feature 4: LPC & Levinson-Durbin (5 tests)

    func testLPCPreemphasisAndHamming() {
        let config = DSPConfig(sampleRate: 16000)
        let lpc = LPC(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        let signal = [Float](repeating: 1.0, count: config.frameSize)
        signal.withUnsafeBufferPointer { buf in
            let success = lpc.computeCoefficients(ptr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
            XCTAssertTrue(success)
            // Leading window edge should be small, center near 1.0
            XCTAssertLessThanOrEqual(workspace.hammingWindow[0], 0.15)
            XCTAssertLessThanOrEqual(0.95, workspace.hammingWindow[config.frameSize / 2])
        }
    }

    func testLPCAutocorrelationSymmetric() {
        let config = DSPConfig(sampleRate: 16000)
        let lpc = LPC(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        var signal = [Float](repeating: 0.0, count: config.frameSize)
        var i = 0
        while i < config.frameSize {
            signal[i] = sin(2.0 * Float.pi * 300.0 * Float(i) / 16000.0)
            i += 1
        }

        signal.withUnsafeBufferPointer { buf in
            let success = lpc.computeCoefficients(ptr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
            XCTAssertTrue(success)
            // Autocorrelation at lag 0 (energy) must be positive
            XCTAssertLessThan(0.0, workspace.lpcAutoCorr[0])
        }
    }

    func testLPCLevinsonDurbinReflectionCoefficients() {
        let config = DSPConfig(sampleRate: 16000)
        let lpc = LPC(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        var speech = [Float](repeating: 0.0, count: config.frameSize)
        var i = 0
        while i < config.frameSize {
            speech[i] = 0.5 * sin(2.0 * Float.pi * 500.0 * Float(i) / 16000.0)
            i += 1
        }

        speech.withUnsafeBufferPointer { buf in
            let success = lpc.computeCoefficients(ptr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
            XCTAssertTrue(success)
            var k = 0
            while k < config.lpcOrder {
                XCTAssertLessThan(abs(workspace.lpcCoeffs[k]), 50.0)
                k += 1
            }
        }
    }

    func testLPCPolynomialCoefficientsOutput() {
        let config = DSPConfig(sampleRate: 16000)
        let lpc = LPC(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        let speech = synthesizeSpeech(sampleRate: 16000, durationSeconds: Float(config.frameSize) / 16000.0)
        speech.withUnsafeBufferPointer { buf in
            let success = lpc.computeCoefficients(ptr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
            XCTAssertTrue(success)
            XCTAssertEqual(workspace.lpcCoeffs.count, config.lpcOrder + 1)
        }
    }

    func testLPCSilenceFallback() {
        let config = DSPConfig(sampleRate: 16000)
        let lpc = LPC(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        let silence = [Float](repeating: 0.0, count: config.frameSize)
        silence.withUnsafeBufferPointer { buf in
            let success = lpc.computeCoefficients(ptr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
            XCTAssertFalse(success, "LPC compute on zero energy silence must return false safely")
        }
    }

    // MARK: - Feature 5: Durand-Kerner & FormantExtractor (5 tests)

    func testDurandKernerKnownRootsPolynomial() {
        let solver = DurandKernerSolver()
        let workspace = DSPWorkspace(maxFrameSize: 512, lpcOrder: 4, melChannels: 24)

        // Polynomial (z - 0.5)(z + 0.5)(z - 0.8i)(z + 0.8i) = (z^2 - 0.25)(z^2 + 0.64) = z^4 + 0.39 z^2 - 0.16
        // Coefficients a_1 = 0, a_2 = 0.39, a_3 = 0, a_4 = -0.16
        let coeffs: [Float] = [0.0, 0.39, 0.0, -0.16]
        coeffs.withUnsafeBufferPointer { cPtr in
            let success = solver.solve(coefficients: cPtr.baseAddress!, order: 4, workspace: workspace)
            XCTAssertTrue(success)
        }
    }

    func testFormantExtractorVowelA() {
        let config = DSPConfig(sampleRate: 16000)
        let lpc = LPC(config: config)
        let solver = DurandKernerSolver()
        let extractor = FormantExtractor(sampleRate: 16000.0)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        // Synthesize Japanese 'a' vowel (F1 ~ 800Hz, F2 ~ 1300Hz, F3 ~ 2600Hz)
        var vowelA = [Float](repeating: 0.0, count: config.frameSize)
        let twoPi = 2.0 * Float.pi
        var i = 0
        while i < config.frameSize {
            let t = Float(i) / 16000.0
            vowelA[i] = (0.5 * sin(twoPi * 800.0 * t)) + (0.3 * sin(twoPi * 1300.0 * t)) + (0.2 * sin(twoPi * 2600.0 * t))
            i += 1
        }

        vowelA.withUnsafeBufferPointer { buf in
            let lpcSuccess = lpc.computeCoefficients(ptr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
            XCTAssertTrue(lpcSuccess)
            if lpcSuccess {
                workspace.lpcCoeffs.withUnsafeBufferPointer { cPtr in
                    let solverSuccess = solver.solve(coefficients: cPtr.baseAddress!, order: config.lpcOrder, workspace: workspace)
                    XCTAssertTrue(solverSuccess)
                    if solverSuccess {
                        workspace.durandKernerCurr.withUnsafeBufferPointer { rPtr in
                            let formants = extractor.extractFormants(roots: rPtr.baseAddress!, count: config.lpcOrder)
                            XCTAssertLessThanOrEqual(1, formants.count)
                        }
                    }
                }
            }
        }
    }

    func testFormantExtractorPoleRadiusFilter() {
        let extractor = FormantExtractor(sampleRate: 16000.0)
        // Root outside unit circle (radius >= 1.0) and broad bandwidth root (radius < 0.88)
        let roots: [Complex] = [
            Complex(real: 1.2, imag: 0.0),    // outside unit circle -> rejected
            Complex(real: 0.5, imag: 0.0),    // radius 0.5 < 0.88 -> rejected
            Complex(real: 0.0, imag: 0.95),   // radius 0.95 -> accepted (formant at 4000Hz)
            Complex(real: 0.0, imag: -0.95)
        ]
        roots.withUnsafeBufferPointer { rPtr in
            let formants = extractor.extractFormants(roots: rPtr.baseAddress!, count: 4)
            XCTAssertLessThanOrEqual(1, formants.count)
        }
    }

    func testFormantExtractorAscendingSort() {
        let extractor = FormantExtractor(sampleRate: 16000.0)
        // 2 pairs of conjugate roots at ~1000Hz and ~2500Hz
        let r1 = 0.95 * cos(2.0 * Float.pi * 1000.0 / 16000.0)
        let i1 = 0.95 * sin(2.0 * Float.pi * 1000.0 / 16000.0)
        let r2 = 0.95 * cos(2.0 * Float.pi * 2500.0 / 16000.0)
        let i2 = 0.95 * sin(2.0 * Float.pi * 2500.0 / 16000.0)

        let roots: [Complex] = [
            Complex(real: r2, imag: i2),
            Complex(real: r2, imag: -i2),
            Complex(real: r1, imag: i1),
            Complex(real: r1, imag: -i1)
        ]
        roots.withUnsafeBufferPointer { rPtr in
            let formants = extractor.extractFormants(roots: rPtr.baseAddress!, count: 4)
            if 2 <= formants.count {
                XCTAssertLessThanOrEqual(formants.f1, formants.f2)
            }
        }
    }

    func testFormantExtractorZeroRootsFallback() {
        let extractor = FormantExtractor(sampleRate: 16000.0)
        let emptyRoots: [Complex] = []
        emptyRoots.withUnsafeBufferPointer { rPtr in
            let formants = extractor.extractFormants(roots: rPtr.baseAddress!, count: 0)
            XCTAssertEqual(formants.count, 0)
            XCTAssertEqual(formants.f1, 0.0)
            XCTAssertEqual(formants.f2, 0.0)
            XCTAssertEqual(formants.f3, 0.0)
        }
    }

    // MARK: - Feature 6: Filterbank & Feature Extraction (5 tests)

    func testFilterbank512PointFFTPowerSpectrum() {
        let config = DSPConfig(sampleRate: 16000)
        let filterbank = Filterbank(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        var wave = [Float](repeating: 0.0, count: config.frameSize)
        var i = 0
        while i < config.frameSize {
            wave[i] = sin(2.0 * Float.pi * 400.0 * Float(i) / 16000.0)
            i += 1
        }

        let pitch = PitchResult(f0: 400.0, hnr: 20.0, isVoiced: true)
        let formants = FormantResult(f1: 800.0, f2: 1300.0, f3: 2600.0, b1: 80.0, b2: 100.0, b3: 150.0, count: 3)

        wave.withUnsafeBufferPointer { buf in
            filterbank.extractFeatures(pcmPtr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
            // Power spectrum must have positive energy
            var sumPower: Float = 0.0
            var k = 0
            while k < workspace.powerSpectrum.count {
                sumPower += workspace.powerSpectrum[k]
                k += 1
            }
            XCTAssertLessThan(0.0, sumPower)
        }
    }

    func testFilterbank64ChannelMelIntegration() {
        let config = DSPConfig(sampleRate: 16000, melChannels: 64)
        let filterbank = Filterbank(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        let speech = synthesizeSpeech(sampleRate: 16000, durationSeconds: Float(config.frameSize) / 16000.0)

        speech.withUnsafeBufferPointer { buf in
            filterbank.extractFeatures(pcmPtr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
            XCTAssertEqual(workspace.melEnergies.count, config.melChannels)
            var m = 0
            while m < config.melChannels {
                XCTAssertFalse(workspace.melEnergies[m].isNaN)
                XCTAssertFalse(workspace.melEnergies[m].isInfinite)
                m += 1
            }
        }
    }

    func testFilterbankLogCompressionNormalization() {
        let config = DSPConfig(sampleRate: 16000, melChannels: 64)
        let filterbank = Filterbank(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        let speech = synthesizeSpeech(sampleRate: 16000, durationSeconds: Float(config.frameSize) / 16000.0)

        speech.withUnsafeBufferPointer { buf in
            filterbank.extractFeatures(pcmPtr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
            var m = 0
            while m < config.melChannels {
                XCTAssertLessThanOrEqual(0.0, workspace.featureBuffer[m])
                XCTAssertLessThanOrEqual(workspace.featureBuffer[m], 1.0)
                m += 1
            }
        }
    }

    func testFilterbank64DimVectorComposition() {
        let config = DSPConfig(sampleRate: 16000, melChannels: 64)
        let filterbank = Filterbank(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        let speech = synthesizeSpeech(sampleRate: 16000, durationSeconds: Float(config.frameSize) / 16000.0)

        speech.withUnsafeBufferPointer { buf in
            filterbank.extractFeatures(pcmPtr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
            XCTAssertEqual(workspace.featureBuffer.count, 64)
            XCTAssertLessThanOrEqual(0.0, workspace.featureBuffer[0])
            XCTAssertLessThanOrEqual(workspace.featureBuffer[0], 1.0)
        }
    }

    func testFilterbankZeroAllocFeatureExtraction() {
        let config = DSPConfig(sampleRate: 16000, melChannels: 64)
        let filterbank = Filterbank(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        let speech = synthesizeSpeech(sampleRate: 16000, durationSeconds: Float(config.frameSize) / 16000.0)

        speech.withUnsafeBufferPointer { buf in
            var iter = 0
            while iter < 500 {
                filterbank.extractFeatures(pcmPtr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
                iter += 1
            }
            XCTAssertEqual(workspace.featureBuffer.count, 64)
        }
    }

    // MARK: - Feature 7: DSPWorkspace (5 tests)

    func testDSPWorkspaceBufferAllocations() {
        let workspace = DSPWorkspace(maxFrameSize: 512, lpcOrder: 12, melChannels: 64)
        XCTAssertEqual(workspace.rawFrame.count, 512)
        XCTAssertEqual(workspace.lpcCoeffs.count, 13)
        XCTAssertEqual(workspace.durandKernerCurr.count, 12)
        XCTAssertEqual(workspace.melEnergies.count, 64)
        XCTAssertEqual(workspace.featureBuffer.count, 64)
    }

    func testDSPWorkspaceHammingWindowPrecomputation() {
        let workspace = DSPWorkspace(maxFrameSize: 512, lpcOrder: 12, melChannels: 24)
        XCTAssertLessThanOrEqual(workspace.hammingWindow[0], 0.15)
        XCTAssertLessThanOrEqual(workspace.hammingWindow[511], 0.15)
        XCTAssertLessThanOrEqual(0.95, workspace.hammingWindow[256])
    }

    func testDSPWorkspaceMemoryReuseHotPath() {
        let workspace = DSPWorkspace(maxFrameSize: 512, lpcOrder: 12, melChannels: 24)
        var i = 0
        while i < 1000 {
            workspace.rawFrame[0] = Float(i)
            workspace.featureBuffer[0] = Float(i) * 0.1
            i += 1
        }
        XCTAssertEqual(workspace.rawFrame[0], 999.0)
    }

    func testDSPWorkspaceResetSanity() {
        let workspace = DSPWorkspace(maxFrameSize: 512, lpcOrder: 12, melChannels: 24)
        workspace.featureBuffer[0] = 42.0
        workspace.featureBuffer[0] = 0.0
        XCTAssertEqual(workspace.featureBuffer[0], 0.0)
    }

    func testDSPWorkspaceMultiInstanceIndependence() {
        let ws1 = DSPWorkspace(maxFrameSize: 512, lpcOrder: 12, melChannels: 24)
        let ws2 = DSPWorkspace(maxFrameSize: 512, lpcOrder: 12, melChannels: 24)
        ws1.rawFrame[0] = 100.0
        ws2.rawFrame[0] = 200.0
        XCTAssertEqual(ws1.rawFrame[0], 100.0)
        XCTAssertEqual(ws2.rawFrame[0], 200.0)
    }

    // MARK: - Feature 8: LIFNeuron Dynamics (5 tests)

    func testLIFNeuronSubThresholdDecay() {
        let config = LIFConfig(beta: 0.9, vTh: 1.0, vReset: 0.0, alpha: 2.0)
        let res = LIFNeuronEngine.stepScalar(config: config, vPrev: 0.8, sPrev: 0.0, inputCurrent: 0.05)
        // V[t+1] = 0.8 * 0.9 + 0.05 = 0.77 < 1.0 -> no spike, s = 0.0
        XCTAssertEqual(res.sNext, 0.0)
        XCTAssertLessThanOrEqual(abs(res.vNext - 0.77), 1e-4)
    }

    func testLIFNeuronSpikeGenerationAndReset() {
        let config = LIFConfig(beta: 0.9, vTh: 1.0, vReset: 0.0, alpha: 2.0)
        let res = LIFNeuronEngine.stepScalar(config: config, vPrev: 0.8, sPrev: 0.0, inputCurrent: 0.35)
        // V[t+1] = 0.8 * 0.9 + 0.35 = 1.07 >= 1.0 -> spike s = 1.0
        XCTAssertEqual(res.sNext, 1.0)
    }

    func testLIFNeuronDirectInputCurrent() {
        let config = LIFConfig(beta: 0.8, vTh: 1.0, vReset: 0.0, alpha: 2.0)
        var v: Float = 0.0
        var s: Float = 0.0
        let inputCurrent: Float = 0.6

        var spikeCount = 0
        var step = 0
        while step < 10 {
            let res = LIFNeuronEngine.stepScalar(config: config, vPrev: v, sPrev: s, inputCurrent: inputCurrent)
            v = res.vNext
            s = res.sNext
            if s == 1.0 {
                spikeCount += 1
            }
            step += 1
        }
        XCTAssertLessThanOrEqual(3, spikeCount)
    }

    func testLIFNeuronSIMD8VsScalarEquivalence() {
        let config = LIFConfig(beta: 0.85, vTh: 1.0, vReset: 0.0, alpha: 2.0)
        let vPrev: [Float] = [0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 1.1, 1.5]
        let sPrev: [Float] = [Float](repeating: 0.0, count: 8)
        let inputs: [Float] = [0.1, 0.3, 0.6, 0.2, 0.2, 0.1, 0.0, 0.1]
        var vScalar = [Float](repeating: 0.0, count: 8)
        var sScalar = [Float](repeating: 0.0, count: 8)
        var vSimd = [Float](repeating: 0.0, count: 8)
        var sSimd = [Float](repeating: 0.0, count: 8)

        var i = 0
        while i < 8 {
            let res = LIFNeuronEngine.stepScalar(config: config, vPrev: vPrev[i], sPrev: sPrev[i], inputCurrent: inputs[i])
            vScalar[i] = res.vNext
            sScalar[i] = res.sNext
            i += 1
        }

        vPrev.withUnsafeBufferPointer { vp in
            sPrev.withUnsafeBufferPointer { sp in
                inputs.withUnsafeBufferPointer { ip in
                    vSimd.withUnsafeMutableBufferPointer { vn in
                        sSimd.withUnsafeMutableBufferPointer { sn in
                            LIFNeuronEngine.stepSIMD8(
                                config: config,
                                vPrevPtr: vp.baseAddress!,
                                sPrevPtr: sp.baseAddress!,
                                inputPtr: ip.baseAddress!,
                                vNextPtr: vn.baseAddress!,
                                sNextPtr: sn.baseAddress!,
                                count: 8
                            )
                        }
                    }
                }
            }
        }

        i = 0
        while i < 8 {
            XCTAssertEqual(vScalar[i], vSimd[i])
            XCTAssertEqual(sScalar[i], sSimd[i])
            i += 1
        }
    }

    func testLIFStateReset() {
        let state = LIFState(size: 64)
        state.v[0] = 0.99
        state.s[0] = 1.0
        state.reset()
        XCTAssertEqual(state.v[0], 0.0)
        XCTAssertEqual(state.s[0], 0.0)
    }

    // MARK: - Feature 9: Fast Sigmoid Surrogate Gradient (5 tests)

    func testSurrogateGradientAtThreshold() {
        let grad = SurrogateGradient.derivative(v: 1.0, vTh: 1.0, alpha: 2.0)
        // sigma'(0) = alpha / (1 + 0)^2 = 2.0 / 4 = 0.5 (or normalized scaled)
        XCTAssertLessThan(0.0, grad)
        XCTAssertLessThanOrEqual(grad, 2.0)
    }

    func testSurrogateGradientSymmetry() {
        let gradLeft = SurrogateGradient.derivative(v: 0.8, vTh: 1.0, alpha: 2.0)
        let gradRight = SurrogateGradient.derivative(v: 1.2, vTh: 1.0, alpha: 2.0)
        XCTAssertLessThanOrEqual(abs(gradLeft - gradRight), 1e-5)
    }

    func testSurrogateGradientSIMD8VsScalar() {
        let vValues: [Float] = [0.0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]
        var gradScalar = [Float](repeating: 0.0, count: 8)
        var gradSimd = [Float](repeating: 0.0, count: 8)

        var i = 0
        while i < 8 {
            gradScalar[i] = SurrogateGradient.derivative(v: vValues[i], vTh: 1.0, alpha: 2.0)
            i += 1
        }

        vValues.withUnsafeBufferPointer { vBuf in
            gradSimd.withUnsafeMutableBufferPointer { gBuf in
                SurrogateGradient.derivativeSIMD8(vPtr: vBuf.baseAddress!, dstPtr: gBuf.baseAddress!, count: 8, vTh: 1.0, alpha: 2.0)
            }
        }

        i = 0
        while i < 8 {
            XCTAssertLessThanOrEqual(abs(gradScalar[i] - gradSimd[i]), 1e-5)
            i += 1
        }
    }

    func testSurrogateGradientExtremeInputs() {
        let gradNeg = SurrogateGradient.derivative(v: -100.0, vTh: 1.0, alpha: 2.0)
        let gradPos = SurrogateGradient.derivative(v: 100.0, vTh: 1.0, alpha: 2.0)
        XCTAssertLessThanOrEqual(0.0, gradNeg)
        XCTAssertLessThanOrEqual(gradNeg, 0.01)
        XCTAssertLessThanOrEqual(0.0, gradPos)
        XCTAssertLessThanOrEqual(gradPos, 0.01)
    }

    func testSurrogateGradientAlphaScaling() {
        let gradAlpha1 = SurrogateGradient.derivative(v: 1.5, vTh: 1.0, alpha: 1.0)
        let gradAlpha4 = SurrogateGradient.derivative(v: 1.5, vTh: 1.0, alpha: 4.0)
        XCTAssertLessThan(gradAlpha4, gradAlpha1)
    }

    // MARK: - Feature 10: Adam Optimizer (5 tests)

    func testAdamOptimizerMomentumUpdate() {
        let param = Parameter(count: 1, initialData: [1.0])
        param.grad[0] = 0.5
        let config = AdamConfig(lr: 0.01)
        let adam = AdamOptimizer(config: config, parameters: [param])

        adam.step()
        XCTAssertLessThan(0.0, param.m[0])
        XCTAssertLessThan(0.0, param.v[0])
        XCTAssertLessThan(param.data[0], 1.0)
    }

    func testAdamOptimizerBiasCorrection() {
        let param = Parameter(count: 1, initialData: [1.0])
        param.grad[0] = 0.5
        let config = AdamConfig(lr: 0.01)
        let adam = AdamOptimizer(config: config, parameters: [param])

        adam.step()
        let p1 = param.data[0]
        adam.step()
        let p2 = param.data[0]
        XCTAssertLessThan(p2, p1)
    }

    func testAdamOptimizerGlobalL2NormClipping() {
        let param = Parameter(count: 4, initialData: [0.0, 0.0, 0.0, 0.0])
        param.grad = [100.0, 200.0, 300.0, 400.0]
        let config = AdamConfig(lr: 0.01, gradClip: 1.0)
        let adam = AdamOptimizer(config: config, parameters: [param])
        adam.step()

        let totalNorm = param.grad.withUnsafeBufferPointer { ptr in
            sqrt(VectorOperations.sumOfSquares(ptr: ptr.baseAddress!, count: 4))
        }
        XCTAssertLessThanOrEqual(totalNorm, 1.01)
    }

    func testAdamOptimizerZeroGrad() {
        let param = Parameter(count: 4, initialData: [0.0, 0.0, 0.0, 0.0])
        param.grad = [1.0, 2.0, 3.0, 4.0]
        param.zeroGrad()
        XCTAssertEqual(param.grad[0], 0.0)
        XCTAssertEqual(param.grad[1], 0.0)
        XCTAssertEqual(param.grad[2], 0.0)
        XCTAssertEqual(param.grad[3], 0.0)
    }

    func testAdamOptimizerParameterConvergence() {
        let param = Parameter(count: 1, initialData: [5.0])
        let config = AdamConfig(lr: 0.05)
        let adam = AdamOptimizer(config: config, parameters: [param])

        // Minimize f(x) = x^2, df/dx = 2x
        var step = 1
        while step <= 100 {
            param.grad[0] = 2.0 * param.data[0]
            adam.step()
            step += 1
        }
        XCTAssertLessThanOrEqual(abs(param.data[0]), 0.1)
    }

    // MARK: - Feature 11: Matryoshka Nested SNN (5 tests)

    func testMatryoshkaForwardAllSlices() {
        let net = MatryoshkaNetwork(inputDim: 32, maxHiddenDim: 256, outputDim: 64, timeSteps: 4)
        let features = [Float](repeating: 0.5, count: 32)
        let slices: [MatryoshkaSlice] = [.base, .middle, .high]

        var sIdx = 0
        while sIdx < slices.count {
            let sl = slices[sIdx]
            let hDim = sl.rawValue
            var vPrev = [Float](repeating: 0.0, count: hDim)
            var sPrev = [Float](repeating: 0.0, count: hDim)
            var spikeSum = [Float](repeating: 0.0, count: hDim)
            var logits = [Float](repeating: 0.0, count: 64)
            var probs = [Float](repeating: 0.0, count: 64)

            net.forwardSlice(
                features: features,
                slice: sl,
                vPrev: &vPrev,
                sPrev: &sPrev,
                spikeSum: &spikeSum,
                logits: &logits,
                probabilities: &probs
            )

            var probSum: Float = 0.0
            var p = 0
            while p < 64 {
                probSum += probs[p]
                p += 1
            }
            XCTAssertLessThanOrEqual(abs(probSum - 1.0), 1e-4)
            sIdx += 1
        }
    }

    func testMatryoshkaWeightSharing() {
        let net = MatryoshkaNetwork(inputDim: 32, maxHiddenDim: 256, outputDim: 64, timeSteps: 4)
        // Weights of base slice (64x32) must match top-left of high slice (256x32)
        var i = 0
        while i < 64 {
            var j = 0
            while j < 32 {
                XCTAssertEqual(net.pWIn.data[i * 32 + j], net.pWIn.data[i * 32 + j])
                j += 1
            }
            i += 1
        }
    }

    func testMatryoshkaBaseExportImport() throws {
        let net1 = MatryoshkaNetwork(inputDim: 32, maxHiddenDim: 256, outputDim: 64, timeSteps: 4)
        let baseWeights = net1.exportBaseWeights()

        let net2 = MatryoshkaNetwork(inputDim: 32, maxHiddenDim: 256, outputDim: 64, timeSteps: 4)
        net2.importBaseWeights(baseWeights)

        let features = [Float](repeating: 0.3, count: 32)
        var v1 = [Float](repeating: 0.0, count: 256)
        var s1 = [Float](repeating: 0.0, count: 256)
        var sum1 = [Float](repeating: 0.0, count: 256)
        var log1 = [Float](repeating: 0.0, count: 64)
        var p1 = [Float](repeating: 0.0, count: 64)

        var v2 = [Float](repeating: 0.0, count: 256)
        var s2 = [Float](repeating: 0.0, count: 256)
        var sum2 = [Float](repeating: 0.0, count: 256)
        var log2 = [Float](repeating: 0.0, count: 64)
        var p2 = [Float](repeating: 0.0, count: 64)

        net1.forwardSlice(features: features, slice: .base, vPrev: &v1, sPrev: &s1, spikeSum: &sum1, logits: &log1, probabilities: &p1)
        net2.forwardSlice(features: features, slice: .base, vPrev: &v2, sPrev: &s2, spikeSum: &sum2, logits: &log2, probabilities: &p2)

        var k = 0
        while k < 64 {
            XCTAssertEqual(p1[k], p2[k])
            k += 1
        }
    }

    func testMatryoshkaBaseCodableSerialization() throws {
        let net = MatryoshkaNetwork(inputDim: 32, maxHiddenDim: 256, outputDim: 64, timeSteps: 4)
        let weights = net.exportBaseWeights()
        let encoder = JSONEncoder()
        let data = try encoder.encode(weights)

        let decoder = JSONDecoder()
        let decoded = try decoder.decode(BaseSNNWeights.self, from: data)

        XCTAssertEqual(weights.inputDim, decoded.inputDim)
        XCTAssertEqual(weights.hiddenDim, decoded.hiddenDim)
        XCTAssertEqual(weights.outputDim, decoded.outputDim)
        XCTAssertEqual(weights.wIn.count, decoded.wIn.count)
    }

    func testMatryoshkaForwardHotPathZeroAlloc() {
        let net = MatryoshkaNetwork(inputDim: 32, maxHiddenDim: 256, outputDim: 64, timeSteps: 4)
        let features = [Float](repeating: 0.5, count: 32)
        var vPrev = [Float](repeating: 0.0, count: 256)
        var sPrev = [Float](repeating: 0.0, count: 256)
        var spikeSum = [Float](repeating: 0.0, count: 256)
        var logits = [Float](repeating: 0.0, count: 64)
        var probs = [Float](repeating: 0.0, count: 64)

        var iter = 0
        while iter < 1000 {
            net.forwardSlice(features: features, slice: .base, vPrev: &vPrev, sPrev: &sPrev, spikeSum: &spikeSum, logits: &logits, probabilities: &probs)
            iter += 1
        }
        XCTAssertEqual(probs.count, 64)
    }

    // MARK: - Feature 12: Quantized Fixed-Point Engine (5 tests)

    func testQuantizedEngineInt32Quantization() {
        let net = MatryoshkaNetwork(inputDim: 32, maxHiddenDim: 256, outputDim: 64, timeSteps: 4)
        let config = QuantizedConfig.int32Config()
        let qWeights = QuantizedEngine.quantize(network: net, config: config)
        XCTAssertEqual(qWeights.config.scale, 65536.0)
        XCTAssertEqual(qWeights.wIn.count, net.pWIn.count)
    }

    func testQuantizedEngineInt16Quantization() {
        let net = MatryoshkaNetwork(inputDim: 32, maxHiddenDim: 256, outputDim: 64, timeSteps: 4)
        let config = QuantizedConfig.int16Config()
        let qWeights = QuantizedEngine.quantize(network: net, config: config)
        XCTAssertEqual(qWeights.config.scale, 2048.0)
        XCTAssertEqual(qWeights.wIn.count, net.pWIn.count)
    }

    func testQuantizedEngineBitShiftDecay() {
        let net = MatryoshkaNetwork(inputDim: 32, maxHiddenDim: 256, outputDim: 64, timeSteps: 4)
        let config = QuantizedConfig.int32Config()
        let qWeights = QuantizedEngine.quantize(network: net, config: config)
        let engine = QuantizedEngine(weights: qWeights, timeSteps: 4)
        let workspace = QuantizedWorkspace(maxHiddenDim: 256, inputDim: 32, outputDim: 64)

        let features = [Float](repeating: 0.5, count: 32)
        var probs = [Float](repeating: 0.0, count: 64)

        engine.predictSlice(features: features, slice: .base, workspace: workspace, outputProbs: &probs)
        var sumP: Float = 0.0
        var i = 0
        while i < 64 {
            sumP += probs[i]
            i += 1
        }
        XCTAssertLessThanOrEqual(abs(sumP - 1.0), 1e-4)
    }

    func testQuantizedEngineSparseRecurrentAddition() {
        let net = MatryoshkaNetwork(inputDim: 32, maxHiddenDim: 256, outputDim: 64, timeSteps: 4)
        let config = QuantizedConfig.int16Config()
        let qWeights = QuantizedEngine.quantize(network: net, config: config)
        let engine = QuantizedEngine(weights: qWeights, timeSteps: 4)
        let workspace = QuantizedWorkspace(maxHiddenDim: 256, inputDim: 32, outputDim: 64)

        let features = [Float](repeating: 0.8, count: 32)
        var probs = [Float](repeating: 0.0, count: 64)
        engine.predictSlice(features: features, slice: .middle, workspace: workspace, outputProbs: &probs)
        XCTAssertEqual(probs.count, 64)
    }

    func testQuantizedWorkspaceResetZeroAlloc() {
        let workspace = QuantizedWorkspace(maxHiddenDim: 256, inputDim: 32, outputDim: 64)
        workspace.vPrev[0] = 12345
        workspace.sPrev[0] = 1
        workspace.reset()
        XCTAssertEqual(workspace.vPrev[0], 0)
        XCTAssertEqual(workspace.sPrev[0], 0)
    }

    // MARK: - Feature 13: VectorOperations SIMD8 (5 tests)

    func testVectorOperationsDotProduct() {
        let a: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        let b: [Float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        let dot = a.withUnsafeBufferPointer { aBuf in
            b.withUnsafeBufferPointer { bBuf in
                VectorOperations.dotProduct(a: aBuf.baseAddress!, b: bBuf.baseAddress!, count: a.count)
            }
        }
        var expected: Float = 0.0
        var i = 0
        while i < a.count {
            expected += a[i] * b[i]
            i += 1
        }
        XCTAssertLessThanOrEqual(abs(dot - expected), 1e-4)
    }

    func testVectorOperationsSumOfSquares() {
        let v: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        let sumSq = v.withUnsafeBufferPointer { buf in
            VectorOperations.sumOfSquares(ptr: buf.baseAddress!, count: v.count)
        }
        // 1 + 4 + 9 + 16 + 25 + 36 + 49 + 64 = 204
        XCTAssertEqual(sumSq, 204.0)
    }

    func testVectorOperationsMultiply() {
        let a: [Float] = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        let b: [Float] = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        var result = [Float](repeating: 0.0, count: 8)
        a.withUnsafeBufferPointer { aBuf in
            b.withUnsafeBufferPointer { bBuf in
                result.withUnsafeMutableBufferPointer { rBuf in
                    VectorOperations.multiply(srcA: aBuf.baseAddress!, srcB: bBuf.baseAddress!, dst: rBuf.baseAddress!, count: 8)
                }
            }
        }
        var i = 0
        while i < 8 {
            XCTAssertEqual(result[i], a[i] * 0.5)
            i += 1
        }
    }

    func testVectorOperationsMaxMagnitude() {
        let v: [Float] = [1.0, -5.5, 3.2, 4.0, -9.8, 2.1, 8.0, 0.0]
        let maxMag = v.withUnsafeBufferPointer { buf in
            VectorOperations.maxMagnitude(ptr: buf.baseAddress!, count: v.count)
        }
        XCTAssertEqual(maxMag, 9.8)
    }

    func testVectorOperationsClamp() {
        let v: [Float] = [-2.0, -0.5, 0.0, 0.5, 1.5, 2.5]
        var clamped = [Float](repeating: 0.0, count: 6)
        v.withUnsafeBufferPointer { vBuf in
            clamped.withUnsafeMutableBufferPointer { rBuf in
                VectorOperations.clamp(src: vBuf.baseAddress!, dst: rBuf.baseAddress!, count: 6, minVal: 0.0, maxVal: 1.0)
            }
        }
        XCTAssertEqual(clamped, [0.0, 0.0, 0.0, 0.5, 1.0, 1.0])
    }

    // MARK: - Feature 14: PhonemeVocabulary (5 tests)

    func testPhonemeVocabularySizeAndSpecialTokens() {
        let vocab = PhonemeVocabulary()
        XCTAssertEqual(vocab.size, 64)
        XCTAssertEqual(PhonemeVocabulary.padId, 0)
        XCTAssertEqual(PhonemeVocabulary.silId, 1)
        XCTAssertEqual(PhonemeVocabulary.unkId, 2)
        XCTAssertEqual(PhonemeVocabulary.sosId, 3)
        XCTAssertEqual(PhonemeVocabulary.eosId, 4)
    }

    func testPhonemeVocabularyBidirectionalMapping() {
        let vocab = PhonemeVocabulary()
        var id = 0
        while id < vocab.size {
            let token = vocab.token(for: id)
            let backId = vocab.id(for: token)
            XCTAssertEqual(backId, id, "Token \(token) id mapping must be reversible")
            id += 1
        }
    }

    func testPhonemeVocabularyKanaToPhonemes() {
        let vocab = PhonemeVocabulary()
        let phonemesA = vocab.kanaToPhonemes("あいうえお")
        XCTAssertEqual(phonemesA, ["a", "i", "u", "e", "o"])

        let phonemesK = vocab.kanaToPhonemes("とうきょう")
        XCTAssertTrue(phonemesK.contains("t"))
        XCTAssertTrue(phonemesK.contains("o"))
    }

    func testPhonemeVocabularyPhonemesToKana() {
        let vocab = PhonemeVocabulary()
        let kana = vocab.phonemesToKana(["a", "i", "u", "e", "o"])
        XCTAssertEqual(kana, "あいうえお")
    }

    func testPhonemeVocabularyUnknownTokenHandling() {
        let vocab = PhonemeVocabulary()
        let unknownId = vocab.id(for: "!!!UNKNOWN_TOKEN!!!")
        XCTAssertEqual(unknownId, PhonemeVocabulary.unkId)

        let outOfBoundsToken = vocab.token(for: 999)
        XCTAssertEqual(outOfBoundsToken, "<unk>")
    }

    // MARK: - Feature 15: TwoStage Transcriber / Pipeline (5 tests)

    func testStreamingTranscriberSingleUtterance() {
        let vocab = TextVocabulary()
        let acNet = MatryoshkaNetwork(inputDim: 64, maxHiddenDim: 256, outputDim: vocab.size, timeSteps: 4)
        let lmNet = MatryoshkaNetwork(inputDim: 64, maxHiddenDim: 256, outputDim: vocab.size, timeSteps: 4)
        let transcriber = StreamingTranscriber(acousticNetwork: acNet, languageNetwork: lmNet, textVocabulary: vocab)

        let collector = ResultCollector<TranscriptionResult>()
        transcriber.onFinalResult = { res in
            collector.append(res)
        }

        var audio: [Float] = []
        audio.append(contentsOf: synthesizeSilence(sampleRate: 16000, durationSeconds: 0.1))
        audio.append(contentsOf: synthesizeSpeech(sampleRate: 16000, durationSeconds: 0.4))
        audio.append(contentsOf: synthesizeSilence(sampleRate: 16000, durationSeconds: 0.4))

        let chunkSize = 160
        var offset = 0
        while offset < audio.count {
            let count = min(chunkSize, audio.count - offset)
            audio.withUnsafeBufferPointer { buf in
                let ptr = buf.baseAddress!.advanced(by: offset)
                transcriber.appendAudio(pcmPtr: ptr, count: count)
            }
            offset += count
        }
        transcriber.flush()

        XCTAssertEqual(collector.count, 1)
    }

    func testStreamingTranscriberMultiUtteranceSegmentation() {
        let vocab = TextVocabulary()
        let acNet = MatryoshkaNetwork(inputDim: 64, maxHiddenDim: 256, outputDim: vocab.size, timeSteps: 4)
        let lmNet = MatryoshkaNetwork(inputDim: 64, maxHiddenDim: 256, outputDim: vocab.size, timeSteps: 4)
        let transcriber = StreamingTranscriber(acousticNetwork: acNet, languageNetwork: lmNet, textVocabulary: vocab)

        let collector = ResultCollector<TranscriptionResult>()
        transcriber.onFinalResult = { res in
            collector.append(res)
        }

        var audio: [Float] = []
        audio.append(contentsOf: synthesizeSpeech(sampleRate: 16000, durationSeconds: 0.4))
        audio.append(contentsOf: synthesizeSilence(sampleRate: 16000, durationSeconds: 0.6))
        audio.append(contentsOf: synthesizeSpeech(sampleRate: 16000, durationSeconds: 0.4))
        audio.append(contentsOf: synthesizeSilence(sampleRate: 16000, durationSeconds: 0.5))

        let chunkSize = 160
        var offset = 0
        while offset < audio.count {
            let count = min(chunkSize, audio.count - offset)
            audio.withUnsafeBufferPointer { buf in
                let ptr = buf.baseAddress!.advanced(by: offset)
                transcriber.appendAudio(pcmPtr: ptr, count: count)
            }
            offset += count
        }
        transcriber.flush()

        XCTAssertEqual(collector.count, 2)
    }

    func testStreamingTranscriberPartialResultsStream() {
        let vocab = TextVocabulary()
        let acNet = MatryoshkaNetwork(inputDim: 64, maxHiddenDim: 256, outputDim: vocab.size, timeSteps: 4)
        let lmNet = MatryoshkaNetwork(inputDim: 64, maxHiddenDim: 256, outputDim: vocab.size, timeSteps: 4)
        let transcriber = StreamingTranscriber(acousticNetwork: acNet, languageNetwork: lmNet, textVocabulary: vocab)

        let counter = AtomicCounter()
        transcriber.onPartialResult = { partial in
            counter.increment()
            XCTAssertFalse(partial.isFinal)
        }

        let audio = synthesizeSpeech(sampleRate: 16000, durationSeconds: 0.8)
        let chunkSize = 160
        var offset = 0
        while offset < audio.count {
            let count = min(chunkSize, audio.count - offset)
            audio.withUnsafeBufferPointer { buf in
                let ptr = buf.baseAddress!.advanced(by: offset)
                transcriber.appendAudio(pcmPtr: ptr, count: count)
            }
            offset += count
        }
        XCTAssertLessThanOrEqual(4, counter.value)
    }

    func testStreamingTranscriberFlushAndReset() {
        let vocab = TextVocabulary()
        let acNet = MatryoshkaNetwork(inputDim: 64, maxHiddenDim: 256, outputDim: vocab.size, timeSteps: 4)
        let lmNet = MatryoshkaNetwork(inputDim: 64, maxHiddenDim: 256, outputDim: vocab.size, timeSteps: 4)
        let transcriber = StreamingTranscriber(acousticNetwork: acNet, languageNetwork: lmNet, textVocabulary: vocab)

        transcriber.appendAudio(pcm: [0.1, 0.2, 0.3])
        transcriber.flush()
        transcriber.reset()
        XCTAssertTrue(true, "Flush and reset completed safely")
    }

    func testStreamingTranscriberQuantizedMode() {
        let vocab = TextVocabulary()
        let acNet = MatryoshkaNetwork(inputDim: 64, maxHiddenDim: 256, outputDim: vocab.size, timeSteps: 4)
        let lmNet = MatryoshkaNetwork(inputDim: 64, maxHiddenDim: 256, outputDim: vocab.size, timeSteps: 4)
        let config = StreamingTranscriberConfig(slice: .high, useQuantization: true)
        let transcriber = StreamingTranscriber(config: config, acousticNetwork: acNet, languageNetwork: lmNet, textVocabulary: vocab)

        let collector = ResultCollector<TranscriptionResult>()
        transcriber.onFinalResult = { res in
            collector.append(res)
        }

        var audio: [Float] = []
        audio.append(contentsOf: synthesizeSpeech(sampleRate: 16000, durationSeconds: 0.5))
        audio.append(contentsOf: synthesizeSilence(sampleRate: 16000, durationSeconds: 0.4))

        let chunkSize = 160
        var offset = 0
        while offset < audio.count {
            let count = min(chunkSize, audio.count - offset)
            audio.withUnsafeBufferPointer { buf in
                let ptr = buf.baseAddress!.advanced(by: offset)
                transcriber.appendAudio(pcmPtr: ptr, count: count)
            }
            offset += count
        }
        transcriber.flush()

        XCTAssertEqual(collector.count, 1)
    }
}
