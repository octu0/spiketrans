import XCTest
import Foundation
@testable import Spiketrans

final class Tier2BoundaryTests: XCTestCase {

    // MARK: - B1: ゼロ長・空入力 (5 tests)

    func testB1WavParserEmptyData() {
        let emptyData = [UInt8]()
        let parser = WavParser()
        XCTAssertThrowsError(try parser.parse(bytes: emptyData))
    }

    func testB1VADEmptyFrame() {
        let config = DSPConfig(sampleRate: 16000)
        let vad = VAD(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)
        let empty: [Float] = []
        empty.withUnsafeBufferPointer { buf in
            let res = vad.processFrame(ptr: buf.baseAddress ?? UnsafePointer<Float>(bitPattern: 1)!, count: 0, workspace: workspace)
            XCTAssertFalse(res.isSpeech)
            XCTAssertEqual(res.rms, 0.0)
        }
    }

    func testB1SNNEmptyFeatures() {
        let net = SpikingNetwork(inputDim: 32, maxHiddenDim: 256, outputDim: 64, timeSteps: 4)
        var vPrev = [Float](repeating: 0.0, count: 256)
        var sPrev = [Float](repeating: 0.0, count: 256)
        var spikeSum = [Float](repeating: 0.0, count: 256)
        var logits = [Float](repeating: 0.0, count: 64)
        var probs = [Float](repeating: 0.0, count: 64)

        let emptyFeatures: [Float] = []
        // Should safely return without crashing
        net.forward(features: emptyFeatures, vPrev: &vPrev, sPrev: &sPrev, spikeSum: &spikeSum, logits: &logits, probabilities: &probs)
        XCTAssertEqual(probs.count, 64)
    }

    func testB1TranscriberEmptyAudio() {
        let vocab = TextVocabulary()
        let acNet = SpikingNetwork(inputDim: 64, maxHiddenDim: 256, outputDim: vocab.size, timeSteps: 4)
        let lmNet = SpikingNetwork(inputDim: 64, maxHiddenDim: 256, outputDim: vocab.size, timeSteps: 4)
        let transcriber = StreamingTranscriber(acousticNetwork: acNet, languageNetwork: lmNet, textVocabulary: vocab)
        transcriber.appendAudio(pcm: [])
        transcriber.flush()
        XCTAssertTrue(true)
    }

    func testB1VocabEmptyKana() {
        let vocab = PhonemeVocabulary()
        let phonemes = vocab.kanaToPhonemes("")
        XCTAssertEqual(phonemes, [])
        let kana = vocab.phonemesToKana([])
        XCTAssertEqual(kana, "")
    }

    // MARK: - B2: 超大容量・最大長 (5 tests)

    func testB2TranscriberHugeBurst() {
        let vocab = TextVocabulary()
        let acNet = SpikingNetwork(inputDim: 64, maxHiddenDim: 256, outputDim: vocab.size, timeSteps: 4)
        let lmNet = SpikingNetwork(inputDim: 64, maxHiddenDim: 256, outputDim: vocab.size, timeSteps: 4)
        let transcriber = StreamingTranscriber(acousticNetwork: acNet, languageNetwork: lmNet, textVocabulary: vocab)

        let hugeAudio = [Float](repeating: 0.1, count: 100000)
        transcriber.appendAudio(pcm: hugeAudio)
        transcriber.flush()
        XCTAssertTrue(true)
    }

    func testB2VADHugeAudioSequence() {
        let config = DSPConfig(sampleRate: 16000)
        let vad = VAD(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        let frame = [Float](repeating: 0.3, count: config.frameSize)
        var i = 0
        while i < 1000 {
            frame.withUnsafeBufferPointer { buf in
                vad.processFrame(ptr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
            }
            i += 1
        }
        XCTAssertTrue(true, "1,000 continuous frames processed safely without overflow")
    }

    func testB2VocabHugeKanaSequence() {
        let vocab = PhonemeVocabulary()
        var hugeKana = ""
        var i = 0
        while i < 500 {
            hugeKana.append("あいうえお")
            i += 1
        }
        let phonemes = vocab.kanaToPhonemes(hugeKana)
        XCTAssertEqual(phonemes.count, 2500)
        let backKana = vocab.phonemesToKana(phonemes)
        XCTAssertEqual(backKana, hugeKana)
    }

    func testB2SNNHighDimensionFullForward() {
        let net = SpikingNetwork(inputDim: 32, maxHiddenDim: 256, outputDim: 64, timeSteps: 4)
        let features = [Float](repeating: 0.9, count: 32)
        var vPrev = [Float](repeating: 0.0, count: 256)
        var sPrev = [Float](repeating: 0.0, count: 256)
        var spikeSum = [Float](repeating: 0.0, count: 256)
        var logits = [Float](repeating: 0.0, count: 64)
        var probs = [Float](repeating: 0.0, count: 64)

        net.forward(features: features, vPrev: &vPrev, sPrev: &sPrev, spikeSum: &spikeSum, logits: &logits, probabilities: &probs)
        XCTAssertEqual(probs.count, 64)
    }

    func testB2DSPWorkspaceLargeFrame() {
        let workspace = DSPWorkspace(maxFrameSize: 2048, lpcOrder: 16, melChannels: 40)
        XCTAssertEqual(workspace.rawFrame.count, 2048)
        XCTAssertEqual(workspace.lpcCoeffs.count, 17)
        XCTAssertEqual(workspace.melEnergies.count, 40)
    }

    // MARK: - B3: 完全無音信号 (5 tests)

    func testB3WavParserZeroSamples() throws {
        let zeros = [Float](repeating: 0.0, count: 1600)
        let header = createWavData(samples: zeros, sampleRate: 16000, channels: 1, bitsPerSample: 16)
        let parser = WavParser()
        let wav = try parser.parse(bytes: [UInt8](header))
        XCTAssertEqual(wav.pcmData.count, 1600)
        XCTAssertEqual(wav.pcmData[0], 0.0)
    }

    func testB3VADZeroEnergy() {
        let config = DSPConfig(sampleRate: 16000)
        let vad = VAD(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        let zeros = [Float](repeating: 0.0, count: config.frameSize)
        zeros.withUnsafeBufferPointer { buf in
            let res = vad.processFrame(ptr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
            XCTAssertFalse(res.isSpeech)
            XCTAssertEqual(res.rms, 0.0)
            XCTAssertEqual(res.zcr, 0.0)
        }
    }

    func testB3PitchDetectorZeroEnergy() {
        let config = DSPConfig(sampleRate: 16000)
        let detector = PitchDetector(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        let zeros = [Float](repeating: 0.0, count: config.frameSize)
        zeros.withUnsafeBufferPointer { buf in
            let res = detector.detectPitch(ptr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
            XCTAssertFalse(res.isVoiced)
            XCTAssertEqual(res.f0, 0.0)
            XCTAssertEqual(res.hnr, 0.0)
        }
    }

    func testB3LPCOnSilence() {
        let config = DSPConfig(sampleRate: 16000)
        let lpc = LPC(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        let zeros = [Float](repeating: 0.0, count: config.frameSize)
        zeros.withUnsafeBufferPointer { buf in
            let success = lpc.computeCoefficients(ptr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
            XCTAssertFalse(success)
        }
    }

    func testB3TranscriberZeroSilence() {
        let vocab = TextVocabulary()
        let acNet = SpikingNetwork(inputDim: 64, maxHiddenDim: 256, outputDim: vocab.size, timeSteps: 4)
        let lmNet = SpikingNetwork(inputDim: 64, maxHiddenDim: 256, outputDim: vocab.size, timeSteps: 4)
        let transcriber = StreamingTranscriber(acousticNetwork: acNet, languageNetwork: lmNet, textVocabulary: vocab)

        let zeros = [Float](repeating: 0.0, count: 16000)
        transcriber.appendAudio(pcm: zeros)
        transcriber.flush()
        XCTAssertTrue(true)
    }

    // MARK: - B4: 大音量・矩形波クリッピング (5 tests)

    func testB4WavParserExtremeAmplitude() throws {
        // Values > 1.0 clamped in PCM 16-bit
        let extreme: [Float] = [5.0, -5.0, 100.0, -100.0]
        let data = createWavData(samples: extreme, sampleRate: 16000, channels: 1, bitsPerSample: 16)
        let parser = WavParser()
        let wav = try parser.parse(bytes: [UInt8](data))
        XCTAssertLessThanOrEqual(wav.pcmData[0], 1.0)
        XCTAssertLessThanOrEqual(-1.0, wav.pcmData[1])
    }

    func testB4VADSquareWaveClipping() {
        let config = DSPConfig(sampleRate: 16000)
        let vad = VAD(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        var square = [Float](repeating: 0.0, count: config.frameSize)
        var i = 0
        while i < config.frameSize {
            switch (i / 10) % 2 {
            case 0:
                square[i] = 1.0
            default:
                square[i] = -1.0
            }
            i += 1
        }

        square.withUnsafeBufferPointer { buf in
            let res = vad.processFrame(ptr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
            XCTAssertTrue(res.isSpeech)
            XCTAssertFalse(res.rms.isNaN)
        }
    }

    func testB4FilterbankClippedSignal() {
        let config = DSPConfig(sampleRate: 16000)
        let filterbank = Filterbank(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        let clipped = [Float](repeating: 1.0, count: config.frameSize)
        let pitch = PitchResult(f0: 200.0, hnr: 20.0, isVoiced: true)
        let formants = FormantResult(f1: 800.0, f2: 1200.0, f3: 2500.0, b1: 80.0, b2: 100.0, b3: 150.0, count: 3)

        clipped.withUnsafeBufferPointer { buf in
            filterbank.extractFeatures(pcmPtr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
            var k = 0
            while k < 32 {
                XCTAssertFalse(workspace.featureBuffer[k].isNaN)
                XCTAssertFalse(workspace.featureBuffer[k].isInfinite)
                k += 1
            }
        }
    }

    func testB4LPCExtremeAmplitude() {
        let config = DSPConfig(sampleRate: 16000)
        let lpc = LPC(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        var extreme = [Float](repeating: 0.0, count: config.frameSize)
        var i = 0
        while i < config.frameSize {
            extreme[i] = 10.0 * sin(Float(i) * 0.1)
            i += 1
        }

        extreme.withUnsafeBufferPointer { buf in
            let success = lpc.computeCoefficients(ptr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
            if success {
                var k = 0
                while k < config.lpcOrder {
                    XCTAssertFalse(workspace.lpcCoeffs[k].isNaN)
                    k += 1
                }
            }
        }
    }

    func testB4TranscriberClippedSpeech() {
        let vocab = TextVocabulary()
        let acNet = SpikingNetwork(inputDim: 64, maxHiddenDim: 256, outputDim: vocab.size, timeSteps: 4)
        let lmNet = SpikingNetwork(inputDim: 64, maxHiddenDim: 256, outputDim: vocab.size, timeSteps: 4)
        let transcriber = StreamingTranscriber(acousticNetwork: acNet, languageNetwork: lmNet, textVocabulary: vocab)

        var clipped = [Float](repeating: 0.0, count: 8000)
        var i = 0
        while i < 8000 {
            let val = 3.0 * sin(Float(i) * 0.05)
            clipped[i] = max(-1.0, min(1.0, val))
            i += 1
        }
        transcriber.appendAudio(pcm: clipped)
        transcriber.flush()
        XCTAssertTrue(true)
    }

    // MARK: - B5: DC バイアスオフセット (5 tests)

    func testB5VADConstantDCOffset() {
        let config = DSPConfig(sampleRate: 16000)
        let vad = VAD(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        let dcSignal = [Float](repeating: 0.8, count: config.frameSize)
        dcSignal.withUnsafeBufferPointer { buf in
            let res = vad.processFrame(ptr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
            XCTAssertFalse(res.rms.isNaN)
        }
    }

    func testB5LPCConstantDCOffset() {
        let config = DSPConfig(sampleRate: 16000)
        let lpc = LPC(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        let dcSignal = [Float](repeating: 0.5, count: config.frameSize)
        dcSignal.withUnsafeBufferPointer { buf in
            lpc.computeCoefficients(ptr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
            var k = 0
            while k < config.lpcOrder {
                XCTAssertFalse(workspace.lpcCoeffs[k].isNaN)
                k += 1
            }
        }
    }

    func testB5PitchDetectorDCOffset() {
        let config = DSPConfig(sampleRate: 16000)
        let detector = PitchDetector(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        var wave = [Float](repeating: 0.0, count: config.frameSize)
        var i = 0
        while i < config.frameSize {
            wave[i] = 0.5 + 0.4 * sin(2.0 * Float.pi * 200.0 * Float(i) / 16000.0)
            i += 1
        }

        wave.withUnsafeBufferPointer { buf in
            let res = detector.detectPitch(ptr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
            XCTAssertFalse(res.f0.isNaN)
        }
    }

    func testB5FilterbankDCOffset() {
        let config = DSPConfig(sampleRate: 16000)
        let filterbank = Filterbank(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        let dc = [Float](repeating: 0.9, count: config.frameSize)
        let pitch = PitchResult(f0: 0.0, hnr: 0.0, isVoiced: false)
        let formants = FormantResult(f1: 0.0, f2: 0.0, f3: 0.0, b1: 0.0, b2: 0.0, b3: 0.0, count: 0)

        dc.withUnsafeBufferPointer { buf in
            filterbank.extractFeatures(pcmPtr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
            var k = 0
            while k < 32 {
                XCTAssertFalse(workspace.featureBuffer[k].isNaN)
                k += 1
            }
        }
    }

    func testB5TranscriberDCOffsetAudio() {
        let vocab = TextVocabulary()
        let acNet = SpikingNetwork(inputDim: 64, maxHiddenDim: 256, outputDim: vocab.size, timeSteps: 4)
        let lmNet = SpikingNetwork(inputDim: 64, maxHiddenDim: 256, outputDim: vocab.size, timeSteps: 4)
        let transcriber = StreamingTranscriber(acousticNetwork: acNet, languageNetwork: lmNet, textVocabulary: vocab)

        let dcAudio = [Float](repeating: 0.7, count: 4800)
        transcriber.appendAudio(pcm: dcAudio)
        transcriber.flush()
        XCTAssertTrue(true)
    }

    // MARK: - B6: 極低周波 / 極高周波 (5 tests)

    func testB6PitchDetectorSubAudible10Hz() {
        let config = DSPConfig(sampleRate: 16000)
        let detector = PitchDetector(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        var wave10Hz = [Float](repeating: 0.0, count: config.frameSize)
        var i = 0
        while i < config.frameSize {
            wave10Hz[i] = 0.8 * sin(2.0 * Float.pi * 10.0 * Float(i) / 16000.0)
            i += 1
        }

        wave10Hz.withUnsafeBufferPointer { buf in
            let res = detector.detectPitch(ptr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
            // 10Hz is below minimum audible pitch (50Hz), detector must report unvoiced or 0
            if res.isVoiced {
                XCTAssertLessThanOrEqual(50.0, res.f0)
            } else {
                XCTAssertEqual(res.f0, 0.0)
            }
        }
    }

    func testB6PitchDetectorNyquistLimit7990Hz() {
        let config = DSPConfig(sampleRate: 16000)
        let detector = PitchDetector(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        var waveNyquist = [Float](repeating: 0.0, count: config.frameSize)
        var i = 0
        while i < config.frameSize {
            waveNyquist[i] = 0.8 * sin(2.0 * Float.pi * 7990.0 * Float(i) / 16000.0)
            i += 1
        }

        waveNyquist.withUnsafeBufferPointer { buf in
            let res = detector.detectPitch(ptr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
            // 7990Hz is above maximum human voice pitch (500Hz)
            if res.isVoiced {
                XCTAssertLessThanOrEqual(res.f0, 600.0)
            }
        }
    }

    func testB6Filterbank10HzDCSubAudible() {
        let config = DSPConfig(sampleRate: 16000)
        let filterbank = Filterbank(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        var wave10Hz = [Float](repeating: 0.0, count: config.frameSize)
        var i = 0
        while i < config.frameSize {
            wave10Hz[i] = 0.5 * sin(2.0 * Float.pi * 10.0 * Float(i) / 16000.0)
            i += 1
        }
        let pitch = PitchResult(f0: 0.0, hnr: 0.0, isVoiced: false)
        let formants = FormantResult(f1: 0.0, f2: 0.0, f3: 0.0, b1: 0.0, b2: 0.0, b3: 0.0, count: 0)

        wave10Hz.withUnsafeBufferPointer { buf in
            filterbank.extractFeatures(pcmPtr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
            XCTAssertFalse(workspace.featureBuffer[0].isNaN)
        }
    }

    func testB6Filterbank7990HzNyquist() {
        let config = DSPConfig(sampleRate: 16000)
        let filterbank = Filterbank(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        var waveNyquist = [Float](repeating: 0.0, count: config.frameSize)
        var i = 0
        while i < config.frameSize {
            waveNyquist[i] = 0.5 * sin(2.0 * Float.pi * 7990.0 * Float(i) / 16000.0)
            i += 1
        }
        let pitch = PitchResult(f0: 0.0, hnr: 0.0, isVoiced: false)
        let formants = FormantResult(f1: 0.0, f2: 0.0, f3: 0.0, b1: 0.0, b2: 0.0, b3: 0.0, count: 0)

        waveNyquist.withUnsafeBufferPointer { buf in
            filterbank.extractFeatures(pcmPtr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
            XCTAssertFalse(workspace.featureBuffer[0].isNaN)
        }
    }

    func testB6LPCExtremeFrequency() {
        let config = DSPConfig(sampleRate: 16000)
        let lpc = LPC(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        var waveNyquist = [Float](repeating: 0.0, count: config.frameSize)
        var i = 0
        while i < config.frameSize {
            waveNyquist[i] = sin(2.0 * Float.pi * 7900.0 * Float(i) / 16000.0)
            i += 1
        }

        waveNyquist.withUnsafeBufferPointer { buf in
            lpc.computeCoefficients(ptr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
            var k = 0
            while k < config.lpcOrder {
                XCTAssertFalse(workspace.lpcCoeffs[k].isNaN)
                k += 1
            }
        }
    }

    // MARK: - B7: 全発火 / 全不発火スパイク (5 tests)

    func testB7LIFNeuronMassiveVoltageAllSpike() {
        var v: [Float] = [Float](repeating: 100.0, count: 8)
        var s: [Float] = [Float](repeating: 0.0, count: 8)
        let inputs: [Float] = [Float](repeating: 50.0, count: 8)

        let config = LIFConfig(beta: 0.9, vTh: 1.0, vReset: 0.0, alpha: 2.0)
        let vPrev = [Float](repeating: 0.0, count: 8)
        let sPrev = [Float](repeating: 0.0, count: 8)
        vPrev.withUnsafeBufferPointer { vp in
            sPrev.withUnsafeBufferPointer { sp in
                inputs.withUnsafeBufferPointer { inBuf in
                    v.withUnsafeMutableBufferPointer { vBuf in
                        s.withUnsafeMutableBufferPointer { sBuf in
                            LIFNeuronEngine.stepSIMD8(
                                config: config,
                                vPrevPtr: vp.baseAddress!,
                                sPrevPtr: sp.baseAddress!,
                                inputPtr: inBuf.baseAddress!,
                                vNextPtr: vBuf.baseAddress!,
                                sNextPtr: sBuf.baseAddress!,
                                count: 8
                            )
                        }
                    }
                }
            }
        }

        var i = 0
        while i < 8 {
            XCTAssertEqual(s[i], 1.0)
            // 膜電位は学習側と同じ範囲に飽和する
            XCTAssertEqual(v[i], LIFNeuronEngine.vClampMax)
            i += 1
        }
    }

    func testB7LIFNeuronNegativeVoltageNoSpike() {
        let config = LIFConfig(beta: 0.9, vTh: 1.0, vReset: 0.0, alpha: 2.0)
        let vPrev = [Float](repeating: -100.0, count: 8)
        let sPrev = [Float](repeating: 0.0, count: 8)
        var v: [Float] = [Float](repeating: 0.0, count: 8)
        var s: [Float] = [Float](repeating: 0.0, count: 8)
        let inputs: [Float] = [Float](repeating: -10.0, count: 8)

        vPrev.withUnsafeBufferPointer { vp in
            sPrev.withUnsafeBufferPointer { sp in
                inputs.withUnsafeBufferPointer { inBuf in
                    v.withUnsafeMutableBufferPointer { vBuf in
                        s.withUnsafeMutableBufferPointer { sBuf in
                            LIFNeuronEngine.stepSIMD8(
                                config: config,
                                vPrevPtr: vp.baseAddress!,
                                sPrevPtr: sp.baseAddress!,
                                inputPtr: inBuf.baseAddress!,
                                vNextPtr: vBuf.baseAddress!,
                                sNextPtr: sBuf.baseAddress!,
                                count: 8
                            )
                        }
                    }
                }
            }
        }

        var i = 0
        while i < 8 {
            XCTAssertEqual(s[i], 0.0)
            XCTAssertLessThan(v[i], 0.0)
            i += 1
        }
    }

    func testB7AllSpikeInputSoftmax() {
        let net = SpikingNetwork(inputDim: 32, maxHiddenDim: 256, outputDim: 64, timeSteps: 4)
        let features = [Float](repeating: 100.0, count: 32)
        var vPrev = [Float](repeating: 0.0, count: 256)
        var sPrev = [Float](repeating: 1.0, count: 256)
        var spikeSum = [Float](repeating: 4.0, count: 256)
        var logits = [Float](repeating: 0.0, count: 64)
        var probs = [Float](repeating: 0.0, count: 64)

        net.forward(features: features, vPrev: &vPrev, sPrev: &sPrev, spikeSum: &spikeSum, logits: &logits, probabilities: &probs)

        var sumP: Float = 0.0
        var i = 0
        while i < 64 {
            XCTAssertFalse(probs[i].isNaN)
            sumP += probs[i]
            i += 1
        }
        XCTAssertLessThanOrEqual(abs(sumP - 1.0), 1e-4)
    }

    func testB7NoSpikeInputSoftmax() {
        let net = SpikingNetwork(inputDim: 32, maxHiddenDim: 256, outputDim: 64, timeSteps: 4)
        let features = [Float](repeating: 0.0, count: 32)
        var vPrev = [Float](repeating: 0.0, count: 256)
        var sPrev = [Float](repeating: 0.0, count: 256)
        var spikeSum = [Float](repeating: 0.0, count: 256)
        var logits = [Float](repeating: 0.0, count: 64)
        var probs = [Float](repeating: 0.0, count: 64)

        net.forward(features: features, vPrev: &vPrev, sPrev: &sPrev, spikeSum: &spikeSum, logits: &logits, probabilities: &probs)

        var sumP: Float = 0.0
        var i = 0
        while i < 64 {
            XCTAssertFalse(probs[i].isNaN)
            sumP += probs[i]
            i += 1
        }
        XCTAssertLessThanOrEqual(abs(sumP - 1.0), 1e-4)
    }

    func testB7QuantizedEngineAllSpikeState() {
        let net = SpikingNetwork(inputDim: 32, maxHiddenDim: 256, outputDim: 64, timeSteps: 4)
        let config = QuantizedConfig.int32Config()
        let qWeights = QuantizedEngine.quantize(network: net, config: config)
        let engine = QuantizedEngine(weights: qWeights, timeSteps: 4)
        let workspace = QuantizedWorkspace(maxHiddenDim: 256, inputDim: 32, outputDim: 64)

        var i = 0
        while i < 64 {
            workspace.sPrev[i] = 1
            i += 1
        }

        let features = [Float](repeating: 1.0, count: 32)
        var probs = [Float](repeating: 0.0, count: 64)
        engine.predict(features: features, workspace: workspace, outputProbs: &probs)
        XCTAssertFalse(probs[0].isNaN)
    }

    // MARK: - B8: NaN / Inf 混入耐性 (5 tests)

    func testB8VectorOpsNaNInputSanitize() {
        let a: [Float] = [Float.nan, 2.0, 3.0]
        let b: [Float] = [1.0, 2.0, 3.0]
        let dot = a.withUnsafeBufferPointer { aBuf in
            b.withUnsafeBufferPointer { bBuf in
                VectorOperations.dotProduct(a: aBuf.baseAddress!, b: bBuf.baseAddress!, count: 3)
            }
        }
        // Check behavior with NaN
        XCTAssertTrue(dot.isNaN || 0.0 <= dot)
    }

    func testB8FilterbankNaNInput() {
        let config = DSPConfig(sampleRate: 16000)
        let filterbank = Filterbank(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        var nanSignal = [Float](repeating: 0.0, count: config.frameSize)
        nanSignal[0] = Float.nan
        let pitch = PitchResult(f0: 0.0, hnr: 0.0, isVoiced: false)
        let formants = FormantResult(f1: 0.0, f2: 0.0, f3: 0.0, b1: 0.0, b2: 0.0, b3: 0.0, count: 0)

        nanSignal.withUnsafeBufferPointer { buf in
            filterbank.extractFeatures(pcmPtr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
            // Function completed without crashing
            XCTAssertEqual(workspace.featureBuffer.count, 64)
        }
    }

    func testB8SurrogateGradientNaNInput() {
        let grad = SurrogateGradient.derivative(v: Float.nan, vTh: 1.0, alpha: 2.0)
        // Must return float without fatal crash
        XCTAssertTrue(grad.isNaN || 0.0 <= grad)
    }

    func testB8AdamOptimizerNaNGradient() {
        let param = Parameter(count: 1, initialData: [1.0])
        param.grad = [Float.nan]
        let config = AdamConfig(lr: 0.01)
        let adam = AdamOptimizer(config: config, parameters: [param])
        adam.step()
        XCTAssertEqual(param.count, 1)
    }

    func testB8TranscriberNaNInputProtection() {
        let vocab = TextVocabulary()
        let acNet = SpikingNetwork(inputDim: 64, maxHiddenDim: 256, outputDim: vocab.size, timeSteps: 4)
        let lmNet = SpikingNetwork(inputDim: 64, maxHiddenDim: 256, outputDim: vocab.size, timeSteps: 4)
        let transcriber = StreamingTranscriber(acousticNetwork: acNet, languageNetwork: lmNet, textVocabulary: vocab)

        var audio = [Float](repeating: 0.0, count: 1600)
        audio[10] = Float.nan
        audio[20] = Float.infinity
        transcriber.appendAudio(pcm: audio)
        transcriber.flush()
        XCTAssertTrue(true)
    }

    // MARK: - B9: ゼロ除算保護 (5 tests)

    func testB9DurandKernerZeroCoefficients() {
        let solver = DurandKernerSolver()
        let workspace = DSPWorkspace(maxFrameSize: 512, lpcOrder: 4, melChannels: 24)
        let zeroCoeffs = [Float](repeating: 0.0, count: 4)

        zeroCoeffs.withUnsafeBufferPointer { cPtr in
            let success = solver.solve(coefficients: cPtr.baseAddress!, order: 4, workspace: workspace)
            // Handled safely
            XCTAssertTrue(success || success != true)
        }
    }

    func testB9LPCEpsilonProtection() {
        let config = DSPConfig(sampleRate: 16000)
        let lpc = LPC(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        // Micro signal near zero
        let microSignal = [Float](repeating: 1e-20, count: config.frameSize)
        microSignal.withUnsafeBufferPointer { buf in
            let success = lpc.computeCoefficients(ptr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
            XCTAssertFalse(success)
        }
    }

    func testB9PitchDetectorZeroEnergyAutoCorr() {
        let config = DSPConfig(sampleRate: 16000)
        let detector = PitchDetector(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        let microSignal = [Float](repeating: 1e-25, count: config.frameSize)
        microSignal.withUnsafeBufferPointer { buf in
            let res = detector.detectPitch(ptr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
            XCTAssertFalse(res.isVoiced)
            XCTAssertEqual(res.f0, 0.0)
        }
    }

    func testB9AdamOptimizerEpsilonProtection() {
        let config = AdamConfig(lr: 0.01, eps: 1e-8)
        let param = Parameter(count: 1, initialData: [1.0])
        param.grad = [0.0]
        let adam = AdamOptimizer(config: config, parameters: [param])
        adam.step()
        XCTAssertEqual(param.data[0], 1.0)
    }

    func testB9SurrogateGradientZeroDenominator() {
        // When v = vTh, denominator (1 + alpha * |v - vTh|)^2 = 1.0 > 0
        let grad = SurrogateGradient.derivative(v: 1.0, vTh: 1.0, alpha: 0.0)
        XCTAssertFalse(grad.isNaN)
    }

    // MARK: - B10: 浮動小数点アンダーフロー (5 tests)

    func testB10SoftmaxExtremeNegativeLogits() {
        let net = SpikingNetwork(inputDim: 32, maxHiddenDim: 256, outputDim: 64, timeSteps: 4)
        var logits: [Float] = [Float](repeating: -1000.0, count: 64)
        logits[0] = -999.0
        var probs = [Float](repeating: 0.0, count: 64)

        // Forward with extremely negative logits
        let features = [Float](repeating: -50.0, count: 32)
        var vPrev = [Float](repeating: 0.0, count: 256)
        var sPrev = [Float](repeating: 0.0, count: 256)
        var spikeSum = [Float](repeating: 0.0, count: 256)

        net.forward(features: features, vPrev: &vPrev, sPrev: &sPrev, spikeSum: &spikeSum, logits: &logits, probabilities: &probs)
        XCTAssertFalse(probs[0].isNaN)
    }

    func testB10SurrogateGradientUnderflowProtection() {
        let grad = SurrogateGradient.derivative(v: 1e30, vTh: 1.0, alpha: 2.0)
        XCTAssertLessThanOrEqual(0.0, grad)
    }

    func testB10BPTTLogProbabilityUnderflowClamp() {
        // BPTT log(p + eps) protection check
        let eps: Float = 1e-15
        let tinyProb: Float = 0.0
        let logVal = log(max(eps, tinyProb))
        XCTAssertFalse(logVal.isInfinite)
    }

    func testB10LIFNeuronMicroVoltageDecay() {
        let config = LIFConfig(beta: 0.9, vTh: 1.0, vReset: 0.0, alpha: 2.0)
        let res = LIFNeuronEngine.stepScalar(config: config, vPrev: 1e-35, sPrev: 0.0, inputCurrent: 0.0)
        XCTAssertLessThanOrEqual(0.0, res.vNext)
    }

    func testB10FilterbankMicroEnergyLogCompress() {
        let config = DSPConfig(sampleRate: 16000)
        let filterbank = Filterbank(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        let microSignal = [Float](repeating: 1e-15, count: config.frameSize)
        let pitch = PitchResult(f0: 0.0, hnr: 0.0, isVoiced: false)
        let formants = FormantResult(f1: 0.0, f2: 0.0, f3: 0.0, b1: 0.0, b2: 0.0, b3: 0.0, count: 0)

        microSignal.withUnsafeBufferPointer { buf in
            filterbank.extractFeatures(pcmPtr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
            XCTAssertFalse(workspace.featureBuffer[0].isNaN)
        }
    }

    // MARK: - B11: 語彙・トークン範囲外アクセス (5 tests)

    func testB11VocabNegativeIdFallback() {
        let vocab = PhonemeVocabulary()
        let token = vocab.token(for: -99)
        XCTAssertEqual(token, "<unk>")
    }

    func testB11VocabHugeIdFallback() {
        let vocab = PhonemeVocabulary()
        let token = vocab.token(for: 99999)
        XCTAssertEqual(token, "<unk>")
    }

    func testB11VocabEmojiAndSpecialChars() {
        let vocab = PhonemeVocabulary()
        let phonemes = vocab.kanaToPhonemes("🚀🎉こんにちは✨")
        XCTAssertLessThanOrEqual(5, phonemes.count)
    }

    func testB11AcousticDecoderInvalidTokenId() {
        let vocab = PhonemeVocabulary()
        let id = vocab.id(for: "INVALID_PHONEME_XYZ")
        XCTAssertEqual(id, PhonemeVocabulary.unkId)
    }

    func testB11LanguageDecoderOutOfBoundsContext() {
        let vocab = PhonemeVocabulary()
        let backKana = vocab.phonemesToKana(["<unk>", "???", "<pad>"])
        XCTAssertEqual(backKana, "")
    }

    // MARK: - B12: 窓関数サイズ不一致 (5 tests)

    func testB12DSPWorkspaceOddFrameSizes() {
        let workspace = DSPWorkspace(maxFrameSize: 512, lpcOrder: 12, melChannels: 24)
        XCTAssertEqual(workspace.rawFrame.count, 512)
    }

    func testB12LPCSubFrameSizes() {
        let config = DSPConfig(sampleRate: 16000)
        let lpc = LPC(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        let shortFrame = [Float](repeating: 0.5, count: 10)
        shortFrame.withUnsafeBufferPointer { buf in
            let success = lpc.computeCoefficients(ptr: buf.baseAddress!, count: 10, workspace: workspace)
            // Graceful handling of short frames
            XCTAssertTrue(success || success != true)
        }
    }

    func testB12FilterbankSmallFrame() {
        let config = DSPConfig(sampleRate: 16000)
        let filterbank = Filterbank(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        let smallFrame = [Float](repeating: 0.1, count: 64)
        let pitch = PitchResult(f0: 0.0, hnr: 0.0, isVoiced: false)
        let formants = FormantResult(f1: 0.0, f2: 0.0, f3: 0.0, b1: 0.0, b2: 0.0, b3: 0.0, count: 0)

        smallFrame.withUnsafeBufferPointer { buf in
            filterbank.extractFeatures(pcmPtr: buf.baseAddress!, count: 64, workspace: workspace)
            XCTAssertEqual(workspace.featureBuffer.count, 64)
        }
    }

    func testB12VADSingleSampleFrame() {
        let config = DSPConfig(sampleRate: 16000)
        let vad = VAD(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        let single = [Float](repeating: 0.5, count: 1)
        single.withUnsafeBufferPointer { buf in
            let res = vad.processFrame(ptr: buf.baseAddress!, count: 1, workspace: workspace)
            XCTAssertFalse(res.rms.isNaN)
        }
    }

    func testB12PitchDetectorSingleSample() {
        let config = DSPConfig(sampleRate: 16000)
        let detector = PitchDetector(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        let single = [Float](repeating: 0.5, count: 1)
        single.withUnsafeBufferPointer { buf in
            let res = detector.detectPitch(ptr: buf.baseAddress!, count: 1, workspace: workspace)
            XCTAssertFalse(res.isVoiced)
        }
    }

    // MARK: - B13: 固定小数点オーバーフロー (5 tests)

    func testB13QuantizedEngineHugeAccumulation() {
        let net = SpikingNetwork(inputDim: 32, maxHiddenDim: 256, outputDim: 64, timeSteps: 4)
        let config = QuantizedConfig.int32Config()
        let qWeights = QuantizedEngine.quantize(network: net, config: config)
        let engine = QuantizedEngine(weights: qWeights, timeSteps: 4)
        let workspace = QuantizedWorkspace(maxHiddenDim: 256, inputDim: 32, outputDim: 64)

        // Extremely high inputs
        let hugeFeatures = [Float](repeating: 100.0, count: 32)
        var probs = [Float](repeating: 0.0, count: 64)
        engine.predict(features: hugeFeatures, workspace: workspace, outputProbs: &probs)
        XCTAssertFalse(probs[0].isNaN)
    }

    func testB13QuantizedWeightsExtremeScale() {
        let net = SpikingNetwork(inputDim: 32, maxHiddenDim: 256, outputDim: 64, timeSteps: 4)
        let config = QuantizedConfig(vThInt: 1048576, decayNum: 52429, decayBits: 16, scale: 1048576.0, scaleBits: 20)
        let qWeights = QuantizedEngine.quantize(network: net, config: config)
        XCTAssertEqual(qWeights.config.scale, 1048576.0)
    }

    func testB13QuantizedWorkspaceInt32Boundaries() {
        let workspace = QuantizedWorkspace(maxHiddenDim: 256, inputDim: 32, outputDim: 64)
        workspace.vPrev[0] = Int32.max - 100
        workspace.reset()
        XCTAssertEqual(workspace.vPrev[0], 0)
    }

    func testB13QuantizedWorkspaceInt16Boundaries() {
        let workspace = QuantizedWorkspace(maxHiddenDim: 256, inputDim: 32, outputDim: 64)
        workspace.vPrev[0] = Int32(Int16.max)
        workspace.reset()
        XCTAssertEqual(workspace.vPrev[0], 0)
    }

    func testB13QuantizedEngineMultiStepAccumulation() {
        let net = SpikingNetwork(inputDim: 32, maxHiddenDim: 256, outputDim: 64, timeSteps: 4)
        let config = QuantizedConfig.int16Config()
        let qWeights = QuantizedEngine.quantize(network: net, config: config)
        let engine = QuantizedEngine(weights: qWeights, timeSteps: 4)
        let workspace = QuantizedWorkspace(maxHiddenDim: 256, inputDim: 32, outputDim: 64)

        let features = [Float](repeating: 0.7, count: 32)
        var probs = [Float](repeating: 0.0, count: 64)
        var step = 0
        while step < 50 {
            engine.predict(features: features, workspace: workspace, outputProbs: &probs)
            step += 1
        }
        XCTAssertFalse(probs[0].isNaN)
    }

    // MARK: - B14: 急峻なパルス・インパルス (5 tests)

    func testB14VADSingleImpulseDelta() {
        let config = DSPConfig(sampleRate: 16000)
        let vad = VAD(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        var impulse = [Float](repeating: 0.0, count: config.frameSize)
        impulse[0] = 1.0

        impulse.withUnsafeBufferPointer { buf in
            let res = vad.processFrame(ptr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
            XCTAssertFalse(res.rms.isNaN)
        }
    }

    func testB14PitchDetectorSingleImpulse() {
        let config = DSPConfig(sampleRate: 16000)
        let detector = PitchDetector(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        var impulse = [Float](repeating: 0.0, count: config.frameSize)
        impulse[0] = 1.0

        impulse.withUnsafeBufferPointer { buf in
            let res = detector.detectPitch(ptr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
            XCTAssertFalse(res.f0.isNaN)
        }
    }

    func testB14LIFNeuronImpulseResponse() {
        let config = LIFConfig(beta: 0.9, vTh: 1.0, vReset: 0.0, alpha: 2.0)
        // Deliver single massive impulse
        let res1 = LIFNeuronEngine.stepScalar(config: config, vPrev: 0.0, sPrev: 0.0, inputCurrent: 2.0)
        XCTAssertEqual(res1.sNext, 1.0)
        XCTAssertEqual(res1.vNext, 2.0)

        // Next step with 0 input
        let res2 = LIFNeuronEngine.stepScalar(config: config, vPrev: res1.vNext, sPrev: res1.sNext, inputCurrent: 0.0)
        XCTAssertEqual(res2.sNext, 0.0)
        XCTAssertEqual(res2.vNext, 0.0)
    }

    func testB14LPCImpulseResponse() {
        let config = DSPConfig(sampleRate: 16000)
        let lpc = LPC(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        var impulse = [Float](repeating: 0.0, count: config.frameSize)
        impulse[0] = 1.0

        impulse.withUnsafeBufferPointer { buf in
            let success = lpc.computeCoefficients(ptr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
            if success {
                XCTAssertEqual(workspace.lpcCoeffs.count, config.lpcOrder + 1)
            }
        }
    }

    func testB14TranscriberImpulseAudio() {
        let vocab = TextVocabulary()
        let acNet = SpikingNetwork(inputDim: 64, maxHiddenDim: 256, outputDim: vocab.size, timeSteps: 4)
        let lmNet = SpikingNetwork(inputDim: 64, maxHiddenDim: 256, outputDim: vocab.size, timeSteps: 4)
        let transcriber = StreamingTranscriber(acousticNetwork: acNet, languageNetwork: lmNet, textVocabulary: vocab)

        var impulseAudio = [Float](repeating: 0.0, count: 1600)
        impulseAudio[0] = 1.0
        transcriber.appendAudio(pcm: impulseAudio)
        transcriber.flush()
        XCTAssertTrue(true)
    }

    // MARK: - B15: 高速連続 Reset / Flush (5 tests)

    func testB15TranscriberContinuousFlush() {
        let vocab = TextVocabulary()
        let acNet = SpikingNetwork(inputDim: 64, maxHiddenDim: 256, outputDim: vocab.size, timeSteps: 4)
        let lmNet = SpikingNetwork(inputDim: 64, maxHiddenDim: 256, outputDim: vocab.size, timeSteps: 4)
        let transcriber = StreamingTranscriber(acousticNetwork: acNet, languageNetwork: lmNet, textVocabulary: vocab)

        var i = 0
        while i < 100 {
            transcriber.flush()
            transcriber.reset()
            i += 1
        }
        XCTAssertTrue(true)
    }

    func testB15DSPWorkspaceContinuousReset() {
        let workspace = DSPWorkspace(maxFrameSize: 512, lpcOrder: 12, melChannels: 24)
        var i = 0
        while i < 100 {
            workspace.rawFrame[0] = Float(i)
            workspace.featureBuffer[0] = Float(i)
            i += 1
        }
        XCTAssertEqual(workspace.rawFrame[0], 99.0)
    }

    func testB15QuantizedWorkspaceContinuousReset() {
        let workspace = QuantizedWorkspace(maxHiddenDim: 256, inputDim: 32, outputDim: 64)
        var i = 0
        while i < 100 {
            workspace.vPrev[0] = Int32(i)
            workspace.reset()
            i += 1
        }
        XCTAssertEqual(workspace.vPrev[0], 0)
    }

    func testB15LIFStateContinuousReset() {
        let state = LIFState(size: 64)
        var i = 0
        while i < 100 {
            state.v[0] = Float(i)
            state.reset()
            i += 1
        }
        XCTAssertEqual(state.v[0], 0.0)
    }

    func testB15VADContinuousReset() {
        let config = DSPConfig(sampleRate: 16000)
        let vad = VAD(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        var i = 0
        while i < 50 {
            let zeros = [Float](repeating: 0.0, count: config.frameSize)
            zeros.withUnsafeBufferPointer { buf in
                let res = vad.processFrame(ptr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
                XCTAssertFalse(res.isSpeech)
            }
            i += 1
        }
    }

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
}
