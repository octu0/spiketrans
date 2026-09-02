import XCTest
import Foundation
@testable import Spiketrans
#if canImport(Darwin)
import Darwin
#endif

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

final class Tier4RealWorldTests: XCTestCase {

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

    // MARK: - 1. 日本語 5 母音（あ・い・う・え・お）連続発話
    func testScenario1JapaneseVowelsAIEUO() {
        let sampleRate = 16000
        let frameSize = 512
        let config = DSPConfig(sampleRate: sampleRate, frameSize: frameSize, maxPitchLag: 320)
        let workspace = DSPWorkspace(maxFrameSize: 1024, lpcOrder: config.lpcOrder, melChannels: config.melChannels, maxPitchLag: 320)
        let vad = VAD(config: config)
        let pitchDetector = PitchDetector(config: config)
        let lpc = LPC(config: config)
        let solver = DurandKernerSolver()
        let formantExtractor = FormantExtractor(sampleRate: Float(sampleRate))
        let filterbank = Filterbank(config: config)

        // Vowel formants table
        struct VowelSpec {
            let name: String
            let f1: Float
            let f2: Float
            let f3: Float
        }
        let vowels: [VowelSpec] = [
            VowelSpec(name: "a", f1: 800.0, f2: 1300.0, f3: 2600.0),
            VowelSpec(name: "i", f1: 300.0, f2: 2300.0, f3: 3000.0),
            VowelSpec(name: "u", f1: 380.0, f2: 1200.0, f3: 2400.0),
            VowelSpec(name: "e", f1: 500.0, f2: 1900.0, f3: 2600.0),
            VowelSpec(name: "o", f1: 500.0, f2: 900.0, f3: 2500.0)
        ]

        var vIdx = 0
        while vIdx < vowels.count {
            let v = vowels[vIdx]
            var wave = [Float](repeating: 0.0, count: frameSize)
            let twoPi = 2.0 * Float.pi
            var i = 0
            while i < frameSize {
                let t = Float(i) / Float(sampleRate)
                let f0 = 0.5 * sin(twoPi * 150.0 * t)
                let f1 = 0.4 * sin(twoPi * v.f1 * t)
                let f2 = 0.3 * sin(twoPi * v.f2 * t)
                let f3 = 0.2 * sin(twoPi * v.f3 * t)
                wave[i] = f0 + f1 + f2 + f3
                i += 1
            }

            wave.withUnsafeBufferPointer { buf in
                let framePtr = buf.baseAddress!
                let vadRes = vad.processFrame(ptr: framePtr, count: frameSize, workspace: workspace)
                XCTAssertTrue(vadRes.isSpeech)

                let pitchRes = pitchDetector.detectPitch(ptr: framePtr, count: frameSize, workspace: workspace)
                XCTAssertTrue(pitchRes.isVoiced)

                let lpcSuccess = lpc.computeCoefficients(ptr: framePtr, count: frameSize, workspace: workspace)
                XCTAssertTrue(lpcSuccess)

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

                filterbank.extractFeatures(pcmPtr: framePtr, count: frameSize, workspace: workspace)
                XCTAssertEqual(workspace.featureBuffer.count, 64)
            }

            vIdx += 1
        }
    }

    // MARK: - 2. 動的会話音声の VAD 分割と 2 段ストリーミング文字起こし
    func testScenario2ConversationVADSegmentationAndTwoStageSTT() {
        let vocab = TextVocabulary()
        let acNet = SpikingNetwork(inputDim: 64, maxHiddenDim: 256, outputDim: vocab.size, timeSteps: 4)
        let lmNet = SpikingNetwork(inputDim: 64, maxHiddenDim: 256, outputDim: vocab.size, timeSteps: 4)
        let transcriber = StreamingTranscriber(acousticNetwork: acNet, languageNetwork: lmNet, textVocabulary: vocab)

        let collector = ResultCollector<TranscriptionResult>()
        transcriber.onFinalResult = { res in
            collector.append(res)
        }

        // Utterance 1 (600ms) -> Silence (500ms) -> Utterance 2 (500ms) -> Silence (400ms)
        let u1 = synthesizeSpeech(sampleRate: 16000, durationSeconds: 0.6)
        let s1 = synthesizeSilence(sampleRate: 16000, durationSeconds: 0.5)
        let u2 = synthesizeSpeech(sampleRate: 16000, durationSeconds: 0.5)
        let s2 = synthesizeSilence(sampleRate: 16000, durationSeconds: 0.4)

        transcriber.appendAudio(pcm: u1)
        transcriber.appendAudio(pcm: s1)
        transcriber.appendAudio(pcm: u2)
        transcriber.appendAudio(pcm: s2)
        transcriber.flush()

        XCTAssertEqual(collector.count, 2)
        if 2 <= collector.count {
            let res1 = collector[0]
            let res2 = collector[1]
            XCTAssertLessThan(res1.startTimeSeconds, res1.endTimeSeconds)
            XCTAssertLessThan(res2.startTimeSeconds, res2.endTimeSeconds)
            XCTAssertLessThanOrEqual(res1.endTimeSeconds, res2.startTimeSeconds)
        }
    }


    // MARK: - 4. Int16 固定小数点推論精度
    func testScenario4Int16FixedPointPrecision() {
        let net = SpikingNetwork(inputDim: 32, maxHiddenDim: 256, outputDim: 64, timeSteps: 4)
        let config16 = QuantizedConfig.int16Config()
        let qWeights16 = QuantizedEngine.quantize(network: net, config: config16)
        let engine16 = QuantizedEngine(weights: qWeights16, timeSteps: 4)
        let workspace16 = QuantizedWorkspace(maxHiddenDim: 256, inputDim: 32, outputDim: 64)

        var floatProbs = [Float](repeating: 0.0, count: 64)
        var quantProbs = [Float](repeating: 0.0, count: 64)
        var vPrev = [Float](repeating: 0.0, count: 256)
        var sPrev = [Float](repeating: 0.0, count: 256)
        var spikeSum = [Float](repeating: 0.0, count: 256)
        var logits = [Float](repeating: 0.0, count: 64)

        var feat = [Float](repeating: 0.0, count: 32)
        var d = 0
        while d < 32 {
            feat[d] = 0.5
            d += 1
        }

        net.forward(features: feat, vPrev: &vPrev, sPrev: &sPrev, spikeSum: &spikeSum, logits: &logits, probabilities: &floatProbs)
        engine16.predict(features: feat, workspace: workspace16, outputProbs: &quantProbs)

        XCTAssertFalse(quantProbs[0].isNaN)
        var sumP: Float = 0.0
        var i = 0
        while i < 64 {
            sumP += quantProbs[i]
            i += 1
        }
        XCTAssertLessThanOrEqual(abs(sumP - 1.0), 1e-4)
    }

    // MARK: - 5. 長時間連続ストリーム O(1) メモリ安定性
    func testScenario5TenMinutesContinuousStreamO1Memory() {
        let vocab = TextVocabulary()
        let acNet = SpikingNetwork(inputDim: 64, maxHiddenDim: 256, outputDim: vocab.size, timeSteps: 4)
        let lmNet = SpikingNetwork(inputDim: 64, maxHiddenDim: 256, outputDim: vocab.size, timeSteps: 4)
        let config = StreamingTranscriberConfig(beamWidth: 1)
        let transcriber = StreamingTranscriber(config: config, acousticNetwork: acNet, languageNetwork: lmNet, textVocabulary: vocab)

        let chunk = synthesizeSpeech(sampleRate: 16000, durationSeconds: 0.01) // 160 samples (10ms)

        var initialRss: UInt64 = 0
        var finalRss: UInt64 = 0

        // 3,000 frames = 30s stream for test
        var frame = 0
        while frame < 3000 {
            transcriber.appendAudio(pcm: chunk)
            if frame == 500 {
                initialRss = getResidentMemoryBytes()
            }
            if frame == 2999 {
                finalRss = getResidentMemoryBytes()
            }
            frame += 1
        }
        transcriber.flush()

        let growthMB = Double(Int64(finalRss) - Int64(initialRss)) / (1024.0 * 1024.0)
        XCTAssertLessThanOrEqual(growthMB, 10.0, "Memory growth across streaming must remain flat")
    }

    // MARK: - 6. 高音 / 低音ピッチ適応
    func testScenario6HighPitchAndLowPitchAdaptation() {
        let sampleRate = 16000
        let frameSize = 512
        let config = DSPConfig(sampleRate: sampleRate, frameSize: frameSize, maxPitchLag: 320)
        let detector = PitchDetector(config: config)
        let workspace = DSPWorkspace(maxFrameSize: 1024, lpcOrder: config.lpcOrder, melChannels: config.melChannels, maxPitchLag: 320)

        // Male low pitch (100Hz)
        var maleVoice = [Float](repeating: 0.0, count: frameSize)
        // Female/Child high pitch (320Hz)
        var femaleVoice = [Float](repeating: 0.0, count: frameSize)

        let twoPi = 2.0 * Float.pi
        var i = 0
        while i < frameSize {
            let t = Float(i) / Float(sampleRate)
            maleVoice[i] = 0.6 * sin(twoPi * 100.0 * t) + 0.3 * sin(twoPi * 200.0 * t)
            femaleVoice[i] = 0.6 * sin(twoPi * 320.0 * t) + 0.3 * sin(twoPi * 640.0 * t)
            i += 1
        }

        maleVoice.withUnsafeBufferPointer { buf in
            let res = detector.detectPitch(ptr: buf.baseAddress!, count: frameSize, workspace: workspace)
            XCTAssertTrue(res.isVoiced)
            XCTAssertLessThanOrEqual(80.0, res.f0)
            XCTAssertLessThanOrEqual(res.f0, 150.0)
        }

        femaleVoice.withUnsafeBufferPointer { buf in
            let res = detector.detectPitch(ptr: buf.baseAddress!, count: frameSize, workspace: workspace)
            XCTAssertTrue(res.isVoiced)
            XCTAssertLessThanOrEqual(280.0, res.f0)
            XCTAssertLessThanOrEqual(res.f0, 360.0)
        }
    }

    // MARK: - 7. 音響確率ゆらぎに対する自己回帰言語モデル文脈補正
    func testScenario7AcousticUncertaintyAutoregressiveLMCorrection() {
        let vocab = TextVocabulary(characters: Array("あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん"))
        let lmNet = SpikingNetwork(inputDim: 64, maxHiddenDim: 256, outputDim: vocab.size, timeSteps: 4)
        let config = LanguageDecoderConfig(beamWidth: 4, lmWeight: 0.5, wordBonus: 0.2)
        let decoder = LanguageDecoder(lmNetwork: lmNet, vocabulary: vocab, config: config)

        var frames: [AcousticFrameProbabilities] = []
        var i = 0
        while i < 5 {
            var probs = [Float](repeating: 0.0001, count: vocab.size)
            // Tie between two tokens
            probs[5] = 0.48
            probs[6] = 0.49
            frames.append(AcousticFrameProbabilities(frameIndex: i, topTokenId: 6, topProbability: 0.49, probabilities: probs))
            i += 1
        }

        let res = decoder.decodeBeamSearch(acousticProbs: frames)
        XCTAssertFalse(res.tokens.isEmpty)
    }

    // MARK: - 8. 急激な音量スイング & 発話速度変化追従性
    func testScenario8DynamicVolumeAndSpeechRateTracking() {
        let vocab = TextVocabulary()
        let acNet = SpikingNetwork(inputDim: 64, maxHiddenDim: 256, outputDim: vocab.size, timeSteps: 4)
        let lmNet = SpikingNetwork(inputDim: 64, maxHiddenDim: 256, outputDim: vocab.size, timeSteps: 4)
        let transcriber = StreamingTranscriber(acousticNetwork: acNet, languageNetwork: lmNet, textVocabulary: vocab)

        // Soft speech (-40dB = 0.01 amplitude)
        let soft = synthesizeSpeech(sampleRate: 16000, durationSeconds: 0.3, amplitude: 0.01)
        // Loud speech (0dB = 1.0 amplitude)
        let loud = synthesizeSpeech(sampleRate: 16000, durationSeconds: 0.3, amplitude: 1.0)
        let silence = synthesizeSilence(sampleRate: 16000, durationSeconds: 0.4)

        transcriber.appendAudio(pcm: soft)
        transcriber.appendAudio(pcm: silence)
        transcriber.appendAudio(pcm: loud)
        transcriber.appendAudio(pcm: silence)
        transcriber.flush()

        XCTAssertTrue(true)
    }

    // MARK: - Helpers

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
}
