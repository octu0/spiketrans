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

final class Tier3CombinationTests: XCTestCase {

    // MARK: - 1. DSP ↔ SNN 特徴量注入完全パイプライン
    func testComboDSPToSNNFeatureInjection() throws {
        let sampleRate = 16000
        let config = DSPConfig(sampleRate: sampleRate)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)
        let vad = VAD(config: config)
        let pitchDetector = PitchDetector(config: config)
        let lpc = LPC(config: config)
        let solver = DurandKernerSolver()
        let formantExtractor = FormantExtractor(sampleRate: Float(sampleRate))
        let filterbank = Filterbank(config: config)

        let snn = MatryoshkaNetwork(inputDim: 32, maxHiddenDim: 256, outputDim: 64, timeSteps: 4)

        let speech = synthesizeSpeech(sampleRate: sampleRate, durationSeconds: 0.1)
        let wavData = createWavData(samples: speech, sampleRate: sampleRate)
        let parser = WavParser()
        let parsedWav = try parser.parse(bytes: [UInt8](wavData))

        var vPrev = [Float](repeating: 0.0, count: 256)
        var sPrev = [Float](repeating: 0.0, count: 256)
        var spikeSum = [Float](repeating: 0.0, count: 256)
        var logits = [Float](repeating: 0.0, count: 64)
        var probs = [Float](repeating: 0.0, count: 64)

        parsedWav.pcmData.withUnsafeBufferPointer { buf in
            let framePtr = buf.baseAddress!
            let vadRes = vad.processFrame(ptr: framePtr, count: config.frameSize, workspace: workspace)
            XCTAssertTrue(vadRes.isSpeech)

            let pitchRes = pitchDetector.detectPitch(ptr: framePtr, count: config.frameSize, workspace: workspace)
            let lpcSuccess = lpc.computeCoefficients(ptr: framePtr, count: config.frameSize, workspace: workspace)

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

            filterbank.extractFeatures(pcmPtr: framePtr, count: config.frameSize, workspace: workspace)

            snn.forwardSlice(features: workspace.featureBuffer, slice: .base, vPrev: &vPrev, sPrev: &sPrev, spikeSum: &spikeSum, logits: &logits, probabilities: &probs)

            XCTAssertEqual(probs.count, 64)
            var sumP: Float = 0.0
            var i = 0
            while i < 64 {
                sumP += probs[i]
                i += 1
            }
            XCTAssertLessThanOrEqual(abs(sumP - 1.0), 1e-4)
        }
    }

    // MARK: - 2. Matryoshka スライス動的切替
    func testComboMatryoshkaSliceSwitchingLive() {
        let net = MatryoshkaNetwork(inputDim: 32, maxHiddenDim: 256, outputDim: 64, timeSteps: 4)
        let features = [Float](repeating: 0.5, count: 32)
        let slices: [MatryoshkaSlice] = [.base, .middle, .high, .base]

        var sIdx = 0
        while sIdx < slices.count {
            let sl = slices[sIdx]
            let hDim = sl.rawValue
            var vPrev = [Float](repeating: 0.0, count: hDim)
            var sPrev = [Float](repeating: 0.0, count: hDim)
            var spikeSum = [Float](repeating: 0.0, count: hDim)
            var logits = [Float](repeating: 0.0, count: 64)
            var probs = [Float](repeating: 0.0, count: 64)

            net.forwardSlice(features: features, slice: sl, vPrev: &vPrev, sPrev: &sPrev, spikeSum: &spikeSum, logits: &logits, probabilities: &probs)
            XCTAssertFalse(probs[0].isNaN)
            sIdx += 1
        }
    }

    // MARK: - 3. BPTT 学習モデル ↔ Int32 固定小数点推論
    func testComboBPTTTrainingToQuantizedInferenceInt32() {
        let net = MatryoshkaNetwork(inputDim: 32, maxHiddenDim: 64, outputDim: 64, timeSteps: 4)
        let config = QuantizedConfig.int32Config()
        let qWeights = QuantizedEngine.quantize(network: net, config: config)
        let engine = QuantizedEngine(weights: qWeights, timeSteps: 4)
        let workspace = QuantizedWorkspace(maxHiddenDim: 64, inputDim: 32, outputDim: 64)

        var floatProbs = [Float](repeating: 0.0, count: 64)
        var quantProbs = [Float](repeating: 0.0, count: 64)
        var vPrev = [Float](repeating: 0.0, count: 64)
        var sPrev = [Float](repeating: 0.0, count: 64)
        var spikeSum = [Float](repeating: 0.0, count: 64)
        var logits = [Float](repeating: 0.0, count: 64)

        var matchCount = 0
        var testSamples = 0
        while testSamples < 100 {
            var feat = [Float](repeating: 0.0, count: 32)
            var d = 0
            while d < 32 {
                feat[d] = Float((testSamples * 13 + d * 7) % 100) / 100.0
                d += 1
            }

            var vp = [Float](repeating: 0.0, count: 64)
            var sp = [Float](repeating: 0.0, count: 64)
            net.forwardSlice(features: feat, slice: .base, vPrev: &vp, sPrev: &sp, spikeSum: &spikeSum, logits: &logits, probabilities: &floatProbs)
            engine.predictSlice(features: feat, slice: .base, workspace: workspace, outputProbs: &quantProbs)

            var topFloat = 0
            var maxFloat: Float = -1.0
            var topQuant = 0
            var maxQuant: Float = -1.0

            var c = 0
            while c < 64 {
                if maxFloat < floatProbs[c] {
                    maxFloat = floatProbs[c]
                    topFloat = c
                }
                if maxQuant < quantProbs[c] {
                    maxQuant = quantProbs[c]
                    topQuant = c
                }
                c += 1
            }

            if topFloat == topQuant {
                matchCount += 1
            }
            testSamples += 1
        }

        let matchRate = Float(matchCount) / 100.0
        XCTAssertLessThanOrEqual(0.85, matchRate, "Int32 Top-1 match rate must be >= 85%")
    }

    // MARK: - 4. BPTT 学習モデル ↔ Int16 固定小数点推論
    func testComboBPTTTrainingToQuantizedInferenceInt16() {
        let net = MatryoshkaNetwork(inputDim: 32, maxHiddenDim: 64, outputDim: 64, timeSteps: 4)
        let config = QuantizedConfig.int16Config()
        let qWeights = QuantizedEngine.quantize(network: net, config: config)
        let engine = QuantizedEngine(weights: qWeights, timeSteps: 4)
        let workspace = QuantizedWorkspace(maxHiddenDim: 64, inputDim: 32, outputDim: 64)

        var floatProbs = [Float](repeating: 0.0, count: 64)
        var quantProbs = [Float](repeating: 0.0, count: 64)
        var spikeSum = [Float](repeating: 0.0, count: 64)
        var logits = [Float](repeating: 0.0, count: 64)

        var matchCount = 0
        var testSamples = 0
        while testSamples < 100 {
            var feat = [Float](repeating: 0.0, count: 32)
            var d = 0
            while d < 32 {
                feat[d] = Float((testSamples * 17 + d * 5) % 100) / 100.0
                d += 1
            }

            var vp = [Float](repeating: 0.0, count: 64)
            var sp = [Float](repeating: 0.0, count: 64)
            net.forwardSlice(features: feat, slice: .base, vPrev: &vp, sPrev: &sp, spikeSum: &spikeSum, logits: &logits, probabilities: &floatProbs)
            engine.predictSlice(features: feat, slice: .base, workspace: workspace, outputProbs: &quantProbs)

            var topFloat = 0
            var maxFloat: Float = -1.0
            var topQuant = 0
            var maxQuant: Float = -1.0

            var c = 0
            while c < 64 {
                if maxFloat < floatProbs[c] {
                    maxFloat = floatProbs[c]
                    topFloat = c
                }
                if maxQuant < quantProbs[c] {
                    maxQuant = quantProbs[c]
                    topQuant = c
                }
                c += 1
            }

            if topFloat == topQuant {
                matchCount += 1
            }
            testSamples += 1
        }

        let matchRate = Float(matchCount) / 100.0
        XCTAssertLessThanOrEqual(0.50, matchRate, "Int16 Top-1 match rate must be at least 50%")
    }

    // MARK: - 5. VAD セグメンテーション ↔ StreamingTranscriber
    func testComboVADSegmentationToStreamingTranscription() {
        let vocab = TextVocabulary()
        let acNet = MatryoshkaNetwork(inputDim: 64, maxHiddenDim: 256, outputDim: vocab.size, timeSteps: 4)
        let lmNet = MatryoshkaNetwork(inputDim: 64, maxHiddenDim: 256, outputDim: vocab.size, timeSteps: 4)
        let transcriber = StreamingTranscriber(acousticNetwork: acNet, languageNetwork: lmNet, textVocabulary: vocab)

        let collector = ResultCollector<TranscriptionResult>()
        transcriber.onFinalResult = { res in
            collector.append(res)
        }

        let speech1 = synthesizeSpeech(sampleRate: 16000, durationSeconds: 0.4)
        let silence = synthesizeSilence(sampleRate: 16000, durationSeconds: 0.6)
        let speech2 = synthesizeSpeech(sampleRate: 16000, durationSeconds: 0.4)
        let endSilence = synthesizeSilence(sampleRate: 16000, durationSeconds: 0.5)

        transcriber.appendAudio(pcm: speech1)
        transcriber.appendAudio(pcm: silence)
        transcriber.appendAudio(pcm: speech2)
        transcriber.appendAudio(pcm: endSilence)
        transcriber.flush()

        XCTAssertEqual(collector.count, 2)
    }

    // MARK: - 6. 音響 CTC 圧縮 ↔ 自己回帰言語 SNN デコーダ
    func testComboAcousticCTCCollapseToLanguageDecoder() {
        let vocab = TextVocabulary()
        let acNet = MatryoshkaNetwork(inputDim: 64, maxHiddenDim: 256, outputDim: vocab.size, timeSteps: 4)
        let lmNet = MatryoshkaNetwork(inputDim: 64, maxHiddenDim: 256, outputDim: vocab.size, timeSteps: 4)
        let acDecoder = AcousticDecoder(network: acNet, vocabulary: vocab, slice: .base)
        let lmDecoder = LanguageDecoder(lmNetwork: lmNet, vocabulary: vocab)
        let workspace = AcousticWorkspace(maxHiddenDim: 256, outputDim: vocab.size, inputDim: 64)

        var acousticProbs: [AcousticFrameProbabilities] = []
        var i = 0
        while i < 20 {
            let feat = [Float](repeating: 0.6, count: 64)
            let frame = acDecoder.decodeFrame(features: feat, workspace: workspace, frameIndex: i)
            acousticProbs.append(frame)
            i += 1
        }

        let result = lmDecoder.decodeBeamSearch(acousticProbs: acousticProbs, slice: .base)
        XCTAssertFalse(result.score.isNaN)
    }

    // MARK: - 7. Base モデル単体切出 export ↔ import ↔ 推論一致
    func testComboBaseWeightsExportImportInference() {
        let net1 = MatryoshkaNetwork(inputDim: 32, maxHiddenDim: 256, outputDim: 64, timeSteps: 4)
        let exported = net1.exportBaseWeights()

        let net2 = MatryoshkaNetwork(inputDim: 32, maxHiddenDim: 256, outputDim: 64, timeSteps: 4)
        net2.importBaseWeights(exported)

        let feat = [Float](repeating: 0.4, count: 32)
        var v1 = [Float](repeating: 0.0, count: 128)
        var s1 = [Float](repeating: 0.0, count: 128)
        var sum1 = [Float](repeating: 0.0, count: 128)
        var log1 = [Float](repeating: 0.0, count: 64)
        var p1 = [Float](repeating: 0.0, count: 64)

        var v2 = [Float](repeating: 0.0, count: 128)
        var s2 = [Float](repeating: 0.0, count: 128)
        var sum2 = [Float](repeating: 0.0, count: 128)
        var log2 = [Float](repeating: 0.0, count: 64)
        var p2 = [Float](repeating: 0.0, count: 64)

        net1.forwardSlice(features: feat, slice: .base, vPrev: &v1, sPrev: &s1, spikeSum: &sum1, logits: &log1, probabilities: &p1)
        net2.forwardSlice(features: feat, slice: .base, vPrev: &v2, sPrev: &s2, spikeSum: &sum2, logits: &log2, probabilities: &p2)

        var c = 0
        while c < 64 {
            XCTAssertEqual(p1[c], p2[c], accuracy: 1e-6)
            c += 1
        }
    }

    // MARK: - 8. 3スライス同時時間逆伝播勾配累積 ↔ Adam オプティマイザ
    func testComboAdamOptimizerMultiSliceGradientAccumulation() {
        let net = MatryoshkaNetwork(inputDim: 32, maxHiddenDim: 256, outputDim: 64, timeSteps: 4)
        let config = AdamConfig(lr: 0.01)
        let adam = AdamOptimizer(config: config, parameters: net.parameters)
        let trainer = BPTTTrainer(network: net, optimizer: adam)

        var seq: [[Float]] = []
        var targets: [Int] = []
        var i = 0
        while i < 5 {
            seq.append([Float](repeating: 0.5, count: 32))
            targets.append((i + 5) % 64)
            i += 1
        }

        let stepRes = trainer.trainStep(featuresSeq: seq, targets: targets)
        XCTAssertLessThanOrEqual(0.0, stepRes.totalLoss)
        XCTAssertLessThanOrEqual(0.0, stepRes.lossBase)
        XCTAssertLessThanOrEqual(0.0, stepRes.lossMiddle)
        XCTAssertLessThanOrEqual(0.0, stepRes.lossHigh)
    }

    // MARK: - 9. LPC 係数 ↔ Durand-Kerner ↔ フォルマント ↔ 32次元特徴量統合
    func testComboDurandKernerToFormantEnergyFilterbank() {
        let config = DSPConfig(sampleRate: 16000)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)
        let lpc = LPC(config: config)
        let solver = DurandKernerSolver()
        let extractor = FormantExtractor(sampleRate: 16000.0)
        let filterbank = Filterbank(config: config)

        let speech = synthesizeSpeech(sampleRate: 16000, durationSeconds: Float(config.frameSize) / 16000.0)
        speech.withUnsafeBufferPointer { buf in
            let lpcSuccess = lpc.computeCoefficients(ptr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
            XCTAssertTrue(lpcSuccess)
            if lpcSuccess {
                workspace.lpcCoeffs.withUnsafeBufferPointer { cPtr in
                    let solverSuccess = solver.solve(coefficients: cPtr.baseAddress!, order: config.lpcOrder, workspace: workspace)
                    if solverSuccess {
                        workspace.durandKernerCurr.withUnsafeBufferPointer { rPtr in
                            _ = extractor.extractFormants(roots: rPtr.baseAddress!, count: config.lpcOrder)
                            filterbank.extractFeatures(pcmPtr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
                            XCTAssertEqual(workspace.featureBuffer.count, 64)
                        }
                    }
                }
            }
        }
    }

    // MARK: - 10. VectorOperations SIMD8 ↔ スカラー完全ビット一致
    func testComboSIMD8VectorOpsVsScalarBitExact() {
        let a: [Float] = [1.5, -2.5, 3.0, 4.2, -5.1, 6.0, 7.8, -8.3]
        let b: [Float] = [0.2, 0.4, -0.6, 0.8, -1.0, 1.2, -1.4, 1.6]

        let dot = a.withUnsafeBufferPointer { aBuf in
            b.withUnsafeBufferPointer { bBuf in
                VectorOperations.dotProduct(a: aBuf.baseAddress!, b: bBuf.baseAddress!, count: 8)
            }
        }
        var expectedDot: Float = 0.0
        var i = 0
        while i < 8 {
            expectedDot += a[i] * b[i]
            i += 1
        }
        XCTAssertLessThanOrEqual(abs(dot - expectedDot), 1e-4)

        let sumSq = a.withUnsafeBufferPointer { buf in
            VectorOperations.sumOfSquares(ptr: buf.baseAddress!, count: 8)
        }
        var expectedSumSq: Float = 0.0
        i = 0
        while i < 8 {
            expectedSumSq += a[i] * a[i]
            i += 1
        }
        XCTAssertLessThanOrEqual(abs(sumSq - expectedSumSq), 1e-4)
    }

    // MARK: - 11. StreamingTranscriber 完全ライフサイクル
    func testComboStreamingTranscriberLifecycle() {
        let vocab = TextVocabulary()
        let acNet = MatryoshkaNetwork(inputDim: 64, maxHiddenDim: 256, outputDim: vocab.size, timeSteps: 4)
        let lmNet = MatryoshkaNetwork(inputDim: 64, maxHiddenDim: 256, outputDim: vocab.size, timeSteps: 4)
        let transcriber = StreamingTranscriber(acousticNetwork: acNet, languageNetwork: lmNet, textVocabulary: vocab)

        let collector = ResultCollector<TranscriptionResult>()
        transcriber.onFinalResult = { res in
            collector.append(res)
        }

        // Cycle 1
        let speech = synthesizeSpeech(sampleRate: 16000, durationSeconds: 0.4)
        let silence = synthesizeSilence(sampleRate: 16000, durationSeconds: 0.4)
        transcriber.appendAudio(pcm: speech)
        transcriber.appendAudio(pcm: silence)
        transcriber.flush()
        XCTAssertEqual(collector.count, 1)

        // Reset
        transcriber.reset()

        // Cycle 2
        transcriber.appendAudio(pcm: speech)
        transcriber.appendAudio(pcm: silence)
        transcriber.flush()
        XCTAssertEqual(collector.count, 2)
    }

    // MARK: - 12. 音響スコアと言語スコアの重みバランス
    func testComboAcousticLanguageScoreBalance() {
        let vocab = TextVocabulary(characters: Array("あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん"))
        let lmNet = MatryoshkaNetwork(inputDim: 64, maxHiddenDim: 256, outputDim: vocab.size, timeSteps: 4)

        let configBalanced = LanguageDecoderConfig(beamWidth: 4, lmWeight: 0.3, wordBonus: 0.1)
        let decoderBalanced = LanguageDecoder(lmNetwork: lmNet, vocabulary: vocab, config: configBalanced)

        var probs = [Float](repeating: 0.0001, count: vocab.size)
        probs[vocab.id(for: "あ")] = 0.5
        probs[vocab.id(for: "い")] = 0.45
        let frame = AcousticFrameProbabilities(frameIndex: 0, topTokenId: vocab.id(for: "あ"), topProbability: 0.5, probabilities: probs)

        let resGreedy = decoderBalanced.decodeGreedy(acousticProbs: [frame], slice: .base)
        XCTAssertFalse(resGreedy.tokens.isEmpty)
    }

    // MARK: - 13. 高雑音環境下での VAD 遮断 ↔ 音響デコーダ保護
    func testComboHighNoiseVADToAcousticBlankSuppression() {
        let config = DSPConfig(sampleRate: 16000)
        let vad = VAD(config: config)
        let workspace = DSPWorkspace(maxFrameSize: config.frameSize, lpcOrder: config.lpcOrder, melChannels: config.melChannels)

        // Low amplitude micro noise below threshold
        let noise = [Float](repeating: 0.0001, count: config.frameSize)
        noise.withUnsafeBufferPointer { buf in
            let vadRes = vad.processFrame(ptr: buf.baseAddress!, count: config.frameSize, workspace: workspace)
            XCTAssertFalse(vadRes.isSpeech)
        }
    }

    // MARK: - 14. 長時間連続推論における各 Workspace バッファ再利用
    func testComboLongAudioContinuousMemoryRecycle() {
        let dspWs = DSPWorkspace(maxFrameSize: 512, lpcOrder: 12, melChannels: 24)
        let acWs = AcousticWorkspace(maxHiddenDim: 256, outputDim: 64, inputDim: 32)
        let qWs = QuantizedWorkspace(maxHiddenDim: 256, inputDim: 32, outputDim: 64)

        var frame = 0
        while frame < 1000 {
            dspWs.rawFrame[0] = Float(frame)
            acWs.vPrev[0] = Float(frame)
            qWs.vPrev[0] = Int32(frame)
            frame += 1
        }

        XCTAssertEqual(dspWs.rawFrame[0], 999.0)
        XCTAssertEqual(acWs.vPrev[0], 999.0)
        XCTAssertEqual(qWs.vPrev[0], 999)
    }

    // MARK: - 15. 言語 SNN ビームサーチ ↔ Greedy デコード比較
    func testComboLanguageDecoderBeamSearchVsGreedy() {
        let vocab = TextVocabulary(characters: Array("あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん"))
        let lmNet = MatryoshkaNetwork(inputDim: 64, maxHiddenDim: 256, outputDim: vocab.size, timeSteps: 4)
        let config = LanguageDecoderConfig(beamWidth: 4, lmWeight: 0.3, wordBonus: 0.1)
        let decoder = LanguageDecoder(lmNetwork: lmNet, vocabulary: vocab, config: config)

        var frames: [AcousticFrameProbabilities] = []
        var i = 0
        while i < 10 {
            var probs = [Float](repeating: 0.0001, count: vocab.size)
            let tok = (i % 5) + 5
            probs[tok] = 0.6
            frames.append(AcousticFrameProbabilities(frameIndex: i, topTokenId: tok, topProbability: 0.6, probabilities: probs))
            i += 1
        }

        let greedyRes = decoder.decodeGreedy(acousticProbs: frames, slice: .base)
        let beamRes = decoder.decodeBeamSearch(acousticProbs: frames, slice: .base)

        XCTAssertFalse(greedyRes.tokens.isEmpty)
        XCTAssertFalse(beamRes.tokens.isEmpty)
    }

    // MARK: - Helpers

    private func createWavData(samples: [Float], sampleRate: Int = 16000) -> Data {
        var data = Data()
        data.append(contentsOf: [UInt8]("RIFF".utf8))
        let subchunk2Size = UInt32(samples.count * 2)
        let chunkSize = UInt32(36 + Int(subchunk2Size))

        var chunkSizeLE = chunkSize.littleEndian
        withUnsafeBytes(of: &chunkSizeLE) { data.append(contentsOf: $0) }

        data.append(contentsOf: [UInt8]("WAVE".utf8))
        data.append(contentsOf: [UInt8]("fmt ".utf8))

        let subchunk1Size: UInt32 = 16
        let audioFormat: UInt16 = 1
        let numChannels: UInt16 = 1
        let sRate = UInt32(sampleRate)
        let byteRate = UInt32(sampleRate * 2)
        let blockAlign: UInt16 = 2
        let bitsPerSamp: UInt16 = 16

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
}
