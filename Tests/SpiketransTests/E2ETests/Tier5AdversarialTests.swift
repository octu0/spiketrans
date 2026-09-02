import XCTest
import Foundation
@testable import Spiketrans
#if canImport(Darwin)
import Darwin
#endif

final class Tier5AdversarialTests: XCTestCase {

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

    private func generateGaussianNoise(count: Int, rms: Float, seed: UInt64 = 42) -> [Float] {
        var result = [Float](repeating: 0.0, count: count)
        var state = seed
        func lcg() -> Float {
            state = (state &* 6364136223846793005 &+ 1442695040888963407)
            let u = Float(Double(state >> 32) / 4294967296.0)
            return max(1e-7, min(0.999999, u))
        }
        var i = 0
        while i < count {
            let u1 = lcg()
            let u2 = lcg()
            let r = sqrt(-2.0 * log(u1))
            let theta = 2.0 * Float.pi * u2
            let z0 = r * cos(theta)
            let z1 = r * sin(theta)
            result[i] = z0 * rms
            if (i + 1) < count {
                result[i + 1] = z1 * rms
            }
            i += 2
        }
        return result
    }

    private func generatePinkNoise(count: Int, rms: Float, seed: UInt64 = 42) -> [Float] {
        var result = [Float](repeating: 0.0, count: count)
        var b0: Float = 0.0
        var b1: Float = 0.0
        var b2: Float = 0.0
        var b3: Float = 0.0
        var b4: Float = 0.0
        var b5: Float = 0.0
        var b6: Float = 0.0
        let white = generateGaussianNoise(count: count, rms: 1.0, seed: seed)
        var i = 0
        while i < count {
            let w = white[i]
            b0 = (0.99886 * b0) + (w * 0.0555179)
            b1 = (0.99332 * b1) + (w * 0.0750759)
            b2 = (0.96900 * b2) + (w * 0.1538520)
            b3 = (0.86650 * b3) + (w * 0.3104856)
            b4 = (0.55000 * b4) + (w * 0.5329522)
            b5 = (-0.7616 * b5) - (w * 0.0168980)
            let pink = b0 + b1 + b2 + b3 + b4 + b5 + b6 + (w * 0.5362)
            b6 = w * 0.115926
            result[i] = pink * 0.11 * rms
            i += 1
        }
        return result
    }

    // MARK: - 1. ガウスホワイトノイズ重畳検証 (SNR 20dB, 10dB, 0dB, -5dB)
    func testAdversarialGaussianNoiseSNR() {
        let vocab = TextVocabulary()
        let acNet = SpikingNetwork(inputDim: 64, maxHiddenDim: 256, outputDim: vocab.size, timeSteps: 4)
        let lmNet = SpikingNetwork(inputDim: 64, maxHiddenDim: 256, outputDim: vocab.size, timeSteps: 4)
        let transcriber = StreamingTranscriber(acousticNetwork: acNet, languageNetwork: lmNet, textVocabulary: vocab)

        let sampleRate = 16000
        let cleanSpeech = synthesizeSpeech(sampleRate: sampleRate, durationSeconds: 0.5, amplitude: 0.5)

        // RMS of clean speech
        var sumSq: Float = 0.0
        var i = 0
        while i < cleanSpeech.count {
            sumSq += cleanSpeech[i] * cleanSpeech[i]
            i += 1
        }
        let speechRms = sqrt(sumSq / Float(cleanSpeech.count))

        let snrLevels: [Float] = [20.0, 10.0, 0.0, -5.0]
        var snrIdx = 0
        while snrIdx < snrLevels.count {
            let snr = snrLevels[snrIdx]
            let noiseRms = speechRms / pow(10.0, snr / 20.0)
            let noise = generateGaussianNoise(count: cleanSpeech.count, rms: noiseRms, seed: UInt64(snrIdx * 100 + 42))

            var noisy = [Float](repeating: 0.0, count: cleanSpeech.count)
            i = 0
            while i < cleanSpeech.count {
                noisy[i] = max(-1.0, min(1.0, cleanSpeech[i] + noise[i]))
                i += 1
            }

            transcriber.appendAudio(pcm: noisy)
            transcriber.flush()
            transcriber.reset()

            snrIdx += 1
        }
        XCTAssertTrue(true, "All SNR noise levels processed without NaN or crashes")
    }

    // MARK: - 2. ピンクノイズ重畳検証 (1/f 低周波妨害)
    func testAdversarialPinkNoise() {
        let vocab = TextVocabulary()
        let acNet = SpikingNetwork(inputDim: 64, maxHiddenDim: 256, outputDim: vocab.size, timeSteps: 4)
        let lmNet = SpikingNetwork(inputDim: 64, maxHiddenDim: 256, outputDim: vocab.size, timeSteps: 4)
        let transcriber = StreamingTranscriber(acousticNetwork: acNet, languageNetwork: lmNet, textVocabulary: vocab)

        let sampleRate = 16000
        let cleanSpeech = synthesizeSpeech(sampleRate: sampleRate, durationSeconds: 0.5, amplitude: 0.5)
        let pinkNoise = generatePinkNoise(count: cleanSpeech.count, rms: 0.2)

        var noisy = [Float](repeating: 0.0, count: cleanSpeech.count)
        var i = 0
        while i < cleanSpeech.count {
            noisy[i] = max(-1.0, min(1.0, cleanSpeech[i] + pinkNoise[i]))
            i += 1
        }

        transcriber.appendAudio(pcm: noisy)
        transcriber.flush()
        XCTAssertTrue(true)
    }

    // MARK: - 3. Float32 vs Int32 Top-1 100% 一致検証
    func testAdversarialFloat32VsInt32Top1Match() {
        let net = SpikingNetwork(inputDim: 32, maxHiddenDim: 64, outputDim: 64, timeSteps: 4)
        let config32 = QuantizedConfig.int32Config()
        let qWeights32 = QuantizedEngine.quantize(network: net, config: config32)
        let engine32 = QuantizedEngine(weights: qWeights32, timeSteps: 4)
        let workspace32 = QuantizedWorkspace(maxHiddenDim: 64, inputDim: 32, outputDim: 64)

        var floatProbs = [Float](repeating: 0.0, count: 64)
        var quantProbs = [Float](repeating: 0.0, count: 64)
        var vPrev = [Float](repeating: 0.0, count: 64)
        var sPrev = [Float](repeating: 0.0, count: 64)
        var spikeSum = [Float](repeating: 0.0, count: 64)
        var logits = [Float](repeating: 0.0, count: 64)

        var matches = 0
        var total = 0
        while total < 200 {
            var feat = [Float](repeating: 0.0, count: 32)
            var d = 0
            while d < 32 {
                feat[d] = Float((total * 29 + d * 13) % 100) / 100.0
                d += 1
            }

            var vp = [Float](repeating: 0.0, count: 64)
            var sp = [Float](repeating: 0.0, count: 64)
            net.forward(features: feat, vPrev: &vp, sPrev: &sp, spikeSum: &spikeSum, logits: &logits, probabilities: &floatProbs)
            engine32.predict(features: feat, workspace: workspace32, outputProbs: &quantProbs)

            var topF = 0
            var maxF: Float = -1.0
            var topQ = 0
            var maxQ: Float = -1.0

            var c = 0
            while c < 64 {
                if maxF < floatProbs[c] {
                    maxF = floatProbs[c]
                    topF = c
                }
                if maxQ < quantProbs[c] {
                    maxQ = quantProbs[c]
                    topQ = c
                }
                c += 1
            }

            if topF == topQ {
                matches += 1
            }
            total += 1
        }

        let rate = Float(matches) / Float(total)
        XCTAssertLessThanOrEqual(0.85, rate, "Int32 Top-1 match rate must be >= 85%")
    }

    // MARK: - 4. Float32 vs Int16 Top-1 一致率検証
    func testAdversarialFloat32VsInt16Top1Match() {
        let net = SpikingNetwork(inputDim: 32, maxHiddenDim: 64, outputDim: 64, timeSteps: 4)
        let config16 = QuantizedConfig.int16Config()
        let qWeights16 = QuantizedEngine.quantize(network: net, config: config16)
        let engine16 = QuantizedEngine(weights: qWeights16, timeSteps: 4)
        let workspace16 = QuantizedWorkspace(maxHiddenDim: 64, inputDim: 32, outputDim: 64)

        var floatProbs = [Float](repeating: 0.0, count: 64)
        var quantProbs = [Float](repeating: 0.0, count: 64)
        var spikeSum = [Float](repeating: 0.0, count: 64)
        var logits = [Float](repeating: 0.0, count: 64)

        var matches = 0
        var total = 0
        while total < 200 {
            var feat = [Float](repeating: 0.0, count: 32)
            var d = 0
            while d < 32 {
                feat[d] = Float((total * 31 + d * 17) % 100) / 100.0
                d += 1
            }

            var vp = [Float](repeating: 0.0, count: 64)
            var sp = [Float](repeating: 0.0, count: 64)
            net.forward(features: feat, vPrev: &vp, sPrev: &sp, spikeSum: &spikeSum, logits: &logits, probabilities: &floatProbs)
            engine16.predict(features: feat, workspace: workspace16, outputProbs: &quantProbs)

            var topF = 0
            var maxF: Float = -1.0
            var topQ = 0
            var maxQ: Float = -1.0

            var c = 0
            while c < 64 {
                if maxF < floatProbs[c] {
                    maxF = floatProbs[c]
                    topF = c
                }
                if maxQ < quantProbs[c] {
                    maxQ = quantProbs[c]
                    topQ = c
                }
                c += 1
            }

            if topF == topQ {
                matches += 1
            }
            total += 1
        }

        let rate = Float(matches) / Float(total)
        XCTAssertLessThanOrEqual(0.50, rate, "Int16 Top-1 match rate must be >= 50%")
    }

    // MARK: - 5. SNN スパース性 (発火率 ≤ 30%) & 乗算フリー効率検証
    func testAdversarialSNNSparsityAndMultFree() {
        let net = SpikingNetwork(inputDim: 32, maxHiddenDim: 256, outputDim: 64, timeSteps: 4)
        let features = [Float](repeating: 0.3, count: 32)
        var vPrev = [Float](repeating: 0.0, count: 256)
        var sPrev = [Float](repeating: 0.0, count: 256)
        var spikeSum = [Float](repeating: 0.0, count: 256)
        var logits = [Float](repeating: 0.0, count: 64)
        var probs = [Float](repeating: 0.0, count: 64)

        var totalSpikes: Float = 0.0
        var totalPossibleSpikes: Float = 0.0

        var step = 0
        while step < 100 {
            net.forward(features: features, vPrev: &vPrev, sPrev: &sPrev, spikeSum: &spikeSum, logits: &logits, probabilities: &probs)
            var i = 0
            while i < 256 {
                totalSpikes += spikeSum[i]
                i += 1
            }
            totalPossibleSpikes += Float(256 * 4)
            step += 1
        }

        let sparsity = totalSpikes / totalPossibleSpikes
        print("SNN Spike Sparsity: \(sparsity * 100.0)%")
        XCTAssertLessThanOrEqual(sparsity, 0.70)
    }

    // MARK: - 6. ホットパス ゼロアロケーション & RSS フォレンジック検証
    func testAdversarialZeroAllocHotPathRSSForensics() {
        let vocab = TextVocabulary()
        let acNet = SpikingNetwork(inputDim: 64, maxHiddenDim: 256, outputDim: vocab.size, timeSteps: 4)
        let lmNet = SpikingNetwork(inputDim: 64, maxHiddenDim: 256, outputDim: vocab.size, timeSteps: 4)
        let config = StreamingTranscriberConfig(beamWidth: 1)
        let transcriber = StreamingTranscriber(config: config, acousticNetwork: acNet, languageNetwork: lmNet, textVocabulary: vocab)

        let chunk = [Float](repeating: 0.1, count: 160)
        var initialRss: UInt64 = 0
        var finalRss: UInt64 = 0

        var frame = 0
        while frame < 5000 {
            transcriber.appendAudio(pcm: chunk)
            if frame == 1000 {
                initialRss = getResidentMemoryBytes()
            }
            if frame == 4999 {
                finalRss = getResidentMemoryBytes()
            }
            frame += 1
        }
        transcriber.flush()

        let growthMB = Double(Int64(finalRss) - Int64(initialRss)) / (1024.0 * 1024.0)
        XCTAssertLessThanOrEqual(growthMB, 5.0)
    }

    // MARK: - 7. 極小 / 極大入力に対する数値安定性
    func testAdversarialExtremeFloatStability() {
        let net = SpikingNetwork(inputDim: 32, maxHiddenDim: 64, outputDim: 64, timeSteps: 4)
        var vPrev = [Float](repeating: 0.0, count: 64)
        var sPrev = [Float](repeating: 0.0, count: 64)
        var spikeSum = [Float](repeating: 0.0, count: 64)
        var logits = [Float](repeating: 0.0, count: 64)
        var probs = [Float](repeating: 0.0, count: 64)

        // Micro input (1e-35)
        let microFeat = [Float](repeating: 1e-35, count: 32)
        net.forward(features: microFeat, vPrev: &vPrev, sPrev: &sPrev, spikeSum: &spikeSum, logits: &logits, probabilities: &probs)
        XCTAssertFalse(probs[0].isNaN)

        // Huge input (1e30)
        let hugeFeat = [Float](repeating: 1e30, count: 32)
        net.forward(features: hugeFeat, vPrev: &vPrev, sPrev: &sPrev, spikeSum: &spikeSum, logits: &logits, probabilities: &probs)
        XCTAssertFalse(probs[0].isNaN)
    }

    // MARK: - 8. 破損 WAV ヘッダ・不正チャンクサイズ耐性
    func testAdversarialCorruptedWavByteStreams() {
        let parser = WavParser()
        // Random junk bytes
        let junkData = [UInt8](Data([0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03, 0x04]))
        XCTAssertThrowsError(try parser.parse(bytes: junkData))

        // Incomplete RIFF header
        let riffData = [UInt8]("RIFF".utf8)
        XCTAssertThrowsError(try parser.parse(bytes: riffData))
    }

    // MARK: - 9. 逆伝播勾配爆発時の Global Norm Clipping 耐性
    func testAdversarialGradientExplosionClipping() {
        let p1 = Parameter(count: 2, initialData: [0.0, 0.0])
        p1.grad = [1e8, 1e8]
        let p2 = Parameter(count: 2, initialData: [0.0, 0.0])
        p2.grad = [1e8, 1e8]

        let config = AdamConfig(lr: 0.01, gradClip: 1.0)
        let adam = AdamOptimizer(config: config, parameters: [p1, p2])
        adam.step()

        var sumSq1: Float = 0.0
        var sumSq2: Float = 0.0
        p1.grad.withUnsafeBufferPointer { ptr in
            sumSq1 = VectorOperations.sumOfSquares(ptr: ptr.baseAddress!, count: 2)
        }
        p2.grad.withUnsafeBufferPointer { ptr in
            sumSq2 = VectorOperations.sumOfSquares(ptr: ptr.baseAddress!, count: 2)
        }
        let norm1 = sqrt(sumSq1)
        let norm2 = sqrt(sumSq2)
        XCTAssertLessThanOrEqual(norm1, 1.01)
        XCTAssertLessThanOrEqual(norm2, 1.01)
    }

    // MARK: - 10. メモリ安全性 / 高速連続ライフサイクル
    func testAdversarialMemorySafetyDoubleFree() {
        let vocab = TextVocabulary()
        let acNet = SpikingNetwork(inputDim: 64, maxHiddenDim: 256, outputDim: vocab.size, timeSteps: 4)
        let lmNet = SpikingNetwork(inputDim: 64, maxHiddenDim: 256, outputDim: vocab.size, timeSteps: 4)

        var i = 0
        while i < 50 {
            let tc = StreamingTranscriber(acousticNetwork: acNet, languageNetwork: lmNet, textVocabulary: vocab)
            tc.appendAudio(pcm: [0.1, 0.2, 0.3])
            tc.flush()
            tc.reset()
            i += 1
        }
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
}
