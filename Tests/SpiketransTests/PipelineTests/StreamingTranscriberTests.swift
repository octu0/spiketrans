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

final class StreamingTranscriberTests: XCTestCase {

    // 音声信号合成ヘルパー (ピッチ 200Hz + フォルマント 800Hz / 1200Hz / 2500Hz)
    private func synthesizeSpeech(sampleRate: Int, durationSeconds: Float, amplitude: Float = 0.5) -> [Float] {
        let sampleCount = Int(Float(sampleRate) * durationSeconds)
        var pcm = [Float](repeating: 0.0, count: sampleCount)
        let twoPi = 2.0 * Float.pi

        var i = 0
        while i < sampleCount {
            let t = Float(i) / Float(sampleRate)
            // 基本周波数 200Hz + 高調波 + フォルマント模倣
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

    private func createTestNetworks() -> (acoustic: SpikingNetwork, language: SpikingNetwork, vocab: TextVocabulary) {
        let vocab = TextVocabulary()
        let ac = SpikingNetwork(inputDim: 64, maxHiddenDim: 256, outputDim: vocab.size, timeSteps: 4)
        let lm = SpikingNetwork(inputDim: 64, maxHiddenDim: 256, outputDim: vocab.size, timeSteps: 4)
        return (acoustic: ac, language: lm, vocab: vocab)
    }

    // MARK: - 1. 単一発話のストリーミング文字起こしテスト

    func testStreamingSingleUtterance() {
        let (acNet, lmNet, vocab) = createTestNetworks()

        let transcriber = StreamingTranscriber(
            acousticNetwork: acNet,
            languageNetwork: lmNet,
            textVocabulary: vocab
        )

        let collector = ResultCollector<TranscriptionResult>()
        transcriber.onFinalResult = { res in
            collector.append(res)
        }

        let sampleRate = 16000
        // 前後無音 + 400ms 発話 + 後続無音
        var audio: [Float] = []
        audio.append(contentsOf: synthesizeSilence(sampleRate: sampleRate, durationSeconds: 0.1))
        audio.append(contentsOf: synthesizeSpeech(sampleRate: sampleRate, durationSeconds: 0.5))
        audio.append(contentsOf: synthesizeSilence(sampleRate: sampleRate, durationSeconds: 0.4))

        // 160 サンプル (10ms) chunk ずつストリーミング供給
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

        XCTAssertEqual(collector.count, 1, "Single utterance must trigger exactly 1 onFinalResult")
        if 1 <= collector.count {
            let res = collector[0]
            XCTAssertTrue(res.isFinal)
            XCTAssertLessThan(res.startTimeSeconds, res.endTimeSeconds)
        }
    }

    // MARK: - 2. 無音挟み多重発話ストリーミングテスト

    func testStreamingMultipleUtterancesWithSilence() {
        let (acNet, lmNet, vocab) = createTestNetworks()

        let transcriber = StreamingTranscriber(
            acousticNetwork: acNet,
            languageNetwork: lmNet,
            textVocabulary: vocab
        )

        let collector = ResultCollector<TranscriptionResult>()
        transcriber.onFinalResult = { res in
            collector.append(res)
        }

        let sampleRate = 16000
        // 発話1 (400ms) -> 無音 (600ms) -> 発話2 (400ms) -> 無音 (500ms)
        var audio: [Float] = []
        audio.append(contentsOf: synthesizeSpeech(sampleRate: sampleRate, durationSeconds: 0.4))
        audio.append(contentsOf: synthesizeSilence(sampleRate: sampleRate, durationSeconds: 0.6))
        audio.append(contentsOf: synthesizeSpeech(sampleRate: sampleRate, durationSeconds: 0.4))
        audio.append(contentsOf: synthesizeSilence(sampleRate: sampleRate, durationSeconds: 0.5))

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

        XCTAssertEqual(collector.count, 2, "Two distinct utterances must trigger exactly 2 onFinalResult callbacks")
        if 2 <= collector.count {
            let res1 = collector[0]
            let res2 = collector[1]
            XCTAssertLessThan(res1.startTimeSeconds, res1.endTimeSeconds)
            XCTAssertLessThan(res2.startTimeSeconds, res2.endTimeSeconds)
            XCTAssertLessThanOrEqual(res1.endTimeSeconds, res2.startTimeSeconds)
        }
    }

    // MARK: - 3. O(1) メモリ消費量・長時間ストリーミング連続処理テスト

    func testStreamingO1MemoryFootprint() {
        let (acNet, lmNet, vocab) = createTestNetworks()

        let config = StreamingTranscriberConfig(
            beamWidth: 1
        )

        let transcriber = StreamingTranscriber(
            config: config,
            acousticNetwork: acNet,
            languageNetwork: lmNet,
            textVocabulary: vocab
        )

        let sampleRate = 16000
        let chunk = synthesizeSpeech(sampleRate: sampleRate, durationSeconds: 0.01) // 160 samples (10ms)

        func getMemoryUsageMB() -> Float {
            var info = mach_task_basic_info()
            var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / 4)
            let kerr = withUnsafeMutablePointer(to: &info) {
                $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                    task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
                }
            }
            if kerr == KERN_SUCCESS {
                return Float(info.resident_size) / (1024.0 * 1024.0)
            }
            return 0.0
        }

        var memAt2000: Float = 0.0
        var memAt5000: Float = 0.0
        var memAt10000: Float = 0.0

        // 10,000 フレーム = 100秒分のストリーミング処理
        var frame = 0
        while frame < 10000 {
            transcriber.appendAudio(pcm: chunk)

            if frame == 2000 {
                memAt2000 = getMemoryUsageMB()
            }
            if frame == 5000 {
                memAt5000 = getMemoryUsageMB()
            }
            if frame == 9999 {
                memAt10000 = getMemoryUsageMB()
            }
            frame += 1
        }
        transcriber.flush()

        let memGrowth = memAt10000 - memAt2000
        print("M3 Streaming Transcriber Memory: frame2000=\(memAt2000)MB, frame5000=\(memAt5000)MB, frame10000=\(memAt10000)MB, growth=\(memGrowth)MB")
        XCTAssertLessThanOrEqual(memGrowth, 5.0, "Memory growth must remain flat across 10,000 frames")
    }

    // MARK: - 4. 部分認識結果 (Partial Results) 逐次発火テスト

    func testStreamingPartialResults() {
        let (acNet, lmNet, vocab) = createTestNetworks()

        let transcriber = StreamingTranscriber(
            acousticNetwork: acNet,
            languageNetwork: lmNet,
            textVocabulary: vocab
        )

        let counter = AtomicCounter()
        transcriber.onPartialResult = { partial in
            counter.increment()
            XCTAssertFalse(partial.isFinal)
        }

        let sampleRate = 16000
        // 1.0 秒 (100フレーム) の発話音声
        let audio = synthesizeSpeech(sampleRate: sampleRate, durationSeconds: 1.0)
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

        XCTAssertLessThanOrEqual(5, counter.value, "Partial results must be triggered multiple times during active speech")
    }

    // MARK: - 5. 異常系・敵対的入力 (Adversarial Inputs) 堅牢性テスト

    func testStreamingAdversarialInputs() {
        let (acNet, lmNet, vocab) = createTestNetworks()

        let transcriber = StreamingTranscriber(
            acousticNetwork: acNet,
            languageNetwork: lmNet,
            textVocabulary: vocab
        )

        // 空配列
        transcriber.appendAudio(pcm: [])

        // 1サンプルのみ
        transcriber.appendAudio(pcm: [0.5])

        // 超巨大バースト (50,000 サンプル)
        let hugeAudio = [Float](repeating: 0.1, count: 50000)
        transcriber.appendAudio(pcm: hugeAudio)

        // 完全無音
        let silence = [Float](repeating: 0.0, count: 16000)
        transcriber.appendAudio(pcm: silence)

        // DC オフセット (定数信号)
        let dc = [Float](repeating: 0.8, count: 8000)
        transcriber.appendAudio(pcm: dc)

        // ホワイトノイズ
        var noise = [Float](repeating: 0.0, count: 8000)
        var i = 0
        while i < 8000 {
            noise[i] = Float.random(in: -0.5...0.5)
            i += 1
        }
        transcriber.appendAudio(pcm: noise)

        transcriber.flush()
        transcriber.reset()
    }

    // MARK: - 6. 固定小数点量子化モードストリーミングテスト

    func testStreamingQuantizedEngineMode() {
        let (acNet, lmNet, vocab) = createTestNetworks()

        let config = StreamingTranscriberConfig(
            useQuantization: true
        )

        let transcriber = StreamingTranscriber(
            config: config,
            acousticNetwork: acNet,
            languageNetwork: lmNet,
            textVocabulary: vocab
        )

        let collector = ResultCollector<TranscriptionResult>()
        transcriber.onFinalResult = { res in
            collector.append(res)
        }

        let sampleRate = 16000
        var audio: [Float] = []
        audio.append(contentsOf: synthesizeSpeech(sampleRate: sampleRate, durationSeconds: 0.5))
        audio.append(contentsOf: synthesizeSilence(sampleRate: sampleRate, durationSeconds: 0.4))

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

        XCTAssertEqual(collector.count, 1, "Quantized transcriber must process speech stream correctly")
    }
}
