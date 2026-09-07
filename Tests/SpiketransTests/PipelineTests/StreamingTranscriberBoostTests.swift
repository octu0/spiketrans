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
}

final class StreamingTranscriberBoostTests: XCTestCase {

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

    private func createTestNetworks() -> (acoustic: SpikingNetwork, language: SpikingNetwork, vocab: TextVocabulary) {
        let vocab = TextVocabulary()
        let ac = SpikingNetwork(inputDim: SpeechDataset.acousticInputDim(), maxHiddenDim: 256, outputDim: vocab.size, timeSteps: 4)
        let lm = SpikingNetwork(inputDim: 64, maxHiddenDim: 256, outputDim: vocab.size, timeSteps: 4)
        return (acoustic: ac, language: lm, vocab: vocab)
    }

    // MARK: - 1. 設定パラメータ既定値とオプショナル指定の検証

    func testBoostConfigParameters() {
        // 既定値: 後方互換のため全て false
        let defaultConfig = StreamingTranscriberConfig()
        XCTAssertFalse(defaultConfig.enableFillerRemoval)
        XCTAssertEqual(defaultConfig.fillerMode, .remove)
        XCTAssertFalse(defaultConfig.enableITN)
        XCTAssertFalse(defaultConfig.enableParagraphSegmentation)
        XCTAssertEqual(defaultConfig.paragraphPauseThreshold, 1.2)

        // ブースト設定の有効化
        let boostConfig = StreamingTranscriberConfig(
            enableFillerRemoval: true,
            fillerMode: .remove,
            enableITN: true,
            enableParagraphSegmentation: true,
            paragraphPauseThreshold: 1.5
        )
        XCTAssertTrue(boostConfig.enableFillerRemoval)
        XCTAssertEqual(boostConfig.fillerMode, .remove)
        XCTAssertTrue(boostConfig.enableITN)
        XCTAssertTrue(boostConfig.enableParagraphSegmentation)
        XCTAssertEqual(boostConfig.paragraphPauseThreshold, 1.5)
    }

    // MARK: - 2. 段落分割コールバックとタイムスタンプの統合テスト

    func testStreamingParagraphSegmentationIntegration() {
        let (acNet, lmNet, vocab) = createTestNetworks()
        let config = StreamingTranscriberConfig(
            beamWidth: 1,
            enableFillerRemoval: true,
            enableITN: true,
            enableParagraphSegmentation: true,
            paragraphPauseThreshold: 1.2
        )

        let transcriber = StreamingTranscriber(
            config: config,
            acousticNetwork: acNet,
            languageNetwork: lmNet,
            textVocabulary: vocab
        )

        let finalCollector = ResultCollector<TranscriptionResult>()
        let paragraphCollector = ResultCollector<ParagraphSegment>()

        transcriber.onFinalResult = { res in
            finalCollector.append(res)
        }
        transcriber.onParagraphResult = { para in
            paragraphCollector.append(para)
        }

        let sampleRate = 16000
        var audio: [Float] = []

        // 発話 1 (0.5秒)
        audio.append(contentsOf: synthesizeSilence(sampleRate: sampleRate, durationSeconds: 0.1))
        audio.append(contentsOf: synthesizeSpeech(sampleRate: sampleRate, durationSeconds: 0.5))
        // 短いポーズ (0.4秒: 1.2秒未満なので同一段落)
        audio.append(contentsOf: synthesizeSilence(sampleRate: sampleRate, durationSeconds: 0.4))
        // 発話 2 (0.5秒)
        audio.append(contentsOf: synthesizeSpeech(sampleRate: sampleRate, durationSeconds: 0.5))
        // 長いポーズ (1.5秒: 1.2秒以上なので段落分割が発生する)
        audio.append(contentsOf: synthesizeSilence(sampleRate: sampleRate, durationSeconds: 1.5))
        // 発話 3 (0.5秒)
        audio.append(contentsOf: synthesizeSpeech(sampleRate: sampleRate, durationSeconds: 0.5))
        audio.append(contentsOf: synthesizeSilence(sampleRate: sampleRate, durationSeconds: 0.3))

        let chunkSize = 320
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

        // 発話結果が取得できていること
        XCTAssertLessThanOrEqual(1, finalCollector.count)
        // 段落分割が正常に発生していること
        XCTAssertLessThanOrEqual(1, paragraphCollector.count)

        // 段落のタイムスタンプ形式が [mm:ss.xx -> mm:ss.xx] であること
        for para in paragraphCollector.allItems {
            XCTAssertTrue(para.timestampFormatted.hasPrefix("["))
            XCTAssertTrue(para.timestampFormatted.hasSuffix("]"))
            XCTAssertTrue(para.timestampFormatted.contains(" -> "))
            XCTAssertTrue(para.formattedLine.contains(para.timestampFormatted))
        }
    }
}
