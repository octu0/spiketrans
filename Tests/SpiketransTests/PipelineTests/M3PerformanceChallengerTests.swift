import XCTest
import Foundation
@testable import Spiketrans
#if canImport(Darwin)
import Darwin
#endif

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

/// Milestone M3 (Two-Stage STT Pipeline & Streaming Decoder) パフォーマンステスト・負荷検証担当 Challenger (Challenger 2) テストスイート
final class M3PerformanceChallengerTests: XCTestCase {

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

    private func synthesizeVowelSpeech(sampleRate: Int, durationSeconds: Float, amplitude: Float = 0.5) -> [Float] {
        let sampleCount = Int(Float(sampleRate) * durationSeconds)
        var pcm = [Float](repeating: 0.0, count: sampleCount)
        let twoPi = 2.0 * Float.pi

        var i = 0
        while i < sampleCount {
            let t = Float(i) / Float(sampleRate)
            let f0 = sin(twoPi * 220.0 * t)
            let f1 = 0.6 * sin(twoPi * 800.0 * t)
            let f2 = 0.4 * sin(twoPi * 1200.0 * t)
            let f3 = 0.2 * sin(twoPi * 2600.0 * t)
            pcm[i] = amplitude * (f0 + f1 + f2 + f3)
            i += 1
        }
        return pcm
    }

    private func synthesizeSilence(sampleRate: Int, durationSeconds: Float) -> [Float] {
        let sampleCount = Int(Float(sampleRate) * durationSeconds)
        return [Float](repeating: 0.0, count: sampleCount)
    }

    private func createTestNetworks() -> (acoustic: MatryoshkaNetwork, language: MatryoshkaNetwork, vocab: TextVocabulary) {
        let vocab = TextVocabulary()
        let ac = MatryoshkaNetwork(inputDim: 64, maxHiddenDim: 256, outputDim: vocab.size, timeSteps: 4)
        let lm = MatryoshkaNetwork(inputDim: 64, maxHiddenDim: 256, outputDim: vocab.size, timeSteps: 4)
        return (acoustic: ac, language: lm, vocab: vocab)
    }

    // MARK: - 1. 20,000 フレーム (約400秒分) 連続ストリーミング & O(1) メモリ空間維持実証

    func testStreaming20000FramesO1MemoryFootprint() {
        let (acNet, lmNet, vocab) = createTestNetworks()

        let config = StreamingTranscriberConfig(
            slice: .base,
            beamWidth: 1
        )

        let transcriber = StreamingTranscriber(
            config: config,
            acousticNetwork: acNet,
            languageNetwork: lmNet,
            textVocabulary: vocab
        )

        let finalCounter = AtomicCounter()
        transcriber.onFinalResult = { _ in
            finalCounter.increment()
        }

        let sampleRate = 16000
        let hopSize = config.dspConfig.hopSize // 160 samples (10ms)
        let totalFrames = 20000
        let totalAudioDuration = Float(totalFrames * hopSize) / Float(sampleRate) // 200.0秒

        // 変化するピッチ・倍音を持つ合成音声チャンク
        var syntheticChunk = [Float](repeating: 0.0, count: hopSize)

        var rssAt2000: UInt64 = 0
        var rssAt5000: UInt64 = 0
        var rssAt10000: UInt64 = 0
        var rssAt15000: UInt64 = 0
        var rssAt20000: UInt64 = 0

        let startTime = CFAbsoluteTimeGetCurrent()

        var frame = 0
        while frame < totalFrames {
            // 発話 (100フレーム = 1秒) と無音 (30フレーム = 0.3秒) の周期的繰り返し
            let cycle = frame % 130
            if cycle < 100 {
                let f0: Float = 150.0 + 80.0 * sin(Float(frame) * 0.05)
                let factor = (2.0 * Float.pi * f0) / Float(sampleRate)
                var s = 0
                while s < hopSize {
                    let t = Float(s) * factor
                    syntheticChunk[s] = 0.4 * sin(t) + 0.2 * sin(2.0 * t) + 0.1 * sin(3.0 * t)
                    s += 1
                }
            } else {
                var s = 0
                while s < hopSize {
                    syntheticChunk[s] = 0.0
                    s += 1
                }
            }

            syntheticChunk.withUnsafeBufferPointer { buf in
                transcriber.appendAudio(pcmPtr: buf.baseAddress!, count: hopSize)
            }

            if frame == 2000 {
                rssAt2000 = getResidentMemoryBytes()
            }
            if frame == 5000 {
                rssAt5000 = getResidentMemoryBytes()
            }
            if frame == 10000 {
                rssAt10000 = getResidentMemoryBytes()
            }
            if frame == 15000 {
                rssAt15000 = getResidentMemoryBytes()
            }
            if frame == 19999 {
                rssAt20000 = getResidentMemoryBytes()
            }

            frame += 1
        }
        transcriber.flush()

        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        let rtf = Float(elapsed) / totalAudioDuration
        let fps = Float(totalFrames) / Float(elapsed)

        let mem2kMB = Double(rssAt2000) / (1024.0 * 1024.0)
        let mem5kMB = Double(rssAt5000) / (1024.0 * 1024.0)
        let mem10kMB = Double(rssAt10000) / (1024.0 * 1024.0)
        let mem15kMB = Double(rssAt15000) / (1024.0 * 1024.0)
        let mem20kMB = Double(rssAt20000) / (1024.0 * 1024.0)
        let growthMB = Double(Int64(rssAt20000) - Int64(rssAt2000)) / (1024.0 * 1024.0)

        print("\n==================================================")
        print("=== M3 20,000 Frames Streaming Load Test Result ===")
        print("==================================================")
        print("Total frames processed: \(totalFrames)")
        print("Simulated audio:        \(String(format: "%.2f", totalAudioDuration)) s (\(Int(totalAudioDuration / 60)) min \(Int(totalAudioDuration) % 60) s)")
        print("Elapsed time:           \(String(format: "%.4f", elapsed)) s")
        print("Throughput:             \(String(format: "%.1f", fps)) frames/sec")
        print("Real-Time Factor (RTF): \(String(format: "%.6f", rtf)) xRT")
        print("Final results emitted:  \(finalCounter.value)")
        print("\n--- Memory Profile (RSS) ---")
        print("RSS at frame  2,000 (warm-up): \(String(format: "%.3f", mem2kMB)) MB")
        print("RSS at frame  5,000:          \(String(format: "%.3f", mem5kMB)) MB")
        print("RSS at frame 10,000:          \(String(format: "%.3f", mem10kMB)) MB")
        print("RSS at frame 15,000:          \(String(format: "%.3f", mem15kMB)) MB")
        print("RSS at frame 20,000 (final):   \(String(format: "%.3f", mem20kMB)) MB")
        print("Memory Growth (2k -> 20k):     \(String(format: "%.3f", growthMB)) MB")
        print("==================================================")

        // メモリリークなし検証 (OS ページキャッシュ変動許容 5.0 MB 以内)
        XCTAssertLessThanOrEqual(growthMB, 5.0, "Memory growth across 20,000 frames must remain flat (O(1) memory)")
        // 発話セグメントが正常に複数回確定したこと
        XCTAssertLessThanOrEqual(10, finalCounter.value, "Multiple speech segments must be finalized over 20,000 frames")
    }

    // MARK: - 2. End-to-End スループット & レイテンシ (RTF) 測定

    func testStreamingEndToEndLatencyAndThroughput() {
        let (acNet, lmNet, vocab) = createTestNetworks()

        let slices: [MatryoshkaSlice] = [.base, .middle, .high]
        let beamOptions = [1, 4]

        print("\n==================================================")
        print("=== M3 Two-Stage STT End-to-End Benchmark ===")
        print("==================================================")

        var sIdx = 0
        while sIdx < slices.count {
            let slice = slices[sIdx]

            var bIdx = 0
            while bIdx < beamOptions.count {
                let beam = beamOptions[bIdx]

                let config = StreamingTranscriberConfig(
                    slice: slice,
                    useQuantization: false,
                    beamWidth: beam
                )

                let transcriber = StreamingTranscriber(
                    config: config,
                    acousticNetwork: acNet,
                    languageNetwork: lmNet,
                    textVocabulary: vocab
                )

                let finalCounter = AtomicCounter()
                transcriber.onFinalResult = { _ in
                    finalCounter.increment()
                }

                let sampleRate = 16000
                // 1.0秒発話 + 0.3秒無音 (計 130 フレーム) x 10回 = 13秒音声 (1,300フレーム)
                var audio: [Float] = []
                var rep = 0
                while rep < 10 {
                    audio.append(contentsOf: self.synthesizeVowelSpeech(sampleRate: sampleRate, durationSeconds: 1.0))
                    audio.append(contentsOf: self.synthesizeSilence(sampleRate: sampleRate, durationSeconds: 0.3))
                    rep += 1
                }

                let totalAudioSec = Float(audio.count) / Float(sampleRate)
                let chunkSize = 160
                let startT = CFAbsoluteTimeGetCurrent()

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

                let elapsedT = CFAbsoluteTimeGetCurrent() - startT
                let rtf = Float(elapsedT) / totalAudioSec
                let fps = Float(audio.count / chunkSize) / Float(elapsedT)
                let latencyPerFrameMs = (elapsedT / Double(audio.count / chunkSize)) * 1000.0

                print("[Slice: \(slice), Beam: \(beam)]")
                print("  Audio Duration:  \(String(format: "%.2f", totalAudioSec)) s")
                print("  Processing Time: \(String(format: "%.4f", elapsedT)) s")
                print("  Throughput:      \(String(format: "%.1f", fps)) frames/sec")
                print("  Per-Frame Time:  \(String(format: "%.3f", latencyPerFrameMs)) ms/frame")
                print("  RTF:             \(String(format: "%.6f", rtf)) xRT")
                print("  Final Results:   \(finalCounter.value)")

                XCTAssertEqual(finalCounter.value, 10, "Each utterance must yield 1 final result")
                // デバッグビルドでもリアルタイム(RTF < 1.0)を上回るスループット
                XCTAssertLessThan(rtf, 1.0, "RTF must be faster than real-time in debug build")

                bIdx += 1
            }

            sIdx += 1
        }
        print("==================================================")
    }

    // MARK: - 3. 量子化エンジン併用時の 10,000 フレーム連続ストリーミング検証

    func testStreamingQuantizedEngineLongRunningStability() {
        let (acNet, lmNet, vocab) = createTestNetworks()

        let config = StreamingTranscriberConfig(
            slice: .high,
            useQuantization: true,
            beamWidth: 1
        )

        let transcriber = StreamingTranscriber(
            config: config,
            acousticNetwork: acNet,
            languageNetwork: lmNet,
            textVocabulary: vocab
        )

        let finalCounter = AtomicCounter()
        transcriber.onFinalResult = { _ in
            finalCounter.increment()
        }

        let sampleRate = 16000
        let hopSize = config.dspConfig.hopSize
        let totalFrames = 10000
        let totalAudioSec = Float(totalFrames * hopSize) / Float(sampleRate)

        var syntheticChunk = [Float](repeating: 0.0, count: hopSize)
        var initialRss: UInt64 = 0
        var finalRss: UInt64 = 0

        let startT = CFAbsoluteTimeGetCurrent()

        var frame = 0
        while frame < totalFrames {
            let cycle = frame % 100
            if cycle < 70 {
                let f0: Float = 200.0 + 50.0 * sin(Float(frame) * 0.1)
                let factor = (2.0 * Float.pi * f0) / Float(sampleRate)
                var s = 0
                while s < hopSize {
                    syntheticChunk[s] = 0.5 * sin(Float(s) * factor)
                    s += 1
                }
            } else {
                var s = 0
                while s < hopSize {
                    syntheticChunk[s] = 0.0
                    s += 1
                }
            }

            syntheticChunk.withUnsafeBufferPointer { buf in
                transcriber.appendAudio(pcmPtr: buf.baseAddress!, count: hopSize)
            }

            if frame == 2000 {
                initialRss = getResidentMemoryBytes()
            }

            frame += 1
        }
        transcriber.flush()
        finalRss = getResidentMemoryBytes()

        let elapsedT = CFAbsoluteTimeGetCurrent() - startT
        let rtf = Float(elapsedT) / totalAudioSec
        let growthMB = Double(Int64(finalRss) - Int64(initialRss)) / (1024.0 * 1024.0)

        print("\n--- Quantized Streaming 10,000 Frames Result ---")
        print("Simulated audio: \(totalAudioSec) s")
        print("Elapsed time:   \(String(format: "%.4f", elapsedT)) s")
        print("RTF:            \(String(format: "%.6f", rtf)) xRT")
        print("Memory growth:  \(String(format: "%.3f", growthMB)) MB")
        print("Final results:  \(finalCounter.value)")

        XCTAssertLessThanOrEqual(growthMB, 5.0, "Quantized transcriber memory growth must be bounded")
        XCTAssertLessThanOrEqual(10, finalCounter.value, "Quantized transcriber must finalize segments")
    }

    // MARK: - 4. 部分認識結果 (Partial Results) の遅延 & 発火レート検証

    func testStreamingPartialResultLatencyAndCadence() {
        let (acNet, lmNet, vocab) = createTestNetworks()

        let transcriber = StreamingTranscriber(
            acousticNetwork: acNet,
            languageNetwork: lmNet,
            textVocabulary: vocab
        )

        let collector = ResultCollector<TranscriptionResult>()
        transcriber.onPartialResult = { res in
            collector.append(res)
        }

        let sampleRate = 16000
        // 2.0 秒 (200 フレーム) の発話音声
        let audio = synthesizeVowelSpeech(sampleRate: sampleRate, durationSeconds: 2.0)
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

        print("\n--- Partial Results Verification ---")
        print("Partial count: \(collector.count)")
        if 0 < collector.count {
            print("First partial: '\(collector[0].text)' (start: \(collector[0].startTimeSeconds)s, end: \(collector[0].endTimeSeconds)s)")
            print("Last partial:  '\(collector[collector.count - 1].text)' (start: \(collector[collector.count - 1].startTimeSeconds)s, end: \(collector[collector.count - 1].endTimeSeconds)s)")
        }

        // 200フレームで 10フレーム毎発火するため、約15〜20回の部分結果が通知されること
        XCTAssertLessThanOrEqual(10, collector.count, "Must emit frequent partial results during 2s speech")
        for res in collector.allItems {
            XCTAssertFalse(res.isFinal)
            XCTAssertLessThanOrEqual(res.startTimeSeconds, res.endTimeSeconds)
        }
    }
}
