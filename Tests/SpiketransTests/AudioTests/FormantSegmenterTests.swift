import Testing
import Foundation
@testable import Spiketrans

@Suite("FormantSegmenterTests - フォルマント動的音素境界検出テスト")
struct FormantSegmenterTests {

    @Test("合成正弦波音声からの動的境界検出")
    func testSyntheticToneBoundaryDetection() {
        let sampleRate = 16000
        let durationSec: Float = 2.0
        let totalSamples = Int(Float(sampleRate) * durationSec)

        // 周波数が段階的に切り替わる合成波形 (音素遷移のシミュレーション)
        var pcm = [Float](repeating: 0.0, count: totalSamples)
        var i = 0
        while i < totalSamples {
            let t = Float(i) / Float(sampleRate)
            let freq: Float
            switch t {
            case ..<0.5:
                freq = 300.0 // 第1音素
            case ..<1.0:
                freq = 800.0 // 第2音素 (急変)
            case ..<1.5:
                freq = 1500.0 // 第3音素
            default:
                freq = 0.0 // 無音
            }
            pcm[i] = sin(2.0 * Float.pi * freq * t) * 0.5
            i += 1
        }

        let boundaries = FormantSegmenter.detectBoundaries(
            pcmData: pcm,
            sampleRate: sampleRate,
            minChunkFrames: 8,
            maxChunkFrames: 36
        )

        #expect(0 < boundaries.count)
        // 境界インデックスが昇順にソートされていること
        var k = 1
        while k < boundaries.count {
            #expect(boundaries[k - 1] < boundaries[k])
            k += 1
        }
    }

    @Test("CosineLRScheduler 学習率スケジューラの挙動検証")
    func testCosineScheduler() {
        let scheduler = CosineLRScheduler(lrMax: 0.025, lrMin: 0.002, totalEpochs: 60, warmupEpochs: 3)
        
        let lr1 = scheduler.learningRate(forEpoch: 1)
        let lr3 = scheduler.learningRate(forEpoch: 3)
        let lr30 = scheduler.learningRate(forEpoch: 30)
        let lr60 = scheduler.learningRate(forEpoch: 60)

        // Warmup: lr1 < lr3
        #expect(lr1 < lr3)
        #expect(abs(lr3 - 0.025) <= 1e-4)

        // Decay: lr60 < lr30 < lr3
        #expect(lr30 < lr3)
        #expect(lr60 < lr30)
        #expect(abs(lr60 - 0.002) <= 1e-4)
    }
}
