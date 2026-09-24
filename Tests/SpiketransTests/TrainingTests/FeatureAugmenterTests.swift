import XCTest
@testable import Spiketrans

final class FeatureAugmenterTests: XCTestCase {
    private struct SeededGenerator: RandomNumberGenerator {
        var state: UInt64
        mutating func next() -> UInt64 {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return state
        }
    }

    func testResampleChangesLengthAndKeepsShape() {
        var pcm = [Float](repeating: 0.0, count: 16000)
        var i = 0
        while i < pcm.count {
            pcm[i] = sin(Float(i) * 0.01)
            i += 1
        }
        let faster = FeatureAugmenter.resample(pcm, speed: 1.1)
        let slower = FeatureAugmenter.resample(pcm, speed: 0.9)
        XCTAssertEqual(faster.count, Int(16000.0 / 1.1))
        XCTAssertEqual(slower.count, Int(16000.0 / 0.9))
        // 1.1 倍速なら出力の i 番目は入力の 1.1 i 番目付近の値
        XCTAssertEqual(faster[1000], sin(1100.0 * 0.01), accuracy: 1e-3)
        XCTAssertEqual(FeatureAugmenter.resample(pcm, speed: 1.0), pcm)
    }

    func testSpecAugmentMasksWholeFramesAndMatchingBandsAcrossStack() {
        let dim = StreamingFeatureFrontEnd.acousticInputDim(stack: 4)
        let frames = 50
        let features = [[Float]](repeating: [Float](repeating: 1.0, count: dim), count: frames)
        var rng = SeededGenerator(state: 7)
        let out = FeatureAugmenter.specAugment(features, using: &rng)
        XCTAssertEqual(out.count, frames)
        var maskedFrames = 0
        var bandZeroPattern: [Bool]? = nil
        var t = 0
        while t < frames {
            if out[t].allSatisfy({ $0 == 0.0 }) {
                maskedFrames += 1
            } else {
                // 周波数マスク: 各ホップの平滑メルと差分で同じ帯が 0 になっている
                let mel = StreamingFeatureFrontEnd.melChannels
                let tap = StreamingFeatureFrontEnd.tapDim
                var pattern = [Bool](repeating: false, count: mel)
                var b = 0
                while b < mel {
                    pattern[b] = out[t][b] == 0.0
                    var s = 0
                    while s < 4 {
                        XCTAssertEqual(out[t][s * tap + b] == 0.0, pattern[b], "ホップ \(s) の平滑メル帯 \(b) が揃っていない")
                        XCTAssertEqual(out[t][s * tap + mel + b] == 0.0, pattern[b], "ホップ \(s) の差分帯 \(b) が揃っていない")
                        s += 1
                    }
                    b += 1
                }
                if let previous = bandZeroPattern {
                    XCTAssertEqual(previous, pattern, "周波数マスクがフレームで変わっている")
                }
                bandZeroPattern = pattern
            }
            t += 1
        }
        XCTAssertLessThanOrEqual(maskedFrames, FeatureAugmenter.timeMasks * min(FeatureAugmenter.maxTimeMaskFrames, frames / 10))
        let maskedBands = bandZeroPattern?.filter { $0 }.count ?? 0
        XCTAssertLessThanOrEqual(maskedBands, FeatureAugmenter.frequencyMasks * FeatureAugmenter.maxFrequencyMaskBands)
    }

    func testDisabledAugmenterHasNoHooks() {
        let off = FeatureAugmenter(speed: false, specAugment: false)
        XCTAssertNil(off.pcmTransform)
        XCTAssertNil(off.featureTransform)
        let on = FeatureAugmenter(speed: true, specAugment: true)
        XCTAssertNotNil(on.pcmTransform)
        XCTAssertNotNil(on.featureTransform)
    }
}
