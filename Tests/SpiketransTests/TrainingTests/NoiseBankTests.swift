import XCTest
@testable import Spiketrans

final class NoiseBankTests: XCTestCase {
    private func tone(count: Int, amplitude: Float, period: Int) -> [Float] {
        var out = [Float](repeating: 0.0, count: count)
        var i = 0
        while i < count {
            out[i] = amplitude * sin(Float(i) * 2.0 * Float.pi / Float(period))
            i += 1
        }
        return out
    }

    func testMixMatchesRequestedSNR() {
        let speech = tone(count: 16000, amplitude: 0.2, period: 100)
        let noise = tone(count: 4000, amplitude: 0.7, period: 37)
        let mixed = NoiseBank.mix(speech: speech, noise: noise, offset: 123, snrDB: 10.0)
        XCTAssertEqual(mixed.count, speech.count)
        var residual = [Float](repeating: 0.0, count: speech.count)
        var i = 0
        while i < speech.count {
            residual[i] = mixed[i] - speech[i]
            i += 1
        }
        let snr = 20.0 * log10(NoiseBank.rms(speech) / NoiseBank.rms(residual))
        XCTAssertEqual(snr, 10.0, accuracy: 0.2)
    }

    func testShortNoiseIsTiledFromOffset() {
        let speech = [Float](repeating: 0.1, count: 10)
        let noise: [Float] = [1.0, 2.0, 3.0]
        let mixed = NoiseBank.mix(speech: speech, noise: noise, offset: 1, snrDB: 0.0)
        // 雑音は 2,3,1,2,3,... の並びで、SNR 0 dB なら雑音 RMS が発話 RMS (0.1) に揃う
        let residual0 = mixed[0] - speech[0]
        let residual1 = mixed[1] - speech[1]
        let residual2 = mixed[2] - speech[2]
        XCTAssertEqual(residual1 / residual0, 1.5, accuracy: 1e-4)
        XCTAssertEqual(residual2 / residual0, 0.5, accuracy: 1e-4)
        var acc: Float = 0.0
        var i = 0
        while i < mixed.count {
            let r = mixed[i] - speech[i]
            acc += r * r
            i += 1
        }
        XCTAssertEqual((acc / Float(mixed.count)).squareRoot(), 0.1, accuracy: 1e-3)
    }

    func testSilentSpeechAndEmptyNoiseAreLeftUntouched() {
        let silence = [Float](repeating: 0.0, count: 100)
        XCTAssertEqual(NoiseBank.mix(speech: silence, noise: [0.5, -0.5], offset: 0, snrDB: 5.0), silence)
        let speech = tone(count: 100, amplitude: 0.3, period: 10)
        XCTAssertEqual(NoiseBank.mix(speech: speech, noise: [], offset: 0, snrDB: 5.0), speech)
    }

    func testBankMixesWithinConfiguredSNRRange() {
        let bank = NoiseBank(clips: [tone(count: 8000, amplitude: 0.5, period: 41), []])
        XCTAssertEqual(bank.count, 1)
        XCTAssertEqual(bank.totalSeconds, 0.5, accuracy: 1e-6)
        let speech = tone(count: 16000, amplitude: 0.2, period: 100)
        var rng = SystemRandomNumberGenerator()
        var mixedCount = 0
        var trial = 0
        while trial < 40 {
            let mixed = bank.mix(speech, using: &rng)
            var acc: Float = 0.0
            var i = 0
            while i < mixed.count {
                let r = mixed[i] - speech[i]
                acc += r * r
                i += 1
            }
            let residualRMS = (acc / Float(mixed.count)).squareRoot()
            if 1e-6 < residualRMS {
                mixedCount += 1
                let snr = 20.0 * log10(NoiseBank.rms(speech) / residualRMS)
                XCTAssertGreaterThanOrEqual(snr, NoiseBank.snrRange.lowerBound - 0.3)
                XCTAssertLessThanOrEqual(snr, NoiseBank.snrRange.upperBound + 0.3)
            }
            trial += 1
        }
        // 確率 0.5 なので 40 回のうち一部は素通し、一部は付加される
        XCTAssertGreaterThan(mixedCount, 5)
        XCTAssertLessThan(mixedCount, 35)
    }
}
