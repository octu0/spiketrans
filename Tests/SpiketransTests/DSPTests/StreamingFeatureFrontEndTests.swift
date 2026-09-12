import XCTest
@testable import Spiketrans

final class StreamingFeatureFrontEndTests: XCTestCase {
    func testTapLayoutIncludesProsodyAndShorterStack() {
        XCTAssertEqual(StreamingFeatureFrontEnd.melTapDim, 128)
        XCTAssertEqual(StreamingFeatureFrontEnd.prosodyDim, 4)
        XCTAssertEqual(StreamingFeatureFrontEnd.tapDim, 132)
        XCTAssertEqual(StreamingFeatureFrontEnd.defaultStack, 2)
        XCTAssertEqual(StreamingFeatureFrontEnd.acousticInputDim(), 264)
    }

    func testSilenceIsUnvoiced() {
        let n = 16000
        let pcm = [Float](repeating: 0.0, count: n)
        let frames = SpeechDataset.extractFeaturesFromPCM(pcmData: pcm, frameStack: 2)
        XCTAssertLessThan(0, frames.count)
        let voicedIdx = StreamingFeatureFrontEnd.melTapDim + 2
        var i = 0
        while i < frames.count {
            XCTAssertEqual(frames[i].count, 264)
            XCTAssertEqual(frames[i][voicedIdx], 0.0, accuracy: 1e-5)
            XCTAssertEqual(frames[i][voicedIdx - 2], 0.0, accuracy: 1e-5)
            i += 1
        }
    }

    func testSine200HzSetsVoicedAndLogF0() {
        let sampleRate: Float = 16000.0
        let f0: Float = 200.0
        let n = 8000
        var pcm = [Float](repeating: 0.0, count: n)
        var i = 0
        while i < n {
            pcm[i] = 0.2 * sinf(2.0 * Float.pi * f0 * Float(i) / sampleRate)
            i += 1
        }
        let frames = SpeechDataset.extractFeaturesFromPCM(pcmData: pcm, frameStack: 2)
        XCTAssertLessThan(4, frames.count)
        let logIdx = StreamingFeatureFrontEnd.melTapDim
        let voicedIdx = logIdx + 2
        let expectedLog = logf(f0 / StreamingFeatureFrontEnd.f0RefHz) / StreamingFeatureFrontEnd.f0Span
        var voicedCount = 0
        var logSum: Float = 0.0
        i = 0
        while i < frames.count {
            if 0.5 <= frames[i][voicedIdx] {
                voicedCount += 1
                logSum += frames[i][logIdx]
            }
            i += 1
        }
        XCTAssertLessThan(frames.count / 4, voicedCount)
        let meanLog = logSum / Float(voicedCount)
        XCTAssertEqual(meanLog, expectedLog, accuracy: 0.15)
    }

    func testStack2EmitsMoreFramesThanStack4() {
        let n = 8000
        var pcm = [Float](repeating: 0.0, count: n)
        var i = 0
        while i < n {
            pcm[i] = 0.1 * sinf(Float(i) * 0.1)
            i += 1
        }
        let s2 = SpeechDataset.extractFeaturesFromPCM(pcmData: pcm, frameStack: 2)
        let s4 = SpeechDataset.extractFeaturesFromPCM(pcmData: pcm, frameStack: 4)
        XCTAssertLessThan(s4.count, s2.count)
        XCTAssertEqual(s2[0].count, 264)
        XCTAssertEqual(s4[0].count, 528)
    }
}
