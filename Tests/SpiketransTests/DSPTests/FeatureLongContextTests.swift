import XCTest
@testable import Spiketrans

/// 入力特徴の末尾に付ける背景音の要約 (平滑メルの指数移動平均)
final class FeatureLongContextTests: XCTestCase {
    private func tone(seconds: Float, amplitude: Float) -> [Float] {
        let n = Int(seconds * 16000.0)
        var pcm = [Float](repeating: 0.0, count: n)
        var i = 0
        while i < n {
            pcm[i] = amplitude * sin(2.0 * Float.pi * 440.0 * Float(i) / 16000.0)
            i += 1
        }
        return pcm
    }

    func testInputDimLayoutHelpers() {
        XCTAssertEqual(StreamingFeatureFrontEnd.acousticInputDim(stack: 4), 512)
        XCTAssertEqual(StreamingFeatureFrontEnd.acousticInputDim(stack: 4, longContext: true), 576)
        XCTAssertEqual(StreamingFeatureFrontEnd.frameStack(forInputDim: 512), 4)
        XCTAssertEqual(StreamingFeatureFrontEnd.frameStack(forInputDim: 576), 4)
        XCTAssertEqual(StreamingFeatureFrontEnd.hasLongContext(inputDim: 512), false)
        XCTAssertEqual(StreamingFeatureFrontEnd.hasLongContext(inputDim: 576), true)
    }

    /// 要約を付けても前半 512 次元は付けない場合と同じで、末尾 64 次元は平滑メルを遅れて追う
    func testContextAppendsWithoutChangingStackedFeatures() {
        let pcm = tone(seconds: 3.0, amplitude: 0.1)
        let plain = SpeechDataset.extractFeaturesFromPCM(pcmData: pcm, frameStack: 4)
        let withContext = SpeechDataset.extractFeaturesFromPCM(pcmData: pcm, frameStack: 4, longContext: true)
        XCTAssertEqual(plain.count, withContext.count)
        XCTAssertEqual(withContext[0].count, 576)
        var t = 0
        while t < plain.count {
            XCTAssertEqual(Array(withContext[t][0..<512]), plain[t], "frame \(t)")
            t += 1
        }
        // 定常音なので、十分時間が経てば要約は最後のホップの平滑メルに近づく
        let last = withContext[withContext.count - 2]
        let lastMel = Array(last[(3 * 128)..<(3 * 128 + 64)])
        let context = Array(last[512..<576])
        var maxDiff: Float = 0.0
        var c = 0
        while c < 64 {
            maxDiff = max(maxDiff, abs(lastMel[c] - context[c]))
            c += 1
        }
        XCTAssertLessThan(maxDiff, 0.05)
    }

    /// 音量が急に変わっても、要約は 1 秒の時定数でゆっくり追う
    func testContextFollowsSlowly() {
        var pcm = tone(seconds: 2.0, amplitude: 0.01)
        pcm.append(contentsOf: tone(seconds: 1.0, amplitude: 0.3))
        let feats = SpeechDataset.extractFeaturesFromPCM(pcmData: pcm, frameStack: 4, longContext: true)
        // 切り替わり直後 (2.0 秒 = 50 フレーム目の少し後) の要約は、その時点の平滑メルより十分小さい
        let t = 52
        let mel = feats[t][(3 * 128)..<(3 * 128 + 64)].reduce(0, +)
        let ctx = feats[t][512..<576].reduce(0, +)
        XCTAssertLessThan(ctx, mel)
    }
}
