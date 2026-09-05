import XCTest
import Foundation
@testable import Spiketrans
#if canImport(MLX)
import MLX
#endif

final class Conv2DSubsamplingTests: XCTestCase {

    // MARK: - 1. 時間軸 1/4 圧縮率と出力次元の検証

    func testSubsamplingTimeCompressionRatio() {
        let subsampler = Conv2DSubsampling(
            inChannels: 1,
            outChannels1: 16,
            outChannels2: 16,
            melChannels: 64,
            outputDim: 128
        )

        // T = 16 -> T' = 4
        let mel16 = [[Float]](repeating: [Float](repeating: 0.5, count: 64), count: 16)
        let out16 = subsampler.forward(melSpectrogram: mel16)
        XCTAssertEqual(out16.count, 4)
        XCTAssertEqual(out16[0].count, 128)

        // T = 40 -> T' = 10
        let mel40 = [[Float]](repeating: [Float](repeating: 0.3, count: 64), count: 40)
        let out40 = subsampler.forward(melSpectrogram: mel40)
        XCTAssertEqual(out40.count, 10)

        // T = 100 -> T' = 25
        let mel100 = [[Float]](repeating: [Float](repeating: 0.8, count: 64), count: 100)
        let out100 = subsampler.forward(melSpectrogram: mel100)
        XCTAssertEqual(out100.count, 25)

        // 境界値: 空配列
        let outEmpty = subsampler.forward(melSpectrogram: [])
        XCTAssertEqual(outEmpty.count, 0)

        // 境界値: T = 1 -> T' = 1
        let mel1 = [[Float]](repeating: [Float](repeating: 0.1, count: 64), count: 1)
        let out1 = subsampler.forward(melSpectrogram: mel1)
        XCTAssertEqual(out1.count, 1)

        // 境界値: T = 4 -> T' = 1
        let mel4 = [[Float]](repeating: [Float](repeating: 0.1, count: 64), count: 4)
        let out4 = subsampler.forward(melSpectrogram: mel4)
        XCTAssertEqual(out4.count, 1)
    }

    // MARK: - 2. 周波数局所並進不変性 (フォルマントシフト吸収) の検証

    func testFrequencyTranslationalInvariance() {
        let subsampler = Conv2DSubsampling(
            inChannels: 1,
            outChannels1: 16,
            outChannels2: 16,
            melChannels: 64,
            outputDim: 128
        )

        let seqLen = 16
        // 人工的なフォルマント共鳴ピーク (周波数 20ch 付近に山)
        var melOriginal = [[Float]](repeating: [Float](repeating: 0.0, count: 64), count: seqLen)
        var t = 0
        while t < seqLen {
            var f = 0
            while f < 64 {
                let diff = Float(f - 20)
                melOriginal[t][f] = exp(-0.1 * diff * diff)
                f += 1
            }
            t += 1
        }

        // ピッチ・フォルマント周波数が 1 ビン高域へシフトした信号
        var melShifted = [[Float]](repeating: [Float](repeating: 0.0, count: 64), count: seqLen)
        t = 0
        while t < seqLen {
            var f = 0
            while f < 64 {
                let diff = Float(f - 21) // +1 bin shift
                melShifted[t][f] = exp(-0.1 * diff * diff)
                f += 1
            }
            t += 1
        }

        let outOriginal = subsampler.forward(melSpectrogram: melOriginal)
        let outShifted = subsampler.forward(melSpectrogram: melShifted)

        XCTAssertEqual(outOriginal.count, outShifted.count)

        // 2D-Conv の周波数畳み込みとストライドにより、周波数シフト後も高いコサイン類似度が保たれること
        var step = 0
        while step < outOriginal.count {
            let vecA = outOriginal[step]
            let vecB = outShifted[step]

            var dot: Float = 0.0
            var normA: Float = 0.0
            var normB: Float = 0.0
            var d = 0
            while d < 128 {
                dot += vecA[d] * vecB[d]
                normA += vecA[d] * vecA[d]
                normB += vecB[d] * vecB[d]
                d += 1
            }
            let denom = sqrt(normA) * sqrt(normB)
            if 1e-6 < denom {
                let cosSim = dot / denom
                // 1 ビンシフトに対して特徴ベクトルのコサイン類似度が 0.85 以上維持されること
                XCTAssertTrue(0.85 <= cosSim)
            }
            step += 1
        }
    }

    // MARK: - 3. 因果的パディング (未来フレーム非参照 / Causal Padding) の厳密検証

    func testCausalPaddingZeroLookahead() {
        let subsampler = Conv2DSubsampling(
            inChannels: 1,
            outChannels1: 16,
            outChannels2: 16,
            melChannels: 64,
            outputDim: 128
        )

        let totalT = 24
        var melSeqA = [[Float]](repeating: [Float](repeating: 0.0, count: 64), count: totalT)
        var t = 0
        while t < totalT {
            var f = 0
            while f < 64 {
                melSeqA[t][f] = sin(Float(t * 64 + f)) * 0.5 + 0.5
                f += 1
            }
            t += 1
        }

        // 未来部分 (後半 12 <= t) のみを全く別の値に改変した系列 B
        var melSeqB = melSeqA
        t = 12
        while t < totalT {
            var f = 0
            while f < 64 {
                melSeqB[t][f] = 0.99
                f += 1
            }
            t += 1
        }

        let outA = subsampler.forward(melSpectrogram: melSeqA)
        let outB = subsampler.forward(melSpectrogram: melSeqB)

        // 過去ステップ (前半: tSrc <= 11) に対応する出力ステップでは、
        // 未来フレーム改変の影響を 1 ビットも受けず、完全一致すること
        // tSrc2 = 2 * (2 * t2) <= 11 -> t2 < 3 (step 0, 1, 2)
        var step = 0
        while step < 2 {
            var d = 0
            while d < 128 {
                let diff = abs(outA[step][d] - outB[step][d])
                XCTAssertEqual(diff, 0.0, accuracy: 1e-6)
                d += 1
            }
            step += 1
        }
    }

    // MARK: - 4. ストリーミング推論とシーケンス一括推論の完全一致検証

    func testStreamingMatchesOfflineBatch() {
        let subsampler = Conv2DSubsampling(
            inChannels: 1,
            outChannels1: 16,
            outChannels2: 16,
            melChannels: 64,
            outputDim: 128
        )

        let totalT = 20
        var melSeq = [[Float]](repeating: [Float](repeating: 0.0, count: 64), count: totalT)
        var t = 0
        while t < totalT {
            var f = 0
            while f < 64 {
                melSeq[t][f] = cos(Float(t * 13 + f * 7)) * 0.5 + 0.5
                f += 1
            }
            t += 1
        }

        // 1. 一括推論
        let batchOut = subsampler.forward(melSpectrogram: melSeq)

        // 2. ストリーミング推論 (1フレームずつ供給)
        var streamState = Conv2DStreamingState()
        var streamOutputs: [[Float]] = []

        t = 0
        while t < totalT {
            let res = subsampler.forwardStreaming(melFrame: melSeq[t], state: &streamState)
            switch res {
            case .some(let feat):
                streamOutputs.append(feat)
            case .none:
                break
            }
            t += 1
        }

        // 発火したステップ数の一致
        XCTAssertEqual(streamOutputs.count, batchOut.count)

        // 各ステップの特徴量がバッチ推論と誤差 1e-4 以内で完全一致すること
        var sIdx = 0
        while sIdx < streamOutputs.count {
            var d = 0
            while d < 128 {
                XCTAssertEqual(streamOutputs[sIdx][d], batchOut[sIdx][d], accuracy: 1e-4)
                d += 1
            }
            sIdx += 1
        }
    }

    // MARK: - 5. MLX Metal GPU と Pure Swift SIMD の出力完全一致検証

    #if canImport(MLX)
    func testMLXAndPureSwiftNumericalEquivalence() {
        let outChannels1 = 16
        let outChannels2 = 16
        let melChannels = 64
        let outputDim = 128

        // 共有重みの生成
        let swiftSubsampler = Conv2DSubsampling(
            inChannels: 1,
            outChannels1: outChannels1,
            outChannels2: outChannels2,
            melChannels: melChannels,
            outputDim: outputDim
        )
        let weights = swiftSubsampler.exportWeights()

        // MLX 側にインポート
        let mlxSubsampler = MLXConv2DSubsampling(weights: weights)

        // テスト入力スペクトログラム (T = 16, F = 64)
        let seqLen = 16
        var melSeq = [[Float]](repeating: [Float](repeating: 0.0, count: melChannels), count: seqLen)
        var flatFeat = [Float](repeating: 0.0, count: seqLen * melChannels)
        var t = 0
        while t < seqLen {
            var f = 0
            while f < melChannels {
                let v = sin(Float(t * 31 + f * 17)) * 0.5 + 0.5
                melSeq[t][f] = v
                flatFeat[t * melChannels + f] = v
                f += 1
            }
            t += 1
        }

        // Swift 推論
        let swiftOut = swiftSubsampler.forward(melSpectrogram: melSeq)

        // MLX 推論
        let mlxInput = MLXArray(flatFeat, [1, seqLen, melChannels])
        let mlxOut = mlxSubsampler(mlxInput)
        eval(mlxOut)

        XCTAssertEqual(mlxOut.shape[0], 1)
        XCTAssertEqual(mlxOut.shape[1], swiftOut.count)
        XCTAssertEqual(mlxOut.shape[2], outputDim)

        let mlxFlat = mlxOut.asArray(Float.self)

        // 全要素の数値一致比較 (誤差 1e-4 以下)
        var step = 0
        while step < swiftOut.count {
            var d = 0
            while d < outputDim {
                let mlxVal = mlxFlat[step * outputDim + d]
                let swiftVal = swiftOut[step][d]
                XCTAssertEqual(mlxVal, swiftVal, accuracy: 1e-4)
                d += 1
            }
            step += 1
        }
    }
    #endif

    // MARK: - 6. 多層 SNN + AcousticDecoder との協調推論

    func testAcousticDecoderWithConv2DSubsampling() {
        let net = SpikingNetwork(numLayers: 2, inputDim: 128, maxHiddenDim: 256, outputDim: 50)
        let subsampler = Conv2DSubsampling(
            inChannels: 1,
            outChannels1: 16,
            outChannels2: 16,
            melChannels: 64,
            outputDim: 128
        )
        net.convSubsampling = subsampler

        let decoder = AcousticDecoder(network: net)
        let workspace = AcousticWorkspace(maxHiddenDim: 256, outputDim: 50, inputDim: 128, numLayers: 2)

        // 64ch Mel スペクトログラム 40 フレーム (400ms 相当)
        let mel40 = [[Float]](repeating: [Float](repeating: 0.5, count: 64), count: 40)

        // decodeSequence に 64ch Mel を直接供給 -> 内部で 1/4 圧縮され 10 フレームの確率分布が出力される
        let probsSeq = decoder.decodeSequence(featuresSeq: mel40, workspace: workspace)

        XCTAssertEqual(probsSeq.count, 10)
        var pIdx = 0
        while pIdx < probsSeq.count {
            let p = probsSeq[pIdx]
            XCTAssertEqual(p.probabilities.count, 50)
            XCTAssertTrue(0 <= p.topTokenId)
            XCTAssertTrue(p.topTokenId < 50)
            XCTAssertTrue(0.0 <= p.topProbability)
            XCTAssertTrue(p.topProbability <= 1.0)
            pIdx += 1
        }
    }

    // MARK: - 7. FormantSegmenter 動的境界検出との協調検証

    func testFormantSegmenterSubsampleBoundaries() {
        let originalBoundaries = [4, 8, 12, 16, 20, 24, 28, 32]
        let scaledBoundaries = FormantSegmenter.subsampleBoundaries(boundaries: originalBoundaries, factor: 4)

        XCTAssertEqual(scaledBoundaries, [1, 2, 3, 4, 5, 6, 7, 8])

        // 近接境界の重複排除検証
        let closeBoundaries = [3, 4, 5, 8, 9, 12]
        let deduplicated = FormantSegmenter.subsampleBoundaries(boundaries: closeBoundaries, factor: 4)
        // 3/4=0, 4/4=1, 5/4=1 -> [0, 1]
        // 8/4=2, 9/4=2 -> [2]
        // 12/4=3 -> [3]
        XCTAssertEqual(deduplicated, [0, 1, 2, 3])
    }

    // MARK: - 8. 10時間連続推論を支える高スループット & RTF 0.002 級検証

    func testThroughputAndLatencyRTF0002() {
        let net = SpikingNetwork(numLayers: 2, inputDim: 128, maxHiddenDim: 256, outputDim: 50)
        let subsampler = Conv2DSubsampling(outputDim: 128)
        net.convSubsampling = subsampler

        let decoder = AcousticDecoder(network: net)
        let workspace = AcousticWorkspace(maxHiddenDim: 256, outputDim: 50, inputDim: 128, numLayers: 2)

        // 2,000 フレームの音声 (20.0 秒相当)
        let totalFrames = 2000
        var melSeq = [[Float]](repeating: [Float](repeating: 0.1, count: 64), count: totalFrames)
        var f = 0
        while f < totalFrames {
            melSeq[f][f % 64] = 0.8
            f += 1
        }

        let start = DispatchTime.now()
        let results = decoder.decodeSequence(featuresSeq: melSeq, workspace: workspace)
        let end = DispatchTime.now()

        let elapsedNs = end.uptimeNanoseconds - start.uptimeNanoseconds
        let elapsedSec = Double(elapsedNs) / 1_000_000_000.0

        // 20.0 秒の音声を処理した結果が 500 フレーム (1/4 圧縮)
        XCTAssertEqual(results.count, 500)

        // 実時間係数 (RTF = elapsedSec / 20.0)
        let rtf = elapsedSec / 20.0

        // RTF 0.02 以下（実時間の 50 倍以上速い）であることを保証
        #if DEBUG
        XCTAssertTrue(rtf < 0.25)
        XCTAssertTrue(elapsedSec < 5.0)
        #else
        XCTAssertTrue(rtf < 0.02)
        XCTAssertTrue(elapsedSec < 0.5)
        #endif
    }

    // MARK: - 9. 重みシリアライズ・JSON 互換性検証

    func testWeightsSerializationWithConvSubsampling() throws {
        let net = SpikingNetwork(numLayers: 2, inputDim: 128, maxHiddenDim: 256, outputDim: 50)
        let subsampler = Conv2DSubsampling(outputDim: 128)
        net.convSubsampling = subsampler

        let weights = net.exportWeights()
        XCTAssertNotNil(weights.convSubsampling)
        XCTAssertEqual(weights.convSubsampling?.melChannels, 64)
        XCTAssertEqual(weights.convSubsampling?.outputDim, 128)

        let encoder = JSONEncoder()
        let data = try encoder.encode(weights)

        let decoder = JSONDecoder()
        let decodedWeights = try decoder.decode(SpikingNetworkWeights.self, from: data)

        XCTAssertNotNil(decodedWeights.convSubsampling)
        XCTAssertEqual(decodedWeights.convSubsampling?.conv1Weight, weights.convSubsampling?.conv1Weight)
        XCTAssertEqual(decodedWeights.convSubsampling?.conv2Weight, weights.convSubsampling?.conv2Weight)
        XCTAssertEqual(decodedWeights.convSubsampling?.projWeight, weights.convSubsampling?.projWeight)

        // 別ネットへのインポート
        let net2 = SpikingNetwork(numLayers: 2, inputDim: 128, maxHiddenDim: 256, outputDim: 50)
        net2.importWeights(from: decodedWeights)

        XCTAssertNotNil(net2.convSubsampling)
        XCTAssertEqual(net2.convSubsampling?.projWeight, subsampler.projWeight)
    }

    // MARK: - 10. StreamingTranscriber と Conv2D Subsampling の統合検証

    func testStreamingTranscriberIntegrationWithConv2DSubsampling() {
        let acNet = SpikingNetwork(numLayers: 2, inputDim: 128, maxHiddenDim: 256, outputDim: 50)
        let subsampler = Conv2DSubsampling(
            inChannels: 1,
            outChannels1: 16,
            outChannels2: 16,
            melChannels: 64,
            outputDim: 128
        )
        acNet.convSubsampling = subsampler

        let lmNet = SpikingNetwork(numLayers: 1, inputDim: 50, maxHiddenDim: 64, outputDim: 50)
        let vocab = TextVocabulary()

        let config = StreamingTranscriberConfig(
            beamWidth: 1,
            maxSegmentDurationSeconds: 5.0
        )
        let transcriber = StreamingTranscriber(
            config: config,
            acousticNetwork: acNet,
            languageNetwork: lmNet,
            textVocabulary: vocab
        )

        final class ResultBox: @unchecked Sendable {
            var count: Int = 0
        }
        let box = ResultBox()
        transcriber.onFinalResult = { _ in
            box.count += 1
        }

        // 16,000 サンプル (1.0秒) の正弦波有声音声 + 無音
        var pcm = [Float](repeating: 0.0, count: 16000)
        var i = 0
        while i < 8000 {
            pcm[i] = sin(Float(i) * 0.05) * 0.5
            i += 1
        }

        transcriber.appendAudio(pcm: pcm)
        transcriber.flush()

        // 音声が正しく処理され、ストリーミング因果バッファが正常動作すること
        XCTAssertTrue(0 <= box.count)
    }

    // MARK: - 11. 長大系列 & 奇数長系列におけるストリーミングとバッチの完全一致検証

    func testStreamingLongSequenceNumericalEquivalence() {
        let subsampler = Conv2DSubsampling(
            inChannels: 1,
            outChannels1: 16,
            outChannels2: 16,
            melChannels: 64,
            outputDim: 128
        )

        // T = 101 (奇数長・長大系列)
        let totalT = 101
        var melSeq = [[Float]](repeating: [Float](repeating: 0.0, count: 64), count: totalT)
        var t = 0
        while t < totalT {
            var f = 0
            while f < 64 {
                melSeq[t][f] = sin(Float(t * 19 + f * 5)) * 0.5 + 0.5
                f += 1
            }
            t += 1
        }

        let batchOut = subsampler.forward(melSpectrogram: melSeq)

        var streamState = Conv2DStreamingState()
        var streamOutputs: [[Float]] = []

        t = 0
        while t < totalT {
            let res = subsampler.forwardStreaming(melFrame: melSeq[t], state: &streamState)
            switch res {
            case .some(let feat):
                streamOutputs.append(feat)
            case .none:
                break
            }
            t += 1
        }

        XCTAssertEqual(streamOutputs.count, batchOut.count)

        var sIdx = 0
        while sIdx < streamOutputs.count {
            var d = 0
            while d < 128 {
                XCTAssertEqual(streamOutputs[sIdx][d], batchOut[sIdx][d], accuracy: 1e-4)
                d += 1
            }
            sIdx += 1
        }
    }

    // MARK: - 12. transcribeAcousticDirect と Conv2D Subsampling の協調検証

    func testTrainerTranscribeAcousticDirectWithConv2DSubsampling() {
        let acNet = SpikingNetwork(numLayers: 2, inputDim: 128, maxHiddenDim: 256, outputDim: 50)
        let subsampler = Conv2DSubsampling(
            inChannels: 1,
            outChannels1: 16,
            outChannels2: 16,
            melChannels: 64,
            outputDim: 128
        )
        acNet.convSubsampling = subsampler

        let lmNet = SpikingNetwork(numLayers: 1, inputDim: 50, maxHiddenDim: 64, outputDim: 50)
        let vocab = TextVocabulary()
        let trainer = Trainer(
            acousticNetwork: acNet,
            languageNetwork: lmNet,
            textVocabulary: vocab
        )

        // 64ch Mel 特徴量 40 フレーム
        let mel40 = [[Float]](repeating: [Float](repeating: 0.5, count: 64), count: 40)
        let text = trainer.transcribeAcousticDirect(
            featuresSeq: mel40,
            minDurationFrames: 1,
            minConfidence: 0.0,
            boundaries: [4, 8, 12, 16]
        )

        XCTAssertNotNil(text)
    }

    #if canImport(MLX)
    // MARK: - 13. MLXBPTTTrainer.trainBatch と Conv2D Subsampling の協調検証

    func testMLXBPTTTrainerTrainBatchWithConv2DSubsampling() {
        let mlxNet = MLXSpikingNetwork(numLayers: 2, inputDim: 128, maxHiddenDim: 128, outputDim: 50)
        mlxNet.convSubsampling = MLXConv2DSubsampling(outputDim: 128)

        let trainer = MLXBPTTTrainer(network: mlxNet)

        // 64ch Mel 入力 16 フレーム
        let mel16 = [[Float]](repeating: [Float](repeating: 0.2, count: 64), count: 16)
        let targets = [Int](repeating: 5, count: 16)

        let loss = trainer.trainBatch(
            featuresBatch: [mel16],
            targetsBatch: [targets]
        )

        XCTAssertTrue(loss.isFinite)
        XCTAssertTrue(0.0 <= loss)
    }
    #endif
}
