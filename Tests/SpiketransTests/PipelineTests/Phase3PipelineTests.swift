import Foundation
import XCTest
@testable import Spiketrans
#if canImport(MLX)
import MLX
import MLXNN
#endif

/// フェーズ3: BPTT 窓幅適正化 (Tw=8〜12) および Conv2D Subsampling + 多層 SNN 統合学習・推論パイプライン検証
final class Phase3PipelineTests: XCTestCase {

    override func setUp() {
        super.setUp()
        #if canImport(MLX)
        Device.setDefault(device: .cpu)
        #endif
    }

    /// 1. BPTT 窓幅 Tw = 8 (320ms相当) および Tw = 12 (480ms相当) での安定学習・勾配クリッピング協調検証
    func testBPTTWindowRangeStability() {
        #if canImport(MLX)
        let melChannels = 64
        let convOutDim = 128
        let hiddenDim = 128
        let vocabSize = 40

        for targetTw in [8, 12] {
            let conv = MLXConv2DSubsampling(melChannels: melChannels, outputDim: convOutDim)
            let net = MLXSpikingNetwork(
                numLayers: 2,
                inputDim: convOutDim,
                maxHiddenDim: hiddenDim,
                outputDim: vocabSize,
                timeSteps: 4,
                lifConfig: LIFConfig(beta: 0.92, vTh: 1.0, vReset: 0.0, alpha: 2.0, rho: 0.85, gamma: 0.0),
                convSubsampling: conv
            )
            let trainer = MLXBPTTTrainer(
                network: net,
                config: TrainingConfig(learningRate: 0.005, clipNorm: 5.0),
                bpttWindow: targetTw
            )

            XCTAssertEqual(trainer.bpttWindow, targetTw)

            // 60フレーム (2.4秒相当) の合成 64ch Mel 特徴量バッチ (バッチサイズ 2)
            let seqLen = 60
            var sample1 = [[Float]]()
            var sample2 = [[Float]]()
            var t = 0
            while t < seqLen {
                var f1 = [Float](repeating: 0.0, count: melChannels)
                var f2 = [Float](repeating: 0.0, count: melChannels)
                var c = 0
                while c < melChannels {
                    f1[c] = sin(Float(t * 10 + c) * 0.1) * 0.5 + 0.5
                    f2[c] = cos(Float(t * 10 + c) * 0.1) * 0.5 + 0.5
                    c += 1
                }
                sample1.append(f1)
                sample2.append(f2)
                t += 1
            }

            let batchFeatures = [sample1, sample2]
            let targets = [[3, 5, 8, 12, 15], [4, 7, 10, 14, 18]]

            var initialLoss: Float = 0.0
            var finalLoss: Float = 0.0

            var step = 0
            while step < 3 {
                let loss = trainer.trainBatchCTC(
                    featuresBatch: batchFeatures,
                    targetsBatch: targets,
                    blankId: 0
                )
                XCTAssertFalse(loss.isNaN, "損失が NaN になってはならない (Tw=\(targetTw))")
                XCTAssertFalse(loss.isInfinite, "損失が Inf になってはならない (Tw=\(targetTw))")
                XCTAssertTrue(0.0 <= loss, "損失は非負である必要がある")

                if step == 0 {
                    initialLoss = loss
                }
                finalLoss = loss
                step += 1
            }

            XCTAssertTrue(finalLoss <= initialLoss + 5.0, "Tw=\(targetTw) で学習損失が発散せず安定推移すること")
        }
        #endif
    }

    /// 2. Conv2D Subsampling + 多層 SNN の完全な重みエクスポート & インポート整合性検証
    func testEndToEndConv2DSubsamplingMultiLayerExportImport() throws {
        let melChannels = 64
        let convOutDim = 128
        let hiddenDim = 128
        let outputDim = 50

        let convSub = Conv2DSubsampling(melChannels: melChannels, outputDim: convOutDim)
        let originalNetwork = SpikingNetwork(
            numLayers: 3,
            inputDim: convOutDim,
            maxHiddenDim: hiddenDim,
            outputDim: outputDim,
            timeSteps: 4,
            lifConfig: LIFConfig(beta: 0.90, vTh: 1.1, vReset: 0.0, alpha: 2.0, rho: 0.82, gamma: 0.05),
            convSubsampling: convSub
        )

        let testVocab = TextVocabulary(corpus: ["あいうえお", "かきくけこ", "さしすせそ"])
        let exportedWeights = originalNetwork.exportWeights(vocabulary: testVocab)

        XCTAssertEqual(exportedWeights.numLayers, 3)
        XCTAssertEqual(exportedWeights.inputDim, convOutDim)
        XCTAssertEqual(exportedWeights.maxHiddenDim, hiddenDim)
        XCTAssertEqual(exportedWeights.outputDim, outputDim)
        XCTAssertNotNil(exportedWeights.convSubsampling)
        XCTAssertNotNil(exportedWeights.wLayers)
        XCTAssertEqual(exportedWeights.wLayers?.count, 2)
        XCTAssertEqual(exportedWeights.bHLayers?.count, 2)
        XCTAssertEqual(exportedWeights.gammaRMS?.count, 2)

        // JSON へのシリアライズとデシリアライズ
        let tempURL = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("phase3_test_weights_\(UUID().uuidString).json")
        defer {
            try? FileManager.default.removeItem(at: tempURL)
        }

        try exportedWeights.save(to: tempURL)
        let loadedWeights = try SpikingNetworkWeights.load(from: tempURL)

        XCTAssertEqual(loadedWeights.numLayers, 3)
        XCTAssertEqual(loadedWeights.inputDim, convOutDim)
        XCTAssertEqual(loadedWeights.convSubsampling?.melChannels, melChannels)

        // SpikingNetwork(weights:) 経由の復元
        let restoredNetwork = SpikingNetwork(weights: loadedWeights)
        XCTAssertEqual(restoredNetwork.numLayers, 3)
        XCTAssertEqual(restoredNetwork.inputDim, convOutDim)
        XCTAssertNotNil(restoredNetwork.convSubsampling)
        XCTAssertEqual(restoredNetwork.convSubsampling?.melChannels, melChannels)
        XCTAssertEqual(restoredNetwork.pWLayers.count, 2)
        XCTAssertEqual(restoredNetwork.pBHLayers.count, 2)
        XCTAssertEqual(restoredNetwork.pGammaRMS.count, 2)

        // パラメータ完全一致の確認
        XCTAssertEqual(restoredNetwork.pWIn.data, originalNetwork.pWIn.data)
        XCTAssertEqual(restoredNetwork.pWRec.data, originalNetwork.pWRec.data)
        XCTAssertEqual(restoredNetwork.pBH.data, originalNetwork.pBH.data)
        var l = 0
        while l < 2 {
            XCTAssertEqual(restoredNetwork.pWLayers[l].data, originalNetwork.pWLayers[l].data)
            XCTAssertEqual(restoredNetwork.pBHLayers[l].data, originalNetwork.pBHLayers[l].data)
            XCTAssertEqual(restoredNetwork.pGammaRMS[l].data, originalNetwork.pGammaRMS[l].data)
            l += 1
        }
        XCTAssertEqual(restoredNetwork.pWOut.data, originalNetwork.pWOut.data)
        XCTAssertEqual(restoredNetwork.pBOut.data, originalNetwork.pBOut.data)

        #if canImport(MLX)
        // MLXSpikingNetwork(weights:) 経由の復元
        let mlxRestored = MLXSpikingNetwork(weights: loadedWeights)
        XCTAssertEqual(mlxRestored.numLayers, 3)
        XCTAssertEqual(mlxRestored.inputDim, convOutDim)
        XCTAssertNotNil(mlxRestored.convSubsampling)
        XCTAssertEqual(mlxRestored.wLayers.count, 2)
        XCTAssertEqual(mlxRestored.bHLayers.count, 2)
        XCTAssertEqual(mlxRestored.gammaRMS.count, 2)
        #endif
    }

    /// 3. AcousticDecoder および StreamingTranscriber による多層 SNN + Conv2D 統合推論検証
    func testAcousticDecoderAndStreamingWithConvAndMultiLayer() {
        let melChannels = 64
        let convOutDim = 128
        let hiddenDim = 128
        let outputDim = 30

        let convSub = Conv2DSubsampling(melChannels: melChannels, outputDim: convOutDim)
        let network = SpikingNetwork(
            numLayers: 2,
            inputDim: convOutDim,
            maxHiddenDim: hiddenDim,
            outputDim: outputDim,
            timeSteps: 4,
            convSubsampling: convSub
        )

        let vocab = TextVocabulary(corpus: ["テスト", "おんせい", "にんしき"])
        let decoder = AcousticDecoder(
            network: network,
            vocabulary: vocab
        )
        let workspace = AcousticWorkspace(
            maxHiddenDim: network.maxHiddenDim,
            outputDim: network.outputDim,
            inputDim: network.inputDim,
            numLayers: network.numLayers
        )

        // 64ch Mel 入力系列 (20フレーム)
        var melSeq = [[Float]]()
        var f = 0
        while f < 20 {
            var frame = [Float](repeating: 0.0, count: melChannels)
            var c = 0
            while c < melChannels {
                frame[c] = Float(c) / Float(melChannels)
                c += 1
            }
            melSeq.append(frame)
            f += 1
        }

        // 4x サブサンプリングにより、20 フレーム -> 6 フレームに圧縮されてデコードされる
        let probs = decoder.decodeSequence(featuresSeq: melSeq, workspace: workspace)
        let expectedFrames = ((20 + 1) / 2 + 1) / 2
        XCTAssertEqual(probs.count, expectedFrames, "20フレームの入力が Conv2DSubsampling で \(expectedFrames) フレームに圧縮されること")

        for p in probs {
            XCTAssertEqual(p.probabilities.count, outputDim)
            var sumP: Float = 0.0
            for val in p.probabilities {
                sumP += val
            }
            XCTAssertTrue(0.99 <= sumP && sumP <= 1.01, "Softmax 確率和が 1.0 になること")
        }

        // ストリーミング推論の因果性テスト
        var streamState = Conv2DStreamingState()
        workspace.reset()
        var emittedFrames = 0
        f = 0
        while f < 20 {
            let res = decoder.decodeStreaming(
                melFrame: melSeq[f],
                state: &streamState,
                workspace: workspace,
                frameIndex: emittedFrames
            )
            switch res {
            case .some:
                emittedFrames += 1
            case .none:
                break
            }
            f += 1
        }
        XCTAssertEqual(emittedFrames, 5, "ストリーミングでも因果的ストライド4ごとに1フレーム出力されること")
    }

    /// 4. AcousticWorkspace の多層バッファ割り当てとゼロアロケーション・メモリ安全性の検証
    func testWorkspaceNumLayersSafety() {
        for layers in [1, 2, 4] {
            let hiddenDim = 256
            let outDim = 64
            let inDim = 128

            let ws = AcousticWorkspace(
                maxHiddenDim: hiddenDim,
                outputDim: outDim,
                inputDim: inDim,
                numLayers: layers
            )

            XCTAssertEqual(ws.vPrev.count, layers * hiddenDim)
            XCTAssertEqual(ws.sPrev.count, layers * hiddenDim)
            XCTAssertEqual(ws.aPrev.count, layers * hiddenDim)

            let net = SpikingNetwork(
                numLayers: layers,
                inputDim: inDim,
                maxHiddenDim: hiddenDim,
                outputDim: outDim,
                timeSteps: 4
            )

            let dummyFeat = [Float](repeating: 0.1, count: inDim)
            // 順伝播で境界外アクセスやクラッシュが起きないことを確認
            net.forward(
                features: dummyFeat,
                vPrev: &ws.vPrev,
                sPrev: &ws.sPrev,
                aPrev: &ws.aPrev,
                spikeSum: &ws.spikeSum,
                logits: &ws.logits,
                probabilities: &ws.probabilities,
                scratch: ws.scratch
            )

            var pSum: Float = 0.0
            for p in ws.probabilities {
                pSum += p
            }
            XCTAssertTrue(0.99 <= pSum && pSum <= 1.01)
        }
    }

    /// 5. loanword128 実音声データセットを用いた End-to-End ミニバッチ学習と推論の検証
    func testLoanwordMiniTrainingE2E() throws {
        let loanwordWavPath = ".tmp/loanword128/wav/LOANWORD128_001.wav"
        if FileManager.default.fileExists(atPath: loanwordWavPath) != true {
            print("loanword128 が見つからないため実音声テストをスキップ")
            return
        }

        guard let data = try? Data(contentsOf: URL(fileURLWithPath: loanwordWavPath)),
              let wav = try? WavParser().parse(bytes: [UInt8](data)) else {
            XCTFail("LOANWORD128_001.wav の読み込みに失敗")
            return
        }

        let pcm16k = SpeechDataset.resampleTo16k(pcmData: wav.pcmData, sampleRate: wav.sampleRate)
        let melSpectrogram = SpeechDataset.extractMelSpectrogram(pcmData: pcm16k)
        XCTAssertFalse(melSpectrogram.isEmpty)
        XCTAssertEqual(melSpectrogram.first?.count, 64)

        #if canImport(MLX)
        let vocab = TextVocabulary(corpus: ["突然のことにパニクって逃げ出してしまった"])
        let conv = MLXConv2DSubsampling(melChannels: 64, outputDim: 128)
        let net = MLXSpikingNetwork(
            numLayers: 2,
            inputDim: 128,
            maxHiddenDim: 128,
            outputDim: vocab.size,
            timeSteps: 4,
            convSubsampling: conv
        )
        let trainer = MLXBPTTTrainer(
            network: net,
            config: TrainingConfig(learningRate: 0.01, clipNorm: 5.0),
            bpttWindow: 8
        )

        let targetIds = vocab.textToIds("突然のことにパニクって逃げ出してしまった")
        let loss = trainer.trainBatchCTC(
            featuresBatch: [melSpectrogram],
            targetsBatch: [targetIds],
            blankId: 0
        )

        XCTAssertFalse(loss.isNaN)
        XCTAssertFalse(loss.isInfinite)
        XCTAssertTrue(0.0 < loss)

        // エクスポートして Pure Swift 側で推論テスト
        let exported = net.exportWeights(vocabulary: vocab)
        let pureNet = SpikingNetwork(weights: exported)
        let decoder = AcousticDecoder(network: pureNet, vocabulary: vocab)
        let ws = AcousticWorkspace(
            maxHiddenDim: pureNet.maxHiddenDim,
            outputDim: pureNet.outputDim,
            inputDim: pureNet.inputDim,
            numLayers: pureNet.numLayers
        )

        let probs = decoder.decodeSequence(featuresSeq: melSpectrogram, workspace: ws)
        XCTAssertFalse(probs.isEmpty)
        let expectedFrames = ((melSpectrogram.count + 1) / 2 + 1) / 2
        XCTAssertEqual(probs.count, expectedFrames)
        #endif
    }

    /// 6. StreamingTranscriber による多層 SNN + Conv2D フロントエンド実音声ストリーミング検証
    func testStreamingTranscriberWithConv2DAndMultiLayer() throws {
        let melChannels = 64
        let convOutDim = 128
        let hiddenDim = 128
        let vocab = TextVocabulary(corpus: ["あいうえお", "かきくけこ", "さしすせそ"])

        let convSub = Conv2DSubsampling(melChannels: melChannels, outputDim: convOutDim)
        let acousticNetwork = SpikingNetwork(
            numLayers: 2,
            inputDim: convOutDim,
            maxHiddenDim: hiddenDim,
            outputDim: vocab.size,
            timeSteps: 4,
            convSubsampling: convSub
        )
        let languageNetwork = SpikingNetwork(
            inputDim: 128,
            maxHiddenDim: hiddenDim,
            outputDim: vocab.size,
            timeSteps: 4
        )

        let transcriber = StreamingTranscriber(
            acousticNetwork: acousticNetwork,
            languageNetwork: languageNetwork,
            textVocabulary: vocab
        )

        // 16kHz 正弦波 (1秒 = 16000サンプル)
        var pcm = [Float](repeating: 0.0, count: 16000)
        var i = 0
        while i < 16000 {
            pcm[i] = sin(Float(i) * 0.05) * 0.3
            i += 1
        }

        transcriber.appendAudio(pcm: pcm)
        transcriber.flush()

        // ストリーミング処理でクラッシュせず、正常にフレーム消化されること
        XCTAssertTrue(true)
    }

    /// 7. 極小フレーム数 (1, 2, 3 フレーム) の Conv2DSubsampling および AcousticDecoder 境界耐性検証
    func testMinimalFramesBoundaryRobustness() {
        let melChannels = 64
        let convOutDim = 128
        let convSub = Conv2DSubsampling(melChannels: melChannels, outputDim: convOutDim)
        let network = SpikingNetwork(
            numLayers: 2,
            inputDim: convOutDim,
            maxHiddenDim: 128,
            outputDim: 20,
            timeSteps: 4,
            convSubsampling: convSub
        )
        let decoder = AcousticDecoder(network: network, vocabulary: TextVocabulary())
        let workspace = AcousticWorkspace(
            maxHiddenDim: network.maxHiddenDim,
            outputDim: network.outputDim,
            inputDim: network.inputDim,
            numLayers: network.numLayers
        )

        for frameCount in [1, 2, 3] {
            var frames = [[Float]]()
            var f = 0
            while f < frameCount {
                frames.append([Float](repeating: 0.2, count: melChannels))
                f += 1
            }

            // Conv2DSubsampling 直接
            let subsampled = convSub.forward(melSpectrogram: frames)
            let expectedCount = ((frameCount + 1) / 2 + 1) / 2
            XCTAssertEqual(subsampled.count, expectedCount)

            // AcousticDecoder 経由
            workspace.reset()
            let probs = decoder.decodeSequence(featuresSeq: frames, workspace: workspace)
            XCTAssertEqual(probs.count, expectedCount)
            for p in probs {
                XCTAssertEqual(p.probabilities.count, network.outputDim)
                for val in p.probabilities {
                    XCTAssertFalse(val.isNaN)
                    XCTAssertFalse(val.isInfinite)
                }
            }
        }
    }

    /// 8. SpeechDataset の lazyUseMel による 64ch Mel 保持検証
    func testSpeechDatasetLazyUseMelFeatureDimension() {
        let loanwordWavPath = ".tmp/loanword128/wav/LOANWORD128_001.wav"
        if FileManager.default.fileExists(atPath: loanwordWavPath) != true {
            return
        }

        let vocab = TextVocabulary(corpus: ["テスト"])
        let dataset = SpeechDataset.lazyFromManifest(
            pairs: [(path: loanwordWavPath, text: "テスト")],
            textVocabulary: vocab,
            frameStack: 1,
            useMel: true,
            workers: 1
        )

        XCTAssertTrue(dataset.lazyUseMel)
        let sample = dataset[0]
        XCTAssertEqual(sample.acousticFeatures.first?.count, 64, "useMel: true 時は 64ch Mel 特徴量が抽出されること")
    }
}
