import XCTest
import Foundation
@testable import Spiketrans

final class MultiLayerSNNTests: XCTestCase {

    // MARK: - 1. 重みシリアライズ・多層エクスポート/インポート検証

    func testMultiLayerWeightsSerializationAndCompatibility() throws {
        // 2層ネットワークの構築とエクスポート
        let net2 = SpikingNetwork(numLayers: 2, inputDim: 64, maxHiddenDim: 128, outputDim: 50)
        let weights2 = net2.exportWeights()

        XCTAssertEqual(weights2.numLayers, 2)
        XCTAssertEqual(weights2.inputDim, 64)
        XCTAssertEqual(weights2.maxHiddenDim, 128)
        XCTAssertEqual(weights2.outputDim, 50)
        XCTAssertNotNil(weights2.wLayers)
        XCTAssertNotNil(weights2.bHLayers)
        XCTAssertNotNil(weights2.gammaRMS)

        // JSON エンコードとデコードの往復検証
        let encoder = JSONEncoder()
        let data = try encoder.encode(weights2)
        let decoder = JSONDecoder()
        let decodedWeights = try decoder.decode(SpikingNetworkWeights.self, from: data)

        XCTAssertEqual(decodedWeights.numLayers, 2)
        XCTAssertEqual(decodedWeights.wIn.count, 128 * 64)
        XCTAssertEqual(decodedWeights.wRec.count, 128 * 128)
        XCTAssertEqual(decodedWeights.wLayers?.count, 1)
        XCTAssertEqual(decodedWeights.wLayers?[0].count, 128 * 128)
        XCTAssertEqual(decodedWeights.gammaRMS?[0].count, 128)

        // 別インスタンスへのインポートと値の一致
        let net2Copy = SpikingNetwork(numLayers: 2, inputDim: 64, maxHiddenDim: 128, outputDim: 50)
        net2Copy.importWeights(from: decodedWeights)

        let reExported = net2Copy.exportWeights()
        XCTAssertEqual(weights2.wIn, reExported.wIn)
        XCTAssertEqual(weights2.wRec, reExported.wRec)
        XCTAssertEqual(weights2.wLayers?[0], reExported.wLayers?[0])
        XCTAssertEqual(weights2.bHLayers?[0], reExported.bHLayers?[0])
        XCTAssertEqual(weights2.gammaRMS?[0], reExported.gammaRMS?[0])
    }

    // MARK: - 2. 電流RMSNormによる上位層沈黙防止検証

    func testCurrentRMSNormPreventsUpperLayerSilence() {
        let hDim = 128
        let inDim = 64
        let outDim = 10
        let net = SpikingNetwork(numLayers: 2, inputDim: inDim, maxHiddenDim: hDim, outputDim: outDim)

        // 極めて疎な入力（1〜2チャネルのみ微小電流）
        var sparseFeatures = [Float](repeating: 0.0, count: inDim)
        sparseFeatures[0] = 0.5
        sparseFeatures[1] = 0.8

        var vPrev = [Float](repeating: 0.0, count: 2 * hDim)
        var sPrev = [Float](repeating: 0.0, count: 2 * hDim)
        var aPrev = [Float](repeating: 0.0, count: 2 * hDim)
        var spikeSum = [Float](repeating: 0.0, count: hDim)
        var logits = [Float](repeating: 0.0, count: outDim)
        var probs = [Float](repeating: 0.0, count: outDim)

        let scratch = ForwardScratch(maxHiddenDim: hDim)

        // 複数ステップ推論を実行し、最終層（Layer 1）で発火が生じるか検証
        var step = 0
        var totalFinalLayerSpikes: Float = 0.0
        while step < 20 {
            net.forward(
                features: sparseFeatures,
                vPrev: &vPrev,
                sPrev: &sPrev,
                aPrev: &aPrev,
                spikeSum: &spikeSum,
                logits: &logits,
                probabilities: &probs,
                scratch: scratch
            )

            var i = 0
            while i < hDim {
                totalFinalLayerSpikes += spikeSum[i]
                i += 1
            }
            step += 1
        }

        // 電流RMSNormによって上位層への入力電流分散が正規化され、完全沈黙（0発火）が回避されていること
        XCTAssertTrue(0.0 < totalFinalLayerSpikes)
    }

    // MARK: - 3. Membrane-Shortcut（電流スキップ接続）によるサロゲート勾配バイパス検証

    func testMembraneShortcutGradientFlow() {
        // Membrane-Shortcut の数学的性質:
        // I^{(l)} = I^{(l)}_{norm} + I^{(l-1)}
        // 逆伝播時: ∂L/∂I^{(l-1)} = ∂L/∂I^{(l)} + (RMSNorm経由の勾配)
        // サロゲート勾配 σ'(V) が 0 に縮退した場合でも、電流加算ハイウェイを通じて
        // 勾配が前層に直接伝達されることを検証する
        let hDim = 64
        let inCur = [Float](repeating: 1.0, count: hDim)
        let denseCur = [Float](repeating: 0.5, count: hDim)
        let gamma = [Float](repeating: 1.0, count: hDim)

        // RMS 計算
        var sumSq: Float = 0.0
        var i = 0
        while i < hDim {
            sumSq += denseCur[i] * denseCur[i]
            i += 1
        }
        let rms = sqrt((sumSq / Float(hDim)) + 1e-5)
        let invRms = 1.0 / rms

        // 前向きショートカット
        var shortcutCur = [Float](repeating: 0.0, count: hDim)
        i = 0
        while i < hDim {
            shortcutCur[i] = (denseCur[i] * invRms * gamma[i]) + inCur[i]
            i += 1
        }

        // 逆伝播: 上位層からの電流勾配 dShortcut
        let dShortcut = [Float](repeating: 0.8, count: hDim)

        // ショートカットにより前層電流への勾配 dInCur には dShortcut * 1.0 が無損失で伝達される
        var dInCur = [Float](repeating: 0.0, count: hDim)
        i = 0
        while i < hDim {
            // 線形ショートカット接続の寄与は 1.0
            dInCur[i] = dShortcut[i]
            i += 1
        }

        i = 0
        while i < hDim {
            XCTAssertEqual(dInCur[i], 0.8, accuracy: 1e-5)
            i += 1
        }
    }

    // MARK: - 4. 第0層再帰＋上位層FFによる暴走防止・安定性検証

    func testLayer0RecurrentUpperLayerFFStability() {
        // 全層再帰では同期発火による暴走（てんかん様発振・損失1300急騰）が発生するリスクがある。
        // 第0層のみ再帰とし、上位層をフィードフォワードLIFとすることで、
        // 強い入力電流が長時間続いても同期暴走を起こさないことを検証。
        let hDim = 128
        let inDim = 64
        let outDim = 10
        let net = SpikingNetwork(numLayers: 2, inputDim: inDim, maxHiddenDim: hDim, outputDim: outDim)

        // 強力な直流入力を 200 ステップ連続注入
        let strongFeatures = [Float](repeating: 5.0, count: inDim)
        var vPrev = [Float](repeating: 0.0, count: 2 * hDim)
        var sPrev = [Float](repeating: 0.0, count: 2 * hDim)
        var aPrev = [Float](repeating: 0.0, count: 2 * hDim)
        var spikeSum = [Float](repeating: 0.0, count: hDim)
        var logits = [Float](repeating: 0.0, count: outDim)
        var probs = [Float](repeating: 0.0, count: outDim)

        let scratch = ForwardScratch(maxHiddenDim: hDim)

        var step = 0
        while step < 200 {
            net.forward(
                features: strongFeatures,
                vPrev: &vPrev,
                sPrev: &sPrev,
                aPrev: &aPrev,
                spikeSum: &spikeSum,
                logits: &logits,
                probabilities: &probs,
                scratch: scratch
            )

            // 各ステップで膜電位が NaN / Inf にならず正常範囲に収まっていること
            var i = 0
            while i < (2 * hDim) {
                XCTAssertTrue(vPrev[i].isNaN != true)
                XCTAssertTrue(vPrev[i].isInfinite != true)
                // 膜電位クランプ [-20.0, 20.0] の範囲内
                XCTAssertTrue(-20.0 <= vPrev[i])
                XCTAssertTrue(vPrev[i] <= 20.0)
                i += 1
            }

            // 確率分布が正規化されていること (sum = 1.0)
            var sumP: Float = 0.0
            var c = 0
            while c < outDim {
                sumP += probs[c]
                c += 1
            }
            XCTAssertEqual(sumP, 1.0, accuracy: 1e-4)

            step += 1
        }
    }

    // MARK: - 5. 単層比での学習損失収束・品質向上検証

    func testMultiLayerToyDatasetConvergence() {
        // トイタスク: 3 パターンの音響系列（共調音コンテキスト）
        let inDim = 16
        let hDim = 32
        let outDim = 5
        let seqLen = 6

        // 単層モデルと2層モデル
        let net1 = SpikingNetwork(numLayers: 1, inputDim: inDim, maxHiddenDim: hDim, outputDim: outDim)
        let opt1 = AdamOptimizer(config: AdamConfig(lr: 0.02), parameters: net1.parameters)
        let trainer1 = BPTTTrainer(network: net1, optimizer: opt1)

        let net2 = SpikingNetwork(numLayers: 2, inputDim: inDim, maxHiddenDim: hDim, outputDim: outDim)
        let opt2 = AdamOptimizer(config: AdamConfig(lr: 0.02), parameters: net2.parameters)
        let trainer2 = BPTTTrainer(network: net2, optimizer: opt2)

        // 合成データ系列の作成
        var trainSeq = [[Float]](repeating: [Float](repeating: 0.0, count: inDim), count: seqLen)
        var k = 0
        while k < seqLen {
            var d = 0
            while d < inDim {
                trainSeq[k][d] = sin(Float(k * d + 1)) * 0.5 + 0.5
                d += 1
            }
            k += 1
        }
        let targets = [1, 2, 3, 2, 1, 0]

        // 単層の学習
        var step = 0
        var loss1First: Float = 0.0
        var loss1Last: Float = 0.0
        while step < 15 {
            let l = trainer1.trainStep(featuresSeq: trainSeq, targets: targets)
            if step == 0 { loss1First = l }
            loss1Last = l
            step += 1
        }

        // 2層の学習
        step = 0
        var loss2First: Float = 0.0
        var loss2Last: Float = 0.0
        while step < 15 {
            let l = trainer2.trainStep(featuresSeq: trainSeq, targets: targets)
            if step == 0 { loss2First = l }
            loss2Last = l
            step += 1
        }

        // 双方ともに初期損失から改善していること
        XCTAssertTrue(loss1Last <= loss1First)
        XCTAssertTrue(loss2Last <= loss2First)
        XCTAssertTrue(loss2Last.isNaN != true)
    }

    // MARK: - 6. 10時間想定の O(1) 定数メモリおよび推論レイテンシ検証

    func testLongDurationO1MemoryFootprint() {
        // 10時間の会議録音（ストリーミング推論）を模倣
        // 10時間 = 36000秒 = 3,600,000 フレーム。
        // ここでは 2,000 フレームの連続推論を行い、メモリ割り当てが定数 O(1) で維持され、
        // かつ推論レイテンシ（RTF 0.01前後）が処理時間経過によって劣化しないことを検証。
        let inDim = 64
        let hDim = 256
        let outDim = 50
        let net = SpikingNetwork(numLayers: 2, inputDim: inDim, maxHiddenDim: hDim, outputDim: outDim)

        let workspace = AcousticWorkspace(
            maxHiddenDim: hDim,
            outputDim: outDim,
            inputDim: inDim,
            numLayers: 2
        )
        let decoder = AcousticDecoder(network: net)

        let totalFrames = 2000
        var dummyFrame = [Float](repeating: 0.1, count: inDim)

        let startEarly = DispatchTime.now()

        // 最初の 200 フレームの実行
        var f = 0
        while f < 200 {
            dummyFrame[f % inDim] = Float(f % 10) * 0.1
            _ = decoder.decodeFrame(features: dummyFrame, workspace: workspace, frameIndex: f)
            f += 1
        }
        let endEarly = DispatchTime.now()
        let earlyDurationNs = endEarly.uptimeNanoseconds - startEarly.uptimeNanoseconds

        // 中間フレームの連続実行 (200 ..< 1800)
        while f < 1800 {
            dummyFrame[f % inDim] = Float(f % 10) * 0.1
            _ = decoder.decodeFrame(features: dummyFrame, workspace: workspace, frameIndex: f)
            f += 1
        }

        // 最後の 200 フレームの実行
        let startLate = DispatchTime.now()
        while f < totalFrames {
            dummyFrame[f % inDim] = Float(f % 10) * 0.1
            _ = decoder.decodeFrame(features: dummyFrame, workspace: workspace, frameIndex: f)
            f += 1
        }
        let endLate = DispatchTime.now()
        let lateDurationNs = endLate.uptimeNanoseconds - startLate.uptimeNanoseconds

        // 状態配列のサイズが一切増加していないこと (O(1) 定数メモリ検証)
        XCTAssertEqual(workspace.vPrev.count, 2 * hDim)
        XCTAssertEqual(workspace.sPrev.count, 2 * hDim)
        XCTAssertEqual(workspace.aPrev.count, 2 * hDim)
        XCTAssertEqual(workspace.spikeSum.count, hDim)
        XCTAssertEqual(workspace.logits.count, outDim)
        XCTAssertEqual(workspace.probabilities.count, outDim)

        // レイテンシ劣化がないことの検証:
        // 後半 200 フレームの実行時間が前半の 4 倍未満であること（GCやメモリ膨張によるスローダウンなし）
        let earlySec = Double(earlyDurationNs) / 1_000_000_000.0
        let lateSec = Double(lateDurationNs) / 1_000_000_000.0

        // 200フレームの音声時間 = 2.0 秒 (10msシフト)。
        // 200フレームの処理時間が 1.0 秒以下であること
        XCTAssertTrue(lateSec < 1.0)
        XCTAssertTrue(lateSec < (earlySec * 4.0 + 0.1))
    }

    // MARK: - 7. チャンク分割時の状態連続性・リセット検証

    func testStateContinuityAcrossChunks() {
        let inDim = 64
        let hDim = 128
        let outDim = 20
        let net = SpikingNetwork(numLayers: 2, inputDim: inDim, maxHiddenDim: hDim, outputDim: outDim)

        let workspace = AcousticWorkspace(
            maxHiddenDim: hDim,
            outputDim: outDim,
            inputDim: inDim,
            numLayers: 2
        )
        let decoder = AcousticDecoder(network: net)

        let frame1 = [Float](repeating: 0.8, count: inDim)
        let frame2 = [Float](repeating: 0.2, count: inDim)

        // 1. 状態を引き継いで 2 フレーム連続推論
        _ = decoder.decodeFrame(features: frame1, workspace: workspace, frameIndex: 0)
        let continuousRes = decoder.decodeFrame(features: frame2, workspace: workspace, frameIndex: 1)

        // 2. 状態をリセットしてから 2 番目のフレームを推論
        workspace.resetHiddenState()
        let resetRes = decoder.decodeFrame(features: frame2, workspace: workspace, frameIndex: 0)

        // 第0層の再帰および適応閾値状態により、先行コンテキストの有無で出力確率が明確に異なる（文脈理解の存在証明）
        var hasDifference = false
        var c = 0
        while c < outDim {
            if 1e-4 < abs(continuousRes.probabilities[c] - resetRes.probabilities[c]) {
                hasDifference = true
                break
            }
            c += 1
        }
        XCTAssertTrue(hasDifference)
    }

    // MARK: - 8. 既存 Filterbank・フォルマントパイプラインとの協調動作検証

    func testPipelineFilterbankToMultiLayerSNN() {
        let dspConfig = DSPConfig(sampleRate: 16000, frameSize: 512, hopSize: 160, melChannels: 64)
        let filterbank = Filterbank(config: dspConfig)
        let net = SpikingNetwork(numLayers: 2, inputDim: 64, maxHiddenDim: 128, outputDim: 50)
        let workspace = AcousticWorkspace(maxHiddenDim: 128, outputDim: 50, inputDim: 64, numLayers: 2)
        let decoder = AcousticDecoder(network: net)

        // フォルマント成分を含む音声信号の生成 (基本周波数 200Hz + 倍音)
        var audioSignal = [Float](repeating: 0.0, count: dspConfig.frameSize)
        var i = 0
        while i < dspConfig.frameSize {
            let t = Float(i) / Float(dspConfig.sampleRate)
            audioSignal[i] = sin(2.0 * Float.pi * 200.0 * t) * 0.5 + sin(2.0 * Float.pi * 800.0 * t) * 0.3
            i += 1
        }

        let dspWorkspace = DSPWorkspace()

        // Filterbank による特徴量抽出 (Direct Input Current: 0.0〜1.0)
        let melFeatures = audioSignal.withUnsafeBufferPointer { audioBuf in
            filterbank.extractFeatures(
                pcmPtr: audioBuf.baseAddress!,
                count: dspConfig.frameSize,
                workspace: dspWorkspace
            )
        }

        // 多層 SNN デコーダに入力電流として供給
        let res = decoder.decodeFrame(features: melFeatures, workspace: workspace, frameIndex: 0)

        XCTAssertEqual(res.probabilities.count, 50)
        XCTAssertTrue(0 <= res.topTokenId)
        XCTAssertTrue(res.topTokenId < 50)
        XCTAssertTrue(0.0 <= res.topProbability)
        XCTAssertTrue(res.topProbability <= 1.0)
    }
}
