import XCTest
import MLX
@testable import Spiketrans

/// compile() 済みの学習ステップが学習率の変更に追従することを確かめる。
/// 学習率が Float のままだとトレース時の値が焼き込まれ、lr=0 にしても重みが動く
final class MLXCompiledLearningRateTests: XCTestCase {
    private func makeBatch(inputDim: Int, frames: Int) -> ([[[Float]]], [[Int]]) {
        var seq: [[Float]] = []
        var t = 0
        while t < frames {
            var frame = [Float](repeating: 0.0, count: inputDim)
            var d = 0
            while d < inputDim {
                frame[d] = 0.5 + 0.5 * sin(Float(t * 7 + d * 13) * 0.37)
                d += 1
            }
            seq.append(frame)
            t += 1
        }
        return ([seq, seq], [[1, 2, 3, 2], [4, 5, 6]])
    }

    private func maxAbsDiff(_ a: [Float], _ b: [Float]) -> Float {
        var m: Float = 0.0
        var i = 0
        while i < min(a.count, b.count) {
            m = max(m, abs(a[i] - b[i]))
            i += 1
        }
        return m
    }

    /// lr=0 のステップで重みが動かず、lr を戻すと再び動く
    private func assertFollowsLearningRate(compiled: Bool) {
        let inputDim = 16
        let (feats, targets) = makeBatch(inputDim: inputDim, frames: 32)
        let net = MLXSpikingNetwork(numLayers: 2, inputDim: inputDim, maxHiddenDim: 64, outputDim: 8)
        let trainer = MLXBPTTTrainer(network: net, config: TrainingConfig(learningRate: 0.01), bpttWindow: 4)
        trainer.trainBatchCTC(featuresBatch: feats, targetsBatch: targets, compiled: compiled)
        trainer.trainBatchCTC(featuresBatch: feats, targetsBatch: targets, compiled: compiled)

        trainer.updateLearningRate(0.0)
        let frozen = net.exportWeights()
        trainer.trainBatchCTC(featuresBatch: feats, targetsBatch: targets, compiled: compiled)
        let afterZero = net.exportWeights()
        XCTAssertEqual(maxAbsDiff(frozen.wIn, afterZero.wIn), 0.0, "compiled=\(compiled): 学習率 0 で重みが動いた")

        trainer.updateLearningRate(0.01)
        trainer.trainBatchCTC(featuresBatch: feats, targetsBatch: targets, compiled: compiled)
        let afterRestore = net.exportWeights()
        XCTAssertLessThan(0.0, maxAbsDiff(afterZero.wIn, afterRestore.wIn), "compiled=\(compiled): 学習率を戻しても重みが動かない")
    }

    func testEagerStepFollowsLearningRate() {
        assertFollowsLearningRate(compiled: false)
    }

    func testCompiledStepFollowsLearningRate() {
        assertFollowsLearningRate(compiled: true)
    }

    /// compile の上限を超える長い系列は eager に落ちる (コンパイル回数が増えない)
    func testLongSequenceFallsBackToEager() {
        let inputDim = 16
        let (feats, targets) = makeBatch(inputDim: inputDim, frames: compiledMaxFrames + 1)
        let net = MLXSpikingNetwork(numLayers: 1, inputDim: inputDim, maxHiddenDim: 32, outputDim: 8)
        let trainer = MLXBPTTTrainer(network: net, config: TrainingConfig(learningRate: 0.01), bpttWindow: 4)
        let loss = trainer.trainBatchCTC(featuresBatch: feats, targetsBatch: targets, compiled: true)
        XCTAssertFalse(loss.isNaN)
        XCTAssertEqual(trainer.ctcCompileCount, 0)
    }

    /// 巻き戻しで重みが保存時の値に戻り、Adam のモーメントが 0 になり、そのあとも学習が続く
    func testRollbackRestoresWeightsAndResetsOptimizer() {
        let inputDim = 16
        let (feats, targets) = makeBatch(inputDim: inputDim, frames: 32)
        let net = MLXSpikingNetwork(numLayers: 2, inputDim: inputDim, maxHiddenDim: 64, outputDim: 8)
        let trainer = MLXBPTTTrainer(network: net, config: TrainingConfig(learningRate: 0.01), bpttWindow: 4)
        trainer.trainBatchCTC(featuresBatch: feats, targetsBatch: targets, compiled: true)
        let snapshot = net.exportWeights()
        trainer.trainBatchCTC(featuresBatch: feats, targetsBatch: targets, compiled: true)
        trainer.trainBatchCTC(featuresBatch: feats, targetsBatch: targets, compiled: true)
        XCTAssertLessThan(0.0, maxAbsDiff(snapshot.wIn, net.exportWeights().wIn), "学習で重みが動いていない")

        trainer.rollback(to: snapshot)
        XCTAssertEqual(maxAbsDiff(snapshot.wIn, net.exportWeights().wIn), 0.0, "巻き戻しで重みが戻らない")
        XCTAssertEqual(maxAbsDiff(snapshot.wOut, net.exportWeights().wOut), 0.0)
        for state in trainer.optimizer.innerState() {
            eval(state)
            XCTAssertEqual(abs(state).max().item(Float.self), 0.0, "Adam のモーメントが 0 に戻っていない")
        }

        // 巻き戻し後も学習が続く (importWeights がプロパティ代入だった頃は Module のキャッシュが
        // 古い配列を指したままになり、ここで重みが動かなかった)
        trainer.trainBatchCTC(featuresBatch: feats, targetsBatch: targets, compiled: false)
        XCTAssertLessThan(0.0, maxAbsDiff(snapshot.wIn, net.exportWeights().wIn), "巻き戻し後に eager の学習が止まった")
        let afterEager = net.exportWeights()
        trainer.trainBatchCTC(featuresBatch: feats, targetsBatch: targets, compiled: true)
        XCTAssertLessThan(0.0, maxAbsDiff(afterEager.wIn, net.exportWeights().wIn), "巻き戻し後に compile 済みステップの学習が止まった")
    }

    /// importWeights のあとに学習が続く (追加学習の前提)
    func testTrainingContinuesAfterImportWeights() {
        let inputDim = 16
        let (feats, targets) = makeBatch(inputDim: inputDim, frames: 32)
        let net = MLXSpikingNetwork(numLayers: 2, inputDim: inputDim, maxHiddenDim: 64, outputDim: 8)
        let trainer = MLXBPTTTrainer(network: net, config: TrainingConfig(learningRate: 0.01), bpttWindow: 4)
        trainer.trainBatchCTC(featuresBatch: feats, targetsBatch: targets, compiled: false)
        let snapshot = net.exportWeights()
        net.importWeights(from: snapshot)
        XCTAssertEqual(maxAbsDiff(snapshot.wIn, net.exportWeights().wIn), 0.0)
        trainer.trainBatchCTC(featuresBatch: feats, targetsBatch: targets, compiled: false)
        XCTAssertLessThan(0.0, maxAbsDiff(snapshot.wIn, net.exportWeights().wIn), "importWeights 後に重みが動かない")
    }

    /// 長い系列のチャンク分割の勾配は、一括で微分した勾配と一致する
    func testChunkedLongSequenceMatchesOneShot() {
        let inputDim = 16
        let frames = compiledMaxFrames + 64
        let (feats, targets) = makeBatch(inputDim: inputDim, frames: frames)
        let netA = MLXSpikingNetwork(numLayers: 3, inputDim: inputDim, maxHiddenDim: 64, outputDim: 8)
        let netB = MLXSpikingNetwork(numLayers: 3, inputDim: inputDim, maxHiddenDim: 64, outputDim: 8)
        netB.importWeights(from: netA.exportWeights())
        let trainerA = MLXBPTTTrainer(network: netA, config: TrainingConfig(learningRate: 0.01), bpttWindow: 4)
        let trainerB = MLXBPTTTrainer(network: netB, config: TrainingConfig(learningRate: 0.01), bpttWindow: 4)
        trainerA.chunkLongSequences = false
        let lossA = trainerA.trainBatchCTC(featuresBatch: feats, targetsBatch: targets)
        let lossB = trainerB.trainBatchCTC(featuresBatch: feats, targetsBatch: targets)
        XCTAssertEqual(lossA, lossB, accuracy: 1e-3 * max(1.0, abs(lossA)), "チャンク分割の損失が一括と違う")
        let wA = netA.exportWeights()
        let wB = netB.exportWeights()
        XCTAssertLessThan(maxAbsDiff(wA.wIn, wB.wIn), 1e-4, "wIn の更新がチャンク分割と一括で違う")
        XCTAssertLessThan(maxAbsDiff(wA.wOut, wB.wOut), 1e-4, "wOut の更新がチャンク分割と一括で違う")
        XCTAssertLessThan(maxAbsDiff(wA.wRec, wB.wRec), 1e-4, "wRec の更新がチャンク分割と一括で違う")
    }

    /// compile したチャンク分割 (端の短いチャンクは 0 埋め) が、compile なしのチャンク分割と同じ更新をする。
    /// 2 ステップ回して 2 回目がキャッシュ済みの compile 関数を使う経路も通す
    func testCompiledChunksMatchEagerChunks() {
        let inputDim = 16
        let frames = compiledMaxFrames + 40
        let (feats, targets) = makeBatch(inputDim: inputDim, frames: frames)
        let netA = MLXSpikingNetwork(numLayers: 3, inputDim: inputDim, maxHiddenDim: 64, outputDim: 8)
        let netB = MLXSpikingNetwork(numLayers: 3, inputDim: inputDim, maxHiddenDim: 64, outputDim: 8)
        netB.importWeights(from: netA.exportWeights())
        let trainerA = MLXBPTTTrainer(network: netA, config: TrainingConfig(learningRate: 0.01), bpttWindow: 4)
        let trainerB = MLXBPTTTrainer(network: netB, config: TrainingConfig(learningRate: 0.01), bpttWindow: 4)
        var step = 0
        while step < 2 {
            let lossA = trainerA.trainBatchCTC(featuresBatch: feats, targetsBatch: targets, compiled: false)
            let lossB = trainerB.trainBatchCTC(featuresBatch: feats, targetsBatch: targets, compiled: true)
            XCTAssertEqual(lossA, lossB, accuracy: 1e-3 * max(1.0, abs(lossA)), "compile 済みチャンクの損失が違う (step \(step))")
            step += 1
        }
        let wA = netA.exportWeights()
        let wB = netB.exportWeights()
        XCTAssertLessThan(maxAbsDiff(wA.wIn, wB.wIn), 1e-4, "wIn の更新が compile 済みチャンクで違う")
        XCTAssertLessThan(maxAbsDiff(wA.wOut, wB.wOut), 1e-4, "wOut の更新が compile 済みチャンクで違う")
        XCTAssertLessThan(maxAbsDiff(wA.wRec, wB.wRec), 1e-4, "wRec の更新が compile 済みチャンクで違う")
    }

    private func makeVoiceTargets(frames: Int) -> [[Int]] {
        // 前半は配信者 (1)、真ん中に bot (2)、末尾はラベルなし (-1)。2 件目は「なし (0)」だけ
        var a = [Int](repeating: 1, count: frames)
        var t = frames / 3
        while t < frames / 2 {
            a[t] = 2
            t += 1
        }
        t = frames - 8
        while t < frames {
            a[t] = -1
            t += 1
        }
        return [a, [Int](repeating: 0, count: frames)]
    }

    func testVoiceLossMatchesManualCrossEntropy() {
        let logits = MLXArray([1.0, 0.0, -1.0, 0.5, 0.2, 0.2, 0.2, 0.2] as [Float], [1, 2, 4])
        let targets = MLXArray([Int32(2), Int32(-1)], [1, 2])
        let loss = MLXBPTTTrainer.voiceLoss(logits: logits, targets: targets)
        eval(loss)
        // 有効なのは 1 フレーム目だけ: -log softmax([1, 0, -1, 0.5])[2]
        let z: [Float] = [1.0, 0.0, -1.0, 0.5]
        let m = z.max()!
        let lse = m + log(z.map { exp($0 - m) }.reduce(0, +))
        XCTAssertEqual(loss.item(Float.self), lse - z[2], accuracy: 1e-5)
        let grad = MLXBPTTTrainer.voiceLossGradient(logits: logits, targets: targets)
        eval(grad)
        let g = grad.asArray(Float.self)
        // ラベルなしフレームの勾配は 0、有効フレームは softmax - onehot
        XCTAssertEqual(g[4], 0.0, accuracy: 1e-7)
        XCTAssertEqual(g[2], exp(z[2] - lse) - 1.0, accuracy: 1e-5)
    }

    func testVoiceHeadWeightsRoundTrip() throws {
        let net = MLXSpikingNetwork(numLayers: 3, inputDim: 16, maxHiddenDim: 32, outputDim: 8, voiceHead: true)
        XCTAssertEqual(net.voiceLayer, 1)
        let w = net.exportWeights()
        XCTAssertEqual(w.hasVoiceHead, true)
        XCTAssertEqual(w.wVoice?.count, SpikingNetworkWeights.voiceClasses * 32)
        let data = try JSONEncoder().encode(w)
        let decoded = try JSONDecoder().decode(SpikingNetworkWeights.self, from: data)
        XCTAssertEqual(decoded, w)
        XCTAssertEqual(MLXSpikingNetwork(weights: decoded).exportWeights(), w)
        // CPU 推論側は補助ヘッドを持たずに読める
        XCTAssertEqual(SpikingNetwork(weights: decoded).numLayers, 3)
        XCTAssertEqual(MLXSpikingNetwork(numLayers: 3, inputDim: 16, maxHiddenDim: 32, outputDim: 8).exportWeights().hasVoiceHead, false)
    }

    /// 補助ヘッド付きで、compile 済みチャンク (解析的な補助勾配) と eager (自動微分) の更新が一致する
    func testVoiceHeadChunkedMatchesEager() {
        let inputDim = 16
        let frames = compiledMaxFrames + 40
        let (feats, targets) = makeBatch(inputDim: inputDim, frames: frames)
        let voice = makeVoiceTargets(frames: frames)
        let netA = MLXSpikingNetwork(numLayers: 3, inputDim: inputDim, maxHiddenDim: 64, outputDim: 8, voiceHead: true)
        let netB = MLXSpikingNetwork(numLayers: 3, inputDim: inputDim, maxHiddenDim: 64, outputDim: 8, voiceHead: true)
        netB.importWeights(from: netA.exportWeights())
        let trainerA = MLXBPTTTrainer(network: netA, config: TrainingConfig(learningRate: 0.01), bpttWindow: 4)
        let trainerB = MLXBPTTTrainer(network: netB, config: TrainingConfig(learningRate: 0.01), bpttWindow: 4)
        trainerA.chunkLongSequences = false
        var step = 0
        while step < 2 {
            let lossA = trainerA.trainBatchCTC(featuresBatch: feats, targetsBatch: targets, compiled: false, voiceTargetsBatch: voice)
            let lossB = trainerB.trainBatchCTC(featuresBatch: feats, targetsBatch: targets, compiled: true, voiceTargetsBatch: voice)
            XCTAssertEqual(lossA, lossB, accuracy: 1e-3 * max(1.0, abs(lossA)), "補助ヘッド付きの損失が違う (step \(step))")
            XCTAssertEqual(trainerA.lastVoiceLoss, trainerB.lastVoiceLoss, accuracy: 1e-3, "補助損失が違う (step \(step))")
            XCTAssertLessThan(0.0, trainerA.lastVoiceLoss)
            step += 1
        }
        let wA = netA.exportWeights()
        let wB = netB.exportWeights()
        XCTAssertLessThan(maxAbsDiff(wA.wIn, wB.wIn), 1e-4, "wIn の更新が違う")
        XCTAssertLessThan(maxAbsDiff(wA.wVoice ?? [], wB.wVoice ?? []), 1e-4, "wVoice の更新が違う")
        XCTAssertLessThan(maxAbsDiff(wA.wLayers[0], wB.wLayers[0]), 1e-4, "wLayers の更新が違う")
    }

    /// 短い系列 (compile 済みステップ) でも補助損失が下がる
    func testVoiceHeadCompiledStepLearns() {
        let inputDim = 16
        let (feats, targets) = makeBatch(inputDim: inputDim, frames: 32)
        let voice = makeVoiceTargets(frames: 32)
        let net = MLXSpikingNetwork(numLayers: 2, inputDim: inputDim, maxHiddenDim: 64, outputDim: 8, voiceHead: true)
        let trainer = MLXBPTTTrainer(network: net, config: TrainingConfig(learningRate: 0.01), bpttWindow: 4)
        trainer.trainBatchCTC(featuresBatch: feats, targetsBatch: targets, compiled: true, voiceTargetsBatch: voice)
        let first = trainer.lastVoiceLoss
        var i = 0
        while i < 30 {
            trainer.trainBatchCTC(featuresBatch: feats, targetsBatch: targets, compiled: true, voiceTargetsBatch: voice)
            i += 1
        }
        XCTAssertLessThan(trainer.lastVoiceLoss, first * 0.8, "補助損失が下がらない (\(first) → \(trainer.lastVoiceLoss))")
    }

    /// 事前学習 (APC) のステップで予測損失が下がる。短い系列が混ざっても (0 埋め・マスク) 有限のまま
    func testAPCStepLearns() {
        let inputDim = 16
        let (long, _) = makeBatch(inputDim: inputDim, frames: 64)
        let short = Array(long[0][0..<40])
        let batch = [long[0], short]
        let net = MLXSpikingNetwork(numLayers: 3, inputDim: inputDim, maxHiddenDim: 64, outputDim: 1, predictionHead: true)
        let trainer = MLXBPTTTrainer(network: net, config: TrainingConfig(learningRate: 0.01), bpttWindow: 4)
        let first = trainer.trainBatchAPC(featuresBatch: batch)
        var last = first
        var i = 0
        while i < 40 {
            last = trainer.trainBatchAPC(featuresBatch: batch)
            XCTAssertTrue(last.isFinite)
            i += 1
        }
        XCTAssertLessThan(last, first * 0.7, "予測損失が下がらない (\(first) → \(last))")
    }

    /// 事前学習の重みから CTC を始めると、隠れ層は同じで、読み出しは新しいものに替わり予測ヘッドは無くなる
    func testStartingCTCFromPretrainedWeights() throws {
        let pre = MLXSpikingNetwork(numLayers: 3, inputDim: 16, maxHiddenDim: 32, outputDim: 1, predictionHead: true)
        let pw = pre.exportWeights()
        XCTAssertEqual(pw.hasPredictionHead, true)
        let decoded = try JSONDecoder().decode(SpikingNetworkWeights.self, from: try JSONEncoder().encode(pw))
        XCTAssertEqual(decoded, pw)
        XCTAssertEqual(MLXSpikingNetwork(weights: decoded).exportWeights(), pw)

        let fresh = MLXSpikingNetwork(numLayers: 3, inputDim: 16, maxHiddenDim: 32, outputDim: 8).exportWeights()
        let start = pw.startingCTC(readout: fresh)
        XCTAssertEqual(start.hasPredictionHead, false)
        XCTAssertEqual(start.outputDim, 8)
        XCTAssertEqual(start.wIn, pw.wIn)
        XCTAssertEqual(start.wLayers, pw.wLayers)
        XCTAssertEqual(start.wOut, fresh.wOut)
        let net = MLXSpikingNetwork(weights: start)
        XCTAssertEqual(net.predictionHead.isEmpty, true)
        XCTAssertEqual(net.exportWeights().wIn, pw.wIn)
    }

    /// 予測先を複数にすると、ヘッドの出力が予測先の数ぶんになり、重みの往復で保たれ、損失も下がる
    func testAPCMultipleShifts() throws {
        let inputDim = 16
        let (long, _) = makeBatch(inputDim: inputDim, frames: 64)
        let net = MLXSpikingNetwork(numLayers: 2, inputDim: inputDim, maxHiddenDim: 64, outputDim: 1,
                                    predictionHead: true, predictionTargets: 3)
        XCTAssertEqual(net.predictionHead[0].shape[1], 3 * inputDim)
        let trainer = MLXBPTTTrainer(network: net, config: TrainingConfig(learningRate: 0.01), bpttWindow: 4)
        let first = trainer.trainBatchAPC(featuresBatch: long, shifts: [3, 6, 10])
        var last = first
        var i = 0
        while i < 40 {
            last = trainer.trainBatchAPC(featuresBatch: long, shifts: [3, 6, 10])
            i += 1
        }
        XCTAssertLessThan(last, first * 0.8)
        let w = net.exportWeights()
        XCTAssertEqual(w.predictionTargets, 3)
        let decoded = try JSONDecoder().decode(SpikingNetworkWeights.self, from: try JSONEncoder().encode(w))
        XCTAssertEqual(MLXSpikingNetwork(weights: decoded).exportWeights(), w)
    }

    /// k-means が離れた 2 つの塊を別のクラスに分け、assign が正しい中心を返し、保存して読み戻せる
    func testClusterCodebookSeparatesBlobs() throws {
        var frames: [[Float]] = []
        var i = 0
        while i < 200 {
            let base: Float = (i % 2 == 0) ? -3.0 : 3.0
            frames.append([base + Float(i % 7) * 0.01, base * 0.5, Float(i % 5) * 0.02])
            i += 1
        }
        let book = ClusterCodebook.fit(frames: frames, k: 2, iterations: 10, seed: 1)
        XCTAssertEqual(book.count, 2)
        let ids = ClusterCodebook.assign(features: MLXArray(frames.flatMap { $0 }, [200, 3]), codebook: book.arrays())
        eval(ids)
        let a = ids.asArray(Int32.self)
        var k = 0
        while k < 200 {
            XCTAssertEqual(a[k], a[k % 2], "フレーム \(k) のクラスが塊と一致しない")
            k += 1
        }
        XCTAssertNotEqual(a[0], a[1])
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("codebook_test.json")
        try book.save(to: url)
        let back = try ClusterCodebook.load(from: url)
        XCTAssertEqual(back.centroids, book.centroids)
    }

    /// 区間を隠してクラスを当てる事前学習のステップで、損失が有限のまま下がる
    func testMaskedClusterStepLearns() {
        let inputDim = 16
        let (feats, _) = makeBatch(inputDim: inputDim, frames: 64)
        var frames: [[Float]] = []
        for seq in feats {
            frames.append(contentsOf: seq)
        }
        let book = ClusterCodebook.fit(frames: frames, k: 8, iterations: 10, seed: 2)
        let codebook = book.arrays()
        let net = MLXSpikingNetwork(numLayers: 2, inputDim: inputDim, maxHiddenDim: 64, outputDim: 1,
                                    predictionHead: true, predictionClasses: 8)
        let trainer = MLXBPTTTrainer(network: net, config: TrainingConfig(learningRate: 0.01), bpttWindow: 4)
        // 隠す位置は毎回変わるので、損失は数ステップの平均で比べる
        func average(_ steps: Int) -> Float {
            var sum: Float = 0.0
            var i = 0
            while i < steps {
                let loss = trainer.trainBatchMaskedCluster(featuresBatch: feats, codebook: codebook, startProb: 0.2, spanFrames: 2)
                XCTAssertTrue(loss.isFinite)
                sum += loss
                i += 1
            }
            return sum / Float(steps)
        }
        let first = average(5)
        var i = 0
        while i < 150 {
            trainer.trainBatchMaskedCluster(featuresBatch: feats, codebook: codebook, startProb: 0.2, spanFrames: 2)
            i += 1
        }
        let last = average(5)
        XCTAssertLessThan(last, first * 0.8, "クラス予測の損失が下がらない (\(first) → \(last))")
        XCTAssertEqual(net.exportWeights().bPred?.count, 8)
    }

    /// 先生の層の発火率は compile 版と eager 版で一致し、値は 0〜1 (サブステップ平均の発火率)
    func testTeacherLayerRatesCompiledMatchesEager() {
        let inputDim = 16
        let (feats, _) = makeBatch(inputDim: inputDim, frames: 32)
        let net = MLXSpikingNetwork(numLayers: 3, inputDim: inputDim, maxHiddenDim: 64, outputDim: 1)
        net.wIn = net.wIn * 4.0
        let trainer = MLXBPTTTrainer(network: net, bpttWindow: 4)
        let x = MLXArray(feats.flatMap { $0 }.flatMap { $0 }, [2, 32, inputDim])
        let a = trainer.layerRates(network: net, features: x, layer: 1)
        let b = trainer.layerRatesEager(network: net, features: x, layer: 1)
        eval(a, b)
        let fa = a.asArray(Float.self)
        let fb = b.asArray(Float.self)
        XCTAssertEqual(a.shape, [2, 32, 64])
        XCTAssertLessThan(maxAbsDiff(fa, fb), 1e-5)
        XCTAssertLessThanOrEqual(fa.max() ?? 2.0, 1.0)
        XCTAssertLessThan(0.0, fa.reduce(0, +))
    }
}
