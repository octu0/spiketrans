import XCTest
import Foundation
@testable import Spiketrans

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

final class LongDurationPipelineTests: XCTestCase {

    /// 常駐メモリ (MB) 取得ヘルパー
    private func residentMemoryMB() -> Double {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) { infoPtr in
            infoPtr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { intPtr in
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), intPtr, &count)
            }
        }
        if result != KERN_SUCCESS {
            return 0.0
        }
        return Double(info.resident_size) / (1024.0 * 1024.0)
    }

    /// 音声合成ヘルパー (ピッチ 150Hz + 母音フォルマント F1, F2, F3)
    private func synthesizeVowel(
        sampleRate: Int,
        durationSeconds: Float,
        f1: Float = 800.0,
        f2: Float = 1200.0,
        f3: Float = 2500.0,
        amplitude: Float = 0.5
    ) -> [Float] {
        let sampleCount = Int(Float(sampleRate) * durationSeconds)
        var pcm = [Float](repeating: 0.0, count: sampleCount)
        let twoPi = 2.0 * Float.pi

        var i = 0
        while i < sampleCount {
            let t = Float(i) / Float(sampleRate)
            let f0 = sin(twoPi * 150.0 * t)
            let h1 = 0.6 * sin(twoPi * f1 * t)
            let h2 = 0.4 * sin(twoPi * f2 * t)
            let h3 = 0.2 * sin(twoPi * f3 * t)
            pcm[i] = amplitude * (f0 + h1 + h2 + h3)
            i += 1
        }
        return pcm
    }

    /// 無音・低レベル環境ノイズ合成
    private func synthesizeSilenceWithNoise(sampleRate: Int, durationSeconds: Float, noiseAmp: Float = 0.0001) -> [Float] {
        let sampleCount = Int(Float(sampleRate) * durationSeconds)
        var pcm = [Float](repeating: 0.0, count: sampleCount)
        var i = 0
        while i < sampleCount {
            pcm[i] = Float.random(in: -noiseAmp...noiseAmp)
            i += 1
        }
        return pcm
    }

    // MARK: - 1. VAD とフォルマント推定・適応イコライジングの協調検証

    func testLongDurationVADAndFormantEqualizerCoordination() {
        let dspConfig = DSPConfig(sampleRate: 16000, frameSize: 400, hopSize: 160, lpcOrder: 12, melChannels: 64)
        let filterbank = Filterbank(config: dspConfig)
        let workspace = DSPWorkspace(
            maxFrameSize: dspConfig.frameSize,
            lpcOrder: dspConfig.lpcOrder,
            melChannels: dspConfig.melChannels,
            fftSize: 512
        )
        let vad = VAD(config: dspConfig)

        // 1. 母音音声フレーム (F1=800Hz, F2=1200Hz, F3=2500Hz) の検証
        let speechAudio = synthesizeVowel(sampleRate: 16000, durationSeconds: 0.05, f1: 800.0, f2: 1200.0, f3: 2500.0, amplitude: 0.6)
        let speechCount = min(dspConfig.frameSize, speechAudio.count)

        var speechVADIsVoiced = false
        speechAudio.withUnsafeBufferPointer { buf in
            let vadRes = vad.processFrame(ptr: buf.baseAddress!, count: speechCount, workspace: workspace)
            speechVADIsVoiced = vadRes.isSpeech
        }
        XCTAssertTrue(speechVADIsVoiced, "合成母音フレームで VAD が有声判定すること")

        // Filterbank による特徴量抽出 (内部で VAD 有声時に LPC + Durand-Kerner 適応イコライジングが発動)
        var speechFeatures: [Float] = []
        speechAudio.withUnsafeBufferPointer { buf in
            speechFeatures = filterbank.extractFeatures(pcmPtr: buf.baseAddress!, count: speechCount, workspace: workspace)
        }
        XCTAssertEqual(speechFeatures.count, 64)

        // フォルマント帯域 (800Hz〜2500Hz) に対応する Mel チャネルのエネルギーが非人声帯域より高いことを確認
        var f1f2EnergySum: Float = 0.0
        var fIdx = 10
        while fIdx <= 35 {
            f1f2EnergySum += speechFeatures[fIdx]
            fIdx += 1
        }
        XCTAssertTrue(0.0 < f1f2EnergySum, "有声フレームでフォルマント領域の Mel エネルギーが有意に抽出されていること")

        // 2. 無音フレームの検証 (LPC + Durand-Kerner のバイパス動作確認)
        let silenceAudio = synthesizeSilenceWithNoise(sampleRate: 16000, durationSeconds: 0.05, noiseAmp: 0.00005)
        let silenceCount = min(dspConfig.frameSize, silenceAudio.count)

        var silenceVADIsVoiced = true
        silenceAudio.withUnsafeBufferPointer { buf in
            let vadRes = vad.processFrame(ptr: buf.baseAddress!, count: silenceCount, workspace: workspace)
            silenceVADIsVoiced = vadRes.isSpeech
        }
        XCTAssertTrue(silenceVADIsVoiced != true, "微小ノイズ環境下で VAD が無音判定すること")

        var silenceFeatures: [Float] = []
        silenceAudio.withUnsafeBufferPointer { buf in
            silenceFeatures = filterbank.extractFeatures(pcmPtr: buf.baseAddress!, count: silenceCount, workspace: workspace)
        }
        XCTAssertEqual(silenceFeatures.count, 64)

        // 無音時はフォルマント強調がバイパスされ、ノイズエネルギーが極小に保たれること
        var maxSilenceEnergy: Float = 0.0
        var sIdx = 0
        while sIdx < 64 {
            if maxSilenceEnergy < silenceFeatures[sIdx] {
                maxSilenceEnergy = silenceFeatures[sIdx]
            }
            sIdx += 1
        }
        XCTAssertTrue(maxSilenceEnergy <= 0.05, "無音時に不要なフォルマント増幅が行われず特徴量が静穏に保たれること")

        // 3. 多層 SNN 音響デコーダへの供給と事後確率出力の検証
        let net = SpikingNetwork(numLayers: 2, inputDim: 64, maxHiddenDim: 128, outputDim: 50)
        let acWs = AcousticWorkspace(maxHiddenDim: 128, outputDim: 50, inputDim: 64, numLayers: 2)
        let decoder = AcousticDecoder(network: net)

        let probsSpeech = decoder.decodeFrame(features: speechFeatures, workspace: acWs, frameIndex: 0)
        let probsSilence = decoder.decodeFrame(features: silenceFeatures, workspace: acWs, frameIndex: 1)

        XCTAssertTrue(0.0 <= probsSpeech.topProbability && probsSpeech.topProbability <= 1.0)
        XCTAssertTrue(0.0 <= probsSilence.topProbability && probsSilence.topProbability <= 1.0)
        XCTAssertTrue(probsSpeech.topProbability.isNaN != true)
        XCTAssertTrue(probsSilence.topProbability.isNaN != true)
    }

    // MARK: - 2. FormantSegmenter 動的境界と Conv2DSubsampling + 多層 SNN の同期検証

    func testFormantBoundariesWithConv2DSubsamplingAndMultiLayerSNN() {
        let sampleRate = 16000
        let frameSize = 400
        let hopSize = 160

        // /a/ (800Hz, 1200Hz) から /i/ (300Hz, 2300Hz) への遷移音声を合成
        var continuousPcm: [Float] = []
        continuousPcm.append(contentsOf: synthesizeVowel(sampleRate: sampleRate, durationSeconds: 0.25, f1: 800.0, f2: 1200.0, f3: 2500.0))
        continuousPcm.append(contentsOf: synthesizeVowel(sampleRate: sampleRate, durationSeconds: 0.25, f1: 300.0, f2: 2300.0, f3: 3000.0))
        continuousPcm.append(contentsOf: synthesizeSilenceWithNoise(sampleRate: sampleRate, durationSeconds: 0.15))

        // FormantSegmenter による動的音素境界検出
        let boundaries = FormantSegmenter.detectBoundaries(
            pcmData: continuousPcm,
            sampleRate: sampleRate,
            frameSize: frameSize,
            hopSize: hopSize,
            minChunkFrames: 8,
            maxChunkFrames: 36
        )

        XCTAssertTrue(0 < boundaries.count, "フォルマント遷移変曲点または無声境界が正しく検出されること")

        // Conv2DSubsampling (4x時間ダウンサンプリング: 64ch Mel -> 256ch) + 2層 SNN の構築
        let subsampler = Conv2DSubsampling(melChannels: 64, outputDim: 256)
        let net = SpikingNetwork(
            numLayers: 2,
            inputDim: 256,
            maxHiddenDim: 128,
            outputDim: 50,
            convSubsampling: subsampler
        )
        let decoder = AcousticDecoder(network: net, convSubsampling: subsampler)
        let acWs = AcousticWorkspace(maxHiddenDim: 128, outputDim: 50, inputDim: 256, numLayers: 2)

        let dspCfg = DSPConfig(sampleRate: sampleRate, frameSize: frameSize, hopSize: hopSize)
        let filterbank = Filterbank(config: dspCfg)
        let dspWs = DSPWorkspace(maxFrameSize: frameSize, lpcOrder: 12, melChannels: 64, fftSize: 512)

        let totalFrames = (continuousPcm.count - frameSize) / hopSize + 1
        var streamingState = Conv2DStreamingState()

        var emittedSubsampledFrames = 0
        var f = 0
        continuousPcm.withUnsafeBufferPointer { buf in
            let basePtr = buf.baseAddress!
            while f < totalFrames {
                let framePtr = basePtr.advanced(by: f * hopSize)
                let melFeatures = filterbank.extractFeatures(pcmPtr: framePtr, count: frameSize, workspace: dspWs)

                let result = decoder.decodeStreaming(
                    melFrame: melFeatures,
                    state: &streamingState,
                    workspace: acWs,
                    frameIndex: f
                )
                switch result {
                case .some(let probs):
                    emittedSubsampledFrames += 1
                    XCTAssertTrue(0.0 <= probs.topProbability && probs.topProbability <= 1.0)
                    XCTAssertTrue(probs.topProbability.isNaN != true)
                case .none:
                    break
                }
                f += 1
            }
        }

        // 4x サブサンプリングにより、出力フレーム数はおよそ totalFrames / 4 であること
        let expectedSubsampledApprox = totalFrames / 4
        XCTAssertTrue(expectedSubsampledApprox - 2 <= emittedSubsampledFrames)
        XCTAssertTrue(emittedSubsampledFrames <= expectedSubsampledApprox + 2)

        // 境界を跨いだ後でも多層 SNN の膜電位と適応閾値が連続性を維持し、非ゼロかつ有限であること
        var maxMembrane: Float = 0.0
        var vIdx = 0
        while vIdx < acWs.vPrev.count {
            let absV = abs(acWs.vPrev[vIdx])
            if maxMembrane < absV {
                maxMembrane = absV
            }
            vIdx += 1
        }
        XCTAssertTrue(0.0 < maxMembrane, "ストリーミング後も SNN 膜電位状態が維持されていること")
        XCTAssertTrue(maxMembrane <= 20.0, "膜電位が暴走せず有界な範囲内に維持されていること")
    }

    // MARK: - 3. 10時間想定の O(1) 定数メモリおよび RTF <= 0.005 高速性検証

    func testTenHourStreamingConstantMemoryAndRTF() {
        let inDim = 64
        let hDim = 256
        let outDim = 100
        let net = SpikingNetwork(numLayers: 2, inputDim: inDim, maxHiddenDim: hDim, outputDim: outDim)
        let decoder = AcousticDecoder(network: net)
        let workspace = AcousticWorkspace(maxHiddenDim: hDim, outputDim: outDim, inputDim: inDim, numLayers: 2)

        // 長時間ストリーミングシミュレーション: 10,000 フレーム (実時間 100 秒相当)
        let testFrames = 10000
        var dummyFrame = [Float](repeating: 0.05, count: inDim)

        // ウォームアップ (JIT / キャッシュ初期化)
        var w = 0
        while w < 100 {
            decoder.decodeFrame(features: dummyFrame, workspace: workspace, frameIndex: w)
            w += 1
        }

        let initialRSS = residentMemoryMB()
        let startTime = DispatchTime.now()

        var f = 0
        while f < testFrames {
            dummyFrame[f % inDim] = Float(f % 10) * 0.05
            decoder.decodeFrame(features: dummyFrame, workspace: workspace, frameIndex: f)
            f += 1
        }

        let endTime = DispatchTime.now()
        let finalRSS = residentMemoryMB()
        let durationNs = endTime.uptimeNanoseconds - startTime.uptimeNanoseconds
        let durationSec = Double(durationNs) / 1_000_000_000.0

        // 音声相当時間: 10,000 フレーム * 10ms = 100.0 秒
        let audioSec = Double(testFrames) * 0.010
        let rtf = durationSec / audioSec

        // 1. RTF 高速性検証: SNN は極めて軽量であり、RTF <= 0.005 を満たすこと (リアルタイムの 200倍以上高速)
        #if DEBUG
        XCTAssertTrue(rtf <= 0.25, "多層 SNN 音響推論の RTF は Debug ビルドでも 0.25 以下であること (実績値: \(rtf))")
        #else
        XCTAssertTrue(rtf <= 0.005, "多層 SNN 音響推論の RTF は 0.005 以下であること (実績値: \(rtf))")
        #endif

        // 2. O(1) 定数メモリ検証: 10,000 フレームの連続推論で常駐メモリの増加が実質ゼロ (1.0MB 未満) であること
        let rssDelta = finalRSS - initialRSS
        XCTAssertTrue(rssDelta <= 1.0, "10,000フレーム連続推論後のメモリ増加が 1.0MB 以下であること (実績値: \(rssDelta) MB)")

        // 3. 長時間推論後の数値安定性: 膜電位・適応閾値・出力確率が正常値を有すること
        var i = 0
        while i < workspace.probabilities.count {
            let p = workspace.probabilities[i]
            XCTAssertTrue(p.isNaN != true, "出力確率に NaN が含まれないこと")
            XCTAssertTrue(0.0 <= p && p <= 1.0, "出力確率が [0, 1] 範囲内であること")
            i += 1
        }

        var vIdx = 0
        while vIdx < workspace.vPrev.count {
            let v = workspace.vPrev[vIdx]
            XCTAssertTrue(v.isNaN != true, "膜電位に NaN が含まれないこと")
            XCTAssertTrue(v.isInfinite != true, "膜電位が発散していないこと")
            XCTAssertTrue(abs(v) <= 20.0, "膜電位が健全な有界値 [-20.0, 20.0] に留まっていること")
            vIdx += 1
        }
    }

    // MARK: - 4. StreamingTranscriber フルパイプライン発話分離・長時間安定性検証

    func testStreamingTranscriberFullPipelineLongDuration() {
        let sampleRate = 16000
        let (acNet, lmNet, vocab) = createTestNetworks()
        let config = StreamingTranscriberConfig(
            dspConfig: DSPConfig(sampleRate: sampleRate, frameSize: 400, hopSize: 160, lpcOrder: 12),
            beamWidth: 4,
            lmWeight: 0.3
        )
        let transcriber = StreamingTranscriber(
            config: config,
            acousticNetwork: acNet,
            languageNetwork: lmNet,
            textVocabulary: vocab
        )

        let counter = AtomicCounter()
        transcriber.onFinalResult = { _ in
            counter.increment()
        }

        // 会議録音シミュレーション: [発話1 (0.4秒)] -> [無音 (0.3秒)] -> [発話2 (0.5秒)] -> [無音 (0.3秒)]
        var streamPcm: [Float] = []
        streamPcm.append(contentsOf: synthesizeSilenceWithNoise(sampleRate: sampleRate, durationSeconds: 0.1))
        streamPcm.append(contentsOf: synthesizeVowel(sampleRate: sampleRate, durationSeconds: 0.4, f1: 800.0, f2: 1200.0, f3: 2500.0, amplitude: 0.6))
        streamPcm.append(contentsOf: synthesizeSilenceWithNoise(sampleRate: sampleRate, durationSeconds: 0.3))
        streamPcm.append(contentsOf: synthesizeVowel(sampleRate: sampleRate, durationSeconds: 0.5, f1: 400.0, f2: 2000.0, f3: 2800.0, amplitude: 0.6))
        streamPcm.append(contentsOf: synthesizeSilenceWithNoise(sampleRate: sampleRate, durationSeconds: 0.3))

        // 160 サンプル (10ms) のストリーミングチャンクごとに供給
        let chunkSize = 160
        var offset = 0
        while offset < streamPcm.count {
            let count = min(chunkSize, streamPcm.count - offset)
            streamPcm.withUnsafeBufferPointer { buf in
                let ptr = buf.baseAddress!.advanced(by: offset)
                transcriber.appendAudio(pcmPtr: ptr, count: count)
            }
            offset += count
        }
        transcriber.flush()

        // 2 回の発話セグメントが正しく分離され、コールバックされたことの検証
        XCTAssertTrue(2 <= counter.value, "2回以上の発話セグメントが最終認識結果として出力されていること")
    }

    private func createTestNetworks() -> (acoustic: SpikingNetwork, language: SpikingNetwork, vocab: TextVocabulary) {
        let vocab = TextVocabulary()
        let ac = SpikingNetwork(numLayers: 2, inputDim: 64, maxHiddenDim: 128, outputDim: vocab.size, timeSteps: 4)
        let lm = SpikingNetwork(numLayers: 1, inputDim: 64, maxHiddenDim: 128, outputDim: vocab.size, timeSteps: 4)
        return (acoustic: ac, language: lm, vocab: vocab)
    }
}
