import XCTest
import Foundation
@testable import Spiketrans

/// スレッドセーフな結果収集ヘルパー
private final class ThreadSafeCollector<T>: @unchecked Sendable {
    private let lock = NSLock()
    private var items: [T] = []

    func append(_ item: T) {
        lock.lock()
        defer { lock.unlock() }
        items.append(item)
    }

    var count: Int {
        lock.lock()
        defer { lock.unlock() }
        return items.count
    }

    var allItems: [T] {
        lock.lock()
        defer { lock.unlock() }
        return items
    }

    subscript(index: Int) -> T {
        lock.lock()
        defer { lock.unlock() }
        return items[index]
    }

    func reset() {
        lock.lock()
        defer { lock.unlock() }
        items.removeAll()
    }
}

/// スレッドセーフなカウンター
private final class ThreadSafeCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var count: Int = 0

    func increment() {
        lock.lock()
        defer { lock.unlock() }
        count += 1
    }

    var value: Int {
        lock.lock()
        defer { lock.unlock() }
        return count
    }

    func reset() {
        lock.lock()
        defer { lock.unlock() }
        count = 0
    }
}

/// Milestone M3 (Two-Stage STT Pipeline & Streaming Decoder) 敵対的・極限入力検証テストスイート (Challenger 1)
final class M3AdversarialChallengerTests: XCTestCase {

    private func createTestNetworks() -> (acoustic: SpikingNetwork, language: SpikingNetwork) {
        let vocabSize = TextVocabulary().size
        let ac = SpikingNetwork(inputDim: 64, maxHiddenDim: 256, outputDim: vocabSize, timeSteps: 4)
        let lm = SpikingNetwork(inputDim: 64, maxHiddenDim: 256, outputDim: vocabSize, timeSteps: 4)
        return (acoustic: ac, language: lm)
    }

    private func synthesizeVowelSpeech(sampleRate: Int, durationSeconds: Float, amplitude: Float = 0.5) -> [Float] {
        let sampleCount = Int(Float(sampleRate) * durationSeconds)
        var pcm = [Float](repeating: 0.0, count: sampleCount)
        let twoPi = 2.0 * Float.pi

        var i = 0
        while i < sampleCount {
            let t = Float(i) / Float(sampleRate)
            let f0 = sin(twoPi * 220.0 * t)
            let f1 = 0.6 * sin(twoPi * 800.0 * t)
            let f2 = 0.4 * sin(twoPi * 1200.0 * t)
            let f3 = 0.2 * sin(twoPi * 2600.0 * t)
            pcm[i] = amplitude * (f0 + f1 + f2 + f3)
            i += 1
        }
        return pcm
    }

    // MARK: - 1. 完全無音・長時間無音・微小信号入力に対する安全性

    /// 長時間完全無音 (100,000 サンプル = 6.25秒) を連続ストリーミングしても発話判定・不要アロケーションが発生しないこと
    func testLongPureSilenceStreaming() {
        let (acNet, lmNet) = createTestNetworks()
        let transcriber = StreamingTranscriber(acousticNetwork: acNet, languageNetwork: lmNet)

        let finalCollector = ThreadSafeCollector<TranscriptionResult>()
        let partialCollector = ThreadSafeCollector<TranscriptionResult>()

        transcriber.onFinalResult = { res in
            finalCollector.append(res)
        }
        transcriber.onPartialResult = { res in
            partialCollector.append(res)
        }

        // 100,000 サンプルの完全ゼロ PCM (160 サンプル/チャンク)
        let totalSamples = 100000
        let chunkSize = 160
        let silenceChunk = [Float](repeating: 0.0, count: chunkSize)

        var processed = 0
        while processed < totalSamples {
            silenceChunk.withUnsafeBufferPointer { buf in
                transcriber.appendAudio(pcmPtr: buf.baseAddress!, count: chunkSize)
            }
            processed += chunkSize
        }

        // フラッシュ実行
        transcriber.flush()

        // 完全無音なので発話検出は 0 回でなければならない
        XCTAssertEqual(finalCollector.count, 0, "Pure silence must not trigger any final result")
        XCTAssertEqual(partialCollector.count, 0, "Pure silence must not trigger any partial result")
    }

    /// 極小振幅 (量子化ノイズ未満・微小浮動小数点 1e-9) の無音入力テスト
    func testNearZeroSubAudibleNoiseStreaming() {
        let (acNet, lmNet) = createTestNetworks()
        let transcriber = StreamingTranscriber(acousticNetwork: acNet, languageNetwork: lmNet)

        let finalCollector = ThreadSafeCollector<TranscriptionResult>()
        transcriber.onFinalResult = { res in
            finalCollector.append(res)
        }

        var subAudibleChunk = [Float](repeating: 0.0, count: 160)
        var i = 0
        while i < 160 {
            subAudibleChunk[i] = 1e-9 * Float(i % 5)
            i += 1
        }

        var frame = 0
        while frame < 500 { // 5秒分
            subAudibleChunk.withUnsafeBufferPointer { buf in
                transcriber.appendAudio(pcmPtr: buf.baseAddress!, count: 160)
            }
            frame += 1
        }
        transcriber.flush()

        XCTAssertEqual(finalCollector.count, 0, "Sub-audible tiny float values must not trigger speech")
    }

    /// 0 サンプル、1 サンプル、非アライメント長サンプルの連続 appendAudio 境界テスト
    func testIrregularChunkSizeAndEmptyAudioAppend() {
        let (acNet, lmNet) = createTestNetworks()
        let transcriber = StreamingTranscriber(acousticNetwork: acNet, languageNetwork: lmNet)

        // 空配列
        transcriber.appendAudio(pcm: [])
        // 1 サンプル
        transcriber.appendAudio(pcm: [0.0])
        // 7 サンプル (素数長)
        transcriber.appendAudio(pcm: [0.1, -0.1, 0.2, -0.2, 0.05, -0.05, 0.0])
        // 161 サンプル (hopSize + 1)
        let oddAudio = [Float](repeating: 0.01, count: 161)
        transcriber.appendAudio(pcm: oddAudio)
        // 32769 サンプル (ringBufferCapacity + 1 の巨大チャンク)
        let hugeAudio = [Float](repeating: 0.005, count: 32769)
        transcriber.appendAudio(pcm: hugeAudio)

        transcriber.flush()
        transcriber.reset()
    }

    // MARK: - 2. 短時間ノイズバースト & 100回連続短発話・長発話ストレステスト

    /// 100 回連続の短時間ノイズバースト (10ms 〜 30ms) に対するステートマシン整合性と堅牢性
    func test100ConsecutiveShortNoiseBursts() {
        let (acNet, lmNet) = createTestNetworks()
        let transcriber = StreamingTranscriber(acousticNetwork: acNet, languageNetwork: lmNet)

        let finalCollector = ThreadSafeCollector<TranscriptionResult>()
        transcriber.onFinalResult = { res in
            finalCollector.append(res)
        }

        let sampleRate = 16000
        let burstLength = 160 // 10ms (1フレーム)
        let silenceLength = 480 // 30ms (3フレーム)

        var burst = [Float](repeating: 0.0, count: burstLength)
        var s = 0
        while s < burstLength {
            burst[s] = Float.random(in: -0.9...0.9)
            s += 1
        }
        let silence = [Float](repeating: 0.0, count: silenceLength)

        var rep = 0
        while rep < 100 {
            burst.withUnsafeBufferPointer { buf in
                transcriber.appendAudio(pcmPtr: buf.baseAddress!, count: burstLength)
            }
            silence.withUnsafeBufferPointer { buf in
                transcriber.appendAudio(pcmPtr: buf.baseAddress!, count: silenceLength)
            }
            rep += 1
        }
        transcriber.flush()

        // 10msの単発ノイズは VAD の連続判定 (3フレーム以上) によりフィルタされ、クラッシュしないこと
        print("100 short noise bursts: final results emitted = \(finalCollector.count)")
        XCTAssertLessThanOrEqual(finalCollector.count, 100)
    }

    /// 100 回連続の正規短発話 (500ms 発話 + 400ms 無音) の完全処理とタイムスタンプ整合性
    func test100ConsecutiveRealUtterances() {
        let (acNet, lmNet) = createTestNetworks()
        let transcriber = StreamingTranscriber(
            config: StreamingTranscriberConfig(beamWidth: 1),
            acousticNetwork: acNet,
            languageNetwork: lmNet
        )

        let finalCollector = ThreadSafeCollector<TranscriptionResult>()
        transcriber.onFinalResult = { res in
            finalCollector.append(res)
        }

        let sampleRate = 16000
        let speech = synthesizeVowelSpeech(sampleRate: sampleRate, durationSeconds: 0.4) // 400ms
        let silence = [Float](repeating: 0.0, count: Int(Float(sampleRate) * 0.3)) // 300ms

        let chunkSize = 160
        var rep = 0
        while rep < 100 {
            var offset = 0
            while offset < speech.count {
                let count = min(chunkSize, speech.count - offset)
                speech.withUnsafeBufferPointer { buf in
                    let ptr = buf.baseAddress!.advanced(by: offset)
                    transcriber.appendAudio(pcmPtr: ptr, count: count)
                }
                offset += count
            }

            offset = 0
            while offset < silence.count {
                let count = min(chunkSize, silence.count - offset)
                silence.withUnsafeBufferPointer { buf in
                    let ptr = buf.baseAddress!.advanced(by: offset)
                    transcriber.appendAudio(pcmPtr: ptr, count: count)
                }
                offset += count
            }

            rep += 1
        }
        transcriber.flush()

        XCTAssertEqual(finalCollector.count, 100, "100 distinct speech utterances must emit exactly 100 final results")

        // タイムスタンプの単調増加性チェック
        var prevEnd: Float = 0.0
        var idx = 0
        while idx < finalCollector.count {
            let res = finalCollector[idx]
            XCTAssertLessThan(res.startTimeSeconds, res.endTimeSeconds)
            XCTAssertLessThanOrEqual(prevEnd, res.startTimeSeconds + 0.001)
            prevEnd = res.endTimeSeconds
            idx += 1
        }
    }

    /// 最大セグメント長 (15 秒) を超える超長発話 (20 秒連続音声) に対する自動セグメント分割
    func testMaxSegmentDurationAutoSplit() {
        let (acNet, lmNet) = createTestNetworks()
        let config = StreamingTranscriberConfig(
            beamWidth: 1,
            maxSegmentDurationSeconds: 5.0 // テスト短縮のため 5 秒に設定
        )
        let transcriber = StreamingTranscriber(
            config: config,
            acousticNetwork: acNet,
            languageNetwork: lmNet
        )

        let finalCollector = ThreadSafeCollector<TranscriptionResult>()
        transcriber.onFinalResult = { res in
            finalCollector.append(res)
        }

        let sampleRate = 16000
        // 12 秒間の無休連続発話音声
        let longSpeech = synthesizeVowelSpeech(sampleRate: sampleRate, durationSeconds: 12.0)
        let chunkSize = 160

        var offset = 0
        while offset < longSpeech.count {
            let count = min(chunkSize, longSpeech.count - offset)
            longSpeech.withUnsafeBufferPointer { buf in
                let ptr = buf.baseAddress!.advanced(by: offset)
                transcriber.appendAudio(pcmPtr: ptr, count: count)
            }
            offset += count
        }
        transcriber.flush()

        // 12秒の音声が 5秒の maxSegmentDuration により複数セグメントに自動分割されること
        XCTAssertLessThanOrEqual(2, finalCollector.count, "12s continuous speech must be auto-split into at least 2 segments")
        let maxAllowedDuration = config.maxSegmentDurationSeconds + 1.5
        for res in finalCollector.allItems {
            let segDuration = res.endTimeSeconds - res.startTimeSeconds
            XCTAssertLessThanOrEqual(segDuration, maxAllowedDuration, "Each segment must not exceed max duration limit + margin")
        }
    }

    // MARK: - 3. 異常音素列・未知トークン・促音連続に対する PhonemeVocabulary の堅牢性

    /// 促音 `Q` の連続、長音 `_` の連続、母音連続、子音連続に対する復元堅牢性
    func testPhonemeVocabularyExtremeSequences() {
        let vocab = PhonemeVocabulary()

        // 1. 促音連続 (QQQQ)
        let qSeq = ["Q", "Q", "Q", "Q"]
        let qKana = vocab.phonemesToKana(qSeq)
        XCTAssertEqual(qKana, "っっっっ")

        // 2. 長音連続 (____)
        let barSeq = ["_", "_", "_", "_"]
        let barKana = vocab.phonemesToKana(barSeq)
        XCTAssertEqual(barKana, "ーーーー")

        // 3. 子音のみの連続 (母音なし: k, s, t, n, h, m, r, w, g, z, d, b, p)
        let consonantSeq = ["k", "s", "t", "n", "h", "m", "r", "w", "g", "z", "d", "b", "p"]
        let consonantKana = vocab.phonemesToKana(consonantSeq)
        XCTAssertEqual(consonantKana, "くすとんふむるうぐずどぶぷ")

        // 4. 拗音のみの連続 (母音なし: ky, sh, ch, ts, ny, hy, my, ry, gy, j, by, py)
        let ySeq = ["ky", "sh", "ch", "ts", "ny", "hy", "my", "ry", "gy", "j", "by", "py"]
        let yKana = vocab.phonemesToKana(ySeq)
        // 拗音+母音なしはスキップまたは安全に処理
        XCTAssertFalse(yKana.isEmpty)

        // 5. 促音 + 子音 + 母音 (k, i, Q, t, e -> きって)
        let kitte = vocab.phonemesToKana(["k", "i", "Q", "t", "e"])
        XCTAssertEqual(kitte, "きって")

        // 6. 促音 + 促音 + 子音 + 母音 (k, i, Q, Q, t, e -> きっって)
        let kitte2 = vocab.phonemesToKana(["k", "i", "Q", "Q", "t", "e"])
        XCTAssertEqual(kitte2, "きっって")
    }

    /// 特殊トークン・予約トークン・未知トークン混在時の日本語復元
    func testPhonemeVocabularySpecialAndUnknownTokens() {
        let vocab = PhonemeVocabulary()

        // 特殊トークンのみ
        let specialPhonemes = ["<pad>", "<sil>", "<sos>", "<eos>", "<unk>"]
        let emptyKana = vocab.phonemesToKana(specialPhonemes)
        XCTAssertEqual(emptyKana, "", "Special tokens must be ignored during kana reconstruction")

        // 予約トークン (<res40> など)
        let resPhonemes = ["<res40>", "<res45>", "<res63>", "a"]
        let resKana = vocab.phonemesToKana(resPhonemes)
        XCTAssertEqual(resKana, "あ")

        // 未知の音素文字列
        let unknownPhonemes = ["invalid_phoneme_1", "invalid_phoneme_2", "k", "a"]
        let unknownKana = vocab.phonemesToKana(unknownPhonemes)
        XCTAssertEqual(unknownKana, "か")

        // 未定義 ID 列の tokensToText
        let invalidIds = [-100, -1, 64, 100, 99999, PhonemeVocabulary.padId, PhonemeVocabulary.silId, vocab.id(for: "a")]
        let validText = vocab.tokensToText(invalidIds)
        XCTAssertEqual(validText, "あ")
    }

    /// 日本語ひらがな/カタカナ以外の特殊入力 (漢字、英語、記号、絵文字、空文字) に対する kanaToPhonemes の安全性
    func testKanaToPhonemesNonKanaInputs() {
        let vocab = PhonemeVocabulary()

        // 空文字列
        XCTAssertTrue(vocab.kanaToPhonemes("").isEmpty)

        // 漢字・英語・記号・絵文字
        let weirdText = "漢字English1234!@#$%^&*()_+-=😀🎉🍣"
        let phonemes = vocab.kanaToPhonemes(weirdText)
        // クラッシュせず、安全に空または無視されること
        XCTAssertNotNil(phonemes)

        // 混在文字列
        let mixed = "こんにちはHello世界"
        let mixedPhonemes = vocab.kanaToPhonemes(mixed)
        let mixedKana = vocab.phonemesToKana(mixedPhonemes)
        XCTAssertEqual(mixedKana, "こんにちは")
    }

    // MARK: - 4. 音響デコーダ & 言語デコーダの極限入力耐性

    /// 特異な事後確率分布 (全0、全均等、全無音、全1.0) に対するデコーダの堅牢性
    func testDecodersWithPathologicalProbabilities() {
        let (acNet, lmNet) = createTestNetworks()
        let acousticDecoder = AcousticDecoder(network: acNet)
        let langDecoder = LanguageDecoder(lmNetwork: lmNet)

        let vocabSize = TextVocabulary().size
        // 1. 全確率が 0.0 のフレーム
        let zeroProbs = [Float](repeating: 0.0, count: vocabSize)
        let frameZero = AcousticFrameProbabilities(
            frameIndex: 0,
            topTokenId: 0,
            topProbability: 0.0,
            probabilities: zeroProbs
        )

        // 2. 全確率が 1/vocabSize の一様分布フレーム
        let uniformProbs = [Float](repeating: 1.0 / Float(vocabSize), count: vocabSize)
        let frameUniform = AcousticFrameProbabilities(
            frameIndex: 1,
            topTokenId: 5,
            topProbability: 1.0 / Float(vocabSize),
            probabilities: uniformProbs
        )

        // 3. 全確率が極小値 (1e-12) のフレーム
        let tinyProbs = [Float](repeating: 1e-12, count: vocabSize)
        let frameTiny = AcousticFrameProbabilities(
            frameIndex: 2,
            topTokenId: 0,
            topProbability: 1e-12,
            probabilities: tinyProbs
        )

        let badFrames = [frameZero, frameUniform, frameTiny]

        // collapseTokens がクラッシュしないこと
        let collapsed = acousticDecoder.collapseTokens(badFrames)
        XCTAssertNotNil(collapsed)

        // Greedy デコードがクラッシュせず finite なスコアを返すこと
        let greedyRes = langDecoder.decodeGreedy(acousticProbs: badFrames)
        XCTAssertFalse(greedyRes.score.isNaN)
        XCTAssertFalse(greedyRes.score.isInfinite)

        // BeamSearch デコードがクラッシュせず finite なスコアを返すこと
        let beamRes = langDecoder.decodeBeamSearch(acousticProbs: badFrames)
        XCTAssertFalse(beamRes.score.isNaN)
        XCTAssertFalse(beamRes.score.isInfinite)
    }

    /// 特徴量入力に NaN / Inf / 極大値が存在する場合の音響推論の耐性
    func testAcousticDecoderWithNaNAndInfInputs() {
        let (acNet, _) = createTestNetworks()
        let decoder = AcousticDecoder(network: acNet)
        let vocabSize = TextVocabulary().size
        let workspace = AcousticWorkspace(maxHiddenDim: 256, outputDim: vocabSize, inputDim: 64)

        // NaN 混入特徴量
        var nanFeatures = [Float](repeating: 0.5, count: 64)
        nanFeatures[0] = Float.nan
        nanFeatures[15] = Float.infinity
        nanFeatures[31] = -Float.infinity

        let result = decoder.decodeFrame(features: nanFeatures, workspace: workspace, frameIndex: 0)
        XCTAssertLessThanOrEqual(0, result.topTokenId)
        XCTAssertLessThan(result.topTokenId, vocabSize)

        // 超大数 (1e10) と超小数 (-1e10)
        var extremeFeatures = [Float](repeating: 1e10, count: 64)
        extremeFeatures[10] = -1e10
        let extremeResult = decoder.decodeFrame(features: extremeFeatures, workspace: workspace, frameIndex: 1)
        XCTAssertLessThanOrEqual(0, extremeResult.topTokenId)
        XCTAssertLessThan(extremeResult.topTokenId, vocabSize)
    }

    /// 言語デコーダ設定パラメータの極限値 (beamWidth = 1, maxSequenceLength = 0, 1000)
    func testLanguageDecoderExtremeConfigs() {
        let vocab = TextVocabulary(characters: Array("あいうえお"))
        let lmNet = SpikingNetwork(inputDim: 64, maxHiddenDim: 256, outputDim: vocab.size, timeSteps: 4)

        // 1. maxSequenceLength = 0
        let configZeroLen = LanguageDecoderConfig(beamWidth: 4, maxSequenceLength: 0)
        let decoderZeroLen = LanguageDecoder(lmNetwork: lmNet, vocabulary: vocab, config: configZeroLen)

        let aId = vocab.id(for: "あ")
        var dummyProbs = [Float](repeating: 0.0001, count: vocab.size)
        dummyProbs[aId] = 0.95
        let dummyFrame = AcousticFrameProbabilities(
            frameIndex: 0,
            topTokenId: aId,
            topProbability: 0.95,
            probabilities: dummyProbs
        )

        let zeroGreedy = decoderZeroLen.decodeGreedy(acousticProbs: [dummyFrame, dummyFrame])
        XCTAssertTrue(zeroGreedy.tokens.isEmpty, "maxSequenceLength=0 must yield 0 tokens")

        // 2. beamWidth = 1 (実質 Greedy な BeamSearch)
        let configBeam1 = LanguageDecoderConfig(beamWidth: 1, maxSequenceLength: 50)
        let decoderBeam1 = LanguageDecoder(lmNetwork: lmNet, vocabulary: vocab, config: configBeam1)
        let beam1Res = decoderBeam1.decodeBeamSearch(acousticProbs: [dummyFrame, dummyFrame])
        XCTAssertFalse(beam1Res.tokens.isEmpty)
        XCTAssertEqual(beam1Res.text, "あ")
    }

    // MARK: - 5. スレッド安全性・複数インスタンス同時実行

    /// 複数スレッドから同時に PhonemeVocabulary の変換メソッドを呼び出す並行テスト
    func testPhonemeVocabularyConcurrentAccess() {
        let vocab = PhonemeVocabulary()
        let expectation = XCTestExpectation(description: "Concurrent PhonemeVocabulary Access")
        expectation.expectedFulfillmentCount = 10

        let queue = DispatchQueue(label: "org.spiketrans.vocab.concurrent", attributes: .concurrent)

        var t = 0
        while t < 10 {
            queue.async {
                var iter = 0
                while iter < 200 {
                    let words = ["とうきょう", "しんかんせん", "にほん", "からす", "らーめん", "きって"]
                    let w = words[iter % words.count]
                    let phonemes = vocab.kanaToPhonemes(w)
                    let kana = vocab.phonemesToKana(phonemes)
                    XCTAssertFalse(kana.isEmpty)
                    let tokens = vocab.textToTokens(w)
                    let text = vocab.tokensToText(tokens)
                    XCTAssertFalse(text.isEmpty)
                    iter += 1
                }
                expectation.fulfill()
            }
            t += 1
        }

        wait(for: [expectation], timeout: 5.0)
    }
}
