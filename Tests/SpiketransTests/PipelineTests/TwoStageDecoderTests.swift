import XCTest
@testable import Spiketrans

final class TwoStageDecoderTests: XCTestCase {

    // MARK: - 1. 特殊トークン定義と整合性テスト

    func testPhonemeVocabularySpecialTokens() {
        let vocab = PhonemeVocabulary()

        XCTAssertEqual(PhonemeVocabulary.padId, 0)
        XCTAssertEqual(PhonemeVocabulary.silId, 1)
        XCTAssertEqual(PhonemeVocabulary.unkId, 2)
        XCTAssertEqual(PhonemeVocabulary.sosId, 3)
        XCTAssertEqual(PhonemeVocabulary.eosId, 4)

        XCTAssertEqual(vocab.token(for: 0), "<pad>")
        XCTAssertEqual(vocab.token(for: 1), "<sil>")
        XCTAssertEqual(vocab.token(for: 2), "<unk>")
        XCTAssertEqual(vocab.token(for: 3), "<sos>")
        XCTAssertEqual(vocab.token(for: 4), "<eos>")

        XCTAssertTrue(vocab.isSilence(0))
        XCTAssertTrue(vocab.isSilence(1))
        XCTAssertFalse(vocab.isSilence(2))
        XCTAssertFalse(vocab.isSilence(3))
        XCTAssertFalse(vocab.isSilence(4))
        XCTAssertFalse(vocab.isSilence(5))
    }

    // MARK: - 2. 語彙テーブルの双方向変換テスト

    func testPhonemeVocabularyBidirectionalMapping() {
        let vocab = PhonemeVocabulary()
        XCTAssertEqual(vocab.size, 64)

        var i = 0
        while i < vocab.size {
            let tok = vocab.token(for: i)
            let retrievedId = vocab.id(for: tok)
            XCTAssertEqual(retrievedId, i, "Bidirectional mismatch for token \(tok) at id \(i)")
            i += 1
        }

        // 範囲外・未知トークン
        XCTAssertEqual(vocab.id(for: "unknown_xyz"), PhonemeVocabulary.unkId)
        XCTAssertEqual(vocab.token(for: -1), "<unk>")
        XCTAssertEqual(vocab.token(for: 100), "<unk>")
    }

    // MARK: - 3. 日本語かな ↔ 音素列の相互変換テスト

    func testKanaToPhonemesAndPhonemesToKana() {
        let vocab = PhonemeVocabulary()

        let testCases: [(input: String, expectedPhonemes: [String], expectedKana: String)] = [
            ("あいうえお", ["a", "i", "u", "e", "o"], "あいうえお"),
            ("からす", ["k", "a", "r", "a", "s", "u"], "からす"),
            ("にほん", ["n", "i", "h", "o", "N"], "にほん"),
            ("とうきょう", ["t", "o", "u", "ky", "o", "u"], "とうきょう"),
            ("きって", ["k", "i", "Q", "t", "e"], "きって"),
            ("しんかんせん", ["sh", "i", "N", "k", "a", "N", "s", "e", "N"], "しんかんせん"),
            ("ラーメン", ["r", "a", "_", "m", "e", "N"], "らーめん")
        ]

        var idx = 0
        while idx < testCases.count {
            let tc = testCases[idx]
            let phonemes = vocab.kanaToPhonemes(tc.input)
            XCTAssertEqual(phonemes, tc.expectedPhonemes, "Phoneme mismatch for \(tc.input)")

            let reconstructedKana = vocab.phonemesToKana(phonemes)
            XCTAssertEqual(reconstructedKana, tc.expectedKana, "Reconstruction mismatch for \(tc.input)")

            let tokenIds = vocab.textToTokens(tc.input)
            let textFromTokens = vocab.tokensToText(tokenIds)
            XCTAssertEqual(textFromTokens, tc.expectedKana, "Token ID reconstruction mismatch for \(tc.input)")

            idx += 1
        }
    }

    // MARK: - 4. CTC 重複圧縮 (Collapse) ロジックテスト

    func testAcousticDecoderCollapseIdenticalTokens() {
        let textVocab = TextVocabulary(characters: Array("からす"))
        let net = SpikingNetwork(inputDim: 64, maxHiddenDim: 256, outputDim: textVocab.size)
        let decoder = AcousticDecoder(network: net, vocabulary: textVocab)

        let kId = textVocab.id(for: "か")
        let aId = textVocab.id(for: "ら")
        let padId = TextVocabulary.padId

        // [k, k, k, a, a, pad, a, a]
        let tokenSeq = [kId, kId, kId, aId, aId, padId, aId, aId]
        var frameProbs: [AcousticFrameProbabilities] = []

        var i = 0
        while i < tokenSeq.count {
            var probs = [Float](repeating: 0.0001, count: textVocab.size)
            probs[tokenSeq[i]] = 0.95
            frameProbs.append(AcousticFrameProbabilities(
                frameIndex: i,
                topTokenId: tokenSeq[i],
                topProbability: 0.95,
                probabilities: probs
            ))
            i += 1
        }

        let collapsed = decoder.collapseTokens(frameProbs)
        // [k, a, a] (pad を挟んだ同一文字は別トークンとして維持される)
        XCTAssertEqual(collapsed, [kId, aId, aId])

        // 全無音フレーム
        var allSil: [AcousticFrameProbabilities] = []
        i = 0
        while i < 10 {
            var probs = [Float](repeating: 0.0001, count: textVocab.size)
            probs[padId] = 0.99
            allSil.append(AcousticFrameProbabilities(
                frameIndex: i,
                topTokenId: padId,
                topProbability: 0.99,
                probabilities: probs
            ))
            i += 1
        }
        let collapsedSil = decoder.collapseTokens(allSil)
        XCTAssertTrue(collapsedSil.isEmpty)
    }

    // MARK: - 5. Float32 vs 量子化エンジン推論結果比較テスト

    func testAcousticDecoderQuantizedVsFloat32() {
        let textVocab = TextVocabulary(characters: Array("あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん"))
        let net = SpikingNetwork(inputDim: 64, maxHiddenDim: 256, outputDim: textVocab.size, timeSteps: 4)
        let qConfig = QuantizedConfig.int32Config()
        let qWeights = QuantizedEngine.quantize(network: net, config: qConfig)
        let qEngine = QuantizedEngine(weights: qWeights, timeSteps: 4)

        let floatDecoder = AcousticDecoder(network: net, vocabulary: textVocab)
        let quantDecoder = AcousticDecoder(network: net, quantizedEngine: qEngine, vocabulary: textVocab)

        let floatWs = AcousticWorkspace(maxHiddenDim: 256, outputDim: textVocab.size, inputDim: 64)
        let quantWs = AcousticWorkspace(maxHiddenDim: 256, outputDim: textVocab.size, inputDim: 64)

        var matchCount = 0
        let totalFrames = 50

        var f = 0
        while f < totalFrames {
            floatWs.reset()
            quantWs.reset()

            var features = [Float](repeating: 0.0, count: 64)
            var d = 0
            while d < 64 {
                features[d] = Float((f * 29 + d * 13) % 100) / 100.0
                d += 1
            }

            let fRes = floatDecoder.decodeFrame(features: features, workspace: floatWs, frameIndex: f)
            let qRes = quantDecoder.decodeFrame(features: features, workspace: quantWs, frameIndex: f)

            if fRes.topTokenId == qRes.topTokenId {
                matchCount += 1
            }

            f += 1
        }

        let matchRate = Float(matchCount) / Float(totalFrames)
        XCTAssertLessThanOrEqual(0.85, matchRate, "Quantized match rate must be at least 85%, got \(matchRate)")
    }

    // MARK: - 6. 言語デコーダ 貪欲法 (Greedy) テスト

    func testLanguageDecoderGreedyDecoding() {
        let vocab = TextVocabulary(characters: Array("からす"))
        let lmNet = SpikingNetwork(inputDim: 64, maxHiddenDim: 256, outputDim: vocab.size, timeSteps: 4)
        let langDecoder = LanguageDecoder(lmNetwork: lmNet, vocabulary: vocab)

        let kId = vocab.id(for: "か")
        let rId = vocab.id(for: "ら")
        let sId = vocab.id(for: "す")

        // "からす" (か, ら, す)
        let tokenSeq = [kId, kId, rId, rId, sId, sId]
        var acousticProbs: [AcousticFrameProbabilities] = []

        var i = 0
        while i < tokenSeq.count {
            var probs = [Float](repeating: 0.0001, count: vocab.size)
            probs[tokenSeq[i]] = 0.95
            acousticProbs.append(AcousticFrameProbabilities(
                frameIndex: i,
                topTokenId: tokenSeq[i],
                topProbability: 0.95,
                probabilities: probs
            ))
            i += 1
        }

        let result = langDecoder.decodeGreedy(acousticProbs: acousticProbs)
        XCTAssertFalse(result.tokens.isEmpty)
        XCTAssertFalse(result.text.isEmpty)
        XCTAssertEqual(result.text, "からす")
    }

    // MARK: - 7. 言語デコーダ ビーム探索 (Beam Search) テスト

    func testLanguageDecoderBeamSearchDecoding() {
        let vocab = TextVocabulary(characters: Array("にほん"))
        let lmNet = SpikingNetwork(inputDim: 64, maxHiddenDim: 256, outputDim: vocab.size, timeSteps: 4)
        let config = LanguageDecoderConfig(beamWidth: 4, lmWeight: 0.3, wordBonus: 0.1)
        let langDecoder = LanguageDecoder(lmNetwork: lmNet, vocabulary: vocab, config: config)

        let nId = vocab.id(for: "に")
        let hId = vocab.id(for: "ほ")
        let nBigId = vocab.id(for: "ん")

        // "にほん" (に, ほ, ん)
        let tokenSeq = [nId, nId, hId, hId, nBigId, nBigId]
        var acousticProbs: [AcousticFrameProbabilities] = []

        var i = 0
        while i < tokenSeq.count {
            var probs = [Float](repeating: 0.0001, count: vocab.size)
            probs[tokenSeq[i]] = 0.92
            acousticProbs.append(AcousticFrameProbabilities(
                frameIndex: i,
                topTokenId: tokenSeq[i],
                topProbability: 0.92,
                probabilities: probs
            ))
            i += 1
        }

        let beamResult = langDecoder.decodeBeamSearch(acousticProbs: acousticProbs)
        XCTAssertFalse(beamResult.tokens.isEmpty)
        XCTAssertEqual(beamResult.text, "にほん")
    }

    // MARK: - 8. スライス動的切り替えテスト

}
