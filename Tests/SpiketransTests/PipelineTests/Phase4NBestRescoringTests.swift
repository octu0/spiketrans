import Foundation
import XCTest
@testable import Spiketrans

private final class ResultCollector<T>: @unchecked Sendable {
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
}

/// フェーズ4: Stage 2 デコーダの言語モデル統合と N-best リスコアリングの包括的単体テストスイート
final class Phase4NBestRescoringTests: XCTestCase {

    // MARK: - 1. 音響 CTC デコーダ N-best 抽出検証

    /// CTCStreamingDecoder および CTCBeamDecoder からの N-best 音響仮説抽出、ソート順、重複排除検証
    func testCTCStreamingDecoderAndBeamDecoderNBest() {
        let vocab = TextVocabulary(corpus: ["あいうえお", "かきくけこ", "さしすせそ"])
        let beamDecoder = CTCBeamDecoder(
            vocabulary: vocab,
            blankId: TextVocabulary.padId,
            beamWidth: 8,
            lengthBonus: 0.1,
            tokenPruneCount: 4
        )

        // 空入力の境界値検証
        let emptyNBest = beamDecoder.decodeNBest(logProbs: [], n: 5)
        XCTAssertTrue(emptyNBest.isEmpty, "空フレーム入力に対して空の N-best を返すこと")

        // 3フレームの対数確率系列を構築
        // frame 0: "あ"(id: 4) が高確率、"か"(id: 9) が次点
        // frame 1: blank が高確率
        // frame 2: "い"(id: 5) が高確率、"き"(id: 10) が次点
        let vSize = vocab.size
        var logProbs = [[Float]](repeating: [Float](repeating: -10.0, count: vSize), count: 3)

        logProbs[0][TextVocabulary.padId] = -2.0
        logProbs[0][4] = -0.2   // "あ"
        logProbs[0][9] = -1.5   // "か"

        logProbs[1][TextVocabulary.padId] = -0.1 // blank
        logProbs[1][4] = -4.0
        logProbs[1][5] = -4.0

        logProbs[2][TextVocabulary.padId] = -2.0
        logProbs[2][5] = -0.3   // "い"
        logProbs[2][10] = -1.2  // "き"

        let nBest = beamDecoder.decodeNBest(logProbs: logProbs, n: 4)
        XCTAssertTrue(nBest.isEmpty != true, "N-best 仮説が抽出されること")
        XCTAssertTrue(nBest.count <= 4, "指定件数 (4件) 以下の仮説が返されること")

        // スコア降順ソートの検証
        var idx = 1
        while idx < nBest.count {
            XCTAssertTrue(nBest[idx].score <= nBest[idx - 1].score, "N-best はスコア降順で並んでいること")
            idx += 1
        }

        // 最上位仮説の妥当性検証 ("あい" が最尤)
        let topHyp = nBest[0]
        XCTAssertEqual(topHyp.text, "あい")
        XCTAssertTrue(topHyp.acousticScore.isFinite, "音響対数確率は有限値であること")
        XCTAssertTrue(topHyp.score.isFinite, "総合スコアは有限値であること")

        // CTCBeamSearch 型エイリアスの検証
        let beamSearch: CTCBeamSearch = beamDecoder
        let nBestFromAlias = beamSearch.decodeNBest(logProbs: logProbs, n: 2)
        XCTAssertEqual(nBestFromAlias.count, min(2, nBest.count))
        XCTAssertEqual(nBestFromAlias[0].text, nBest[0].text)

        // ストリーミングデコーダからの N-best 抽出検証
        let streaming = beamDecoder.makeStreamingDecoder()
        streaming.push(frame: logProbs[0])
        streaming.push(frame: logProbs[1])
        streaming.push(frame: logProbs[2])

        let streamNBest = streaming.nBest(n: 4)
        XCTAssertEqual(streamNBest.count, nBest.count)
        XCTAssertEqual(streamNBest[0].text, nBest[0].text)
        XCTAssertEqual(streamNBest[0].acousticScore, nBest[0].acousticScore, accuracy: 1e-4)
    }

    /// AcousticDecoder による音響特徴量からの対数確率算出および N-best 抽出検証
    func testAcousticDecoderNBestExtraction() {
        let vocab = TextVocabulary(corpus: ["テスト", "おんせい", "にんしき"])
        let network = SpikingNetwork(
            inputDim: 32,
            maxHiddenDim: 64,
            outputDim: vocab.size,
            timeSteps: 2
        )
        let decoder = AcousticDecoder(network: network, vocabulary: vocab)
        let workspace = AcousticWorkspace(
            maxHiddenDim: network.maxHiddenDim,
            outputDim: network.outputDim,
            inputDim: network.inputDim
        )

        // 10フレームのダミー音響特徴量
        var featSeq = [[Float]]()
        var f = 0
        while f < 10 {
            var frame = [Float](repeating: 0.0, count: 32)
            var c = 0
            while c < 32 {
                frame[c] = sin(Float(f * 32 + c) * 0.1) * 0.5 + 0.5
                c += 1
            }
            featSeq.append(frame)
            f += 1
        }

        let logProbs = decoder.decodeLogProbabilities(featuresSeq: featSeq, workspace: workspace)
        XCTAssertEqual(logProbs.count, 10)
        XCTAssertEqual(logProbs[0].count, vocab.size)

        // 各フレームで確率最大要素が 0.0 以下 (対数確率) であること
        for lp in logProbs {
            for val in lp {
                XCTAssertTrue(val <= 0.0, "対数確率は 0.0 以下であること")
                XCTAssertTrue(val.isFinite, "対数確率は有限値であること")
            }
        }

        let nBest = decoder.decodeNBest(
            featuresSeq: featSeq,
            workspace: workspace,
            beamWidth: 8,
            n: 3
        )
        XCTAssertTrue(nBest.count <= 3)
    }

    // MARK: - 2. 言語モデル・文脈スコアラー検証

    /// LanguageModelScorer プロトコルおよび CharLanguageModel, StatisticalNGramScorer の動作検証
    func testLanguageModelScorerImplementations() {
        let corpus = [
            "きしゃのきしゃ",
            "きしゃのかいけん",
            "きしゃがはしる",
            "いちといっち",
            "いちのばしょ",
            "いっちするいけん"
        ]
        let dict = KanaKanjiDictionary()
        dict.buildFromCorpus(rawTexts: [
            "貴社の記者",
            "記者の会見",
            "汽車が走る",
            "位置と一致",
            "位置の場所",
            "一致する意見"
        ])

        // 1. StatisticalNGramScorer の検証
        let statisticalScorer = dict.makeContextScorer()
        let pFrequent = statisticalScorer.logProbability(of: "貴社の記者")
        let pUnseen = statisticalScorer.logProbability(of: "貴社の走る")

        XCTAssertTrue(pFrequent.isFinite)
        XCTAssertTrue(pUnseen.isFinite)
        XCTAssertTrue(pUnseen < pFrequent, "頻出連接 (貴社の記者) の対数確率が未観測連接 (貴社の走る) より高いこと")

        // 空文字列境界値
        XCTAssertEqual(statisticalScorer.logProbability(of: ""), 0.0)

        // 2. CharLanguageModel (LanguageModelScorer 適合) の検証
        let kanjiVocab = TextVocabulary(corpus: corpus)
        let charLM = CharLanguageModel.make(vocabulary: kanjiVocab, hiddenDim: 32, timeSteps: 2)

        let lmScore = charLM.logProbability(of: "きしゃ")
        XCTAssertTrue(lmScore.isFinite)
        XCTAssertEqual(charLM.logProbability(of: ""), 0.0)
    }

    // MARK: - 3. セカンドパス・リスコアリング計算式検証

    /// S(Y, X) = S_acoustic(X) + \lambda_{lex} S_{lexical}(Y|X) + \lambda_{LM} S_{LM}(Y) + LenBonus * |Y| の厳密一致検証
    func testKanaKanjiDecoderRescoreFormulaEquivalence() {
        let dict = KanaKanjiDictionary()
        dict.addEntry(KanaKanjiEntry(reading: "きしゃ", surface: "貴社", frequency: 10))
        dict.addEntry(KanaKanjiEntry(reading: "きしゃ", surface: "記者", frequency: 8))
        dict.addEntry(KanaKanjiEntry(reading: "の", surface: "の", frequency: 20, isParticle: true))
        dict.addBigram(from: "貴社", to: "の")
        dict.addBigram(from: "の", to: "記者")

        let lmScorer = dict.makeContextScorer(wordWeight: 1.0)
        let decoder = KanaKanjiDecoder(
            dictionary: dict,
            languageModel: lmScorer,
            lexicalWeight: 1.5,
            lmWeight: 0.8,
            lengthBonus: 0.2
        )

        let hyp = AcousticHypothesis(
            text: "きしゃのきしゃ",
            tokens: [4, 5, 6],
            acousticScore: -1.25,
            score: -1.25
        )

        let rescoredList = decoder.rescoreNBest(
            hypotheses: [hyp],
            lexicalWeight: 1.5,
            lmWeight: 0.8,
            lengthBonus: 0.2,
            kanjiBeamWidth: 5
        )

        XCTAssertTrue(rescoredList.isEmpty != true, "リスコアリング候補が生成されること")

        for item in rescoredList {
            let expectedLenBonus = Float(item.text.count) * 0.2
            let expectedTotal = item.acousticScore + (1.5 * item.lexicalScore) + (0.8 * item.lmScore) + expectedLenBonus

            XCTAssertEqual(item.acousticScore, -1.25, accuracy: 1e-5)
            XCTAssertEqual(item.lengthBonus, expectedLenBonus, accuracy: 1e-5)
            XCTAssertEqual(item.totalScore, expectedTotal, accuracy: 1e-4, "計算式 S(Y, X) と各項の積算が厳密に一致すること")
        }

        // スコア降順の検証
        var r = 1
        while r < rescoredList.count {
            XCTAssertTrue(rescoredList[r].totalScore <= rescoredList[r - 1].totalScore)
            r += 1
        }
    }

    // MARK: - 4. 会議・日常会話同音異義語解決検証

    /// 「貴社の記者」や「位置と一致」など音響的に曖昧な同音異義語が文脈連接と LM で正しく選択される検証
    func testHomophoneResolutionWithLMRescoring() {
        let dict = KanaKanjiDictionary()
        // ビジネス会話コーパス
        dict.buildFromCorpus(rawTexts: [
            "貴社の記者会見に出席する",
            "貴社の記者から質問を受けました",
            "本日の記者会見",
            "自動車の位置と一致する情報",
            "現在位置と一致した",
            "位置の確認を行う"
        ])

        let lmScorer = dict.makeContextScorer(wordWeight: 1.2)
        let decoder = KanaKanjiDecoder(
            dictionary: dict,
            languageModel: lmScorer,
            lexicalWeight: 1.0,
            lmWeight: 0.6,
            lengthBonus: 0.0
        )

        // 1. 同音異義語: 「きしゃのきしゃ」 -> 「貴社の記者」
        let kishaHyp = AcousticHypothesis(
            text: "きしゃのきしゃ",
            tokens: [],
            acousticScore: -0.5,
            score: -0.5
        )
        let kishaResult = decoder.decode(acousticHypotheses: [kishaHyp])
        XCTAssertEqual(kishaResult, "貴社の記者", "文脈連接と LM により「貴社の記者」が選択されること")

        // 2. 同音異義語: 「いちといっち」 -> 「位置と一致」
        let ichiHyp = AcousticHypothesis(
            text: "いちといっち",
            tokens: [],
            acousticScore: -0.4,
            score: -0.4
        )
        let ichiResult = decoder.decode(acousticHypotheses: [ichiHyp])
        XCTAssertEqual(ichiResult, "位置と一致", "文脈連接と LM により「位置と一致」が選択されること")
    }

    // MARK: - 5. 濁音・促音ブレの文脈補正検証

    /// 音響認識で生じた濁音ブレ (ぎしゃ -> 貴社) や促音ブレ (いちといち -> 位置と一致) の文脈補正検証
    func testVoicingAndGeminateBlurCorrection() {
        let dict = KanaKanjiDictionary()
        dict.buildFromCorpus(rawTexts: [
            "貴社の記者に回答する",
            "位置と一致した結果",
            "結果の一致を確認する"
        ])

        let lmScorer = dict.makeContextScorer(wordWeight: 1.2)
        let decoder = KanaKanjiDecoder(
            dictionary: dict,
            languageModel: lmScorer,
            lexicalWeight: 1.0,
            lmWeight: 0.8
        )

        // ケース A: 音響第1位に濁音ブレがあるが、N-best 第2位に正しいかながある場合
        // 第1位: "ぎしゃのきしゃ" (音響スコア -0.30)
        // 第2位: "きしゃのきしゃ" (音響スコア -0.45)
        let hypsA = [
            AcousticHypothesis(text: "ぎしゃのきしゃ", tokens: [], acousticScore: -0.30, score: -0.30),
            AcousticHypothesis(text: "きしゃのきしゃ", tokens: [], acousticScore: -0.45, score: -0.45)
        ]

        let resultA = decoder.decode(acousticHypotheses: hypsA)
        XCTAssertEqual(resultA, "貴社の記者", "濁音ブレの音響第1位を退け、文脈整合性の高い第2位から「貴社の記者」が選択されること")

        // ケース B: 音響第1位に促音落ち (促音ブレ) があるが、N-best 第2位に正しい促音がある場合
        // 第1位: "いちといち" (音響スコア -0.20)
        // 第2位: "いちといっち" (音響スコア -0.50)
        let hypsB = [
            AcousticHypothesis(text: "いちといち", tokens: [], acousticScore: -0.20, score: -0.20),
            AcousticHypothesis(text: "いちといっち", tokens: [], acousticScore: -0.50, score: -0.50)
        ]

        let resultB = decoder.decode(acousticHypotheses: hypsB)
        XCTAssertEqual(resultB, "位置と一致", "促音ブレの音響第1位を退け、文脈整合性の高い第2位から「位置と一致」が選択されること")

        // ケース C: N-best が音響ブレ単独 ("ぎしゃのきしゃ") のみの場合でも調音ファジー検索と文脈で補正
        let hypsC = [
            AcousticHypothesis(text: "ぎしゃのきしゃ", tokens: [], acousticScore: -0.50, score: -0.50)
        ]
        let resultC = decoder.decode(acousticHypotheses: hypsC)
        XCTAssertEqual(resultC, "貴社の記者", "単独の濁音ブレ入力でも調音ファジー検索と連接確率で「貴社の記者」に補正されること")
    }

    // MARK: - 6. StreamingTranscriber 統合検証

    /// StreamingTranscriber に KanaKanjiDecoder を統合したストリーミング N-best リスコアリング検証
    func testStreamingTranscriberWithKanaKanjiDecoderNBest() {
        let textCorpus = [
            "テストのけっか",
            "にんしきのせいど"
        ]
        let kanjiCorpus = [
            "テストの結果",
            "認識の精度"
        ]

        let vocab = TextVocabulary(corpus: textCorpus)
        let dict = KanaKanjiDictionary()
        dict.buildFromCorpus(rawTexts: kanjiCorpus)

        let acousticNet = SpikingNetwork(
            inputDim: 32,
            maxHiddenDim: 64,
            outputDim: vocab.size,
            timeSteps: 2
        )
        let languageNet = SpikingNetwork(
            inputDim: vocab.size,
            maxHiddenDim: 64,
            outputDim: vocab.size,
            timeSteps: 2
        )

        let kkDecoder = KanaKanjiDecoder(
            dictionary: dict,
            languageModel: dict.makeContextScorer(),
            languageBonus: 0.0,
            lexicalWeight: 1.0,
            lmWeight: 0.3
        )

        let transcriber = StreamingTranscriber(
            config: StreamingTranscriberConfig(beamWidth: 4),
            acousticNetwork: acousticNet,
            languageNetwork: languageNet,
            textVocabulary: vocab,
            kanaKanjiDecoder: kkDecoder
        )

        let collector = ResultCollector<TranscriptionResult>()
        transcriber.onFinalResult = { res in
            collector.append(res)
        }

        // 1秒間の有声 PCM (440Hz 正弦波) を入力
        let sampleRate = 16000
        var pcm = [Float](repeating: 0.0, count: sampleRate)
        var i = 0
        while i < sampleRate {
            pcm[i] = sin(Float(i) * 2.0 * .pi * 440.0 / Float(sampleRate)) * 0.5
            i += 1
        }

        transcriber.appendAudio(pcm: pcm)

        // 0.3秒間の無音を入力してセグメント確定をトリガー
        let silence = [Float](repeating: 0.0, count: 4800)
        transcriber.appendAudio(pcm: silence)
        transcriber.flush()

        XCTAssertTrue(0 < collector.count, "発話区間終了により finalResult コールバックが発火すること")
        if 0 < collector.count {
            let firstRes = collector.allItems[0]
            XCTAssertTrue(firstRes.isFinal)
            XCTAssertTrue(0.0 <= firstRes.startTimeSeconds)
            XCTAssertTrue(firstRes.startTimeSeconds <= firstRes.endTimeSeconds)
        }
    }

    // MARK: - 7. 超高速スループット & RTF < 0.005 検証

    /// N-best リスコアリング処理のリアルタイム係数 (RTF) が 0.005 未満であることを実測検証
    func testThroughputAndRTFPerformance() {
        let dict = KanaKanjiDictionary()
        var sentences: [String] = []
        var s = 0
        while s < 50 {
            sentences.append("貴社の記者会見に出席して現在位置と一致する情報を確認しました")
            s += 1
        }
        dict.buildFromCorpus(rawTexts: sentences)

        let decoder = KanaKanjiDecoder(
            dictionary: dict,
            languageModel: dict.makeContextScorer(),
            lexicalWeight: 1.0,
            lmWeight: 0.3
        )

        let sampleHyps = [
            AcousticHypothesis(text: "きしゃのきしゃかいけんにしゅっせきして", tokens: [], acousticScore: -1.0, score: -1.0),
            AcousticHypothesis(text: "ぎしゃのきしゃかいけんにしゅっせきして", tokens: [], acousticScore: -1.2, score: -1.2),
            AcousticHypothesis(text: "きしゃのぎしゃかいけんにしゅっせきして", tokens: [], acousticScore: -1.5, score: -1.5),
            AcousticHypothesis(text: "きしゃのきしゃかいけんにすっせきして", tokens: [], acousticScore: -1.8, score: -1.8),
            AcousticHypothesis(text: "きしゃのきしゃかいけんにしゅせきして", tokens: [], acousticScore: -2.0, score: -2.0)
        ]

        // 3秒相当の発話テキストに対するリスコアリングを 100 回実行
        let iterations = 100
        let audioDurationSec: Double = 3.0 * Double(iterations)

        let startTime = CFAbsoluteTimeGetCurrent()
        var iter = 0
        while iter < iterations {
            let res = decoder.decode(acousticHypotheses: sampleHyps)
            XCTAssertTrue(res.isEmpty != true)
            iter += 1
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        let rtf = elapsed / audioDurationSec

        XCTAssertTrue(rtf < 0.005, "N-best リスコアリングの RTF は 0.005 未満であること (実測: \(String(format: "%.6f", rtf)))")
    }

    // MARK: - 8. 無音・空仮説での幻覚抑止検証

    /// CTC の Blank（無音）仮説が最上位の場合に、下位のノイズ仮説を拾って誤発話（幻覚）を出力しない検証
    func testSilenceHypothesisDoesNotHallucinate() {
        let dict = KanaKanjiDictionary()
        dict.buildFromCorpus(rawTexts: ["テスト", "音声認識"])
        let decoder = KanaKanjiDecoder(
            dictionary: dict,
            languageModel: dict.makeContextScorer(),
            lexicalWeight: 1.0,
            lmWeight: 0.3
        )

        // 最上位: 無音（score: -0.01）
        // 次点: 微小ノイズからの誤検出（score: -15.0）
        let hyps = [
            AcousticHypothesis(text: "", tokens: [0], acousticScore: -0.01, score: -0.01),
            AcousticHypothesis(text: "てすと", tokens: [4, 5, 6], acousticScore: -15.0, score: -15.0)
        ]

        let rescored = decoder.rescoreNBest(hypotheses: hyps)
        XCTAssertTrue(rescored.isEmpty != true, "無音候補を含むリスコアリング結果が生成されること")
        switch rescored.first {
        case .some(let best):
            XCTAssertEqual(best.text, "", "最上位の無音仮説が維持され、誤検出テキストを出力しないこと")
            XCTAssertEqual(best.tokens, [0], "無音仮説のトークン ID が保持されること")
        case .none:
            XCTFail("候補が返されること")
        }

        let decodedText = decoder.decode(acousticHypotheses: hyps)
        XCTAssertEqual(decodedText, "", "decode() が空文字列を返すこと")
    }

    // MARK: - 9. 句読点を含む文脈スコアリングの頑健性検証

    /// 句読点（。、）を含むテキストでも辞書外未知語ペナルティを受けず、文脈連接が正しく評価される検証
    func testStatisticalNGramScorerPunctuationRobustness() {
        let dict = KanaKanjiDictionary()
        dict.buildFromCorpus(rawTexts: [
            "貴社の記者会見に出席する",
            "本日の記者会見"
        ])
        let scorer = dict.makeContextScorer()

        let pWithoutPunct = scorer.logProbability(of: "貴社の記者会見")
        let pWithPunct = scorer.logProbability(of: "貴社の記者会見。")

        XCTAssertTrue(pWithoutPunct.isFinite)
        XCTAssertTrue(pWithPunct.isFinite)
        XCTAssertEqual(pWithoutPunct, pWithPunct, accuracy: 1e-4, "句読点が付与されても単語連接確率は同一に保たれること")
    }

    // MARK: - 10. トークン ID 整合性検証

    /// リスコアリングによって第2位仮説が逆転勝利した際、トークン ID が正しく勝者の系列を反映している検証
    func testRescoreNBestTokenIdIntegrityOnRankInversion() {
        let dict = KanaKanjiDictionary()
        dict.buildFromCorpus(rawTexts: [
            "貴社の記者に連絡する"
        ])
        let decoder = KanaKanjiDecoder(
            dictionary: dict,
            languageModel: dict.makeContextScorer(),
            lexicalWeight: 1.0,
            lmWeight: 1.0
        )

        // 第1位: ぎしゃ (tokens: [10, 20]), score: -0.30
        // 第2位: きしゃ (tokens: [5, 20]), score: -0.45
        let hyps = [
            AcousticHypothesis(text: "ぎしゃのきしゃ", tokens: [10, 20, 30, 10, 20], acousticScore: -0.30, score: -0.30),
            AcousticHypothesis(text: "きしゃのきしゃ", tokens: [5, 20, 30, 5, 20], acousticScore: -0.45, score: -0.45)
        ]

        let rescored = decoder.rescoreNBest(hypotheses: hyps)
        switch rescored.first {
        case .some(let best):
            XCTAssertEqual(best.text, "貴社の記者")
            XCTAssertEqual(best.tokens, [5, 20, 30, 5, 20], "勝者となった第2仮説のトークン ID 系列が反映されること")
        case .none:
            XCTFail("最良候補が存在すること")
        }
    }
}
