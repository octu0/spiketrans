import Foundation

/// SNN 重みの JSON 表現。語彙文字を同梱できる。
public struct SpikingNetworkWeights: Sendable, Codable, Equatable {
    public let inputDim: Int
    public let maxHiddenDim: Int
    public let outputDim: Int
    public let timeSteps: Int
    public let beta: Float
    public let vTh: Float
    public let vReset: Float
    public let alpha: Float
    public let rho: Float      // 適応閾値減衰率
    public let gamma: Float    // 発火時閾値上昇幅 (0.0 で固定閾値)

    /// 層 0 (再帰 LIF 層)
    public let wIn: [Float]    // [maxHiddenDim * inputDim]
    public let wRec: [Float]   // [maxHiddenDim * maxHiddenDim]
    public let bH: [Float]     // [maxHiddenDim]

    /// 層 1 以降 (前層スパイクを受けるフィードフォワード LIF 層)。
    /// 各層は結合重み・バイアス・電流 RMSNorm ゲインを持つ。1 層構成では空
    public let wLayers: [[Float]]    // [numLayers - 1][maxHiddenDim * maxHiddenDim]
    public let bHLayers: [[Float]]   // [numLayers - 1][maxHiddenDim]
    public let gammaRMS: [[Float]]   // [numLayers - 1][maxHiddenDim]
    /// 層 1 以降の再帰結合 (同じ層の直前サブステップのスパイクから)。上位層に再帰を持たない構成では nil
    public let wRecLayers: [[Float]]?  // [numLayers - 1][maxHiddenDim * maxHiddenDim]
    /// ニューロンごとの膜電位減衰率 (学習する構成のみ。無ければ全ニューロン beta 共通)
    public let betaLayers: [[Float]]?  // [numLayers][maxHiddenDim]
    /// 各層の LIF に入る電流全体を RMSNorm するときのゲイン (無ければ残差をそのまま入れる)
    public let inputNormGains: [[Float]]?  // [numLayers][maxHiddenDim]
    /// 声の種類 (なし / 配信者 / bot / その他) を中間層の発火率から判定する補助ヘッド。学習時だけ使う
    public let wVoice: [Float]?   // [voiceClasses * maxHiddenDim]
    public let bVoice: [Float]?   // [voiceClasses]
    /// 自己教師ありの事前学習 (APC) で、最終層の読み出しから先のフレームの入力特徴量を予測するヘッド。
    /// 予測先が n 個なら出力は n × inputDim。事前学習の重みだけが持つ。CTC の学習を始めるときに捨てる
    public let wPred: [Float]?    // [(n * inputDim) * maxHiddenDim]
    public let bPred: [Float]?    // [n * inputDim]

    public let wOut: [Float]   // [outputDim * maxHiddenDim]
    public let bOut: [Float]   // [outputDim]

    /// 出力層の ID 割当に対応する文字列 (特殊トークンを除く)。
    /// 言語モデルなど語彙を持たないネットワークでは nil になる
    public let vocabularyCharacters: String?

    public init(
        inputDim: Int,
        maxHiddenDim: Int,
        outputDim: Int,
        timeSteps: Int,
        lifConfig: LIFConfig,
        wIn: [Float],
        wRec: [Float],
        bH: [Float],
        wLayers: [[Float]] = [],
        bHLayers: [[Float]] = [],
        gammaRMS: [[Float]] = [],
        wRecLayers: [[Float]]? = nil,
        betaLayers: [[Float]]? = nil,
        inputNormGains: [[Float]]? = nil,
        wVoice: [Float]? = nil,
        bVoice: [Float]? = nil,
        wPred: [Float]? = nil,
        bPred: [Float]? = nil,
        wOut: [Float],
        bOut: [Float],
        vocabularyCharacters: String? = nil
    ) {
        self.inputDim = inputDim
        self.maxHiddenDim = maxHiddenDim
        self.outputDim = outputDim
        self.timeSteps = timeSteps
        self.beta = lifConfig.beta
        self.vTh = lifConfig.vTh
        self.vReset = lifConfig.vReset
        self.alpha = lifConfig.alpha
        self.rho = lifConfig.rho
        self.gamma = lifConfig.gamma
        self.wIn = wIn
        self.wRec = wRec
        self.bH = bH
        self.wLayers = wLayers
        self.bHLayers = bHLayers
        self.gammaRMS = gammaRMS
        self.wRecLayers = wRecLayers
        self.betaLayers = betaLayers
        self.inputNormGains = inputNormGains
        self.wVoice = wVoice
        self.bVoice = bVoice
        self.wPred = wPred
        self.bPred = bPred
        self.wOut = wOut
        self.bOut = bOut
        self.vocabularyCharacters = vocabularyCharacters
    }

    /// 上位層が再帰結合を持つか
    public var hasUpperRecurrence: Bool {
        guard let rec = wRecLayers else {
            return false
        }
        return rec.isEmpty != true
    }

    /// 減衰率を学習する構成の初期値の範囲。ニューロンの番号順に等間隔に並べ、
    /// 速い (0.92: 1 フレームで 0.72 倍) ものから秒単位で保つ (0.995: 1 フレームで 0.98 倍) ものまで混ぜる
    public static let learnedBetaRange: ClosedRange<Float> = 0.92...0.995

    /// 事前学習の予測ヘッドを持つか (= APC で事前学習した重み)
    public var hasPredictionHead: Bool {
        guard let w = wPred, let b = bPred else {
            return false
        }
        return w.isEmpty != true && b.isEmpty != true
    }

    /// 事前学習の既定の予測先 (フレーム数。40 ms × 3 = 120 ms 先)
    public static let predictShift = 3

    /// 予測ヘッドの予測先の数 (出力次元 / inputDim)
    public var predictionTargets: Int {
        guard let b = bPred, 0 < inputDim else {
            return 0
        }
        return b.count / inputDim
    }

    /// 事前学習の重みから CTC の学習を始めるための重み。隠れ層はそのまま、
    /// 読み出し (wOut / bOut / 語彙) は `readout` (新しく作ったネットワーク) のものに替え、予測ヘッドを捨てる。
    /// 事前学習に無い減衰率は `readout` の初期値を使う
    public func startingCTC(readout: SpikingNetworkWeights) -> SpikingNetworkWeights {
        return SpikingNetworkWeights(
            inputDim: inputDim,
            maxHiddenDim: maxHiddenDim,
            outputDim: readout.outputDim,
            timeSteps: timeSteps,
            lifConfig: lifConfig,
            wIn: wIn,
            wRec: wRec,
            bH: bH,
            wLayers: wLayers,
            bHLayers: bHLayers,
            gammaRMS: gammaRMS,
            wRecLayers: wRecLayers,
            betaLayers: betaLayers ?? readout.betaLayers,
            inputNormGains: inputNormGains,
            wVoice: readout.wVoice,
            bVoice: readout.bVoice,
            wOut: readout.wOut,
            bOut: readout.bOut,
            vocabularyCharacters: readout.vocabularyCharacters
        )
    }

    /// 声の種類の補助ヘッドのクラス数 (0 なし / 1 配信者 / 2 bot / 3 その他の声)
    public static let voiceClasses = 4

    /// 声の種類の補助ヘッドを持つか
    public var hasVoiceHead: Bool {
        guard let w = wVoice, let b = bVoice else {
            return false
        }
        return w.isEmpty != true && b.isEmpty != true
    }

    /// 各層の入力電流を RMSNorm するか
    public var hasInputNorm: Bool {
        guard let g = inputNormGains else {
            return false
        }
        return g.isEmpty != true
    }

    /// 減衰率をニューロンごとに持つか
    public var hasLearnedBeta: Bool {
        guard let b = betaLayers else {
            return false
        }
        return b.isEmpty != true
    }

    /// 層数 (層 0 + 上位層)
    public var numLayers: Int {
        return 1 + wLayers.count
    }

    /// 保持している LIF / ALIF パラメータを LIFConfig として復元
    public var lifConfig: LIFConfig {
        return LIFConfig(
            beta: beta,
            vTh: vTh,
            vReset: vReset,
            alpha: alpha,
            rho: rho,
            gamma: gamma
        )
    }

    /// 同梱された語彙。無ければ nil (学習時テキストからの再構築が必要)
    public var vocabulary: TextVocabulary? {
        guard let chars = vocabularyCharacters, chars.isEmpty != true else {
            return nil
        }
        return TextVocabulary(serializedCharacters: chars)
    }

    public func save(to url: URL) throws {
        let encoder = JSONEncoder()
        let data = try encoder.encode(self)
        try data.write(to: url, options: .atomic)
    }

    public static func load(from url: URL) throws -> SpikingNetworkWeights {
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        return try decoder.decode(SpikingNetworkWeights.self, from: data)
    }
}
