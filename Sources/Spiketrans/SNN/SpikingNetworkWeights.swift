import Foundation

/// SNN 重みのシリアライゼーションコンテナ (JSON)
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
        self.wOut = wOut
        self.bOut = bOut
        self.vocabularyCharacters = vocabularyCharacters
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

    /// ファイルに保存
    public func save(to url: URL) throws {
        let encoder = JSONEncoder()
        let data = try encoder.encode(self)
        try data.write(to: url, options: .atomic)
    }

    /// ファイルから読み込み
    public static func load(from url: URL) throws -> SpikingNetworkWeights {
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        return try decoder.decode(SpikingNetworkWeights.self, from: data)
    }
}
