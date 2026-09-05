import Foundation

/// SNN 重みのシリアライゼーションコンテナ (JSON)
public struct SpikingNetworkWeights: Sendable, Codable, Equatable {
    public let numLayers: Int
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

    public let wIn: [Float]    // [maxHiddenDim * inputDim]
    public let wRec: [Float]   // [maxHiddenDim * maxHiddenDim]
    public let bH: [Float]     // [maxHiddenDim]

    /// 上位層（Layer 1 以降）の結合重み・バイアス・電流RMSNormゲイン
    public let wLayers: [[Float]]?   // [numLayers - 1][maxHiddenDim * maxHiddenDim]
    public let bHLayers: [[Float]]?  // [numLayers - 1][maxHiddenDim]
    public let gammaRMS: [[Float]]?  // [numLayers - 1][maxHiddenDim]

    public let wOut: [Float]   // [outputDim * maxHiddenDim]
    public let bOut: [Float]   // [outputDim]

    /// 出力層の ID 割当に対応する文字列 (特殊トークンを除く)。
    /// 言語モデルなど語彙を持たないネットワークでは nil になる
    public let vocabularyCharacters: String?

    /// 2D-Conv Subsampling フロントエンド重み (Phase 2)
    public let convSubsampling: Conv2DSubsamplingWeights?

    public init(
        numLayers: Int = 1,
        inputDim: Int,
        maxHiddenDim: Int,
        outputDim: Int,
        timeSteps: Int,
        beta: Float,
        vTh: Float,
        vReset: Float,
        alpha: Float,
        rho: Float = 0.85,
        gamma: Float = 0.0,
        wIn: [Float],
        wRec: [Float],
        bH: [Float],
        wLayers: [[Float]]? = nil,
        bHLayers: [[Float]]? = nil,
        gammaRMS: [[Float]]? = nil,
        wOut: [Float],
        bOut: [Float],
        vocabularyCharacters: String? = nil,
        convSubsampling: Conv2DSubsamplingWeights? = nil
    ) {
        self.numLayers = max(1, numLayers)
        self.inputDim = inputDim
        self.maxHiddenDim = maxHiddenDim
        self.outputDim = outputDim
        self.timeSteps = timeSteps
        self.beta = beta
        self.vTh = vTh
        self.vReset = vReset
        self.alpha = alpha
        self.rho = rho
        self.gamma = gamma
        self.wIn = wIn
        self.wRec = wRec
        self.bH = bH
        self.wLayers = wLayers
        self.bHLayers = bHLayers
        self.gammaRMS = gammaRMS
        self.wOut = wOut
        self.bOut = bOut
        self.vocabularyCharacters = vocabularyCharacters
        self.convSubsampling = convSubsampling
    }

    /// 単層互換イニシャライザ
    public init(
        inputDim: Int,
        maxHiddenDim: Int,
        outputDim: Int,
        timeSteps: Int,
        beta: Float,
        vTh: Float,
        vReset: Float,
        alpha: Float,
        rho: Float = 0.85,
        gamma: Float = 0.0,
        wIn: [Float],
        wRec: [Float],
        bH: [Float],
        wOut: [Float],
        bOut: [Float],
        vocabularyCharacters: String? = nil,
        convSubsampling: Conv2DSubsamplingWeights? = nil
    ) {
        self.init(
            numLayers: 1,
            inputDim: inputDim,
            maxHiddenDim: maxHiddenDim,
            outputDim: outputDim,
            timeSteps: timeSteps,
            beta: beta,
            vTh: vTh,
            vReset: vReset,
            alpha: alpha,
            rho: rho,
            gamma: gamma,
            wIn: wIn,
            wRec: wRec,
            bH: bH,
            wLayers: nil,
            bHLayers: nil,
            gammaRMS: nil,
            wOut: wOut,
            bOut: bOut,
            vocabularyCharacters: vocabularyCharacters,
            convSubsampling: convSubsampling
        )
    }

    /// 単層 LIFConfig から構築
    public init(
        inputDim: Int,
        maxHiddenDim: Int,
        outputDim: Int,
        timeSteps: Int,
        lifConfig: LIFConfig,
        wIn: [Float],
        wRec: [Float],
        bH: [Float],
        wOut: [Float],
        bOut: [Float],
        vocabularyCharacters: String? = nil,
        convSubsampling: Conv2DSubsamplingWeights? = nil
    ) {
        self.init(
            numLayers: 1,
            inputDim: inputDim,
            maxHiddenDim: maxHiddenDim,
            outputDim: outputDim,
            timeSteps: timeSteps,
            beta: lifConfig.beta,
            vTh: lifConfig.vTh,
            vReset: lifConfig.vReset,
            alpha: lifConfig.alpha,
            rho: lifConfig.rho,
            gamma: lifConfig.gamma,
            wIn: wIn,
            wRec: wRec,
            bH: bH,
            wLayers: nil,
            bHLayers: nil,
            gammaRMS: nil,
            wOut: wOut,
            bOut: bOut,
            vocabularyCharacters: vocabularyCharacters,
            convSubsampling: convSubsampling
        )
    }

    /// 多層 LIFConfig から構築
    public init(
        numLayers: Int,
        inputDim: Int,
        maxHiddenDim: Int,
        outputDim: Int,
        timeSteps: Int,
        lifConfig: LIFConfig,
        wIn: [Float],
        wRec: [Float],
        bH: [Float],
        wLayers: [[Float]]? = nil,
        bHLayers: [[Float]]? = nil,
        gammaRMS: [[Float]]? = nil,
        wOut: [Float],
        bOut: [Float],
        vocabularyCharacters: String? = nil,
        convSubsampling: Conv2DSubsamplingWeights? = nil
    ) {
        self.init(
            numLayers: numLayers,
            inputDim: inputDim,
            maxHiddenDim: maxHiddenDim,
            outputDim: outputDim,
            timeSteps: timeSteps,
            beta: lifConfig.beta,
            vTh: lifConfig.vTh,
            vReset: lifConfig.vReset,
            alpha: lifConfig.alpha,
            rho: lifConfig.rho,
            gamma: lifConfig.gamma,
            wIn: wIn,
            wRec: wRec,
            bH: bH,
            wLayers: wLayers,
            bHLayers: bHLayers,
            gammaRMS: gammaRMS,
            wOut: wOut,
            bOut: bOut,
            vocabularyCharacters: vocabularyCharacters,
            convSubsampling: convSubsampling
        )
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

    private enum CodingKeys: String, CodingKey {
        case numLayers
        case inputDim
        case maxHiddenDim
        case outputDim
        case timeSteps
        case beta
        case vTh
        case vReset
        case alpha
        case rho
        case gamma
        case wIn
        case wRec
        case bH
        case wLayers
        case bHLayers
        case gammaRMS
        case wOut
        case bOut
        case vocabularyCharacters
        case convSubsampling
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.inputDim = try container.decode(Int.self, forKey: .inputDim)
        self.maxHiddenDim = try container.decode(Int.self, forKey: .maxHiddenDim)
        self.outputDim = try container.decode(Int.self, forKey: .outputDim)
        self.timeSteps = try container.decode(Int.self, forKey: .timeSteps)
        self.beta = try container.decode(Float.self, forKey: .beta)
        self.vTh = try container.decode(Float.self, forKey: .vTh)
        self.vReset = try container.decode(Float.self, forKey: .vReset)
        self.alpha = try container.decode(Float.self, forKey: .alpha)
        
        switch try container.decodeIfPresent(Float.self, forKey: .rho) {
        case .some(let val):
            self.rho = val
        case .none:
            self.rho = 0.85
        }
        
        switch try container.decodeIfPresent(Float.self, forKey: .gamma) {
        case .some(let val):
            self.gamma = val
        case .none:
            self.gamma = 0.0
        }

        self.wIn = try container.decode([Float].self, forKey: .wIn)
        self.wRec = try container.decode([Float].self, forKey: .wRec)
        self.bH = try container.decode([Float].self, forKey: .bH)
        self.wLayers = try container.decodeIfPresent([[Float]].self, forKey: .wLayers)
        self.bHLayers = try container.decodeIfPresent([[Float]].self, forKey: .bHLayers)
        self.gammaRMS = try container.decodeIfPresent([[Float]].self, forKey: .gammaRMS)
        self.wOut = try container.decode([Float].self, forKey: .wOut)
        self.bOut = try container.decode([Float].self, forKey: .bOut)
        self.vocabularyCharacters = try container.decodeIfPresent(String.self, forKey: .vocabularyCharacters)
        self.convSubsampling = try container.decodeIfPresent(Conv2DSubsamplingWeights.self, forKey: .convSubsampling)

        switch try container.decodeIfPresent(Int.self, forKey: .numLayers) {
        case .some(let n):
            self.numLayers = n
        case .none:
            switch self.wLayers {
            case .some(let layers):
                self.numLayers = 1 + layers.count
            case .none:
                self.numLayers = 1
            }
        }
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
