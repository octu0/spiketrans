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
    public let betaFast: Float // 高周波用減衰率

    public let wIn: [Float]    // [maxHiddenDim * inputDim]
    public let wRec: [Float]   // [maxHiddenDim * maxHiddenDim]
    public let bH: [Float]     // [maxHiddenDim]
    public let wOut: [Float]   // [outputDim * maxHiddenDim]
    public let bOut: [Float]   // [outputDim]

    enum CodingKeys: String, CodingKey {
        case inputDim, maxHiddenDim, outputDim, timeSteps
        case beta, vTh, vReset, alpha, rho, gamma, betaFast
        case wIn, wRec, bH, wOut, bOut
    }

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
        betaFast: Float = 0.0,
        wIn: [Float],
        wRec: [Float],
        bH: [Float],
        wOut: [Float],
        bOut: [Float]
    ) {
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
        self.betaFast = betaFast
        self.wIn = wIn
        self.wRec = wRec
        self.bH = bH
        self.wOut = wOut
        self.bOut = bOut
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
        self.rho = try container.decodeIfPresent(Float.self, forKey: .rho) ?? 0.85
        self.gamma = try container.decodeIfPresent(Float.self, forKey: .gamma) ?? 0.0
        self.betaFast = try container.decodeIfPresent(Float.self, forKey: .betaFast) ?? 0.0
        self.wIn = try container.decode([Float].self, forKey: .wIn)
        self.wRec = try container.decode([Float].self, forKey: .wRec)
        self.bH = try container.decode([Float].self, forKey: .bH)
        self.wOut = try container.decode([Float].self, forKey: .wOut)
        self.bOut = try container.decode([Float].self, forKey: .bOut)
    }

    /// LIFConfig から直接構築
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
        bOut: [Float]
    ) {
        self.init(
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
            betaFast: lifConfig.betaFast,
            wIn: wIn,
            wRec: wRec,
            bH: bH,
            wOut: wOut,
            bOut: bOut
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
            gamma: gamma,
            betaFast: betaFast
        )
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
