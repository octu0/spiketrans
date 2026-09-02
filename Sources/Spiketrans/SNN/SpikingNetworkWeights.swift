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

    public let wIn: [Float]    // [maxHiddenDim * inputDim]
    public let wRec: [Float]   // [maxHiddenDim * maxHiddenDim]
    public let bH: [Float]     // [maxHiddenDim]
    public let wOut: [Float]   // [outputDim * maxHiddenDim]
    public let bOut: [Float]   // [outputDim]

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
        self.wIn = wIn
        self.wRec = wRec
        self.bH = bH
        self.wOut = wOut
        self.bOut = bOut
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
            gamma: gamma
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
