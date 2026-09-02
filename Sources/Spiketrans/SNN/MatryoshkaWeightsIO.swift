import Foundation

/// マトリョーシカ SNN 全体重みシリアライゼーションコンテナ (JSON / バイナリ対応)
///
/// version 1: beta / vTh / vReset / alpha のみを保持 (ALIF 導入前)
/// version 2: 適応型発火閾値 (ALIF) の rho / gamma を追加
/// version 3: スライス別出力バイアス (bOutBase / bOutMiddle) を追加
public struct MatryoshkaWeightsData: Sendable, Codable, Equatable {
    /// 現行フォーマットバージョン
    public static let currentVersion: Int = 3

    public let version: Int
    public let inputDim: Int
    public let maxHiddenDim: Int
    public let outputDim: Int
    public let timeSteps: Int
    public let beta: Float
    public let vTh: Float
    public let vReset: Float
    public let alpha: Float
    public let rho: Float      // 適応閾値減衰率 (version 1 の重みでは 0.85 とみなす)
    public let gamma: Float    // 発火時閾値上昇幅 (version 1 の重みでは 0.0 = 固定閾値)

    public let wIn: [Float]    // [maxHiddenDim * inputDim]
    public let wRec: [Float]   // [maxHiddenDim * maxHiddenDim]
    public let bH: [Float]     // [maxHiddenDim]
    public let wOut: [Float]   // [outputDim * maxHiddenDim]
    public let bOut: [Float]   // [outputDim] (High スライス用)
    // スライスごとに blank の閾値を独立に較正できるようにするための出力バイアス。
    // 共有すると sliceNorm がバイアスに掛からず、小さいスライスほど非 blank 側が
    // 相対的に持ち上がって挿入過多になる。version 2 以前の重みでは bOut を複製する。
    public let bOutBase: [Float]     // [outputDim]
    public let bOutMiddle: [Float]   // [outputDim]

    public init(
        version: Int = MatryoshkaWeightsData.currentVersion,
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
        bOutBase: [Float]? = nil,
        bOutMiddle: [Float]? = nil
    ) {
        self.version = version
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
        self.bOutBase = bOutBase ?? bOut
        self.bOutMiddle = bOutMiddle ?? bOut
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
        bOut: [Float],
        bOutBase: [Float]? = nil,
        bOutMiddle: [Float]? = nil
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
            bOut: bOut,
            bOutBase: bOutBase,
            bOutMiddle: bOutMiddle
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

    /// version 1 (rho / gamma 無し) の JSON も読めるようにするデコーダ
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.version = try c.decode(Int.self, forKey: .version)
        self.inputDim = try c.decode(Int.self, forKey: .inputDim)
        self.maxHiddenDim = try c.decode(Int.self, forKey: .maxHiddenDim)
        self.outputDim = try c.decode(Int.self, forKey: .outputDim)
        self.timeSteps = try c.decode(Int.self, forKey: .timeSteps)
        self.beta = try c.decode(Float.self, forKey: .beta)
        self.vTh = try c.decode(Float.self, forKey: .vTh)
        self.vReset = try c.decode(Float.self, forKey: .vReset)
        self.alpha = try c.decode(Float.self, forKey: .alpha)
        // version 1 の重みには rho / gamma が存在しないため固定閾値相当にフォールバック
        self.rho = try c.decodeIfPresent(Float.self, forKey: .rho) ?? 0.85
        self.gamma = try c.decodeIfPresent(Float.self, forKey: .gamma) ?? 0.0
        self.wIn = try c.decode([Float].self, forKey: .wIn)
        self.wRec = try c.decode([Float].self, forKey: .wRec)
        self.bH = try c.decode([Float].self, forKey: .bH)
        self.wOut = try c.decode([Float].self, forKey: .wOut)
        let bOutDecoded = try c.decode([Float].self, forKey: .bOut)
        self.bOut = bOutDecoded
        // version 2 以前はスライス別バイアスが無いので High のバイアスを複製する
        self.bOutBase = try c.decodeIfPresent([Float].self, forKey: .bOutBase) ?? bOutDecoded
        self.bOutMiddle = try c.decodeIfPresent([Float].self, forKey: .bOutMiddle) ?? bOutDecoded
    }

    /// ファイルに保存
    public func save(to url: URL) throws {
        let encoder = JSONEncoder()
        let data = try encoder.encode(self)
        try data.write(to: url, options: .atomic)
    }

    /// ファイルから読み込み
    public static func load(from url: URL) throws -> MatryoshkaWeightsData {
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        return try decoder.decode(MatryoshkaWeightsData.self, from: data)
    }
}
