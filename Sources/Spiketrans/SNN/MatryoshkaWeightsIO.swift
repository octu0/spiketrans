import Foundation

/// マトリョーシカ SNN 全体重みシリアライゼーションコンテナ (JSON / バイナリ対応)
public struct MatryoshkaWeightsData: Sendable, Codable, Equatable {
    public let version: Int
    public let inputDim: Int
    public let maxHiddenDim: Int
    public let outputDim: Int
    public let timeSteps: Int
    public let beta: Float
    public let vTh: Float
    public let vReset: Float
    public let alpha: Float

    public let wIn: [Float]    // [maxHiddenDim * inputDim]
    public let wRec: [Float]   // [maxHiddenDim * maxHiddenDim]
    public let bH: [Float]     // [maxHiddenDim]
    public let wOut: [Float]   // [outputDim * maxHiddenDim]
    public let bOut: [Float]   // [outputDim]

    public init(
        version: Int = 1,
        inputDim: Int,
        maxHiddenDim: Int,
        outputDim: Int,
        timeSteps: Int,
        beta: Float,
        vTh: Float,
        vReset: Float,
        alpha: Float,
        wIn: [Float],
        wRec: [Float],
        bH: [Float],
        wOut: [Float],
        bOut: [Float]
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
        self.wIn = wIn
        self.wRec = wRec
        self.bH = bH
        self.wOut = wOut
        self.bOut = bOut
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
