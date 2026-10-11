import Foundation

/// Hawk 型 (Griffin の再帰だけの構成、De et al. 2024) の音響モデルの重み。SNN との比較用。
/// 1 層は「RMSNorm → 再帰ブロック → 残差」と「RMSNorm → ゲート付き MLP → 残差」。
/// 再帰ブロックは GeLU の枝と、線形 → 因果畳み込み → RG-LRU の枝の積を出力の線形で戻す。
/// 行列はすべて [入力, 出力] の行優先 (x · W の形)
public struct HawkWeights: Sendable, Codable, Equatable {
    public let inputDim: Int
    public let modelDim: Int
    /// 再帰ブロックの幅
    public let rnnDim: Int
    /// MLP の中間の幅
    public let mlpDim: Int
    /// RG-LRU のゲートのブロック対角の数
    public let heads: Int
    public let convWidth: Int
    public let outputDim: Int

    public let wIn: [Float]          // [inputDim, modelDim]
    /// 入力の線形の偏り。無音 (特徴量がすべて 0) のフレームで残差の流れが 0 になると、
    /// 各層の RMSNorm の逆伝播が 1/√eps 倍ずつ勾配を増やして 8 層であふれるため、流れに下駄をはかせる
    public let bIn: [Float]          // [modelDim]
    public let layers: [HawkLayerWeights]
    public let finalNorm: [Float]    // [modelDim]
    public let wOut: [Float]         // [modelDim, outputDim]
    public let bOut: [Float]         // [outputDim]
    public let vocabularyCharacters: String?

    /// RG-LRU の減衰の指数 c (a_t = σ(Λ)^(c r_t))。ゲート付き記憶と同じ値
    public static let decayExponent: Float = SpikingNetworkWeights.gateDecayExponent

    public init(
        inputDim: Int,
        modelDim: Int,
        rnnDim: Int,
        mlpDim: Int,
        heads: Int,
        convWidth: Int,
        outputDim: Int,
        wIn: [Float],
        bIn: [Float],
        layers: [HawkLayerWeights],
        finalNorm: [Float],
        wOut: [Float],
        bOut: [Float],
        vocabularyCharacters: String? = nil
    ) {
        self.inputDim = inputDim
        self.modelDim = modelDim
        self.rnnDim = rnnDim
        self.mlpDim = mlpDim
        self.heads = heads
        self.convWidth = convWidth
        self.outputDim = outputDim
        self.wIn = wIn
        self.bIn = bIn
        self.layers = layers
        self.finalNorm = finalNorm
        self.wOut = wOut
        self.bOut = bOut
        self.vocabularyCharacters = vocabularyCharacters
    }

    public var numLayers: Int {
        return layers.count
    }

    public var headDim: Int {
        return rnnDim / heads
    }

    public var parameterCount: Int {
        var n = wIn.count + bIn.count + finalNorm.count + wOut.count + bOut.count
        for layer in layers {
            n += layer.parameterCount
        }
        return n
    }

    /// 同梱された語彙。無ければ nil
    public var vocabulary: TextVocabulary? {
        guard let chars = vocabularyCharacters, chars.isEmpty != true else {
            return nil
        }
        return TextVocabulary(serializedCharacters: chars)
    }

    /// Λ の初期値。ゲート付き記憶と同じく、ゲートが半開き (r = 0.5) のとき時定数が 1〜100 フレームに広がる
    public static func initialLambda(rnnDim: Int) -> [Float] {
        return Array(SpikingNetworkWeights.initialGateLayers(numLayers: 1, hidden: rnnDim)[0].prefix(rnnDim))
    }

    public func save(to url: URL) throws {
        let data = try JSONEncoder().encode(self)
        try data.write(to: url, options: .atomic)
    }

    public static func load(from url: URL) throws -> HawkWeights {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(HawkWeights.self, from: data)
    }
}

/// Hawk 型の 1 層
public struct HawkLayerWeights: Sendable, Codable, Equatable {
    public let norm1: [Float]       // [modelDim]
    public let wGate: [Float]       // [modelDim, rnnDim] GeLU の枝
    public let wX: [Float]          // [modelDim, rnnDim] 再帰の枝
    public let convW: [Float]       // [convWidth, rnnDim] 因果畳み込み (最後の行が現在のフレーム)
    public let convB: [Float]       // [rnnDim]
    public let wA: [Float]          // [heads, headDim, headDim] 残す割合のゲート (ブロック対角)
    public let bA: [Float]          // [rnnDim]
    public let wI: [Float]          // [heads, headDim, headDim] 取り込むゲート
    public let bI: [Float]          // [rnnDim]
    public let lambda: [Float]      // [rnnDim] 減衰率 σ(Λ) の logit
    public let wO: [Float]          // [rnnDim, modelDim]
    public let norm2: [Float]       // [modelDim]
    public let wMLPGate: [Float]    // [modelDim, mlpDim]
    public let wMLPUp: [Float]      // [modelDim, mlpDim]
    public let wMLPDown: [Float]    // [mlpDim, modelDim]

    public init(
        norm1: [Float], wGate: [Float], wX: [Float], convW: [Float], convB: [Float],
        wA: [Float], bA: [Float], wI: [Float], bI: [Float], lambda: [Float], wO: [Float],
        norm2: [Float], wMLPGate: [Float], wMLPUp: [Float], wMLPDown: [Float]
    ) {
        self.norm1 = norm1
        self.wGate = wGate
        self.wX = wX
        self.convW = convW
        self.convB = convB
        self.wA = wA
        self.bA = bA
        self.wI = wI
        self.bI = bI
        self.lambda = lambda
        self.wO = wO
        self.norm2 = norm2
        self.wMLPGate = wMLPGate
        self.wMLPUp = wMLPUp
        self.wMLPDown = wMLPDown
    }

    public var parameterCount: Int {
        return norm1.count + wGate.count + wX.count + convW.count + convB.count + wA.count + bA.count
            + wI.count + bI.count + lambda.count + wO.count + norm2.count + wMLPGate.count + wMLPUp.count + wMLPDown.count
    }
}
