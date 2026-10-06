import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// 上位層の電流 RMSNorm で分散 0 のときの除算を避ける微小値 (推論側と共有)
internal let rmsNormEpsilon: Float = 1e-5
/// 入力正規化 (LIF に入る残差電流の RMSNorm) の下限。パディングや発話先頭では電流がちょうど 0 になり、
/// 1e-5 だと勾配が 1/sqrt(eps) = 316 倍に跳ねてゲイン 10 では float32 があふれる (RMS の下限 0.1 に相当)
internal let inputNormEpsilon: Float = 1e-2

/// compile() する系列長の上限 (32 の倍数に切り上げた後のフレーム数)。
/// compile は時間ループを静的に展開するため、1 ステップのピークメモリは系列長と層数に比例する
/// (B=64・T=256 で 4 層 26 GB、5 層 37 GB。8 層なら実メモリ 64 GB を超える)。
/// これを超える系列はチャンク分割の逆伝播で流す。学習中の compile 済み 256 バケットは 7.3 ms/フレームで
/// チャンク分割 (7 ms/フレーム) と同等なので、128 に下げても速度はほぼ変わらずピークだけ下がる
public let compiledMaxFrames = 128

/// MLX 上の切り詰め BPTT。系列が `compiledMaxFrames` を超えると eager。
public final class MLXBPTTTrainer: @unchecked Sendable {
    public let network: MLXSpikingNetwork
    public let optimizer: ArrayLRAdam
    public let config: TrainingConfig
    /// 切り詰め BPTT の窓幅 (フレーム単位)。
    ///
    /// この本数ごとに膜電位・スパイク・適応閾値の勾配を切り離す。1 にすると
    /// フレームをまたぐ信用割り当てが完全に消え、第1段は実質フレーム独立の
    /// 分類器になる。大きくすると時間文脈を学習できる一方、計算グラフが深くなる。
    public private(set) var bpttWindow: Int
    /// 声の種類の補助損失の重み (CTC 損失に足す)
    public var voiceLossWeight: Float = 0.3
    /// 直近のバッチの補助損失 (重みを掛ける前)。補助ヘッドが無ければ 0
    public private(set) var lastVoiceLoss: Float = 0.0

    /// 系列長 (padded maxT) をキーとするコンパイル済み CTC 学習ステップのキャッシュ
    private var compiledCTCSteps: [Int: ([MLXArray]) -> [MLXArray]] = [:]
    private var compiledAPCSteps: [Int: ([MLXArray]) -> [MLXArray]] = [:]
    private var compiledClusterSteps: [Int: ([MLXArray]) -> [MLXArray]] = [:]
    private var compiledTeacherRates: [Int: ([MLXArray]) -> [MLXArray]] = [:]

    /// 系列長 (T) をキーとするコンパイル済み logitsBatch のキャッシュ
    private var compiledLogitsSteps: [Int: ([MLXArray]) -> [MLXArray]] = [:]

    /// (バッチ件数 << 16 | チャンクのフレーム数) をキーとする、チャンク分割の前向き (勾配なし) と逆伝播の
    /// コンパイル済み関数。チャンクの形は 32 単位の 4 種類なので、件数ごとに数回のトレースで済む
    private var compiledChunkForward: [Int: ([MLXArray]) -> [MLXArray]] = [:]
    private var compiledChunkBackward: [Int: ([MLXArray]) -> [MLXArray]] = [:]
    /// `chunkBackward` が返す勾配配列の並び (ModuleParameters の flattened キー)
    private var chunkGradKeys: [String] = []

    /// compiledMaxFrames を超える系列をチャンク分割で学習するときの 1 チャンクのフレーム数
    public let longSequenceChunkFrames = compiledMaxFrames
    /// false にすると長い系列も一括で微分する (比較用)
    public var chunkLongSequences = true

    /// CTC 学習ステップのコンパイル（キャッシュミス）回数
    public private(set) var ctcCompileCount: Int = 0

    /// CTC 学習ステップのキャッシュヒット回数
    public private(set) var ctcCacheHitCount: Int = 0

    /// MLX が使い回しのために抱えるバッファキャッシュの上限。
    /// 系列長バケットごとに別サイズのバッファが要るため、無制限だと約 50 GB まで膨らんで実メモリ 64 GB を使い切る。
    /// 逆に 1 ステップの作業領域 (B=64・T=256 で 15〜27 GB) より小さいと、同じ形が続いても毎ステップ確保し直して
    /// 2 倍遅くなる (256 フレーム: 0.8 → 1.6 秒)。その間に置く
    public static let cacheLimitBytes = 24 << 30

    public init(
        network: MLXSpikingNetwork,
        config: TrainingConfig = TrainingConfig(learningRate: 0.015),
        bpttWindow: Int = 16
    ) {
        self.network = network
        self.config = config
        self.optimizer = ArrayLRAdam(learningRate: config.learningRate)
        self.bpttWindow = max(1, bpttWindow)
        MLX.GPU.set(cacheLimit: Self.cacheLimitBytes)
    }

    /// 学習率を更新。配列で持つので compile 済みステップにも入力として渡る
    public func updateLearningRate(_ lr: Float) {
        self.optimizer.learningRate = MLXArray(lr)
    }

    /// 切り詰め BPTT の窓幅を変える。窓幅はトレース済みのステップに焼き込まれているので作り直す
    public func setBPTTWindow(_ frames: Int) {
        let next = max(1, frames)
        if next == bpttWindow {
            return
        }
        bpttWindow = next
        compiledCTCSteps.removeAll()
        compiledAPCSteps.removeAll()
        compiledClusterSteps.removeAll()
        compiledLogitsSteps.removeAll()
        compiledChunkForward.removeAll()
        compiledChunkBackward.removeAll()
    }

    /// 重みを保存済みの状態へ巻き戻す。Adam のモーメントも捨て、
    /// 古い重みを参照している compile 済みステップも作り直す
    public func rollback(to weights: SpikingNetworkWeights) {
        network.importWeights(from: weights)
        optimizer.resetState()
        compiledCTCSteps.removeAll()
        compiledAPCSteps.removeAll()
        compiledClusterSteps.removeAll()
        compiledLogitsSteps.removeAll()
        compiledChunkForward.removeAll()
        compiledChunkBackward.removeAll()
        eval(network, optimizer)
    }

    /// 1 サブステップ分の全層 LIF 更新。
    /// 入力 [層 0 の入力電流, v_0...v_L-1, s_0...s_L-1, a_0...a_L-1]、出力 [v..., s..., a..., 最終層の読み出し]。
    /// a は適応閾値の状態。ゲート付き記憶の構成では、その層の記憶 (適応閾値は使わない)。
    /// 層 0 は再帰、層 1 以降は前層スパイクの RMSNorm 電流 + 前層電流の残差 (+ 再帰構成なら同じ層の直前スパイクの再帰電流)。
    /// 入力正規化の構成では、各層の LIF に入るのは残差電流全体を RMSNorm してゲインを掛けたもの (残差そのものは正規化しない)。
    /// 最終層だけハードリセットせず、閾値単位の膜電位を読んでから余りを残す
    func substep(network: MLXSpikingNetwork, arrays: [MLXArray], updatesMemory: Bool) -> [MLXArray] {
        let numLayers = network.numLayers
        let beta = network.lifConfig.beta
        let vTh = network.lifConfig.vTh
        let alpha = network.lifConfig.alpha
        let rho = network.lifConfig.rho
        let gamma = network.lifConfig.gamma
        let vMin = LIFNeuronEngine.vClampMin
        let vMax = LIFNeuronEngine.vClampMax
        let readoutK = LIFNeuronEngine.readoutClipInThresholdUnits

        let current0 = arrays[0]
        var v = Array(arrays[1..<(1 + numLayers)])
        var s = Array(arrays[(1 + numLayers)..<(1 + 2 * numLayers)])
        var a = Array(arrays[(1 + 2 * numLayers)..<(1 + 3 * numLayers)])
        var readout = MLXArray.zeros(like: v[0])

        var stream = current0 + matmul(s[0], network.wRec)
        var l = 0
        while l < numLayers {
            // この層が残差に足す電流 (層 0 は入力と再帰の電流そのもの)
            var own = stream
            if 0 < l {
                let upperIdx = l - 1
                let denseCur = matmul(s[l - 1], network.wLayers[upperIdx]) + network.bHLayers[upperIdx]
                // MLXFast.rmsNorm は valueAndGrad + compile で Metal 生存バッファ上限を超える
                let meanSq = mean(denseCur * denseCur, axis: -1, keepDims: true)
                let rms = sqrt(meanSq + rmsNormEpsilon)
                own = (denseCur / rms) * network.gammaRMS[upperIdx]
                if upperIdx < network.wRecLayers.count {
                    own = own + matmul(s[l], network.wRecLayers[upperIdx])
                }
                stream = own + stream
            }
            let gated = l < network.gateParams.count
            if gated {
                // ゲート付き記憶 (RG-LRU 型): 発火でリセットされない連続値の記憶。状態は a[l] に持つ。
                // 1 フレームに 1 回 (最初のサブステップで) 残す割合を入力に応じて決めて更新し、
                // 出力の重みを掛けて毎サブステップ残差に足す
                let g = network.gateParams[l]
                if updatesMemory {
                    let keep = sigmoid(g[1] * own + g[2])
                    let admit = sigmoid(g[3] * own + g[4])
                    let decay = exp((SpikingNetworkWeights.gateDecayExponent * keep) * log(sigmoid(g[0])))
                    a[l] = decay * a[l] + sqrt(1.0 - decay * decay + 1e-6) * (admit * own)
                }
                stream = stream + g[5] * a[l]
            }
            var current = stream
            if l < network.inputNormGains.count {
                let streamMeanSq = mean(stream * stream, axis: -1, keepDims: true)
                current = (stream / sqrt(streamMeanSq + inputNormEpsilon)) * network.inputNormGains[l]
            }

            let isLast = (l + 1) == numLayers
            var decayed = v[l] * beta
            if l < network.betaLogits.count {
                decayed = v[l] * sigmoid(network.betaLogits[l])
            }
            if isLast {
                v[l] = clip(decayed + current, min: vMin, max: vMax)
            } else {
                v[l] = clip(decayed * (1.0 - s[l]) + current, min: vMin, max: vMax)
            }

            var dynVTh = MLXArray(vTh)
            if gated != true {
                a[l] = (a[l] * rho) + (s[l] * gamma)
                dynVTh = vTh + a[l]
            }
            let vRel = (v[l] - dynVTh) * alpha
            let sSurrogate = 0.5 * (vRel / (1.0 + abs(vRel)) + 1.0)
            var sReset = (dynVTh .<= v[l]).asType(.float32)
            if network.continuousSpikes {
                // 閾値の前後 0.5 だけ 0〜1 の連続値、それより下は 0・上は 1 (疎なまま)。勾配は発火と同じ代理勾配
                sReset = clip(v[l] - dynVTh + 0.5, min: 0.0, max: 1.0)
            }
            s[l] = stopGradient(sReset - sSurrogate) + sSurrogate

            if isLast {
                // 読み出しは閾値単位でクリップするが、勾配はクリップ前の値を通す (straight-through)。
                // clip の勾配は |v| >= vTh で 0 になり、発火しているニューロンから勾配が流れず
                // 学習が 4 倍以上遅くなった (JSUT 20ep 未学習 16.7% → 43.4%)
                let scaled = v[l] / vTh
                let clipped = clip(scaled, min: -readoutK, max: readoutK)
                readout = stopGradient(clipped - scaled) + scaled
                v[l] = clip(v[l] - sReset * vTh, min: vMin, max: vMax)
            }
            l += 1
        }
        return v + s + a + [readout]
    }

    /// バッチ（複数発話）に対するフォワードとロジット系列 [B, T, outputDim] の計算
    public func logitsBatch(
        network: MLXSpikingNetwork,
        features: MLXArray,          // [B, T, inputDim]
        compiled: Bool = false
    ) -> MLXArray {
        if compiled {
            let seqLen = features.shape[1]
            if let cached = compiledLogitsSteps[seqLen] {
                return cached([features])[0]
            }
            let fn = compile(inputs: [network]) { arrays in
                return [self.logitsBatch(network: network, features: arrays[0], compiled: false)]
            }
            compiledLogitsSteps[seqLen] = fn
            return fn([features])[0]
        }
        let batchSize = features.shape[0]
        let seqLen = features.shape[1]
        let hMax = network.maxHiddenDim
        let numLayers = network.numLayers

        // 層 0 の入力電流系列: [B, T, hMax] = [B, T, inputDim] @ [inputDim, hMax] + bH
        let currentSeq0 = matmul(features, network.wIn) + network.bH
        var v = [MLXArray](repeating: MLXArray.zeros([batchSize, hMax]), count: numLayers)
        var s = [MLXArray](repeating: MLXArray.zeros([batchSize, hMax]), count: numLayers)
        var a = [MLXArray](repeating: MLXArray.zeros([batchSize, hMax]), count: numLayers)
        var rates: [MLXArray] = []
        let readoutList = forwardFrames(network: network, currentSeq0: currentSeq0, from: 0, to: seqLen, v: &v, s: &s, a: &a, voiceRates: &rates)
        return matmul(stacked(readoutList, axis: 1), network.wOut) + network.bOut
    }

    /// 推論用のロジット系列 [B, T, outputDim]。chunkFrames フレームずつ状態を引き継いで前向きに進め、
    /// 区切りごとに評価してグラフを切る (長い系列を一度に組むと Metal のバッファ数の上限を超える)
    public func logitsInChunks(
        network: MLXSpikingNetwork,
        features: MLXArray,          // [B, T, inputDim]
        chunkFrames: Int = 32
    ) -> MLXArray {
        let batchSize = features.shape[0]
        let seqLen = features.shape[1]
        let hMax = network.maxHiddenDim
        let numLayers = network.numLayers
        var v = [MLXArray](repeating: MLXArray.zeros([batchSize, hMax]), count: numLayers)
        var s = [MLXArray](repeating: MLXArray.zeros([batchSize, hMax]), count: numLayers)
        var a = [MLXArray](repeating: MLXArray.zeros([batchSize, hMax]), count: numLayers)
        var rates: [MLXArray] = []
        var outputs: [MLXArray] = []
        var t0 = 0
        while t0 < seqLen {
            let t1 = min(seqLen, t0 + max(1, chunkFrames))
            let current = matmul(features[0..., t0..<t1, 0...], network.wIn) + network.bH
            let readouts = forwardFrames(network: network, currentSeq0: current, from: 0, to: t1 - t0, v: &v, s: &s, a: &a, voiceRates: &rates)
            let logits = matmul(stacked(readouts, axis: 1), network.wOut) + network.bOut
            rates.removeAll()
            eval(v + s + a + [logits])
            outputs.append(logits)
            t0 = t1
        }
        return concatenated(outputs, axis: 1)
    }

    /// フレーム [t0, t1) を前向きに進め、各フレームの読み出し (サブステップ平均、[B, hMax]) を返す。
    /// 状態 v, s, a は更新して返す。bpttWindow フレームごとに状態の勾配を切り離す (切り詰め BPTT)
    func forwardFrames(
        network: MLXSpikingNetwork,
        currentSeq0: MLXArray,
        from t0: Int,
        to t1: Int,
        v: inout [MLXArray],
        s: inout [MLXArray],
        a: inout [MLXArray],
        voiceRates: inout [MLXArray]
    ) -> [MLXArray] {
        let tSteps = network.timeSteps
        let numLayers = network.numLayers
        let voiceLayer = network.voiceLayer
        let collectVoice = network.voiceHead.isEmpty != true
        var readoutList: [MLXArray] = []
        readoutList.reserveCapacity(t1 - t0)

        var t = t0
        while t < t1 {
            let current0_t = currentSeq0[0..., t, 0...]
            var readoutSum = MLXArray.zeros(like: v[0])
            var voiceSum = MLXArray.zeros(like: v[0])

            if (t % bpttWindow) == 0 {
                var l = 0
                while l < numLayers {
                    v[l] = stopGradient(v[l])
                    s[l] = stopGradient(s[l])
                    a[l] = stopGradient(a[l])
                    l += 1
                }
            }

            var step = 0
            while step < tSteps {
                var inputs: [MLXArray] = [current0_t]
                inputs.append(contentsOf: v)
                inputs.append(contentsOf: s)
                inputs.append(contentsOf: a)
                let out = substep(network: network, arrays: inputs, updatesMemory: step == 0)
                v = Array(out[0..<numLayers])
                s = Array(out[numLayers..<(2 * numLayers)])
                a = Array(out[(2 * numLayers)..<(3 * numLayers)])
                readoutSum = readoutSum + out[3 * numLayers]
                if collectVoice {
                    voiceSum = voiceSum + s[voiceLayer]
                }
                step += 1
            }

            readoutList.append(readoutSum / Float(tSteps))
            if collectVoice {
                voiceRates.append(voiceSum / Float(tSteps))
            }
            t += 1
        }
        return readoutList
    }

    /// 事前学習 (APC) の損失。最終層の読み出しから shifts の各フレーム先の入力特徴量を予測し
    /// (ヘッドの出力を予測先ごとに inputDim ずつ使う)、有効なフレーム (mask [B, T] が 1、かつ t + shift が発話内) の
    /// L1 誤差を予測先ごとに平均してから、予測先の間で平均する
    func predictionLoss(network: MLXSpikingNetwork, features: MLXArray, mask: MLXArray, shifts: [Int]) -> MLXArray {
        let batchSize = features.shape[0]
        let seqLen = features.shape[1]
        let hMax = network.maxHiddenDim
        let numLayers = network.numLayers
        let currentSeq0 = matmul(features, network.wIn) + network.bH
        var v = [MLXArray](repeating: MLXArray.zeros([batchSize, hMax]), count: numLayers)
        var s = [MLXArray](repeating: MLXArray.zeros([batchSize, hMax]), count: numLayers)
        var a = [MLXArray](repeating: MLXArray.zeros([batchSize, hMax]), count: numLayers)
        var rates: [MLXArray] = []
        let readoutList = forwardFrames(network: network, currentSeq0: currentSeq0, from: 0, to: seqLen, v: &v, s: &s, a: &a, voiceRates: &rates)
        let pred = matmul(stacked(readoutList, axis: 1), network.predictionHead[0]) + network.predictionHead[1]
        let inDim = features.shape[2]
        var total = MLXArray(Float(0.0))
        var used = 0
        var k = 0
        while k < shifts.count {
            let shift = shifts[k]
            if shift < seqLen {
                let past = pred[0..., 0..<(seqLen - shift), (k * inDim)..<((k + 1) * inDim)]
                let future = features[0..., shift..<seqLen, 0...]
                let valid = mask[0..., shift..<seqLen].expandedDimensions(axis: -1)
                let count = maximum(sum(valid) * Float(inDim), MLXArray(Float(1.0)))
                total = total + sum(abs(past - future) * valid) / count
                used += 1
            }
            k += 1
        }
        return total / Float(max(1, used))
    }

    /// 事前学習 (HuBERT 型) の損失。入力の隠したフレーム (spanMask が 1) を 0 にして流し、
    /// 隠したフレームのクラス番号 targets [B, T] (Int32) を最終層の読み出しから当てる交差エントロピー。
    /// 因果なので、隠したフレームの番号はそれより前の文脈だけから推測することになる
    func maskedClusterLoss(network: MLXSpikingNetwork, features: MLXArray, valid: MLXArray, spanMask: MLXArray,
                           targets: MLXArray) -> MLXArray {
        let batchSize = features.shape[0]
        let seqLen = features.shape[1]
        let hMax = network.maxHiddenDim
        let numLayers = network.numLayers
        let masked = features * (1.0 - spanMask).expandedDimensions(axis: -1)
        let currentSeq0 = matmul(masked, network.wIn) + network.bH
        var v = [MLXArray](repeating: MLXArray.zeros([batchSize, hMax]), count: numLayers)
        var s = [MLXArray](repeating: MLXArray.zeros([batchSize, hMax]), count: numLayers)
        var a = [MLXArray](repeating: MLXArray.zeros([batchSize, hMax]), count: numLayers)
        var rates: [MLXArray] = []
        let readoutList = forwardFrames(network: network, currentSeq0: currentSeq0, from: 0, to: seqLen, v: &v, s: &s, a: &a, voiceRates: &rates)
        let logits = matmul(stacked(readoutList, axis: 1), network.predictionHead[0]) + network.predictionHead[1]
        let logProbs = logSoftmax(logits, axis: -1)
        let picked = takeAlong(logProbs, targets.expandedDimensions(axis: -1), axis: -1).squeezed(axis: -1)
        let weight = valid * spanMask
        let count = maximum(sum(weight), MLXArray(Float(1.0)))
        return -sum(picked * weight) / count
    }

    /// 層 layer の発火率 (サブステップ平均) [B, T, H]。HuBERT の 2 周目で、先生のモデルから目標を作るのに使う (勾配なし)
    public func layerRates(network: MLXSpikingNetwork, features: MLXArray, layer: Int) -> MLXArray {
        let key = features.shape[0] << 16 | features.shape[1]
        if let cached = compiledTeacherRates[key] {
            return cached([features])[0]
        }
        let fn = compile(inputs: [network]) { arrays in
            return [self.layerRatesEager(network: network, features: arrays[0], layer: layer)]
        }
        compiledTeacherRates[key] = fn
        return fn([features])[0]
    }

    public func layerRatesEager(network: MLXSpikingNetwork, features: MLXArray, layer: Int) -> MLXArray {
        let batchSize = features.shape[0]
        let seqLen = features.shape[1]
        let hMax = network.maxHiddenDim
        let numLayers = network.numLayers
        let tSteps = network.timeSteps
        let currentSeq0 = matmul(features, network.wIn) + network.bH
        var v = [MLXArray](repeating: MLXArray.zeros([batchSize, hMax]), count: numLayers)
        var s = [MLXArray](repeating: MLXArray.zeros([batchSize, hMax]), count: numLayers)
        var a = [MLXArray](repeating: MLXArray.zeros([batchSize, hMax]), count: numLayers)
        var rates: [MLXArray] = []
        var t = 0
        while t < seqLen {
            var sum = MLXArray.zeros([batchSize, hMax])
            var step = 0
            while step < tSteps {
                let out = substep(network: network, arrays: [currentSeq0[0..., t, 0...]] + v + s + a, updatesMemory: step == 0)
                v = Array(out[0..<numLayers])
                s = Array(out[numLayers..<(2 * numLayers)])
                a = Array(out[(2 * numLayers)..<(3 * numLayers)])
                sum = sum + s[layer]
                step += 1
            }
            rates.append(sum / Float(tSteps))
            t += 1
        }
        return stopGradient(stacked(rates, axis: 1))
    }

    /// 事前学習 (HuBERT 型) の 1 ステップ。隠す区間は spanFrames フレーム単位で、各フレームを開始点にする確率 startProb。
    /// 目標は codebook (中心 [mean, invStd, centroids, centroidSq]) に最も近いクラス番号で、
    /// teacher があればその層 teacherLayer の発火率から (2 周目)、無ければ入力の特徴量から (1 周目) 求める。
    /// 系列は 32 フレーム単位に切り上げて 0 で埋め、形ごとに compile する
    @discardableResult
    public func trainBatchMaskedCluster(featuresBatch: [[[Float]]], codebook: [MLXArray],
                                        startProb: Double, spanFrames: Int,
                                        teacher: MLXBPTTTrainer? = nil, teacherLayer: Int = 0) -> Float {
        var maxT = 0
        for f in featuresBatch {
            maxT = max(maxT, f.count)
        }
        if featuresBatch.isEmpty || maxT == 0 {
            return 0.0
        }
        maxT = ((maxT + 31) / 32) * 32
        let bSize = featuresBatch.count
        let inDim = network.inputDim
        var flatFeat = [Float](repeating: 0.0, count: bSize * maxT * inDim)
        var flatValid = [Float](repeating: 0.0, count: bSize * maxT)
        var flatSpan = [Float](repeating: 0.0, count: bSize * maxT)
        var rng = SystemRandomNumberGenerator()
        var b = 0
        while b < bSize {
            let seq = featuresBatch[b]
            var t = 0
            while t < seq.count {
                let offset = ((b * maxT) + t) * inDim
                var d = 0
                while d < inDim {
                    flatFeat[offset + d] = seq[t][d]
                    d += 1
                }
                flatValid[b * maxT + t] = 1.0
                if Double.random(in: 0.0..<1.0, using: &rng) < startProb {
                    var k = 0
                    while k < spanFrames && (t + k) < seq.count {
                        flatSpan[b * maxT + t + k] = 1.0
                        k += 1
                    }
                }
                t += 1
            }
            b += 1
        }
        let featArray = MLXArray(flatFeat, [bSize, maxT, inDim])
        let validArray = MLXArray(flatValid, [bSize, maxT])
        let spanArray = MLXArray(flatSpan, [bSize, maxT])
        var targets: MLXArray
        if let teacher = teacher {
            let rates = teacher.layerRates(network: teacher.network, features: featArray, layer: teacherLayer)
            targets = ClusterCodebook.assign(features: rates, codebook: codebook)
        } else {
            targets = ClusterCodebook.assign(features: featArray, codebook: codebook)
        }
        eval(targets)
        let key = bSize << 16 | maxT
        let stepFn: ([MLXArray]) -> [MLXArray]
        if let cached = compiledClusterSteps[key] {
            stepFn = cached
        } else {
            let lg = valueAndGrad(model: self.network) { (model: MLXSpikingNetwork, arrays: [MLXArray]) -> [MLXArray] in
                return [self.maskedClusterLoss(network: model, features: arrays[0], valid: arrays[1], spanMask: arrays[2],
                                               targets: arrays[3])]
            }
            func step(arrays: [MLXArray]) -> [MLXArray] {
                let (lossValues, grads) = lg(self.network, Array(arrays[0..<4]))
                let (clippedGrads, _) = clipGradNorm(gradients: grads, maxNorm: 5.0)
                let outerLearningRate = self.optimizer.learningRate
                self.optimizer.learningRate = arrays[4]
                self.optimizer.update(model: self.network, gradients: clippedGrads)
                self.optimizer.learningRate = outerLearningRate
                return lossValues
            }
            let newStep = compile(inputs: [self.network, self.optimizer], outputs: [self.network, self.optimizer], step)
            compiledClusterSteps[key] = newStep
            stepFn = newStep
        }
        let lossValues = stepFn([featArray, validArray, spanArray, targets, optimizer.learningRate])
        eval(network, optimizer, lossValues)
        return lossValues[0].item(Float.self)
    }

    /// 事前学習 (APC) の 1 ステップ。系列は 32 フレーム単位に切り上げて 0 で埋め、形ごとに compile する
    @discardableResult
    public func trainBatchAPC(featuresBatch: [[[Float]]], shifts: [Int] = [SpikingNetworkWeights.predictShift]) -> Float {
        var maxT = 0
        for f in featuresBatch {
            maxT = max(maxT, f.count)
        }
        let minShift = shifts.min() ?? SpikingNetworkWeights.predictShift
        if featuresBatch.isEmpty || maxT <= minShift {
            return 0.0
        }
        maxT = ((maxT + 31) / 32) * 32
        let bSize = featuresBatch.count
        let inDim = network.inputDim
        var flatFeat = [Float](repeating: 0.0, count: bSize * maxT * inDim)
        var flatMask = [Float](repeating: 0.0, count: bSize * maxT)
        var b = 0
        while b < bSize {
            let seq = featuresBatch[b]
            var t = 0
            while t < seq.count {
                let offset = ((b * maxT) + t) * inDim
                var d = 0
                while d < inDim {
                    flatFeat[offset + d] = seq[t][d]
                    d += 1
                }
                flatMask[b * maxT + t] = 1.0
                t += 1
            }
            b += 1
        }
        let featArray = MLXArray(flatFeat, [bSize, maxT, inDim])
        let maskArray = MLXArray(flatMask, [bSize, maxT])
        let key = bSize << 16 | maxT
        let stepFn: ([MLXArray]) -> [MLXArray]
        if let cached = compiledAPCSteps[key] {
            stepFn = cached
        } else {
            let lg = valueAndGrad(model: self.network) { (model: MLXSpikingNetwork, arrays: [MLXArray]) -> [MLXArray] in
                return [self.predictionLoss(network: model, features: arrays[0], mask: arrays[1], shifts: shifts)]
            }
            func step(arrays: [MLXArray]) -> [MLXArray] {
                let (lossValues, grads) = lg(self.network, Array(arrays[0..<2]))
                let (clippedGrads, _) = clipGradNorm(gradients: grads, maxNorm: 5.0)
                let outerLearningRate = self.optimizer.learningRate
                self.optimizer.learningRate = arrays[2]
                self.optimizer.update(model: self.network, gradients: clippedGrads)
                self.optimizer.learningRate = outerLearningRate
                return lossValues
            }
            let newStep = compile(inputs: [self.network, self.optimizer], outputs: [self.network, self.optimizer], step)
            compiledAPCSteps[key] = newStep
            stepFn = newStep
        }
        let lossValues = stepFn([featArray, maskArray, optimizer.learningRate])
        eval(network, optimizer, lossValues)
        return lossValues[0].item(Float.self)
    }

    /// 補助ヘッドのロジット [B, T, voiceClasses]。中間層の発火率 (サブステップ平均) の線形読み出し
    func voiceLogits(network: MLXSpikingNetwork, rates: [MLXArray]) -> MLXArray {
        return matmul(stacked(rates, axis: 1), network.voiceHead[0]) + network.voiceHead[1]
    }

    /// 声の種類の交差エントロピー (targets が -1 のフレームは除く。有効フレームの平均)
    public static func voiceLoss(logits: MLXArray, targets: MLXArray) -> MLXArray {
        let mask = (MLXArray(Int32(0)) .<= targets).asType(.float32)
        let safe = maximum(targets, MLXArray(Int32(0))).asType(.int32)
        let logProbs = logSoftmax(logits, axis: -1)
        let picked = takeAlong(logProbs, safe.expandedDimensions(axis: -1), axis: -1).squeezed(axis: -1)
        let count = maximum(sum(mask), MLXArray(Float(1.0)))
        return -sum(picked * mask) / count
    }

    /// voiceLoss のロジットについての勾配 (softmax - onehot をマスクして有効フレーム数で割る)
    static func voiceLossGradient(logits: MLXArray, targets: MLXArray) -> MLXArray {
        let mask = (MLXArray(Int32(0)) .<= targets).asType(.float32)
        let safe = maximum(targets, MLXArray(Int32(0))).asType(.int32)
        let classes = logits.shape[2]
        let probs = softmax(logits, axis: -1)
        let onehot = (safe.expandedDimensions(axis: -1) .== MLXArray(0..<Int32(classes))).asType(.float32)
        let count = maximum(sum(mask), MLXArray(Float(1.0)))
        return (probs - onehot) * mask.expandedDimensions(axis: -1) / count
    }

    /// 全フレームの CTC ロジットと補助ヘッドのロジット (ヘッドが無ければ nil)
    func logitsAndVoiceBatch(network: MLXSpikingNetwork, features: MLXArray) -> (MLXArray, MLXArray?) {
        let batchSize = features.shape[0]
        let seqLen = features.shape[1]
        let hMax = network.maxHiddenDim
        let numLayers = network.numLayers
        let currentSeq0 = matmul(features, network.wIn) + network.bH
        var v = [MLXArray](repeating: MLXArray.zeros([batchSize, hMax]), count: numLayers)
        var s = [MLXArray](repeating: MLXArray.zeros([batchSize, hMax]), count: numLayers)
        var a = [MLXArray](repeating: MLXArray.zeros([batchSize, hMax]), count: numLayers)
        var rates: [MLXArray] = []
        let readoutList = forwardFrames(network: network, currentSeq0: currentSeq0, from: 0, to: seqLen, v: &v, s: &s, a: &a, voiceRates: &rates)
        let logits = matmul(stacked(readoutList, axis: 1), network.wOut) + network.bOut
        if rates.isEmpty {
            return (logits, nil)
        }
        return (logits, voiceLogits(network: network, rates: rates))
    }

    /// 長い系列の損失とパラメータ勾配をチャンク分割で求める。
    ///
    /// 全系列を一度に微分すると逆伝播用の中間値が系列長に比例して残り (64 件 × 256 フレームで
    /// 15 GB、768 フレームで 110 GB)、実メモリを超えると 1 バッチ 30 秒かかる。切り詰め BPTT は
    /// 窓の外へ勾配を流さないので、次の 3 段で同じ勾配を小さなメモリで得られる。
    ///   1. 勾配なしの前向きで、チャンク先頭の状態と全フレームのロジットを取る
    ///   2. CTC 損失のロジットについての勾配を vjp で求める
    ///   3. チャンクごとに前向きをやり直し、ロジットとその勾配の内積を逆伝播してパラメータ勾配を足す
    /// チャンク境界は窓境界に揃えるので、一括で微分した結果と一致する
    func chunkedLossAndGradients(
        features: MLXArray,
        extTargets: MLXCTCLoss.ExtendedTargets,
        voiceTargets: MLXArray? = nil,
        compiled: Bool = true
    ) -> (loss: MLXArray, gradients: ModuleParameters) {
        let batchSize = features.shape[0]
        let seqLen = features.shape[1]
        let hMax = network.maxHiddenDim
        let numLayers = network.numLayers
        let chunkFrames = max(bpttWindow, (longSequenceChunkFrames / bpttWindow) * bpttWindow)
        let useVoice = network.voiceHead.isEmpty != true && voiceTargets != nil
        var headCount = 1
        if useVoice {
            headCount = 2
        }

        // チャンクの形 [B, フレーム数] ごとに compile して、以後はトレース済みの関数を使う。
        // 端のチャンクは 32 フレーム単位に切り上げて 0 で埋める (因果系なので前のフレームの結果は変わらない)。
        // 形の種類は 32・64・96・128 の 4 つで済み、160 フレームの系列を 256 まで埋める無駄を避ける
        func paddedFrames(_ frames: Int) -> Int {
            return min(chunkFrames, ((frames + 31) / 32) * 32)
        }
        func padded(_ x: MLXArray, to frames: Int) -> MLXArray {
            if x.shape[1] == frames {
                return x
            }
            return concatenated([x, MLXArray.zeros([x.shape[0], frames - x.shape[1], x.shape[2]])], axis: 1)
        }
        func forwardFn(_ frames: Int) -> ([MLXArray]) -> [MLXArray] {
            if compiled != true {
                return { arrays in self.chunkForward(arrays, chunkFrames: frames) }
            }
            let key = batchSize << 16 | frames
            if let cached = compiledChunkForward[key] {
                return cached
            }
            let fn = compile(inputs: [network]) { arrays in self.chunkForward(arrays, chunkFrames: frames) }
            compiledChunkForward[key] = fn
            return fn
        }
        func backwardFn(_ frames: Int) -> ([MLXArray]) -> [MLXArray] {
            if compiled != true {
                return { arrays in self.chunkBackward(arrays, chunkFrames: frames) }
            }
            let key = batchSize << 16 | frames
            if let cached = compiledChunkBackward[key] {
                return cached
            }
            let fn = compile(inputs: [network]) { arrays in self.chunkBackward(arrays, chunkFrames: frames) }
            compiledChunkBackward[key] = fn
            return fn
        }

        // 1. 勾配なしの前向き
        var states = [MLXArray](repeating: MLXArray.zeros([batchSize, hMax]), count: 3 * numLayers)
        var chunkStarts: [[MLXArray]] = []
        var logitsChunks: [MLXArray] = []
        var voiceChunks: [MLXArray] = []
        var t0 = 0
        while t0 < seqLen {
            let t1 = min(seqLen, t0 + chunkFrames)
            let frames = paddedFrames(t1 - t0)
            chunkStarts.append(states)
            let out = forwardFn(frames)([padded(features[0..., t0..<t1, 0...], to: frames)] + states)
            eval(out)
            logitsChunks.append(out[0][0..., 0..<(t1 - t0), 0...])
            if useVoice {
                voiceChunks.append(out[1][0..., 0..<(t1 - t0), 0...])
            }
            states = Array(out[headCount...])
            t0 = t1
        }
        let logits = concatenated(logitsChunks, axis: 1)

        // 2. CTC 損失のロジット勾配 (+ 補助損失のロジット勾配は解析的に求める)
        let (lossOut, cotangents) = vjp(
            { (arrays: [MLXArray]) -> [MLXArray] in
                return [MLXCTCLoss.loss(logits: arrays[0], targets: extTargets)]
            },
            primals: [logits],
            cotangents: [MLXArray(Float(1.0))]
        )
        let gradLogits = cotangents[0]
        var totalLoss = lossOut[0]
        var gradVoice: MLXArray? = nil
        if useVoice, let vt = voiceTargets {
            let voice = concatenated(voiceChunks, axis: 1)
            let vLoss = Self.voiceLoss(logits: voice, targets: vt)
            totalLoss = totalLoss + vLoss * voiceLossWeight
            gradVoice = Self.voiceLossGradient(logits: voice, targets: vt) * voiceLossWeight
            eval(vLoss)
            lastVoiceLoss = vLoss.item(Float.self)
        }
        eval(totalLoss, gradLogits)

        // 3. チャンクごとの逆伝播 (パラメータ勾配を足し込む)
        var total: ModuleParameters? = nil
        var c = 0
        t0 = 0
        while t0 < seqLen {
            let t1 = min(seqLen, t0 + chunkFrames)
            let frames = paddedFrames(t1 - t0)
            var inputs = [padded(features[0..., t0..<t1, 0...], to: frames), padded(gradLogits[0..., t0..<t1, 0...], to: frames)]
            if let gv = gradVoice {
                inputs.append(padded(gv[0..., t0..<t1, 0...], to: frames))
            }
            inputs.append(contentsOf: chunkStarts[c])
            let gradValues = backwardFn(frames)(inputs)
            let grads = ModuleParameters.unflattened(Array(zip(chunkGradKeys, gradValues)))
            switch total {
            case .some(let acc):
                total = acc.mapValues(grads) { (x: MLXArray, y: MLXArray?) -> MLXArray in
                    return x + (y ?? MLXArray.zeros(like: x))
                }
            case .none:
                total = grads
            }
            eval(total!.flattenedValues())
            c += 1
            t0 = t1
        }
        return (totalLoss, total!)
    }

    /// チャンク 1 つの勾配なし前向き。入力は [特徴量 [B, C, inDim]] + 状態 (v, s, a を層ごとに)、
    /// 出力は [ロジット [B, C, V]] + チャンク末尾の状態
    private func chunkForward(_ arrays: [MLXArray], chunkFrames: Int) -> [MLXArray] {
        let numLayers = network.numLayers
        let cur = matmul(arrays[0], network.wIn) + network.bH
        var v = Array(arrays[1..<(1 + numLayers)])
        var s = Array(arrays[(1 + numLayers)..<(1 + 2 * numLayers)])
        var a = Array(arrays[(1 + 2 * numLayers)..<(1 + 3 * numLayers)])
        var rates: [MLXArray] = []
        let readouts = forwardFrames(network: network, currentSeq0: cur, from: 0, to: chunkFrames, v: &v, s: &s, a: &a, voiceRates: &rates)
        let logits = matmul(stacked(readouts, axis: 1), network.wOut) + network.bOut
        if rates.isEmpty {
            return [logits] + v + s + a
        }
        return [logits, voiceLogits(network: network, rates: rates)] + v + s + a
    }

    /// チャンク 1 つの逆伝播。入力は [特徴量, ロジット勾配] + チャンク先頭の状態、
    /// 出力はパラメータ勾配を `chunkGradKeys` の順に並べたもの
    private func chunkBackward(_ arrays: [MLXArray], chunkFrames: Int) -> [MLXArray] {
        let numLayers = network.numLayers
        // 補助ヘッドがあるときは入力が [特徴量, ロジット勾配, 補助ロジット勾配] + 状態
        let useVoice = network.voiceHead.isEmpty != true
        var base = 2
        if useVoice {
            base = 3
        }
        let lg = valueAndGrad(model: network) { (model: MLXSpikingNetwork, arrays: [MLXArray]) -> [MLXArray] in
            let cur = matmul(arrays[0], model.wIn) + model.bH
            var cv = Array(arrays[base..<(base + numLayers)])
            var cs = Array(arrays[(base + numLayers)..<(base + 2 * numLayers)])
            var ca = Array(arrays[(base + 2 * numLayers)..<(base + 3 * numLayers)])
            var rates: [MLXArray] = []
            let readouts = self.forwardFrames(network: model, currentSeq0: cur, from: 0, to: chunkFrames, v: &cv, s: &cs, a: &ca, voiceRates: &rates)
            let chunkLogits = matmul(stacked(readouts, axis: 1), model.wOut) + model.bOut
            var inner = sum(chunkLogits * arrays[1])
            if useVoice {
                inner = inner + sum(self.voiceLogits(network: model, rates: rates) * arrays[2])
            }
            return [inner]
        }
        let (_, grads) = lg(network, arrays)
        let flat = grads.flattened()
        chunkGradKeys = flat.map { $0.0 }
        return flat.map { $0.1 }
    }

    /// バッチ（複数発話）に対するフレーム整列教師の交差エントロピー損失
    public func lossBatch(
        network: MLXSpikingNetwork,
        features: MLXArray,          // [B, T, inputDim]
        targets: MLXArray            // [B, T] (Int32, パディングは -1)
    ) -> MLXArray {
        let logits = logitsBatch(network: network, features: features)

        // 重み: targets == -1 (パディング) は 0.0, targets == 0 (padId) は 0.3, 0 < targets は 1.0
        let weights = which(targets .== -1, MLXArray(0.0), which(targets .== 0, MLXArray(0.3), MLXArray(1.0)))
        let totalWeight = sum(weights) + 1e-6
        let cleanTargets = clip(targets, min: 0, max: Float(network.outputDim - 1)).asType(.int32)

        let lossArray = crossEntropy(logits: logits, targets: cleanTargets, weights: weights, reduction: .sum)
        return lossArray / totalWeight
    }

    /// ミニバッチ学習ステップ
    public func trainBatch(
        featuresBatch: [[[Float]]],
        targetsBatch: [[Int]]
    ) -> Float {
        let bSize = featuresBatch.count
        if bSize == 0 { return 0.0 }

        var maxT = 0
        for f in featuresBatch {
            maxT = max(maxT, f.count)
        }
        if maxT == 0 { return 0.0 }

        let inDim = network.inputDim
        var flatFeat = [Float](repeating: 0.0, count: bSize * maxT * inDim)
        var flatTgt = [Int32](repeating: -1, count: bSize * maxT)

        var b = 0
        while b < bSize {
            let fSeq = featuresBatch[b]
            let tSeq = targetsBatch[b]
            let curT = fSeq.count
            var t = 0
            while t < curT {
                let fVec = fSeq[t]
                let offset = ((b * maxT) + t) * inDim
                var d = 0
                while d < inDim {
                    flatFeat[offset + d] = fVec[d]
                    d += 1
                }
                t += 1
            }

            var ot = 0
            while ot < min(curT, tSeq.count) {
                flatTgt[(b * maxT) + ot] = Int32(tSeq[ot])
                ot += 1
            }
            b += 1
        }

        let featArray = MLXArray(flatFeat, [bSize, maxT, inDim])
        let targetArray = MLXArray(flatTgt, [bSize, maxT])

        let lg = valueAndGrad(model: network) { model, fArr, tArr -> MLXArray in
            return self.lossBatch(network: model, features: fArr, targets: tArr)
        }

        let (lossVal, grads) = lg(network, featArray, targetArray)
        let (clippedGrads, _) = clipGradNorm(gradients: grads, maxNorm: 5.0)
        optimizer.update(model: network, gradients: clippedGrads)
        eval(network, lossVal)

        return lossVal.item(Float.self)
    }

    /// 単一サンプル学習ステップ
    public func trainStep(
        features: [[Float]],
        targets: [Int]
    ) -> Float {
        return trainBatch(featuresBatch: [features], targetsBatch: [targets])
    }

    /// CTC ミニバッチ学習ステップ (GPU)
    ///
    /// CTC の前向き再帰を MLX 演算で構成しているため、フォワードは 1 回で済み、
    /// ロジットを CPU へ取り出す必要もない。
    ///
    /// - Parameter targetsBatch: フレームに整列していないラベル列 (かな ID 列)。
    @discardableResult
    public func trainBatchCTC(
        featuresBatch: [[[Float]]],
        targetsBatch: [[Int]],
        blankId: Int = 0,
        compiled: Bool = true,
        voiceTargetsBatch: [[Int]]? = nil
    ) -> Float {
        let bSize = featuresBatch.count
        if bSize == 0 {
            return 0.0
        }

        // ラベルが空のサンプルは CTC を定義できないので除外する
        var validFeatures: [[[Float]]] = []
        var validTargets: [[Int]] = []
        var validVoice: [[Int]] = []
        var frameCounts: [Int] = []
        var maxT = 0
        var b = 0
        while b < bSize {
            let frames = featuresBatch[b].count
            if 0 < frames && targetsBatch[b].isEmpty != true {
                validFeatures.append(featuresBatch[b])
                validTargets.append(targetsBatch[b])
                validVoice.append(voiceTargetsBatch?[b] ?? [])
                frameCounts.append(frames)
                if maxT < frames {
                    maxT = frames
                }
            }
            b += 1
        }
        if validFeatures.isEmpty || maxT == 0 {
            return 0.0
        }

        // 系列長を 32 の倍数へ切り上げる。
        // MLX は解放したバッファを形ごとに使い回すため、毎バッチ長さが違うと
        // 使い回せないバッファが溜まり Metal のリソース上限に達する。
        // 増えたフレームは inputLengths で凍結されるので損失は変わらない
        maxT = ((maxT + 31) / 32) * 32

        let validCount = validFeatures.count
        let inDim = network.inputDim
        var flatFeat = [Float](repeating: 0.0, count: validCount * maxT * inDim)
        b = 0
        while b < validCount {
            let fSeq = validFeatures[b]
            var t = 0
            while t < fSeq.count {
                let fVec = fSeq[t]
                let offset = ((b * maxT) + t) * inDim
                var d = 0
                while d < inDim {
                    flatFeat[offset + d] = fVec[d]
                    d += 1
                }
                t += 1
            }
            b += 1
        }

        let featArray = MLXArray(flatFeat, [validCount, maxT, inDim])
        let extTargets = MLXCTCLoss.ExtendedTargets(
            targetsBatch: validTargets,
            frameCounts: frameCounts,
            blankId: blankId
        )
        // 補助ヘッドがあるときは声の種類のフレームラベル [B, maxT] (無い項目・パディングは -1) を常に渡す。
        // compile 済みの関数の入力の並びを一定にするため
        let useVoice = network.voiceHead.isEmpty != true
        var voiceArray: MLXArray? = nil
        if useVoice {
            var flatVoice = [Int32](repeating: -1, count: validCount * maxT)
            b = 0
            while b < validCount {
                let labels = validVoice[b]
                var t = 0
                while t < min(labels.count, maxT) {
                    flatVoice[b * maxT + t] = Int32(labels[t])
                    t += 1
                }
                b += 1
            }
            voiceArray = MLXArray(flatVoice, [validCount, maxT])
        }
        let voiceWeight = voiceLossWeight
        lastVoiceLoss = 0.0

        if compiledMaxFrames < maxT && chunkLongSequences {
            let (loss, grads) = chunkedLossAndGradients(features: featArray, extTargets: extTargets, voiceTargets: voiceArray, compiled: compiled)
            let (clippedGrads, _) = clipGradNorm(gradients: grads, maxNorm: 5.0)
            optimizer.update(model: network, gradients: clippedGrads)
            eval(network, optimizer, loss)
            return loss.item(Float.self)
        }

        if compiled != true || compiledMaxFrames < maxT {
            let lg = valueAndGrad(model: network) { (model: MLXSpikingNetwork, arrays: [MLXArray]) -> [MLXArray] in
                let (logits, voice) = self.logitsAndVoiceBatch(network: model, features: arrays[0])
                let ctc = MLXCTCLoss.loss(logits: logits, targets: extTargets)
                if let voice = voice, 1 < arrays.count {
                    let vLoss = Self.voiceLoss(logits: voice, targets: arrays[1])
                    return [ctc + vLoss * voiceWeight, vLoss]
                }
                return [ctc, MLXArray(Float(0.0))]
            }

            var inputs = [featArray]
            if let va = voiceArray {
                inputs.append(va)
            }
            let (lossValues, grads) = lg(network, inputs)
            let (clippedGrads, _) = clipGradNorm(gradients: grads, maxNorm: 5.0)
            optimizer.update(model: network, gradients: clippedGrads)
            eval(network, optimizer, lossValues)
            lastVoiceLoss = lossValues[1].item(Float.self)

            return lossValues[0].item(Float.self)
        }

        let stepFn: ([MLXArray]) -> [MLXArray]
        if let cached = compiledCTCSteps[maxT] {
            ctcCacheHitCount += 1
            stepFn = cached
        } else {
            ctcCompileCount += 1
            let computeLoss = { (model: MLXSpikingNetwork, arrays: [MLXArray]) -> [MLXArray] in
                let (logits, voice) = self.logitsAndVoiceBatch(network: model, features: arrays[0])
                let targets = MLXCTCLoss.ExtendedTargets(
                    extTargets: arrays[1],
                    skipMask: arrays[2],
                    validMask: arrays[3],
                    finalIndex1: arrays[4],
                    finalIndex2: arrays[5],
                    hasSecondFinal: arrays[6],
                    inputLengths: arrays[7]
                )
                let ctc = MLXCTCLoss.loss(logits: logits, targets: targets)
                if let voice = voice, 8 < arrays.count {
                    let vLoss = Self.voiceLoss(logits: voice, targets: arrays[8])
                    return [ctc + vLoss * voiceWeight, vLoss]
                }
                return [ctc, MLXArray(Float(0.0))]
            }
            let lg = valueAndGrad(model: self.network, computeLoss)
            // 学習率は最後の入力配列。トレース時にここで差し替えるので、以後の呼び出しでは
            // 入力に渡した値がそのまま更新式に使われる。トレース用のプレースホルダ配列は
            // 実体を持たないので、更新後は元の配列へ戻す
            var lossInputCount = 8
            if useVoice {
                lossInputCount = 9
            }
            func step(arrays: [MLXArray]) -> [MLXArray] {
                let (lossValues, grads) = lg(self.network, Array(arrays[0..<lossInputCount]))
                let (clippedGrads, _) = clipGradNorm(gradients: grads, maxNorm: 5.0)
                let outerLearningRate = self.optimizer.learningRate
                self.optimizer.learningRate = arrays[lossInputCount]
                self.optimizer.update(model: self.network, gradients: clippedGrads)
                self.optimizer.learningRate = outerLearningRate
                return lossValues
            }
            let newStep = compile(
                inputs: [self.network, self.optimizer],
                outputs: [self.network, self.optimizer],
                step
            )
            compiledCTCSteps[maxT] = newStep
            stepFn = newStep
        }

        var inputs: [MLXArray] = [featArray]
        inputs.append(contentsOf: extTargets.toArrays())
        if let va = voiceArray {
            inputs.append(va)
        }
        inputs.append(optimizer.learningRate)

        let lossValues = stepFn(inputs)
        // オプティマイザの状態も評価する。network と損失だけ評価すると
        // Adam の m/v が遅延グラフとして積み上がり、生存バッファ数が
        // Metal のリソース上限 (約 50 万) に達して落ちる
        eval(network, optimizer, lossValues)
        lastVoiceLoss = lossValues[1].item(Float.self)

        return lossValues[0].item(Float.self)
    }
}
