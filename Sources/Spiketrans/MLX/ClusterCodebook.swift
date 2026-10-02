import Foundation
import MLX

/// 事前学習 (HuBERT 型) の目標を作るためのクラスタ中心。入力特徴量を次元ごとに標準化してから
/// k-means で分け、各フレームを最も近い中心の番号に置き換える
public struct ClusterCodebook: Codable, Sendable {
    public let mean: [Float]      // [dim]
    public let std: [Float]       // [dim]
    public let centroids: [[Float]]  // [k][dim] (標準化後の空間)

    public var count: Int {
        return centroids.count
    }

    public var dim: Int {
        return mean.count
    }

    public init(mean: [Float], std: [Float], centroids: [[Float]]) {
        self.mean = mean
        self.std = std
        self.centroids = centroids
    }

    public func save(to url: URL) throws {
        try JSONEncoder().encode(self).write(to: url)
    }

    public static func load(from url: URL) throws -> ClusterCodebook {
        return try JSONDecoder().decode(ClusterCodebook.self, from: Data(contentsOf: url))
    }

    /// 中心を MLX の配列にまとめたもの: [mean [D], invStd [D], centroids [K, D], centroidSq [K]]
    public func arrays() -> [MLXArray] {
        let d = dim
        let flat = centroids.flatMap { $0 }
        let c = MLXArray(flat, [count, d])
        let invStd = MLXArray(std.map { 1.0 / max($0, 1e-6) }, [d])
        return [MLXArray(mean, [d]), invStd, c, sum(c * c, axis: 1)]
    }

    /// 特徴量 [..., D] の各フレームに最も近い中心の番号 [...] (Int32)
    public static func assign(features: MLXArray, codebook: [MLXArray]) -> MLXArray {
        let x = (features - codebook[0]) * codebook[1]
        // |x - c|^2 = |x|^2 - 2 x·c + |c|^2 の最小 (|x|^2 はフレームごとに共通なので省く)
        let dots = matmul(x, codebook[2].transposed())
        let dist = codebook[3] - 2.0 * dots
        return argMin(dist, axis: -1).asType(.int32)
    }

    /// フレーム [N, D] から k-means で中心を作る (初期値は k-means++)。
    /// 標準化の平均・標準偏差もこのフレームから求める
    public static func fit(frames: [[Float]], k: Int, iterations: Int, seed: UInt64 = 0) -> ClusterCodebook {
        let n = frames.count
        let d = frames.first?.count ?? 0
        var meanV = [Float](repeating: 0.0, count: d)
        var sqV = [Float](repeating: 0.0, count: d)
        for f in frames {
            var j = 0
            while j < d {
                meanV[j] += f[j]
                sqV[j] += f[j] * f[j]
                j += 1
            }
        }
        var stdV = [Float](repeating: 1.0, count: d)
        var j = 0
        while j < d {
            meanV[j] /= Float(max(1, n))
            stdV[j] = max(1e-3, (sqV[j] / Float(max(1, n)) - meanV[j] * meanV[j]).squareRoot())
            j += 1
        }
        let x = (MLXArray(frames.flatMap { $0 }, [n, d]) - MLXArray(meanV, [d])) / MLXArray(stdV, [d])
        eval(x)
        // 初期値は k-means++: 既に選んだ中心から遠いフレームほど (距離の 2 乗に比例して) 選ばれやすくする
        var rng = SplitMix64(seed: seed)
        var initIdx: [Int32] = [Int32(rng.next() % UInt64(max(1, n)))]
        var nearest = sum(square(x - x[Int(initIdx[0])]), axis: 1)
        while initIdx.count < min(k, n) {
            eval(nearest)
            let dist = nearest.asArray(Float.self)
            var total: Double = 0.0
            for v in dist {
                total += Double(v)
            }
            var pick = Int(rng.next() % UInt64(n))
            if 0.0 < total {
                var r = Double(rng.next() % 1_000_000_007) / 1_000_000_007.0 * total
                var i = 0
                while i < n {
                    r -= Double(dist[i])
                    if r <= 0.0 {
                        pick = i
                        break
                    }
                    i += 1
                }
            }
            initIdx.append(Int32(pick))
            nearest = minimum(nearest, sum(square(x - x[pick]), axis: 1))
        }
        var c = x[MLXArray(initIdx)]
        var it = 0
        while it < iterations {
            let dist = sum(c * c, axis: 1) - 2.0 * matmul(x, c.transposed())
            let assign = argMin(dist, axis: -1)
            let onehot = (assign.expandedDimensions(axis: -1) .== MLXArray(0..<Int32(c.shape[0]))).asType(.float32)
            let counts = sum(onehot, axis: 0)
            let sums = matmul(onehot.transposed(), x)
            // 空になった中心は元の位置に残す
            let has = (MLXArray(Float(0.5)) .< counts).expandedDimensions(axis: -1)
            c = which(has, sums / maximum(counts, MLXArray(Float(1.0))).expandedDimensions(axis: -1), c)
            eval(c)
            it += 1
        }
        let flat = c.asArray(Float.self)
        var out: [[Float]] = []
        var r = 0
        while r < c.shape[0] {
            out.append(Array(flat[(r * d)..<((r + 1) * d)]))
            r += 1
        }
        return ClusterCodebook(mean: meanV, std: stdV, centroids: out)
    }
}

/// 再現できる乱数 (k-means の初期値用)
struct SplitMix64: RandomNumberGenerator {
    var state: UInt64

    init(seed: UInt64) {
        self.state = seed
    }

    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
}
