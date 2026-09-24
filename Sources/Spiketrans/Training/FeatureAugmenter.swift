import Foundation

/// 学習時のデータ拡張 (話速摂動と SpecAugment)。推論側は変更しない。
///
/// - 話速摂動: PCM を 0.9〜1.1 倍に線形補間で伸縮する (Kaldi の speed perturbation と同じく音高も動く)。
///   早口・ゆっくりの話し方への汎化を狙う。ラベルは変わらない。
/// - SpecAugment: 特徴量の時間区間 (フレーム) と周波数帯 (メル帯) を 0 で隠す。
///   特徴量は 10 ms ホップの [平滑メル 64 + 時間差分 64] = 128 を `stack` 本束ねた並びなので、
///   周波数帯のマスクは全ホップ・平滑と差分の両方の同じ帯に掛ける。
public final class FeatureAugmenter: @unchecked Sendable {
    public static let speedRange: ClosedRange<Float> = 0.9...1.1
    /// 時間マスクの本数と、1 本の最大長 (フレーム数と系列長に対する比率の小さい方)
    public static let timeMasks = 2
    public static let maxTimeMaskFrames = 20
    public static let maxTimeMaskRatio: Float = 0.1
    /// 周波数マスクの本数と 1 本の最大帯数 (メル 64 帯のうち)
    public static let frequencyMasks = 2
    public static let maxFrequencyMaskBands = 12

    public let speed: Bool
    public let specAugment: Bool
    public let noiseBank: NoiseBank?

    public init(speed: Bool, specAugment: Bool, noiseBank: NoiseBank? = nil) {
        self.speed = speed
        self.specAugment = specAugment
        self.noiseBank = noiseBank
    }

    /// 何も掛けないなら nil (呼び出し側がフックを省ける)
    public var pcmTransform: (([Float]) -> [Float])? {
        if speed != true && noiseBank == nil {
            return nil
        }
        return { pcm in self.transformPCM(pcm) }
    }

    public var featureTransform: (([[Float]]) -> [[Float]])? {
        if specAugment != true {
            return nil
        }
        return { features in
            var rng = SystemRandomNumberGenerator()
            return Self.specAugment(features, using: &rng)
        }
    }

    func transformPCM(_ pcm: [Float]) -> [Float] {
        var rng = SystemRandomNumberGenerator()
        var out = pcm
        if speed {
            let factor = Float.random(in: Self.speedRange, using: &rng)
            out = Self.resample(out, speed: factor)
        }
        if let bank = noiseBank {
            out = bank.mix(out, using: &rng)
        }
        return out
    }

    /// 話速 `speed` 倍 (1.1 なら 10% 速く = サンプル数が 1/1.1 倍) に線形補間で伸縮する
    public static func resample(_ pcm: [Float], speed: Float) -> [Float] {
        if pcm.count < 2 || speed <= 0.0 || abs(speed - 1.0) < 1e-4 {
            return pcm
        }
        let outCount = max(1, Int(Float(pcm.count) / speed))
        var out = [Float](repeating: 0.0, count: outCount)
        var i = 0
        while i < outCount {
            let position = Float(i) * speed
            let base = Int(position)
            if pcm.count - 1 <= base {
                out[i] = pcm[pcm.count - 1]
            } else {
                let frac = position - Float(base)
                out[i] = pcm[base] * (1.0 - frac) + pcm[base + 1] * frac
            }
            i += 1
        }
        return out
    }

    /// SpecAugment。マスクの位置と長さは乱数、値は 0
    public static func specAugment<G: RandomNumberGenerator>(_ features: [[Float]], using rng: inout G) -> [[Float]] {
        let frames = features.count
        if frames == 0 {
            return features
        }
        let dim = features[0].count
        let tapDim = StreamingFeatureFrontEnd.tapDim
        let mel = StreamingFeatureFrontEnd.melChannels
        let stack = max(1, dim / tapDim)
        var out = features

        let maxTime = min(maxTimeMaskFrames, max(1, Int(Float(frames) * maxTimeMaskRatio)))
        var m = 0
        while m < timeMasks {
            let length = Int.random(in: 0...maxTime, using: &rng)
            if 0 < length && length < frames {
                let start = Int.random(in: 0...(frames - length), using: &rng)
                var t = start
                while t < start + length {
                    out[t] = [Float](repeating: 0.0, count: dim)
                    t += 1
                }
            }
            m += 1
        }

        m = 0
        while m < frequencyMasks {
            let bands = Int.random(in: 0...maxFrequencyMaskBands, using: &rng)
            if 0 < bands && bands < mel {
                let startBand = Int.random(in: 0...(mel - bands), using: &rng)
                var t = 0
                while t < frames {
                    var s = 0
                    while s < stack {
                        var b = startBand
                        while b < startBand + bands {
                            let melIndex = s * tapDim + b
                            let deltaIndex = s * tapDim + mel + b
                            if melIndex < dim {
                                out[t][melIndex] = 0.0
                            }
                            if deltaIndex < dim {
                                out[t][deltaIndex] = 0.0
                            }
                            b += 1
                        }
                        s += 1
                    }
                    t += 1
                }
            }
            m += 1
        }
        return out
    }
}
