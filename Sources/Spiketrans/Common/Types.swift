import Foundation

/// 音声フレームの解析結果
public struct AudioFrameResult: Sendable, Equatable {
    public let isSpeech: Bool
    public let pitchHz: Float
    public let hnrDb: Float
    public let formants: (f1: Float, f2: Float, f3: Float)
    public let formantBandwidths: (b1: Float, b2: Float, b3: Float)
    public let features: [Float] // 64次元 (Direct Input Current: 0.0〜1.0)
    
    public init(
        isSpeech: Bool,
        pitchHz: Float,
        hnrDb: Float,
        formants: (f1: Float, f2: Float, f3: Float),
        formantBandwidths: (b1: Float, b2: Float, b3: Float),
        features: [Float]
    ) {
        self.isSpeech = isSpeech
        self.pitchHz = pitchHz
        self.hnrDb = hnrDb
        self.formants = formants
        self.formantBandwidths = formantBandwidths
        self.features = features
    }
    
    public static func == (lhs: AudioFrameResult, rhs: AudioFrameResult) -> Bool {
        if lhs.isSpeech != rhs.isSpeech {
            return false
        }
        if lhs.pitchHz != rhs.pitchHz {
            return false
        }
        if lhs.hnrDb != rhs.hnrDb {
            return false
        }
        if lhs.formants.f1 != rhs.formants.f1 || lhs.formants.f2 != rhs.formants.f2 || lhs.formants.f3 != rhs.formants.f3 {
            return false
        }
        if lhs.formantBandwidths.b1 != rhs.formantBandwidths.b1 || lhs.formantBandwidths.b2 != rhs.formantBandwidths.b2 || lhs.formantBandwidths.b3 != rhs.formantBandwidths.b3 {
            return false
        }
        return lhs.features == rhs.features
    }
}

/// ピッチおよび有声度解析結果
public struct PitchResult: Sendable, Equatable {
    public let f0: Float
    public let hnr: Float
    public let isVoiced: Bool
    
    public init(f0: Float, hnr: Float, isVoiced: Bool) {
        self.f0 = f0
        self.hnr = hnr
        self.isVoiced = isVoiced
    }
}

/// フォルマント解析結果
public struct FormantResult: Sendable, Equatable {
    public let f1: Float
    public let f2: Float
    public let f3: Float
    public let b1: Float
    public let b2: Float
    public let b3: Float
    public let count: Int
    
    public init(f1: Float, f2: Float, f3: Float, b1: Float, b2: Float, b3: Float, count: Int) {
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
        self.count = count
    }
}

/// VAD（Voice Activity Detection）フレーム判定結果
public struct VADResult: Sendable, Equatable {
    public let isSpeech: Bool
    public let rms: Float
    public let zcr: Float
    public let voicingRatio: Float
    public let noiseFloor: Float
    
    public init(isSpeech: Bool, rms: Float, zcr: Float, voicingRatio: Float, noiseFloor: Float) {
        self.isSpeech = isSpeech
        self.rms = rms
        self.zcr = zcr
        self.voicingRatio = voicingRatio
        self.noiseFloor = noiseFloor
    }
}

/// 発話区間セグメント
public struct SpeechSegment: Sendable, Equatable {
    public let startIndex: Int
    public let endIndex: Int
    public let durationSeconds: Float
    
    public init(startIndex: Int, endIndex: Int, durationSeconds: Float) {
        self.startIndex = startIndex
        self.endIndex = endIndex
        self.durationSeconds = durationSeconds
    }
}

/// パイプライン設定パラメータ
public struct DSPConfig: Sendable {
    public let sampleRate: Int
    public let frameSize: Int
    public let hopSize: Int
    public let lpcOrder: Int
    public let melChannels: Int
    public let minPitchLag: Int
    public let maxPitchLag: Int
    public let preemphasisCoeff: Float
    public let vadEnergyThresholdRatio: Float
    public let vadZcrThreshold: Float
    public let vadVoicingRatioThreshold: Float
    
    public init(
        sampleRate: Int = 16000,
        frameSize: Int = 320,
        hopSize: Int = 160,
        lpcOrder: Int = 12,
        melChannels: Int = 64,
        minPitchLag: Int = 32,
        maxPitchLag: Int = 320,
        preemphasisCoeff: Float = 0.97,
        vadEnergyThresholdRatio: Float = 2.5,
        vadZcrThreshold: Float = 0.35,
        vadVoicingRatioThreshold: Float = 0.30
    ) {
        self.sampleRate = sampleRate
        self.frameSize = frameSize
        self.hopSize = hopSize
        self.lpcOrder = lpcOrder
        self.melChannels = melChannels
        self.minPitchLag = minPitchLag
        self.maxPitchLag = maxPitchLag
        self.preemphasisCoeff = preemphasisCoeff
        self.vadEnergyThresholdRatio = vadEnergyThresholdRatio
        self.vadZcrThreshold = vadZcrThreshold
        self.vadVoicingRatioThreshold = vadVoicingRatioThreshold
    }
}
