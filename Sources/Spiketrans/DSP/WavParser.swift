import Foundation

public struct WavData: Sendable, Equatable {
    public let sampleRate: Int
    public let channels: Int
    public let bitsPerSample: Int
    public let pcmData: [Float]

    public init(sampleRate: Int, channels: Int, bitsPerSample: Int, pcmData: [Float]) {
        self.sampleRate = sampleRate
        self.channels = channels
        self.bitsPerSample = bitsPerSample
        self.pcmData = pcmData
    }
}

public enum WavParserError: Error, Equatable {
    case invalidHeader
    case fmtChunkNotFound
    case dataChunkNotFound
    case unsupportedFormat(String)
}

public struct WavParser: Sendable {
    public init() {}

    /// メモリ上の WAV バイト列をパースする。ファイルは `WavStreamReader`。
    public func parse(bytes: [UInt8]) throws -> WavData {
        let (format, dataOffset) = try WavFormat.parse(bytes: bytes)
        let available = max(0, bytes.count - dataOffset)
        let byteCount = min(format.dataByteCount, available)
        var pcmData = [Float](repeating: 0.0, count: byteCount / max(1, format.frameBytes))
        let n = bytes.withUnsafeBufferPointer { bytePtr in
            pcmData.withUnsafeMutableBufferPointer { pcmPtr in
                WavPCM.decode(
                    raw: bytePtr.baseAddress! + dataOffset,
                    byteCount: byteCount,
                    format: format,
                    into: pcmPtr.baseAddress!
                )
            }
        }
        if n < pcmData.count {
            pcmData.removeLast(pcmData.count - n)
        }
        return WavData(
            sampleRate: format.sampleRate,
            channels: format.channels,
            bitsPerSample: format.bitsPerSample,
            pcmData: pcmData
        )
    }
}
