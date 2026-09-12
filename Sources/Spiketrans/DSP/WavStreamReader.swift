import Foundation

/// ファイルから WAV をチャンク探索して PCM を逐次読む。形式は `WavParser` と同じ。
public final class WavStreamReader: @unchecked Sendable {
    public let format: WavFormat
    public var sampleRate: Int { format.sampleRate }
    public var numChannels: Int { format.channels }
    public var bitsPerSample: Int { format.bitsPerSample }
    public var dataByteCount: Int { format.dataByteCount }
    public var totalSamples: Int { format.totalSamples }

    private let handle: FileHandle
    private var remainingBytes: Int
    private var isClosed = false

    public init(filePath: String) throws {
        let url = URL(fileURLWithPath: filePath)
        let attrs = try FileManager.default.attributesOfItem(atPath: filePath)
        let fileSize = (attrs[.size] as? NSNumber)?.uint64Value ?? 0
        let handle = try FileHandle(forReadingFrom: url)
        self.handle = handle
        let format = try WavFormat.read(from: handle, fileSize: fileSize)
        self.format = format
        self.remainingBytes = format.dataByteCount
    }

    deinit {
        try? close()
    }

    /// 最大 maxSamples 個のモノラル PCM Float を読む。戻り値 0 は EOF。
    public func readSamples(
        into buffer: UnsafeMutablePointer<Float>,
        maxSamples: Int
    ) throws -> Int {
        if isClosed || maxSamples <= 0 || remainingBytes <= 0 {
            return 0
        }
        let frameBytes = format.frameBytes
        if frameBytes <= 0 {
            return 0
        }
        let maxBytes = min(remainingBytes, maxSamples * frameBytes)
        let wantBytes = maxBytes - (maxBytes % frameBytes)
        if wantBytes <= 0 {
            remainingBytes = 0
            return 0
        }
        let raw = try readBytes(from: handle, count: wantBytes, allowShort: true)
        if raw.isEmpty {
            remainingBytes = 0
            return 0
        }
        let complete = raw.count - (raw.count % frameBytes)
        remainingBytes -= raw.count
        return raw.withUnsafeBytes { rawBuf in
            let base = rawBuf.bindMemory(to: UInt8.self).baseAddress!
            return WavPCM.decode(
                raw: base,
                byteCount: complete,
                format: format,
                into: buffer
            )
        }
    }

    public func readToEnd() throws -> WavData {
        var pcm = [Float](repeating: 0.0, count: totalSamples)
        var filled = 0
        while filled < pcm.count {
            let remaining = pcm.count - filled
            let n = try pcm.withUnsafeMutableBufferPointer { buf in
                try readSamples(into: buf.baseAddress! + filled, maxSamples: remaining)
            }
            if n <= 0 {
                break
            }
            filled += n
        }
        if filled < pcm.count {
            pcm.removeLast(pcm.count - filled)
        }
        return WavData(
            sampleRate: sampleRate,
            channels: numChannels,
            bitsPerSample: bitsPerSample,
            pcmData: pcm
        )
    }

    public func close() throws {
        if isClosed {
            return
        }
        isClosed = true
        try handle.close()
    }
}
