import Foundation

/// WAV の fmt / data から得た形式。メモリ経路とファイル経路で同じ値を使う。
public struct WavFormat: Sendable, Equatable {
    public let sampleRate: Int
    public let channels: Int
    public let bitsPerSample: Int
    public let isFloat: Bool
    public let dataByteCount: Int

    public var bytesPerSample: Int { bitsPerSample / 8 }
    public var frameBytes: Int { bytesPerSample * max(1, channels) }
    public var totalSamples: Int {
        let n = frameBytes
        if n <= 0 {
            return 0
        }
        return dataByteCount / n
    }

    /// メモリ上の RIFF を歩く。data チャンク先頭のオフセットも返す。
    public static func parse(bytes: [UInt8]) throws -> (format: WavFormat, dataOffset: Int) {
        if bytes.count < 12 {
            throw WavParserError.invalidHeader
        }
        if bytes[0] != 0x52 || bytes[1] != 0x49 || bytes[2] != 0x46 || bytes[3] != 0x46 {
            throw WavParserError.invalidHeader
        }
        if bytes[8] != 0x57 || bytes[9] != 0x41 || bytes[10] != 0x56 || bytes[11] != 0x45 {
            throw WavParserError.invalidHeader
        }

        var sampleRate = 0
        var channels = 0
        var bitsPerSample = 0
        var dataOffset = 0
        var dataSize = 0
        var isFloat = false
        var offset = 12
        let fileLimit = bytes.count

        while (offset + 8) <= fileLimit {
            let chunkSize = int32LE(bytes, offset + 4)
            if chunkSize < 0 || fileLimit < (offset + 8 + chunkSize) {
                if 0 < dataOffset && 0 < sampleRate {
                    break
                }
                throw WavParserError.invalidHeader
            }

            if isChunk(bytes, offset, 0x66, 0x6d, 0x74, 0x20) {
                let parsed = try parseFmt(bytes, bodyOffset: offset + 8, chunkSize: chunkSize)
                sampleRate = parsed.sampleRate
                channels = parsed.channels
                bitsPerSample = parsed.bitsPerSample
                isFloat = parsed.isFloat
            } else if isChunk(bytes, offset, 0x64, 0x61, 0x74, 0x61) {
                dataOffset = offset + 8
                dataSize = chunkSize
            }

            offset += 8 + chunkSize + (chunkSize & 1)
        }

        let format = try makeFormat(
            sampleRate: sampleRate,
            channels: channels,
            bitsPerSample: bitsPerSample,
            isFloat: isFloat,
            dataByteCount: dataSize
        )
        if dataOffset == 0 {
            throw WavParserError.dataChunkNotFound
        }
        return (format, dataOffset)
    }

    /// ファイルを歩く。戻ったときハンドルは data チャンク先頭にある。
    public static func read(from handle: FileHandle, fileSize: UInt64) throws -> WavFormat {
        let riff = try readBytes(from: handle, count: 12, allowShort: false)
        if riff.count < 12
            || riff[0] != 0x52 || riff[1] != 0x49 || riff[2] != 0x46 || riff[3] != 0x46
            || riff[8] != 0x57 || riff[9] != 0x41 || riff[10] != 0x56 || riff[11] != 0x45 {
            throw WavParserError.invalidHeader
        }

        var sampleRate = 0
        var channels = 0
        var bitsPerSample = 0
        var isFloat = false
        var dataOffset: UInt64 = 0
        var dataSize = 0
        var pos: UInt64 = 12

        while (pos + 8) <= fileSize {
            let head = try readBytes(from: handle, count: 8, allowShort: false)
            let chunkSize = int32LE(head, 4)
            let payloadPos = pos + 8
            if chunkSize < 0 || fileSize < (payloadPos + UInt64(chunkSize)) {
                if 0 < dataOffset && 0 < sampleRate {
                    break
                }
                throw WavParserError.invalidHeader
            }

            if isChunk(head, 0, 0x66, 0x6d, 0x74, 0x20) {
                let body = try readBytes(from: handle, count: chunkSize, allowShort: false)
                let parsed = try parseFmt(Array(body), bodyOffset: 0, chunkSize: chunkSize)
                sampleRate = parsed.sampleRate
                channels = parsed.channels
                bitsPerSample = parsed.bitsPerSample
                isFloat = parsed.isFloat
                try skip(handle, chunkSize & 1)
            } else if isChunk(head, 0, 0x64, 0x61, 0x74, 0x61) {
                dataOffset = payloadPos
                dataSize = chunkSize
                if 0 < sampleRate {
                    try handle.seek(toOffset: dataOffset)
                    return try makeFormat(
                        sampleRate: sampleRate,
                        channels: channels,
                        bitsPerSample: bitsPerSample,
                        isFloat: isFloat,
                        dataByteCount: dataSize
                    )
                }
                try skip(handle, chunkSize + (chunkSize & 1))
            } else {
                try skip(handle, chunkSize + (chunkSize & 1))
            }
            pos += 8 + UInt64(chunkSize) + UInt64(chunkSize & 1)
        }

        if dataOffset == 0 {
            throw WavParserError.dataChunkNotFound
        }
        try handle.seek(toOffset: dataOffset)
        return try makeFormat(
            sampleRate: sampleRate,
            channels: channels,
            bitsPerSample: bitsPerSample,
            isFloat: isFloat,
            dataByteCount: dataSize
        )
    }

    private static func makeFormat(
        sampleRate: Int,
        channels: Int,
        bitsPerSample: Int,
        isFloat: Bool,
        dataByteCount: Int
    ) throws -> WavFormat {
        if sampleRate == 0 || channels == 0 || bitsPerSample == 0 {
            throw WavParserError.fmtChunkNotFound
        }
        if isFloat {
            if bitsPerSample != 32 {
                throw WavParserError.unsupportedFormat("Only 32-bit IEEE float is supported, got \(bitsPerSample)")
            }
        } else if bitsPerSample != 16 && bitsPerSample != 24 && bitsPerSample != 32 {
            throw WavParserError.unsupportedFormat("Only 16/24/32-bit integer PCM is supported, got \(bitsPerSample)")
        }
        return WavFormat(
            sampleRate: sampleRate,
            channels: channels,
            bitsPerSample: bitsPerSample,
            isFloat: isFloat,
            dataByteCount: dataByteCount
        )
    }

    private static func parseFmt(_ bytes: [UInt8], bodyOffset: Int, chunkSize: Int) throws -> (
        sampleRate: Int, channels: Int, bitsPerSample: Int, isFloat: Bool
    ) {
        if chunkSize < 16 || bytes.count < (bodyOffset + 16) {
            throw WavParserError.invalidHeader
        }
        let formatCode = int16LE(bytes, bodyOffset)
        var effective = formatCode
        if formatCode == 0xFFFE && 40 <= chunkSize && bytes.count >= (bodyOffset + 26) {
            effective = int16LE(bytes, bodyOffset + 24)
        }
        if effective != 1 && effective != 3 {
            throw WavParserError.unsupportedFormat("Only integer PCM and IEEE float are supported, got format \(formatCode)")
        }
        return (
            sampleRate: int32LE(bytes, bodyOffset + 4),
            channels: int16LE(bytes, bodyOffset + 2),
            bitsPerSample: int16LE(bytes, bodyOffset + 14),
            isFloat: effective == 3
        )
    }
}

enum WavPCM {
    static let pcmScale: Float = 1.0 / 32768.0

    /// `raw` は data チャンク先頭。書いたモノラルサンプル数を返す。
    @inline(__always)
    static func decode(
        raw: UnsafePointer<UInt8>,
        byteCount: Int,
        format: WavFormat,
        into dst: UnsafeMutablePointer<Float>
    ) -> Int {
        let frameBytes = format.frameBytes
        if frameBytes <= 0 {
            return 0
        }
        let totalSamples = byteCount / frameBytes
        if format.isFloat {
            decodeFloat32(raw: raw, dst: dst, totalSamples: totalSamples, channels: format.channels, byteCount: byteCount)
            return totalSamples
        }
        if format.bitsPerSample != 16 {
            decodeWideInteger(
                raw: raw, dst: dst, totalSamples: totalSamples,
                channels: format.channels, bytesPerSample: format.bytesPerSample, byteCount: byteCount
            )
            return totalSamples
        }
        switch format.channels {
        case 1:
            decodeMono16(raw: raw, dst: dst, totalSamples: totalSamples, byteCount: byteCount)
        case 2:
            decodeStereo16(raw: raw, dst: dst, totalSamples: totalSamples, byteCount: byteCount)
        default:
            break
        }
        return totalSamples
    }

    @inline(__always)
    private static func decodeWideInteger(
        raw: UnsafePointer<UInt8>,
        dst: UnsafeMutablePointer<Float>,
        totalSamples: Int,
        channels: Int,
        bytesPerSample: Int,
        byteCount: Int
    ) {
        let frameBytes = bytesPerSample * channels
        let invChannels = 1.0 / Float(channels)
        let highOffset = bytesPerSample - 2
        var i = 0
        while i < totalSamples {
            let base = i * frameBytes
            if byteCount < (base + frameBytes) {
                break
            }
            var sum: Float = 0.0
            var ch = 0
            while ch < channels {
                let o = base + (ch * bytesPerSample) + highOffset
                let bits = UInt16(raw[o]) | (UInt16(raw[o + 1]) << 8)
                sum += Float(Int16(bitPattern: bits)) * pcmScale
                ch += 1
            }
            dst[i] = sum * invChannels
            i += 1
        }
    }

    @inline(__always)
    private static func decodeFloat32(
        raw: UnsafePointer<UInt8>,
        dst: UnsafeMutablePointer<Float>,
        totalSamples: Int,
        channels: Int,
        byteCount: Int
    ) {
        let frameBytes = 4 * channels
        let invChannels = 1.0 / Float(channels)
        var i = 0
        while i < totalSamples {
            let base = i * frameBytes
            if byteCount < (base + frameBytes) {
                break
            }
            var sum: Float = 0.0
            var ch = 0
            while ch < channels {
                let o = base + (ch * 4)
                let bits = UInt32(raw[o])
                    | (UInt32(raw[o + 1]) << 8)
                    | (UInt32(raw[o + 2]) << 16)
                    | (UInt32(raw[o + 3]) << 24)
                sum += Float(bitPattern: bits)
                ch += 1
            }
            dst[i] = sum * invChannels
            i += 1
        }
    }

    @inline(__always)
    private static func decodeMono16(
        raw: UnsafePointer<UInt8>,
        dst: UnsafeMutablePointer<Float>,
        totalSamples: Int,
        byteCount: Int
    ) {
        let width = 8
        let limit = totalSamples - (totalSamples % width)
        var i = 0
        while i < limit {
            let sampleIndex = i * 2
            if byteCount < (sampleIndex + (width * 2)) {
                break
            }
            let pRaw = raw.advanced(by: sampleIndex).withMemoryRebound(to: Int16.self, capacity: width) { ptr in
                return ptr
            }
            dst[i + 0] = Float(pRaw[0]) * pcmScale
            dst[i + 1] = Float(pRaw[1]) * pcmScale
            dst[i + 2] = Float(pRaw[2]) * pcmScale
            dst[i + 3] = Float(pRaw[3]) * pcmScale
            dst[i + 4] = Float(pRaw[4]) * pcmScale
            dst[i + 5] = Float(pRaw[5]) * pcmScale
            dst[i + 6] = Float(pRaw[6]) * pcmScale
            dst[i + 7] = Float(pRaw[7]) * pcmScale
            i += width
        }
        while i < totalSamples {
            let sampleIndex = i * 2
            if byteCount < (sampleIndex + 2) {
                break
            }
            let bits = UInt16(raw[sampleIndex]) | (UInt16(raw[sampleIndex + 1]) << 8)
            dst[i] = Float(Int16(bitPattern: bits)) * pcmScale
            i += 1
        }
    }

    @inline(__always)
    private static func decodeStereo16(
        raw: UnsafePointer<UInt8>,
        dst: UnsafeMutablePointer<Float>,
        totalSamples: Int,
        byteCount: Int
    ) {
        let halfScale = pcmScale * 0.5
        var i = 0
        while i < totalSamples {
            let sampleIndex = i * 4
            if byteCount < (sampleIndex + 4) {
                break
            }
            let l = UInt16(raw[sampleIndex]) | (UInt16(raw[sampleIndex + 1]) << 8)
            let r = UInt16(raw[sampleIndex + 2]) | (UInt16(raw[sampleIndex + 3]) << 8)
            dst[i] = (Float(Int16(bitPattern: l)) + Float(Int16(bitPattern: r))) * halfScale
            i += 1
        }
    }
}

@inline(__always)
private func isChunk(_ bytes: [UInt8], _ offset: Int, _ a: UInt8, _ b: UInt8, _ c: UInt8, _ d: UInt8) -> Bool {
    return bytes[offset] == a && bytes[offset + 1] == b && bytes[offset + 2] == c && bytes[offset + 3] == d
}

@inline(__always)
private func isChunk(_ data: Data, _ offset: Int, _ a: UInt8, _ b: UInt8, _ c: UInt8, _ d: UInt8) -> Bool {
    return data[offset] == a && data[offset + 1] == b && data[offset + 2] == c && data[offset + 3] == d
}

@inline(__always)
private func int16LE(_ bytes: [UInt8], _ offset: Int) -> Int {
    return Int(bytes[offset]) | (Int(bytes[offset + 1]) << 8)
}

@inline(__always)
private func int32LE(_ bytes: [UInt8], _ offset: Int) -> Int {
    return Int(bytes[offset]) | (Int(bytes[offset + 1]) << 8) | (Int(bytes[offset + 2]) << 16) | (Int(bytes[offset + 3]) << 24)
}

@inline(__always)
private func int32LE(_ data: Data, _ offset: Int) -> Int {
    return Int(data[offset]) | (Int(data[offset + 1]) << 8) | (Int(data[offset + 2]) << 16) | (Int(data[offset + 3]) << 24)
}

func readBytes(from handle: FileHandle, count: Int, allowShort: Bool) throws -> Data {
    if count <= 0 {
        return Data()
    }
    var out = Data()
    out.reserveCapacity(count)
    while out.count < count {
        let chunk = try handle.read(upToCount: count - out.count)
        switch chunk {
        case .some(let data) where 0 < data.count:
            out.append(data)
        default:
            if allowShort {
                return out
            }
            throw WavParserError.invalidHeader
        }
    }
    return out
}

private func skip(_ handle: FileHandle, _ n: Int) throws {
    if n <= 0 {
        return
    }
    try handle.seek(toOffset: handle.offsetInFile + UInt64(n))
}
