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
    
    /// バイト配列から WAV データをパース
    public func parse(bytes: [UInt8]) throws -> WavData {
        if bytes.count < 44 {
            throw WavParserError.invalidHeader
        }
        
        // RIFF マジックナンバー
        if bytes[0] != 0x52 || bytes[1] != 0x49 || bytes[2] != 0x46 || bytes[3] != 0x46 {
            throw WavParserError.invalidHeader
        }
        // WAVE マジックナンバー
        if bytes[8] != 0x57 || bytes[9] != 0x41 || bytes[10] != 0x56 || bytes[11] != 0x45 {
            throw WavParserError.invalidHeader
        }
        
        var sampleRate = 0
        var channels = 0
        var bitsPerSample = 0
        var dataOffset = 0
        var dataSize = 0
        
        var offset = 12
        let fileLimit = bytes.count
        
        while (offset + 8) <= fileLimit {
            let chunkId0 = bytes[offset+0]
            let chunkId1 = bytes[offset+1]
            let chunkId2 = bytes[offset+2]
            let chunkId3 = bytes[offset+3]
            
            let chunkSize = Int(bytes[offset+4]) | (Int(bytes[offset+5]) << 8) | (Int(bytes[offset+6]) << 16) | (Int(bytes[offset+7]) << 24)
            
            if chunkSize < 0 || fileLimit < (offset + 8 + chunkSize) {
                throw WavParserError.invalidHeader
            }
            
            // "fmt " chunk
            if chunkId0 == 0x66 && chunkId1 == 0x6d && chunkId2 == 0x74 && chunkId3 == 0x20 {
                let formatCode = Int(bytes[offset+8]) | (Int(bytes[offset+9]) << 8)
                // 0xFFFE は WAVE_FORMAT_EXTENSIBLE。実体は末尾のサブフォーマット GUID が示し、
                // 先頭 2 バイトが 1 なら PCM。macOS の afconvert 等はこの形式で書き出す
                var isPCM = (formatCode == 1)
                if formatCode == 0xFFFE && 40 <= chunkSize {
                    let subFormat = Int(bytes[offset+32]) | (Int(bytes[offset+33]) << 8)
                    isPCM = (subFormat == 1)
                }
                if isPCM != true {
                    throw WavParserError.unsupportedFormat("Only PCM is supported, got format \(formatCode)")
                }
                channels = Int(bytes[offset+10]) | (Int(bytes[offset+11]) << 8)
                sampleRate = Int(bytes[offset+12]) | (Int(bytes[offset+13]) << 8) | (Int(bytes[offset+14]) << 16) | (Int(bytes[offset+15]) << 24)
                bitsPerSample = Int(bytes[offset+22]) | (Int(bytes[offset+23]) << 8)
            }
            
            // "data" chunk
            if chunkId0 == 0x64 && chunkId1 == 0x61 && chunkId2 == 0x74 && chunkId3 == 0x61 {
                dataOffset = offset + 8
                dataSize = chunkSize
            }
            
            offset += (8 + chunkSize)
        }
        
        if sampleRate == 0 || channels == 0 || bitsPerSample == 0 {
            throw WavParserError.fmtChunkNotFound
        }
        if dataOffset == 0 {
            throw WavParserError.dataChunkNotFound
        }
        if bitsPerSample != 16 {
            throw WavParserError.unsupportedFormat("Only 16-bit PCM is supported, got \(bitsPerSample)")
        }
        
        let bytesPerSample = 2
        let totalSamples = (dataSize / (bytesPerSample * channels))
        var pcmData = [Float](repeating: 0.0, count: totalSamples)
        
        let scale: Float = 1.0 / 32768.0
        
        bytes.withUnsafeBufferPointer { bytePtr in
            let rawBytes = bytePtr.baseAddress!
            pcmData.withUnsafeMutableBufferPointer { pcmPtr in
                let dst = pcmPtr.baseAddress!
                
                switch channels {
                case 1:
                    Self.decodeMono(
                        rawBytes: rawBytes,
                        dst: dst,
                        dataOffset: dataOffset,
                        totalSamples: totalSamples,
                        fileLimit: fileLimit,
                        scale: scale
                    )
                case 2:
                    Self.decodeStereoDownmix(
                        rawBytes: rawBytes,
                        dst: dst,
                        dataOffset: dataOffset,
                        totalSamples: totalSamples,
                        fileLimit: fileLimit,
                        scale: scale
                    )
                default:
                    break
                }
            }
        }
        
        return WavData(
            sampleRate: sampleRate,
            channels: channels,
            bitsPerSample: bitsPerSample,
            pcmData: pcmData
        )
    }
    
    @inline(__always)
    private static func decodeMono(
        rawBytes: UnsafePointer<UInt8>,
        dst: UnsafeMutablePointer<Float>,
        dataOffset: Int,
        totalSamples: Int,
        fileLimit: Int,
        scale: Float
    ) {
        let width = 8
        let limit = totalSamples - (totalSamples % width)
        var i = 0
        
        while i < limit {
            let sampleIndex = dataOffset + (i * 2)
            if fileLimit < (sampleIndex + (width * 2)) {
                break
            }
            
            let pRaw = rawBytes.advanced(by: sampleIndex).withMemoryRebound(to: Int16.self, capacity: width) { ptr in
                return ptr
            }
            
            let s0 = Float(pRaw[0]) * scale
            let s1 = Float(pRaw[1]) * scale
            let s2 = Float(pRaw[2]) * scale
            let s3 = Float(pRaw[3]) * scale
            let s4 = Float(pRaw[4]) * scale
            let s5 = Float(pRaw[5]) * scale
            let s6 = Float(pRaw[6]) * scale
            let s7 = Float(pRaw[7]) * scale
            
            let v = SIMD8<Float>(s0, s1, s2, s3, s4, s5, s6, s7)
            dst[i+0] = v[0]
            dst[i+1] = v[1]
            dst[i+2] = v[2]
            dst[i+3] = v[3]
            dst[i+4] = v[4]
            dst[i+5] = v[5]
            dst[i+6] = v[6]
            dst[i+7] = v[7]
            i += width
        }
        
        while i < totalSamples {
            let sampleIndex = dataOffset + (i * 2)
            if fileLimit < (sampleIndex + 2) {
                break
            }
            let b0 = UInt16(rawBytes[sampleIndex+0])
            let b1 = UInt16(rawBytes[sampleIndex+1])
            let rawVal = Int16(bitPattern: b0 | (b1 << 8))
            dst[i] = Float(rawVal) * scale
            i += 1
        }
    }
    
    @inline(__always)
    private static func decodeStereoDownmix(
        rawBytes: UnsafePointer<UInt8>,
        dst: UnsafeMutablePointer<Float>,
        dataOffset: Int,
        totalSamples: Int,
        fileLimit: Int,
        scale: Float
    ) {
        let halfScale = scale * 0.5
        var i = 0
        while i < totalSamples {
            let sampleIndex = dataOffset + (i * 4)
            if fileLimit < (sampleIndex + 4) {
                break
            }
            let l0 = UInt16(rawBytes[sampleIndex+0])
            let l1 = UInt16(rawBytes[sampleIndex+1])
            let r0 = UInt16(rawBytes[sampleIndex+2])
            let r1 = UInt16(rawBytes[sampleIndex+3])
            
            let rawL = Float(Int16(bitPattern: l0 | (l1 << 8)))
            let rawR = Float(Int16(bitPattern: r0 | (r1 << 8)))
            dst[i] = (rawL + rawR) * halfScale
            i += 1
        }
    }
}
