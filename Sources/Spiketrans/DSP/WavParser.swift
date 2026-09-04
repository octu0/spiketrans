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
        var isFloatFormat = false
        
        var offset = 12
        let fileLimit = bytes.count
        
        while (offset + 8) <= fileLimit {
            let chunkId0 = bytes[offset+0]
            let chunkId1 = bytes[offset+1]
            let chunkId2 = bytes[offset+2]
            let chunkId3 = bytes[offset+3]
            
            let chunkSize = Int(bytes[offset+4]) | (Int(bytes[offset+5]) << 8) | (Int(bytes[offset+6]) << 16) | (Int(bytes[offset+7]) << 24)
            
            if chunkSize < 0 || fileLimit < (offset + 8 + chunkSize) {
                // fmt と data が揃っていれば、末尾の壊れたメタデータは無視して打ち切る。
                // 録音機材が書く bext/iXML/id3 等は破損していることがある
                if 0 < dataOffset && 0 < sampleRate {
                    break
                }
                throw WavParserError.invalidHeader
            }
            
            // "fmt " chunk
            if chunkId0 == 0x66 && chunkId1 == 0x6d && chunkId2 == 0x74 && chunkId3 == 0x20 {
                let formatCode = Int(bytes[offset+8]) | (Int(bytes[offset+9]) << 8)
                // 1 = 整数 PCM, 3 = IEEE 浮動小数。
                // 0xFFFE は WAVE_FORMAT_EXTENSIBLE で、実体は末尾のサブフォーマット GUID が示す
                // (macOS の afconvert 等はこの形式で書き出す)
                var effectiveFormat = formatCode
                if formatCode == 0xFFFE && 40 <= chunkSize {
                    effectiveFormat = Int(bytes[offset+32]) | (Int(bytes[offset+33]) << 8)
                }
                if effectiveFormat != 1 && effectiveFormat != 3 {
                    throw WavParserError.unsupportedFormat("Only integer PCM and IEEE float are supported, got format \(formatCode)")
                }
                isFloatFormat = (effectiveFormat == 3)
                channels = Int(bytes[offset+10]) | (Int(bytes[offset+11]) << 8)
                sampleRate = Int(bytes[offset+12]) | (Int(bytes[offset+13]) << 8) | (Int(bytes[offset+14]) << 16) | (Int(bytes[offset+15]) << 24)
                bitsPerSample = Int(bytes[offset+22]) | (Int(bytes[offset+23]) << 8)
            }
            
            // "data" chunk
            if chunkId0 == 0x64 && chunkId1 == 0x61 && chunkId2 == 0x74 && chunkId3 == 0x61 {
                dataOffset = offset + 8
                dataSize = chunkSize
            }
            
            // RIFF はチャンクをワード境界に揃えるため、奇数長なら詰め物が 1 バイト入る
            offset += 8 + chunkSize + (chunkSize & 1)
        }
        
        if sampleRate == 0 || channels == 0 || bitsPerSample == 0 {
            throw WavParserError.fmtChunkNotFound
        }
        if dataOffset == 0 {
            throw WavParserError.dataChunkNotFound
        }
        if isFloatFormat {
            if bitsPerSample != 32 {
                throw WavParserError.unsupportedFormat("Only 32-bit IEEE float is supported, got \(bitsPerSample)")
            }
        } else {
            if bitsPerSample != 16 && bitsPerSample != 24 && bitsPerSample != 32 {
                throw WavParserError.unsupportedFormat("Only 16/24/32-bit integer PCM is supported, got \(bitsPerSample)")
            }
        }

        let bytesPerSample = bitsPerSample / 8
        let totalSamples = (dataSize / (bytesPerSample * channels))
        var pcmData = [Float](repeating: 0.0, count: totalSamples)
        
        let scale: Float = 1.0 / 32768.0
        
        bytes.withUnsafeBufferPointer { bytePtr in
            let rawBytes = bytePtr.baseAddress!
            pcmData.withUnsafeMutableBufferPointer { pcmPtr in
                let dst = pcmPtr.baseAddress!

                if isFloatFormat {
                    // 32bit 浮動小数は既に -1.0〜1.0 の範囲なのでスケール不要
                    Self.decodeFloat32(
                        rawBytes: rawBytes,
                        dst: dst,
                        dataOffset: dataOffset,
                        totalSamples: totalSamples,
                        channels: channels,
                        fileLimit: fileLimit
                    )
                    return
                }

                // 24/32bit 整数は上位ビットだけを見れば 16bit 相当に落とせる
                if bitsPerSample != 16 {
                    Self.decodeWideInteger(
                        rawBytes: rawBytes,
                        dst: dst,
                        dataOffset: dataOffset,
                        totalSamples: totalSamples,
                        channels: channels,
                        bytesPerSample: bytesPerSample,
                        fileLimit: fileLimit
                    )
                    return
                }

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
    
    /// 24bit / 32bit 整数 PCM のデコード。多チャネルは平均でモノラルへ落とす。
    /// リトルエンディアンの最上位 2 バイトを 16bit サンプルとして読む
    @inline(__always)
    private static func decodeWideInteger(
        rawBytes: UnsafePointer<UInt8>,
        dst: UnsafeMutablePointer<Float>,
        dataOffset: Int,
        totalSamples: Int,
        channels: Int,
        bytesPerSample: Int,
        fileLimit: Int
    ) {
        let frameBytes = bytesPerSample * channels
        let invChannels = 1.0 / Float(channels)
        let scale: Float = 1.0 / 32768.0
        let highOffset = bytesPerSample - 2
        var i = 0
        while i < totalSamples {
            let base = dataOffset + (i * frameBytes)
            if fileLimit < (base + frameBytes) {
                break
            }
            var sum: Float = 0.0
            var ch = 0
            while ch < channels {
                let o = base + (ch * bytesPerSample) + highOffset
                let raw = UInt16(rawBytes[o]) | (UInt16(rawBytes[o+1]) << 8)
                sum += Float(Int16(bitPattern: raw)) * scale
                ch += 1
            }
            dst[i] = sum * invChannels
            i += 1
        }
    }

    /// 32bit IEEE 浮動小数のデコード。多チャネルは平均でモノラルへ落とす
    @inline(__always)
    private static func decodeFloat32(
        rawBytes: UnsafePointer<UInt8>,
        dst: UnsafeMutablePointer<Float>,
        dataOffset: Int,
        totalSamples: Int,
        channels: Int,
        fileLimit: Int
    ) {
        let frameBytes = 4 * channels
        let invChannels = 1.0 / Float(channels)
        var i = 0
        while i < totalSamples {
            let base = dataOffset + (i * frameBytes)
            if fileLimit < (base + frameBytes) {
                break
            }
            var sum: Float = 0.0
            var ch = 0
            while ch < channels {
                let o = base + (ch * 4)
                let bits = UInt32(rawBytes[o])
                    | (UInt32(rawBytes[o+1]) << 8)
                    | (UInt32(rawBytes[o+2]) << 16)
                    | (UInt32(rawBytes[o+3]) << 24)
                sum += Float(bitPattern: bits)
                ch += 1
            }
            dst[i] = sum * invChannels
            i += 1
        }
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
