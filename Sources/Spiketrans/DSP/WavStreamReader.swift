import Foundation

/// WAV ファイルストリームをチャンク単位で読み出すストリーミングリーダー
public final class WavStreamReader: Sendable {
    public let reader: any ReadCloser
    public let sampleRate: Int
    public let numChannels: Int
    public let bitsPerSample: Int
    public let dataByteCount: Int
    public let totalSamples: Int

    private let isClosedBox: BoxBool

    private final class BoxBool: @unchecked Sendable {
        var value: Bool
        init(_ value: Bool) {
            self.value = value
        }
    }

    public init(reader: any ReadCloser) throws {
        self.reader = reader
        self.isClosedBox = BoxBool(false)

        // 1. WAV ヘッダー 44 バイトの読み込み
        var headerBuffer = [UInt8](repeating: 0, count: 44)
        var totalHeaderRead = 0

        try headerBuffer.withUnsafeMutableBufferPointer { bufPtr in
            let base = bufPtr.baseAddress!
            while totalHeaderRead < 44 {
                let n = try reader.read(into: base.advanced(by: totalHeaderRead), count: 44 - totalHeaderRead)
                if n <= 0 {
                    throw NSError(
                        domain: "WavStreamReaderError",
                        code: -1,
                        userInfo: [NSLocalizedDescriptionKey: "Unexpected EOF while reading WAV header"]
                    )
                }
                totalHeaderRead += n
            }
        }

        // 2. RIFF / WAVE マジックチェック
        let isRiff = headerBuffer[0] == 0x52 && headerBuffer[1] == 0x49 && headerBuffer[2] == 0x46 && headerBuffer[3] == 0x46 // "RIFF"
        let isWave = headerBuffer[8] == 0x57 && headerBuffer[9] == 0x41 && headerBuffer[10] == 0x56 && headerBuffer[11] == 0x45 // "WAVE"
        if isRiff != true || isWave != true {
            try? reader.close()
            throw NSError(
                domain: "WavStreamReaderError",
                code: -2,
                userInfo: [NSLocalizedDescriptionKey: "Invalid WAV format: Missing RIFF/WAVE header"]
            )
        }

        // 3. チャンネル数、サンプリングレート、ビット深度の取得 (リトルエンディアン)
        let channels = Int(headerBuffer[22]) | (Int(headerBuffer[23]) << 8)
        let sRate = Int(headerBuffer[24]) | (Int(headerBuffer[25]) << 8) | (Int(headerBuffer[26]) << 16) | (Int(headerBuffer[27]) << 24)
        let bits = Int(headerBuffer[34]) | (Int(headerBuffer[35]) << 8)
        let dataLen = Int(headerBuffer[40]) | (Int(headerBuffer[41]) << 8) | (Int(headerBuffer[42]) << 16) | (Int(headerBuffer[43]) << 24)

        self.sampleRate = sRate
        self.numChannels = channels
        self.bitsPerSample = bits
        self.dataByteCount = dataLen

        let bytesPerSample = (bits / 8) * channels
        if 0 < bytesPerSample {
            self.totalSamples = dataLen / bytesPerSample
        } else {
            self.totalSamples = 0
        }
    }

    /// ファイルパスから直接開くコンビニエンスイニシャライザ
    public convenience init(filePath: String) throws {
        let fileReader = try FileReadCloser(path: filePath)
        try self.init(reader: fileReader)
    }

    /// 最大 maxSamples 個のモノラル PCM Float サンプルをバッファに読み出し (戻り値は読み込んだサンプル数、0 は EOF)
    public func readSamples(
        into buffer: UnsafeMutablePointer<Float>,
        maxSamples: Int
    ) throws -> Int {
        if isClosedBox.value {
            return 0
        }
        if maxSamples <= 0 {
            return 0
        }

        // 16-bit モノラル または ステレオの処理
        let bytesToRead = maxSamples * (bitsPerSample / 8) * numChannels
        var byteBuffer = [UInt8](repeating: 0, count: bytesToRead)

        var totalBytes = 0
        try byteBuffer.withUnsafeMutableBufferPointer { bPtr in
            let base = bPtr.baseAddress!
            while totalBytes < bytesToRead {
                let n = try reader.read(into: base.advanced(by: totalBytes), count: bytesToRead - totalBytes)
                if n <= 0 {
                    break // EOF
                }
                totalBytes += n
            }
        }

        if totalBytes <= 0 {
            return 0
        }

        let bytesPerFrame = (bitsPerSample / 8) * numChannels
        let framesRead = totalBytes / bytesPerFrame
        let invScale: Float = 1.0 / 32768.0

        byteBuffer.withUnsafeBufferPointer { rawPtr in
            let rawBase = rawPtr.baseAddress!
            var f = 0
            while f < framesRead {
                switch numChannels {
                case 1:
                    let byteIdx = f * 2
                    let low = Int16(rawBase[byteIdx])
                    let high = Int16(Int8(bitPattern: rawBase[byteIdx + 1])) << 8
                    let int16Val = high | (low & 0x00FF)
                    buffer[f] = Float(int16Val) * invScale
                default:
                    // ステレオダウンミックス
                    let byteIdx = f * 4
                    let lLow = Int16(rawBase[byteIdx])
                    let lHigh = Int16(Int8(bitPattern: rawBase[byteIdx + 1])) << 8
                    let left = Float(lHigh | (lLow & 0x00FF)) * invScale

                    let rLow = Int16(rawBase[byteIdx + 2])
                    let rHigh = Int16(Int8(bitPattern: rawBase[byteIdx + 3])) << 8
                    let right = Float(rHigh | (rLow & 0x00FF)) * invScale

                    buffer[f] = (left + right) * 0.5
                }
                f += 1
            }
        }

        return framesRead
    }

    /// ストリームを閉じる
    public func close() throws {
        if isClosedBox.value != true {
            isClosedBox.value = true
            try reader.close()
        }
    }
}
