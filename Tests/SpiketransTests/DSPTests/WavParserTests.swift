import XCTest
@testable import Spiketrans

final class WavParserTests: XCTestCase {
    
    /// ヘルパー: 16-bit PCM WAV バイナリの構築
    private func createWavBytes(
        sampleRate: Int = 16000,
        channels: Int = 1,
        bitsPerSample: Int = 16,
        pcmSamples: [Int16]
    ) -> [UInt8] {
        var bytes: [UInt8] = []
        
        let dataSize = pcmSamples.count * 2
        let riffSize = 36 + dataSize
        
        // "RIFF"
        bytes.append(contentsOf: [0x52, 0x49, 0x46, 0x46])
        // RIFF size (Little Endian)
        bytes.append(UInt8(riffSize & 0xFF))
        bytes.append(UInt8((riffSize >> 8) & 0xFF))
        bytes.append(UInt8((riffSize >> 16) & 0xFF))
        bytes.append(UInt8((riffSize >> 24) & 0xFF))
        // "WAVE"
        bytes.append(contentsOf: [0x57, 0x41, 0x56, 0x45])
        
        // "fmt "
        bytes.append(contentsOf: [0x66, 0x6d, 0x74, 0x20])
        // Subchunk1Size (16 for PCM)
        bytes.append(contentsOf: [16, 0, 0, 0])
        // AudioFormat (1 = PCM)
        bytes.append(contentsOf: [1, 0])
        // NumChannels
        bytes.append(UInt8(channels & 0xFF))
        bytes.append(UInt8((channels >> 8) & 0xFF))
        // SampleRate
        bytes.append(UInt8(sampleRate & 0xFF))
        bytes.append(UInt8((sampleRate >> 8) & 0xFF))
        bytes.append(UInt8((sampleRate >> 16) & 0xFF))
        bytes.append(UInt8((sampleRate >> 24) & 0xFF))
        // ByteRate
        let byteRate = (sampleRate * channels * bitsPerSample) / 8
        bytes.append(UInt8(byteRate & 0xFF))
        bytes.append(UInt8((byteRate >> 8) & 0xFF))
        bytes.append(UInt8((byteRate >> 16) & 0xFF))
        bytes.append(UInt8((byteRate >> 24) & 0xFF))
        // BlockAlign
        let blockAlign = (channels * bitsPerSample) / 8
        bytes.append(UInt8(blockAlign & 0xFF))
        bytes.append(UInt8((blockAlign >> 8) & 0xFF))
        // BitsPerSample
        bytes.append(UInt8(bitsPerSample & 0xFF))
        bytes.append(UInt8((bitsPerSample >> 8) & 0xFF))
        
        // "data"
        bytes.append(contentsOf: [0x64, 0x61, 0x74, 0x61])
        // Subchunk2Size
        bytes.append(UInt8(dataSize & 0xFF))
        bytes.append(UInt8((dataSize >> 8) & 0xFF))
        bytes.append(UInt8((dataSize >> 16) & 0xFF))
        bytes.append(UInt8((dataSize >> 24) & 0xFF))
        
        // Data samples
        var i = 0
        while i < pcmSamples.count {
            let sample = pcmSamples[i]
            let uVal = UInt16(bitPattern: sample)
            bytes.append(UInt8(uVal & 0xFF))
            bytes.append(UInt8((uVal >> 8) & 0xFF))
            i += 1
        }
        
        return bytes
    }
    
    func testParse16kMonoPCM() throws {
        let parser = WavParser()
        let sampleCount = 320
        var samples = [Int16](repeating: 0, count: sampleCount)
        var i = 0
        while i < sampleCount {
            samples[i] = Int16(i * 100)
            i += 1
        }
        
        let wavBytes = createWavBytes(sampleRate: 16000, channels: 1, bitsPerSample: 16, pcmSamples: samples)
        let wavData = try parser.parse(bytes: wavBytes)
        
        XCTAssertEqual(wavData.sampleRate, 16000)
        XCTAssertEqual(wavData.channels, 1)
        XCTAssertEqual(wavData.bitsPerSample, 16)
        XCTAssertEqual(wavData.pcmData.count, sampleCount)
        
        let expected0: Float = Float(samples[0]) / 32768.0
        let expected10: Float = Float(samples[10]) / 32768.0
        XCTAssertLessThan(abs(wavData.pcmData[0] - expected0), 1e-4)
        XCTAssertLessThan(abs(wavData.pcmData[10] - expected10), 1e-4)
    }
    
    func testParse16kStereoDownmix() throws {
        let parser = WavParser()
        let frameCount = 100
        var stereoSamples = [Int16](repeating: 0, count: frameCount * 2)
        var i = 0
        while i < frameCount {
            stereoSamples[i * 2] = 1000 // Left
            stereoSamples[(i * 2) + 1] = 2000 // Right
            i += 1
        }
        
        let wavBytes = createWavBytes(sampleRate: 16000, channels: 2, bitsPerSample: 16, pcmSamples: stereoSamples)
        let wavData = try parser.parse(bytes: wavBytes)
        
        XCTAssertEqual(wavData.sampleRate, 16000)
        XCTAssertEqual(wavData.channels, 2)
        XCTAssertEqual(wavData.pcmData.count, frameCount)
        
        // ダウンミックス期待値: (1000 + 2000) / 2 = 1500 / 32768
        let expected: Float = 1500.0 / 32768.0
        var k = 0
        while k < frameCount {
            XCTAssertLessThan(abs(wavData.pcmData[k] - expected), 1e-4)
            k += 1
        }
    }
    
    func testRejectInvalidMagic() {
        let parser = WavParser()
        var corrupted = createWavBytes(sampleRate: 16000, channels: 1, bitsPerSample: 16, pcmSamples: [100, 200])
        corrupted[0] = 0x00 // Corrupt 'R'
        
        XCTAssertThrowsError(try parser.parse(bytes: corrupted)) { error in
            XCTAssertEqual(error as? WavParserError, WavParserError.invalidHeader)
        }
    }
    
    func testRejectUnsupportedFormat() {
        let parser = WavParser()
        // Format 3 (IEEE Float)
        var bytes = createWavBytes(sampleRate: 16000, channels: 1, bitsPerSample: 16, pcmSamples: [100, 200])
        bytes[20] = 3 // Format code 3
        
        XCTAssertThrowsError(try parser.parse(bytes: bytes))
    }
    
    func testRejectTruncatedData() {
        let parser = WavParser()
        let wavBytes = createWavBytes(sampleRate: 16000, channels: 1, bitsPerSample: 16, pcmSamples: [100, 200, 300])
        let truncated = Array(wavBytes[0..<30]) // 30 bytes < 44 bytes
        
        XCTAssertThrowsError(try parser.parse(bytes: truncated)) { error in
            XCTAssertEqual(error as? WavParserError, WavParserError.invalidHeader)
        }
    }
    
    func testEmptyData() {
        let parser = WavParser()
        let emptyBytes: [UInt8] = []
        XCTAssertThrowsError(try parser.parse(bytes: emptyBytes)) { error in
            XCTAssertEqual(error as? WavParserError, WavParserError.invalidHeader)
        }
    }
}
