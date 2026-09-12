import XCTest
@testable import Spiketrans

final class WavStreamReaderTests: XCTestCase {
    private var tempDir: String = ""

    override func setUp() {
        super.setUp()
        tempDir = (NSTemporaryDirectory() as NSString).appendingPathComponent("spiketrans_wavstream_\(UUID().uuidString)")
        try? FileManager.default.createDirectory(atPath: tempDir, withIntermediateDirectories: true)
    }

    override func tearDown() {
        try? FileManager.default.removeItem(atPath: tempDir)
        super.tearDown()
    }

    private func writeFile(_ bytes: [UInt8], name: String) throws -> String {
        let path = (tempDir as NSString).appendingPathComponent(name)
        try Data(bytes).write(to: URL(fileURLWithPath: path))
        return path
    }

    func testClassicMonoMatchesParser() throws {
        var samples = [Int16](repeating: 0, count: 320)
        var i = 0
        while i < samples.count {
            samples[i] = Int16(i * 80)
            i += 1
        }
        let bytes = classicMono16(samples: samples)
        let path = try writeFile(bytes, name: "clip.wav")
        let parsed = try WavParser().parse(bytes: bytes)
        let streamed = try WavStreamReader(filePath: path).readToEnd()
        XCTAssertEqual(streamed.sampleRate, parsed.sampleRate)
        XCTAssertEqual(streamed.pcmData.count, parsed.pcmData.count)
        XCTAssertEqual(streamed.pcmData[1], parsed.pcmData[1], accuracy: 1e-6)
    }

    func testJunkChunkBeforeFmtMatchesParser() throws {
        let bytes = junkThenPcm16()
        let path = try writeFile(bytes, name: "junk.wav")
        let parsed = try WavParser().parse(bytes: bytes)
        let streamed = try WavStreamReader(filePath: path).readToEnd()
        XCTAssertEqual(parsed.pcmData.count, 8)
        XCTAssertEqual(streamed.pcmData, parsed.pcmData)
        XCTAssertEqual(streamed.pcmData[0], 4096.0 / 32768.0, accuracy: 1e-6)
    }

    func testInt24MatchesParser() throws {
        var payload: [UInt8] = []
        var i = 0
        while i < 4 {
            payload.append(contentsOf: [0, 0x00, 0x10])
            i += 1
        }
        let bytes = buildCustomWav(audioFormat: 1, bitsPerSample: 24, payload: payload)
        let path = try writeFile(bytes, name: "int24.wav")
        let parsed = try WavParser().parse(bytes: bytes)
        let streamed = try WavStreamReader(filePath: path).readToEnd()
        XCTAssertEqual(streamed.pcmData, parsed.pcmData)
        XCTAssertEqual(streamed.pcmData[0], 4096.0 / 32768.0, accuracy: 1e-6)
    }

    func testFloat32MatchesParser() throws {
        var payload: [UInt8] = []
        var i = 0
        while i < 4 {
            let bits = Float(0.5).bitPattern
            payload.append(UInt8(bits & 0xFF))
            payload.append(UInt8((bits >> 8) & 0xFF))
            payload.append(UInt8((bits >> 16) & 0xFF))
            payload.append(UInt8((bits >> 24) & 0xFF))
            i += 1
        }
        let bytes = buildCustomWav(audioFormat: 3, bitsPerSample: 32, payload: payload)
        let path = try writeFile(bytes, name: "f32.wav")
        let parsed = try WavParser().parse(bytes: bytes)
        let streamed = try WavStreamReader(filePath: path).readToEnd()
        XCTAssertEqual(streamed.pcmData, parsed.pcmData)
        XCTAssertEqual(streamed.pcmData[0], 0.5, accuracy: 1e-6)
    }

    func testChunkedReadThenEOF() throws {
        let bytes = classicMono16(samples: [0, 80, 160, 240, 320])
        let path = try writeFile(bytes, name: "short.wav")
        let stream = try WavStreamReader(filePath: path)
        var buf = [Float](repeating: 0.0, count: 3)
        let n1 = try buf.withUnsafeMutableBufferPointer { p in
            try stream.readSamples(into: p.baseAddress!, maxSamples: 3)
        }
        XCTAssertEqual(n1, 3)
        let n2 = try buf.withUnsafeMutableBufferPointer { p in
            try stream.readSamples(into: p.baseAddress!, maxSamples: 3)
        }
        XCTAssertEqual(n2, 2)
        let n3 = try buf.withUnsafeMutableBufferPointer { p in
            try stream.readSamples(into: p.baseAddress!, maxSamples: 3)
        }
        XCTAssertEqual(n3, 0)
        try stream.close()
    }

    func testRejectsNonWav() throws {
        let path = try writeFile([UInt8](repeating: 0, count: 44), name: "not.wav")
        XCTAssertThrowsError(try WavStreamReader(filePath: path)) { error in
            XCTAssertEqual(error as? WavParserError, .invalidHeader)
        }
    }

    func testRejectsTruncatedHeader() throws {
        let path = try writeFile([0x52, 0x49, 0x46, 0x46], name: "short.wav")
        XCTAssertThrowsError(try WavStreamReader(filePath: path)) { error in
            XCTAssertEqual(error as? WavParserError, .invalidHeader)
        }
    }

    func testLoadWavFileHelper() throws {
        let bytes = classicMono16(samples: [0, 100, 200])
        let path = try writeFile(bytes, name: "helper.wav")
        let wav = SpeechDataset.loadWavFile(path: path)
        XCTAssertEqual(wav?.pcmData.count, 3)
        XCTAssertEqual(wav?.sampleRate, 16000)
    }

    private func classicMono16(samples: [Int16]) -> [UInt8] {
        var bytes: [UInt8] = []
        let dataSize = samples.count * 2
        let riffSize = 36 + dataSize
        bytes.append(contentsOf: [0x52, 0x49, 0x46, 0x46])
        appendU32(&bytes, UInt32(riffSize))
        bytes.append(contentsOf: [0x57, 0x41, 0x56, 0x45])
        bytes.append(contentsOf: [0x66, 0x6d, 0x74, 0x20])
        appendU32(&bytes, 16)
        bytes.append(contentsOf: [1, 0])
        bytes.append(contentsOf: [1, 0])
        appendU32(&bytes, 16000)
        appendU32(&bytes, 32000)
        bytes.append(contentsOf: [2, 0])
        bytes.append(contentsOf: [16, 0])
        bytes.append(contentsOf: [0x64, 0x61, 0x74, 0x61])
        appendU32(&bytes, UInt32(dataSize))
        var i = 0
        while i < samples.count {
            let uVal = UInt16(bitPattern: samples[i])
            bytes.append(UInt8(uVal & 0xFF))
            bytes.append(UInt8((uVal >> 8) & 0xFF))
            i += 1
        }
        return bytes
    }

    private func junkThenPcm16() -> [UInt8] {
        var bytes: [UInt8] = []
        bytes.append(contentsOf: [0x52, 0x49, 0x46, 0x46])
        appendU32(&bytes, 36 + 12 + 16)
        bytes.append(contentsOf: [0x57, 0x41, 0x56, 0x45])
        bytes.append(contentsOf: [0x4a, 0x55, 0x4e, 0x4b])
        appendU32(&bytes, 4)
        bytes.append(contentsOf: [0xDE, 0xAD, 0xBE, 0xEF])
        bytes.append(contentsOf: [0x66, 0x6d, 0x74, 0x20])
        appendU32(&bytes, 16)
        bytes.append(contentsOf: [1, 0, 1, 0])
        bytes.append(contentsOf: [0x80, 0x3E, 0, 0])
        bytes.append(contentsOf: [0x00, 0x7D, 0, 0])
        bytes.append(contentsOf: [2, 0, 16, 0])
        bytes.append(contentsOf: [0x64, 0x61, 0x74, 0x61])
        appendU32(&bytes, 16)
        var s = 0
        while s < 8 {
            bytes.append(0)
            bytes.append(0x10)
            s += 1
        }
        return bytes
    }

    private func buildCustomWav(audioFormat: UInt16, bitsPerSample: UInt16, payload: [UInt8]) -> [UInt8] {
        var bytes: [UInt8] = []
        bytes.append(contentsOf: [0x52, 0x49, 0x46, 0x46])
        appendU32(&bytes, UInt32(36 + payload.count))
        bytes.append(contentsOf: [0x57, 0x41, 0x56, 0x45])
        bytes.append(contentsOf: [0x66, 0x6d, 0x74, 0x20])
        appendU32(&bytes, 16)
        bytes.append(UInt8(audioFormat & 0xFF))
        bytes.append(UInt8((audioFormat >> 8) & 0xFF))
        bytes.append(contentsOf: [1, 0])
        appendU32(&bytes, 16000)
        let byteRate = (16000 * Int(bitsPerSample)) / 8
        appendU32(&bytes, UInt32(byteRate))
        let blockAlign = bitsPerSample / 8
        bytes.append(UInt8(blockAlign & 0xFF))
        bytes.append(UInt8((blockAlign >> 8) & 0xFF))
        bytes.append(UInt8(bitsPerSample & 0xFF))
        bytes.append(UInt8((bitsPerSample >> 8) & 0xFF))
        bytes.append(contentsOf: [0x64, 0x61, 0x74, 0x61])
        appendU32(&bytes, UInt32(payload.count))
        bytes.append(contentsOf: payload)
        return bytes
    }

    private func appendU32(_ bytes: inout [UInt8], _ v: UInt32) {
        bytes.append(UInt8(v & 0xFF))
        bytes.append(UInt8((v >> 8) & 0xFF))
        bytes.append(UInt8((v >> 16) & 0xFF))
        bytes.append(UInt8((v >> 24) & 0xFF))
    }
}
