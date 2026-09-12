import XCTest
@testable import Spiketrans

final class FeatureDiskCacheTests: XCTestCase {
    var tempDir: String = ""

    override func setUp() {
        super.setUp()
        let uniqueId = UUID().uuidString
        tempDir = (NSTemporaryDirectory() as NSString).appendingPathComponent("spiketrans_test_cache_\(uniqueId)")
        try? FileManager.default.createDirectory(atPath: tempDir, withIntermediateDirectories: true)
    }

    override func tearDown() {
        try? FileManager.default.removeItem(atPath: tempDir)
        super.tearDown()
    }

    // MARK: - 1. 基本的な保存と読み込みのラウンドトリップ検証

    func testSaveAndLoadRoundTrip() {
        let cache = FeatureDiskCache(baseDirectory: tempDir)
        let numFrames = 30
        let frameDim = 512
        var dummyFeatures = [[Float]]()
        dummyFeatures.reserveCapacity(numFrames)

        var f = 0
        while f < numFrames {
            var row = [Float](repeating: 0.0, count: frameDim)
            var d = 0
            while d < frameDim {
                row[d] = Float(f * frameDim + d) * 0.01
                d += 1
            }
            dummyFeatures.append(row)
            f += 1
        }

        let testAudioPath = "/path/to/test_utterance_001.wav"
        let saved = cache.save(path: testAudioPath, frameStack: 4, features: dummyFeatures)
        XCTAssertTrue(saved)

        // キャッシュからの読み出し検証
        let loaded = cache.load(path: testAudioPath, frameStack: 4)
        XCTAssertNotNil(loaded)
        guard let features = loaded else {
            return
        }

        XCTAssertEqual(features.count, numFrames)
        XCTAssertEqual(features[0].count, frameDim)

        var checkFrame = 0
        while checkFrame < numFrames {
            var checkDim = 0
            while checkDim < frameDim {
                let expected = Float(checkFrame * frameDim + checkDim) * 0.01
                let actual = features[checkFrame][checkDim]
                XCTAssertEqual(actual, expected, accuracy: 1e-6)
                checkDim += 1
            }
            checkFrame += 1
        }
    }

    // MARK: - 2. 32 バイトヘッダーのみによるフレーム数高速取得検証

    func testGetFrameCountHeaderOnly() {
        let cache = FeatureDiskCache(baseDirectory: tempDir)
        let numFrames = 75
        let frameDim = 128
        var dummyFeatures = [[Float]]()
        var f = 0
        while f < numFrames {
            dummyFeatures.append([Float](repeating: 0.5, count: frameDim))
            f += 1
        }

        let testAudioPath = "/path/to/test_utterance_002.wav"
        cache.save(path: testAudioPath, frameStack: 1, features: dummyFeatures)

        let count = cache.getFrameCount(path: testAudioPath, frameStack: 1)
        XCTAssertEqual(count, numFrames)
    }

    // MARK: - 3. キャッシュミスおよび異なる frameStack での独立性検証

    func testCacheMissAndFrameStackSeparation() {
        let cache = FeatureDiskCache(baseDirectory: tempDir)
        let testAudioPath = "/path/to/test_utterance_003.wav"

        // 存在しないキー
        XCTAssertNil(cache.load(path: testAudioPath, frameStack: 4))
        XCTAssertNil(cache.getFrameCount(path: testAudioPath, frameStack: 4))

        // frameStack = 4 で保存
        let dummy = [[Float]](repeating: [Float](repeating: 1.0, count: 512), count: 20)
        cache.save(path: testAudioPath, frameStack: 4, features: dummy)

        // frameStack = 2 で照会した場合は nil であること
        XCTAssertNil(cache.load(path: testAudioPath, frameStack: 2))
        XCTAssertNil(cache.getFrameCount(path: testAudioPath, frameStack: 2))

        // frameStack = 4 では正しく取得できること
        XCTAssertNotNil(cache.load(path: testAudioPath, frameStack: 4))
        XCTAssertEqual(cache.getFrameCount(path: testAudioPath, frameStack: 4), 20)
    }

    // MARK: - 4. 破損ファイル・不完全データの安全な拒絶テスト

    func testCorruptedAndTruncatedFiles() {
        let cache = FeatureDiskCache(baseDirectory: tempDir)
        let testAudioPath = "/path/to/corrupt_test.wav"
        let filePath = cache.cacheFilePath(for: testAudioPath, frameStack: 4)
        let parentDir = (filePath as NSString).deletingLastPathComponent
        try? FileManager.default.createDirectory(atPath: parentDir, withIntermediateDirectories: true)

        // A. 32 バイト未満の極小ファイル
        let tinyData = Data([0x53, 0x50, 0x4B])
        try? tinyData.write(to: URL(fileURLWithPath: filePath))
        XCTAssertNil(cache.load(path: testAudioPath, frameStack: 4))
        XCTAssertNil(cache.getFrameCount(path: testAudioPath, frameStack: 4))

        // B. マジックナンバー破損 (SPKF でない)
        var badMagicData = Data(count: 32 + (10 * 128 * 4))
        badMagicData.withUnsafeMutableBytes { rawBuf in
            let u32 = rawBuf.baseAddress!.bindMemory(to: UInt32.self, capacity: 6)
            u32[0] = 0xDEADBEEF // 不正マジック
            u32[1] = 1
            u32[2] = 4
            u32[3] = 128
            u32[4] = 10
            u32[5] = 0
        }
        try? badMagicData.write(to: URL(fileURLWithPath: filePath))
        XCTAssertNil(cache.load(path: testAudioPath, frameStack: 4))
        XCTAssertNil(cache.getFrameCount(path: testAudioPath, frameStack: 4))

        // C. ペイロードが途中で切れている (Truncated)
        var truncatedData = Data(count: 32 + (10 * 128 * 2)) // 本来 4 バイト必要なところ 2 バイト
        truncatedData.withUnsafeMutableBytes { rawBuf in
            let u32 = rawBuf.baseAddress!.bindMemory(to: UInt32.self, capacity: 6)
            u32[0] = 0x53504B46 // "SPKF"
            u32[1] = 1
            u32[2] = 4
            u32[3] = 128
            u32[4] = 10
            u32[5] = 0
        }
        try? truncatedData.write(to: URL(fileURLWithPath: filePath))
        XCTAssertNil(cache.load(path: testAudioPath, frameStack: 4))
        XCTAssertNil(cache.getFrameCount(path: testAudioPath, frameStack: 4))
    }

    // MARK: - 5. ディレクトリシャーディング階層構造の検証

    func testDirectoryShardingStructure() {
        let cache = FeatureDiskCache(baseDirectory: tempDir)
        let testPath = "/audio/corpus/sample_0099.wav"
        let filePath = cache.cacheFilePath(for: testPath, frameStack: 4)

        // tempDir/xx/yy/<hash>.feat の 2 階層サブディレクトリ構造であることを検証
        let relative = filePath.replacingOccurrences(of: tempDir + "/", with: "")
        let components = relative.components(separatedBy: "/")
        XCTAssertEqual(components.count, 3)
        XCTAssertEqual(components[0].count, 2)
        XCTAssertEqual(components[1].count, 2)
        XCTAssertTrue(components[2].hasSuffix(".feat"))
    }

    // MARK: - 6. マルチスレッド並行アクセステスト

    func testConcurrentMultiThreadedAccess() {
        let cache = FeatureDiskCache(baseDirectory: tempDir)
        let concurrency = 8
        let iterations = 10

        DispatchQueue.concurrentPerform(iterations: concurrency) { worker in
            var iter = 0
            while iter < iterations {
                let path = "/test/audio_\(worker)_\(iter).wav"
                let feat = [[Float]](repeating: [Float](repeating: Float(worker + iter), count: 64), count: 15)
                cache.save(path: path, frameStack: 1, features: feat)

                let loaded = cache.load(path: path, frameStack: 1)
                XCTAssertNotNil(loaded)
                if let l = loaded {
                    XCTAssertEqual(l.count, 15)
                    XCTAssertEqual(l[0].count, 64)
                    XCTAssertEqual(l[0][0], Float(worker + iter))
                }

                let count = cache.getFrameCount(path: path, frameStack: 1)
                XCTAssertEqual(count, 15)
                iter += 1
            }
        }
    }

    // MARK: - 7. SpeechDataset との連携検証

    func testSpeechDatasetLazyWithFeatureCache() {
        let cache = FeatureDiskCache(baseDirectory: tempDir)

        // 1 秒分 (16,000 サンプル) の正弦波 WAV を一時ファイルに生成
        let wavPath = (tempDir as NSString).appendingPathComponent("synth_speech.wav")
        let sampleRate = 16000
        var pcm = [Float](repeating: 0.0, count: sampleRate)
        var s = 0
        while s < sampleRate {
            let t = Float(s) / Float(sampleRate)
            pcm[s] = sin(2.0 * Float.pi * 440.0 * t) * 0.4
            s += 1
        }

        var int16Data = [Int16](repeating: 0, count: sampleRate)
        var p = 0
        while p < sampleRate {
            int16Data[p] = Int16(max(-32768.0, min(32767.0, pcm[p] * 32767.0)))
            p += 1
        }

        var wavBytes = [UInt8]()
        wavBytes.append(contentsOf: [0x52, 0x49, 0x46, 0x46]) // RIFF
        let totalChunkSize = UInt32(36 + sampleRate * 2)
        withUnsafeBytes(of: totalChunkSize.littleEndian) { wavBytes.append(contentsOf: $0) }
        wavBytes.append(contentsOf: [0x57, 0x41, 0x56, 0x45]) // WAVE
        wavBytes.append(contentsOf: [0x66, 0x6D, 0x74, 0x20]) // fmt
        let fmtSize = UInt32(16)
        withUnsafeBytes(of: fmtSize.littleEndian) { wavBytes.append(contentsOf: $0) }
        let audioFormat = UInt16(1)
        withUnsafeBytes(of: audioFormat.littleEndian) { wavBytes.append(contentsOf: $0) }
        let numChannels = UInt16(1)
        withUnsafeBytes(of: numChannels.littleEndian) { wavBytes.append(contentsOf: $0) }
        let sr = UInt32(sampleRate)
        withUnsafeBytes(of: sr.littleEndian) { wavBytes.append(contentsOf: $0) }
        let byteRate = UInt32(sampleRate * 2)
        withUnsafeBytes(of: byteRate.littleEndian) { wavBytes.append(contentsOf: $0) }
        let blockAlign = UInt16(2)
        withUnsafeBytes(of: blockAlign.littleEndian) { wavBytes.append(contentsOf: $0) }
        let bitsPerSample = UInt16(16)
        withUnsafeBytes(of: bitsPerSample.littleEndian) { wavBytes.append(contentsOf: $0) }
        wavBytes.append(contentsOf: [0x64, 0x61, 0x74, 0x61]) // data
        let dataSize = UInt32(sampleRate * 2)
        withUnsafeBytes(of: dataSize.littleEndian) { wavBytes.append(contentsOf: $0) }
        int16Data.withUnsafeBufferPointer { buf in
            let rawBytes = UnsafeRawBufferPointer(buf)
            wavBytes.append(contentsOf: rawBytes)
        }

        try? Data(wavBytes).write(to: URL(fileURLWithPath: wavPath))

        let vocab = TextVocabulary(corpus: ["テスト"])
        let pairs = [(path: wavPath, text: "テスト")]

        // 1 回目: キャッシュなし状態からの構築 (キャッシュへ自動書き込み)
        let ds1 = SpeechDataset.lazyFromManifest(
            pairs: pairs,
            textVocabulary: vocab,
            frameStack: 4,
            workers: 1,
            cache: cache
        )
        XCTAssertEqual(ds1.count, 1)
        let frames1 = ds1.frameCount(at: 0)
        XCTAssertTrue(0 < frames1)

        // キャッシュファイルが存在することを確認
        let cachedCount = cache.getFrameCount(path: wavPath, frameStack: 4)
        XCTAssertEqual(cachedCount, frames1)

        // 2 回目: キャッシュからの高速構築 (WAV を再展開せずヘッダーのみでメタデータ構築)
        let ds2 = SpeechDataset.lazyFromManifest(
            pairs: pairs,
            textVocabulary: vocab,
            frameStack: 4,
            workers: 1,
            cache: cache
        )
        XCTAssertEqual(ds2.count, 1)
        XCTAssertEqual(ds2.frameCount(at: 0), frames1)

        // loadFeatures がディスクキャッシュから特徴量を正しく取得すること
        let (pcmDirect, featDirect) = SpeechDataset.loadFeatures(
            path: wavPath,
            frameStack: 4,
            cache: cache,
            loadPCM: false
        )
        XCTAssertEqual(featDirect.count, frames1)
        XCTAssertTrue(pcmDirect.isEmpty) // loadPCM=false なので空
    }

    // MARK: - 8. 整数オーバーフロー攻撃および不揃い系列 (Ragged Array) の拒絶検証

    func testIntegerOverflowAndRaggedArrayRejection() {
        let cache = FeatureDiskCache(baseDirectory: tempDir)

        // A. 悪意ある/破損ヘッダー (極大なフレーム数・次元数による乗算オーバーフロー誘発)
        let overflowPath = "/path/to/overflow_attack.wav"
        let filePath = cache.cacheFilePath(for: overflowPath, frameStack: 4)
        let parentDir = (filePath as NSString).deletingLastPathComponent
        try? FileManager.default.createDirectory(atPath: parentDir, withIntermediateDirectories: true)

        var badHeader = Data(count: 32)
        badHeader.withUnsafeMutableBytes { rawBuf in
            let u32 = rawBuf.baseAddress!.bindMemory(to: UInt32.self, capacity: 6)
            u32[0] = 0x53504B46 // "SPKF"
            u32[1] = 1
            u32[2] = 4
            u32[3] = 0x7FFFFFFF // 極大な次元数
            u32[4] = 0x7FFFFFFF // 極大なフレーム数
            u32[5] = 0
        }
        try? badHeader.write(to: URL(fileURLWithPath: filePath))
        // 乗算オーバーフローでクラッシュせず安全に nil を返すこと
        XCTAssertNil(cache.load(path: overflowPath, frameStack: 4))
        XCTAssertNil(cache.getFrameCount(path: overflowPath, frameStack: 4))

        // B. 各行の次元が不揃いな配列 (Ragged Array) を保存しようとした場合の拒絶
        let raggedFeatures: [[Float]] = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0] // 行長が異なる
        ]
        let raggedSaved = cache.save(path: "/path/to/ragged.wav", frameStack: 1, features: raggedFeatures)
        XCTAssertFalse(raggedSaved)

        // C. 空の特徴量配列の保存拒絶
        let emptySaved = cache.save(path: "/path/to/empty.wav", frameStack: 1, features: [])
        XCTAssertFalse(emptySaved)
    }

    // MARK: - 9. パス正規化および PCM オンデマンド読出制御の検証

    func testPathStandardizationAndPCMLoading() {
        // 末尾スラッシュ付きベースディレクトリでも同一に正規化されること
        let cacheWithSlash = FeatureDiskCache(baseDirectory: tempDir + "/")
        let cacheWithoutSlash = FeatureDiskCache(baseDirectory: tempDir)

        let testPath1 = "./relative/audio.wav"
        let testPath2 = "relative/audio.wav"

        let path1 = cacheWithSlash.cacheFilePath(for: testPath1, frameStack: 4)
        let path2 = cacheWithoutSlash.cacheFilePath(for: testPath2, frameStack: 4)
        XCTAssertEqual(path1, path2)

        // SpeechDataset の subscript と sample(at:loadPCM:) の挙動検証
        let wavPath = (tempDir as NSString).appendingPathComponent("synth_pcm_test.wav")
        let sampleRate = 16000
        let pcm = [Float](repeating: 0.25, count: sampleRate)
        var int16Data = [Int16](repeating: 0, count: sampleRate)
        var p = 0
        while p < sampleRate {
            int16Data[p] = Int16(pcm[p] * 32767.0)
            p += 1
        }

        var wavBytes = [UInt8]()
        wavBytes.append(contentsOf: [0x52, 0x49, 0x46, 0x46]) // RIFF
        let totalChunkSize = UInt32(36 + sampleRate * 2)
        withUnsafeBytes(of: totalChunkSize.littleEndian) { wavBytes.append(contentsOf: $0) }
        wavBytes.append(contentsOf: [0x57, 0x41, 0x56, 0x45, 0x66, 0x6D, 0x74, 0x20]) // WAVEfmt 
        let fmtSize = UInt32(16)
        withUnsafeBytes(of: fmtSize.littleEndian) { wavBytes.append(contentsOf: $0) }
        let audioFormat = UInt16(1)
        withUnsafeBytes(of: audioFormat.littleEndian) { wavBytes.append(contentsOf: $0) }
        let numChannels = UInt16(1)
        withUnsafeBytes(of: numChannels.littleEndian) { wavBytes.append(contentsOf: $0) }
        let sr = UInt32(sampleRate)
        withUnsafeBytes(of: sr.littleEndian) { wavBytes.append(contentsOf: $0) }
        let byteRate = UInt32(sampleRate * 2)
        withUnsafeBytes(of: byteRate.littleEndian) { wavBytes.append(contentsOf: $0) }
        let blockAlign = UInt16(2)
        withUnsafeBytes(of: blockAlign.littleEndian) { wavBytes.append(contentsOf: $0) }
        let bitsPerSample = UInt16(16)
        withUnsafeBytes(of: bitsPerSample.littleEndian) { wavBytes.append(contentsOf: $0) }
        wavBytes.append(contentsOf: [0x64, 0x61, 0x74, 0x61]) // data
        let dataSize = UInt32(sampleRate * 2)
        withUnsafeBytes(of: dataSize.littleEndian) { wavBytes.append(contentsOf: $0) }
        int16Data.withUnsafeBufferPointer { buf in
            wavBytes.append(contentsOf: UnsafeRawBufferPointer(buf))
        }
        try? Data(wavBytes).write(to: URL(fileURLWithPath: wavPath))

        let vocab = TextVocabulary(corpus: ["テスト"])
        let dataset = SpeechDataset.lazyFromManifest(
            pairs: [(path: wavPath, text: "テスト")],
            textVocabulary: vocab,
            frameStack: 4,
            workers: 1,
            cache: cacheWithoutSlash
        )

        // 既定の subscript では loadPCM: false となり、PCM 配列は空（WAV 読出オーバーヘッドゼロ）
        let sampleFast = dataset[0]
        XCTAssertTrue(sampleFast.audioPCM.isEmpty)
        XCTAssertFalse(sampleFast.acousticFeatures.isEmpty)

        // 明示的に loadPCM: true を指定した場合は PCM が読み込まれること
        let sampleWithPCM = dataset.sample(at: 0, loadPCM: true)
        XCTAssertEqual(sampleWithPCM.audioPCM.count, sampleRate)
        XCTAssertFalse(sampleWithPCM.acousticFeatures.isEmpty)
    }

    // MARK: - 10. ファイルサイズ試算 (estimateFileBytes) と実サイズの厳密一致検証

    func testEstimateFileBytesCalculation() {
        let frameStack = 4
        let numFrames = 25
        let estimated = FeatureDiskCache.estimateFileBytes(frameCount: numFrames, frameStack: frameStack)
        let frameDim = StreamingFeatureFrontEnd.tapDim * frameStack
        XCTAssertEqual(estimated, Int64(32 + numFrames * frameDim * MemoryLayout<Float>.size))

        let cache = FeatureDiskCache(baseDirectory: tempDir)
        let dummy = [[Float]](repeating: [Float](repeating: 1.0, count: frameDim), count: numFrames)
        let testPath = "/path/to/estimate_test.wav"
        let saved = cache.save(path: testPath, frameStack: frameStack, features: dummy)
        XCTAssertTrue(saved)

        let filePath = cache.cacheFilePath(for: testPath, frameStack: frameStack)
        let attrs = try? FileManager.default.attributesOfItem(atPath: filePath)
        let actualSize = (attrs?[.size] as? UInt64) ?? 0
        XCTAssertEqual(Int64(actualSize), estimated)

        // 境界値・不正値でのゼロ返却検証
        XCTAssertEqual(FeatureDiskCache.estimateFileBytes(frameCount: 0, frameStack: frameStack), 0)
        XCTAssertEqual(FeatureDiskCache.estimateFileBytes(frameCount: -10, frameStack: frameStack), 0)
    }

    // MARK: - 11. ホワイトリスト (allowedPaths) による保存・読出・メタデータフィルタ検証

    func testAllowedPathsFiltering() {
        let cache = FeatureDiskCache(baseDirectory: tempDir)
        let longPath = "/path/to/long_audio.wav"
        let shortPath = "/path/to/short_audio.wav"

        // 初期状態 (allowedPaths == nil) ではすべて許可
        XCTAssertTrue(cache.isAllowed(path: longPath))
        XCTAssertTrue(cache.isAllowed(path: shortPath))

        // longPath のみ許可するホワイトリストを設定
        cache.allowedPaths = [longPath]
        XCTAssertTrue(cache.isAllowed(path: longPath))
        XCTAssertFalse(cache.isAllowed(path: shortPath))

        let dummy = [[Float]](repeating: [Float](repeating: 0.5, count: 512), count: 20)

        // 許可されていない shortPath は保存が拒絶されること
        let savedShort = cache.save(path: shortPath, frameStack: 4, features: dummy)
        XCTAssertFalse(savedShort)
        XCTAssertNil(cache.load(path: shortPath, frameStack: 4))
        XCTAssertNil(cache.getFrameCount(path: shortPath, frameStack: 4))

        // 許可されている longPath は正常に保存・読込・ヘッダー取得できること
        let savedLong = cache.save(path: longPath, frameStack: 4, features: dummy)
        XCTAssertTrue(savedLong)
        XCTAssertNotNil(cache.load(path: longPath, frameStack: 4))
        XCTAssertEqual(cache.getFrameCount(path: longPath, frameStack: 4), 20)

        // ホワイトリストを nil に戻すとすべて許可されること
        cache.allowedPaths = nil
        XCTAssertTrue(cache.isAllowed(path: shortPath))
    }

    // MARK: - 12. キャッシュファイルの明示的削除 (remove) 検証

    func testCacheFileRemoval() {
        let cache = FeatureDiskCache(baseDirectory: tempDir)
        let testPath = "/path/to/delete_me.wav"
        let dummy = [[Float]](repeating: [Float](repeating: 0.1, count: 512), count: 10)

        cache.save(path: testPath, frameStack: 4, features: dummy)
        let filePath = cache.cacheFilePath(for: testPath, frameStack: 4)
        XCTAssertTrue(FileManager.default.fileExists(atPath: filePath))

        // 削除実行
        let removed = cache.remove(path: testPath, frameStack: 4)
        XCTAssertTrue(removed)
        XCTAssertFalse(FileManager.default.fileExists(atPath: filePath))

        // 存在しないファイルの削除は false
        let removedAgain = cache.remove(path: testPath, frameStack: 4)
        XCTAssertFalse(removedAgain)
    }

    // MARK: - 13. SpeechDataset.selectQuotaPaths の長尺優先選別ロジック検証

    func testSelectQuotaPathsLongestFirst() {
        let metas = [
            SpeechDataset.SampleMeta(
                path: "/audio/short.wav",
                rawText: "短",
                hiraganaText: "たん",
                textIds: [1],
                phonemeIds: [1],
                frameCount: 100 // ~0.2 MB
            ),
            SpeechDataset.SampleMeta(
                path: "/audio/long_a.wav",
                rawText: "長A",
                hiraganaText: "ちょうえー",
                textIds: [2],
                phonemeIds: [2],
                frameCount: 1000 // ~2.05 MB
            ),
            SpeechDataset.SampleMeta(
                path: "/audio/medium.wav",
                rawText: "中",
                hiraganaText: "ちゅう",
                textIds: [3],
                phonemeIds: [3],
                frameCount: 500 // ~1.02 MB
            ),
            SpeechDataset.SampleMeta(
                path: "/audio/long_b.wav",
                rawText: "長B",
                hiraganaText: "ちょうびー",
                textIds: [4],
                phonemeIds: [4],
                frameCount: 1000 // ~2.05 MB
            )
        ]

        // 1. 容量無制限 (maxGigabytes <= 0) の場合: 全件選別
        let unlimited = SpeechDataset.selectQuotaPaths(metas: metas, frameStack: 4, maxGigabytes: 0.0)
        XCTAssertEqual(unlimited.count, 4)

        // 2. 約 3.5 MB 上限の場合:
        // 1000 フレーム (~2.05 MB) のうち 1 件と、500 フレーム (~1.02 MB) の合計 ~3.07 MB が収まる
        // 3.5 MB = 3.5 / 1024 / 1024 GB ≈ 0.00334 GB
        let selected3MB = SpeechDataset.selectQuotaPaths(
            metas: metas,
            frameStack: 4,
            maxGigabytes: 3.5 / 1024.0
        )
        // 最長サンプル long_a または long_b が確実に選別されていること
        XCTAssertTrue(selected3MB.contains("/audio/long_a.wav") || selected3MB.contains("/audio/long_b.wav"))
        // 合計容量が上限以下であること
        var totalBytes: Int64 = 0
        for m in metas {
            if selected3MB.contains(m.path) {
                totalBytes += FeatureDiskCache.estimateFileBytes(frameCount: m.frameCount, frameStack: 4)
            }
        }
        let maxAllowedBytes = Int64((3.5 / 1024.0) * 1024.0 * 1024.0 * 1024.0)
        XCTAssertTrue(totalBytes <= maxAllowedBytes)

        // 3. 極小上限 (0.0001 GB ≈ 100 KB): どのサンプルも収まらない場合は空
        let selectedTiny = SpeechDataset.selectQuotaPaths(
            metas: metas,
            frameStack: 4,
            maxGigabytes: 0.0001
        )
        XCTAssertTrue(selectedTiny.isEmpty)
    }

    // MARK: - 14. SpeechDataset クォータキャッシュのエンドツーエンド動作検証

    func testSpeechDatasetQuotaCachingEndToEnd() {
        let cache = FeatureDiskCache(baseDirectory: tempDir)
        let sampleRate = 16000

        // 長尺音声 (1.0 秒 = 16,000 サンプル) と短尺音声 (0.2 秒 = 3,200 サンプル) を生成
        let longWavPath = (tempDir as NSString).appendingPathComponent("long_speech.wav")
        let shortWavPath = (tempDir as NSString).appendingPathComponent("short_speech.wav")

        func createWav(path: String, sampleCount: Int) {
            var pcm = [Float](repeating: 0.0, count: sampleCount)
            var s = 0
            while s < sampleCount {
                let t = Float(s) / Float(sampleRate)
                pcm[s] = sin(2.0 * Float.pi * 440.0 * t) * 0.3
                s += 1
            }
            var int16Data = [Int16](repeating: 0, count: sampleCount)
            var p = 0
            while p < sampleCount {
                int16Data[p] = Int16(pcm[p] * 32767.0)
                p += 1
            }
            var wavBytes = [UInt8]()
            wavBytes.append(contentsOf: [0x52, 0x49, 0x46, 0x46]) // RIFF
            let totalChunkSize = UInt32(36 + sampleCount * 2)
            withUnsafeBytes(of: totalChunkSize.littleEndian) { wavBytes.append(contentsOf: $0) }
            wavBytes.append(contentsOf: [0x57, 0x41, 0x56, 0x45, 0x66, 0x6D, 0x74, 0x20])
            let fmtSize = UInt32(16)
            withUnsafeBytes(of: fmtSize.littleEndian) { wavBytes.append(contentsOf: $0) }
            let audioFormat = UInt16(1)
            withUnsafeBytes(of: audioFormat.littleEndian) { wavBytes.append(contentsOf: $0) }
            let numChannels = UInt16(1)
            withUnsafeBytes(of: numChannels.littleEndian) { wavBytes.append(contentsOf: $0) }
            let sr = UInt32(sampleRate)
            withUnsafeBytes(of: sr.littleEndian) { wavBytes.append(contentsOf: $0) }
            let byteRate = UInt32(sampleRate * 2)
            withUnsafeBytes(of: byteRate.littleEndian) { wavBytes.append(contentsOf: $0) }
            let blockAlign = UInt16(2)
            withUnsafeBytes(of: blockAlign.littleEndian) { wavBytes.append(contentsOf: $0) }
            let bitsPerSample = UInt16(16)
            withUnsafeBytes(of: bitsPerSample.littleEndian) { wavBytes.append(contentsOf: $0) }
            wavBytes.append(contentsOf: [0x64, 0x61, 0x74, 0x61])
            let dataSize = UInt32(sampleCount * 2)
            withUnsafeBytes(of: dataSize.littleEndian) { wavBytes.append(contentsOf: $0) }
            int16Data.withUnsafeBufferPointer { buf in
                wavBytes.append(contentsOf: UnsafeRawBufferPointer(buf))
            }
            try? Data(wavBytes).write(to: URL(fileURLWithPath: path))
        }

        createWav(path: longWavPath, sampleCount: sampleRate)
        createWav(path: shortWavPath, sampleCount: 3200)

        let vocab = TextVocabulary(corpus: ["テスト"])
        let pairs = [
            (path: longWavPath, text: "テスト長い"),
            (path: shortWavPath, text: "テスト短い")
        ]

        // 長尺音声 (約 51 KB) のみ収まり、短尺音声 (~10 KB) を足すと超過する容量制約 (~55 KB)
        let quotaGB = 55_000.0 / (1024.0 * 1024.0 * 1024.0)
        let dataset = SpeechDataset.lazyFromManifest(
            pairs: pairs,
            textVocabulary: vocab,
            frameStack: 4,
            workers: 1,
            cache: cache,
            maxCacheGigabytes: quotaGB
        )

        XCTAssertEqual(dataset.count, 2)
        XCTAssertNotNil(cache.allowedPaths)
        guard let allowed = cache.allowedPaths else {
            XCTFail("allowedPaths should not be nil")
            return
        }

        // 長尺音声のみが選別され、短尺音声は選別対象外であること
        let normLong = (longWavPath as NSString).standardizingPath
        let normShort = (shortWavPath as NSString).standardizingPath
        XCTAssertTrue(allowed.contains(normLong))
        XCTAssertFalse(allowed.contains(normShort))

        // 各サンプルの特徴量読み込み（学習ステップを模擬）
        let longSample = dataset[0]
        let shortSample = dataset[1]
        XCTAssertFalse(longSample.acousticFeatures.isEmpty)
        XCTAssertFalse(shortSample.acousticFeatures.isEmpty)

        // 長尺音声のキャッシュファイルはディスクに生成されること
        let longCachePath = cache.cacheFilePath(for: longWavPath, frameStack: 4)
        XCTAssertTrue(FileManager.default.fileExists(atPath: longCachePath))

        // 短尺音声のキャッシュファイルはディスクに生成されないこと (オンデマンド抽出)
        let shortCachePath = cache.cacheFilePath(for: shortWavPath, frameStack: 4)
        XCTAssertFalse(FileManager.default.fileExists(atPath: shortCachePath))
    }

    // MARK: - 15. selectQuotaPaths における同一パス重複時のクォータ二重加算防止検証

    func testSelectQuotaPathsDeduplication() {
        // 同一パスのサンプルが複数回出現した場合 (同一音声に複数アノテーションがある場合等)
        // ディスク容量は 1 ファイル分のみ消費されるべきであり、二重加算されて他の長尺音声が弾かれてはならない
        let path1 = "/audio/duplicate_long.wav" // 1000 フレーム ≈ 2.05 MB
        let path2 = "/audio/second_long.wav"    // 800 フレーム ≈ 1.64 MB

        let metas = [
            SpeechDataset.SampleMeta(
                path: path1,
                rawText: "重複1",
                hiraganaText: "ちょうふく1",
                textIds: [1],
                phonemeIds: [1],
                frameCount: 1000
            ),
            SpeechDataset.SampleMeta(
                path: path1,
                rawText: "重複2",
                hiraganaText: "ちょうふく2",
                textIds: [1],
                phonemeIds: [1],
                frameCount: 1000
            ),
            SpeechDataset.SampleMeta(
                path: path2,
                rawText: "二番目",
                hiraganaText: "にばんめ",
                textIds: [2],
                phonemeIds: [2],
                frameCount: 800
            )
        ]

        // 1 ファイル目 (~2.05 MB) + 2 ファイル目 (~1.64 MB) の合計は約 3.69 MB。
        // 上限を 4.0 MB (≈ 0.0039 GB) に設定した場合、二重加算バグがあると
        // 2.05 + 2.05 = 4.10 MB となり path2 が弾かれてしまう。
        // 重複排除が正しければ、両方のパスが選別される。
        let quotaGB = 4.0 / 1024.0
        let selected = SpeechDataset.selectQuotaPaths(
            metas: metas,
            frameStack: 4,
            maxGigabytes: quotaGB
        )

        XCTAssertTrue(selected.contains(path1))
        XCTAssertTrue(selected.contains(path2))
        XCTAssertEqual(selected.count, 2)
    }

    // MARK: - 16. パス表記揺れ (相対パス・非正規化パス) におけるクォータ保護と誤削除防止

    func testSelectQuotaPathsPathNormalization() {
        let rawRelativePath = "./audio/../audio/sample_norm.wav"
        let standardPath = (rawRelativePath as NSString).standardizingPath

        let metas = [
            SpeechDataset.SampleMeta(
                path: rawRelativePath,
                rawText: "正規化テスト",
                hiraganaText: "せいきかてすと",
                textIds: [1],
                phonemeIds: [1],
                frameCount: 500
            )
        ]

        let selected = SpeechDataset.selectQuotaPaths(
            metas: metas,
            frameStack: 4,
            maxGigabytes: 1.0
        )

        // 標準化されたパスで選別集合に格納されていること
        XCTAssertTrue(selected.contains(standardPath))
    }

    // MARK: - 17. クォータ縮小時の旧キャッシュ自動退避 (Eviction) 検証

    func testQuotaReductionEviction() {
        let cache = FeatureDiskCache(baseDirectory: tempDir)
        let sampleRate = 16000

        let wavPath1 = (tempDir as NSString).appendingPathComponent("evict_long1.wav")
        let wavPath2 = (tempDir as NSString).appendingPathComponent("evict_long2.wav")

        func writeWav(path: String, samples: Int) {
            let pcm = [Float](repeating: 0.2, count: samples)
            var int16Data = [Int16](repeating: 0, count: samples)
            var p = 0
            while p < samples {
                int16Data[p] = Int16(pcm[p] * 32767.0)
                p += 1
            }
            var wavBytes = [UInt8]()
            wavBytes.append(contentsOf: [0x52, 0x49, 0x46, 0x46]) // RIFF
            let totalChunkSize = UInt32(36 + samples * 2)
            withUnsafeBytes(of: totalChunkSize.littleEndian) { wavBytes.append(contentsOf: $0) }
            wavBytes.append(contentsOf: [0x57, 0x41, 0x56, 0x45, 0x66, 0x6D, 0x74, 0x20])
            let fmtSize = UInt32(16)
            withUnsafeBytes(of: fmtSize.littleEndian) { wavBytes.append(contentsOf: $0) }
            let audioFormat = UInt16(1)
            withUnsafeBytes(of: audioFormat.littleEndian) { wavBytes.append(contentsOf: $0) }
            let numChannels = UInt16(1)
            withUnsafeBytes(of: numChannels.littleEndian) { wavBytes.append(contentsOf: $0) }
            let sr = UInt32(sampleRate)
            withUnsafeBytes(of: sr.littleEndian) { wavBytes.append(contentsOf: $0) }
            let byteRate = UInt32(sampleRate * 2)
            withUnsafeBytes(of: byteRate.littleEndian) { wavBytes.append(contentsOf: $0) }
            let blockAlign = UInt16(2)
            withUnsafeBytes(of: blockAlign.littleEndian) { wavBytes.append(contentsOf: $0) }
            let bitsPerSample = UInt16(16)
            withUnsafeBytes(of: bitsPerSample.littleEndian) { wavBytes.append(contentsOf: $0) }
            wavBytes.append(contentsOf: [0x64, 0x61, 0x74, 0x61])
            let dataSize = UInt32(samples * 2)
            withUnsafeBytes(of: dataSize.littleEndian) { wavBytes.append(contentsOf: $0) }
            int16Data.withUnsafeBufferPointer { buf in
                wavBytes.append(contentsOf: UnsafeRawBufferPointer(buf))
            }
            try? Data(wavBytes).write(to: URL(fileURLWithPath: path))
        }

        // 2 本の長尺音声を生成 (各 1 秒 = ~51 KB 特徴量)
        writeWav(path: wavPath1, samples: sampleRate)
        writeWav(path: wavPath2, samples: sampleRate)

        let vocab = TextVocabulary(corpus: ["テスト"])
        let pairs = [
            (path: wavPath1, text: "音声1"),
            (path: wavPath2, text: "音声2")
        ]

        // 1. 最初は両方収まる容量 (~120 KB) でデータセット構築
        let largeQuotaGB = 120_000.0 / (1024.0 * 1024.0 * 1024.0)
        let ds1 = SpeechDataset.lazyFromManifest(
            pairs: pairs,
            textVocabulary: vocab,
            frameStack: 4,
            workers: 1,
            cache: cache,
            maxCacheGigabytes: largeQuotaGB
        )
        // 両方の特徴量を読み込んでキャッシュを生成
        XCTAssertFalse(ds1[0].acousticFeatures.isEmpty)
        XCTAssertFalse(ds1[1].acousticFeatures.isEmpty)

        let cache1 = cache.cacheFilePath(for: wavPath1, frameStack: 4)
        let cache2 = cache.cacheFilePath(for: wavPath2, frameStack: 4)
        XCTAssertTrue(FileManager.default.fileExists(atPath: cache1))
        XCTAssertTrue(FileManager.default.fileExists(atPath: cache2))

        // 2. 次に 1 本しか収まらない極小クォータ (~60 KB) で再実行
        let smallQuotaGB = 60_000.0 / (1024.0 * 1024.0 * 1024.0)
        let ds2 = SpeechDataset.lazyFromManifest(
            pairs: pairs,
            textVocabulary: vocab,
            frameStack: 4,
            workers: 1,
            cache: cache,
            maxCacheGigabytes: smallQuotaGB
        )
        XCTAssertEqual(ds2.count, 2)

        // 収まった 1 本は残るが、上限を超過した 2 本目はディスクから削除されていること
        let exists1 = FileManager.default.fileExists(atPath: cache1)
        let exists2 = FileManager.default.fileExists(atPath: cache2)
        // どちらか一方のみが残り、もう一方は削除されていること
        XCTAssertTrue((exists1 && exists2 != true) || (exists2 && exists1 != true))
    }

    // MARK: - 18. 過去ホワイトリストが残存するキャッシュの再利用時のヘッダー直読検証

    func testReusedCacheWithStaleWhitelist() {
        let cache = FeatureDiskCache(baseDirectory: tempDir)
        let sampleRate = 16000
        let wavPath = (tempDir as NSString).appendingPathComponent("reused_cache.wav")

        let pcm = [Float](repeating: 0.1, count: sampleRate)
        var int16Data = [Int16](repeating: 0, count: sampleRate)
        var p = 0
        while p < sampleRate {
            int16Data[p] = Int16(pcm[p] * 32767.0)
            p += 1
        }
        var wavBytes = [UInt8]()
        wavBytes.append(contentsOf: [0x52, 0x49, 0x46, 0x46])
        let totalChunkSize = UInt32(36 + sampleRate * 2)
        withUnsafeBytes(of: totalChunkSize.littleEndian) { wavBytes.append(contentsOf: $0) }
        wavBytes.append(contentsOf: [0x57, 0x41, 0x56, 0x45, 0x66, 0x6D, 0x74, 0x20])
        let fmtSize = UInt32(16)
        withUnsafeBytes(of: fmtSize.littleEndian) { wavBytes.append(contentsOf: $0) }
        let audioFormat = UInt16(1)
        withUnsafeBytes(of: audioFormat.littleEndian) { wavBytes.append(contentsOf: $0) }
        let numChannels = UInt16(1)
        withUnsafeBytes(of: numChannels.littleEndian) { wavBytes.append(contentsOf: $0) }
        let sr = UInt32(sampleRate)
        withUnsafeBytes(of: sr.littleEndian) { wavBytes.append(contentsOf: $0) }
        let byteRate = UInt32(sampleRate * 2)
        withUnsafeBytes(of: byteRate.littleEndian) { wavBytes.append(contentsOf: $0) }
        let blockAlign = UInt16(2)
        withUnsafeBytes(of: blockAlign.littleEndian) { wavBytes.append(contentsOf: $0) }
        let bitsPerSample = UInt16(16)
        withUnsafeBytes(of: bitsPerSample.littleEndian) { wavBytes.append(contentsOf: $0) }
        wavBytes.append(contentsOf: [0x64, 0x61, 0x74, 0x61])
        let dataSize = UInt32(sampleRate * 2)
        withUnsafeBytes(of: dataSize.littleEndian) { wavBytes.append(contentsOf: $0) }
        int16Data.withUnsafeBufferPointer { buf in
            wavBytes.append(contentsOf: UnsafeRawBufferPointer(buf))
        }
        try? Data(wavBytes).write(to: URL(fileURLWithPath: wavPath))

        // まずキャッシュを作成
        let vocab = TextVocabulary(corpus: ["テスト"])
        let ds = SpeechDataset.lazyFromManifest(
            pairs: [(path: wavPath, text: "テスト")],
            textVocabulary: vocab,
            frameStack: 4,
            workers: 1,
            cache: cache,
            maxCacheGigabytes: 0.0
        )
        XCTAssertFalse(ds[0].acousticFeatures.isEmpty)

        // 過去の別データセット用のホワイトリスト（wavPath を含まない）がキャッシュに残存している状態をシミュレート
        cache.allowedPaths = ["/other/path.wav"]

        // この状態で新しい容量付きで lazyFromManifest を呼び出す
        let dsNew = SpeechDataset.lazyFromManifest(
            pairs: [(path: wavPath, text: "テスト")],
            textVocabulary: vocab,
            frameStack: 4,
            workers: 1,
            cache: cache,
            maxCacheGigabytes: 1.0
        )
        XCTAssertEqual(dsNew.count, 1)
        // 過去のホワイトリストによって阻害されることなく、キャッシュ対象として選別されること
        XCTAssertTrue(cache.isAllowed(path: wavPath))
    }

    // MARK: - 19. textIds(at:) による特徴量・音声デコードを伴わない高速テキスト取得検証

    func testDatasetTextIdsAccessWithoutAudioDecoding() {
        let text = "テスト音声"
        let vocab = TextVocabulary(corpus: [text])
        let expectedIds = vocab.textToIds(text)

        // 1. 遅延モード (lazyFromManifest) での検証
        let wavPath = (tempDir as NSString).appendingPathComponent("text_ids_test.wav")
        let sampleRate = 16000
        let pcm = [Float](repeating: 0.05, count: sampleRate)
        var int16Data = [Int16](repeating: 0, count: sampleRate)
        var p = 0
        while p < sampleRate {
            int16Data[p] = Int16(pcm[p] * 32767.0)
            p += 1
        }
        var wavBytes = [UInt8]()
        wavBytes.append(contentsOf: [0x52, 0x49, 0x46, 0x46])
        let totalChunkSize = UInt32(36 + sampleRate * 2)
        withUnsafeBytes(of: totalChunkSize.littleEndian) { wavBytes.append(contentsOf: $0) }
        wavBytes.append(contentsOf: [0x57, 0x41, 0x56, 0x45, 0x66, 0x6D, 0x74, 0x20])
        let fmtSize = UInt32(16)
        withUnsafeBytes(of: fmtSize.littleEndian) { wavBytes.append(contentsOf: $0) }
        let audioFormat = UInt16(1)
        withUnsafeBytes(of: audioFormat.littleEndian) { wavBytes.append(contentsOf: $0) }
        let numChannels = UInt16(1)
        withUnsafeBytes(of: numChannels.littleEndian) { wavBytes.append(contentsOf: $0) }
        let sr = UInt32(sampleRate)
        withUnsafeBytes(of: sr.littleEndian) { wavBytes.append(contentsOf: $0) }
        let byteRate = UInt32(sampleRate * 2)
        withUnsafeBytes(of: byteRate.littleEndian) { wavBytes.append(contentsOf: $0) }
        let blockAlign = UInt16(2)
        withUnsafeBytes(of: blockAlign.littleEndian) { wavBytes.append(contentsOf: $0) }
        let bitsPerSample = UInt16(16)
        withUnsafeBytes(of: bitsPerSample.littleEndian) { wavBytes.append(contentsOf: $0) }
        wavBytes.append(contentsOf: [0x64, 0x61, 0x74, 0x61])
        let dataSize = UInt32(sampleRate * 2)
        withUnsafeBytes(of: dataSize.littleEndian) { wavBytes.append(contentsOf: $0) }
        int16Data.withUnsafeBufferPointer { buf in
            wavBytes.append(contentsOf: UnsafeRawBufferPointer(buf))
        }
        try? Data(wavBytes).write(to: URL(fileURLWithPath: wavPath))

        let dsLazy = SpeechDataset.lazyFromManifest(
            pairs: [(path: wavPath, text: text)],
            textVocabulary: vocab,
            frameStack: 4,
            workers: 1
        )
        XCTAssertEqual(dsLazy.textIds(at: 0), expectedIds)
        XCTAssertEqual(dsLazy.hiraganaText(at: 0), KanjiConverter().convertToHiragana(text))

        // 2. 即時モード (eager) での検証
        let sample = AudioTextSample(
            audioPCM: [],
            rawText: text,
            hiraganaText: "てすとおんせい",
            textIds: expectedIds,
            phonemeIds: [],
            acousticFeatures: []
        )
        let dsEager = SpeechDataset(samples: [sample])
        XCTAssertEqual(dsEager.textIds(at: 0), expectedIds)
        XCTAssertEqual(dsEager.hiraganaText(at: 0), "てすとおんせい")
    }
}
