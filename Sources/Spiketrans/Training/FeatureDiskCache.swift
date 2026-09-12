import Foundation

/// 音響特徴のディスクキャッシュ。パスを FNV-1a で 2 階層 (`ab/cd/<hash>.feat`) に置く。
public final class FeatureDiskCache: @unchecked Sendable {
    public let baseDirectory: String
    private let lock = NSLock()
    private var _allowedPaths: Set<String>? = nil

    /// 32 バイト固定ヘッダー定義およびバリデーション
    private struct Header {
        static let magic: UInt32 = 0x53504B46 // "SPKF"
        static let version: UInt32 = 2
        static let size = 32

        let frameStack: Int
        let frameDim: Int
        let frameCount: Int
        let elementCount: Int
        let totalBytes: Int

        init?(from rawBuffer: UnsafeRawBufferPointer, expectedStack: Int) {
            if rawBuffer.count < Self.size {
                return nil
            }
            let m = rawBuffer.loadUnaligned(fromByteOffset: 0, as: UInt32.self)
            let v = rawBuffer.loadUnaligned(fromByteOffset: 4, as: UInt32.self)
            let fs = Int(rawBuffer.loadUnaligned(fromByteOffset: 8, as: UInt32.self))
            let fd = Int(rawBuffer.loadUnaligned(fromByteOffset: 12, as: UInt32.self))
            let nf = Int(rawBuffer.loadUnaligned(fromByteOffset: 16, as: UInt32.self))
            let dt = rawBuffer.loadUnaligned(fromByteOffset: 20, as: UInt32.self)

            if m != Self.magic || v != Self.version || fs != expectedStack || fd <= 0 || 4096 < fd || nf <= 0 || 100_000_000 < nf || dt != 0 {
                return nil
            }

            let (elemCount, overflow1) = nf.multipliedReportingOverflow(by: fd)
            if overflow1 || elemCount <= 0 {
                return nil
            }
            let (payloadBytes, overflow2) = elemCount.multipliedReportingOverflow(by: MemoryLayout<Float>.size)
            if overflow2 || payloadBytes <= 0 {
                return nil
            }
            let (totBytes, overflow3) = Self.size.addingReportingOverflow(payloadBytes)
            if overflow3 {
                return nil
            }

            self.frameStack = fs
            self.frameDim = fd
            self.frameCount = nf
            self.elementCount = elemCount
            self.totalBytes = totBytes
        }
    }

    /// キャッシュ対象のホワイトリスト（指定時はこのパスのみ保存・再利用）
    public var allowedPaths: Set<String>? {
        get {
            lock.lock()
            defer { lock.unlock() }
            return _allowedPaths
        }
        set {
            lock.lock()
            defer { lock.unlock() }
            if let paths = newValue {
                _allowedPaths = Set(paths.map { ($0 as NSString).standardizingPath })
            } else {
                _allowedPaths = nil
            }
        }
    }

    public init(baseDirectory: String, allowedPaths: Set<String>? = nil) {
        self.baseDirectory = (baseDirectory as NSString).standardizingPath
        if let paths = allowedPaths {
            self._allowedPaths = Set(paths.map { ($0 as NSString).standardizingPath })
        } else {
            self._allowedPaths = nil
        }
    }

    /// 指定されたパスがキャッシュ対象（ホワイトリスト内）であるかを判定
    public func isAllowed(path: String) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        if let allowed = _allowedPaths {
            let normalized = (path as NSString).standardizingPath
            return allowed.contains(normalized)
        }
        return true
    }

    private static func fnv1a64(_ string: String, seed: UInt64) -> UInt64 {
        var hash = seed
        for byte in string.utf8 {
            hash ^= UInt64(byte)
            hash = hash &* 1099511628211
        }
        return hash
    }

    /// キャッシュキーと保存先ファイルパスの導出
    public func cacheFilePath(for path: String, frameStack: Int) -> String {
        let normalizedPath = (path as NSString).standardizingPath
        let keyString = "\(normalizedPath):stack=\(frameStack)"
        let h1 = Self.fnv1a64(keyString, seed: 14695981039346656037)
        let h2 = Self.fnv1a64(keyString, seed: 1099511628211)
        let hex = String(format: "%016llx%016llx", h1, h2)

        let prefix1 = String(hex.prefix(2))
        let prefix2 = String(hex.dropFirst(2).prefix(2))
        return "\(baseDirectory)/\(prefix1)/\(prefix2)/\(hex).feat"
    }

    /// キャッシュファイルのヘッダー情報からフレーム数のみを取得
    ///
    /// データセット初期化時に全 WAV を展開することなく、瞬時にフレーム数を確定する
    public func getFrameCount(path: String, frameStack: Int) -> Int? {
        if isAllowed(path: path) != true {
            return nil
        }
        let filePath = cacheFilePath(for: path, frameStack: frameStack)
        guard let fileHandle = try? FileHandle(forReadingFrom: URL(fileURLWithPath: filePath)) else {
            return nil
        }
        defer {
            try? fileHandle.close()
        }
        guard let headerData = try? fileHandle.read(upToCount: Header.size) else {
            return nil
        }
        let fileSize = (try? fileHandle.seekToEnd()) ?? 0
        return headerData.withUnsafeBytes { rawBuffer -> Int? in
            guard let header = Header(from: rawBuffer, expectedStack: frameStack) else {
                return nil
            }
            if fileSize < UInt64(header.totalBytes) {
                return nil
            }
            return header.frameCount
        }
    }

    /// キャッシュから特徴量系列 [frames][dim] を読込
    public func load(path: String, frameStack: Int) -> [[Float]]? {
        if isAllowed(path: path) != true {
            return nil
        }
        let filePath = cacheFilePath(for: path, frameStack: frameStack)
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: filePath)) else {
            return nil
        }
        return data.withUnsafeBytes { rawBuffer -> [[Float]]? in
            guard let header = Header(from: rawBuffer, expectedStack: frameStack) else {
                return nil
            }
            if rawBuffer.count < header.totalBytes {
                return nil
            }

            let floatBase = rawBuffer.baseAddress!.advanced(by: Header.size).bindMemory(to: Float.self, capacity: header.elementCount)
            var result = [[Float]]()
            result.reserveCapacity(header.frameCount)

            let fDim = header.frameDim
            var f = 0
            while f < header.frameCount {
                let rowPtr = floatBase.advanced(by: f * fDim)
                let row = [Float](unsafeUninitializedCapacity: fDim) { destBuf, initializedCount in
                    destBuf.baseAddress!.update(from: rowPtr, count: fDim)
                    initializedCount = fDim
                }
                result.append(row)
                f += 1
            }
            return result
        }
    }

    /// 特徴量系列をアトミックにディスクキャッシュへ保存
    @discardableResult
    public func save(path: String, frameStack: Int, features: [[Float]]) -> Bool {
        if isAllowed(path: path) != true {
            return false
        }
        if features.isEmpty {
            return false
        }
        let nFrames = features.count
        let fDim = features[0].count
        if nFrames <= 0 || 100_000_000 < nFrames || fDim <= 0 || 4096 < fDim || frameStack <= 0 {
            return false
        }

        // 不正・不揃いな系列長（ragged array）を検知してデータ破損を未然に防止
        var validateIdx = 0
        while validateIdx < nFrames {
            if features[validateIdx].count != fDim {
                return false
            }
            validateIdx += 1
        }

        let (elemCount, overflow1) = nFrames.multipliedReportingOverflow(by: fDim)
        if overflow1 || elemCount <= 0 {
            return false
        }
        let (payloadBytes, overflow2) = elemCount.multipliedReportingOverflow(by: MemoryLayout<Float>.size)
        if overflow2 || payloadBytes <= 0 {
            return false
        }
        let (totalBytes, overflow3) = Header.size.addingReportingOverflow(payloadBytes)
        if overflow3 {
            return false
        }

        let filePath = cacheFilePath(for: path, frameStack: frameStack)
        let parentDir = (filePath as NSString).deletingLastPathComponent
        if FileManager.default.fileExists(atPath: parentDir) != true {
            do {
                try FileManager.default.createDirectory(atPath: parentDir, withIntermediateDirectories: true)
            } catch {
                return false
            }
        }

        var data = Data(count: totalBytes)

        data.withUnsafeMutableBytes { rawBuffer in
            rawBuffer.storeBytes(of: Header.magic, toByteOffset: 0, as: UInt32.self)
            rawBuffer.storeBytes(of: Header.version, toByteOffset: 4, as: UInt32.self)
            rawBuffer.storeBytes(of: UInt32(frameStack), toByteOffset: 8, as: UInt32.self)
            rawBuffer.storeBytes(of: UInt32(fDim), toByteOffset: 12, as: UInt32.self)
            rawBuffer.storeBytes(of: UInt32(nFrames), toByteOffset: 16, as: UInt32.self)
            rawBuffer.storeBytes(of: UInt32(0), toByteOffset: 20, as: UInt32.self) // Float32
            rawBuffer.storeBytes(of: UInt64(0), toByteOffset: 24, as: UInt64.self) // Reserved

            let floatBase = rawBuffer.baseAddress!.advanced(by: Header.size).bindMemory(to: Float.self, capacity: elemCount)
            var f = 0
            while f < nFrames {
                let row = features[f]
                row.withUnsafeBufferPointer { rowBuf in
                    floatBase.advanced(by: f * fDim).update(from: rowBuf.baseAddress!, count: fDim)
                }
                f += 1
            }
        }

        do {
            try data.write(to: URL(fileURLWithPath: filePath), options: .atomic)
            return true
        } catch {
            return false
        }
    }

    /// 指定された音声パスに対応するキャッシュファイルをディスクから削除
    @discardableResult
    public func remove(path: String, frameStack: Int) -> Bool {
        let filePath = cacheFilePath(for: path, frameStack: frameStack)
        if FileManager.default.fileExists(atPath: filePath) {
            do {
                try FileManager.default.removeItem(atPath: filePath)
                return true
            } catch {
                return false
            }
        }
        return false
    }

    /// 指定されたフレーム数とスタック数におけるキャッシュファイル容量
    public static func estimateFileBytes(frameCount: Int, frameStack: Int) -> Int64 {
        let fDim = StreamingFeatureFrontEnd.tapDim * frameStack
        let (elemCount, overflow1) = frameCount.multipliedReportingOverflow(by: fDim)
        if overflow1 || elemCount <= 0 {
            return 0
        }
        let (payloadBytes, overflow2) = elemCount.multipliedReportingOverflow(by: MemoryLayout<Float>.size)
        if overflow2 || payloadBytes <= 0 {
            return 0
        }
        let (totalBytes, overflow3) = Header.size.addingReportingOverflow(payloadBytes)
        if overflow3 {
            return 0
        }
        return Int64(totalBytes)
    }
}
