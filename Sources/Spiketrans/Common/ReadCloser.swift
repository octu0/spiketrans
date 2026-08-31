import Foundation
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

/// ストリーム読み込みおよびリソース解放を抽象化するプロトコル (Go の io.ReadCloser 相当)
public protocol ReadCloser: AnyObject, Sendable {
    /// 指定されたバッファに最大 count バイトを読み込む (戻り値 0 は EOF)
    func read(into buffer: UnsafeMutablePointer<UInt8>, count: Int) throws -> Int

    /// ストリームを閉じ、保持しているリソース (ファイルディスクリプタ等) を解放する
    func close() throws
}

/// ファイルディスクリプタを用いた Pure Swift ファイルストリームリーダー
public final class FileReadCloser: ReadCloser, @unchecked Sendable {
    private var fd: Int32
    private var isClosed: Bool

    public init(path: String) throws {
        let descriptor = path.withCString { cPath in
            open(cPath, O_RDONLY)
        }
        if descriptor < 0 {
            throw NSError(
                domain: "FileReadCloserError",
                code: Int(errno),
                userInfo: [NSLocalizedDescriptionKey: "Failed to open file at path: \(path) (errno: \(errno))"]
            )
        }
        self.fd = descriptor
        self.isClosed = false
    }

    deinit {
        if isClosed != true {
            _ = Darwin.close(fd)
        }
    }

    /// 最大 count バイトをバッファに読み出し
    public func read(into buffer: UnsafeMutablePointer<UInt8>, count: Int) throws -> Int {
        if isClosed {
            throw NSError(
                domain: "FileReadCloserError",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Attempted to read from a closed FileReadCloser"]
            )
        }
        if count <= 0 {
            return 0
        }

        let bytesRead = Darwin.read(fd, buffer, count)
        switch true {
        case bytesRead < 0:
            throw NSError(
                domain: "FileReadCloserError",
                code: Int(errno),
                userInfo: [NSLocalizedDescriptionKey: "Failed to read from file descriptor (errno: \(errno))"]
            )
        default:
            return bytesRead
        }
    }

    /// ファイルを閉じる
    public func close() throws {
        if isClosed != true {
            let res = Darwin.close(fd)
            isClosed = true
            if res < 0 {
                throw NSError(
                    domain: "FileReadCloserError",
                    code: Int(errno),
                    userInfo: [NSLocalizedDescriptionKey: "Failed to close file descriptor (errno: \(errno))"]
                )
            }
        }
    }
}

/// メモリバッファをストリーミング形式で読み出す ReadCloser
public final class MemoryReadCloser: ReadCloser, @unchecked Sendable {
    private let data: [UInt8]
    private var position: Int
    private var isClosed: Bool

    public init(data: [UInt8]) {
        self.data = data
        self.position = 0
        self.isClosed = false
    }

    public func read(into buffer: UnsafeMutablePointer<UInt8>, count: Int) throws -> Int {
        if isClosed {
            throw NSError(
                domain: "MemoryReadCloserError",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Attempted to read from a closed MemoryReadCloser"]
            )
        }
        if count <= 0 {
            return 0
        }

        let available = data.count - position
        if available <= 0 {
            return 0
        }

        let toRead = min(available, count)
        data.withUnsafeBufferPointer { srcPtr in
            let srcBase = srcPtr.baseAddress!.advanced(by: position)
            buffer.initialize(from: srcBase, count: toRead)
        }
        position += toRead
        return toRead
    }

    public func close() throws {
        isClosed = true
    }
}
