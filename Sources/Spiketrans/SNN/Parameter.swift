import Foundation

/// CPU 推論用ネットワークの重み配列
public final class Parameter: @unchecked Sendable {
    public var data: [Float]
    public let count: Int

    public init(count: Int, initialData: [Float]? = nil) {
        self.count = count
        if let initD = initialData {
            self.data = initD
        } else {
            self.data = [Float](repeating: 0.0, count: count)
        }
    }
}
