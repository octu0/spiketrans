import XCTest
@testable import Spiketrans

final class VectorOperationsTests: XCTestCase {
    
    // MARK: - Dot Product Tests
    
    func testDotProductEquivalence() {
        let testCounts = [0, 1, 7, 8, 9, 15, 16, 17, 319, 320, 321, 512, 1024]
        
        var tIdx = 0
        while tIdx < testCounts.count {
            let count = testCounts[tIdx]
            var a = [Float](repeating: 0.0, count: count)
            var b = [Float](repeating: 0.0, count: count)
            
            var i = 0
            while i < count {
                a[i] = sin(Float(i) * 0.1) * 2.5
                b[i] = cos(Float(i) * 0.2) * 1.8
                i += 1
            }
            
            // スカラー基準値
            var scalarSum: Float = 0.0
            i = 0
            while i < count {
                scalarSum += a[i] * b[i]
                i += 1
            }
            
            // SIMD8 実装
            var simdSum: Float = 0.0
            if 0 < count {
                simdSum = a.withUnsafeBufferPointer { aPtr in
                    b.withUnsafeBufferPointer { bPtr in
                        VectorOperations.dotProduct(a: aPtr.baseAddress!, b: bPtr.baseAddress!, count: count)
                    }
                }
            }
            
            let diff = abs(simdSum - scalarSum)
            let maxVal = max(abs(scalarSum), 1.0)
            let relError = diff / maxVal
            
            XCTAssertLessThan(relError, 1e-4, "DotProduct count=\(count) failed: simd=\(simdSum), scalar=\(scalarSum)")
            tIdx += 1
        }
    }
    
    func testDotProductCornerCases() {
        let count = 64
        var a = [Float](repeating: 0.0, count: count)
        var b = [Float](repeating: 0.0, count: count)
        
        // ゼロ配列
        let zeroRes = a.withUnsafeBufferPointer { aPtr in
            b.withUnsafeBufferPointer { bPtr in
                VectorOperations.dotProduct(a: aPtr.baseAddress!, b: bPtr.baseAddress!, count: count)
            }
        }
        XCTAssertEqual(zeroRes, 0.0)
        
        // 直交ベクトル
        var i = 0
        while i < count {
            if i % 2 == 0 {
                a[i] = 1.0
                b[i] = 0.0
            } else {
                a[i] = 0.0
                b[i] = 1.0
            }
            i += 1
        }
        
        let orthoRes = a.withUnsafeBufferPointer { aPtr in
            b.withUnsafeBufferPointer { bPtr in
                VectorOperations.dotProduct(a: aPtr.baseAddress!, b: bPtr.baseAddress!, count: count)
            }
        }
        XCTAssertEqual(orthoRes, 0.0)
    }
    
    // MARK: - Sum of Squares Tests
    
    func testSumOfSquaresEquivalence() {
        let testCounts = [0, 1, 7, 8, 9, 15, 16, 17, 319, 320, 321, 512, 1024]
        
        var tIdx = 0
        while tIdx < testCounts.count {
            let count = testCounts[tIdx]
            var a = [Float](repeating: 0.0, count: count)
            
            var i = 0
            while i < count {
                a[i] = sin(Float(i) * 0.3) * 1.5
                i += 1
            }
            
            var scalarSum: Float = 0.0
            i = 0
            while i < count {
                scalarSum += a[i] * a[i]
                i += 1
            }
            
            var simdSum: Float = 0.0
            if 0 < count {
                simdSum = a.withUnsafeBufferPointer { aPtr in
                    VectorOperations.sumOfSquares(ptr: aPtr.baseAddress!, count: count)
                }
            }
            
            let diff = abs(simdSum - scalarSum)
            let maxVal = max(abs(scalarSum), 1.0)
            let relError = diff / maxVal
            
            XCTAssertLessThan(relError, 1e-4, "SumOfSquares count=\(count) failed: simd=\(simdSum), scalar=\(scalarSum)")
            tIdx += 1
        }
    }
    
    // MARK: - Element-wise Multiply Tests
    
    func testMultiplyEquivalence() {
        let testCounts = [0, 1, 7, 8, 9, 15, 16, 17, 319, 320, 321, 512, 1024]
        
        var tIdx = 0
        while tIdx < testCounts.count {
            let count = testCounts[tIdx]
            var a = [Float](repeating: 0.0, count: count)
            var b = [Float](repeating: 0.0, count: count)
            var dstSimd = [Float](repeating: 0.0, count: count)
            var dstScalar = [Float](repeating: 0.0, count: count)
            
            var i = 0
            while i < count {
                a[i] = Float(i) * 0.125
                b[i] = Float(count - i) * 0.25
                dstScalar[i] = a[i] * b[i]
                i += 1
            }
            
            if 0 < count {
                a.withUnsafeBufferPointer { aPtr in
                    b.withUnsafeBufferPointer { bPtr in
                        dstSimd.withUnsafeMutableBufferPointer { dPtr in
                            VectorOperations.multiply(
                                srcA: aPtr.baseAddress!,
                                srcB: bPtr.baseAddress!,
                                dst: dPtr.baseAddress!,
                                count: count
                            )
                        }
                    }
                }
            }
            
            i = 0
            while i < count {
                XCTAssertEqual(dstSimd[i], dstScalar[i], "Multiply mismatch at index \(i) for count \(count)")
                i += 1
            }
            tIdx += 1
        }
    }
    
    // MARK: - Max Magnitude Tests
    
    func testMaxMagnitudeEquivalence() {
        let testCounts = [0, 1, 7, 8, 9, 15, 16, 17, 319, 320, 321, 512, 1024]
        
        var tIdx = 0
        while tIdx < testCounts.count {
            let count = testCounts[tIdx]
            var a = [Float](repeating: 0.0, count: count)
            
            var i = 0
            while i < count {
                var sign: Float = 1.0
                if i % 2 == 0 {
                    sign = -1.0
                }
                a[i] = sign * Float(i + 1) * 0.5
                i += 1
            }
            
            var scalarMax: Float = 0.0
            i = 0
            while i < count {
                let mag = abs(a[i])
                if scalarMax < mag {
                    scalarMax = mag
                }
                i += 1
            }
            
            var simdMax: Float = 0.0
            if 0 < count {
                simdMax = a.withUnsafeBufferPointer { aPtr in
                    VectorOperations.maxMagnitude(ptr: aPtr.baseAddress!, count: count)
                }
            }
            
            XCTAssertEqual(simdMax, scalarMax, "MaxMagnitude mismatch for count \(count)")
            tIdx += 1
        }
    }
    
    func testMaxMagnitudeNegativeOnly() {
        let count = 32
        var a = [Float](repeating: 0.0, count: count)
        var i = 0
        while i < count {
            a[i] = -1.0 * Float(i + 1)
            i += 1
        }
        
        let simdMax = a.withUnsafeBufferPointer { ptr in
            VectorOperations.maxMagnitude(ptr: ptr.baseAddress!, count: count)
        }
        XCTAssertEqual(simdMax, 32.0)
    }
    
    // MARK: - Clamp Tests
    
    func testClampEquivalence() {
        let testCounts = [0, 1, 7, 8, 9, 15, 16, 17, 319, 320, 321, 512, 1024]
        let minVal: Float = -0.5
        let maxVal: Float = 0.8
        
        var tIdx = 0
        while tIdx < testCounts.count {
            let count = testCounts[tIdx]
            var src = [Float](repeating: 0.0, count: count)
            var dstSimd = [Float](repeating: 0.0, count: count)
            var dstScalar = [Float](repeating: 0.0, count: count)
            
            var i = 0
            while i < count {
                src[i] = sin(Float(i) * 0.4) * 2.0
                var v = src[i]
                if v < minVal {
                    v = minVal
                }
                if maxVal < v {
                    v = maxVal
                }
                dstScalar[i] = v
                i += 1
            }
            
            if 0 < count {
                src.withUnsafeBufferPointer { sPtr in
                    dstSimd.withUnsafeMutableBufferPointer { dPtr in
                        VectorOperations.clamp(
                            src: sPtr.baseAddress!,
                            dst: dPtr.baseAddress!,
                            count: count,
                            minVal: minVal,
                            maxVal: maxVal
                        )
                    }
                }
            }
            
            i = 0
            while i < count {
                XCTAssertEqual(dstSimd[i], dstScalar[i], "Clamp mismatch at index \(i) for count \(count)")
                i += 1
            }
            tIdx += 1
        }
    }
}
