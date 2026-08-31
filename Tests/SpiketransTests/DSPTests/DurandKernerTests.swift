import XCTest
@testable import Spiketrans

final class DurandKernerTests: XCTestCase {
    
    func testSolveQuadratic() {
        let solver = DurandKernerSolver()
        let workspace = DSPWorkspace(maxFrameSize: 1024, lpcOrder: 12)
        
        // P(z) = z^2 - 3z + 2 = 0 -> 根は z=1, z=2
        // 係数配列: [c0=1.0, c1=-3.0, c2=2.0]
        let coeffs: [Float] = [1.0, -3.0, 2.0]
        let success = coeffs.withUnsafeBufferPointer { ptr in
            return solver.solve(coefficients: ptr.baseAddress!, order: 2, workspace: workspace)
        }
        
        XCTAssertTrue(success)
        
        let r0 = workspace.durandKernerCurr[0]
        let r1 = workspace.durandKernerCurr[1]
        
        // 2つの根が 1.0 と 2.0 に収束していることを検証
        let dist0 = min(
            (r0 - Complex(real: 1.0, imag: 0.0)).magnitude,
            (r0 - Complex(real: 2.0, imag: 0.0)).magnitude
        )
        let dist1 = min(
            (r1 - Complex(real: 1.0, imag: 0.0)).magnitude,
            (r1 - Complex(real: 2.0, imag: 0.0)).magnitude
        )
        
        XCTAssertLessThan(dist0, 1e-3)
        XCTAssertLessThan(dist1, 1e-3)
    }
    
    func testSolve12thOrderLPC() {
        let solver = DurandKernerSolver()
        let workspace = DSPWorkspace(maxFrameSize: 1024, lpcOrder: 12)
        let pi = Float.pi
        
        // 6組の共振極 (r = 0.92, 角度: 30°, 55°, 80°, 105°, 130°, 155°)
        let r: Float = 0.92
        let angles: [Float] = [
            (30.0 * pi) / 180.0,
            (55.0 * pi) / 180.0,
            (80.0 * pi) / 180.0,
            (105.0 * pi) / 180.0,
            (130.0 * pi) / 180.0,
            (155.0 * pi) / 180.0
        ]
        
        // 各極ペア: (z - r e^{j\theta})(z - r e^{-j\theta}) = z^2 - (2 r cos\theta) z + r^2
        // 6個の2次多項式を畳み込んで12次多項式係数 c を計算
        var currentPoly: [Float] = [1.0]
        
        var pIdx = 0
        while pIdx < angles.count {
            let theta = angles[pIdx]
            let quad: [Float] = [1.0, -2.0 * r * cos(theta), r * r]
            
            // 多項式乗算 (畳み込み)
            let newDegree = (currentPoly.count - 1) + 2
            var newPoly = [Float](repeating: 0.0, count: newDegree + 1)
            
            var i = 0
            while i < currentPoly.count {
                var j = 0
                while j < quad.count {
                    newPoly[i + j] += currentPoly[i] * quad[j]
                    j += 1
                }
                i += 1
            }
            currentPoly = newPoly
            pIdx += 1
        }
        
        let success = currentPoly.withUnsafeBufferPointer { ptr in
            return solver.solve(coefficients: ptr.baseAddress!, order: 12, workspace: workspace)
        }
        
        XCTAssertTrue(success)
        
        // 各根の絶対値が r = 0.92 の近傍に収束していることを検証
        var idx = 0
        while idx < 12 {
            let root = workspace.durandKernerCurr[idx]
            XCTAssertLessThan(abs(root.magnitude - r), 0.05)
            idx += 1
        }
    }
    
    func testExtractFormantsVowelA() {
        let extractor = FormantExtractor(sampleRate: 16000.0)
        let pi = Float.pi
        
        // 母音 /a/: F1 = 800Hz, F2 = 1200Hz, F3 = 2500Hz
        let targetF1: Float = 800.0
        let targetF2: Float = 1200.0
        let targetF3: Float = 2500.0
        let r: Float = 0.96
        
        let theta1 = (2.0 * pi * targetF1) / 16000.0
        let theta2 = (2.0 * pi * targetF2) / 16000.0
        let theta3 = (2.0 * pi * targetF3) / 16000.0
        
        let roots: [Complex] = [
            Complex(real: r * cos(theta1), imag: r * sin(theta1)),
            Complex(real: r * cos(theta1), imag: -r * sin(theta1)),
            Complex(real: r * cos(theta2), imag: r * sin(theta2)),
            Complex(real: r * cos(theta2), imag: -r * sin(theta2)),
            Complex(real: r * cos(theta3), imag: r * sin(theta3)),
            Complex(real: r * cos(theta3), imag: -r * sin(theta3))
        ]
        
        let result = roots.withUnsafeBufferPointer { ptr in
            return extractor.extractFormants(roots: ptr.baseAddress!, count: roots.count)
        }
        
        XCTAssertLessThan(abs(result.f1 - targetF1), 1.0)
        XCTAssertLessThan(abs(result.f2 - targetF2), 1.0)
        XCTAssertLessThan(abs(result.f3 - targetF3), 1.0)
    }
    
    func testExtractFormantsVowelI() {
        let extractor = FormantExtractor(sampleRate: 16000.0)
        let pi = Float.pi
        
        // 母音 /i/: F1 = 300Hz, F2 = 2300Hz, F3 = 3000Hz
        let targetF1: Float = 300.0
        let targetF2: Float = 2300.0
        let targetF3: Float = 3000.0
        let r: Float = 0.96
        
        let theta1 = (2.0 * pi * targetF1) / 16000.0
        let theta2 = (2.0 * pi * targetF2) / 16000.0
        let theta3 = (2.0 * pi * targetF3) / 16000.0
        
        let roots: [Complex] = [
            Complex(real: r * cos(theta1), imag: r * sin(theta1)),
            Complex(real: r * cos(theta1), imag: -r * sin(theta1)),
            Complex(real: r * cos(theta2), imag: r * sin(theta2)),
            Complex(real: r * cos(theta2), imag: -r * sin(theta2)),
            Complex(real: r * cos(theta3), imag: r * sin(theta3)),
            Complex(real: r * cos(theta3), imag: -r * sin(theta3))
        ]
        
        let result = roots.withUnsafeBufferPointer { ptr in
            return extractor.extractFormants(roots: ptr.baseAddress!, count: roots.count)
        }
        
        XCTAssertLessThan(abs(result.f1 - targetF1), 1.0)
        XCTAssertLessThan(abs(result.f2 - targetF2), 1.0)
        XCTAssertLessThan(abs(result.f3 - targetF3), 1.0)
    }
    
    func testFilterOutPolesOutsideUnitCircle() {
        let extractor = FormantExtractor(sampleRate: 16000.0)
        let pi = Float.pi
        
        // 半径 r=0.70 (しきい値 0.88 未満) の極
        let freq: Float = 1000.0
        let theta = (2.0 * pi * freq) / 16000.0
        let weakRoot = Complex(real: 0.70 * cos(theta), imag: 0.70 * sin(theta))
        
        let roots: [Complex] = [weakRoot]
        let result = roots.withUnsafeBufferPointer { ptr in
            return extractor.extractFormants(roots: ptr.baseAddress!, count: roots.count)
        }
        
        // 0.88 未満の極はフィルタアウトされてフォルマント数が 0 になる
        XCTAssertEqual(result.count, 0)
        XCTAssertEqual(result.f1, 0.0)
    }
    
    func testBandwidthCalculation() {
        let extractor = FormantExtractor(sampleRate: 16000.0)
        let pi = Float.pi
        let r: Float = 0.96
        let freq: Float = 1000.0
        let theta = (2.0 * pi * freq) / 16000.0
        let root = Complex(real: r * cos(theta), imag: r * sin(theta))
        
        let roots: [Complex] = [root]
        let result = roots.withUnsafeBufferPointer { ptr in
            return extractor.extractFormants(roots: ptr.baseAddress!, count: roots.count)
        }
        
        let expectedBw = -1.0 * (16000.0 / pi) * log(r)
        XCTAssertLessThan(abs(result.b1 - expectedBw), 1.0)
    }
    
    func testFilterOutPolesOnOrOutsideUnitCircle() {
        let extractor = FormantExtractor(sampleRate: 16000.0)
        let pi = Float.pi
        
        // 単位円外の極 (r = 1.15)
        let freq: Float = 1000.0
        let theta = (2.0 * pi * freq) / 16000.0
        let divergingRoot = Complex(real: 1.15 * cos(theta), imag: 1.15 * sin(theta))
        
        // 単位円上の極 (r = 1.00)
        let unitRoot = Complex(real: 1.00 * cos(theta), imag: 1.00 * sin(theta))
        
        let roots: [Complex] = [divergingRoot, unitRoot]
        let result = roots.withUnsafeBufferPointer { ptr in
            return extractor.extractFormants(roots: ptr.baseAddress!, count: roots.count)
        }
        
        // r >= 1.0 の極はすべて除外され、フォルマント数は 0 となること
        XCTAssertEqual(result.count, 0)
        XCTAssertEqual(result.f1, 0.0)
        XCTAssertEqual(result.b1, 0.0)
    }
}
