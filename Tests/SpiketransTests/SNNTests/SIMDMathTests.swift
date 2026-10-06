import XCTest
@testable import Spiketrans

/// SIMD8 の exp・sigmoid が標準の expf と相対誤差 1e-6 以内で一致する
final class SIMDMathTests: XCTestCase {
    func testExpAndSigmoidMatchScalar() {
        var x: Float = -30.0
        while x < 30.0 {
            var lanes = SIMD8<Float>(repeating: 0.0)
            var k = 0
            while k < 8 {
                lanes[k] = x + Float(k) * 0.0137
                k += 1
            }
            let e = SIMDMath.exp(lanes)
            let sg = SIMDMath.sigmoid(lanes)
            k = 0
            while k < 8 {
                let ref = expf(lanes[k])
                XCTAssertEqual(e[k], ref, accuracy: max(1e-30, ref * 1e-6), "exp(\(lanes[k]))")
                XCTAssertEqual(sg[k], 1.0 / (1.0 + expf(-lanes[k])), accuracy: 1e-6, "sigmoid(\(lanes[k]))")
                k += 1
            }
            x += 0.173
        }
        let extreme = SIMDMath.exp(SIMD8<Float>(-200.0, -100.0, -87.0, 0.0, 87.0, 88.0, 100.0, 200.0))
        XCTAssertEqual(extreme[3], 1.0, accuracy: 1e-6)
        XCTAssertTrue(extreme[0].isFinite && extreme[7].isFinite)
        XCTAssertLessThan(extreme[0], 1e-37)
    }
}
