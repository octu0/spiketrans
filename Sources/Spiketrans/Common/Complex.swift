import Foundation

/// 単精度浮動小数点複素数
public struct Complex: Sendable, Equatable {
    public var real: Float
    public var imag: Float
    
    @inline(__always)
    public init(real: Float, imag: Float) {
        self.real = real
        self.imag = imag
    }
    
    @inline(__always)
    public static func + (lhs: Complex, rhs: Complex) -> Complex {
        return Complex(real: lhs.real + rhs.real, imag: lhs.imag + rhs.imag)
    }
    
    @inline(__always)
    public static func - (lhs: Complex, rhs: Complex) -> Complex {
        return Complex(real: lhs.real - rhs.real, imag: lhs.imag - rhs.imag)
    }
    
    @inline(__always)
    public static func * (lhs: Complex, rhs: Complex) -> Complex {
        return Complex(
            real: (lhs.real * rhs.real) - (lhs.imag * rhs.imag),
            imag: (lhs.real * rhs.imag) + (lhs.imag * rhs.real)
        )
    }
    
    @inline(__always)
    public static func * (lhs: Complex, rhs: Float) -> Complex {
        return Complex(real: lhs.real * rhs, imag: lhs.imag * rhs)
    }
    
    @inline(__always)
    public static func / (lhs: Complex, rhs: Complex) -> Complex {
        let denom = (rhs.real * rhs.real) + (rhs.imag * rhs.imag)
        if denom < 1e-10 {
            return Complex(real: 0.0, imag: 0.0)
        }
        let inv = 1.0 / denom
        return Complex(
            real: ((lhs.real * rhs.real) + (lhs.imag * rhs.imag)) * inv,
            imag: ((lhs.imag * rhs.real) - (lhs.real * rhs.imag)) * inv
        )
    }
    
    public var magnitude: Float {
        @inline(__always)
        get {
            return sqrt((real * real) + (imag * imag))
        }
    }
    
    public var magnitudeSquared: Float {
        @inline(__always)
        get {
            return (real * real) + (imag * imag)
        }
    }
    
    public var phase: Float {
        @inline(__always)
        get {
            return atan2(imag, real)
        }
    }
}
