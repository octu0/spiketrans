import Foundation

/// Pure Swift による 512点 Cooley-Tukey Radix-2 FFT プロセッサ
public struct FFT: Sendable {
    public let size: Int
    public let log2Size: Int
    
    private let bitReversedIndices: [Int]
    private let twiddleReal: [Float]
    private let twiddleImag: [Float]
    
    public init(size: Int = 512) {
        self.size = size
        
        // log2(size) の計算
        var m = 0
        var temp = size
        while 1 < temp {
            temp = temp >> 1
            m += 1
        }
        self.log2Size = m
        
        // 1. Bit-reversal インデックステーブルの事前計算
        var bitRev = [Int](repeating: 0, count: size)
        var i = 0
        while i < size {
            var rev = 0
            var b = 0
            while b < m {
                if (i & (1 << b)) != 0 {
                    rev |= (1 << ((m - 1) - b))
                }
                b += 1
            }
            bitRev[i] = rev
            i += 1
        }
        self.bitReversedIndices = bitRev
        
        // 2. Twiddle factor (回転因子: e^{-j 2\pi k / N}) テーブルの事前計算
        let half = size / 2
        var twReal = [Float](repeating: 0.0, count: half)
        var twImag = [Float](repeating: 0.0, count: half)
        
        var k = 0
        let factor = (2.0 * Float.pi) / Float(size)
        while k < half {
            let theta = -1.0 * Float(k) * factor
            twReal[k] = cos(theta)
            twImag[k] = sin(theta)
            k += 1
        }
        self.twiddleReal = twReal
        self.twiddleImag = twImag
    }
    
    /// In-place 順方向 FFT
    @inline(__always)
    public func forward(
        real: UnsafeMutablePointer<Float>,
        imag: UnsafeMutablePointer<Float>
    ) {
        // 1. Bit-reversal スワップ
        var i = 0
        while i < size {
            let rev = bitReversedIndices[i]
            if i < rev {
                let tr = real[i]
                real[i] = real[rev]
                real[rev] = tr
                
                let ti = imag[i]
                imag[i] = imag[rev]
                imag[rev] = ti
            }
            i += 1
        }
        
        // 2. Cooley-Tukey Butterfly 演算
        var s = 1
        while s <= log2Size {
            let len = 1 << s
            let halfLen = len >> 1
            let step = size / len
            
            var j = 0
            while j < halfLen {
                let wIdx = j * step
                let wr = twiddleReal[wIdx]
                let wi = twiddleImag[wIdx]
                
                var k = 0
                while k < size {
                    let idx1 = k + j
                    let idx2 = idx1 + halfLen
                    
                    let ur = real[idx1]
                    let ui = imag[idx1]
                    let vr = real[idx2]
                    let vi = imag[idx2]
                    
                    let tr = (wr * vr) - (wi * vi)
                    let ti = (wr * vi) + (wi * vr)
                    
                    real[idx1] = ur + tr
                    imag[idx1] = ui + ti
                    real[idx2] = ur - tr
                    imag[idx2] = ui - ti
                    
                    k += len
                }
                j += 1
            }
            s += 1
        }
    }
    
    /// パワースペクトル (|X[k]|^2) の高速算出 (SIMD8 + ポインタ)
    @inline(__always)
    public func computePowerSpectrum(
        real: UnsafePointer<Float>,
        imag: UnsafePointer<Float>,
        powerSpectrum: UnsafeMutablePointer<Float>,
        halfSize: Int
    ) {
        let width = 8
        let limit = halfSize - (halfSize % width)
        var i = 0
        
        while i < limit {
            let vr = SIMD8<Float>(
                real[i+0], real[i+1], real[i+2], real[i+3],
                real[i+4], real[i+5], real[i+6], real[i+7]
            )
            let vi = SIMD8<Float>(
                imag[i+0], imag[i+1], imag[i+2], imag[i+3],
                imag[i+4], imag[i+5], imag[i+6], imag[i+7]
            )
            let vPow = (vr * vr) + (vi * vi)
            powerSpectrum[i+0] = vPow[0]
            powerSpectrum[i+1] = vPow[1]
            powerSpectrum[i+2] = vPow[2]
            powerSpectrum[i+3] = vPow[3]
            powerSpectrum[i+4] = vPow[4]
            powerSpectrum[i+5] = vPow[5]
            powerSpectrum[i+6] = vPow[6]
            powerSpectrum[i+7] = vPow[7]
            i += width
        }
        
        while i <= halfSize {
            let r = real[i]
            let im = imag[i]
            powerSpectrum[i] = (r * r) + (im * im)
            i += 1
        }
    }
}
