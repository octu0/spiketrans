import Foundation

/// フォルマント遷移変曲点および音響境界の動的セグメンテーション
public struct FormantSegmenter: Sendable {

    /// PCM データからフォルマント周波数の変化率変曲点および有声/無声境界を検出し、
    /// TBPTT および推論同期用の動的境界フレームインデックス（0-indexed）リストを返す
    public static func detectBoundaries(
        pcmData: [Float],
        sampleRate: Int = 16000,
        frameSize: Int = 400,
        hopSize: Int = 160,
        minChunkFrames: Int = 8,
        maxChunkFrames: Int = 36
    ) -> [Int] {
        let totalSamples = pcmData.count
        if totalSamples < frameSize {
            return []
        }

        let numFrames = (totalSamples - frameSize) / hopSize + 1
        if numFrames <= minChunkFrames {
            return [numFrames - 1]
        }

        let dspCfg = DSPConfig(sampleRate: sampleRate, frameSize: frameSize, hopSize: hopSize, lpcOrder: 12)
        let dspWs = DSPWorkspace(lpcOrder: 12, melChannels: 64, fftSize: 512)
        let lpc = LPC(config: dspCfg)
        let solver = DurandKernerSolver()
        let formantExtractor = FormantExtractor(sampleRate: Float(sampleRate))

        var f1List = [Float](repeating: 0.0, count: numFrames)
        var f2List = [Float](repeating: 0.0, count: numFrames)
        var energyList = [Float](repeating: 0.0, count: numFrames)

        pcmData.withUnsafeBufferPointer { pcmPtr in
            let basePtr = pcmPtr.baseAddress!
            var fIdx = 0
            while fIdx < numFrames {
                let offset = fIdx * hopSize
                let framePtr = basePtr.advanced(by: offset)

                // フレームエネルギーの計算
                var sumSq: Float = 0.0
                var i = 0
                while i < frameSize {
                    let s = framePtr[i]
                    sumSq += s * s
                    i += 1
                }
                energyList[fIdx] = sumSq / Float(frameSize)

                // LPC フォルマント解析
                let success = lpc.computeCoefficients(ptr: framePtr, count: frameSize, workspace: dspWs)
                if success {
                    let coeffPtr = dspWs.lpcCoeffs.withUnsafeBufferPointer { $0.baseAddress! }
                    let rootsSuccess = solver.solve(coefficients: coeffPtr, order: 12, workspace: dspWs)
                    if rootsSuccess {
                        let rootPtr = dspWs.durandKernerCurr.withUnsafeBufferPointer { $0.baseAddress! }
                        let formantRes = formantExtractor.extractFormants(roots: rootPtr, count: 12, workspace: dspWs)
                        f1List[fIdx] = formantRes.f1
                        f2List[fIdx] = formantRes.f2
                    }
                }

                fIdx += 1
            }
        }

        // フォルマント遷移速度 Delta F の計算
        var deltaF = [Float](repeating: 0.0, count: numFrames)
        var k = 1
        while k < numFrames {
            let df1 = f1List[k] - f1List[k - 1]
            let df2 = f2List[k] - f2List[k - 1]
            deltaF[k] = sqrt(df1 * df1 + df2 * df2)
            k += 1
        }

        // 動的音素境界の決定 (minChunkFrames <= chunk <= maxChunkFrames)
        var boundaries: [Int] = []
        var lastBoundary = 0
        var t = 1
        while t < (numFrames - 1) {
            let framesSinceLast = t - lastBoundary
            var isBoundary = false

            if maxChunkFrames <= framesSinceLast {
                isBoundary = true
            }

            if minChunkFrames <= framesSinceLast && framesSinceLast < maxChunkFrames {
                // フォルマント変化量の局所ピーク (変曲点) かつ有意な変化
                let prevD = deltaF[t - 1]
                let curD = deltaF[t]
                let nextD = deltaF[t + 1]
                if prevD <= curD && nextD <= curD && 150.0 <= curD {
                    isBoundary = true
                }

                // 無音/有声の切り替わり (エネルギーの急変)
                let prevE = energyList[t - 1]
                let curE = energyList[t]
                if prevE < 1e-4 && 1e-3 <= curE {
                    isBoundary = true
                }
                if 1e-3 <= prevE && curE < 1e-4 {
                    isBoundary = true
                }
            }

            if isBoundary {
                boundaries.append(t)
                lastBoundary = t
            }

            t += 1
        }

        if lastBoundary < (numFrames - 1) {
            boundaries.append(numFrames - 1)
        }

        return boundaries
    }

    /// 境界フレームインデックスを Subsampling 比率（既定 4）に合わせて圧縮・重複排除する
    public static func subsampleBoundaries(boundaries: [Int], factor: Int = 4) -> [Int] {
        if boundaries.isEmpty || factor <= 1 {
            return boundaries
        }
        var scaled: [Int] = []
        scaled.reserveCapacity(boundaries.count)
        var lastIdx: Int? = nil

        var i = 0
        while i < boundaries.count {
            let s = boundaries[i] / factor
            switch lastIdx {
            case .some(let prev):
                if prev < s {
                    scaled.append(s)
                    lastIdx = s
                }
            case .none:
                scaled.append(s)
                lastIdx = s
            }
            i += 1
        }
        return scaled
    }
}
