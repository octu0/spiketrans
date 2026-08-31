import Foundation
import Spiketrans
#if canImport(Darwin)
import Darwin
#endif

setbuf(stdout, nil)

private func getResidentMemoryBytes() -> UInt64 {
    #if canImport(Darwin)
    var taskInfo = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
    let kerr: kern_return_t = withUnsafeMutablePointer(to: &taskInfo) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
        }
    }
    if kerr == KERN_SUCCESS {
        return taskInfo.resident_size
    }
    return 0
    #else
    return 0
    #endif
}

print("==================================================")
print("=== Spiketrans Audio DSP Benchmark & Stress Test ===")
print("==================================================")

let sampleRate = 16000
let config = DSPConfig(sampleRate: sampleRate)
let workspace = DSPWorkspace(maxFrameSize: 1024, lpcOrder: config.lpcOrder, melChannels: config.melChannels)
let vad = VAD(config: config)
let pitchDetector = PitchDetector(config: config)
let lpc = LPC(config: config)
let solver = DurandKernerSolver()
let formantExtractor = FormantExtractor(sampleRate: Float(sampleRate))
let filterbank = Filterbank(config: config)

let frameSize = config.frameSize
let hopSize = config.hopSize

// ----------------------------------------------------
// 1. 標準 5秒 音声ベンチマーク
// ----------------------------------------------------
print("\n--- 1. 標準 5秒音声ベンチマーク ---")
let shortDuration = 5
let shortTotalSamples = sampleRate * shortDuration
var shortPcm = [Float](repeating: 0.0, count: shortTotalSamples)
let factor = (2.0 * Float.pi * 440.0) / Float(sampleRate)
var i = 0
while i < shortTotalSamples {
    shortPcm[i] = 0.5 * sin(Float(i) * factor)
    i += 1
}

let shortStartTime = CFAbsoluteTimeGetCurrent()
var shortProcessedFrames = 0
var shortOffset = 0

shortPcm.withUnsafeBufferPointer { pcmPtr in
    let base = pcmPtr.baseAddress!
    while (shortOffset + frameSize) <= shortTotalSamples {
        let framePtr = base.advanced(by: shortOffset)
        
        vad.processFrame(ptr: framePtr, count: frameSize, workspace: workspace)
        let pitchRes = pitchDetector.detectPitch(ptr: framePtr, count: frameSize, workspace: workspace)
        let lpcSuccess = lpc.computeCoefficients(ptr: framePtr, count: frameSize, workspace: workspace)
        
        var formantRes = FormantResult(f1: 0.0, f2: 0.0, f3: 0.0, b1: 0.0, b2: 0.0, b3: 0.0, count: 0)
        if lpcSuccess {
            workspace.lpcCoeffs.withUnsafeBufferPointer { cPtr in
                let solverSuccess = solver.solve(coefficients: cPtr.baseAddress!, order: config.lpcOrder, workspace: workspace)
                if solverSuccess {
                    workspace.durandKernerCurr.withUnsafeBufferPointer { rPtr in
                        formantRes = formantExtractor.extractFormants(roots: rPtr.baseAddress!, count: config.lpcOrder)
                    }
                }
            }
        }
        
        filterbank.extractFeatures(
            pcmPtr: framePtr,
            count: frameSize,
            workspace: workspace
        )
        
        shortProcessedFrames += 1
        shortOffset += hopSize
    }
}

let shortElapsed = CFAbsoluteTimeGetCurrent() - shortStartTime
let shortRtf = Float(shortElapsed) / Float(shortDuration)

print("処理フレーム数: \(shortProcessedFrames) フレーム")
print("実音声時間: \(shortDuration) 秒")
print("処理所要時間: \(String(format: "%.4f", shortElapsed)) 秒")
print("Real-Time Factor (RTF): \(String(format: "%.6f", shortRtf)) xRT")

// ----------------------------------------------------
// 2. SNN コア推論スループット計測
// ----------------------------------------------------
print("\n==================================================")
print("=== SNN Core Inference Throughput Benchmark ===")
print("==================================================")

let net = MatryoshkaNetwork(inputDim: 64, maxHiddenDim: 256, outputDim: 64, timeSteps: 4)
for slice in MatryoshkaSlice.allCases {
    let hSize = slice.rawValue
    var v = [Float](repeating: 0.0, count: hSize)
    var s = [Float](repeating: 0.0, count: hSize)
    var sum = [Float](repeating: 0.0, count: hSize)
    var logits = [Float](repeating: 0.0, count: 64)
    var probs = [Float](repeating: 0.0, count: 64)
    let feat = [Float](repeating: 0.5, count: 64)
    
    let benchSteps = 10000
    let start = CFAbsoluteTimeGetCurrent()
    var step = 0
    while step < benchSteps {
        net.forwardSlice(
            features: feat,
            slice: slice,
            vPrev: &v,
            sPrev: &s,
            spikeSum: &sum,
            logits: &logits,
            probabilities: &probs
        )
        step += 1
    }
    let elapsed = CFAbsoluteTimeGetCurrent() - start
    let throughput = Double(benchSteps) / elapsed
    let latency = (elapsed / Double(benchSteps)) * 1_000_000.0
    print("[Slice: \(slice) (Hidden Dim: \(hSize))]")
    print("  Float32: Throughput = \(String(format: "%.1f", throughput)) steps/sec, Latency = \(String(format: "%.2f", latency)) µs/step")
}

print("\n==================================================")
print("=== ベンチマーク完了 ===")
print("==================================================")
