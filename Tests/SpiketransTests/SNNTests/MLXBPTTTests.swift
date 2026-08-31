import XCTest
import MLX
@testable import Spiketrans

final class MLXBPTTTests: XCTestCase {
    override func setUp() {
        super.setUp()
        Device.setDefault(device: .cpu)
    }

    func testMLXNetworkForwardAndExportImport() throws {
        let mlxNet = MLXMatryoshkaNetwork(inputDim: 128, maxHiddenDim: 1024, outputDim: 523)
        let weights = mlxNet.exportWeights()
        
        XCTAssertEqual(weights.inputDim, 128)
        XCTAssertEqual(weights.maxHiddenDim, 1024)
        XCTAssertEqual(weights.outputDim, 523)
        
        let cpuNet = MatryoshkaNetwork(inputDim: 128, maxHiddenDim: 1024, outputDim: 523)
        cpuNet.importWeights(from: weights)
        
        // 重みデータの比較
        let cpuExported = cpuNet.exportWeights()
        XCTAssertEqual(weights.wIn, cpuExported.wIn)
        XCTAssertEqual(weights.bH, cpuExported.bH)
        XCTAssertEqual(weights.wOut, cpuExported.wOut)
        XCTAssertEqual(weights.bOut, cpuExported.bOut)
    }
    
    func testMLXBPTTTrainerSingleStep() throws {
        let mlxNet = MLXMatryoshkaNetwork(inputDim: 128, maxHiddenDim: 512, outputDim: 50)
        let trainer = MLXBPTTTrainer(network: mlxNet, config: TrainingConfig(learningRate: 0.01))
        
        let seqLen = 20
        var dummyFeat = [[Float]](repeating: [Float](repeating: 0.0, count: 128), count: seqLen)
        var i = 0
        while i < seqLen {
            var c = 0
            while c < 128 {
                dummyFeat[i][c] = Float.random(in: 0.0...1.0)
                c += 1
            }
            i += 1
        }
        
        var dummyTargets = [Int](repeating: 0, count: seqLen)
        i = 5
        while i < 15 {
            dummyTargets[i] = i % 50
            i += 1
        }
        
        let res = trainer.trainStep(features: dummyFeat, targets: dummyTargets)
        XCTAssertGreaterThan(res.totalLoss, 0.0)
        XCTAssertFalse(res.totalLoss.isNaN)
        XCTAssertFalse(res.totalLoss.isInfinite)
    }
}
