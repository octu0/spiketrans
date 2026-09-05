import XCTest
import MLX
@testable import Spiketrans

final class MLXBPTTTests: XCTestCase {
    override func setUp() {
        super.setUp()
        Device.setDefault(device: .cpu)
    }

    func testMLXNetworkForwardAndExportImport() throws {
        let mlxNet = MLXSpikingNetwork(inputDim: 128, maxHiddenDim: 1024, outputDim: 523)
        let weights = mlxNet.exportWeights()
        
        XCTAssertEqual(weights.inputDim, 128)
        XCTAssertEqual(weights.maxHiddenDim, 1024)
        XCTAssertEqual(weights.outputDim, 523)
        
        let cpuNet = SpikingNetwork(inputDim: 128, maxHiddenDim: 1024, outputDim: 523)
        cpuNet.importWeights(from: weights)
        
        // 重みデータの比較
        let cpuExported = cpuNet.exportWeights()
        XCTAssertEqual(weights.wIn, cpuExported.wIn)
        XCTAssertEqual(weights.bH, cpuExported.bH)
        XCTAssertEqual(weights.wOut, cpuExported.wOut)
        XCTAssertEqual(weights.bOut, cpuExported.bOut)
    }
    
    func testMLXBPTTTrainerSingleStep() throws {
        let mlxNet = MLXSpikingNetwork(numLayers: 2, inputDim: 128, maxHiddenDim: 512, outputDim: 50)
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
        XCTAssertTrue(0.0 < res)
        XCTAssertTrue(res.isNaN != true)
        XCTAssertTrue(res.isInfinite != true)
    }

    func testMLXThreeLayerNetworkTrainingAndExport() throws {
        let mlxNet = MLXSpikingNetwork(numLayers: 3, inputDim: 64, maxHiddenDim: 128, outputDim: 20)
        let trainer = MLXBPTTTrainer(network: mlxNet, config: TrainingConfig(learningRate: 0.01))

        let seqLen = 10
        var dummyFeat = [[Float]](repeating: [Float](repeating: 0.0, count: 64), count: seqLen)
        var i = 0
        while i < seqLen {
            var c = 0
            while c < 64 {
                dummyFeat[i][c] = Float.random(in: 0.0...1.0)
                c += 1
            }
            i += 1
        }

        var dummyTargets = [Int](repeating: 0, count: seqLen)
        i = 2
        while i < 8 {
            dummyTargets[i] = i % 20
            i += 1
        }

        let loss = trainer.trainStep(features: dummyFeat, targets: dummyTargets)
        XCTAssertTrue(0.0 < loss)
        XCTAssertTrue(loss.isNaN != true)

        let weights = mlxNet.exportWeights()
        XCTAssertEqual(weights.numLayers, 3)
        XCTAssertEqual(weights.wLayers?.count, 2)
        XCTAssertEqual(weights.bHLayers?.count, 2)
        XCTAssertEqual(weights.gammaRMS?.count, 2)

        let cpuNet = SpikingNetwork(numLayers: 3, inputDim: 64, maxHiddenDim: 128, outputDim: 20)
        cpuNet.importWeights(from: weights)
        let cpuWeights = cpuNet.exportWeights()

        XCTAssertEqual(weights.wIn, cpuWeights.wIn)
        XCTAssertEqual(weights.wLayers?[0], cpuWeights.wLayers?[0])
        XCTAssertEqual(weights.wLayers?[1], cpuWeights.wLayers?[1])
        XCTAssertEqual(weights.gammaRMS?[0], cpuWeights.gammaRMS?[0])
        XCTAssertEqual(weights.gammaRMS?[1], cpuWeights.gammaRMS?[1])
    }

    func testMLXConv2DSubsamplingTrainingAndExport() throws {
        let mlxNet = MLXSpikingNetwork(numLayers: 2, inputDim: 128, maxHiddenDim: 128, outputDim: 20)
        mlxNet.convSubsampling = MLXConv2DSubsampling(outputDim: 128)
        let trainer = MLXBPTTTrainer(network: mlxNet, config: TrainingConfig(learningRate: 0.01))

        let seqLen = 16
        var dummyFeat = [[Float]](repeating: [Float](repeating: 0.0, count: 64), count: seqLen)
        var i = 0
        while i < seqLen {
            var c = 0
            while c < 64 {
                dummyFeat[i][c] = Float.random(in: 0.0...1.0)
                c += 1
            }
            i += 1
        }

        let targets = [2, 5, 8]
        let loss = trainer.trainBatchCTC(featuresBatch: [dummyFeat], targetsBatch: [targets])
        XCTAssertTrue(0.0 < loss)
        XCTAssertTrue(loss.isNaN != true)

        let weights = mlxNet.exportWeights()
        XCTAssertNotNil(weights.convSubsampling)
        XCTAssertEqual(weights.convSubsampling?.outputDim, 128)

        let cpuNet = SpikingNetwork(numLayers: 2, inputDim: 128, maxHiddenDim: 128, outputDim: 20)
        cpuNet.importWeights(from: weights)
        XCTAssertNotNil(cpuNet.convSubsampling)
        XCTAssertEqual(weights.convSubsampling?.projWeight, cpuNet.convSubsampling?.projWeight)
    }
}
