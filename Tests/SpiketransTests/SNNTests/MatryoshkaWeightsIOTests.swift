import XCTest
@testable import Spiketrans

final class MatryoshkaWeightsIOTests: XCTestCase {
    func testExportAndImportWeightsEquivalence() throws {
        let net1 = MatryoshkaNetwork(inputDim: 128, maxHiddenDim: 1024, outputDim: 523)
        let exported = net1.exportWeights()
        
        let net2 = MatryoshkaNetwork(inputDim: 128, maxHiddenDim: 1024, outputDim: 523)
        net2.importWeights(from: exported)
        
        XCTAssertEqual(net1.pWIn.data, net2.pWIn.data)
        XCTAssertEqual(net1.pWRec.data, net2.pWRec.data)
        XCTAssertEqual(net1.pBH.data, net2.pBH.data)
        XCTAssertEqual(net1.pWOut.data, net2.pWOut.data)
        XCTAssertEqual(net1.pBOut.data, net2.pBOut.data)
        
        // 推論結果の完全一致検証
        var feat = [Float](repeating: 0.0, count: 128)
        var i = 0
        while i < 128 {
            feat[i] = Float(i) * 0.01
            i += 1
        }
        
        var v1 = [Float](repeating: 0.0, count: 128)
        var s1 = [Float](repeating: 0.0, count: 128)
        var sp1 = [Float](repeating: 0.0, count: 128)
        var log1 = [Float](repeating: 0.0, count: 523)
        var prob1 = [Float](repeating: 0.0, count: 523)
        
        var v2 = [Float](repeating: 0.0, count: 128)
        var s2 = [Float](repeating: 0.0, count: 128)
        var sp2 = [Float](repeating: 0.0, count: 128)
        var log2 = [Float](repeating: 0.0, count: 523)
        var prob2 = [Float](repeating: 0.0, count: 523)
        
        net1.forwardSlice(
            features: feat,
            slice: MatryoshkaSlice.base,
            vPrev: &v1,
            sPrev: &s1,
            spikeSum: &sp1,
            logits: &log1,
            probabilities: &prob1
        )
        
        net2.forwardSlice(
            features: feat,
            slice: MatryoshkaSlice.base,
            vPrev: &v2,
            sPrev: &s2,
            spikeSum: &sp2,
            logits: &log2,
            probabilities: &prob2
        )
        
        XCTAssertEqual(log1, log2)
        XCTAssertEqual(prob1, prob2)
    }
    
    func testSaveAndLoadFile() throws {
        let net = MatryoshkaNetwork(inputDim: 128, maxHiddenDim: 512, outputDim: 100)
        let exported = net.exportWeights()
        
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("test_weights_\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: tempURL) }
        
        try exported.save(to: tempURL)
        let loaded = try MatryoshkaWeightsData.load(from: tempURL)
        
        XCTAssertEqual(exported, loaded)
    }
}
