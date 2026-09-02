import XCTest
@testable import Spiketrans

final class SpikingNetworkWeightsTests: XCTestCase {
    
    func testSaveAndLoadFile() throws {
        let net = SpikingNetwork(inputDim: 128, maxHiddenDim: 512, outputDim: 100)
        let exported = net.exportWeights()
        
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("test_weights_\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: tempURL) }
        
        try exported.save(to: tempURL)
        let loaded = try SpikingNetworkWeights.load(from: tempURL)
        
        XCTAssertEqual(exported, loaded)
    }
}
