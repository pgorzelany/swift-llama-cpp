import Testing
import Foundation
@testable import SwiftLlama

struct LlamaSamplerTests {
    @Test("Sampler chain builds with config and names are accessible")
    func testSamplerChainBuild() throws {
        let model = try #require(LlamaModel(path: URL.llama1B.path))
        let cfg = LlamaSamplingConfig(temperature: 0.7, seed: 123, topP: 0.9, topK: 10, minKeep: 1)
        let sampler = LlamaSampler(config: cfg, model: model)
        #expect(sampler.count() >= 2)
        _ = sampler.name()
        // Access first sampler name if available
        if sampler.count() > 0 { _ = sampler.name(at: 0) }
    }
}


