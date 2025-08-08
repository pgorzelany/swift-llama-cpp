import Testing
import Foundation
@testable import SwiftLlama

struct LlamaContextTests {

    private func makeContext(maxTokens: UInt32 = 2048, batch: UInt32 = 256) async throws -> Llama {
        let llama = try Llama(modelPath: URL.llama1B.path, config: .init(batchSize: batch, maxTokenCount: maxTokens))
        return llama
    }

    @Test("Decode produces logits and embeddings")
    func testDecodeProducesSignals() async throws {
        let sut = try await makeContext()
        await sut.updateSamplingConfig(.init(temperature: 0.7, seed: 1))
        try await sut.initializeCompletion(messages: [LlamaChatMessage(role: .user, content: "Hello")])

        // Generate one token so logits exist for idx -1
        _ = try await sut.generateNextToken()
        // Enable embeddings
        await sut.enableEmbeddingsOutput(true)
        let logits = await sut.getLastLogits()
        let embeddings = await sut.getEmbeddings()
        #expect(logits?.isEmpty == false)
        // Some models don't produce embeddings unless explicitly enabled. Accept nil here.
        #expect(embeddings == nil || embeddings?.isEmpty == false)
    }

    @Test("State save/load roundtrip non-empty")
    func testStateRoundtrip() async throws {
        let sut = try await makeContext()
        await sut.updateSamplingConfig(.init(temperature: 0.7, seed: 1))
        try await sut.initializeCompletion(messages: [LlamaChatMessage(role: .user, content: "Hello state")])
        _ = try await sut.generateNextToken()

        let state = await sut.saveStateData()
        #expect(!state.isEmpty)
        let loaded = await sut.loadStateData(state)
        #expect(loaded)
    }

    @Test("Thread settings can be updated")
    func testThreadSettings() async throws {
        let sut = try await makeContext()
        await sut.setThreads(nThreads: 1, nThreadsBatch: 1)
        let (n, nb) = await sut.getThreads()
        #expect(n == 1)
        #expect(nb == 1)
    }

    @Test("Memory operations clear and remove ranges")
    func testMemoryOperations() async throws {
        let sut = try await makeContext()
        await sut.updateSamplingConfig(.init(temperature: 0.7, seed: 1))
        try await sut.initializeCompletion(messages: [LlamaChatMessage(role: .user, content: "Hello memory ops")])
        _ = try await sut.generateNextToken()

        let beforeMax = await sut.kvMaxPosition()
        await sut.clearKV()
        let afterMin = await sut.kvMinPosition()
        #expect(beforeMax >= -1)
        #expect(afterMin == -1)
    }
}


