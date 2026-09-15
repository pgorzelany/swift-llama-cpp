import Foundation
import Testing

@testable import SwiftLlama

@Suite(
    "Prompt cache reuse", .serialized, .timeLimit(.minutes(2)),
    .enabled(if: ProcessInfo.processInfo.environment["ENCLAVE_GGUF_TEST_MODEL"] != nil,
             "Set ENCLAVE_GGUF_TEST_MODEL to an instruction-tuned GGUF fixture")
)
struct LlamaCacheReuseTests {
    @Test("Repeating a shortened prompt matches a clean context", arguments: [UInt32(1), 128])
    func repeatedPrompt(batchSize: UInt32) async throws {
        let engine = try makeEngine(batchSize: batchSize)
        let messages = [LlamaChatMessage(role: .user, content: "Count from 1 to 100, writing every number.")]
        let first = try await generate(engine, messages: messages)
        let repeated = try await generate(engine, messages: messages)
        await engine.resetCompletion()
        let fresh = try await generate(engine, messages: messages)

        #expect(first.count == 8)
        #expect(repeated == fresh)
        #expect(repeated == first)
        #expect(!repeated.joined().isEmpty)
        #expect(!repeated.joined().contains("\u{FFFD}"))
        #expect(await engine.kvMaxPosition() == engine.currentTokenPosition - 1)
    }

    @Test("A changed suffix matches full reprocessing")
    func changedSuffix() async throws {
        let engine = try makeEngine(batchSize: 128)
        _ = try await generate(engine, messages: [
            .init(role: .user, content: "Count from 1 to 100, writing every number.")
        ])
        let changed = [LlamaChatMessage(
            role: .user, content: "Count from 1 to 100, writing every number, separated by commas."
        )]
        let reused = try await generate(engine, messages: changed)
        await engine.resetCompletion()
        let fresh = try await generate(engine, messages: changed)
        #expect(reused.count == 8)
        #expect(reused == fresh)
    }

    private func makeEngine(batchSize: UInt32) throws -> Llama {
        let path = try #require(ProcessInfo.processInfo.environment["ENCLAVE_GGUF_TEST_MODEL"])
        try #require(FileManager.default.fileExists(atPath: path))
        return try Llama(modelPath: path, config: .init(batchSize: batchSize, maxTokenCount: 512, useGPU: false))
    }

    private func generate(_ engine: Llama, messages: [LlamaChatMessage]) async throws -> [String] {
        await engine.updateSamplingConfig(.init(temperature: 0, seed: 0))
        try await engine.initializeCompletion(messages: messages)
        var tokens: [String] = []
        for _ in 0..<8 {
            try Task.checkCancellation()
            switch try await engine.generateNextToken() {
            case .token(let text): tokens.append(text)
            case .endOfString: return tokens
            }
        }
        return tokens
    }
}
