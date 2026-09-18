import Foundation
import Testing
@testable import SwiftLlama

@Suite("Model behavior", .serialized)
struct LlamaBehaviorTests {
    private let samplingConfig = LlamaSamplingConfig(temperature: 0.0, seed: 42)

    init() {
        LlamaLog.setLogger(nil)
    }

    @Test("Factual response follows the requested topic")
    func factualResponse() async throws {
        let output = try await generate(
            messages: [
                LlamaChatMessage(
                    role: .user,
                    content: "Answer in one short sentence: What is the capital of France?"
                )
            ]
        )

        print("LLAMA_FACTUAL_RESPONSE_START\n\(output)\nLLAMA_FACTUAL_RESPONSE_END")
        #expect(output.localizedCaseInsensitiveContains("Paris"))
        #expect(!output.contains("<|"))
        #expect(!output.contains("�"))
    }

    @Test("Multi-turn prompt preserves conversation facts")
    func multiTurnRecall() async throws {
        let output = try await generate(
            messages: [
                LlamaChatMessage(role: .system, content: "Answer the final question using the conversation. Be concise."),
                LlamaChatMessage(role: .user, content: "The project passphrase is cobalt-lantern. Remember it."),
                LlamaChatMessage(role: .assistant, content: "I will remember that the project passphrase is cobalt-lantern."),
                LlamaChatMessage(role: .user, content: "What is the project passphrase? Reply only with the passphrase.")
            ]
        )

        let normalized = output.lowercased()
        #expect(normalized.contains("cobalt"))
        #expect(normalized.contains("lantern"))
    }

    @Test("Service returns to a clean deterministic state after cancellation")
    func generationAfterCancellation() async throws {
        let service = makeService(maxTokenCount: 256)
        let firstStream = try await service.streamCompletion(
            of: [LlamaChatMessage(role: .user, content: "Write a long story about a lighthouse.")],
            samplingConfig: samplingConfig
        )

        var firstTokenCount = 0
        for try await _ in firstStream {
            firstTokenCount += 1
            if firstTokenCount == 8 {
                await service.stopCompletion()
                break
            }
        }

        let verificationMessages = [
            LlamaChatMessage(
                role: .user,
                content: "Answer in one word: What color is a clear daytime sky?"
            )
        ]
        let outputAfterCancellation = try await collect(
            from: service,
            messages: verificationMessages,
            maxTokens: 32
        )
        let freshOutput = try await collect(
            from: makeService(maxTokenCount: 256),
            messages: verificationMessages,
            maxTokens: 32
        )

        #expect(firstTokenCount == 8)
        #expect(!outputAfterCancellation.isEmpty)
        #expect(outputAfterCancellation == freshOutput)
    }

    @Test("Oversized prompt reports the context limit")
    func oversizedPrompt() async throws {
        let service = makeService(maxTokenCount: 64)
        let oversizedPrompt = String(repeating: "This prompt exceeds the context window. ", count: 100)

        do {
            _ = try await service.streamCompletion(
                of: [LlamaChatMessage(role: .user, content: oversizedPrompt)],
                samplingConfig: samplingConfig
            )
            Issue.record("Expected the oversized prompt to be rejected")
        } catch LlamaError.contextSizeLimitExeeded {
            // Expected.
        } catch {
            Issue.record("Expected a context limit error, received \(error)")
        }
    }

    @Test("Reported effective capacity matches the inference boundary without mutating state")
    func contextUsageBoundary() async throws {
        let llama = try Llama(
            modelPath: URL.llama1B.path,
            config: .init(batchSize: 64, maxTokenCount: 64, useGPU: false)
        )
        var byCount: [Int: [LlamaChatMessage]] = [:]
        for wordCount in 1...160 {
            let messages = [LlamaChatMessage(role: .user, content: String(repeating: "x ", count: wordCount))]
            let usage = try await llama.contextUsage(messages, addingAssistant: true)
            byCount[usage.usedTokens] = messages
            if byCount[usage.effectiveCapacity] != nil, byCount[usage.effectiveCapacity + 1] != nil { break }
        }
        let accepted = try #require(byCount[59])
        let rejected = try #require(byCount[60])
        #expect(await llama.getProcessedTokenIds().isEmpty)
        try await llama.initializeCompletion(messages: accepted, addAssistant: true)
        #expect(await llama.getProcessedTokenIds().count == 59)
        let before = await llama.getProcessedTokenIds()
        _ = try await llama.contextUsage(accepted, addingAssistant: true)
        #expect(await llama.getProcessedTokenIds() == before)
        do {
            try await llama.initializeCompletion(messages: rejected, addAssistant: true)
            Issue.record("Expected the first prompt above effective capacity to be rejected")
        } catch LlamaError.contextSizeLimitExeeded {
        } catch {
            Issue.record("Expected contextSizeLimitExeeded, got \(error)")
        }
    }

    private func generate(messages: [LlamaChatMessage], maxTokens: Int = 64) async throws -> String {
        try await collect(from: makeService(maxTokenCount: 256), messages: messages, maxTokens: maxTokens)
    }

    private func makeService(maxTokenCount: UInt32) -> LlamaService {
        LlamaService(
            modelUrl: .llama1B,
            config: .init(batchSize: 256, maxTokenCount: maxTokenCount, useGPU: true)
        )
    }

    private func collect(
        from service: LlamaService,
        messages: [LlamaChatMessage],
        maxTokens: Int
    ) async throws -> String {
        let stream = try await service.streamCompletion(of: messages, samplingConfig: samplingConfig)
        var output = ""
        var tokenCount = 0

        for try await token in stream {
            output += token
            tokenCount += 1
            if tokenCount == maxTokens {
                await service.stopCompletion()
                break
            }
        }

        return output
    }
}
