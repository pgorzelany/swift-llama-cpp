import Foundation
import Testing
@testable import SwiftLlama

private let gemma4GGUFPath = ProcessInfo.processInfo.environment["GEMMA4_GGUF_PATH"]

@Suite("Gemma compatibility", .serialized)
struct GemmaCompatibilityTests {
    init() {
        LlamaLog.setLogger(nil)
    }

    @Test("Gemma 4 loads and follows a text instruction", .enabled(if: gemma4GGUFPath != nil))
    func gemma4TextGeneration() async throws {
        let modelPath = try #require(gemma4GGUFPath)
        let service = LlamaService(
            modelUrl: URL(fileURLWithPath: modelPath),
            config: .init(batchSize: 256, maxTokenCount: 512, useGPU: true)
        )
        let stream = try await service.streamCompletion(
            of: [
                LlamaChatMessage(role: .system, content: "Follow the user's output format exactly. Do not explain."),
                LlamaChatMessage(role: .user, content: "Reply with exactly GEMMA4_OK")
            ],
            samplingConfig: .init(temperature: 0.0, seed: 42)
        )

        var output = ""
        var tokenCount = 0
        for try await token in stream {
            output += token
            tokenCount += 1
            if tokenCount == 64 {
                await service.stopCompletion()
                break
            }
        }

        print("GEMMA4_RESPONSE_START\n\(output)\nGEMMA4_RESPONSE_END")
        #expect(!output.isEmpty)
        #expect(output.localizedCaseInsensitiveContains("GEMMA4_OK"))
        #expect(!output.contains("�"))
    }
}
