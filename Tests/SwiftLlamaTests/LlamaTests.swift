import Testing
import Foundation
@testable import SwiftLlama

struct LlamaTests {

    @Test("Token generation baseline speed and non-empty output")
    @MainActor
    func tokenGenerationSpeed() async throws {
        let sut = try Llama(
            modelPath: URL.llama1B.path,
            config: .init(batchSize: 256, maxTokenCount: 2048)
        )

        await sut.updateSamplingConfig(.init(temperature: 0.7, seed: 0))
        try await sut.initializeCompletion(messages: [LlamaChatMessage(role: .system, content: "Tell me a very long story about mars colonization")])

        let numberOfTokensToGenerate = 50
        var tokensGenerated = 0
        var generatedText = ""

        let startTime = CFAbsoluteTimeGetCurrent()
        for _ in 0..<numberOfTokensToGenerate {
            let nextToken = try await sut.generateNextToken()
            switch nextToken {
            case .token(let token):
                tokensGenerated += 1
                generatedText += token
            case .endOfString:
                break
            }
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        let tps = Double(tokensGenerated) / max(elapsed, 0.0001)

        #expect(tokensGenerated > 0)
        #expect(!generatedText.isEmpty)
        #expect(tps > 20.0)
    }

    @Test("Tokenize and detokenize roundtrip")
    func tokenizeDetokenizeRoundtrip() throws {
        let modelOpt = LlamaModel(path: URL.llama1B.path)
        let model = try #require(modelOpt)
        let text = "Hello, 世界! Emojis: 🚀🔥"
        let tokens = model.tokenize(text: text, addBos: model.shouldAddBos(), special: true)
        let detok = model.detokenize(tokens: tokens, removeSpecial: true, unparseSpecial: false)
        #expect(!tokens.isEmpty)
        #expect(!detok.isEmpty)
    }
}
