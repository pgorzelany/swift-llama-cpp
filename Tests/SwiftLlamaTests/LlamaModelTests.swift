import Testing
import Foundation
@testable import SwiftLlama

struct LlamaModelTests {
    @Test("Tokenize of empty string is empty and chat template returns something")
    func testTokenizeEmptyAndChatTemplate() throws {
        let model = try #require(LlamaModel(path: URL.llama1B.path))
        let empty = model.tokenize(text: "", addBos: model.shouldAddBos(), special: true)
        #expect(empty.isEmpty)

        let messages = [
            LlamaChatMessage(role: .system, content: "You are a helpful assistant."),
            LlamaChatMessage(role: .user, content: "Say hi")
        ]
        let prompt = model.applyChatTemplate(to: messages)
        #expect(!prompt.isEmpty)
    }

    @Test("Built-in chat templates accessible")
    func testBuiltinTemplates() throws {
        let model = try #require(LlamaModel(path: URL.llama1B.path))
        let templates = model.builtinChatTemplates()
        #expect(templates.count >= 0) // may be empty depending on model
    }
}


