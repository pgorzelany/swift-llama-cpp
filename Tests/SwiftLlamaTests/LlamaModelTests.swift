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

    @Test("Detokenize round-trip: simple ASCII")
    func testDetokenizeRoundTripSimple() throws {
        let model = try #require(LlamaModel(path: URL.llama1B.path))
        let input = "Hello, world!"
        let t1 = model.tokenize(text: input, addBos: false, special: false)
        #expect(!t1.isEmpty)
        let text = model.detokenize(tokens: t1, removeSpecial: true, unparseSpecial: false)
        let t2 = model.tokenize(text: text, addBos: false, special: false)
        #expect(t1 == t2)
    }

    @Test("Detokenize round-trip: whitespace and newlines")
    func testDetokenizeRoundTripWhitespace() throws {
        let model = try #require(LlamaModel(path: URL.llama1B.path))
        let input = "  Leading and trailing  spaces\nwith\tmixed\twhitespace  "
        let t1 = model.tokenize(text: input, addBos: false, special: false)
        #expect(!t1.isEmpty)
        let text = model.detokenize(tokens: t1, removeSpecial: true, unparseSpecial: false)
        let t2 = model.tokenize(text: text, addBos: false, special: false)
        #expect(t1 == t2)
    }

    @Test("Detokenize round-trip: emojis and symbols")
    func testDetokenizeRoundTripUnicodeEmoji() throws {
        let model = try #require(LlamaModel(path: URL.llama1B.path))
        let input = "Emojis 🎯🚀🔥 — currency €£¥, math ±≈∑, quotes “smart”"
        let t1 = model.tokenize(text: input, addBos: false, special: false)
        #expect(!t1.isEmpty)
        let text = model.detokenize(tokens: t1, removeSpecial: true, unparseSpecial: false)
        let t2 = model.tokenize(text: text, addBos: false, special: false)
        #expect(t1 == t2)
    }

    @Test("Detokenize round-trip: multilingual")
    func testDetokenizeRoundTripMultilingual() throws {
        let model = try #require(LlamaModel(path: URL.llama1B.path))
        let input = "français español 中文 العربية हिन्दी русский"
        let t1 = model.tokenize(text: input, addBos: false, special: false)
        #expect(!t1.isEmpty)
        let text = model.detokenize(tokens: t1, removeSpecial: true, unparseSpecial: false)
        let t2 = model.tokenize(text: text, addBos: false, special: false)
        #expect(t1 == t2)
    }

    @Test("Detokenize round-trip: longer text with punctuation and numbers")
    func testDetokenizeRoundTripLonger() throws {
        let model = try #require(LlamaModel(path: URL.llama1B.path))
        let input = "In 2025, test-cases should cover edge-cases: e.g., URLs, emails, and 100% of basics."
        let t1 = model.tokenize(text: input, addBos: false, special: false)
        #expect(!t1.isEmpty)
        let text = model.detokenize(tokens: t1, removeSpecial: true, unparseSpecial: false)
        let t2 = model.tokenize(text: text, addBos: false, special: false)
        #expect(t1 == t2)
    }

    @Test("Round-trip with model.shouldAddBos() setting")
    func testDetokenizeRoundTripWithModelBosSetting() throws {
        let model = try #require(LlamaModel(path: URL.llama1B.path))
        let input = "Round-trip with optional BOS/EOS handling."
        let addBos = model.shouldAddBos()
        let t1 = model.tokenize(text: input, addBos: addBos, special: false)
        #expect(!t1.isEmpty)
        // Remove special when detokenizing; re-tokenize with same addBos to reproduce the same sequence
        let text = model.detokenize(tokens: t1, removeSpecial: true, unparseSpecial: false)
        let t2 = model.tokenize(text: text, addBos: addBos, special: false)
        #expect(t1 == t2)
    }

    @Test("Special tokens: unparse + parse round-trip for BOS")
    func testSpecialTokenUnparseParseRoundTripBOS() throws {
        let model = try #require(LlamaModel(path: URL.llama1B.path))
        let input = "Special token handling check."
        let core = model.tokenize(text: input, addBos: false, special: false)
        #expect(!core.isEmpty)
        // Prepend BOS explicitly to guarantee the presence of a special token in the stream
        let bos = model.bosToken()
        let withBos = [bos] + core
        // Detokenize preserving special tokens and rendering them into text
        let rendered = model.detokenize(tokens: withBos, removeSpecial: false, unparseSpecial: true)
        // Re-tokenize allowing special token parsing; do not auto-add BOS here
        let reparsed = model.tokenize(text: rendered, addBos: false, special: true)
        #expect(reparsed == withBos)
    }
}


