#if compiler(>=6.4) && canImport(FoundationModels)
import Foundation
import FoundationModels
import Testing
@testable import SwiftLlama

@Suite(.serialized, .timeLimit(.minutes(3)))
struct LlamaToolCallingTests {
    @available(iOS 27.0, macOS 27.0, *)
    static var definitions: [Transcript.ToolDefinition] {
        [.init(name: "calculator", description: "Perform arithmetic", parameters: Calculator.Arguments.generationSchema)]
    }

    @available(iOS 27.0, macOS 27.0, *)
    @Test("Every possible fragment boundary hides call syntax and preserves exact arguments")
    func fragmentation() throws {
        let raw = "Before " + call(17, "multiply", 23)
        for boundary in 0...raw.count {
            var parser = LlamaToolParser()
            let split = raw.index(raw.startIndex, offsetBy: boundary)
            let visible = try parser.append(String(raw[..<split])) + parser.append(String(raw[split...]))
            let calls = try parser.finish(definitions: Self.definitions).1
            #expect(visible == "Before ")
            #expect(calls.count == 1)
            #expect(calls.first?.arguments == #"{"lhs":17,"operation":"multiply","rhs":23}"#)
        }
        var parser = LlamaToolParser()
        var visible = ""
        for character in raw { visible += try parser.append(String(character)) }
        #expect(try parser.finish(definitions: Self.definitions).1.count == 1)
        #expect(visible == "Before ")
    }

    @available(iOS 27.0, macOS 27.0, *)
    @Test("The pinned template renders exact tool turns and omits an empty initial system turn")
    func template() {
        let messages: [LlamaChatMessage] = [
            .init(role: .system, content: ""), .init(role: .user, content: "Compute"),
            .init(role: .assistant, content: call(1, "add", 2)), .init(role: .tool, content: "3")
        ]
        let expected = "<|startoftext|><|im_start|>user\nCompute<|im_end|>\n<|im_start|>assistant\n" + call(1, "add", 2) + "<|im_end|>\n<|im_start|>tool\n3<|im_end|>\n"
        #expect(LlamaModel.renderLFMToolPrompt(messages, bos: "<|startoftext|>", addAssistant: false) == expected)
        #expect(LlamaModel.renderLFMToolPrompt(messages, bos: "<|startoftext|>", addAssistant: true) == expected + "<|im_start|>assistant\n")
    }

    @available(iOS 27.0, macOS 27.0, *)
    @Test("Escaped strings and nested JSON values survive Pythonic argument parsing")
    func argumentValues() throws {
        let schema = try JSONDecoder().decode(GenerationSchema.self, from: Data(#"{"title":"Arguments","type":"object","properties":{"title":{"type":"string"},"items":{"type":"array","items":{"type":"integer"}},"enabled":{"type":"boolean"}},"required":["title","items","enabled"],"x-order":["title","items","enabled"],"additionalProperties":false}"#.utf8))
        let definition = Transcript.ToolDefinition(name: "save", description: "Save", parameters: schema)
        var parser = LlamaToolParser()
        let raw = #"<|tool_call_start|>[save(title='It\'s \u00e9\n', items=[1, -2, 3], enabled=True)]<|tool_call_end|>Done."#
        for character in raw { #expect(try parser.append(String(character)).isEmpty) }
        let (trailing, calls) = try parser.finish(definitions: [definition])
        #expect(trailing == "Done.")
        let call = try #require(calls.first)
        let arguments = try JSONSerialization.jsonObject(with: Data(call.arguments.utf8)) as? [String: Any]
        #expect(arguments?["title"] as? String == "It's é\n")
        #expect(arguments?["items"] as? [Int] == [1, -2, 3])
        #expect(arguments?["enabled"] as? Bool == true)
        let transcript = Transcript(entries: [.instructions(.init(segments: [], toolDefinitions: [definition]))])
        let messages = try LlamaTranscriptMapper.messages(transcript, profile: .lfm2_5InstructQ4_0)
        #expect(messages[0].content.contains(#""title": {"type": "string"}"#))
        #expect(!messages[0].content.contains("x-order"))
    }

    @available(iOS 27.0, macOS 27.0, *)
    @Test("Malformed batches cannot partially execute", arguments: [
        "[calculator(lhs=1, operation='add', rhs=)]", "[calculator(lhs=1, lhs=2, operation='add', rhs=3)]",
        "[calculator(lhs=True, operation='add', rhs=3)]", "[calculator(lhs=1, operation='add')]",
        "[calculator(lhs=1, operation='add', rhs=2, extra=4)]", "[missing(lhs=1)]",
        "[calculator(lhs=1, operation='add', rhs=2), missing()]", "[]",
        "[calculator(lhs=1, operation='add', rhs=2)] garbage"
    ])
    func malformed(body: String) throws {
        var parser = LlamaToolParser()
        #expect(try parser.append(LlamaToolParser.start + body + LlamaToolParser.end).isEmpty)
        #expect(throws: (any Error).self) { try parser.finish(definitions: Self.definitions) }
    }

    @available(iOS 27.0, macOS 27.0, *)
    @Test("Truncated control tokens and calls fail without leaking text")
    func truncated() throws {
        for raw in ["<|tool_call_sta", LlamaToolParser.start + "[calculator(lhs=1", call(1, "add", 2) + "<|tool_call_start|>"] {
            var parser = LlamaToolParser()
            #expect(try parser.append(raw).isEmpty)
            #expect(throws: (any Error).self) { try parser.finish(definitions: Self.definitions) }
        }
    }

    @available(iOS 27.0, macOS 27.0, *)
    @Test("Native sessions execute sequential calls, replay results and keep token accounting")
    func nativeSession() async throws {
        let first = "<think>Compute.</think>" + call(17, "multiply", 23)
        let second = call(391, "add", 9)
        let engine = ToolTestEngine(scripts: [first.map(String.init), second.map(String.init), ["400"]])
        let model = LlamaLanguageModel(capabilityProfile: .lfm2_5InstructQ4_0, engineFactory: { engine })
        let tool = Calculator()
        let session = LanguageModelSession(model: model, tools: [tool])
        let response = try await session.respond(to: "Calculate")
        #expect(response.content == "400")
        #expect(await tool.calls.count == 2)
        #expect(session.usage.input.totalTokenCount == 51)
        #expect(session.usage.output.totalTokenCount == first.count + second.count + 1)
        let history = try LlamaTranscriptMapper.messages(session.transcript, profile: .lfm2_5InstructQ4_0)
        #expect(history.filter { $0.role == .assistant }.map(\.content) == [first, second, "400"])
        #expect(history.filter { $0.role == .tool }.map(\.content) == ["391", "400"])
        #expect(history.first?.content.contains("List of tools:") == true)
        #expect(history.first?.content.contains("\"properties\"") == true)
        await model.unload()
    }

    @available(iOS 27.0, macOS 27.0, *)
    @Test("Tool syntax inside reasoning never executes", arguments: ["<think>", "<|channel>thought\n"])
    func reasoningCallExample(open: String) async throws {
        let close = open == "<think>" ? "</think>" : "<channel|>"
        let raw = open + "Example: " + call(1, "add", 2) + close + "Answer"
        let engine = ToolTestEngine(scripts: [raw.map(String.init)])
        let tool = Calculator()
        let model = LlamaLanguageModel(capabilityProfile: .lfm2_5InstructQ4_0, engineFactory: { engine })
        let session = LanguageModelSession(model: model, tools: [tool])
        #expect(try await session.respond(to: "Explain").content == "Answer")
        #expect(await tool.calls.isEmpty)
        let messages = try LlamaTranscriptMapper.messages(session.transcript, profile: .lfm2_5InstructQ4_0)
        #expect(messages.last?.content == raw)
    }

    @available(iOS 27.0, macOS 27.0, *)
    @Test("Multiple calls in a batch execute once and trailing prose stays separate")
    func batchAndTrailingProse() async throws {
        let raw = "<|tool_call_start|>[calculator(lhs=1, operation='add', rhs=2), calculator(lhs=3, operation='multiply', rhs=4)]<|tool_call_end|>Checking both."
        let engine = ToolTestEngine(scripts: [raw.map(String.init), ["3 and 12"]])
        let model = LlamaLanguageModel(capabilityProfile: .lfm2_5InstructQ4_0, engineFactory: { engine })
        let tool = Calculator()
        let session = LanguageModelSession(model: model, tools: [tool])
        #expect(try await session.respond(to: "Calculate both").content == "3 and 12")
        #expect(await tool.calls.count == 2)
        let messages = try LlamaTranscriptMapper.messages(session.transcript, profile: .lfm2_5InstructQ4_0)
        #expect(messages.filter { $0.role == .assistant }.first?.content == raw)
        #expect(messages.filter { $0.role == .tool }.count == 2)
    }

    @available(iOS 27.0, macOS 27.0, *)
    @Test("The output token limit cannot execute a truncated call")
    func outputLimit() async throws {
        let engine = ToolTestEngine(scripts: [call(17, "multiply", 23).map(String.init)])
        let tool = Calculator()
        let model = LlamaLanguageModel(capabilityProfile: .lfm2_5InstructQ4_0, engineFactory: { engine })
        let session = LanguageModelSession(model: model, tools: [tool])
        await #expect(throws: (any Error).self) { try await session.respond(to: "Compute", options: .init(maximumResponseTokens: 25)) }
        #expect(await tool.calls.isEmpty)
    }

    @available(iOS 27.0, macOS 27.0, *)
    @Test("Cancelling a buffered tool call never dispatches it")
    func cancelBufferedCall() async throws {
        let engine = ToolTestEngine(scripts: [[call(17, "multiply", 23)]], pauseAfterTokens: true)
        let tool = Calculator()
        let model = LlamaLanguageModel(capabilityProfile: .lfm2_5InstructQ4_0, engineFactory: { engine })
        let session = LanguageModelSession(model: model, tools: [tool])
        let response = Task { try await session.respond(to: "Compute") }
        for await _ in engine.paused { break }
        response.cancel()
        await model.cancelAndWait()
        _ = await response.result
        #expect(await tool.calls.isEmpty)
    }

    @available(iOS 27.0, macOS 27.0, *)
    @Test("Unverified profiles reject definitions and never advertise tool calling")
    func capabilityGate() async throws {
        let engine = ToolTestEngine(scripts: [["Hello"]])
        let model = LlamaLanguageModel(engineFactory: { engine })
        #expect(!model.capabilities.contains(.toolCalling))
        let transcript = Transcript(entries: [.instructions(.init(segments: [], toolDefinitions: Self.definitions)), .prompt(.init(segments: [.text(.init(content: "Hi"))]))])
        #expect(throws: (any Error).self) { try LlamaTranscriptMapper.messages(transcript) }
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try Data("not the shipped model".utf8).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(throws: LlamaToolError.unverifiedModel) { try LlamaCapabilityProfile.lfm2_5InstructQ4_0.validate(modelURL: url) }
    }

    @available(iOS 27.0, macOS 27.0, *)
    @Test("Disallowed tools cannot execute and required choice is explicitly rejected")
    func toolModes() async throws {
        for mode in [GenerationOptions.ToolCallingMode.disallowed, .required] {
            let tool = Calculator()
            let engine = ToolTestEngine(scripts: [[call(1, "add", 2)]])
            let model = LlamaLanguageModel(capabilityProfile: .lfm2_5InstructQ4_0, engineFactory: { engine })
            let session = LanguageModelSession(model: model, tools: [tool])
            await #expect(throws: (any Error).self) { try await session.respond(to: "Compute", options: .init(toolCallingMode: mode)) }
            #expect(await tool.calls.isEmpty)
        }
    }

    @available(iOS 27.0, macOS 27.0, *)
    @Test("A tool failure propagates through Apple's session and the executor remains reusable")
    func thrownToolError() async throws {
        let engine = ToolTestEngine(scripts: [[call(1, "throw", 0)], ["Recovered"]])
        let model = LlamaLanguageModel(capabilityProfile: .lfm2_5InstructQ4_0, engineFactory: { engine })
        let session = LanguageModelSession(model: model, tools: [Calculator()])
        await #expect(throws: (any Error).self) { try await session.respond(to: "Fail") }
        let next = LanguageModelSession(model: model)
        #expect(try await next.respond(to: "Continue").content == "Recovered")
        await model.unload()
    }

    @available(iOS 27.0, macOS 27.0, *)
    @Test("A malformed generation executes no tools and leaves the executor reusable")
    func invalidGeneration() async throws {
        let tool = Calculator()
        let engine = ToolTestEngine(scripts: [[call(1, "add", 2).replacingOccurrences(of: "rhs=2", with: "rhs='bad'")], ["Hello"]])
        let model = LlamaLanguageModel(capabilityProfile: .lfm2_5InstructQ4_0, engineFactory: { engine })
        let session = LanguageModelSession(model: model, tools: [tool])
        await #expect(throws: (any Error).self) { try await session.respond(to: "Compute") }
        #expect(await tool.calls.isEmpty)
        let next = LanguageModelSession(model: model)
        #expect(try await next.respond(to: "Hi").content == "Hello")
    }
}

@Suite(.serialized, .timeLimit(.minutes(10)))
struct LlamaToolCallingIntegrationTests {
    @available(iOS 27.0, macOS 27.0, *)
    @Test("Shipped LFM calls exact arguments, uses results, avoids unrelated calls and recovers from an error",
          .enabled(if: ProcessInfo.processInfo.environment["ENCLAVE_GGUF_TEST_MODEL"] != nil))
    func shippedModel() async throws {
        let path = try #require(ProcessInfo.processInfo.environment["ENCLAVE_GGUF_TEST_MODEL"])
        let model = LlamaLanguageModel(modelURL: URL(fileURLWithPath: path), configuration: .init(batchSize: 512, maxTokenCount: 4096), capabilityProfile: .lfm2_5InstructQ4_0)
        let tool = RecordLookup()
        let session = LanguageModelSession(model: model, tools: [tool], instructions: "You are a helpful assistant with access to tools. Use the available tools to answer user requests.")
        let options = GenerationOptions(samplingMode: .greedy, maximumResponseTokens: 256)
        let first = try await session.respond(to: "Look up record alpha and tell me its secret code.", options: options)
        print("LFM_TOOL_FIRST \(first.content)")
        #expect(await tool.calls == ["alpha"])
        #expect(first.content.contains("7319"))
        let second = try await session.respond(to: "Look up the next record mentioned in that result and tell me its secret code.", options: options)
        print("LFM_TOOL_SECOND \(second.content)")
        #expect(await tool.calls == ["alpha", "beta"])
        #expect(second.content.contains("4826"))
        let unrelated = try await session.respond(to: "What is the capital of France?", options: options)
        #expect(!unrelated.content.isEmpty)
        #expect(await tool.calls.count == 2)
        let error = try await session.respond(to: "Look up record delta and tell me its secret code.", options: options)
        print("LFM_TOOL_ERROR \(error.content)")
        #expect(await tool.calls == ["alpha", "beta", "delta"])
        #expect(error.content.localizedCaseInsensitiveContains("not found") || error.content.localizedCaseInsensitiveContains("does not exist"))
        let recovered = try await session.respond(to: "Look up record gamma instead and tell me its secret code.", options: options)
        #expect(recovered.content.contains("9051"))
        #expect(await tool.calls == ["alpha", "beta", "delta", "gamma"])
        await model.unload()
    }
}

@available(iOS 27.0, macOS 27.0, *)
private actor RecordLookup: Tool {
    nonisolated let name = "lookup_record"
    nonisolated let description = "Retrieve a private record by its ID. Returns its secret code and the ID of the next record, or an error when not found."
    @Generable struct Arguments: Sendable { var record_id: String }
    private(set) var calls: [String] = []
    func call(arguments: Arguments) async throws -> String {
        calls.append(arguments.record_id)
        switch arguments.record_id {
        case "alpha": return "Record alpha: secret code 7319; next record beta."
        case "beta": return "Record beta: secret code 4826; next record gamma."
        case "gamma": return "Record gamma: secret code 9051; no next record."
        default: return "Error: record not found. Ask for another record ID."
        }
    }
}

@available(iOS 27.0, macOS 27.0, *)
private actor Calculator: Tool {
    nonisolated let name = "calculator"
    nonisolated let description = "Perform arithmetic on lhs and rhs. Operations: add, multiply, divide. Returns the exact result or an error."
    @Generable struct Arguments: Sendable {
        var lhs: Double
        var operation: String
        var rhs: Double
    }
    private(set) var calls: [Arguments] = []
    func call(arguments: Arguments) async throws -> String {
        calls.append(arguments)
        let value: Double
        switch arguments.operation {
        case "add": value = arguments.lhs + arguments.rhs
        case "multiply": value = arguments.lhs * arguments.rhs
        case "divide":
            guard arguments.rhs != 0 else { return "Error: division by zero is undefined." }
            value = arguments.lhs / arguments.rhs
        default: throw LlamaToolError.invalidArguments(arguments.operation)
        }
        return value.formatted(.number.grouping(.never))
    }
}

private func call(_ lhs: Int, _ operation: String, _ rhs: Int) -> String {
    "<|tool_call_start|>[calculator(lhs=\(lhs), operation='\(operation)', rhs=\(rhs))]<|tool_call_end|>"
}

@available(iOS 27.0, macOS 27.0, *)
private actor ToolTestEngine: LlamaExecutorEngine {
    var scripts: [[String]]
    var tokens: [String] = []
    nonisolated let paused: AsyncStream<Void>
    private let pauseContinuation: AsyncStream<Void>.Continuation
    private let pauseAfterTokens: Bool
    init(scripts: [[String]], pauseAfterTokens: Bool = false) {
        self.scripts = scripts
        self.pauseAfterTokens = pauseAfterTokens
        (paused, pauseContinuation) = AsyncStream.makeStream()
    }
    func prepare(_ messages: [LlamaChatMessage], addingAssistant: Bool) -> Int {
        if addingAssistant { tokens = scripts.isEmpty ? [] : scripts.removeFirst() }
        return 17
    }
    func updateSamplingConfig(_ config: LlamaSamplingConfig) {}
    func generateNextToken() async throws -> NextToken {
        if !tokens.isEmpty { return .token(tokens.removeFirst()) }
        if pauseAfterTokens {
            pauseContinuation.yield(())
            try await Task.sleep(for: .seconds(60))
        }
        return .endOfString
    }
    func resetCompletion() {}
    func contextUsage(_ messages: [LlamaChatMessage], addingAssistant: Bool) -> LlamaContextUsage { .init(usedTokens: 17, effectiveCapacity: 4000) }
}
#endif
