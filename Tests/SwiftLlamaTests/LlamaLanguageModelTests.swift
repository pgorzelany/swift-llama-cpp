#if compiler(>=6.4) && canImport(FoundationModels)
import Foundation
import FoundationModels
import Observation
import Synchronization
import Testing
@testable import SwiftLlama

@Suite(.serialized, .timeLimit(.minutes(2)))
struct LlamaLanguageModelTests {
    @available(iOS 27.0, macOS 27.0, *)
    @Test("Native sessions separate tagged reasoning, report tokens and preserve exact replay")
    func structuredResponse() async throws {
        let engine = ExecutorTestEngine(scripts: [["<thi", "nk>Plan", "</think>", "Hello", " world"]])
        let model = LlamaLanguageModel(engineFactory: { engine })
        let session = LanguageModelSession(model: model, instructions: "Be helpful")
        let response = try await session.respond(to: "Hi")
        #expect(response.content == "Hello world")
        #expect(session.usage.input.totalTokenCount == 17)
        #expect(session.usage.output.totalTokenCount == 5)
        let reasoning = session.transcript.compactMap { entry -> String? in
            guard case .reasoning(let value) = entry else { return nil }
            return value.segments.compactMap { if case .text(let text) = $0 { text.content } else { nil } }.joined()
        }
        #expect(reasoning == ["Plan"])
        let messages = try LlamaTranscriptMapper.messages(session.transcript)
        #expect(messages.last?.content == "<think>Plan</think>Hello world")
        let metadata = try #require(session.transcript.compactMap { entry -> [String: GeneratedContent]? in
            if case .response(let response) = entry { response.metadata } else { nil }
        }.last)
        #expect(try metadata[LlamaLanguageModel.Metadata.reasoningTokensKnown]?.value(Bool.self) == false)
        #expect(try metadata[LlamaLanguageModel.Metadata.timeToFirstToken]?.value(Double.self) != nil)
        await model.unload()
    }

    @available(iOS 27.0, macOS 27.0, *)
    @Test("Stop during reasoning or an answer cancels the engine and permits reuse", arguments: ["<think>Plan", "<think>Plan</think>Partial"])
    @MainActor
    func cancellation(raw: String) async throws {
        let blocked = ExecutorTestGate()
        let engine = ExecutorTestEngine(scripts: [[raw], ["Fresh"]], blockedAfterTokens: blocked)
        let model = LlamaLanguageModel(engineFactory: { engine })
        let session = LanguageModelSession(model: model)
        session.transcriptErrorHandlingPolicy = .preserveTranscript
        let consumer = Task {
            for try await _ in session.streamResponse(to: "First") {}
            try Task.checkCancellation()
        }
        try await engine.paused.wait()
        for await ready in Observations({ session.transcript.contains { entry in
            guard case .reasoning(let value) = entry else { return false }
            return value.segments.contains { if case .text(let text) = $0 { text.content == "Plan" } else { false } }
        } }) {
            if ready { break }
        }
        consumer.cancel()
        await model.cancelAndWait()
        _ = await consumer.result
        #expect(await engine.resetCount == 1)
        #expect(try LlamaTranscriptMapper.messages(session.transcript).contains { $0.content == raw })
        let next = LanguageModelSession(model: model)
        #expect(try await next.respond(to: "Second").content == "Fresh")
        await model.unload()
    }

    @available(iOS 27.0, macOS 27.0, *)
    @Test("Output limits count decoded tokens rather than Apple stream snapshots")
    func outputLimit() async throws {
        let engine = ExecutorTestEngine(scripts: [["One", " two", " three", " four"]])
        let model = LlamaLanguageModel(engineFactory: { engine })
        let session = LanguageModelSession(model: model)
        let response = try await session.respond(to: "Count", options: .init(maximumResponseTokens: 3))
        #expect(response.content == "One two three")
        #expect(session.usage.output.totalTokenCount == 3)
        #expect(await engine.generatedCount == 3)
        await model.unload()
    }

    @available(iOS 27.0, macOS 27.0, *)
    @Test("Repeated thinking blocks replay as one lossless assistant turn")
    func repeatedReasoning() async throws {
        let raw = "<think>First</think>Hello <think>Second</think>world"
        let engine = ExecutorTestEngine(scripts: [["<think>First</think>Hello ", "<think>Second", "</think>world"]])
        let model = LlamaLanguageModel(engineFactory: { engine })
        let session = LanguageModelSession(model: model)
        #expect(try await session.respond(to: "Hi").content == "Hello world")
        let messages = try LlamaTranscriptMapper.messages(session.transcript)
        #expect(messages.filter { $0.role == .assistant }.map(\.content) == [raw])
        let interrupted = Transcript(entries: [
            .reasoning(.init(metadata: [LlamaLanguageModel.Metadata.rawOutput: "<think>Partial"], segments: [.text(.init(content: "Partial"))])),
            .prompt(.init(segments: [.text(.init(content: "Continue"))]))
        ])
        #expect(try LlamaTranscriptMapper.messages(interrupted).map(\.content) == ["<think>Partial", "Continue"])
        await model.unload()
    }

    @available(iOS 27.0, macOS 27.0, *)
    @Test("Independent model owners do not share execution or cancellation")
    func independentOwners() async throws {
        let engine = ExecutorTestEngine(scripts: [["Partial"]], blockedAfterTokens: ExecutorTestGate())
        let first = LlamaLanguageModel(engineFactory: { engine })
        let secondEngine = ExecutorTestEngine(scripts: [["Independent"]])
        let second = LlamaLanguageModel(engineFactory: { secondEngine })
        #expect(first.executorConfiguration != second.executorConfiguration)
        let pending = Task { try await LanguageModelSession(model: first).respond(to: "Wait") }
        try await engine.paused.wait()
        do {
            _ = try await LanguageModelSession(model: first).respond(to: "Overlap")
            Issue.record("A shared model must reject overlapping inference")
        } catch {}
        #expect(try await LanguageModelSession(model: second).respond(to: "Proceed").content == "Independent")
        await first.cancelAndWait()
        _ = await pending.result
        #expect(await engine.resetCount == 1)
        #expect(await secondEngine.resetCount == 0)
        await first.unload()
        await second.unload()
    }

    @available(iOS 27.0, macOS 27.0, *)
    @Test("Concurrent stop and unload await suspended warmup and allow a fresh load")
    func stopWarmup() async throws {
        let block = ExecutorTestGate()
        let engine = ExecutorTestEngine(scripts: [["Ready"]], blockedPreparation: block)
        let factoryCalls = Mutex(0)
        let model = LlamaLanguageModel(engineFactory: {
            factoryCalls.withLock { $0 += 1 }
            return engine
        })
        let preparation = Task { try await model.prewarm(transcript: promptTranscript("Warmup")) }
        try await engine.preparing.wait()
        async let stop: Void = model.cancelAndWait()
        async let unload: Void = model.unload()
        _ = await (stop, unload)
        do { try await preparation.value; Issue.record("Preparation should be cancelled") } catch is CancellationError {}
        #expect(await engine.resetCount == 1)
        let session = LanguageModelSession(model: model)
        #expect(try await session.respond(to: "After unload").content == "Ready")
        #expect(factoryCalls.withLock { $0 } == 2)
        await model.unload()
    }

    @available(iOS 27.0, macOS 27.0, *)
    @Test("Failed inference is reset before the same model is reused")
    func failureRecovery() async throws {
        let engine = ExecutorTestEngine(scripts: [["Partial"], ["Recovered"]], failAfterTokens: true)
        let model = LlamaLanguageModel(engineFactory: { engine })
        do {
            _ = try await LanguageModelSession(model: model).respond(to: "First")
            Issue.record("Expected the scripted engine error")
        } catch {}
        await model.cancelAndWait()
        #expect(await engine.resetCount == 1)
        #expect(try await LanguageModelSession(model: model).respond(to: "Next").content == "Recovered")
        await model.unload()
    }

    @available(iOS 27.0, macOS 27.0, *)
    @Test("Mapping honors sampling, rejects unsupported features and retains raw history")
    func mapping() throws {
        let options = GenerationOptions(samplingMode: .random(top: 12, seed: 42), temperature: 0.4, maximumResponseTokens: 20)
        var request = LanguageModelExecutorGenerationRequest(
            id: UUID(), transcript: promptTranscript("Hello"), enabledTools: [],
            generationOptions: options, contextOptions: .init(), metadata: [:]
        )
        let sampling = try LlamaTranscriptMapper.sampling(request)
        #expect(sampling.topK == 12)
        #expect(sampling.seed == 42)
        #expect(sampling.temperature == 0.4)
        request.contextOptions = .init(reasoningLevel: .light)
        #expect(throws: LanguageModelError.self) { try LlamaTranscriptMapper.sampling(request) }
        request.contextOptions = .init()
        request.generationOptions.maximumResponseTokens = 0
        #expect(throws: LlamaExecutorError.self) { try LlamaTranscriptMapper.sampling(request) }
        let transcript = Transcript(entries: [
            .reasoning(.init(segments: [.text(.init(content: "Plan"))])),
            .response(.init(metadata: [LlamaLanguageModel.Metadata.rawOutput: "<think>Plan"], segments: []))
        ])
        #expect(try LlamaTranscriptMapper.messages(transcript).last?.content == "<think>Plan")
        let model = LlamaLanguageModel(modelURL: URL(fileURLWithPath: "/unused.gguf"), configuration: .init(batchSize: 32, maxTokenCount: 256))
        #expect(!model.capabilities.contains(.vision))
        #expect(!model.capabilities.contains(.toolCalling))
        #expect(!model.capabilities.contains(.guidedGeneration))
    }

    @available(iOS 27.0, macOS 27.0, *)
    @Test("Tagged reasoning works at every delimiter split")
    func parserSplits() {
        let raw = "<think>First</think>Hello <think>Second</think>world"
        for count in 0...raw.count {
            var parser = LlamaReasoningParser()
            _ = parser.append(String(raw.prefix(count)))
            _ = parser.append(String(raw.dropFirst(count)))
            _ = parser.finish()
            #expect(parser.rawText == raw)
            #expect(parser.answer == "Hello world")
            #expect(parser.reasoning == ["First", "Second"])
        }
    }

    @available(iOS 27.0, macOS 27.0, *)
    @Test("One thousand direct-executor tokens preserve output with bounded overhead")
    func syntheticThroughput() async throws {
        let engine = ExecutorTestEngine(scripts: [Array(repeating: "word ", count: 1_000)])
        let model = LlamaLanguageModel(engineFactory: { engine })
        let session = LanguageModelSession(model: model)
        let start = ContinuousClock.now
        let response = try await session.respond(to: "Stream")
        let elapsed = start.duration(to: .now)
        print("LLAMA_NATIVE_SESSION_1000_TOKENS \(elapsed)")
        #expect(response.content == String(repeating: "word ", count: 1_000))
        #expect(session.usage.output.totalTokenCount == 1_000)
        #expect(elapsed < .seconds(10))
        await model.unload()
    }

    @available(iOS 27.0, macOS 27.0, *)
    @Test("A real GGUF answers sensibly through the native Apple executor")
    func realGGUF() async throws {
        let model = LlamaLanguageModel(modelURL: .llama1B, configuration: .init(batchSize: 128, maxTokenCount: 512))
        let session = LanguageModelSession(model: model)
        let start = ContinuousClock.now
        let response = try await session.respond(
            to: "Answer in one word: What is the capital of France?",
            options: .init(samplingMode: .greedy, maximumResponseTokens: 24)
        )
        print("LLAMA_NATIVE_GGUF_OUTPUT \(response.content) ELAPSED \(start.duration(to: .now)) TOKENS \(session.usage.output.totalTokenCount)")
        #expect(response.content.localizedCaseInsensitiveContains("Paris"))
        #expect(!response.content.contains("�"))
        #expect(session.usage.output.totalTokenCount > 0)
        #expect(session.usage.output.totalTokenCount <= 24)
        #expect(session.usage.input.totalTokenCount > 0)
        await model.unload()
    }
}

@available(iOS 27.0, macOS 27.0, *)
private func promptTranscript(_ text: String) -> Transcript {
    Transcript(entries: [.prompt(.init(segments: [.text(.init(content: text))]))])
}

private final class ExecutorTestGate: Sendable {
    private let stream: AsyncStream<Void>
    private let continuation: AsyncStream<Void>.Continuation

    init() { (stream, continuation) = AsyncStream.makeStream() }
    func open() { continuation.finish() }
    func wait() async throws {
        for await _ in stream {}
        try Task.checkCancellation()
    }
}

@available(iOS 27.0, macOS 27.0, *)
private actor ExecutorTestEngine: LlamaExecutorEngine {
    nonisolated let paused = ExecutorTestGate()
    nonisolated let preparing = ExecutorTestGate()
    private var scripts: [[String]]
    private var tokens: [String] = []
    private var index = 0
    private var blockedAfterTokens: ExecutorTestGate?
    private var blockedPreparation: ExecutorTestGate?
    private var failAfterTokens: Bool
    private(set) var resetCount = 0
    private(set) var generatedCount = 0

    init(scripts: [[String]], blockedAfterTokens: ExecutorTestGate? = nil, blockedPreparation: ExecutorTestGate? = nil, failAfterTokens: Bool = false) {
        self.scripts = scripts
        self.blockedAfterTokens = blockedAfterTokens
        self.blockedPreparation = blockedPreparation
        self.failAfterTokens = failAfterTokens
    }

    func prepare(_ messages: [LlamaChatMessage], addingAssistant: Bool) async throws -> Int {
        preparing.open()
        if let block = blockedPreparation {
            blockedPreparation = nil
            try await block.wait()
        }
        if addingAssistant {
            tokens = scripts.isEmpty ? [] : scripts.removeFirst()
            index = 0
        }
        return 17
    }

    func updateSamplingConfig(_ config: LlamaSamplingConfig) {}

    func generateNextToken() async throws -> NextToken {
        if index < tokens.count {
            defer { index += 1; generatedCount += 1 }
            return .token(tokens[index])
        }
        if let block = blockedAfterTokens {
            blockedAfterTokens = nil
            paused.open()
            try await block.wait()
        }
        if failAfterTokens {
            failAfterTokens = false
            throw LlamaContextError.decodingError
        }
        return .endOfString
    }

    func resetCompletion() { resetCount += 1 }
}
#endif
