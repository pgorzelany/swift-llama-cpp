#if compiler(>=6.4) && canImport(FoundationModels)
import Foundation
import FoundationModels
import Testing
@testable import SwiftLlama

@Suite("Stopped native generation cache", .serialized, .timeLimit(.minutes(3)),
       .enabled(if: ProcessInfo.processInfo.environment["ENCLAVE_STOP_CACHE_MODEL"] != nil))
struct LlamaStopCacheTests {
    @available(iOS 27.0, macOS 27.0, *)
    @MainActor
    @Test("A coherent stop reuses the evaluated context for continue")
    func coherentStop() async throws { try await stopThenContinue(inFlight: false) }

    @available(iOS 27.0, macOS 27.0, *)
    @MainActor
    @Test("An unemitted decoded token safely falls back to a clean-context equivalent")
    func inFlightStop() async throws { try await stopThenContinue(inFlight: true) }

    @available(iOS 27.0, macOS 27.0, *)
    @MainActor
    private func stopThenContinue(inFlight: Bool) async throws {
        let path = try #require(ProcessInfo.processInfo.environment["ENCLAVE_STOP_CACHE_MODEL"])
        let engine = try Llama(modelPath: path, config: .init(batchSize: 128, maxTokenCount: 4096), thinkingEnabled: false)
        let probe = NativeStopCacheProbe(engine: engine, inFlight: inFlight)
        let model = LlamaLanguageModel(capabilityProfile: .lfm3QADQ4_0, engineFactory: { probe })
        let session = LanguageModelSession(model: model)
        session.transcriptErrorHandlingPolicy = .preserveTranscript
        #if targetEnvironment(simulator)
        let recordCount = inFlight ? 20 : 100
        #else
        let recordCount = 100
        #endif
        let context = (0..<recordCount).map { "Record \($0): the synthetic blue room contains \($0 + 10) paper cranes." }.joined(separator: "\n")
        let sending = Task {
            for try await _ in session.streamResponse(to: Prompt(context + "\nWrite a long numbered story about these rooms, continuing for at least 100 items."),
                                                     options: .init(samplingMode: .greedy, maximumResponseTokens: 256)) {}
        }
        try await probe.paused.wait()
        let beforeStop = await engine.getProcessedTokenIds().count
        sending.cancel()
        await probe.resume()
        await model.cancelAndWait()
        _ = await sending.result
        let saved = session.transcript
        let afterStop = await engine.getProcessedTokenIds().count
        print("STOP_CACHE in_flight=\(inFlight) before_stop=\(beforeStop) after_stop=\(afterStop)")
        #expect(afterStop == beforeStop)
        #expect(await engine.kvMaxPosition() == Int32(afterStop - 1))
        let continued = LanguageModelSession(model: model, transcript: saved)
        let decodedBefore = await engine.totalPromptDecodedTokens
        try await model.prewarm(transcript: saved)
        let response = try await continued.respond(to: "Continue.", options: .init(samplingMode: .greedy, maximumResponseTokens: 8))
        let decodedAfter = await engine.totalPromptDecodedTokens
        print("STOP_CACHE continuation_prompt_decoded=\(decodedAfter - decodedBefore) cached_before=\(beforeStop) output_tokens=\(response.usage.output.totalTokenCount)")
        if inFlight {
            #expect(decodedAfter - decodedBefore >= beforeStop - 24)
            await engine.resetCompletion()
            let fresh = LanguageModelSession(model: model, transcript: saved)
            let reference = try await fresh.respond(to: "Continue.", options: .init(samplingMode: .greedy, maximumResponseTokens: 8))
            #expect(response.content == reference.content)
        } else {
            #expect(decodedAfter - decodedBefore < 64)
        }
        #expect(response.usage.output.totalTokenCount > 0)
        await model.unload()
    }
}

@available(iOS 27.0, macOS 27.0, *)
private actor NativeStopCacheProbe: LlamaExecutorEngine {
    let engine: Llama
    nonisolated let paused = StopCacheGate()
    private var generated = 0
    private var blocked = false
    private let inFlight: Bool
    private var continuation: CheckedContinuation<Void, Never>?

    init(engine: Llama, inFlight: Bool) { self.engine = engine; self.inFlight = inFlight }
    func resume() { continuation?.resume(); continuation = nil }
    func prepare(_ messages: [LlamaChatMessage], addingAssistant: Bool) async throws -> Int {
        try await engine.prepare(messages, addingAssistant: addingAssistant)
    }
    func updateSamplingConfig(_ config: LlamaSamplingConfig) async throws { try await engine.updateSamplingConfig(config) }
    func generateNextToken() async throws -> NextToken {
        if !inFlight, generated == 24 && !blocked {
            blocked = true
            paused.open()
            try await StopCacheGate().wait()
        }
        let next = try await engine.generateNextToken()
        if case .token = next { generated += 1 }
        if inFlight, generated == 24 && !blocked {
            blocked = true
            paused.open()
            await withCheckedContinuation { continuation = $0 }
        }
        return next
    }
    func finishDecoding() async -> String { await engine.finishDecoding() }
    func resetCompletion() async { await engine.resetCompletion() }
    func contextUsage(_ messages: [LlamaChatMessage], addingAssistant: Bool) async throws -> LlamaContextUsage {
        try await engine.contextUsage(messages, addingAssistant: addingAssistant)
    }
    func hasPrefilledReasoning() async -> Bool { await engine.usesPrefilledReasoning }
}

private final class StopCacheGate: Sendable {
    private let stream: AsyncStream<Void>
    private let continuation: AsyncStream<Void>.Continuation
    init() { (stream, continuation) = AsyncStream.makeStream() }
    func open() { continuation.finish() }
    func wait() async throws {
        for await _ in stream {}
        try Task.checkCancellation()
    }
}
#endif
