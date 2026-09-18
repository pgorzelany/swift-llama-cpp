#if compiler(>=6.4) && canImport(FoundationModels)
import Foundation
import FoundationModels
import Synchronization

/// Internal token-level seam for deterministic lifecycle tests; it is not a public completion API.
@available(iOS 27.0, macOS 27.0, *)
protocol LlamaExecutorEngine: Actor {
    func prepare(_ messages: [LlamaChatMessage], addingAssistant: Bool) async throws -> Int
    func updateSamplingConfig(_ config: LlamaSamplingConfig) async
    func generateNextToken() async throws -> NextToken
    func resetCompletion() async
    func contextUsage(_ messages: [LlamaChatMessage], addingAssistant: Bool) async throws -> LlamaContextUsage
}

@available(iOS 27.0, macOS 27.0, *)
extension Llama: LlamaExecutorEngine {
    func prepare(_ messages: [LlamaChatMessage], addingAssistant: Bool) throws -> Int {
        try Task.checkCancellation()
        try initializeCompletion(messages: messages, addAssistant: addingAssistant)
        try Task.checkCancellation()
        return processedTokens.count
    }
}

@available(iOS 27.0, macOS 27.0, *)
final class LlamaExecutorRuntime: Sendable {
    private struct State {
        var engine: (any LlamaExecutorEngine)?
        var warmup: Task<Void, Error>?
        var response: Task<Void, Error>?
        var responseID: UUID?
        var counts: [UUID: Task<LlamaContextUsage?, Error>] = [:]
        var stopping: Task<Void, Never>?
        var releaseOnStop = false
    }

    private let state = Mutex(State())
    private let engineFactory: @Sendable () throws -> any LlamaExecutorEngine

    init(engineFactory: @escaping @Sendable () throws -> any LlamaExecutorEngine) {
        self.engineFactory = engineFactory
    }

    func scheduleWarmup(_ messages: [LlamaChatMessage]) {
        _ = warmup(messages)
    }

    func prepare(_ messages: [LlamaChatMessage]) async throws {
        guard let task = warmup(messages) else { throw LlamaExecutorError.busy }
        try await withTaskCancellationHandler {
            try await task.value
            try Task.checkCancellation()
        } onCancel: { task.cancel() }
    }

    func contextUsage(_ messages: [LlamaChatMessage], addingAssistant: Bool) async throws -> LlamaContextUsage? {
        let pending = state.withLock { state -> (UUID, Task<LlamaContextUsage?, Error>)? in
            guard state.stopping == nil, let engine = state.engine else { return nil }
            let id = UUID()
            let task = Task<LlamaContextUsage?, Error> { @concurrent in
                try Task.checkCancellation()
                let usage = try await engine.contextUsage(messages, addingAssistant: addingAssistant)
                try Task.checkCancellation()
                return usage
            }
            state.counts[id] = task
            return (id, task)
        }
        guard let (id, task) = pending else { return nil }
        do {
            let value = try await withTaskCancellationHandler {
                try await task.value
            } onCancel: { task.cancel() }
            finishCount(id)
            return value
        } catch {
            finishCount(id)
            throw error
        }
    }

    private func warmup(_ messages: [LlamaChatMessage]) -> Task<Void, Error>? {
        state.withLock { state in
            guard state.stopping == nil, state.response == nil else { return nil }
            let previous = state.warmup
            let task = Task { @concurrent [self] in
                if let previous { await Self.waitForWarmup(previous) }
                try Task.checkCancellation()
                let engine = try loadEngine()
                do {
                    if !messages.isEmpty { _ = try await engine.prepare(messages, addingAssistant: false) }
                    try Task.checkCancellation()
                } catch {
                    await engine.resetCompletion()
                    throw error
                }
            }
            state.warmup = task
            return task
        }
    }

    func respond(
        request: LanguageModelExecutorGenerationRequest,
        messages: [LlamaChatMessage],
        sampling: LlamaSamplingConfig,
        channel: LanguageModelExecutorGenerationChannel
    ) async throws {
        let task = try state.withLock { state -> Task<Void, Error> in
            guard state.response == nil, state.stopping == nil else { throw LlamaExecutorError.busy }
            let warmup = state.warmup
            let id = UUID()
            state.responseID = id
            let task = Task { @concurrent [self] in
                defer { finishResponse(id) }
                if let warmup { await Self.waitForWarmup(warmup) }
                try Task.checkCancellation()
                let engine = try loadEngine()
                do {
                    let start = ContinuousClock.now
                    let inputTokens = try await engine.prepare(messages, addingAssistant: true)
                    try Task.checkCancellation()
                    await engine.updateSamplingConfig(sampling)
                    var output = LlamaResponseEmitter(requestID: request.id, inputTokens: inputTokens, start: start, channel: channel)
                    while request.generationOptions.maximumResponseTokens.map({ output.tokenCount < $0 }) ?? true {
                        try Task.checkCancellation()
                        let next = try await engine.generateNextToken()
                        try Task.checkCancellation()
                        guard case .token(let text) = next else { break }
                        await output.append(text)
                    }
                    try Task.checkCancellation()
                    await output.finish()
                } catch {
                    await engine.resetCompletion()
                    throw error
                }
            }
            state.response = task
            return task
        }
        try await withTaskCancellationHandler {
            try await task.value
            try Task.checkCancellation()
        } onCancel: { task.cancel() }
    }

    func stop(unload: Bool) async {
        let task = state.withLock { state -> Task<Void, Never> in
            state.releaseOnStop = state.releaseOnStop || unload
            if let stopping = state.stopping { return stopping }
            let response = state.response
            let warmup = state.warmup
            let counts = Array(state.counts.values)
            let task = Task { @concurrent [self] in
                // Never run cancellation handlers while holding the lifecycle lock.
                response?.cancel()
                warmup?.cancel()
                counts.forEach { $0.cancel() }
                _ = await response?.result
                _ = await warmup?.result
                for count in counts { _ = await count.result }
                self.state.withLock { state in
                    if state.releaseOnStop { state.engine = nil }
                    state.warmup = nil
                    state.counts = [:]
                    state.releaseOnStop = false
                    state.stopping = nil
                }
            }
            state.stopping = task
            return task
        }
        await task.value
    }

    private func loadEngine() throws -> any LlamaExecutorEngine {
        if let engine = state.withLock({ $0.engine }) { return engine }
        // Operations are serialized, but model loading must not hold the cancellation lock.
        let engine = try engineFactory()
        state.withLock { $0.engine = engine }
        return engine
    }

    private func finishResponse(_ id: UUID) {
        state.withLock { state in
            guard state.responseID == id else { return }
            state.response = nil
            state.responseID = nil
            state.warmup = nil
        }
    }

    private func finishCount(_ id: UUID) {
        state.withLock { $0.counts[id] = nil }
    }

    private static func waitForWarmup(_ task: Task<Void, Error>) async {
        await withTaskCancellationHandler {
            _ = await task.result
        } onCancel: { task.cancel() }
    }
}
#endif
