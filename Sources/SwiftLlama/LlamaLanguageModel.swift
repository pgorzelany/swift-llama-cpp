#if compiler(>=6.4) && canImport(FoundationModels)
import Foundation
import FoundationModels

/// A local GGUF model for Apple's sessions. Create a separate instance for each conversation owner.
@available(iOS 27.0, macOS 27.0, *)
public struct LlamaLanguageModel: LanguageModel {
    public typealias Executor = LlamaLanguageModelExecutor

    public let capabilities = LanguageModelCapabilities([])
    public let executorConfiguration: Executor.Configuration

    public init(modelURL: URL, configuration: LlamaConfig) {
        executorConfiguration = .init(modelURL: modelURL, configuration: configuration)
    }

    init(engineFactory: @escaping @Sendable () throws -> any LlamaExecutorEngine) {
        executorConfiguration = .init(engineFactory: engineFactory)
    }

    /// Loads the model and processes history, waiting until preparation has actually finished.
    public func prewarm(transcript: Transcript) async throws {
        let messages = try transcript.isEmpty ? [] : LlamaTranscriptMapper.messages(transcript)
        do {
            try await executorConfiguration.runtime.prepare(messages)
        } catch LlamaError.contextSizeLimitExeeded {
            throw executorConfiguration.contextSizeError
        }
    }

    /// Cancels and awaits inference and warmup before the model can be reused.
    public func cancelAndWait() async {
        await executorConfiguration.runtime.stop(unload: false)
    }

    /// Stops outstanding work before releasing model weights and context.
    public func unload() async {
        await executorConfiguration.runtime.stop(unload: true)
    }

    /// Metadata supplements Apple's transcript with lossless tagged replay and measurement availability.
    public enum Metadata {
        public static let requestID = "llama.requestID"
        public static let rawOutput = "llama.rawOutput"
        public static let isReasoning = "llama.isReasoning"
        public static let finished = "llama.finished"
        public static let usageReported = "llama.usageReported"
        public static let inputTokens = "llama.inputTokens"
        public static let inputTokensKnown = "llama.inputTokensKnown"
        public static let outputTokens = "llama.outputTokens"
        public static let reasoningTokensKnown = "llama.reasoningTokensKnown"
        public static let timeToFirstToken = "llama.timeToFirstToken"
        public static let tokensPerSecond = "llama.tokensPerSecond"
    }
}

/// Runs GGUF inference directly against SwiftLlama's core, without LlamaService or an intermediate stream.
@available(iOS 27.0, macOS 27.0, *)
public struct LlamaLanguageModelExecutor: LanguageModelExecutor {
    public typealias Model = LlamaLanguageModel

    public struct Configuration: Hashable, Sendable {
        private let id = UUID()
        let runtime: LlamaExecutorRuntime
        let contextWindowTokens: Int

        public init(modelURL: URL, configuration: LlamaConfig) {
            contextWindowTokens = Int(configuration.maxTokenCount)
            runtime = LlamaExecutorRuntime {
                guard configuration.batchSize > 0,
                      configuration.batchSize <= UInt32(Int32.max),
                      configuration.maxTokenCount >= 8,
                      configuration.maxTokenCount <= UInt32(Int32.max) else {
                    throw LlamaExecutorError.invalidConfiguration
                }
                return try Llama(modelPath: modelURL.path(percentEncoded: false), config: configuration)
            }
        }

        init(engineFactory: @escaping @Sendable () throws -> any LlamaExecutorEngine) {
            contextWindowTokens = 0
            runtime = LlamaExecutorRuntime(engineFactory: engineFactory)
        }

        public static func == (lhs: Self, rhs: Self) -> Bool { lhs.id == rhs.id }
        public func hash(into hasher: inout Hasher) { hasher.combine(id) }

        var contextSizeError: LanguageModelError {
            .contextSizeExceeded(.init(
                contextSize: contextWindowTokens, tokenCount: 0,
                debugDescription: "The conversation exceeds the GGUF model's configured context window."
            ))
        }
    }

    private let runtime: LlamaExecutorRuntime

    public init(configuration: Configuration) throws {
        runtime = configuration.runtime
    }

    public func prewarm(model: Model, transcript: Transcript) {
        guard let messages = try? (transcript.isEmpty ? [] : LlamaTranscriptMapper.messages(transcript)) else { return }
        runtime.scheduleWarmup(messages)
    }

    public func respond(
        to request: LanguageModelExecutorGenerationRequest,
        model: Model,
        streamingInto channel: LanguageModelExecutorGenerationChannel
    ) async throws {
        let messages = try LlamaTranscriptMapper.messages(request.transcript)
        let sampling = try LlamaTranscriptMapper.sampling(request)
        do {
            try await runtime.respond(request: request, messages: messages, sampling: sampling, channel: channel)
        } catch LlamaError.contextSizeLimitExeeded {
            throw model.executorConfiguration.contextSizeError
        }
    }
}
#endif
