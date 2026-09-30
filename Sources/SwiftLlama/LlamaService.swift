//
//  LlamaService.swift
//  PrivateAI
//
//  Created by Piotr Gorzelany on 24/01/2024.
//

import Foundation

public final actor LlamaService {

    // MARK: Properties
    private var llama: Llama?
    private var generationID: UUID?
    private var preparing = false
    private var stopping = false
    private var currentTask: Task<(), Never>?
    private let modelUrl: URL
    private let config: LlamaConfig

    // MARK: Lifecycle

    public init(modelUrl: URL, config: LlamaConfig) {
        self.modelUrl = modelUrl
        self.config = config
    }

    // MARK: Methods

    public func processMessages(_ messages: [LlamaChatMessage]) async throws {
        guard !preparing, !stopping else { throw LlamaError.busy }
        preparing = true
        defer { preparing = false }
        let llama = try initializeLlamaIfNecessary()
        await stopCompletion()
        try await llama.initializeCompletion(messages: messages, addAssistant: false)
    }

    /// Generate a typed response constrained by a JSON grammar inferred from `T` and decode it.
    /// - Parameters:
    ///   - messages: Chat messages forming the prompt.
    ///   - type: The `Codable` type to generate and decode.
    /// - Returns: A decoded instance of `T` produced by the model.
    public func respond<T: Codable>(to messages: [LlamaChatMessage], generating type: T.Type) async throws -> T {
        let stream = try await streamCompletion(of: messages, generating: type)
        var text = ""
        for try await token in stream { text += token }
        return try JSONDecoder().decode(T.self, from: Data(text.utf8))
    }

    /// Generate a plain text response using the provided sampling configuration.
    /// - Parameters:
    ///   - messages: Chat messages forming the prompt.
    ///   - samplingConfig: Sampling parameters controlling generation.
    /// - Returns: The full generated text.
    public func respond(to messages: [LlamaChatMessage], samplingConfig: LlamaSamplingConfig) async throws -> String {
        let stream = try await streamCompletion(of: messages, samplingConfig: samplingConfig)
        var output = ""
        for try await token in stream {
            output += token
        }
        return output
    }

    public func streamCompletion<T: Codable>(of messages: [LlamaChatMessage], generating: T.Type) async throws -> AsyncThrowingStream<String, Error> {
        // Default: constrain the output to valid JSON matching the provided type
        let grammarConfig = try LlamaTypedJSONGrammarBuilder.makeGrammarConfig(for: generating)
        let sampling = LlamaSamplingConfig(
            temperature: 0.1,
            seed: 42,
            grammarConfig: grammarConfig
        )
        return try await streamCompletion(of: messages, samplingConfig: sampling)
    }

    public func streamCompletion(of messages: [LlamaChatMessage], samplingConfig: LlamaSamplingConfig) async throws -> AsyncThrowingStream<String, Error> {
        guard !messages.isEmpty else { throw LlamaError.emptyMessageArray }
        guard !preparing, !stopping else { throw LlamaError.busy }
        preparing = true
        defer { preparing = false }
        let llama = try initializeLlamaIfNecessary()
        await stopCompletion()
        do {
            try await llama.initializeCompletion(messages: messages)
            try await llama.updateSamplingConfig(samplingConfig)
        } catch {
            await llama.resetCompletion()
            throw error
        }
        let id = UUID()
        generationID = id

        return AsyncThrowingStream { continuation in
            let task = Task {
                defer { if generationID == id { currentTask = nil; generationID = nil } }
                do {
                    generationLoop: while await (llama.currentTokenPosition < llama.maxTokenCount) {
                        try Task.checkCancellation()
                        let result = try await llama.generateNextToken()
                        switch result {
                        case .token(let token):
                            continuation.yield(token)
                        case .endOfString:
                            break generationLoop
                        }
                    }
                    let trailing = await llama.finishDecoding()
                    if !trailing.isEmpty { continuation.yield(trailing) }
                    continuation.finish()
                } catch {
                    await llama.resetCompletion()
                    continuation.finish(throwing: error)
                }
            }
            currentTask = task
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }

    public func stopCompletion() async {
        stopping = true
        let task = currentTask
        await task?.cancelAndWait()
        currentTask = nil
        generationID = nil
        stopping = false
    }

    private func initializeLlamaIfNecessary() throws -> Llama {
        guard let llama else {
            llama = try Llama(modelPath: modelUrl.path(percentEncoded: false), config: config)
            return llama!
        }
        return llama
    }
}
