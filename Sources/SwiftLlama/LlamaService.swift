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
    private var currentTask: Task<(), Error>?
    private let modelUrl: URL
    private let config: LlamaConfig
    private let tokenBufferSize = 1

    // MARK: Lifecycle

    public init(modelUrl: URL, config: LlamaConfig) {
        self.modelUrl = modelUrl
        self.config = config
    }

    // MARK: Methods

    public func processMessages(_ messages: [LlamaChatMessage]) async throws {
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
        func extractLikelyJSON(from text: String) -> String? {
            // Find first opening brace or bracket
            guard let startIndex = text.firstIndex(where: { $0 == "{" || $0 == "[" }) else { return nil }
            let candidate = text[startIndex...]
            // Simple balance-based termination (ignores strings/escapes, good enough for LLM output)
            var depth: Int = 0
            var closingIndex: String.Index?
            for (i, ch) in candidate.enumerated() {
                let idx = candidate.index(candidate.startIndex, offsetBy: i)
                if ch == "{" || ch == "[" { depth += 1 }
                else if ch == "}" || ch == "]" {
                    depth -= 1
                    if depth == 0 { closingIndex = idx; break }
                }
            }
            if let closingIndex {
                return String(candidate[...closingIndex])
            }
            return nil
        }

        var accumulated = ""
        let decoder = JSONDecoder()
        var decodedValue: T?
        let stream = try await streamCompletion(of: messages, generating: type)
        do {
            for try await token in stream {
                accumulated += token
                if let jsonText = extractLikelyJSON(from: accumulated),
                   let data = jsonText.data(using: .utf8),
                   let value = try? decoder.decode(T.self, from: data) {
                    decodedValue = value
                    break
                }
            }
        } catch {
            // Fall through to final decode attempt below
        }
        if let value = decodedValue {
            await stopCompletion()
            return value
        }
        // Final attempt with trimmed JSON if available, otherwise full text
        let finalText = extractLikelyJSON(from: accumulated) ?? accumulated
        guard let finalData = finalText.data(using: .utf8) else {
            throw LlamaError.decodingError
        }
        return try decoder.decode(T.self, from: finalData)
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
        let llama = try initializeLlamaIfNecessary()
        await stopCompletion()
        try await  llama.initializeCompletion(messages: messages)
        await llama.updateSamplingConfig(samplingConfig)

        return AsyncThrowingStream { continuation in
            currentTask = Task {
                do {
                    var tokenBuffer: [String] = []
                    generationLoop: while await (llama.currentTokenPosition < llama.maxTokenCount) {
                        guard !Task.isCancelled else {
                            if !tokenBuffer.isEmpty {
                                continuation.yield(tokenBuffer.joined())
                                tokenBuffer = []
                            }
                            break
                        }
                        let result = try await llama.generateNextToken()
                        switch result {
                        case .token(let token):
                            tokenBuffer.append(token)
                            if tokenBuffer.count == tokenBufferSize {
                                continuation.yield(tokenBuffer.joined())
                                tokenBuffer = []
                            }
                        case .endOfString:
                            continuation.yield(tokenBuffer.joined())
                            break generationLoop
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    public func stopCompletion() async {
        await currentTask?.cancelAndWait()
    }

    private func initializeLlamaIfNecessary() throws -> Llama {
        guard let llama else {
            llama = try Llama(modelPath: modelUrl.path(percentEncoded: false), config: config)
            return llama!
        }
        return llama
    }
}
