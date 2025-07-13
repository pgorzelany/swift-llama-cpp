//
//  LlamaService.swift
//  PrivateAI
//
//  Created by Piotr Gorzelany on 24/01/2024.
//

import Foundation
import SwiftLlama

public final actor LlamaService {

    // MARK: Properties
    private var llama: Llama?
    private var currentTask: Task<(), Error>?
    private let modelUrl: URL
    private let config: LlamaConfig
    private let tokenBufferSize = 2

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

    public func streamCompletion(of messages: [LlamaChatMessage], samplingConfig: LlamaSamplingConfig) async throws -> AsyncThrowingStream<String, Error> {
        let llama = try initializeLlamaIfNecessary()
        await stopCompletion()
        try await  llama.initializeCompletion(messages: messages)
        await llama.updateSamplingConfig(samplingConfig)

        return AsyncThrowingStream { continuation in
            currentTask = Task {
                do {
                    var tokenBuffer: [String] = []
                    generationLoop: while await (llama.currentTokenPosition < llama.maxTokenCount) {
                        guard !Task.isCancelled else { break }
                        let result = try await llama.generateNextToken()
                        guard !Task.isCancelled else { break }
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
