import Foundation
import SwiftLlama
import llama

public enum LlamaError: Error{
    case couldNotInitializeContext
    case contextSizeLimitExeeded
    case decodingError
}

enum NextToken {
    case token(String)
    case endOfString
}

final actor Llama {
    private let model: LlamaModel
    private let context: LlamaContext
    private var batch: LlamaBatch
    private var sampler: LlamaSampler!

    // Configuration

    private let config: LlamaConfig
    let maxTokenCount: UInt32
    /// Tracks the current position in the token sequence during decoding.
    var currentTokenPosition: Int32 = 0
    var processedTokens: [llama_token] = []

    init(modelPath: String, config: LlamaConfig) throws {
        self.config = config
        llama_backend_init()
        var model_params = llama_model_default_params()

        if !config.useGPU {
            model_params.n_gpu_layers = 0
        }

        #if targetEnvironment(simulator)
                model_params.n_gpu_layers = 0
                print("Running on simulator, force use n_gpu_layers = 0")
        #endif

        let model = LlamaModel(path: modelPath, parameters: model_params)
        guard let model else {
            print("Could not load model at \(modelPath)")
            throw LlamaError.couldNotInitializeContext
        }

        let n_threads = ProcessInfo.processInfo.processorCount - 1
        print("Using \(n_threads) threads")

        var contextParam = llama_context_default_params()
        contextParam.n_ctx = config.maxTokenCount
        contextParam.n_threads       = 1 // UInt32(n_threads) its actually faster if less threads are doing work
        contextParam.n_threads_batch = 1 // UInt32(n_threads)
        contextParam.n_batch = config.batchSize
        contextParam.n_ubatch = config.batchSize
        contextParam.offload_kqv = true

        let context = LlamaContext(model: model, parameters: contextParam)
        guard let context else {
            print("Could not load context!")
            throw LlamaError.couldNotInitializeContext
        }


        self.maxTokenCount = min(UInt32(model.trainedContextSize()), config.maxTokenCount)
        self.model = context.model
        self.context = context
        self.batch = .init(initialSize: Int32(config.batchSize))
    }

    deinit {
        llama_backend_free()
    }

    func initializeCompletion(messages: [LlamaChatMessage], addAssistant: Bool? = nil) throws {
        let formattedPrompt = model.applyChatTemplate(to: messages, addAssistant: addAssistant)
        try initializeCompletion(text: formattedPrompt)
    }

    private func initializeCompletion(text: String) throws {
        print("attempting to complete \"\(text)\"")

        let tokenList = model.tokenize(text: text, addBos: model.shouldAddBos(), special: true)
        guard tokenList.count < maxTokenCount - 4 else {
            throw LlamaError.contextSizeLimitExeeded
        }

        if tokenList.starts(with: processedTokens) {
            print("### Using cached processing")
            try processPrompt(tokens: Array(tokenList[processedTokens.count...]), startIndex: processedTokens.count)
        } else {
            // Check if we can optimize by only clearing from the divergence point
            let divergenceIndex = findDivergenceIndex(newTokenList: tokenList, processedTokens: processedTokens)
            
            if divergenceIndex > 0 && shouldUsePartialOptimization(divergenceIndex: divergenceIndex, totalProcessed: processedTokens.count) {
                print("### Using partial optimization from position \(divergenceIndex)")
                try optimizedReprocessing(newTokenList: tokenList, divergenceIndex: divergenceIndex)
            } else {
                print("### Full reprocessing required")
                clear()
                try processPrompt(tokens: tokenList, startIndex: 0)
            }
        }
    }

    /// Find the index where the two token lists diverge
    private func findDivergenceIndex(newTokenList: [llama_token], processedTokens: [llama_token]) -> Int {
        let minLength = min(newTokenList.count, processedTokens.count)
        for i in 0..<minLength {
            if newTokenList[i] != processedTokens[i] {
                return i
            }
        }
        return minLength
    }
    
    /// Decide whether to use partial optimization based on the divergence point
    private func shouldUsePartialOptimization(divergenceIndex: Int, totalProcessed: Int) -> Bool {
        // Only use partial optimization if:
        // 1. We have a significant amount of processed tokens (at least 10)
        // 2. The divergence is not too early (at least 50% of tokens match)
        // 3. The divergence is not at the very beginning
        
        guard divergenceIndex > 0 && totalProcessed >= 10 else { return false }
        
        let matchPercentage = Double(divergenceIndex) / Double(totalProcessed)
        return matchPercentage >= 0.5 // At least 50% of tokens match
    }
    
    /// Optimized reprocessing that only clears cache from the divergence point
    private func optimizedReprocessing(newTokenList: [llama_token], divergenceIndex: Int) throws {
        // Clear KV cache from the divergence point onward
        context.clearKVCacheFromPosition(Int32(divergenceIndex))
        
        // Update our internal state
        processedTokens = Array(processedTokens[0..<divergenceIndex])
        currentTokenPosition = Int32(divergenceIndex)
        
        // Process only the tokens from the divergence point onward
        let tokensToProcess = Array(newTokenList[divergenceIndex...])
        try processPrompt(tokens: tokensToProcess, startIndex: divergenceIndex)
    }

    func generateNextToken() throws -> NextToken {
        let newTokenId = sampler.sample(context: context, lastTokenIndex: batch.size - 1)
        sampler.accept(token: newTokenId)

        if model.isEogToken(newTokenId) || currentTokenPosition > maxTokenCount {
            return .endOfString
        }

        batch.reset()
        batch.addToken(newTokenId, at: currentTokenPosition, logits: true)
        processedTokens.append(newTokenId)

        currentTokenPosition += 1
        try context.decode(batch: batch)

        return .token(model.piece(from: newTokenId))
    }

    func updateSamplingConfig(_ config: LlamaSamplingConfig) {
        self.sampler = .init(config: config, model: model)
    }

    private func clear() {
        context.clearKVCache()
        processedTokens = []
        batch = .init(initialSize: Int32(config.batchSize))
    }

    private func processBatch() throws {
        do {
            try context.decode(batch: batch)
        } catch {
            print("llama_decode() failed")
            throw LlamaError.decodingError
        }
    }

    private func processPrompt(tokens: [llama_token], startIndex: Int) throws {
        guard !tokens.isEmpty else { return }
        batch.reset()

        for i in 0..<tokens.count {
            let tokenPosition = startIndex + i
            let tokenId = tokens[i]
            batch.addToken(tokenId, at: Int32(tokenPosition), logits: false)
            processedTokens.append(tokenId)
            if batch.size == config.batchSize {
                try processBatch()
                batch.reset()
            }
        }

        batch.setLastTokenLogits(true)
        try processBatch()

        currentTokenPosition = Int32(processedTokens.count)
    }
}
