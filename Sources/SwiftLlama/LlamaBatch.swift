//
//  LlamaBatch.swift
//  LlamaSwift
//
//  Created by Piotr Gorzelany on 11/10/2024.
//

import llama

public final class LlamaBatch {
    private(set) var rawBatch: llama_batch
    public var size: Int32 {
        rawBatch.n_tokens
    }

    /// Allocate a batch that can hold up to `initialSize` tokens for a single sequence.
    public init(initialSize: Int32) {
        self.rawBatch = llama_batch_init(initialSize, 0, 1)
    }

    /// Allocate a batch that will carry external embeddings instead of token ids.
    /// - Parameters:
    ///   - capacity: Max number of embedding slots
    ///   - embeddingSize: Size of each embedding vector (n_embd)
    ///   - maxSequences: Max sequences per token (default 1)
    public init(embeddingCapacity capacity: Int32, embeddingSize: Int32, maxSequences: Int32 = 1) {
        self.rawBatch = llama_batch_init(capacity, embeddingSize, maxSequences)
    }

    deinit {
        llama_batch_free(rawBatch)
    }

    /// Reset batch size to 0 without deallocating memory.
    public func reset() {
        rawBatch.n_tokens = 0
    }

    /// Add a token to the batch for a single sequence (seq_id 0).
    public func addToken(_ tokenId: llama_token, at position: llama_pos, logits: Bool) {
        rawBatch.token[Int(rawBatch.n_tokens)] = tokenId
        rawBatch.pos[Int(rawBatch.n_tokens)] = position

        // this is assuming we are only processing one sequence at a time
        rawBatch.n_seq_id[Int(rawBatch.n_tokens)] = 1
        rawBatch.seq_id[Int(rawBatch.n_tokens)]![0] = 0

        rawBatch.logits[Int(rawBatch.n_tokens)] = logits ? 1 : 0

        rawBatch.n_tokens += 1
    }

    /// Mark whether the last token should output logits.
    public func setLastTokenLogits(_ logits: Bool) {
        rawBatch.logits[Int(rawBatch.n_tokens - 1)] = logits ? 1 : 0
    }

     /// Convenience: build a single-sequence batch for a set of tokens where only the last token emits logits.
     public static func singleSequence(tokens: [llama_token]) -> LlamaBatch {
         var cTokens = tokens
         let batch = llama_batch_get_one(&cTokens, Int32(tokens.count))
         // Wrap into class to ensure free on deinit
         let wrapper = LlamaBatch(initialSize: Int32(tokens.count))
         wrapper.rawBatch = batch
         return wrapper
     }

    /// Set an external embedding vector for the current token index.
    /// Note: Caller must ensure the `LlamaBatch` was initialized with embeddings.
    public func setEmbedding(_ vector: [Float]) {
        vector.withUnsafeBufferPointer { src in
            let idx = Int(rawBatch.n_tokens)
            let n = vector.count
            rawBatch.embd.advanced(by: idx * n).update(from: src.baseAddress!, count: n)
        }
    }
}
