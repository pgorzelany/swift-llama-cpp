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

    public init(initialSize: Int32) {
        self.rawBatch = llama_batch_init(initialSize, 0, 1)
    }

    deinit {
        llama_batch_free(rawBatch)
    }

    public func reset() {
        rawBatch.n_tokens = 0
    }

    public func addToken(_ tokenId: llama_token, at position: llama_pos, logits: Bool) {
        rawBatch.token[Int(rawBatch.n_tokens)] = tokenId
        rawBatch.pos[Int(rawBatch.n_tokens)] = position

        // this is assuming we are only processing one sequence at a time
        rawBatch.n_seq_id[Int(rawBatch.n_tokens)] = 1
        rawBatch.seq_id[Int(rawBatch.n_tokens)]![0] = 0

        rawBatch.logits[Int(rawBatch.n_tokens)] = logits ? 1 : 0

        rawBatch.n_tokens += 1
    }

    public func setLastTokenLogits(_ logits: Bool) {
        rawBatch.logits[Int(rawBatch.n_tokens - 1)] = logits ? 1 : 0
    }
}
