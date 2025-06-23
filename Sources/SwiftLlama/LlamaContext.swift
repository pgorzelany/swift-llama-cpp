//
//  LlamaContext.swift
//  PrivateAI
//
//  Created by Piotr Gorzelany on 12/02/2024.
//

import Foundation
import llama

enum LlamaContextError: Error {
    case decodingError
}

final class LlamaContext {
    // MARK: - Properties

    let model: LlamaModel
    let contextPointer: OpaquePointer

    // MARK: - Lifecycle

    init?(model: LlamaModel, parameters: llama_context_params = llama_context_default_params()) {
        self.model = model
        guard let contextPointer = llama_init_from_model(model.modelPointer, parameters) else {
            return nil
        }
        self.contextPointer = contextPointer
    }

    deinit {
        llama_free(contextPointer)
    }

    // MARK: - Methods

    func contextSize() -> UInt32 {
        llama_n_ctx(contextPointer)
    }

    func batchSize() -> UInt32 {
        llama_n_batch(contextPointer)
    }

    func clearKVCache() {
        llama_kv_self_clear(contextPointer)
    }

    func clearKVCacheFromPosition(_ position: Int32) {
        // Remove KV cache entries from the given position to the end
        // seq_id = -1 means all sequences, p0 = position, p1 = -1 means to the end
        llama_kv_self_seq_rm(contextPointer, -1, position, -1)
    }

    func decode(batch: LlamaBatch) throws {
        let returnCode = llama_decode(contextPointer, batch.rawBatch)
        guard returnCode == 0 else {
            throw LlamaContextError.decodingError
        }
        synchronize()
    }

    func getLogits() -> UnsafeMutablePointer<Float>? {
        llama_get_logits(contextPointer)
    }

    func getLogits(index: Int32) -> UnsafeMutablePointer<Float>? {
        llama_get_logits_ith(contextPointer, index)
    }

    func synchronize() {
        llama_synchronize(contextPointer)
    }
}
