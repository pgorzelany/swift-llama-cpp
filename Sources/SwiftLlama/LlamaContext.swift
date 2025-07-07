//
//  LlamaContext.swift
//  PrivateAI
//
//  Created by Piotr Gorzelany on 12/02/2024.
//

import Foundation
import llama

public enum LlamaContextError: Error {
    case decodingError
    case logitsUnavailable
    case saveStateFailed
    case loadStateFailed
    case loraAdapterFailed(String)
}

public final class LlamaContext {
    // MARK: - Properties

    public let model: LlamaModel
    let contextPointer: OpaquePointer

    // MARK: - Lifecycle

    public init?(model: LlamaModel, parameters: llama_context_params = llama_context_default_params()) {
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

    public func contextSize() -> UInt32 {
        llama_n_ctx(contextPointer)
    }

    public func batchSize() -> UInt32 {
        llama_n_batch(contextPointer)
    }

    public func clearKVCache() {
        llama_kv_self_clear(contextPointer)
    }

    public func clearKVCacheFromPosition(_ position: Int32) {
        // Remove KV cache entries from the given position to the end
        // seq_id = -1 means all sequences, p0 = position, p1 = -1 means to the end
        llama_kv_self_seq_rm(contextPointer, -1, position, -1)
    }

    public func decode(batch: LlamaBatch) throws {
        let returnCode = llama_decode(contextPointer, batch.rawBatch)
        guard returnCode >= 0 else {
            throw LlamaContextError.decodingError
        }
        synchronize()
    }

    public func getLogits() -> UnsafeMutablePointer<Float>? {
        llama_get_logits(contextPointer)
    }

    public func getLogits(index: Int32) -> UnsafeMutablePointer<Float>? {
        llama_get_logits_ith(contextPointer, index)
    }

    public func synchronize() {
        llama_synchronize(contextPointer)
    }

    // MARK: - Adapters

    /// Applies a LoRA adapter to the context.
    /// - Parameters:
    ///   - adapter: The `LlamaLoraAdapter` to apply.
    ///   - scale: The scaling factor for the adapter's influence.
    /// - Throws: `LlamaContextError.loraAdapterFailed` if the operation fails.
    public func apply(loraAdapter: LlamaLoraAdapter, scale: Float = 1.0) throws {
        let result = llama_set_adapter_lora(contextPointer, loraAdapter.adapterPointer, scale)
        if result != 0 {
            throw LlamaContextError.loraAdapterFailed("Failed to apply LoRA adapter.")
        }
    }

    /// Removes a specific LoRA adapter from the context.
    /// - Parameter adapter: The `LlamaLoraAdapter` to remove.
    /// - Throws: `LlamaContextError.loraAdapterFailed` if the adapter is not found or cannot be removed.
    public func remove(loraAdapter: LlamaLoraAdapter) throws {
        let result = llama_rm_adapter_lora(contextPointer, loraAdapter.adapterPointer)
        if result == -1 {
            throw LlamaContextError.loraAdapterFailed("LoRA adapter not found in context.")
        }
    }

    /// Removes all LoRA adapters from the context.
    public func removeAllLoraAdapters() {
        llama_clear_adapter_lora(contextPointer)
    }

    /// Applies a control vector to the context.
    ///
    /// This can be used to apply adjustments to the model's behavior.
    ///
    /// - Parameters:
    ///   - data: A buffer of floats representing the control vector data. Should be `n_embd * n_layers`.
    ///   - n_embd: The size of a single layer's control vector.
    ///   - startLayer: The starting layer index for applying the vector (inclusive).
    ///   - endLayer: The ending layer index for applying the vector (inclusive).
    /// - Throws: `LlamaContextError.loraAdapterFailed` if applying the control vector fails.
    public func apply(controlVector data: [Float], n_embd: Int32, startLayer: Int32, endLayer: Int32) throws {
        let result = data.withUnsafeBufferPointer { bufferPointer in
            llama_apply_adapter_cvec(
                contextPointer,
                bufferPointer.baseAddress,
                size_t(bufferPointer.count),
                n_embd,
                startLayer,
                endLayer
            )
        }
        if result != 0 {
            throw LlamaContextError.loraAdapterFailed("Failed to apply control vector.")
        }
    }

    /// Clears the currently applied control vector.
    /// - Throws: `LlamaContextError.loraAdapterFailed` if clearing fails.
    public func clearControlVector() throws {
        let result = llama_apply_adapter_cvec(contextPointer, nil, 0, 0, 0, 0)
        if result != 0 {
            throw LlamaContextError.loraAdapterFailed("Failed to clear control vector.")
        }
    }
}
