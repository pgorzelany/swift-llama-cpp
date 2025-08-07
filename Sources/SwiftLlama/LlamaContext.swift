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
    var memory: LlamaMemory {
        LlamaMemory(memory: llama_get_memory(contextPointer))
    }
    let contextPointer: OpaquePointer
    private var abortBox: Unmanaged<AbortBox>?

    // MARK: - Lifecycle

    public init?(model: LlamaModel, parameters: llama_context_params = llama_context_default_params()) {
        self.model = model
        guard let contextPointer = llama_init_from_model(model.modelPointer, parameters) else {
            return nil
        }
        self.contextPointer = contextPointer
    }

    deinit {
        if let box = abortBox { box.release() }
        llama_free(contextPointer)
    }

    // MARK: - Methods

    public func contextSize() -> UInt32 {
        llama_n_ctx(contextPointer)
    }

    public func batchSize() -> UInt32 {
        llama_n_batch(contextPointer)
    }

    public func ubatchSize() -> UInt32 {
        llama_n_ubatch(contextPointer)
    }

    public func maxParallelSequences() -> UInt32 {
        UInt32(llama_max_parallel_sequences())
    }

    /// Get the configured maximum number of sequences (n_seq_max).
    public func maxSequences() -> UInt32 {
        llama_n_seq_max(contextPointer)
    }

    /// Get the context's pooling type.
    public func poolingType() -> llama_pooling_type {
        llama_pooling_type(contextPointer)
    }

    public func clearKVCache() {
        memory.clear(data: true)
    }

    public func clearKVCacheFromPosition(_ position: Int32) {
        // Remove KV cache entries from the given position to the end
        // seq_id = -1 means all sequences, p0 = position, p1 = -1 means to the end
        memory.remove(sequenceId: 0, from: position, to: -1)
    }

    public func decode(batch: LlamaBatch) throws {
        let returnCode = llama_decode(contextPointer, batch.rawBatch)
        guard returnCode >= 0 else {
            throw LlamaContextError.decodingError
        }
        synchronize()
    }

    /// Run encoder-only pass for encoder-decoder models or preprocess inputs.
    public func encode(batch: LlamaBatch) throws {
        let rc = llama_encode(contextPointer, batch.rawBatch)
        guard rc >= 0 else { throw LlamaContextError.decodingError }
        synchronize()
    }

    // MARK: - Safe logits & embeddings access

    /// Return the logits for the i-th token from the last decode call.
    /// - Parameter index: Use -1 for the last token.
    /// - Returns: A copy of logits as `[Float]` with size equal to `model.vocabularySize()`, or `nil` if unavailable.
    public func logits(at index: Int32) -> [Float]? {
        guard let ptr = llama_get_logits_ith(contextPointer, index) else { return nil }
        let n = Int(model.vocabularySize())
        var out = [Float](repeating: 0, count: n)
        out.withUnsafeMutableBufferPointer { dst in
            dst.baseAddress!.update(from: ptr, count: n)
        }
        return out
    }

    /// Return the logits for the last token.
    public func lastLogits() -> [Float]? { logits(at: -1) }

    /// Return the embeddings for the i-th token from the last decode/encode call.
    /// - Parameter index: Use -1 for the last embedding.
    /// - Returns: A copy of the embedding vector.
    public func embeddings(at index: Int32) -> [Float]? {
        guard let ptr = llama_get_embeddings_ith(contextPointer, index) else { return nil }
        let n = Int(model.nEmbed())
        var out = [Float](repeating: 0, count: n)
        out.withUnsafeMutableBufferPointer { dst in
            dst.baseAddress!.update(from: ptr, count: n)
        }
        return out
    }

    /// Return the pooled embeddings for a sequence id, if pooling is enabled.
    /// - Returns: A copy of the pooled embeddings: size `n_cls_out` when pooling is RANK, otherwise `n_embd`.
    public func pooledEmbeddings(for sequenceId: llama_seq_id) -> [Float]? {
        guard let ptr = llama_get_embeddings_seq(contextPointer, sequenceId) else { return nil }
        let pooling = llama_pooling_type(contextPointer)
        let n: Int
        if pooling == LLAMA_POOLING_TYPE_RANK {
            n = Int(model.nClassifierOutputs())
        } else {
            n = Int(model.nEmbed())
        }
        var out = [Float](repeating: 0, count: n)
        out.withUnsafeMutableBufferPointer { dst in
            dst.baseAddress!.update(from: ptr, count: n)
        }
        return out
    }

    public func synchronize() {
        llama_synchronize(contextPointer)
    }

    // MARK: - Controls

    public func setThreads(nThreads: Int32, nThreadsBatch: Int32) {
        llama_set_n_threads(contextPointer, nThreads, nThreadsBatch)
    }

    public func nThreads() -> Int32 { llama_n_threads(contextPointer) }
    public func nThreadsBatch() -> Int32 { llama_n_threads_batch(contextPointer) }

    public func setEmbeddingsOutput(_ enabled: Bool) { llama_set_embeddings(contextPointer, enabled) }
    public func setCausalAttention(_ enabled: Bool) { llama_set_causal_attn(contextPointer, enabled) }
    public func setWarmup(_ enabled: Bool) { llama_set_warmup(contextPointer, enabled) }

    // Abort callback bridging
    public func setAbortCallback(_ callback: @escaping () -> Bool) {
        // release previous
        if let box = abortBox { box.release(); abortBox = nil }
        let newBox = Unmanaged.passRetained(AbortBox(cb: callback))
        abortBox = newBox
        llama_set_abort_callback(contextPointer, llamaSwiftAbortCallback, newBox.toOpaque())
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

    // MARK: - State / Sessions

    public func stateSize() -> Int { Int(llama_state_get_size(contextPointer)) }

    public func saveState() -> Data {
        let size = stateSize()
        var buffer = Data(count: size)
        let written = buffer.withUnsafeMutableBytes { raw -> size_t in
            guard let base = raw.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return 0 }
            return llama_state_get_data(contextPointer, base, size_t(size))
        }
        if written == 0 { return Data() }
        if written < buffer.count { buffer.removeSubrange(Int(written)..<buffer.count) }
        return buffer
    }

    @discardableResult
    public func loadState(_ data: Data) -> Bool {
        let read = data.withUnsafeBytes { raw -> size_t in
            guard let base = raw.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return 0 }
            return llama_state_set_data(contextPointer, base, size_t(data.count))
        }
        return read > 0
    }

    public func saveSession(to path: String, tokens: [llama_token]) -> Bool {
        tokens.withUnsafeBufferPointer { ptr in
            llama_state_save_file(contextPointer, path, ptr.baseAddress, size_t(tokens.count))
        }
    }

    public func loadSession(from path: String, capacity: Int) -> (tokens: [llama_token], count: Int)? {
        var tokens = [llama_token](repeating: 0, count: capacity)
        var outCount: size_t = 0
        let ok = tokens.withUnsafeMutableBufferPointer { buf in
            llama_state_load_file(contextPointer, path, buf.baseAddress, size_t(capacity), &outCount)
        }
        if ok { return (Array(tokens.prefix(Int(outCount))), Int(outCount)) }
        return nil
    }

    public func stateForSequenceSize(_ seqId: llama_seq_id) -> Int { Int(llama_state_seq_get_size(contextPointer, seqId)) }

    public func stateForSequence(_ seqId: llama_seq_id) -> Data {
        let size = stateForSequenceSize(seqId)
        var buffer = Data(count: size)
        let written = buffer.withUnsafeMutableBytes { raw -> size_t in
            guard let base = raw.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return 0 }
            return llama_state_seq_get_data(contextPointer, base, size_t(size), seqId)
        }
        if written == 0 { return Data() }
        if written < buffer.count { buffer.removeSubrange(Int(written)..<buffer.count) }
        return buffer
    }

    @discardableResult
    public func loadStateForSequence(_ seqId: llama_seq_id, data: Data, destSeqId: llama_seq_id) -> Bool {
        let read = data.withUnsafeBytes { raw -> size_t in
            guard let base = raw.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return 0 }
            return llama_state_seq_set_data(contextPointer, base, size_t(data.count), destSeqId)
        }
        return read > 0
    }

    // MARK: - Performance
    public func performanceData() -> llama_perf_context_data { llama_perf_context(contextPointer) }
    public func performanceReset() { llama_perf_context_reset(contextPointer) }
    public func performancePrint() { llama_perf_context_print(contextPointer) }

    // MARK: - Per-sequence state files

    /// Save a single sequence state to a file along with token history.
    /// - Returns: Number of bytes written.
    @discardableResult
    public func saveSequenceState(to filepath: String, seqId: llama_seq_id, tokens: [llama_token]) -> Int {
        tokens.withUnsafeBufferPointer { ptr in
            Int(llama_state_seq_save_file(contextPointer, filepath, seqId, ptr.baseAddress, size_t(tokens.count)))
        }
    }

    /// Load a single sequence state from file into the specified destination sequence id.
    /// - Returns: Loaded tokens and count if successful, otherwise nil.
    public func loadSequenceState(from filepath: String, destSeqId: llama_seq_id, capacity: Int) -> (tokens: [llama_token], count: Int)? {
        var tokens = [llama_token](repeating: 0, count: capacity)
        var outCount: size_t = 0
        let ok = tokens.withUnsafeMutableBufferPointer { buf in
            llama_state_seq_load_file(contextPointer, filepath, destSeqId, buf.baseAddress, size_t(capacity), &outCount) > 0
        }
        if ok { return (Array(tokens.prefix(Int(outCount))), Int(outCount)) }
        return nil
    }

    // MARK: - Threadpool

    /// Attach the default ggml auto threadpool to this context.
    public func attachAutoThreadpool() { llama_attach_threadpool(contextPointer, nil, nil) }

    /// Detach any threadpools from this context.
    public func detachThreadpool() { llama_detach_threadpool(contextPointer) }
}

// MARK: - Abort callback bridge
private final class AbortBox {
    let cb: () -> Bool
    init(cb: @escaping () -> Bool) { self.cb = cb }
    func toOpaque() -> UnsafeMutableRawPointer { UnsafeMutableRawPointer(Unmanaged.passUnretained(self).toOpaque()) }
}

@_cdecl("llamaSwiftAbortCallback")
private func llamaSwiftAbortCallback(_ userData: UnsafeMutableRawPointer?) -> Bool {
    guard let userData else { return false }
    let box = Unmanaged<AbortBox>.fromOpaque(userData).takeUnretainedValue()
    return box.cb()
}
