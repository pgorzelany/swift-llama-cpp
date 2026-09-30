import llama

public final class LlamaBatch {
    private(set) var rawBatch: llama_batch
    public let capacity: Int32
    private let embeddingSize: Int32
    public var size: Int32 { rawBatch.n_tokens }

    /// Allocates owned token storage. Invalid capacities create an empty, non-writable batch.
    public init(initialSize: Int32) {
        capacity = max(0, initialSize)
        embeddingSize = 0
        rawBatch = llama_batch_init(max(1, capacity), 0, 1)
    }

    /// Allocates owned embedding storage for vectors of exactly embeddingSize floats.
    public init(embeddingCapacity: Int32, embeddingSize: Int32, maxSequences: Int32 = 1) {
        capacity = embeddingSize > 0 && maxSequences > 0 ? max(0, embeddingCapacity) : 0
        self.embeddingSize = max(1, embeddingSize)
        rawBatch = llama_batch_init(max(1, capacity), self.embeddingSize, max(1, maxSequences))
    }

    deinit { llama_batch_free(rawBatch) }
    public func reset() { rawBatch.n_tokens = 0 }

    /// Returns false for an embedding batch or a full batch without writing any memory.
    @discardableResult
    public func addToken(_ tokenId: llama_token, at position: llama_pos, logits: Bool) -> Bool {
        guard embeddingSize == 0, size < capacity, tokenId >= 0, position >= 0 else { return false }
        rawBatch.token[Int(size)] = tokenId
        appendPosition(position, logits: logits)
        return true
    }

    @discardableResult
    public func setLastTokenLogits(_ logits: Bool) -> Bool {
        guard size > 0 else { return false }
        rawBatch.logits[Int(size - 1)] = logits ? 1 : 0
        return true
    }

    public static func singleSequence(tokens: [llama_token]) -> LlamaBatch {
        precondition(tokens.count <= Int(Int32.max))
        let batch = LlamaBatch(initialSize: Int32(tokens.count))
        for (position, token) in tokens.enumerated() {
            batch.addToken(token, at: Int32(position), logits: position == tokens.count - 1)
        }
        return batch
    }

    /// Appends an embedding with position and output flags; invalid dimensions leave the batch unchanged.
    @discardableResult
    public func addEmbedding(_ vector: [Float], at position: llama_pos, logits: Bool = false) -> Bool {
        guard embeddingSize > 0, vector.count == Int(embeddingSize), size < capacity, position >= 0 else { return false }
        vector.withUnsafeBufferPointer { source in
            rawBatch.embd.advanced(by: Int(size) * Int(embeddingSize)).update(from: source.baseAddress!, count: source.count)
        }
        appendPosition(position, logits: logits)
        return true
    }

    /// Appends an embedding at the next sequential position.
    @discardableResult
    public func setEmbedding(_ vector: [Float]) -> Bool { addEmbedding(vector, at: size) }

    private func appendPosition(_ position: llama_pos, logits: Bool) {
        let index = Int(size)
        rawBatch.pos[index] = position
        rawBatch.n_seq_id[index] = 1
        rawBatch.seq_id[index]![0] = 0
        rawBatch.logits[index] = logits ? 1 : 0
        rawBatch.n_tokens += 1
    }
}
