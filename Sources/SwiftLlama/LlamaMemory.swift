import Foundation
import llama

public typealias LlamaSequenceId = llama_seq_id
public typealias LlamaPosition = llama_pos

public final class LlamaMemory {
    let memory: llama_memory_t

    init(memory: llama_memory_t) {
        self.memory = memory
    }

    // MARK: - Public Methods

    /// Clears the memory contents.
    ///
    /// If `data` is `true`, the data buffers will also be cleared along with the metadata.
    /// - Parameter data: A boolean indicating whether to clear the data buffers as well.
    public func clear(data: Bool) {
        llama_memory_clear(memory, data)
    }

    /// Removes all tokens that belong to the specified sequence and have positions in the range `[p0, p1)`.
    ///
    /// - Parameters:
    ///   - sequenceId: The sequence ID. If less than 0, it matches any sequence.
    ///   - p0: The starting position (inclusive). If less than 0, it starts from position 0.
    ///   - p1: The ending position (exclusive). If less than 0, it goes to the end of the sequence.
    /// - Returns: `false` if a partial sequence cannot be removed. Removing a whole sequence never fails.
    @discardableResult
    public func remove(sequenceId: LlamaSequenceId, from p0: LlamaPosition, to p1: LlamaPosition) -> Bool {
        llama_memory_seq_rm(memory, sequenceId, p0, p1)
    }

    /// Copies all tokens that belong to a source sequence to a destination sequence within a specified position range.
    ///
    /// - Parameters:
    ///   - sourceSequenceId: The ID of the source sequence.
    ///   - destinationSequenceId: The ID of the destination sequence.
    ///   - p0: The starting position (inclusive). If less than 0, it starts from position 0.
    ///   - p1: The ending position (exclusive). If less than 0, it goes to the end of the sequence.
    public func copy(
        sourceSequenceId: LlamaSequenceId,
        destinationSequenceId: LlamaSequenceId,
        from p0: LlamaPosition,
        to p1: LlamaPosition
    ) {
        llama_memory_seq_cp(memory, sourceSequenceId, destinationSequenceId, p0, p1)
    }

    /// Removes all tokens that do not belong to the specified sequence.
    ///
    /// - Parameter sequenceId: The ID of the sequence to keep.
    public func keep(sequenceId: LlamaSequenceId) {
        llama_memory_seq_keep(memory, sequenceId)
    }

    /// Adds a relative position "delta" to all tokens that belong to the specified sequence and have positions in the range `[p0, p1)`.
    ///
    /// - Parameters:
    ///   - sequenceId: The sequence ID.
    ///   - p0: The starting position (inclusive). If less than 0, it starts from position 0.
    ///   - p1: The ending position (exclusive). If less than 0, it goes to the end of the sequence.
    ///   - delta: The relative position to add.
    public func add(
        to sequenceId: LlamaSequenceId,
        from p0: LlamaPosition,
        to p1: LlamaPosition,
        delta: LlamaPosition
    ) {
        llama_memory_seq_add(memory, sequenceId, p0, p1, delta)
    }

    /// Performs integer division on the positions of tokens in a specified sequence by a factor `d > 1`.
    ///
    /// - Parameters:
    ///   - sequenceId: The sequence ID.
    ///   - p0: The starting position (inclusive). If less than 0, it starts from position 0.
    ///   - p1: The ending position (exclusive). If less than 0, it goes to the end of the sequence.
    ///   - d: The integer factor to divide positions by. Must be greater than 1.
    public func divide(
        sequenceId: LlamaSequenceId,
        from p0: LlamaPosition,
        to p1: LlamaPosition,
        by d: Int32
    ) {
        llama_memory_seq_div(memory, sequenceId, p0, p1, d)
    }

    /// Returns the smallest position present in the memory for the specified sequence.
    ///
    /// This is typically non-zero only for SWA caches. All positions in the range `[pos_min, pos_max]` are guaranteed
    /// to be present in the memory.
    ///
    /// - Parameter sequenceId: The sequence ID.
    /// - Returns: The minimum position, or -1 if the sequence is empty.
    public func minPosition(for sequenceId: LlamaSequenceId) -> LlamaPosition {
        llama_memory_seq_pos_min(memory, sequenceId)
    }

    /// Returns the largest position present in the memory for the specified sequence.
    ///
    /// All positions in the range `[pos_min, pos_max]` are guaranteed to be present in the memory.
    ///
    /// - Parameter sequenceId: The sequence ID.
    /// - Returns: The maximum position, or -1 if the sequence is empty.
    public func maxPosition(for sequenceId: LlamaSequenceId) -> LlamaPosition {
        llama_memory_seq_pos_max(memory, sequenceId)
    }

    /// A boolean value indicating whether the memory supports shifting.
    public var canShift: Bool {
        llama_memory_can_shift(memory)
    }
} 
