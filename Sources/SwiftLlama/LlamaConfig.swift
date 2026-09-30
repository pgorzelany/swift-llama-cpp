import Foundation
import Darwin

public struct LlamaConfig: Equatable, Sendable {
    public let batchSize: UInt32
    public let maxTokenCount: UInt32
    public let useGPU: Bool
    /// Physical decode batch, independently of the logical prompt batch.
    public let microBatchSize: UInt32?
    /// Nil selects one thread for GPU and the performance core count for CPU.
    public let nThreads: Int32?
    public let nThreadsBatch: Int32?

    public init(
        batchSize: UInt32,
        maxTokenCount: UInt32,
        useGPU: Bool = true,
        microBatchSize: UInt32? = nil,
        nThreads: Int32? = nil,
        nThreadsBatch: Int32? = nil
    ) {
        self.batchSize = batchSize
        self.maxTokenCount = maxTokenCount
        self.useGPU = useGPU
        self.microBatchSize = microBatchSize
        self.nThreads = nThreads
        self.nThreadsBatch = nThreadsBatch
    }

    static var cpuThreadCount: Int32 {
        var count: Int32 = 0
        var size = MemoryLayout<Int32>.size
        if sysctlbyname("hw.perflevel0.physicalcpu", &count, &size, nil, 0) == 0, count > 0 {
            return count
        }
        return Int32(max(1, min(4, ProcessInfo.processInfo.activeProcessorCount)))
    }
}
