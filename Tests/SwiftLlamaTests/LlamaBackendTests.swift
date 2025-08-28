import Testing
@testable import SwiftLlama

struct LlamaBackendTests {
    @Test("System info not empty and supports flags accessible")
    func testSystemInfoAndFlags() {
        let info = LlamaBackend.systemInfo()
        #expect(!info.isEmpty)
        _ = LlamaBackend.supportsMmap
        _ = LlamaBackend.supportsMlock
        _ = LlamaBackend.supportsGpuOffload
        _ = LlamaBackend.maxDevices
        _ = LlamaBackend.maxParallelSequences
    }
}


