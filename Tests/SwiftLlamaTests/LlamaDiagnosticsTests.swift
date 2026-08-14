import Foundation
import Testing
@testable import SwiftLlama

@Suite("Llama diagnostics", .serialized)
struct LlamaDiagnosticsTests {
    init() {
        LlamaLog.setLogger(nil)
        LlamaLog.resetForTesting()
    }

    @Test("Decode errors preserve the native status and backend diagnostics")
    func decodeErrorDescription() throws {
        let marker = LlamaLog.marker()
        LlamaLog.recordForTesting(
            level: 4,
            message: "ggml_metal: failed to allocate compute buffer"
        )

        let error = LlamaContextError.decodeFailed(
            code: -2,
            diagnostics: LlamaLog.diagnostics(since: marker)
        )
        let description = try #require(error.errorDescription)

        #expect(description.contains("status -2"))
        #expect(description.contains("fatal backend error"))
        #expect(description.contains("failed to allocate compute buffer"))
    }

    @Test("KV-cache warnings provide an actionable recovery message")
    func kvCacheErrorDescription() throws {
        let error = LlamaContextError.decodeFailed(code: 1, diagnostics: nil)
        let description = try #require(error.errorDescription)

        #expect(description.contains("KV-cache slot"))
        #expect(description.contains("Reduce the batch size"))
    }

    @Test("Diagnostics exclude messages emitted before the operation marker")
    func operationMarker() throws {
        LlamaLog.recordForTesting(level: 4, message: "stale model warning")
        let marker = LlamaLog.marker()
        LlamaLog.recordForTesting(level: 4, message: "current decode failure")

        let diagnostics = try #require(LlamaLog.diagnostics(since: marker))

        #expect(!diagnostics.contains("stale model warning"))
        #expect(diagnostics.contains("current decode failure"))
    }

    @Test("Continuation messages retain the preceding error severity")
    func continuationSeverity() throws {
        let marker = LlamaLog.marker()
        LlamaLog.recordForTesting(level: 4, message: "ggml_metal: allocation failed")
        LlamaLog.recordForTesting(level: 5, message: "requested buffer size: 1.2 GiB")
        LlamaLog.recordForTesting(level: 2, message: "unrelated informational message")

        let diagnostics = try #require(LlamaLog.diagnostics(since: marker))

        #expect(diagnostics.contains("allocation failed"))
        #expect(diagnostics.contains("requested buffer size"))
        #expect(!diagnostics.contains("unrelated informational message"))
    }

    @Test("Initialization errors include captured llama.cpp diagnostics")
    func initializationErrorDescription() throws {
        let error = LlamaError.contextInitializationFailed(
            diagnostics: "ggml_backend_metal_buffer_type_alloc_buffer: allocation failed"
        )
        let description = try #require(error.errorDescription)

        #expect(description.contains("could not initialize the model context"))
        #expect(description.contains("allocation failed"))
    }
}
