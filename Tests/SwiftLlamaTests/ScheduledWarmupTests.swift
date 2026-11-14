#if compiler(>=6.4) && canImport(FoundationModels)
import Foundation
import FoundationModels
import Synchronization
import Testing
@testable import SwiftLlama

@Suite(.timeLimit(.minutes(1)))
struct ScheduledWarmupTests {
    @available(iOS 27.0, macOS 27.0, *)
    @Test("Waiting joins the scheduled native preparation without scheduling another pass")
    func joinsExistingPreparation() async throws {
        let engine = WarmupWaitTestEngine()
        let runtime = LlamaExecutorRuntime { engine }
        runtime.scheduleWarmup([.init(role: .user, content: "Warmup")])
        await engine.started.wait()
        let completed = Mutex(false)
        let waiting = Task {
            try await runtime.waitForScheduledWarmup()
            completed.withLock { $0 = true }
        }
        try await Task.sleep(for: .milliseconds(20))
        #expect(!completed.withLock { $0 })
        await engine.release.open()
        try await waiting.value
        try await runtime.waitForScheduledWarmup()
        #expect(await engine.preparations == 1)
        await runtime.stop(unload: true)
    }

    @available(iOS 27.0, macOS 27.0, *)
    @Test("Waiting does not load an unused model")
    func unusedModel() async throws {
        let calls = Mutex(0)
        let model = LlamaLanguageModel(engineFactory: {
            calls.withLock { $0 += 1 }
            return WarmupWaitTestEngine()
        })
        try await model.waitForScheduledWarmup()
        #expect(calls.withLock { $0 } == 0)
        await model.unload()
    }

    @available(iOS 27.0, macOS 27.0, *)
    @Test("Scheduled warmup failure remains observable")
    func preservesFailure() async {
        let runtime = LlamaExecutorRuntime { throw LlamaError.modelInitializationFailed(diagnostics: "private runtime output") }
        runtime.scheduleWarmup([])
        do { try await runtime.waitForScheduledWarmup(); Issue.record("Expected model load failure") }
        catch LlamaError.modelInitializationFailed {} catch { Issue.record("Wrong error: \(error)") }
        await runtime.stop(unload: true)
    }

    @available(iOS 27.0, macOS 27.0, *)
    @Test("Cancelling the waiter cancels and joins the existing warmup")
    func cancellation() async {
        let engine = WarmupWaitTestEngine()
        let runtime = LlamaExecutorRuntime { engine }
        runtime.scheduleWarmup([.init(role: .user, content: "Warmup")])
        await engine.started.wait()
        let waiting = Task { try await runtime.waitForScheduledWarmup() }
        await Task.yield()
        waiting.cancel()
        do { try await waiting.value; Issue.record("Expected cancellation") }
        catch is CancellationError {} catch { Issue.record("Wrong error: \(error)") }
        await runtime.stop(unload: true)
        #expect(await engine.preparations == 1)
    }

    @available(iOS 27.0, macOS 27.0, *)
    @Test("An already-cancelled waiter still joins and cancels native preparation")
    func alreadyCancelledWaiter() async {
        let engine = WarmupWaitTestEngine()
        let runtime = LlamaExecutorRuntime { engine }
        runtime.scheduleWarmup([.init(role: .user, content: "Warmup")])
        await engine.started.wait()
        let waiting = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            try await runtime.waitForScheduledWarmup()
        }
        do { try await waiting.value; Issue.record("Expected cancellation") }
        catch is CancellationError {} catch { Issue.record("Wrong error: \(error)") }
        await runtime.stop(unload: true)
        #expect(await engine.preparations == 1)
    }

    @available(iOS 27.0, macOS 27.0, *)
    @Test("Reported native runtime version follows the package artifact pin")
    func runtimeVersionMatchesArtifact() throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let manifest = try String(contentsOf: root.appendingPathComponent("Package.swift"), encoding: .utf8)
        let expected = LlamaLanguageModel.runtimeVersion.replacingOccurrences(of: "llama.cpp-", with: "")
        #expect(manifest.contains("let llamaVersion = \"\(expected)\""))
    }
}

private actor WarmupWaitGate {
    private var openValue = false
    func wait() async { while !openValue { await Task.yield() } }
    func open() { openValue = true }
}

@available(iOS 27.0, macOS 27.0, *)
private actor WarmupWaitTestEngine: LlamaExecutorEngine {
    nonisolated let started = WarmupWaitGate()
    nonisolated let release = WarmupWaitGate()
    private(set) var preparations = 0
    func prepare(_ messages: [LlamaChatMessage], addingAssistant: Bool) async throws -> Int {
        preparations += 1
        await started.open()
        while true {
            try Task.checkCancellation()
            let released = await release.isOpen
            if released { return 0 }
            try await Task.sleep(for: .milliseconds(2))
        }
    }
    func updateSamplingConfig(_ config: LlamaSamplingConfig) {}
    func generateNextToken() -> NextToken { .endOfString }
    func resetCompletion() {}
    func contextUsage(_ messages: [LlamaChatMessage], addingAssistant: Bool) -> LlamaContextUsage { .init(usedTokens: 0, effectiveCapacity: 64) }
}

private extension WarmupWaitGate { var isOpen: Bool { openValue } }
#endif
