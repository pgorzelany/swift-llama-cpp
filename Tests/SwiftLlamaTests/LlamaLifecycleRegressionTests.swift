import Foundation
import Testing
import llama
@testable import SwiftLlama

@Suite("Inference state and resource regressions", .serialized)
struct LlamaLifecycleRegressionTests {
    private func engine(gpu: Bool = true) throws -> Llama {
        try Llama(modelPath: URL.llama1B.path, config: .init(batchSize: 128, maxTokenCount: 512, useGPU: gpu))
    }

    @Test("Saved state restores C memory, token position and stochastic sampler")
    func stateRestoration() async throws {
        let engine = try engine()
        try await engine.updateSamplingConfig(.init(temperature: 0.7, seed: 42))
        let messages = [LlamaChatMessage(role: .user, content: "Count from 1 to 100, writing every number.")]
        try await engine.initializeCompletion(messages: messages)
        let state = await engine.saveStateData()
        let tokens = await engine.getProcessedTokenIds()
        func generate() async throws -> [String] {
            var output: [String] = []
            for _ in 0..<8 {
                if case .token(let piece) = try await engine.generateNextToken() { output.append(piece) }
            }
            return output
        }
        let original = try await generate()
        #expect(await engine.loadStateData(state))
        #expect(await engine.getProcessedTokenIds() == tokens)
        #expect(await engine.currentTokenPosition == Int32(tokens.count))
        #expect(try await generate() == original)
        #expect(await !engine.loadStateData(Data([0, 1, 2])))
        #expect(await !engine.loadStateData(Data()))
    }

    @Test("CPU decode abort cannot advance the Swift cache and subsequent decode recovers")
    func decodeAbort() async throws {
        let engine = try engine(gpu: false)
        try await engine.updateSamplingConfig(.init(temperature: 0, seed: 42))
        let messages = [LlamaChatMessage(role: .user, content: "Count from 1 to 100, writing every number.")]
        try await engine.initializeCompletion(messages: messages)
        await engine.setTestAbort(true)
        await #expect(throws: (any Error).self) { _ = try await engine.generateNextToken() }
        #expect(await engine.getProcessedTokenIds().isEmpty)
        #expect(await engine.currentTokenPosition == 0)
        #expect(await engine.kvMaxPosition() == -1)
        await engine.setTestAbort(false)
        try await engine.initializeCompletion(messages: messages)
        _ = try await engine.generateNextToken()
        #expect(await engine.kvMaxPosition() == engine.currentTokenPosition - 1)
    }

    @Test("A memory handle retains its context and model through deferred backend shutdown")
    func memoryOwnership() throws {
        var parameters = llama_model_default_params()
        parameters.n_gpu_layers = 0
        var model: LlamaModel? = try #require(LlamaModel(path: URL.llama1B.path, parameters: parameters))
        var contextParameters = llama_context_default_params()
        contextParameters.n_ctx = 512
        contextParameters.offload_kqv = false
        contextParameters.op_offload = false
        var context: LlamaContext? = try #require(LlamaContext(model: model!, parameters: contextParameters))
        weak var weakModel = model
        weak var weakContext = context
        var memory: LlamaMemory? = context!.memory
        context = nil
        model = nil
        LlamaBackend.shutdown()
        #expect(weakContext != nil)
        #expect(weakModel != nil)
        memory!.clear(data: true)
        #expect(memory!.maxPosition(for: 0) == -1)
        memory = nil
        #expect(weakContext == nil)
        #expect(weakModel == nil)
        LlamaBackend.initialize()
    }

    @Test("An applied adapter retains its model until it is removed from the context")
    func adapterOwnership() throws {
        var params = llama_model_default_params()
        params.n_gpu_layers = 0
        var model: LlamaModel? = try #require(LlamaModel(path: URL.llama1B.path, parameters: params))
        weak var weakModel = model
        let fixture = try #require(
            Bundle.module.url(forResource: "zero-rank1-llama-1b", withExtension: "gguf", subdirectory: "Resources/Fixtures")
            ?? Bundle.module.url(forResource: "zero-rank1-llama-1b", withExtension: "gguf", subdirectory: "Fixtures")
        )
        try #require(FileManager.default.fileExists(atPath: fixture.path(percentEncoded: false)))
        let marker = LlamaLog.marker()
        var adapter: LlamaLoraAdapter?
        do {
            adapter = try LlamaLoraAdapter(model: model!, path: fixture.path(percentEncoded: false))
        } catch {
            print("ADAPTER_DIAGNOSTICS \(LlamaLog.diagnostics(since: marker) ?? "none")")
            throw error
        }
        weak var weakAdapter = adapter
        var contextParams = llama_context_default_params()
        contextParams.n_ctx = 512
        contextParams.offload_kqv = false
        contextParams.op_offload = false
        var context: LlamaContext? = try #require(LlamaContext(model: model!, parameters: contextParams))
        try context!.apply(loraAdapter: adapter!)
        adapter = nil
        model = nil
        #expect(weakAdapter != nil)
        #expect(weakModel != nil)
        let batch = LlamaBatch.singleSequence(tokens: [128000, 9906])
        try context!.decode(batch: batch)
        context!.removeAllLoraAdapters()
        #expect(weakAdapter == nil)
        #expect(weakModel != nil)
        context = nil
        #expect(weakModel == nil)
    }

    @Test("Native generation assembles multilingual byte tokens without advancing grammar for the prompt")
    func multilingualGeneration() async throws {
        let engine = try engine()
        let expected = "Zażółć 🎯🚀🔥 中文 العربية हिन्दी"
        try await engine.initializeCompletion(messages: [.init(role: .user, content: "Return the requested text.")])
        try await engine.updateSamplingConfig(.init(temperature: 0, seed: 42,
            grammarConfig: .init(grammar: #"root ::= "\#(expected)""#), repetitionPenaltyConfig: nil))
        var output = ""
        for _ in 0..<128 {
            switch try await engine.generateNextToken() {
            case .token(let piece): output += piece
            case .endOfString:
                output += await engine.finishDecoding()
                #expect(output == expected)
                return
            }
        }
        Issue.record("Constrained multilingual generation did not finish")
    }

    @Test("Invalid engine settings fail before model loading")
    func configurationBounds() {
        for config in [
            LlamaConfig(batchSize: 0, maxTokenCount: 512),
            LlamaConfig(batchSize: 128, maxTokenCount: 0),
            LlamaConfig(batchSize: 128, maxTokenCount: 512, microBatchSize: 0),
            LlamaConfig(batchSize: 128, maxTokenCount: 512, microBatchSize: 256),
            LlamaConfig(batchSize: 128, maxTokenCount: 512, nThreads: 0),
            LlamaConfig(batchSize: 128, maxTokenCount: 512, nThreadsBatch: -1)
        ] {
            #expect(throws: LlamaError.self) {
                _ = try Llama(modelPath: "/nonexistent", config: config)
            }
        }
    }
}


private extension Llama {
    func setTestAbort(_ enabled: Bool) { context.setAbortCallback { enabled } }
}
