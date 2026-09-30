import Foundation
import Testing
import llama
#if compiler(>=6.4) && canImport(FoundationModels)
import FoundationModels
#endif
@testable import SwiftLlama

/// Opt-in observations of the audited implementation; these are not regression expectations.
@Suite("llama.cpp audit probes", .serialized,
       .enabled(if: ProcessInfo.processInfo.environment["LLAMA_AUDIT"] == "1"))
struct LlamaAuditProbes {
    @Test func observations() throws {
        LlamaLog.setLogger(nil)
        LlamaBackend.initialize()
        var parameters = llama_model_default_params()
        parameters.n_gpu_layers = 0
        let model = try #require(LlamaModel(path: URL.llama1B.path, parameters: parameters))
        let rendered = model.applyChatTemplate(to: [.init(role: .user, content: "Hi")], addAssistant: true)
        let formattedTokens = model.tokenize(text: rendered, addBos: model.shouldAddBos(), special: true)
        print("AUDIT_BOS native=\(llama_vocab_get_add_bos(model.vocabPointer)) wrapper=\(model.shouldAddBos()) bos=\(model.bosToken()) first=\(formattedTokens.first ?? -1)")
        for input in ["Zażółć gęślą jaźń 🎯🚀🔥", "中文 العربية हिन्दी русский", "abc\u{0}def"] {
            let tokens = model.tokenize(text: input, addBos: false, special: false)
            let pieces = tokens.map { model.piece(from: $0) }.joined()
            let whole = model.detokenize(tokens: tokens)
            print("AUDIT_TEXT \(String(reflecting: input)) pieces=\(String(reflecting: pieces)) detokenize=\(String(reflecting: whole))")
        }
        var longPieces: [(Int32, Int32)] = []
        for token in 0..<model.vocabularySize() {
            let needed = llama_token_to_piece(model.vocabPointer, token, nil, 0, 0, true)
            if needed < -64 { longPieces.append((token, -needed)) }
        }
        print("AUDIT_LONG_PIECES count=\(longPieces.count) examples=\(longPieces.prefix(5))")
        let sampler = try LlamaSampler(config: .init(temperature: 0.7, seed: 42, topK: 40), model: model)
        print("AUDIT_SAMPLER_ORDER \((0..<sampler.count()).map { sampler.name(at: Int32($0)) })")
        #expect(throws: (any Error).self) {
            _ = try LlamaSampler(config: .init(temperature: 0, seed: 42, grammarConfig: .init(grammar: "invalid")), model: model)
        }
        let longText = String(repeating: "x ", count: Int(model.trainedContextSize()) + 10)
        let required = llama_tokenize(model.vocabPointer, longText, Int32(longText.utf8.count), nil, 0, false, false)
        print("AUDIT_LONG_TOKENIZE required=\(-required) wrapperPasses=\(model.trainedContextSize()) actualAllocation=\(longText.utf8.count + 1)")
    }

    @Test func isolatedFailure() throws {
        guard let mode = ProcessInfo.processInfo.environment["LLAMA_AUDIT_FAILURE"] else { return }
        LlamaLog.setLogger(nil)
        if mode == "borrowed-batch" {
            let batch = LlamaBatch.singleSequence(tokens: [1, 2, 3])
            print("AUDIT_BORROWED_BATCH size=\(batch.size)")
            withExtendedLifetime(batch) {}
            return
        }
        LlamaBackend.initialize()
        var parameters = llama_model_default_params()
        parameters.n_gpu_layers = 0
        parameters.vocab_only = mode == "vocab-only-tokenize"
        let model = try #require(LlamaModel(path: URL.llama1B.path, parameters: parameters))
        switch mode {
        case "long-piece":
            for token in 0..<model.vocabularySize() {
                if llama_token_to_piece(model.vocabPointer, token, nil, 0, 0, true) < -64 {
                    print("AUDIT_LONG_PIECE token=\(token)")
                    _ = model.piece(from: token, renderSpecial: true)
                    break
                }
            }
        case "long-tokenize":
            _ = model.tokenize(text: String(repeating: "x ", count: Int(model.trainedContextSize()) + 10), addBos: false, special: false)
        case "vocab-only-tokenize":
            print("AUDIT_VOCAB_ONLY_CRASH contextTrain=\(model.trainedContextSize())")
            _ = model.tokenize(text: "Hello", addBos: false, special: false)
        case "eos-pointer":
            print("AUDIT_EOS wrapper=\(model.eosToken()) correct=\(llama_vocab_eos(model.vocabPointer))")
        default: throw NSError(domain: "Unknown audit probe", code: 1)
        }
    }

    @Test func repeatedExactPrompt() async throws {
        let model = try Llama(modelPath: URL.llama1B.path, config: .init(batchSize: 64, maxTokenCount: 512, useGPU: false))
        let prompt = [LlamaChatMessage(role: .user, content: "Answer in one word: France's capital?")]
        try await model.initializeCompletion(messages: prompt)
        let first = try #require(await model.getLastLogits())
        try await model.initializeCompletion(messages: prompt)
        let second = await model.getLastLogits()
        print("AUDIT_EXACT_PROMPT logitsBefore=\(first.count) logitsAfter=\(second?.count ?? 0) equal=\(second == first)")
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["GEMMA4_GGUF_PATH"] != nil))
    func controlTokenRendering() throws {
        LlamaLog.setLogger(nil)
        let path = try #require(ProcessInfo.processInfo.environment["GEMMA4_GGUF_PATH"])
        var parameters = llama_model_default_params()
        parameters.vocab_only = true
        parameters.n_gpu_layers = 0
        let model = try #require(LlamaModel(path: path, parameters: parameters))
        let protocolText = "<|channel>thought\nPlan<channel|>"
        // Vocab-only has no trained context; avoid the independently confirmed capacity bug.
        let needed = -llama_tokenize(model.vocabPointer, protocolText, Int32(protocolText.utf8.count), nil, 0, false, true)
        var tokens = [llama_token](repeating: 0, count: Int(needed))
        let written = llama_tokenize(model.vocabPointer, protocolText, Int32(protocolText.utf8.count), &tokens, needed, false, true)
        #expect(written == needed)
        print("AUDIT_VOCAB_ONLY contextTrain=\(model.trainedContextSize()) actualTokens=\(written)")
        print("AUDIT_GEMMA_PROTOCOL tokens=\(tokens) hidden=\(String(reflecting: tokens.map { model.piece(from: $0, renderSpecial: false) }.joined())) visible=\(String(reflecting: tokens.map { model.piece(from: $0, renderSpecial: true) }.joined()))")
    }

    #if compiler(>=6.4) && canImport(FoundationModels)
    @available(iOS 27.0, macOS 27.0, *)
    @Test func syntheticStreamingCost() async throws {
        for count in [1_000, 4_000, 8_000] {
            let model = LlamaLanguageModel(engineFactory: { AuditSyntheticEngine(remaining: count) })
            let session = LanguageModelSession(model: model)
            let start = ContinuousClock.now
            let response = try await session.respond(to: "Synthetic audit")
            #expect(response.content == String(repeating: "word ", count: count))
            print("AUDIT_STREAMING tokens=\(count) elapsed=\(start.duration(to: .now))")
            await model.unload()
        }
    }
    #endif
}

#if compiler(>=6.4) && canImport(FoundationModels)
@available(iOS 27.0, macOS 27.0, *)
private actor AuditSyntheticEngine: LlamaExecutorEngine {
    var remaining: Int
    init(remaining: Int) { self.remaining = remaining }
    func prepare(_ messages: [LlamaChatMessage], addingAssistant: Bool) -> Int { 16 }
    func updateSamplingConfig(_ config: LlamaSamplingConfig) {}
    func generateNextToken() -> NextToken {
        guard remaining > 0 else { return .endOfString }
        remaining -= 1
        return .token("word ")
    }
    func resetCompletion() {}
    func contextUsage(_ messages: [LlamaChatMessage], addingAssistant: Bool) -> LlamaContextUsage {
        .init(usedTokens: 16, effectiveCapacity: 16_384)
    }
}
#endif
