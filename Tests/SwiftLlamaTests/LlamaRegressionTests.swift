import Foundation
import Testing
import llama
@testable import SwiftLlama

@Suite("llama.cpp regression contracts", .serialized)
struct LlamaRegressionTests {
    private func vocabulary() throws -> LlamaModel {
        var parameters = llama_model_default_params()
        parameters.vocab_only = true
        return try #require(LlamaModel(path: URL.llama1B.path, parameters: parameters))
    }

    @Test("Special tokens and automatic BOS follow the vocabulary")
    func specialTokens() throws {
        let model = try vocabulary()
        #expect(model.eosToken() == llama_vocab_eos(model.vocabPointer))
        #expect(model.shouldAddBos() == llama_vocab_get_add_bos(model.vocabPointer))
    }

    @Test("Detokenization preserves embedded NUL and complete Unicode")
    func exactDetokenization() throws {
        let model = try #require(LlamaModel(path: URL.llama1B.path))
        for text in ["abc\u{0}def", "Zażółć gęślą jaźń 🎯🚀🔥", "中文 العربية हिन्दी русский"] {
            let tokens = model.tokenize(text: text, addBos: false, special: false)
            #expect(model.detokenize(tokens: tokens, removeSpecial: false) == text)
        }
    }

    @Test("Penalties precede candidate filtering")
    func samplerOrder() throws {
        let model = try vocabulary()
        let sampler = try LlamaSampler(config: .init(temperature: 0.7, seed: 42, topK: 40), model: model)
        let names = (0..<sampler.count()).map { sampler.name(at: Int32($0)) }
        #expect(names == ["penalties", "top-k", "top-p", "temp", "dist"])
    }

    @Test("Invalid grammar reports an error instead of unconstrained generation")
    func invalidGrammar() throws {
        let model = try vocabulary()
        #expect(throws: (any Error).self) {
            _ = try LlamaSampler(config: .init(temperature: 0, seed: 42, grammarConfig: .init(grammar: "invalid")), model: model)
        }
    }
}

@Suite("Byte, ownership and sampling regressions", .serialized)
struct LlamaBoundaryRegressionTests {
    private func vocabulary() throws -> LlamaModel {
        var parameters = llama_model_default_params()
        parameters.vocab_only = true
        return try #require(LlamaModel(path: URL.llama1B.path, parameters: parameters))
    }

    private func nativeTokens(_ text: String, model: LlamaModel) -> [llama_token] {
        let needed = -llama_tokenize(model.vocabPointer, text, Int32(text.utf8.count), nil, 0, false, false)
        guard needed > 0 else { return [] }
        var tokens = [llama_token](repeating: 0, count: Int(needed))
        let written = llama_tokenize(model.vocabPointer, text, Int32(text.utf8.count), &tokens, needed, false, false)
        #expect(written == needed)
        return tokens
    }

    @Test("Tokenization uses buffer capacity even without model context and beyond training size")
    func tokenizerSizing() throws {
        let model = try vocabulary()
        #expect(model.trainedContextSize() == 0)
        for text in ["", "Hello", "Zażółć 🎯", String(repeating: "x ", count: 131_090)] {
            #expect(model.tokenize(text: text, addBos: false, special: false) == nativeTokens(text, model: model))
        }
    }

    @Test("Every oversized vocabulary piece matches exact C bytes")
    func longPieces() throws {
        let model = try vocabulary()
        var count = 0
        for token in 0..<model.vocabularySize() {
            let required = -llama_token_to_piece(model.vocabPointer, token, nil, 0, 0, true)
            guard required > 64 else { continue }
            var buffer = [CChar](repeating: 0, count: Int(required))
            #expect(llama_token_to_piece(model.vocabPointer, token, &buffer, required, 0, true) == required)
            #expect(model.pieceBytes(from: token, renderSpecial: true) == buffer.map { UInt8(bitPattern: $0) })
            count += 1
        }
        #expect(count > 0)
    }

    @Test("Streaming UTF-8 preserves all byte split positions and token boundaries")
    func unicodeSplits() throws {
        let model = try vocabulary()
        for text in ["Zażółć gęślą jaźń 🎯🚀🔥", "中文 العربية हिन्दी русский", "e\u{301} 👩🏽‍💻", "abc\u{0}def"] {
            let bytes = Array(text.utf8)
            for boundary in 0...bytes.count {
                var decoder = LlamaUTF8Decoder()
                let output = decoder.append(Array(bytes.prefix(boundary))) + decoder.append(Array(bytes.dropFirst(boundary))) + decoder.finish()
                #expect(output == text)
            }
            var decoder = LlamaUTF8Decoder()
            var output = ""
            for token in nativeTokens(text, model: model) { output += decoder.append(model.pieceBytes(from: token)) }
            output += decoder.finish()
            #expect(output == text)
        }
        var decoder = LlamaUTF8Decoder()
        #expect(decoder.append([0xF0, 0x9F]).isEmpty)
        #expect(decoder.finish() == "�")
        #expect(decoder.append(Array("fresh".utf8)) == "fresh")
    }

    @Test("Single sequence owns its token, position and output buffers")
    func ownedBatch() {
        for _ in 0..<1_000 {
            let batch = LlamaBatch.singleSequence(tokens: [1, 2, 3])
            #expect(batch.size == 3)
            #expect(Array(UnsafeBufferPointer(start: batch.rawBatch.token, count: 3)) == [1, 2, 3])
            #expect(Array(UnsafeBufferPointer(start: batch.rawBatch.pos, count: 3)) == [0, 1, 2])
            #expect(Array(UnsafeBufferPointer(start: batch.rawBatch.logits, count: 3)) == [0, 0, 1])
        }
        #expect(LlamaBatch.singleSequence(tokens: []).size == 0)
    }

    @Test("Invalid batch operations fail without mutating memory")
    func batchBounds() {
        let tokens = LlamaBatch(initialSize: 1)
        #expect(!tokens.setLastTokenLogits(true))
        #expect(!tokens.setEmbedding([1, 2]))
        #expect(tokens.addToken(10, at: 0, logits: true))
        #expect(!tokens.addToken(11, at: 1, logits: true))
        #expect(tokens.size == 1)
        let embeddings = LlamaBatch(embeddingCapacity: 2, embeddingSize: 3)
        #expect(!embeddings.addToken(10, at: 0, logits: false))
        #expect(!embeddings.setEmbedding([1, 2]))
        #expect(embeddings.addEmbedding([1, 2, 3], at: 4))
        #expect(embeddings.addEmbedding([4, 5, 6], at: 5, logits: true))
        #expect(!embeddings.setEmbedding([7, 8, 9]))
        #expect(Array(UnsafeBufferPointer(start: embeddings.rawBatch.embd, count: 6)) == [1, 2, 3, 4, 5, 6])
        #expect(Array(UnsafeBufferPointer(start: embeddings.rawBatch.pos, count: 2)) == [4, 5])
        #expect(LlamaBatch(initialSize: -1).capacity == 0)
    }

    @Test("Greedy chooses the best penalized logit even when top-k would have discarded it")
    func penalizedGreedy() throws {
        let model = try vocabulary()
        let sampler = try LlamaSampler(config: .init(temperature: 0, seed: 42, topP: 0.1, topK: 1,
            repetitionPenaltyConfig: .init(repeatPenalty: 2)), model: model)
        sampler.acceptPrompt(tokens: [17])
        var candidates = [llama_token_data(id: 17, logit: 10, p: 0), llama_token_data(id: 18, logit: 9, p: 0)]
        candidates.withUnsafeMutableBufferPointer { buffer in
            var data = llama_token_data_array(data: buffer.baseAddress, size: buffer.count, selected: -1, sorted: false)
            llama_sampler_apply(sampler.samplerPointer, &data)
            #expect(data.data[Int(data.selected)].id == 18)
        }
        #expect((0..<sampler.count()).map { sampler.name(at: Int32($0)) } == ["penalties", "greedy"])
    }

    @Test("Sampler removal frees intermediate stages and preserves its selector")
    func samplerOwnership() throws {
        var model: LlamaModel? = try vocabulary()
        weak var weakModel = model
        var sampler: LlamaSampler? = try LlamaSampler(config: .init(temperature: 0.7, seed: 42), model: model!)
        model = nil
        #expect(weakModel != nil)
        let before = sampler!.count()
        #expect(sampler!.remove(at: 0))
        #expect(sampler!.count() == before - 1)
        #expect(!sampler!.remove(at: Int32(sampler!.count() - 1)))
        #expect(!sampler!.remove(at: -1))
        #expect(!sampler!.perfDataDescription().isEmpty)
        var clone = sampler!.clone()
        sampler = nil
        #expect(weakModel != nil)
        #expect(clone?.count() == before - 1)
        clone = nil
        #expect(weakModel == nil)
    }

    @Test("Long metadata, split paths and invalid split input have explicit results")
    func metadataAndSplits() throws {
        let model = try vocabulary()
        let key = "tokenizer.chat_template"
        let needed = llama_model_meta_val_str(model.modelPointer, key, nil, 0)
        if needed > 0 {
            var buffer = [CChar](repeating: 0, count: Int(needed) + 1)
            _ = llama_model_meta_val_str(model.modelPointer, key, &buffer, buffer.count)
            #expect(model.metaValue(forKey: key) == String(cString: buffer))
        }
        let prefix = String(repeating: "long/", count: 300)
        let path = LlamaModel.splitPath(pathPrefix: prefix, splitNo: 0, splitCount: 2)
        #expect(path == prefix + "-00001-of-00002.gguf")
        #expect(LlamaModel.splitPrefix(splitPath: path, splitNo: 0, splitCount: 2) == prefix)
        #expect(LlamaModel(paths: []) == nil)
        #expect(LlamaModel.splitPath(pathPrefix: "model", splitNo: -1, splitCount: 1).isEmpty)
        #expect(LlamaModel.splitPath(pathPrefix: "model", splitNo: Int32.max, splitCount: Int32.max).isEmpty)
        #expect(LlamaModel.splitPrefix(splitPath: "model", splitNo: 0, splitCount: 0) == nil)
        #expect(model.builtinChatTemplates(maxCount: -1).isEmpty)
        #expect(model.applyChatTemplate(name: "missing", to: [.init(role: .user, content: "Hi")]).isEmpty)
        #expect(model.applyChatTemplate(to: [.init(role: .user, content: "a\u{0}b")]).isEmpty)
    }
}
