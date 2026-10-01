import Foundation
import llama

@main
struct GenerationQuality {
    struct Result: Codable {
        let revision: String
        let model: String
        let sampling: String
        let scenario: String
        let messages: [Message]
        let text: String
        let nativeText: String
        let generatedTokens: [Int32]
        let promptTokens: Int
        let endedNaturally: Bool
        let referenceTokens: [Int32]?
    }

    struct Message: Codable {
        let role: String
        let content: String
    }

    static func main() async throws {
        let args = CommandLine.arguments
        guard args.count == 5 else { throw Failure("Expected model output revision label") }
        guard LlamaBackend.supportsGpuOffload else { throw Failure("GPU unavailable") }
        LlamaLog.setLogger(nil)
        let engine = try Llama(modelPath: args[1], config: .init(batchSize: 1024, maxTokenCount: 4096, useGPU: true))
        let output = URL(fileURLWithPath: args[2])
        _ = FileManager.default.createFile(atPath: output.path, contents: Data())
        let file = try FileHandle(forWritingTo: output)
        defer { try? file.close() }
        let scenarios: [(String, [LlamaChatMessage])] = [
            ("instruction", [.init(role: .user, content: "Reply with exactly the single word READY. Do not add anything else.")]),
            ("arithmetic", [.init(role: .user, content: "What is 17 + 25? Reply only with the integer.")]),
            ("polish", [.init(role: .user, content: "Odpowiedz jednym krótkim zdaniem po polsku: do czego służy lodówka?")]),
            ("translation", [.init(role: .user, content: "Translate into Polish: Good morning, thank you for your help.")]),
            ("summary", [.init(role: .user, content: "Summarize in one English sentence: Marta boarded a train in Gdansk at 09:00 and arrived in Warsaw at 12:00.")]),
            ("conversation", [
                .init(role: .system, content: "Answer the final question using the conversation. Be concise."),
                .init(role: .user, content: "The project passphrase is cobalt-lantern. Remember it."),
                .init(role: .assistant, content: "I will remember that the project passphrase is cobalt-lantern."),
                .init(role: .user, content: "What is the project passphrase? Reply only with the passphrase.")
            ]),
            ("story", [.init(role: .user, content: "Write a concise two-sentence story about a cat living on Mars. Be specific.")]),
            ("unicode", [.init(role: .user, content: "Repeat exactly this text and nothing else: Zażółć gęślą jaźń 🌍 中文 العربية हिन्दी")]),
            ("json", [.init(role: .user, content: "Return only a JSON object, without Markdown, with exactly two keys: name with value Ada and age with integer value 36.")])
        ]
        for (scenario, messages) in scenarios {
            for (sampling, temperature) in [("greedy", Float(0)), ("production", Float(0.5))] {
                await engine.resetCompletion()
                try await engine.updateSamplingConfig(.init(temperature: temperature, seed: 42))
                try await engine.initializeCompletion(messages: messages, addAssistant: true)
                let promptCount = await engine.getProcessedTokenIds().count
                var text = ""
                var endedNaturally = false
                generation: for _ in 0..<256 {
                    switch try await engine.generateNextToken() {
                    case .token(let piece): text += piece
                    case .endOfString:
                        endedNaturally = true
                        break generation
                    }
                }
                text += await engine.qualityFinish()
                let processed = await engine.getProcessedTokenIds()
                let tokens = Array(processed.dropFirst(promptCount))
                let nativeText = try await engine.qualityNativeText(tokens: tokens)
                let referenceTokens = try await engine.qualityReferenceTokens(prompt: Array(processed.prefix(promptCount)), temperature: temperature)
                let result = Result(revision: args[3], model: args[4], sampling: sampling, scenario: scenario,
                    messages: messages.map { .init(role: $0.role.rawValue, content: $0.content) },
                    text: text, nativeText: nativeText, generatedTokens: tokens,
                    promptTokens: promptCount, endedNaturally: endedNaturally, referenceTokens: referenceTokens)
                try file.write(contentsOf: JSONEncoder().encode(result) + Data([10]))
                try file.synchronize()
            }
        }
    }

    struct Failure: Error {
        let reason: String
        init(_ reason: String) { self.reason = reason }
    }
}

extension Llama {
    /// Independent C context, batch, sampler and decode loop for the corrected configuration.
    func qualityReferenceTokens(prompt: [Int32], temperature: Float) throws -> [Int32]? {
        #if QUALITY_FIXED
        var params = llama_context_default_params()
        params.n_ctx = 4096
        params.n_batch = 1024
        params.n_ubatch = 1024
        params.n_threads = 1
        params.n_threads_batch = 1
        params.offload_kqv = true
        params.op_offload = true
        guard let native = llama_init_from_model(context.model.modelPointer, params) else {
            throw GenerationQuality.Failure("Native C context failed")
        }
        defer { llama_free(native) }
        var nativeBatch = llama_batch_init(1024, 0, 1)
        defer { llama_batch_free(nativeBatch) }
        for start in stride(from: 0, to: prompt.count, by: 1024) {
            let count = min(1024, prompt.count - start)
            nativeBatch.n_tokens = Int32(count)
            for index in 0..<count {
                nativeBatch.token[index] = prompt[start + index]
                nativeBatch.pos[index] = Int32(start + index)
                nativeBatch.n_seq_id[index] = 1
                nativeBatch.seq_id[index]![0] = 0
                nativeBatch.logits[index] = start + index == prompt.count - 1 ? 1 : 0
            }
            guard llama_decode(native, nativeBatch) == 0 else { throw GenerationQuality.Failure("Native C prefill failed") }
            llama_synchronize(native)
        }
        guard let chain = llama_sampler_chain_init(llama_sampler_chain_default_params()),
              let penalties = llama_sampler_init_penalties(context.model.vocabularySize(), 64, 1.1, 0, 0) else {
            throw GenerationQuality.Failure("Native C sampler failed")
        }
        defer { llama_sampler_free(chain) }
        llama_sampler_chain_add(chain, penalties)
        for token in prompt { llama_sampler_accept(penalties, token) }
        if temperature == 0 {
            llama_sampler_chain_add(chain, llama_sampler_init_greedy())
        } else {
            llama_sampler_chain_add(chain, llama_sampler_init_top_p(0.95, 1))
            llama_sampler_chain_add(chain, llama_sampler_init_temp(temperature))
            llama_sampler_chain_add(chain, llama_sampler_init_dist(42))
        }
        var tokens: [Int32] = []
        for index in 0..<256 {
            let token = llama_sampler_sample(chain, native, -1)
            if llama_vocab_is_eog(context.model.vocabPointer, token) { break }
            nativeBatch.n_tokens = 1
            nativeBatch.token[0] = token
            nativeBatch.pos[0] = Int32(prompt.count + index)
            nativeBatch.n_seq_id[0] = 1
            nativeBatch.seq_id[0]![0] = 0
            nativeBatch.logits[0] = 1
            guard llama_decode(native, nativeBatch) == 0 else { throw GenerationQuality.Failure("Native C decode failed") }
            llama_synchronize(native)
            tokens.append(token)
        }
        return tokens
        #else
        return nil
        #endif
    }

    func qualityFinish() -> String {
        #if QUALITY_FIXED
        return finishDecoding()
        #else
        return ""
        #endif
    }

    /// Direct C detokenization checks that streaming preserves the sampled bytes.
    func qualityNativeText(tokens: [Int32]) throws -> String {
        var buffer = [CChar](repeating: 0, count: 64)
        let needed = tokens.withUnsafeBufferPointer {
            llama_detokenize(context.model.vocabPointer, $0.baseAddress, Int32($0.count), &buffer, Int32(buffer.count), false, false)
        }
        if needed < 0 {
            guard needed != Int32.min else { throw GenerationQuality.Failure("Invalid detokenization size") }
            buffer = [CChar](repeating: 0, count: Int(-needed))
        }
        let written = tokens.withUnsafeBufferPointer {
            llama_detokenize(context.model.vocabPointer, $0.baseAddress, Int32($0.count), &buffer, Int32(buffer.count), false, false)
        }
        guard written >= 0, written <= buffer.count else { throw GenerationQuality.Failure("Detokenization failed") }
        return String(decoding: buffer.prefix(Int(written)).map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }
}
