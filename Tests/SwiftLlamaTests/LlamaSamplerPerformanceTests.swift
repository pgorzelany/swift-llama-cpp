import Foundation
import Testing
import llama
@testable import SwiftLlama

@Suite("Opt-in legacy and greedy sampling benchmark", .serialized,
       .enabled(if: ProcessInfo.processInfo.environment["LLAMA_SAMPLER_BENCHMARK"] == "1"))
struct LlamaSamplerPerformanceTests {
    struct Result: Codable {
        let implementation: String
        let repetition: Int
        let samples: Int
        let seconds: Double
    }

    @Test("Greedy fast path preserves the chosen token and reduces host sampling overhead")
    func greedyComparison() throws {
        var modelParams = llama_model_default_params()
        modelParams.n_gpu_layers = 0
        let model = try #require(LlamaModel(path: URL.llama1B.path(percentEncoded: false), parameters: modelParams))
        var contextParams = llama_context_default_params()
        contextParams.n_ctx = 512
        contextParams.n_threads = LlamaConfig.cpuThreadCount
        contextParams.n_threads_batch = LlamaConfig.cpuThreadCount
        contextParams.offload_kqv = false
        contextParams.op_offload = false
        let context = try #require(LlamaContext(model: model, parameters: contextParams))
        let prompt = model.applyChatTemplate(to: [.init(role: .user, content: "Count from 1 to 10000.")])
        let tokens = model.tokenize(text: prompt, addBos: false, special: true)
        try context.decode(batch: .singleSequence(tokens: tokens))
        let sampler = try LlamaSampler(config: .init(temperature: 0, seed: 42, repetitionPenaltyConfig: nil), model: model)
        let legacy = try #require(llama_sampler_chain_init(llama_sampler_chain_default_params()))
        defer { llama_sampler_free(legacy) }
        // Original wrapper chain for temperature=0 and penalties disabled.
        llama_sampler_chain_add(legacy, llama_sampler_init_top_p(0.95, 1))
        llama_sampler_chain_add(legacy, llama_sampler_init_temp(0))
        llama_sampler_chain_add(legacy, llama_sampler_init_dist(42))
        let expected = sampler.sample(context: context)
        let output = ProcessInfo.processInfo.environment["LLAMA_SAMPLER_BENCH_OUTPUT"] ?? "/tmp/enclave-llama-sampler-benchmark.jsonl"
        _ = FileManager.default.createFile(atPath: output, contents: Data())
        let file = try FileHandle(forWritingTo: URL(fileURLWithPath: output))
        defer { try? file.close() }
        for repetition in 0...7 {
            // Alternate order to reduce a fixed-order scheduling bias.
            let implementations = repetition % 2 == 0 ? ["legacy", "greedy"] : ["greedy", "legacy"]
            for implementation in implementations {
                let start = ContinuousClock.now
                var matches = true
                for _ in 0..<200 {
                    let token = implementation == "legacy"
                        ? llama_sampler_sample(legacy, context.contextPointer, -1)
                        : sampler.sample(context: context)
                    matches = matches && token == expected
                }
                let duration = start.duration(to: .now).components
                #expect(matches)
                let result = Result(implementation: implementation, repetition: repetition, samples: 200,
                    seconds: Double(duration.seconds) + Double(duration.attoseconds) / 1e18)
                try file.write(contentsOf: JSONEncoder().encode(result) + Data([10]))
            }
        }
    }
}
