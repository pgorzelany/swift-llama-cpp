import Foundation
import Testing
import llama
@testable import SwiftLlama

@Suite("Opt-in Mac CPU and Metal benchmark", .serialized,
       .enabled(if: ProcessInfo.processInfo.environment["LLAMA_BENCHMARK"] == "1"))
struct LlamaPerformanceTests {
    struct Result: Codable {
        let profile: String
        let repetition: Int
        let gpu: Bool
        let threads: Int32
        let batch: UInt32
        let microBatch: UInt32
        let requestedPrompt: Int
        let actualPrompt: Int
        let outputTokens: Int
        let loadSeconds: Double
        let prefillSeconds: Double
        let firstTokenSeconds: Double
        let generationSeconds: Double
    }

    @Test("Identical prompt and generation workloads on CPU and GPU")
    func benchmark() async throws {
        LlamaLog.installDiagnosticCapture()
        llama_log_set({ _, message, _ in
            guard let message else { return }
            let line = String(cString: message)
            if ["offload", "buffer size", "n_threads", "flash_attn"].contains(where: line.contains) {
                print("BENCH_BACKEND " + line.trimmingCharacters(in: .newlines))
            }
        }, nil)
        let output = ProcessInfo.processInfo.environment["LLAMA_BENCH_OUTPUT"] ?? "/tmp/enclave-llama-swift-benchmark.jsonl"
        _ = FileManager.default.createFile(atPath: output, contents: Data())
        let file = try FileHandle(forWritingTo: URL(fileURLWithPath: output))
        defer { try? file.close() }
        let profiles: [(String, Bool, Int32, UInt32, UInt32, [Int])] = [
            ("cpu-1", false, 1, 256, 256, [64]),
            ("cpu-4", false, 4, 256, 256, [64]),
            ("cpu-8", false, 8, 256, 256, [64, 512, 2048]),
            ("metal-1", true, 1, 256, 256, [64, 512, 2048]),
            ("metal-4", true, 4, 256, 256, [64, 512]),
            ("metal-b1024", true, 1, 1024, 1024, [512, 2048]),
            ("metal-b1024-u256", true, 1, 1024, 256, [512, 2048])
        ]
        let selected = ProcessInfo.processInfo.environment["LLAMA_BENCH_PROFILES"]?.split(separator: ",").map(String.init)
        let lengthLimit = ProcessInfo.processInfo.environment["LLAMA_BENCH_MAX_PROMPT"].flatMap(Int.init)
        let modelPath = ProcessInfo.processInfo.environment["LLAMA_BENCH_MODEL"] ?? URL.llama1B.path(percentEncoded: false)
        var measuredCases = 0
        for (name, gpu, threads, batch, microBatch, lengths) in profiles where selected?.contains(name) ?? true {
            try #require(!gpu || LlamaBackend.supportsGpuOffload, "Metal benchmark requires an available GPU backend")
            let loadStart = ContinuousClock.now
            let engine = try Llama(modelPath: modelPath,
                config: .init(batchSize: batch, maxTokenCount: 4096, useGPU: gpu,
                              microBatchSize: microBatch, nThreads: threads, nThreadsBatch: threads))
            let loadSeconds = seconds(loadStart.duration(to: .now))
            try await engine.updateSamplingConfig(.init(temperature: 0, seed: 42, repetitionPenaltyConfig: nil))
            for length in lengths where lengthLimit.map({ length <= $0 }) ?? true {
                measuredCases += 1
                var padding = max(1, length - 30)
                var messages: [LlamaChatMessage] = []
                for _ in 0..<4 {
                    messages = [LlamaChatMessage(role: .user, content:
                        "Context:" + String(repeating: " hello", count: padding) +
                        "\\nCount from 1 to 10000, writing every number on its own line.")]
                    let count = try await engine.contextUsage(messages, addingAssistant: true).usedTokens
                    if count == length { break }
                    padding = max(0, padding + length - count)
                }
                for repetition in 0...3 {
                    await engine.resetCompletion()
                    let prefillStart = ContinuousClock.now
                    try await engine.initializeCompletion(messages: messages)
                    let prefillSeconds = seconds(prefillStart.duration(to: .now))
                    let promptCount = await engine.getProcessedTokenIds().count
                    #expect(promptCount == length)
                    let generationStart = ContinuousClock.now
                    var firstTokenSeconds = 0.0
                    var outputTokens = 0
                    for index in 0..<32 {
                        switch try await engine.generateNextToken() {
                        case .token:
                            outputTokens += 1
                            if index == 0 { firstTokenSeconds = seconds(generationStart.duration(to: .now)) }
                        case .endOfString:
                            break
                        }
                        if outputTokens <= index { break }
                    }
                    let generationSeconds = seconds(generationStart.duration(to: .now))
                    #expect(outputTokens == 32)
                    let result = Result(profile: name, repetition: repetition, gpu: gpu, threads: threads,
                        batch: batch, microBatch: microBatch, requestedPrompt: length,
                        actualPrompt: promptCount, outputTokens: outputTokens, loadSeconds: loadSeconds,
                        prefillSeconds: prefillSeconds, firstTokenSeconds: firstTokenSeconds,
                        generationSeconds: generationSeconds)
                    try file.write(contentsOf: JSONEncoder().encode(result) + Data([10]))
                    try file.synchronize()
                }
            }
        }
        #expect(measuredCases > 0)
    }

    private func seconds(_ duration: Duration) -> Double {
        let components = duration.components
        return Double(components.seconds) + Double(components.attoseconds) / 1e18
    }
}
