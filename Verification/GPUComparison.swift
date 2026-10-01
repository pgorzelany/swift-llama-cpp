import Foundation
import CryptoKit
import llama

/// Compiled alongside each immutable revision's unmodified SwiftLlama sources.
@main
struct GPUComparison {
    struct Workload: Codable {
        let requestedTokens: Int
        let content: String
        let promptTokens: [Int32]
    }

    struct Result: Codable {
        let revision: String
        let model: String
        let block: Int
        let repetition: Int
        let sampling: String
        let requestedPrompt: Int
        let actualPrompt: Int
        let outputTokens: Int
        let loadSeconds: Double
        let prefillSeconds: Double
        let firstTokenSeconds: Double
        let generationSeconds: Double
        let promptHash: String
        let omittedLeadingToken: Int32?
        let generatedTokens: [Int32]
        let threads: Int32
        let batch: UInt32
        let microBatch: UInt32
        let thermalState: Int
    }

    static func main() async throws {
        let arguments = CommandLine.arguments
        guard arguments.count == 9 else { throw Failure("Expected mode model fixture output revision label block repetitions") }
        let mode = arguments[1]
        let path = arguments[2]
        let fixture = URL(fileURLWithPath: arguments[3])
        let output = URL(fileURLWithPath: arguments[4])
        let revision = arguments[5]
        let label = arguments[6]
        guard let block = Int(arguments[7]), let repetitions = Int(arguments[8]) else { throw Failure("Invalid run counters") }
        guard LlamaBackend.supportsGpuOffload else { throw Failure("GPU unavailable") }
        LlamaLog.installDiagnosticCapture()
        llama_log_set(benchmarkLog, nil)
        let loadStart = ContinuousClock.now
        let engine = try Llama(modelPath: path, config: .init(batchSize: 1024, maxTokenCount: 4096, useGPU: true))
        let loadSeconds = seconds(loadStart.duration(to: .now))
        if mode == "prepare" {
            var workloads: [Workload] = []
            for length in [64, 512, 2048] {
                var padding = max(1, length - 30)
                var content = ""
                for _ in 0..<8 {
                    content = "Context:" + String(repeating: " hello", count: padding)
                        + "\nWrite a 1000-word science fiction story about an engineer discovering a new planet. Start the story immediately."
                    let count = try await engine.contextUsage([.init(role: .user, content: content)], addingAssistant: true).usedTokens
                    if count == length { break }
                    padding = max(0, padding + length - count)
                }
                try await engine.initializeCompletion(messages: [.init(role: .user, content: content)], addAssistant: true)
                let tokens = await engine.getProcessedTokenIds()
                guard tokens.count == length else { throw Failure("Cannot construct exact prompt") }
                workloads.append(.init(requestedTokens: length, content: content, promptTokens: tokens))
                await engine.resetCompletion()
            }
            try JSONEncoder().encode(workloads).write(to: fixture)
            return
        }
        guard mode == "benchmark", repetitions > 0 else { throw Failure("Invalid mode") }
        let workloads = try JSONDecoder().decode([Workload].self, from: Data(contentsOf: fixture))
        _ = FileManager.default.createFile(atPath: output.path, contents: Data())
        let file = try FileHandle(forWritingTo: output)
        defer { try? file.close() }
        let names = block.isMultiple(of: 2)
            ? ["greedy-controlled", "greedy-native", "production"]
            : ["production", "greedy-native", "greedy-controlled"]
        for sampling in names {
            for workload in block.isMultiple(of: 2) ? workloads : Array(workloads.reversed()) {
                for repetition in 0...repetitions {
                    await engine.resetCompletion()
                    let config = sampling.hasPrefix("greedy")
                        ? LlamaSamplingConfig(temperature: 0, seed: 42, repetitionPenaltyConfig: nil)
                        : LlamaSamplingConfig(temperature: 0.5, seed: 42)
                    // Rebuild both samplers before each trial: old resetCompletion does not reset sampling history.
                    try await engine.updateSamplingConfig(config)
                    let prefillStart = ContinuousClock.now
                    if sampling == "greedy-controlled" {
                        try await engine.benchmarkInitializePrompt(tokens: workload.promptTokens)
                    } else {
                        try await engine.initializeCompletion(messages: [.init(role: .user, content: workload.content)], addAssistant: true)
                    }
                    let prefillSeconds = seconds(prefillStart.duration(to: .now))
                    let prompt = await engine.getProcessedTokenIds()
                    let omittedLeadingToken: Int32?
                    if prompt == workload.promptTokens {
                        omittedLeadingToken = nil
                    } else if sampling != "greedy-controlled", prompt == Array(workload.promptTokens.dropFirst()) {
                        omittedLeadingToken = workload.promptTokens.first
                    } else {
                        throw Failure("Unexpected prompt mismatch: \(prompt.count) vs \(workload.promptTokens.count); \(prompt.prefix(12)) vs \(workload.promptTokens.prefix(12))")
                    }
                    let generationStart = ContinuousClock.now
                    var firstTokenSeconds = 0.0
                    for index in 0..<32 {
                        switch try await engine.generateNextToken() {
                        case .token:
                            if index == 0 { firstTokenSeconds = seconds(generationStart.duration(to: .now)) }
                        case .endOfString:
                            throw Failure("Early EOS at \(index), workload must have 32 output tokens")
                        }
                    }
                    let generationSeconds = seconds(generationStart.duration(to: .now))
                    let allTokens = await engine.getProcessedTokenIds()
                    let generated = Array(allTokens.dropFirst(prompt.count))
                    guard generated.count == 32 else { throw Failure("Invalid generation count") }
                    let settings = await engine.benchmarkSettings()
                    guard settings.0 == 1, settings.1 == 1, settings.2 == 1024, settings.3 == 1024 else {
                        throw Failure("Unexpected execution settings")
                    }
                    let result = Result(revision: revision, model: label, block: block,
                        repetition: repetition == 0 ? 0 : block * repetitions + repetition,
                        sampling: sampling, requestedPrompt: workload.requestedTokens, actualPrompt: prompt.count,
                        outputTokens: generated.count, loadSeconds: loadSeconds, prefillSeconds: prefillSeconds,
                        firstTokenSeconds: firstTokenSeconds, generationSeconds: generationSeconds,
                        promptHash: SHA256.hash(data: Data(workload.content.utf8)).map { String(format: "%02x", $0) }.joined(),
                        omittedLeadingToken: omittedLeadingToken,
                        generatedTokens: generated, threads: settings.0, batch: settings.2, microBatch: settings.3,
                        thermalState: ProcessInfo.processInfo.thermalState.rawValue)
                    try file.write(contentsOf: JSONEncoder().encode(result) + Data([10]))
                    try file.synchronize()
                }
            }
        }
    }

    private static func seconds(_ duration: Duration) -> Double {
        let value = duration.components
        return Double(value.seconds) + Double(value.attoseconds) / 1e18
    }

    struct Failure: Error, CustomStringConvertible {
        let description: String
        init(_ description: String) { self.description = description }
    }
}

extension Llama {
    func benchmarkSettings() -> (Int32, Int32, UInt32, UInt32) {
        (context.nThreads(), context.nThreadsBatch(), context.batchSize(), context.ubatchSize())
    }

    /// Replays the common fixture solely for a controlled decode/sampling comparison.
    func benchmarkInitializePrompt(tokens: [Int32]) throws {
        let promptBatch = LlamaBatch(initialSize: 1024)
        for (index, token) in tokens.enumerated() {
            promptBatch.addToken(token, at: Int32(index), logits: index == tokens.count - 1)
            if promptBatch.size == 1024 || index == tokens.count - 1 {
                try context.decode(batch: promptBatch)
                promptBatch.reset()
            }
        }
        processedTokens = tokens
        currentTokenPosition = Int32(tokens.count)
    }
}

nonisolated private func benchmarkLog(_ level: ggml_log_level, _ message: UnsafePointer<CChar>?, _ data: UnsafeMutableRawPointer?) {
    guard let message else { return }
    let line = String(cString: message)
    if ["offload", "buffer size", "n_threads", "flash_attn"].contains(where: line.contains) {
        print("AB_BACKEND " + line.trimmingCharacters(in: .newlines))
    }
}
