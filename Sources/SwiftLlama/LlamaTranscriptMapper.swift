#if compiler(>=6.4) && canImport(FoundationModels)
import Foundation
import FoundationModels

@available(iOS 27.0, macOS 27.0, *)
enum LlamaTranscriptMapper {
    static func messages(_ transcript: Transcript) throws -> [LlamaChatMessage] {
        var messages: [LlamaChatMessage] = []
        var assistantText = ""
        var assistantRaw: String?
        var hasAssistant = false
        func flushAssistant() {
            if hasAssistant { messages.append(.init(role: .assistant, content: assistantRaw ?? assistantText)) }
            assistantText = ""
            assistantRaw = nil
            hasAssistant = false
        }
        func retainRaw(_ metadata: [String: GeneratedContent]) {
            guard let raw = metadata[LlamaLanguageModel.Metadata.rawOutput].flatMap({ try? $0.value(String.self) }) else { return }
            // Entries retain insertion order, while their cumulative metadata continues to update.
            if raw.count >= (assistantRaw?.count ?? 0) { assistantRaw = raw }
        }
        for entry in transcript {
            switch entry {
            case .instructions(let value):
                guard value.toolDefinitions.isEmpty else { throw unsupported(.toolCalling) }
                flushAssistant()
                messages.append(.init(role: .system, content: try text(value.segments, entry: entry)))
            case .prompt(let value):
                flushAssistant()
                messages.append(.init(role: .user, content: try text(value.segments, entry: entry)))
            case .reasoning(let value):
                hasAssistant = true
                assistantText += "<think>" + (try text(value.segments, entry: entry)) + "</think>"
                retainRaw(value.metadata)
            case .response(let value):
                hasAssistant = true
                assistantText += try text(value.segments, entry: entry)
                retainRaw(value.metadata)
            default:
                throw LanguageModelError.unsupportedTranscriptContent(.init(
                    unsupportedContent: [entry], debugDescription: "This GGUF executor accepts text and tagged reasoning only."
                ))
            }
        }
        flushAssistant()
        guard !messages.isEmpty else { throw LlamaError.emptyMessageArray }
        return messages
    }

    static func sampling(_ request: LanguageModelExecutorGenerationRequest) throws -> LlamaSamplingConfig {
        if request.schema != nil { throw unsupported(.guidedGeneration) }
        if !request.enabledToolDefinitions.isEmpty { throw unsupported(.toolCalling) }
        if request.contextOptions.reasoningLevel != nil { throw unsupported(.reasoning) }
        if let maximum = request.generationOptions.maximumResponseTokens, maximum <= 0 {
            throw LlamaExecutorError.invalidOutputLimit
        }
        var temperature = CFloat(request.generationOptions.temperature ?? 0.7)
        var seed: UInt32 = 0
        var topP: Float = 0.95
        var topK: Int32?
        switch request.generationOptions.samplingMode?.kind {
        case .greedy: temperature = 0
        case .randomTopK(let k, let value):
            topK = Int32(clamping: k)
            seed = UInt32(truncatingIfNeeded: value ?? 0)
        case .randomProbabilityThreshold(let p, let value):
            topP = Float(p)
            seed = UInt32(truncatingIfNeeded: value ?? 0)
        case nil: break
        @unknown default: break
        }
        return LlamaSamplingConfig(temperature: temperature, seed: seed, topP: topP, topK: topK)
    }

    private static func text(_ segments: [Transcript.Segment], entry: Transcript.Entry) throws -> String {
        try segments.map { segment in
            guard case .text(let text) = segment else {
                throw LanguageModelError.unsupportedTranscriptContent(.init(
                    unsupportedContent: [entry], debugDescription: "This GGUF executor does not support non-text segments."
                ))
            }
            return text.content
        }.joined()
    }

    private static func unsupported(_ capability: LanguageModelCapabilities.Capability) -> LanguageModelError {
        .unsupportedCapability(.init(capability: capability, debugDescription: "This GGUF executor does not implement the requested capability."))
    }
}

@available(iOS 27.0, macOS 27.0, *)
enum LlamaExecutorError: LocalizedError {
    case busy
    case invalidOutputLimit
    case invalidConfiguration

    var errorDescription: String? {
        switch self {
        case .busy: "This GGUF model is already generating or stopping. Use an independent model instance for another session."
        case .invalidOutputLimit: "The maximum response token count must be positive."
        case .invalidConfiguration: "GGUF inference requires a positive batch size and a context window of at least eight tokens."
        }
    }
}
#endif
