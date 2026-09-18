#if compiler(>=6.4) && canImport(FoundationModels)
import Foundation
import FoundationModels

@available(iOS 27.0, macOS 27.0, *)
enum LlamaTranscriptMapper {
    static func messages(_ transcript: Transcript, profile: LlamaCapabilityProfile = .textOnly, tools: [Transcript.ToolDefinition]? = nil) throws -> [LlamaChatMessage] {
        var messages: [LlamaChatMessage] = []
        var definitions: [Transcript.ToolDefinition] = []
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
                guard profile != .textOnly || value.toolDefinitions.isEmpty else { throw unsupported(.toolCalling) }
                definitions += value.toolDefinitions
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
            case .toolCalls(let value):
                guard profile != .textOnly else { throw unsupported(.toolCalling) }
                hasAssistant = true
                if let raw = value.first?.metadata[LlamaLanguageModel.Metadata.rawOutput].flatMap({ try? $0.value(String.self) }) {
                    assistantRaw = raw
                } else {
                    assistantText += try renderCalls(value)
                }
            case .toolOutput(let value):
                guard profile != .textOnly else { throw unsupported(.toolCalling) }
                flushAssistant()
                let output = try value.segments.map { segment -> String in
                    if case .structure(let structured) = segment { return structured.content.jsonString }
                    return try text([segment], entry: entry)
                }.joined()
                messages.append(.init(role: .tool, content: output))
            default:
                throw LanguageModelError.unsupportedTranscriptContent(.init(
                    unsupportedContent: [entry], debugDescription: "This GGUF executor does not support this transcript entry."
                ))
            }
        }
        flushAssistant()
        guard !messages.isEmpty else { throw LlamaError.emptyMessageArray }
        if profile != .textOnly {
            let enabled = tools ?? definitions
            guard Set(enabled.map(\.name)).count == enabled.count,
                  enabled.allSatisfy({ $0.name.range(of: "^[A-Za-z_][A-Za-z0-9_]*$", options: .regularExpression) != nil }) else {
                throw LlamaToolError.malformedCall
            }
            if !enabled.isEmpty {
                let schemas = try enabled.map { definition in
                    let schema = try JSONSerialization.jsonObject(with: JSONEncoder().encode(definition.parameters))
                    return ["name": definition.name, "description": definition.description, "parameters": schema] as [String: Any]
                }
                let preamble = "List of tools: " + (try toolJSON(schemas))
                if messages.first?.role == .system {
                    let content = messages[0].content
                    messages[0] = .init(role: .system, content: content + (content.isEmpty ? "" : "\n") + preamble)
                } else { messages.insert(.init(role: .system, content: preamble), at: 0) }
            }
            for index in messages.indices { messages[index].usesLFMToolTemplate = true }
        }
        return messages
    }

    static func sampling(_ request: LanguageModelExecutorGenerationRequest, profile: LlamaCapabilityProfile = .textOnly) throws -> LlamaSamplingConfig {
        if request.schema != nil { throw unsupported(.guidedGeneration) }
        if profile == .textOnly, !request.enabledToolDefinitions.isEmpty { throw unsupported(.toolCalling) }
        if request.generationOptions.toolCallingMode == .required { throw LlamaToolError.unsupportedToolMode }
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

    private static func toolJSON(_ value: Any) throws -> String {
        if let object = value as? [String: Any] {
            let order = ["name", "description", "parameters", "type", "properties", "required", "additionalProperties"]
            // Schema titles and x-order describe Swift types, not tool arguments.
            let isSchema = object["type"] is String || object["$ref"] is String || object["anyOf"] is [Any]
            let keys = object.keys.filter { !isSchema || ($0 != "title" && $0 != "x-order") }.sorted {
                let lhs = order.firstIndex(of: $0) ?? order.count
                let rhs = order.firstIndex(of: $1) ?? order.count
                return lhs == rhs ? $0 < $1 : lhs < rhs
            }
            return "{" + (try keys.map { try toolJSON($0) + ": " + toolJSON(object[$0]!) }).joined(separator: ", ") + "}"
        }
        if let array = value as? [Any] { return "[" + (try array.map(toolJSON)).joined(separator: ", ") + "]" }
        let data = try JSONSerialization.data(withJSONObject: value, options: [.fragmentsAllowed, .withoutEscapingSlashes])
        return String(decoding: data, as: UTF8.self)
    }

    private static func renderCalls(_ calls: Transcript.ToolCalls) throws -> String {
        let rendered = try calls.map { call in
            guard let arguments = try JSONSerialization.jsonObject(with: Data(call.arguments.jsonString.utf8)) as? [String: Any] else { throw LlamaToolError.malformedCall }
            let values = try arguments.keys.sorted().map { key in
                let data = try JSONSerialization.data(withJSONObject: arguments[key]!, options: [.sortedKeys, .fragmentsAllowed, .withoutEscapingSlashes])
                return key + "=" + String(decoding: data, as: UTF8.self)
            }.joined(separator: ", ")
            return call.toolName + "(" + values + ")"
        }.joined(separator: ", ")
        return LlamaToolParser.start + "[" + rendered + "]" + LlamaToolParser.end
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
