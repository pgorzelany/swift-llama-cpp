#if compiler(>=6.4) && canImport(FoundationModels)
import Foundation
import FoundationModels
import CryptoKit
import CoreFoundation

/// Opts into a tested model/runtime contract; downloaded models default to text only.
@available(iOS 27.0, macOS 27.0, *)
public enum LlamaCapabilityProfile: Sendable {
    case textOnly
    /// LFM2.5-1.2B-Instruct-QAD-Q4_0 with llama.cpp b10964 and the LFM tool protocol.
    case lfm2_5InstructQ4_0

    func validate(modelURL: URL) throws {
        guard self != .textOnly else { return }
        let file = try FileHandle(forReadingFrom: modelURL)
        defer { try? file.close() }
        var hash = SHA256()
        while let data = try file.read(upToCount: 1_048_576), !data.isEmpty {
            try Task.checkCancellation()
            hash.update(data: data)
        }
        guard hash.finalize().map({ String(format: "%02x", $0) }).joined() ==
                "bb741ebb106d543e9de114b843a3d3d73d51c74b5801e69da2abde821a0cb3e1" else {
            throw LlamaToolError.unverifiedModel
        }
    }
}

@available(iOS 27.0, macOS 27.0, *)
enum LlamaToolError: Error, Equatable {
    case unverifiedModel
    case malformedCall
    case unknownTool(String)
    case invalidArguments(String)
    case unsupportedToolMode
}

@available(iOS 27.0, macOS 27.0, *)
struct LlamaParsedToolCall: Sendable {
    let name: String
    let arguments: String
}

/// Buffers control syntax until the whole call batch is valid, never executing partial calls.
@available(iOS 27.0, macOS 27.0, *)
struct LlamaToolParser {
    static let start = "<|tool_call_start|>"
    static let end = "<|tool_call_end|>"
    private var pending = ""
    private var capturesCall = false
    private var reasoningEnd: String?

    mutating func append(_ text: String) throws -> String {
        pending += text
        if capturesCall { return "" }
        var visible = ""
        while !pending.isEmpty {
            let delimiters = reasoningEnd.map { [$0] } ?? [Self.start, Self.end, "<think>", "<|channel>thought\n"]
            let match = delimiters.compactMap { delimiter in
                pending.range(of: delimiter).map { (delimiter, $0) }
            }.min { $0.1.lowerBound < $1.1.lowerBound }
            if let (delimiter, range) = match {
                visible += pending[..<range.lowerBound]
                pending = String(pending[range.upperBound...])
                if delimiter == Self.start {
                    pending = delimiter + pending
                    capturesCall = true
                    return visible
                }
                if delimiter == Self.end { throw LlamaToolError.malformedCall }
                visible += delimiter
                reasoningEnd = reasoningEnd != nil ? nil : (delimiter == "<think>" ? "</think>" : "<channel|>")
            } else {
                let held = delimiters.map { delimiter in
                    (1..<delimiter.count).reversed().first { pending.hasSuffix(delimiter.prefix($0)) } ?? 0
                }.max() ?? 0
                visible += pending.dropLast(held)
                pending = String(pending.suffix(held))
                break
            }
        }
        return visible
    }

    mutating func finish(definitions: [Transcript.ToolDefinition]) throws -> (String, [LlamaParsedToolCall]) {
        guard capturesCall else {
            guard pending.isEmpty || reasoningEnd != nil || ![Self.start, Self.end].contains(where: { $0.hasPrefix(pending) }) else { throw LlamaToolError.malformedCall }
            return (pending, [])
        }
        guard pending.hasPrefix(Self.start), let end = pending.range(of: Self.end),
              !pending[end.upperBound...].contains(Self.start), !pending[end.upperBound...].contains(Self.end) else {
            throw LlamaToolError.malformedCall
        }
        let body = String(pending[pending.index(pending.startIndex, offsetBy: Self.start.count)..<end.lowerBound])
        var reader = LlamaCallReader(body)
        let calls = try reader.calls()
        for call in calls {
            guard let definition = definitions.first(where: { $0.name == call.name }) else {
                throw LlamaToolError.unknownTool(call.name)
            }
            let schema = try JSONSerialization.jsonObject(with: JSONEncoder().encode(definition.parameters))
            let value = try JSONSerialization.jsonObject(with: Data(call.arguments.utf8))
            guard LlamaToolSchema.validate(value, schema: schema, root: schema) else {
                throw LlamaToolError.invalidArguments(call.name)
            }
        }
        return (String(pending[end.upperBound...]), calls)
    }
}

@available(iOS 27.0, macOS 27.0, *)
private struct LlamaCallReader {
    private let input: [Character]
    private var index = 0
    init(_ text: String) { input = Array(text) }
    private mutating func spaces() { while index < input.count && input[index].isWhitespace { index += 1 } }
    private mutating func take(_ char: Character) -> Bool {
        spaces()
        guard index < input.count, input[index] == char else { return false }
        index += 1
        return true
    }
    private mutating func require(_ char: Character) throws { guard take(char) else { throw LlamaToolError.malformedCall } }
    private mutating func name() throws -> String {
        spaces()
        let start = index
        while index < input.count, input[index].isASCII, input[index].isLetter || input[index].isNumber || input[index] == "_" { index += 1 }
        let value = String(input[start..<index])
        guard let first = value.first, first.isLetter || first == "_" else { throw LlamaToolError.malformedCall }
        return value
    }
    mutating func calls() throws -> [LlamaParsedToolCall] {
        try require("[")
        var calls: [LlamaParsedToolCall] = []
        repeat {
            let tool = try name()
            try require("(")
            var args: [String: Any] = [:]
            if !take(")") {
                repeat {
                    let key = try name()
                    guard args[key] == nil else { throw LlamaToolError.malformedCall }
                    try require("=")
                    args[key] = try value(depth: 0)
                } while take(",")
                try require(")")
            }
            let data = try JSONSerialization.data(withJSONObject: args, options: [.sortedKeys, .fragmentsAllowed])
            calls.append(.init(name: tool, arguments: String(decoding: data, as: UTF8.self)))
        } while take(",")
        try require("]")
        spaces()
        guard index == input.count else { throw LlamaToolError.malformedCall }
        return calls
    }
    private mutating func value(depth: Int) throws -> Any {
        spaces()
        guard depth < 64, index < input.count else { throw LlamaToolError.malformedCall }
        if input[index] == "'" || input[index] == "\"" { return try string() }
        if take("[") {
            var values: [Any] = []
            if take("]") { return values }
            repeat { values.append(try value(depth: depth + 1)) } while take(",")
            try require("]")
            return values
        }
        if take("{") {
            var values: [String: Any] = [:]
            if take("}") { return values }
            repeat {
                spaces()
                let key = try string()
                guard values[key] == nil else { throw LlamaToolError.malformedCall }
                try require(":")
                values[key] = try value(depth: depth + 1)
            } while take(",")
            try require("}")
            return values
        }
        let start = index
        while index < input.count, !input[index].isWhitespace, ![",", ")", "]", "}"].contains(input[index]) { index += 1 }
        let literal = String(input[start..<index])
        switch literal {
        case "True", "true": return true
        case "False", "false": return false
        case "None", "null": return NSNull()
        default:
            guard let number = try? JSONSerialization.jsonObject(with: Data(literal.utf8), options: .fragmentsAllowed), number is NSNumber else {
                throw LlamaToolError.malformedCall
            }
            return number
        }
    }
    private mutating func string() throws -> String {
        guard index < input.count, input[index] == "'" || input[index] == "\"" else { throw LlamaToolError.malformedCall }
        let quote = input[index]
        index += 1
        var output = ""
        while index < input.count {
            let char = input[index]; index += 1
            if char == quote { return output }
            if char == "\\" {
                guard index < input.count else { throw LlamaToolError.malformedCall }
                let escaped = input[index]; index += 1
                switch escaped {
                case "\\", "'", "\"", "/": output.append(escaped)
                case "n": output.append("\n")
                case "r": output.append("\r")
                case "t": output.append("\t")
                case "b": output.append("\u{08}")
                case "f": output.append("\u{0C}")
                case "u":
                    guard index + 4 <= input.count else { throw LlamaToolError.malformedCall }
                    var sequence = "\\u" + String(input[index..<index+4]); index += 4
                    if index + 6 <= input.count, input[index] == "\\", input[index+1] == "u" {
                        sequence += String(input[index..<index+6]); index += 6
                    }
                    guard let decoded = try? JSONDecoder().decode(String.self, from: Data(("\"" + sequence + "\"").utf8)) else { throw LlamaToolError.malformedCall }
                    output += decoded
                default: throw LlamaToolError.malformedCall
                }
            } else {
                guard char != "\n", char != "\r" else { throw LlamaToolError.malformedCall }
                output.append(char)
            }
        }
        throw LlamaToolError.malformedCall
    }
}

@available(iOS 27.0, macOS 27.0, *)
enum LlamaToolSchema {
    static func validate(_ value: Any, schema: Any, root: Any, depth: Int = 0) -> Bool {
        guard depth < 64, let schema = schema as? [String: Any] else { return false }
        if let ref = schema["$ref"] as? String {
            guard ref.hasPrefix("#/") else { return false }
            var resolved: Any = root
            for key in ref.dropFirst(2).split(separator: "/") {
                guard let next = (resolved as? [String: Any])?[String(key).replacingOccurrences(of: "~1", with: "/").replacingOccurrences(of: "~0", with: "~")] else { return false }
                resolved = next
            }
            return validate(value, schema: resolved, root: root, depth: depth + 1)
        }
        if let choices = schema["anyOf"] as? [Any] { return choices.contains { validate(value, schema: $0, root: root, depth: depth + 1) } }
        if let values = schema["enum"] as? [Any], !values.contains(where: { ($0 as? NSObject)?.isEqual(value) == true }) { return false }
        switch schema["type"] as? String {
        case "object":
            guard let object = value as? [String: Any], let properties = schema["properties"] as? [String: Any] else { return false }
            guard (schema["required"] as? [String] ?? []).allSatisfy({ object[$0] != nil }) else { return false }
            return object.allSatisfy { key, value in
                guard let property = properties[key] else { return schema["additionalProperties"] as? Bool != false }
                return validate(value, schema: property, root: root, depth: depth + 1)
            }
        case "array":
            guard let array = value as? [Any], let item = schema["items"] else { return false }
            if let min = schema["minItems"] as? Int, array.count < min { return false }
            if let max = schema["maxItems"] as? Int, array.count > max { return false }
            return array.allSatisfy { validate($0, schema: item, root: root, depth: depth + 1) }
        case "string":
            guard let text = value as? String else { return false }
            if let min = schema["minLength"] as? Int, text.unicodeScalars.count < min { return false }
            if let max = schema["maxLength"] as? Int, text.unicodeScalars.count > max { return false }
            if let pattern = schema["pattern"] as? String {
                guard let regex = try? NSRegularExpression(pattern: pattern), regex.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)) != nil else { return false }
            }
            return true
        case "boolean": return (value as? NSNumber).map { CFGetTypeID($0) == CFBooleanGetTypeID() } ?? false
        case "null": return value is NSNull
        case "integer", "number":
            guard let n = value as? NSNumber, CFGetTypeID(n) != CFBooleanGetTypeID(), n.doubleValue.isFinite else { return false }
            if schema["type"] as? String == "integer", n.doubleValue.rounded() != n.doubleValue { return false }
            if let min = schema["minimum"] as? Double, n.doubleValue < min { return false }
            if let max = schema["maximum"] as? Double, n.doubleValue > max { return false }
            return true
        default: return false
        }
    }
}
#endif
