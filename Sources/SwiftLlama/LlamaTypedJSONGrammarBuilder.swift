//
//  LlamaTypedJSONGrammarBuilder.swift
//  SwiftLlama
//
//  Creates a GBNF grammar that constrains the model to emit JSON
//  matching the shape of a given Decodable type (best-effort).
//

import Foundation

// MARK: - Public API

enum LlamaTypedJSONGrammarBuilder {

    /// Build a JSON grammar for the provided `Decodable` type.
    ///
    /// Notes and limitations:
    /// - Enforces allowed keys and value types.
    /// - Does not strictly enforce presence of required keys (to avoid combinatorial explosion).
    ///   Optional properties allow `null` in addition to the underlying value type.
    /// - Dictionaries are not currently supported; prefer concrete structs.
    static func makeGrammarConfig<T: Decodable>(for type: T.Type) throws -> LlamaGrammarConfig {
        let recorder = SchemaRecorder()
        let rootSlot = SchemaSlot()
        let decoder = RecordingDecoder(recorder: recorder, target: rootSlot)
        _ = try? T(from: decoder)
        guard let rootNode = rootSlot.node else {
            throw TypedGrammarError.unsupported("Could not infer schema for type: \(String(describing: T.self))")
        }
        let generator = GrammarGenerator()
        let grammar = generator.generateGrammar(for: rootNode)
        return LlamaGrammarConfig(grammar: grammar, grammarRoot: "root")
    }
}

// MARK: - Errors

enum TypedGrammarError: Error, LocalizedError {
    case unsupported(String)
    var errorDescription: String? {
        switch self {
        case .unsupported(let msg): return msg
        }
    }
}

// MARK: - Schema model

final class SchemaSlot {
    var node: SchemaNode?
}

final class SchemaNode {
    enum Kind {
        case string
        case integer
        case number
        case boolean
        case null
        case array(element: SchemaNode)
        case object(required: [String: SchemaNode], optional: [String: SchemaNode])
    }

    let kind: Kind

    init(kind: Kind) { self.kind = kind }
}

final class SchemaRecorder {
    func registerPrimitiveString(into slot: SchemaSlot) {
        if slot.node == nil { slot.node = SchemaNode(kind: .string) }
    }
    func registerPrimitiveInteger(into slot: SchemaSlot) {
        if slot.node == nil { slot.node = SchemaNode(kind: .integer) }
    }
    func registerPrimitiveNumber(into slot: SchemaSlot) {
        if slot.node == nil { slot.node = SchemaNode(kind: .number) }
    }
    func registerPrimitiveBoolean(into slot: SchemaSlot) {
        if slot.node == nil { slot.node = SchemaNode(kind: .boolean) }
    }
    func registerNull(into slot: SchemaSlot) {
        if slot.node == nil { slot.node = SchemaNode(kind: .null) }
    }
    func beginArray(into slot: SchemaSlot) -> SchemaSlot {
        if case .array(let element)? = slot.node?.kind {
            // Already begun; return element slot equivalent
            let childSlot = SchemaSlot()
            childSlot.node = element
            return childSlot
        }
        let childSlot = SchemaSlot()
        // Temporarily create placeholder; element will be filled via returned slot
        let placeholder = SchemaNode(kind: .array(element: SchemaNode(kind: .null)))
        slot.node = placeholder
        // Replace element reference once filled in generation step
        // We will update in finalizeArray
        // For now, stash childSlot on user of this API
        return childSlot
    }
    func finalizeArray(parent slot: SchemaSlot, element: SchemaSlot) {
        let elementNode = element.node ?? SchemaNode(kind: .null)
        slot.node = SchemaNode(kind: .array(element: elementNode))
    }
    func beginObject(into slot: SchemaSlot) -> ObjectBuilder {
        if case .object(let req, let opt)? = slot.node?.kind {
            return ObjectBuilder(recorder: self, target: slot, required: req, optional: opt)
        }
        slot.node = SchemaNode(kind: .object(required: [:], optional: [:]))
        return ObjectBuilder(recorder: self, target: slot, required: [:], optional: [:])
    }
    func setObject(required: [String: SchemaNode], optional: [String: SchemaNode], into slot: SchemaSlot) {
        slot.node = SchemaNode(kind: .object(required: required, optional: optional))
    }
}

final class ObjectBuilder {
    private let recorder: SchemaRecorder
    private let target: SchemaSlot
    private var required: [String: SchemaNode]
    private var optional: [String: SchemaNode]

    init(recorder: SchemaRecorder, target: SchemaSlot, required: [String: SchemaNode], optional: [String: SchemaNode]) {
        self.recorder = recorder
        self.target = target
        self.required = required
        self.optional = optional
    }

    func addRequired(key: String, build: (SchemaSlot) throws -> Void) rethrows {
        let slot = SchemaSlot()
        try build(slot)
        required[key] = slot.node ?? SchemaNode(kind: .null)
        recorder.setObject(required: required, optional: optional, into: target)
    }

    func addOptional(key: String, build: (SchemaSlot) throws -> Void) rethrows {
        let slot = SchemaSlot()
        try build(slot)
        let child = slot.node ?? SchemaNode(kind: .null)
        // Optional allows null as well
        let optionalNode: SchemaNode
        switch child.kind {
        case .null:
            optionalNode = child
        default:
            // Represent optional by allowing either value or null at value-position during grammar generation
            // We'll encode this later by inlining `| "null"` on the value position.
            optionalNode = child
        }
        optional[key] = optionalNode
        recorder.setObject(required: required, optional: optional, into: target)
    }
}

// MARK: - Recording Decoder

final class RecordingDecoder: Decoder {
    let recorder: SchemaRecorder
    let target: SchemaSlot

    init(recorder: SchemaRecorder, target: SchemaSlot) {
        self.recorder = recorder
        self.target = target
    }

    var codingPath: [CodingKey] { [] }
    var userInfo: [CodingUserInfoKey : Any] { [:] }

    func container<Key>(keyedBy type: Key.Type) throws -> KeyedDecodingContainer<Key> where Key : CodingKey {
        let builder = recorder.beginObject(into: target)
        let container = RecordingKeyedContainer<Key>(recorder: recorder, object: builder)
        return KeyedDecodingContainer(container)
    }

    func unkeyedContainer() throws -> UnkeyedDecodingContainer {
        // Start array with one synthetic element to capture element type
        let elementSlot = recorder.beginArray(into: target)
        return RecordingUnkeyedContainer(recorder: recorder, parent: target, elementSlot: elementSlot)
    }

    func singleValueContainer() throws -> SingleValueDecodingContainer {
        return RecordingSingleValueContainer(recorder: recorder, target: target)
    }
}

// MARK: Keyed Container

struct RecordingKeyedContainer<Key: CodingKey>: KeyedDecodingContainerProtocol {
    let recorder: SchemaRecorder
    let object: ObjectBuilder

    var codingPath: [CodingKey] { [] }
    var allKeys: [Key] { [] }

    func contains(_ key: Key) -> Bool { true }

    func decodeNil(forKey key: Key) throws -> Bool { false }

    // Primitive decoders
    func decode(_ type: String.Type, forKey key: Key) throws -> String {
        try object.addRequired(key: key.stringValue) { slot in
            recorder.registerPrimitiveString(into: slot)
        }
        return ""
    }
    func decode(_ type: Bool.Type, forKey key: Key) throws -> Bool {
        try object.addRequired(key: key.stringValue) { slot in
            recorder.registerPrimitiveBoolean(into: slot)
        }
        return false
    }
    func decode(_ type: Int.Type, forKey key: Key) throws -> Int {
        try object.addRequired(key: key.stringValue) { slot in
            recorder.registerPrimitiveInteger(into: slot)
        }
        return 0
    }
    func decode(_ type: Int8.Type, forKey key: Key) throws -> Int8 { Int8(try decode(Int.self, forKey: key)) }
    func decode(_ type: Int16.Type, forKey key: Key) throws -> Int16 { Int16(try decode(Int.self, forKey: key)) }
    func decode(_ type: Int32.Type, forKey key: Key) throws -> Int32 { Int32(try decode(Int.self, forKey: key)) }
    func decode(_ type: Int64.Type, forKey key: Key) throws -> Int64 { Int64(try decode(Int.self, forKey: key)) }
    func decode(_ type: UInt.Type, forKey key: Key) throws -> UInt { UInt(try decode(Int.self, forKey: key)) }
    func decode(_ type: UInt8.Type, forKey key: Key) throws -> UInt8 { UInt8(try decode(Int.self, forKey: key)) }
    func decode(_ type: UInt16.Type, forKey key: Key) throws -> UInt16 { UInt16(try decode(Int.self, forKey: key)) }
    func decode(_ type: UInt32.Type, forKey key: Key) throws -> UInt32 { UInt32(try decode(Int.self, forKey: key)) }
    func decode(_ type: UInt64.Type, forKey key: Key) throws -> UInt64 { UInt64(try decode(Int.self, forKey: key)) }
    func decode(_ type: Float.Type, forKey key: Key) throws -> Float {
        try object.addRequired(key: key.stringValue) { slot in
            recorder.registerPrimitiveNumber(into: slot)
        }
        return 0
    }
    func decode(_ type: Double.Type, forKey key: Key) throws -> Double {
        try object.addRequired(key: key.stringValue) { slot in
            recorder.registerPrimitiveNumber(into: slot)
        }
        return 0
    }

    // Nested / generic
    func decode<T>(_ type: T.Type, forKey key: Key) throws -> T where T : Decodable {
        // Arrays are handled by nested decoding path as well
        var result: T?
        try object.addRequired(key: key.stringValue) { slot in
            let nestedDecoder = RecordingDecoder(recorder: recorder, target: slot)
            result = try? T(from: nestedDecoder)
        }
        // If T is a primitive or simple type and result is still nil, try to materialize via JSONDecoder as a fallback
        if let value = result { return value }
        // Fallbacks
        if T.self == String.self { return ("" as! T) }
        if T.self == Bool.self { return (false as! T) }
        if T.self == Int.self { return (0 as! T) }
        if T.self == Double.self { return (0.0 as! T) }
        if T.self == Float.self { return (0 as! T) }
        if T.self == [String].self { return ([] as! T) }
        // Last resort empty JSON value decoding for collections
        if let arr = try? JSONDecoder().decode(T.self, from: Data("[]".utf8)) { return arr }
        if let obj = try? JSONDecoder().decode(T.self, from: Data("{}".utf8)) { return obj }
        // Best effort
        return try T(from: RecordingDecoder(recorder: recorder, target: SchemaSlot()))
    }

    func decodeIfPresent(_ type: String.Type, forKey key: Key) throws -> String? {
        try object.addOptional(key: key.stringValue) { slot in
            recorder.registerPrimitiveString(into: slot)
        }
        return nil
    }
    func decodeIfPresent(_ type: Bool.Type, forKey key: Key) throws -> Bool? {
        try object.addOptional(key: key.stringValue) { slot in
            recorder.registerPrimitiveBoolean(into: slot)
        }
        return nil
    }
    func decodeIfPresent(_ type: Int.Type, forKey key: Key) throws -> Int? {
        try object.addOptional(key: key.stringValue) { slot in
            recorder.registerPrimitiveInteger(into: slot)
        }
        return nil
    }
    func decodeIfPresent(_ type: Double.Type, forKey key: Key) throws -> Double? {
        try object.addOptional(key: key.stringValue) { slot in
            recorder.registerPrimitiveNumber(into: slot)
        }
        return nil
    }
    func decodeIfPresent(_ type: Float.Type, forKey key: Key) throws -> Float? {
        try object.addOptional(key: key.stringValue) { slot in
            recorder.registerPrimitiveNumber(into: slot)
        }
        return nil
    }
    func decodeIfPresent<T>(_ type: T.Type, forKey key: Key) throws -> T? where T : Decodable {
        var value: T?
        try object.addOptional(key: key.stringValue) { slot in
            let nestedDecoder = RecordingDecoder(recorder: recorder, target: slot)
            value = try? T(from: nestedDecoder)
        }
        return value
    }

    func nestedContainer<NestedKey>(keyedBy type: NestedKey.Type, forKey key: Key) throws -> KeyedDecodingContainer<NestedKey> where NestedKey : CodingKey {
        let builder = recorder.beginObject(into: SchemaSlot())
        return KeyedDecodingContainer(RecordingKeyedContainer<NestedKey>(recorder: recorder, object: builder))
    }
    func nestedUnkeyedContainer(forKey key: Key) throws -> UnkeyedDecodingContainer {
        return RecordingUnkeyedContainer(recorder: recorder, parent: SchemaSlot(), elementSlot: SchemaSlot())
    }
    func superDecoder() throws -> Decoder { RecordingDecoder(recorder: recorder, target: SchemaSlot()) }
    func superDecoder(forKey key: Key) throws -> Decoder { RecordingDecoder(recorder: recorder, target: SchemaSlot()) }
}

// MARK: Unkeyed Container (arrays)

final class RecordingUnkeyedContainer: UnkeyedDecodingContainer {
    let recorder: SchemaRecorder
    let parent: SchemaSlot
    let elementSlot: SchemaSlot
    var index: Int = 0

    init(recorder: SchemaRecorder, parent: SchemaSlot, elementSlot: SchemaSlot) {
        self.recorder = recorder
        self.parent = parent
        self.elementSlot = elementSlot
    }

    var codingPath: [CodingKey] { [] }
    var count: Int? { 1 }
    var isAtEnd: Bool { index >= 1 }
    var currentIndex: Int { index }

    func decodeNil() throws -> Bool { false }

    func decode(_ type: String.Type) throws -> String { index += 1; recorder.registerPrimitiveString(into: elementSlot); recorder.finalizeArray(parent: parent, element: elementSlot); return "" }
    func decode(_ type: Bool.Type) throws -> Bool { index += 1; recorder.registerPrimitiveBoolean(into: elementSlot); recorder.finalizeArray(parent: parent, element: elementSlot); return false }
    func decode(_ type: Int.Type) throws -> Int { index += 1; recorder.registerPrimitiveInteger(into: elementSlot); recorder.finalizeArray(parent: parent, element: elementSlot); return 0 }
    func decode(_ type: Int8.Type) throws -> Int8 { Int8(try decode(Int.self)) }
    func decode(_ type: Int16.Type) throws -> Int16 { Int16(try decode(Int.self)) }
    func decode(_ type: Int32.Type) throws -> Int32 { Int32(try decode(Int.self)) }
    func decode(_ type: Int64.Type) throws -> Int64 { Int64(try decode(Int.self)) }
    func decode(_ type: UInt.Type) throws -> UInt { UInt(try decode(Int.self)) }
    func decode(_ type: UInt8.Type) throws -> UInt8 { UInt8(try decode(Int.self)) }
    func decode(_ type: UInt16.Type) throws -> UInt16 { UInt16(try decode(Int.self)) }
    func decode(_ type: UInt32.Type) throws -> UInt32 { UInt32(try decode(Int.self)) }
    func decode(_ type: UInt64.Type) throws -> UInt64 { UInt64(try decode(Int.self)) }
    func decode(_ type: Float.Type) throws -> Float { index += 1; recorder.registerPrimitiveNumber(into: elementSlot); recorder.finalizeArray(parent: parent, element: elementSlot); return 0 }
    func decode(_ type: Double.Type) throws -> Double { index += 1; recorder.registerPrimitiveNumber(into: elementSlot); recorder.finalizeArray(parent: parent, element: elementSlot); return 0 }

    func decode<T>(_ type: T.Type) throws -> T where T : Decodable {
        index += 1
        let nested = RecordingDecoder(recorder: recorder, target: elementSlot)
        let value = try T(from: nested)
        recorder.finalizeArray(parent: parent, element: elementSlot)
        return value
    }

    func nestedContainer<NestedKey>(keyedBy type: NestedKey.Type) throws -> KeyedDecodingContainer<NestedKey> where NestedKey : CodingKey {
        index += 1
        let builder = recorder.beginObject(into: elementSlot)
        recorder.finalizeArray(parent: parent, element: elementSlot)
        return KeyedDecodingContainer(RecordingKeyedContainer<NestedKey>(recorder: recorder, object: builder))
    }

    func nestedUnkeyedContainer() throws -> UnkeyedDecodingContainer {
        index += 1
        let inner = SchemaSlot()
        return RecordingUnkeyedContainer(recorder: recorder, parent: elementSlot, elementSlot: inner)
    }

    func superDecoder() throws -> Decoder { RecordingDecoder(recorder: recorder, target: SchemaSlot()) }
}

// MARK: Single Value Container

struct RecordingSingleValueContainer: SingleValueDecodingContainer {
    let recorder: SchemaRecorder
    let target: SchemaSlot

    var codingPath: [CodingKey] { [] }

    func decodeNil() -> Bool { false }
    func decode(_ type: String.Type) throws -> String { recorder.registerPrimitiveString(into: target); return "" }
    func decode(_ type: Bool.Type) throws -> Bool { recorder.registerPrimitiveBoolean(into: target); return false }
    func decode(_ type: Int.Type) throws -> Int { recorder.registerPrimitiveInteger(into: target); return 0 }
    func decode(_ type: Int8.Type) throws -> Int8 { Int8(try decode(Int.self)) }
    func decode(_ type: Int16.Type) throws -> Int16 { Int16(try decode(Int.self)) }
    func decode(_ type: Int32.Type) throws -> Int32 { Int32(try decode(Int.self)) }
    func decode(_ type: Int64.Type) throws -> Int64 { Int64(try decode(Int.self)) }
    func decode(_ type: UInt.Type) throws -> UInt { UInt(try decode(Int.self)) }
    func decode(_ type: UInt8.Type) throws -> UInt8 { UInt8(try decode(Int.self)) }
    func decode(_ type: UInt16.Type) throws -> UInt16 { UInt16(try decode(Int.self)) }
    func decode(_ type: UInt32.Type) throws -> UInt32 { UInt32(try decode(Int.self)) }
    func decode(_ type: UInt64.Type) throws -> UInt64 { UInt64(try decode(Int.self)) }
    func decode(_ type: Float.Type) throws -> Float { recorder.registerPrimitiveNumber(into: target); return 0 }
    func decode(_ type: Double.Type) throws -> Double { recorder.registerPrimitiveNumber(into: target); return 0 }
    func decode<T>(_ type: T.Type) throws -> T where T : Decodable {
        let nested = RecordingDecoder(recorder: recorder, target: target)
        return try T(from: nested)
    }
}

// MARK: - Grammar Generation

final class GrammarGenerator {
    struct RuleRef: Hashable { let id: ObjectIdentifier }

    func generateGrammar(for root: SchemaNode) -> String {
        var rules: [String] = []
        var nameMap: [ObjectIdentifier: String] = [:]
        let rootName = emitRule(for: root, preferredName: "root_value", rules: &rules, names: &nameMap)
        let prelude = baseRules()
        let body = rules.joined(separator: "\n")
        let rootLine = "root ::= \(rootName)\n"
        // Place the root rule first, as expected by llama.cpp grammar parser
        return prelude + "\n" + rootLine + body + "\n"
    }

    private func emitRule(for node: SchemaNode, preferredName: String, rules: inout [String], names: inout [ObjectIdentifier: String]) -> String {
        let key = ObjectIdentifier(node)
        if let name = names[key] { return name }
        let name: String
        switch node.kind {
        case .string:
            name = preferredName
            names[key] = name
            rules.append("\(name) ::= string")
        case .integer:
            name = preferredName
            names[key] = name
            rules.append("\(name) ::= int")
        case .number:
            name = preferredName
            names[key] = name
            rules.append("\(name) ::= number")
        case .boolean:
            name = preferredName
            names[key] = name
            rules.append("\(name) ::= \"true\" | \"false\"")
        case .null:
            name = preferredName
            names[key] = name
            rules.append("\(name) ::= \"null\"")
        case .array(let element):
            name = preferredName.hasPrefix("array_") ? preferredName : "array_\(preferredName)"
            names[key] = name
            let child = emitRule(for: element, preferredName: "elem_\(name)", rules: &rules, names: &names)
            rules.append(#"\#(name) ::= "[" ws ( \#(child) ( ws "," ws \#(child) )* )? ws "]""#)
        case .object(let required, let optional):
            name = preferredName.hasPrefix("object_") ? preferredName : "object_\(preferredName)"
            names[key] = name
            // Build pair alternatives
            var pairRules: [String] = []
            for (k, v) in required {
                let child = emitRule(for: v, preferredName: "val_\(sanitize(k))_\(name)", rules: &rules, names: &names)
                let pairName = "pair_\(sanitize(k))_\(name)"
                rules.append(#"\#(pairName) ::= "\#(escapeJSONStringLiteral(k))" ws ":" ws \#(child)"#)
                pairRules.append(pairName)
            }
            for (k, v) in optional {
                let child = emitRule(for: v, preferredName: "val_\(sanitize(k))_\(name)", rules: &rules, names: &names)
                let pairName = "pair_\(sanitize(k))_\(name)"
                rules.append(#"\#(pairName) ::= "\#(escapeJSONStringLiteral(k))" ws ":" ws ( \#(child) | "null" )"#)
                pairRules.append(pairName)
            }
            if pairRules.isEmpty {
                rules.append(#"\#(name) ::= "{" ws "}""#)
            } else {
                let memberName = "member_\(name)"
                let memberAlt = pairRules.joined(separator: " | ")
                rules.append("\(memberName) ::= \(memberAlt)")
                rules.append(#"\#(name) ::= "{" ws ( \#(memberName) ( ws "," ws \#(memberName) )* )? ws "}""#)
            }
        }
        return name
    }

    private func baseRules() -> String {
        return #"""
        ws     ::= ([ \t\n] ws)?
        string ::= "\"" (
          [^"\\u0000-\u001f] |
          "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4})
        )* "\""
        int    ::= ("-")? ("0" | [1-9] [0-9]*)
        number ::= ("-")? ( ("0" | [1-9] [0-9]*) ("." [0-9]+)? ) ([eE] [-+]? [0-9]+)?
        """#
    }

    private func escapeJSONStringLiteral(_ s: String) -> String {
        var out = ""
        for ch in s {
            switch ch {
            case "\\": out += "\\\\"
            case "\"": out += "\\\""
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            default: out.append(ch)
            }
        }
        return out
    }

    private func sanitize(_ s: String) -> String {
        let allowed = s.unicodeScalars.map { CharacterSet.alphanumerics.contains($0) ? Character($0) : "_" }
        return String(allowed).replacingOccurrences(of: "__+", with: "_", options: .regularExpression)
    }
}


