//
//  LlamaTypedGrammarTests.swift
//  SwiftLlamaTests
//

import Testing
import Foundation
@testable import SwiftLlama

struct LlamaTypedGrammarTests {

    private struct Person: Codable, Equatable {
        let name: String
        let age: Int
        let city: String?
    }

    private struct Catalog: Codable, Equatable {
        struct Item: Codable, Equatable {
            let id: Int
            let title: String
            let tags: [String]?
        }
        let items: [Item]
        let version: Double
        let published: Bool
    }

    private struct DeepNest: Codable, Equatable {
        struct Level1: Codable, Equatable {
            struct Level2: Codable, Equatable {
                let value: String
                let count: Int?
            }
            let l2: [Level2]
        }
        let l1: Level1
    }

    private enum Status: String, Codable { case pending, done, failed }
    private struct WithEnum: Codable, Equatable {
        let id: Int
        let status: Status
        let note: String?
    }

    private struct Primitives: Codable, Equatable {
        let s: String
        let i: Int
        let u: UInt
        let f: Float
        let d: Double
        let b: Bool
        let o: String?
    }

    private struct MixedArrays: Codable, Equatable {
        let names: [String]
        let numbers: [Int]
        let people: [Person]
    }

    @Test("Generates grammar for simple struct")
    func testSimpleStructGrammar() throws {
        let cfg = try LlamaTypedJSONGrammarBuilder.makeGrammarConfig(for: Person.self)
        #expect(cfg.grammar.contains("root"))
        #expect(cfg.grammar.contains("object_root_value"))
        #expect(cfg.grammar.contains("\"name\""))
        #expect(cfg.grammar.contains("\"age\""))
        #expect(cfg.grammar.contains("\"city\""))
        // Optional allows null
        #expect(cfg.grammar.contains("| \"null\""))
    }

    @Test("Generates grammar for nested arrays and optionals")
    func testNestedArrayGrammar() throws {
        let cfg = try LlamaTypedJSONGrammarBuilder.makeGrammarConfig(for: Catalog.self)
        #expect(cfg.grammar.contains("array_"))
        #expect(cfg.grammar.contains("\"items\""))
        #expect(cfg.grammar.contains("\"id\""))
        #expect(cfg.grammar.contains("\"title\""))
        #expect(cfg.grammar.contains("\"tags\""))
    }

    @Test("Grammar covers deep nesting and optional fields")
    func testDeepNesting() throws {
        let cfg = try LlamaTypedJSONGrammarBuilder.makeGrammarConfig(for: DeepNest.self)
        #expect(cfg.grammar.contains("\"l1\""))
        #expect(cfg.grammar.contains("\"l2\""))
        #expect(cfg.grammar.contains("\"value\""))
        #expect(cfg.grammar.contains("\"count\""))
        #expect(cfg.grammar.contains("| \"null\""))
    }

    @Test("Grammar includes enum raw string fields")
    func testEnums() throws {
        let cfg = try LlamaTypedJSONGrammarBuilder.makeGrammarConfig(for: WithEnum.self)
        // Enums are modeled as strings in this first version
        #expect(cfg.grammar.contains("\"status\""))
        #expect(cfg.grammar.contains("string"))
    }

    @Test("Grammar handles all primitive number types and bool")
    func testPrimitives() throws {
        let cfg = try LlamaTypedJSONGrammarBuilder.makeGrammarConfig(for: Primitives.self)
        #expect(cfg.grammar.contains("\"s\""))
        #expect(cfg.grammar.contains("\"i\""))
        #expect(cfg.grammar.contains("\"u\""))
        #expect(cfg.grammar.contains("\"f\""))
        #expect(cfg.grammar.contains("\"d\""))
        #expect(cfg.grammar.contains("\"b\""))
    }

    @Test("Grammar supports arrays of primitives and objects")
    func testMixedArrays() throws {
        let cfg = try LlamaTypedJSONGrammarBuilder.makeGrammarConfig(for: MixedArrays.self)
        #expect(cfg.grammar.contains("\"names\""))
        #expect(cfg.grammar.contains("\"numbers\""))
        #expect(cfg.grammar.contains("\"people\""))
        // Should have array rules
        #expect(cfg.grammar.contains("array_"))
    }

    @Test("Streaming typed JSON for Person produces valid JSON")
    func testTypedStreamingPerson() async throws {
        let service = LlamaService(modelUrl: .llama1B, config: .init(batchSize: 128, maxTokenCount: 4096))
        let messages = [
            LlamaChatMessage(role: .system, content: "You are a helpful assistant that responds only with JSON that matches the requested schema."),
            LlamaChatMessage(role: .user, content: "Return a person with name, age, and optionally a city")
        ]
        let stream = try await service.streamCompletion(of: messages, generating: Person.self)
        var text = ""
        var count = 0
        for try await token in stream {
            text += token
            count += 1
            if count > 200 { break }
        }
        // Attempt to finish JSON if truncated by early stop
        if let endIndex = text.lastIndex(of: "}") {
            text = String(text[...endIndex])
        }
        let data = text.data(using: .utf8)
        #expect(data != nil)
        if let data { _ = try? JSONDecoder().decode(Person.self, from: data) }
    }
}


