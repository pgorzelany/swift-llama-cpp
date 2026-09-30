import Foundation
import Testing
import llama
@testable import SwiftLlama

@Suite("Native GBNF regression contracts", .serialized)
struct LlamaGrammarRegressionTests {
    private struct Person: Codable {
        let age: Int
        let city: String?
        let name: String
    }

    private struct UnusualKeys: Codable {
        let first: String
        let second: String
        let unicode: String
        enum CodingKeys: String, CodingKey {
            case first = "a_b"
            case second = "a-b"
            case unicode = "imię"
        }
    }

    private func accepts<T: Codable>(_ text: String, as type: T.Type) throws -> Bool {
        var params = llama_model_default_params()
        params.vocab_only = true
        let model = try #require(LlamaModel(path: URL.llama1B.path, parameters: params))
        let grammar = try LlamaTypedJSONGrammarBuilder.makeGrammarConfig(for: type)
        let sampler = try #require(llama_sampler_init_grammar(model.vocabPointer, grammar.grammar, grammar.grammarRoot))
        defer { llama_sampler_free(sampler) }
        let tokens = model.tokenize(text: text, addBos: false, special: false) + [model.eosToken()]
        for token in tokens {
            var candidate = llama_token_data(id: token, logit: 0, p: 0)
            let allowed = withUnsafeMutablePointer(to: &candidate) { pointer in
                var candidates = llama_token_data_array(data: pointer, size: 1, selected: -1, sorted: false)
                llama_sampler_apply(sampler, &candidates)
                return pointer.pointee.logit.isFinite
            }
            if !allowed { return false }
            llama_sampler_accept(sampler, token)
        }
        return true
    }

    @Test("Native parser requires quoted, unique keys and required fields")
    func objectContract() throws {
        #expect(try accepts(#"{"age":30,"city":null,"name":"A {quoted} \"name\""}"#, as: Person.self))
        for invalid in [#"{}"#, #"{age:30,city:null,name:"A"}"#,
                        #"{"age":30,"city":null}"#, #"{"age":30,"age":31,"city":null,"name":"A"}"#,
                        #"{"age":30,"city":null,"name":"A","extra":1}"#] {
            #expect(try !accepts(invalid, as: Person.self))
        }
    }

    @Test("Distinct punctuation and Unicode coding keys do not collide")
    func keyNames() throws {
        #expect(try accepts(#"{"a-b":"second","a_b":"first","imię":"Zażółć"}"#, as: UnusualKeys.self))
        #expect(try !accepts(#"{"a-b":"second","a-b":"first","imię":"Zażółć"}"#, as: UnusualKeys.self))
    }

    @Test("Recursive schema fails before overflowing the stack")
    func recursion() {
        struct Recursive: Codable { let children: [Recursive] }
        #expect(throws: (any Error).self) {
            _ = try LlamaTypedJSONGrammarBuilder.makeGrammarConfig(for: Recursive.self)
        }
    }
}
