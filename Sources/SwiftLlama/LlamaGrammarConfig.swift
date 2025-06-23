//
//  LlamaGrammarConfig.swift
//  LlamaSwift
//
//  Created by Piotr Gorzelany on 08/02/2025.
//

/// Configuration for grammar-based sampling.
///
/// A grammar allows you to constrain the model's output to a specific format.
/// You define the format using a set of rules in GBNF (GGML's BNF-like) format.
///
/// **Example: JSON Output**
///
/// To force the model to output a simple JSON object, you can use the following grammar:
///
/// ```
/// let jsonGrammar = """
/// root   ::= object
/// value  ::= object | array | string | number | "true" | "false" | "null"
/// object ::= "{" ws ( string ws ":" ws value ("," ws string ws ":" ws value)* )? ws "}"
/// array  ::= "[" ws ( value ("," ws value)* )? ws "]"
/// string ::= "\"" (
///   [^"\\\u0000-\u001f] |
///   "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4})
/// )* "\"" ws
/// number ::= ("-")? ([0-9] | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws
/// ws     ::= ([ \t\n] ws)?
/// """
///
/// let grammarConfig = LlamaGrammarConfig(
///     grammar: jsonGrammar,
///     grammarRoot: "root" // "root" is the starting rule of our grammar
/// )
/// ```
///
public struct LlamaGrammarConfig: Equatable, Sendable {
    /// The grammar rules in GBNF format.
    public let grammar: String

    /// The name of the starting rule in the grammar.
    /// By convention, this is often "root", but it can be any rule name you define.
    public let grammarRoot: String

    public init(grammar: String, grammarRoot: String = "root") {
        self.grammar = grammar
        self.grammarRoot = grammarRoot
    }
} 