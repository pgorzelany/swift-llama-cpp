//
//  LlamaSamplingConfig.swift
//  LlamaSwift
//
//  Created by Piotr Gorzelany on 07/02/2025.
//

public struct LlamaSamplingConfig: Equatable, Sendable {
    public let temperature: CFloat
    public let seed: UInt32
    public let topP: Float
    public let topK: Int32?
    public let minKeep: Int
    /// An optional grammar to constrain the model's output to a specific format.
    /// If nil, no grammar is used.
    public let grammarConfig: LlamaGrammarConfig?

    public init(
        temperature: CFloat,
        seed: UInt32,
        topP: Float = 0.95,
        topK: Int32? = nil,
        minKeep: Int = 1,
        grammarConfig: LlamaGrammarConfig? = nil
    ) {
        self.temperature = temperature
        self.seed = seed
        self.topK = topK
        self.topP = topP
        self.minKeep = minKeep
        self.grammarConfig = grammarConfig
    }
}
