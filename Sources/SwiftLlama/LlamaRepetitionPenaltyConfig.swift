//
//  LlamaRepetitionPenaltyConfig.swift
//  LlamaSwift
//
//  Created by Piotr Gorzelany on 08/02/2025.
//

/// Configuration for repetition penalty, used to prevent the model from generating repetitive text.
///
/// This configuration wraps the `llama_sampler_init_penalties` function and controls how the model penalizes tokens
/// that have recently appeared in the context.
public struct LlamaRepetitionPenaltyConfig: Equatable, Sendable {
    /// The number of recent tokens to consider for penalization.
    /// A value of 0 disables all repetition penalties.
    public let lastN: Int32

    /// The penalty factor for repeating tokens.
    /// A value of 1.0 disables this specific penalty. Higher values increase the penalty.
    public let repeatPenalty: Float

    /// The penalty factor for token frequency.
    /// A value of 0.0 disables this specific penalty. Higher values increase the penalty for frequently occurring tokens.
    public let freqPenalty: Float

    /// The penalty factor for token presence.
    /// A value of 0.0 disables this specific penalty. Higher values increase the penalty for any token that is already present.
    public let presentPenalty: Float

    /// Initializes a new repetition penalty configuration.
    ///
    /// - Parameters:
    ///   - lastN: The number of recent tokens to consider for penalization. Defaults to 64.
    ///   - repeatPenalty: The penalty for repeating tokens. Defaults to 1.1.
    ///   - freqPenalty: The penalty for token frequency. Defaults to 0.0.
    ///   - presentPenalty: The penalty for token presence. Defaults to 0.0.
    public init(
        lastN: Int32 = 64,
        repeatPenalty: Float = 1.1,
        freqPenalty: Float = 0.0,
        presentPenalty: Float = 0.0
    ) {
        self.lastN = lastN
        self.repeatPenalty = repeatPenalty
        self.freqPenalty = freqPenalty
        self.presentPenalty = presentPenalty
    }
} 