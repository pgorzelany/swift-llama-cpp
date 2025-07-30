//
//  LlamaSampler.swift
//  LlamaSwift
//
//  Created by Piotr Gorzelany on 26/09/2024.
//


import Foundation
import llama

/// A wrapper for the `llama.cpp` sampling chain (`llama_sampler_chain`).
///
/// This class configures and manages a series of samplers to control the token generation process.
/// The chain can include samplers for grammar enforcement, temperature, top-k, top-p, and more.
public final class LlamaSampler {
    private let samplerPointer: UnsafeMutablePointer<llama_sampler>

    /// Initializes a new sampling chain based on the provided configuration.
    ///
    /// The sampler chain is built in a specific order to ensure correctness.
    /// If a grammar is provided in the `config`, it is always added first to constrain the possible tokens early.
    /// Other samplers like top-k, top-p, and temperature are added afterward. The chain always ends with a
    /// distribution sampler to make the final selection.
    ///
    /// - Parameters:
    ///   - config: The `LlamaSamplingConfig` that defines which samplers to use and their parameters.
    ///   - model: The `LlamaModel` is required to access the vocabulary for the grammar sampler.
    public init(config: LlamaSamplingConfig, model: LlamaModel) {
        print(config)
        let sparams = llama_sampler_chain_default_params()
        self.samplerPointer = llama_sampler_chain_init(sparams)

        if let grammarConfig = config.grammarConfig {
            if let grammarSampler = llama_sampler_init_grammar(model.vocabPointer, grammarConfig.grammar, grammarConfig.grammarRoot) {
                llama_sampler_chain_add(samplerPointer, grammarSampler)
            } else {
                #warning("Throw an error instead")
                print("Failed to initialize grammar sampler with grammar: \n\n\(grammarConfig.grammar)")
            }
        }

        // Add samplers based on the configuration
        if let topK = config.topK {
            let topKSampler = llama_sampler_init_top_k(topK)
            llama_sampler_chain_add(samplerPointer, topKSampler)
        }

        let topPSampler = llama_sampler_init_top_p(config.topP, config.minKeep)
        llama_sampler_chain_add(samplerPointer, topPSampler)

        if let penaltyConfig = config.repetitionPenaltyConfig, penaltyConfig.lastN > 0 {
            let penaltiesSampler = llama_sampler_init_penalties(
                penaltyConfig.lastN,
                penaltyConfig.repeatPenalty,
                penaltyConfig.freqPenalty,
                penaltyConfig.presentPenalty
            )
            llama_sampler_chain_add(samplerPointer, penaltiesSampler)
        }

        // Always add temperature sampler
        let tempSampler = llama_sampler_init_temp(config.temperature)
        llama_sampler_chain_add(samplerPointer, tempSampler)

        #warning("The seed doesn't seem to have any effect, the temperature defines the outcome.")
        // temperature of 0 gives predictable results, temperature of more than 0 gives randomized results
        // It seems the issue is only when I clear the context, something is not reset there.
        // If I run the app from scratch with the same params I get reproducible results
        let seed = config.seed
        print("### Seed: \(seed)")
        let distSampler = llama_sampler_init_dist(seed)
        llama_sampler_chain_add(samplerPointer, distSampler)
    }

    deinit {
        llama_sampler_free(samplerPointer)
    }

    /// Samples a token from the model's output and implicitly accepts it.
    ///
    /// This is the primary method for token generation. It wraps the `llama_sampler_sample` C function, which:
    /// 1. Applies the full sampler chain (grammar, top-k, temperature, etc.) to the logits.
    /// 2. Selects a token.
    /// 3. Automatically accepts the token, which updates the internal state of all samplers in the chain (e.g., advancing the grammar parser).
    ///
    /// - Parameters:
    ///   - context: The current `LlamaContext`.
    /// - Returns: The sampled `llama_token`.
    public func sample(context: LlamaContext) -> llama_token {
        #warning("Sampler usage example: https://github.com/ggerganov/llama.cpp/blob/564804b79b78df1469ec8646869972de5e885ec4/include/llama.h#L1065")
        return llama_sampler_sample(samplerPointer, context.contextPointer, -1)
    }

    /// Manually accepts a token to update the state of the samplers in the chain.
    ///
    /// This method is primarily used to initialize the state of the samplers before generation begins.
    /// For example, when using a grammar, you should call this method for each token in your prompt to ensure the
    /// grammar state is correctly synchronized with the prompt's content.
    ///
    /// For the main generation loop, the `sample(context:lastTokenIndex:)` method should be used instead, as it handles acceptance automatically.
    ///
    /// - Parameter token: The `llama_token` to accept.
    public func accept(token: llama_token) {
        llama_sampler_accept(samplerPointer, token)
    }
}
