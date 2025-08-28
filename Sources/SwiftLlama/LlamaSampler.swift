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
                // If grammar init fails, we skip adding it to the chain.
                // Consider surfacing this as a thrown error in the initializer signature.
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

        let seed = config.seed
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
    /// Sample a token from the last evaluation (uses idx=-1) and accept it.
    /// - Returns: The sampled token id.
    public func sample(context: LlamaContext) -> llama_token {
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
    /// Manually accept a token to update the internal state of samplers.
    public func accept(token: llama_token) {
        llama_sampler_accept(samplerPointer, token)
    }

    // Chain management helpers
    /// Returns the sampler chain name if available.
    public func name() -> String {
        guard let c = llama_sampler_name(samplerPointer) else { return "" }
        return String(cString: c)
    }

    /// Reset the sampler chain state.
    public func reset() { llama_sampler_reset(samplerPointer) }

    /// Clone the sampler chain.
    public func clone() -> LlamaSampler? {
        guard let cloned = llama_sampler_clone(samplerPointer) else { return nil }
        // Wrap the returned chain pointer in a new Swift object
        // We cannot directly assign to private let, so build via a minimal init
        return LlamaSampler(adopting: cloned)
    }

    /// Internal initializer to adopt an existing sampler pointer.
    private init(adopting pointer: UnsafeMutablePointer<llama_sampler>) {
        self.samplerPointer = pointer
    }

    // Performance helpers (only valid for chains)
    /// Print sampler performance data via logger and return empty string placeholder.
    public func perfDataDescription() -> String {
        llama_perf_sampler_print(samplerPointer)
        return "" // the C function prints to stderr via log; we expose a no-op string here
    }

    // MARK: - Chain management

    /// Number of samplers in the chain.
    public func count() -> Int { Int(llama_sampler_chain_n(samplerPointer)) }

    /// Get a reference name for the i-th sampler in the chain if available.
    public func name(at index: Int32) -> String {
        guard let s = llama_sampler_chain_get(samplerPointer, index) else { return "" }
        guard let c = llama_sampler_name(s) else { return "" }
        return String(cString: c)
    }

    /// Remove the i-th sampler from the chain (ownership transfers to the chain's previous owner if any).
    public func remove(at index: Int32) {
        _ = llama_sampler_chain_remove(samplerPointer, index)
    }
}
