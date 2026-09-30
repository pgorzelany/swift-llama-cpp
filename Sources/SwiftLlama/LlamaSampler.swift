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
    let samplerPointer: UnsafeMutablePointer<llama_sampler>

    private let model: LlamaModel

    /// Builds a chain with penalties before filters; invalid grammar or parameters throw.
    public init(config: LlamaSamplingConfig, model: LlamaModel) throws {
        guard config.temperature.isFinite, config.temperature >= 0,
              config.topP.isFinite, config.topP > 0, config.topP <= 1,
              config.minKeep > 0, config.topK.map({ $0 >= 0 }) ?? true else {
            throw LlamaError.invalidSamplingConfiguration
        }
        if let penalty = config.repetitionPenaltyConfig {
            guard penalty.lastN >= 0, penalty.repeatPenalty.isFinite, penalty.repeatPenalty > 0,
                  penalty.freqPenalty.isFinite, penalty.presentPenalty.isFinite else {
                throw LlamaError.invalidSamplingConfiguration
            }
        }
        self.model = model
        let pointer = llama_sampler_chain_init(llama_sampler_chain_default_params())!
        if let grammar = config.grammarConfig {
            guard let grammarSampler = llama_sampler_init_grammar(model.vocabPointer, grammar.grammar, grammar.grammarRoot) else {
                llama_sampler_free(pointer)
                throw LlamaError.invalidGrammar
            }
            llama_sampler_chain_add(pointer, grammarSampler)
        }
        if let penalty = config.repetitionPenaltyConfig, penalty.lastN != 0,
           penalty.repeatPenalty != 1 || penalty.freqPenalty != 0 || penalty.presentPenalty != 0 {
            llama_sampler_chain_add(pointer, llama_sampler_init_penalties(
                model.vocabularySize(), penalty.lastN, penalty.repeatPenalty, penalty.freqPenalty, penalty.presentPenalty))
        }
        if config.temperature == 0 {
            llama_sampler_chain_add(pointer, llama_sampler_init_greedy())
        } else {
            if let topK = config.topK { llama_sampler_chain_add(pointer, llama_sampler_init_top_k(topK)) }
            llama_sampler_chain_add(pointer, llama_sampler_init_top_p(config.topP, config.minKeep))
            llama_sampler_chain_add(pointer, llama_sampler_init_temp(config.temperature))
            llama_sampler_chain_add(pointer, llama_sampler_init_dist(config.seed))
        }
        self.samplerPointer = pointer
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

    /// Accepts a generated token, advancing penalties and any output grammar.
    public func accept(token: llama_token) {
        llama_sampler_accept(samplerPointer, token)
    }

    /// Seeds repetition history without advancing the output grammar.
    public func acceptPrompt(tokens: [llama_token]) {
        for index in 0..<count() where name(at: Int32(index)) == "penalties" {
            guard let penalty = llama_sampler_chain_get(samplerPointer, Int32(index)) else { continue }
            for token in tokens { llama_sampler_accept(penalty, token) }
        }
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
        return LlamaSampler(adopting: cloned, model: model)
    }

    /// Internal initializer to adopt an existing sampler pointer.
    private init(adopting pointer: UnsafeMutablePointer<llama_sampler>, model: LlamaModel) {
        self.model = model
        self.samplerPointer = pointer
    }

    // Performance helpers (only valid for chains)
    /// Return raw C counters; they remain zero when chain performance collection is disabled.
    public func perfDataDescription() -> String {
        let data = llama_perf_sampler(samplerPointer)
        return "sampled=\(data.n_sample), samplingMilliseconds=\(data.t_sample_ms)"
    }

    // MARK: - Chain management

    /// Number of samplers in the chain.
    public func count() -> Int { Int(llama_sampler_chain_n(samplerPointer)) }

    /// Get a reference name for the i-th sampler in the chain if available.
    public func name(at index: Int32) -> String {
        guard index >= 0, index < count(), let s = llama_sampler_chain_get(samplerPointer, index) else { return "" }
        guard let c = llama_sampler_name(s) else { return "" }
        return String(cString: c)
    }

    /// Frees a removed stage. The final token selector cannot be removed.
    @discardableResult
    public func remove(at index: Int32) -> Bool {
        guard index >= 0, index < count() - 1,
              let removed = llama_sampler_chain_remove(samplerPointer, index) else { return false }
        llama_sampler_free(removed)
        return true
    }
}
