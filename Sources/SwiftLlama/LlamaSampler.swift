//
//  LlamaSampler.swift
//  LlamaSwift
//
//  Created by Piotr Gorzelany on 26/09/2024.
//


import Foundation
import llama

final class LlamaSampler {
    private let samplerPointer: UnsafeMutablePointer<llama_sampler>

    init(config: LlamaSamplingConfig) {
        print(config)
        let sparams = llama_sampler_chain_default_params()
        self.samplerPointer = llama_sampler_chain_init(sparams)

        // Add samplers based on the configuration
        if let topK = config.topK {
            let topKSampler = llama_sampler_init_top_k(topK)
            llama_sampler_chain_add(samplerPointer, topKSampler)
        }

        let topPSampler = llama_sampler_init_top_p(config.topP, config.minKeep)
        llama_sampler_chain_add(samplerPointer, topPSampler)

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

    func sample(context: LlamaContext, lastTokenIndex: Int32) -> llama_token {
        #warning("Sampler usage example: https://github.com/ggerganov/llama.cpp/blob/564804b79b78df1469ec8646869972de5e885ec4/include/llama.h#L1065")
        return llama_sampler_sample(samplerPointer, context.contextPointer, lastTokenIndex)
    }

    func accept(token: llama_token) {
        llama_sampler_accept(samplerPointer, token)
    }
}
