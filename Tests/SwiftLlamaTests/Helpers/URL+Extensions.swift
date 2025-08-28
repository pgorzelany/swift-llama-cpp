//
//  URL+Extensions.swift
//  swift-llama-cpp
//
//  Created by Piotr Gorzelany on 13/07/2025.
//

import Foundation

extension URL {
    static let llama1B: URL = {
        if let url = Bundle.module.url(forResource: "Llama-3.2-1B-Instruct-Q4_K_M", withExtension: "gguf", subdirectory: "Resources") {
            return url
        }
        // Fallback to previous path style in case resources are laid out differently
        return Bundle.module.url(forResource: "Resources/Llama-3.2-1B-Instruct-Q4_K_M", withExtension: "gguf")!
    }()
}
