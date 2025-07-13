//
//  URL+Extensions.swift
//  swift-llama-cpp
//
//  Created by Piotr Gorzelany on 13/07/2025.
//

import Foundation

extension URL {
    static let llama1B = Bundle.module.url(forResource: "Models/Llama-3.2-1B-Instruct-Q4_K_M", withExtension: "gguf")!
}
