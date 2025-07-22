//
//  LlamaLoger.swift
//  swift-llama-cpp
//
//  Created by Piotr Gorzelany on 22/07/2025.
//

import OSLog

extension Logger {
    static func llama(category: String) -> Logger {
        return Logger(subsystem: "swift-llama-ccp", category: category)
    }
}
