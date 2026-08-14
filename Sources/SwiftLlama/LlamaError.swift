//
//  LlamaError.swift
//  swift-llama-cpp
//
//  Created by Piotr Gorzelany on 30/07/2025.
//

import Foundation

public enum LlamaError: LocalizedError {
    case couldNotInitializeContext
    case modelInitializationFailed(diagnostics: String?)
    case contextInitializationFailed(diagnostics: String?)
    case contextSizeLimitExeeded
    case decodingError
    case emptyMessageArray
    case chatTemplateError

    public var errorDescription: String? {
        switch self {
        case .couldNotInitializeContext:
            return "llama.cpp could not initialize the model context."
        case .modelInitializationFailed(let diagnostics):
            return Self.description(
                summary: "llama.cpp could not load the model.",
                diagnostics: diagnostics
            )
        case .contextInitializationFailed(let diagnostics):
            return Self.description(
                summary: "llama.cpp could not initialize the model context.",
                diagnostics: diagnostics
            )
        case .contextSizeLimitExeeded:
            return "The conversation exceeds the model context window."
        case .decodingError:
            return "llama.cpp could not decode the model input."
        case .emptyMessageArray:
            return "At least one chat message is required."
        case .chatTemplateError:
            return "The model could not format the conversation with its chat template."
        }
    }

    private static func description(summary: String, diagnostics: String?) -> String {
        guard let diagnostics, !diagnostics.isEmpty else { return summary }
        return "\(summary)\n\(diagnostics)"
    }
}
