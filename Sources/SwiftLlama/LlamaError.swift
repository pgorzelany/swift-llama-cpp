//
//  LlamaError.swift
//  swift-llama-cpp
//
//  Created by Piotr Gorzelany on 30/07/2025.
//

public enum LlamaError: Error{
    case couldNotInitializeContext
    case contextSizeLimitExeeded
    case decodingError
    case emptyMessageArray
}
