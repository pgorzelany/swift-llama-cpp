//
//  LlamaServiceTests.swift
//  LlamaSwift
//
//  Created by Piotr Gorzelany on 22/10/2024.
//

import XCTest
@testable import SwiftLlama

final class LlamaServicePerformanceTests: XCTestCase {

    // MARK: - Properties
    var llamaService: LlamaService!
    let targetTokenCount = 500 // the service returns every other token

    // MARK: - Setup and Teardown

    override func setUpWithError() throws {
        try super.setUpWithError()

        // Initialize LlamaService with desired parameters
        let contextLength: UInt32 = 16384
        let batchSize: UInt32 = 1024
        llamaService = LlamaService(modelUrl: .llama1B, config: .init(batchSize: batchSize, maxTokenCount: contextLength))
    }

    override func tearDownWithError() throws {
        // Clean up
        llamaService = nil
        try super.tearDownWithError()
    }

    // MARK: - Performance Tests

    func testStreamCompletionPerformance() async throws {
        // Prepare test messages
        let messages = [
            LlamaChatMessage(role: .system, content: "You are a helpful assistant?"),
            LlamaChatMessage(role: .user, content: "Can you tell me a long story about mars colonization??"),

        ]

        let startTime = CFAbsoluteTimeGetCurrent()

        do {
            let stream = try await llamaService.streamCompletion(of: messages, samplingConfig: .init(temperature: 0.5, seed: 0))
            var generatedTokensCount = 0
            var generatedText = ""

            for try await token in stream where generatedTokensCount < targetTokenCount {
                generatedTokensCount += 1
                generatedText += token
            }

            // Optionally assert on received tokens
            XCTAssertEqual(generatedTokensCount, targetTokenCount)

            let endTime = CFAbsoluteTimeGetCurrent()
            let timeElapsed = endTime - startTime
            let tokensPerSecond = Double(generatedTokensCount) / timeElapsed

            // Assert if needed or print the results
            print("Generated \(generatedTokensCount) tokens in \(timeElapsed) seconds.")
            print("Speed: \(tokensPerSecond) tokens/second.")

            // Example assertion (adjust threshold as needed)
            XCTAssert(tokensPerSecond > 24, "Token generation is too slow: \(tokensPerSecond) tokens/second.")
//            XCTAssertEqual(generatedText, expectedCompletion)

        } catch {
            XCTFail("streamCompletion threw an error: \(error)")
        }
    }
}
