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
            let stream = try await llamaService.streamCompletion(of: messages, samplingConfig: .init(temperature: 0.7, seed: 0))
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

        } catch {
            XCTFail("streamCompletion threw an error: \(error)")
        }
    }

    func testJSONGenerationWithGrammar() async throws {
        // Load the JSON grammar from the bundle
        guard let grammarURL = Bundle.module.url(forResource: "Resources/json", withExtension: "gbnf") else {
            XCTFail("Could not find json.gbnf file in test bundle")
            return
        }
        
        let grammarString = try String(contentsOf: grammarURL)
        
        // Create grammar configuration
        let grammarConfig = LlamaGrammarConfig(
            grammar: grammarString,
            grammarRoot: "root"
        )
        
        // Create sampling configuration with grammar
        let samplingConfig = LlamaSamplingConfig(
            temperature: 0.1, // Low temperature for more deterministic output
            seed: 42,
            grammarConfig: grammarConfig
        )
        
        // Prepare test messages that request JSON output
        let messages = [
            LlamaChatMessage(role: .system, content: "You are a helpful assistant that responds only in valid JSON format."),
            LlamaChatMessage(role: .user, content: "Generate a JSON object describing a person with name, age, and city properties.")
        ]
        
        do {
            let stream = try await llamaService.streamCompletion(of: messages, samplingConfig: samplingConfig)
            var generatedText = ""
            var tokenCount = 0
            let maxTokens = 100 // Limit tokens for JSON generation
            
            for try await token in stream where tokenCount < maxTokens {
                generatedText += token
                tokenCount += 1
                
                // Check if we have a complete JSON object
                if generatedText.contains("}") && generatedText.filter({ $0 == "{" }).count == generatedText.filter({ $0 == "}" }).count {
                    break
                }
            }
            
            print("Generated JSON text: \(generatedText)")
            
            // Validate that the generated text is valid JSON
            XCTAssertFalse(generatedText.isEmpty, "No text was generated")
            
            // Try to parse as JSON to ensure it's valid
            guard let jsonData = generatedText.data(using: .utf8) else {
                XCTFail("Generated text could not be converted to data")
                return
            }
            
            do {
                let jsonObject = try JSONSerialization.jsonObject(with: jsonData, options: [])
                print("Successfully parsed JSON: \(jsonObject)")
                
                // Verify it's a dictionary (JSON object)
                XCTAssertTrue(jsonObject is [String: Any], "Generated JSON should be an object/dictionary")
                
            } catch {
                XCTFail("Generated text is not valid JSON: \(error)")
            }
            
        } catch {
            XCTFail("streamCompletion with grammar threw an error: \(error)")
        }
    }
}
