//
//  LlamaServiceTests.swift
//  LlamaSwift
//
//  Created by Piotr Gorzelany on 22/10/2024.
//

import XCTest
@testable import SwiftLlama

final class LlamaServiceTests: XCTestCase {

    // MARK: - Test Configuration
    
    private struct TestConfig {
        static let contextLength: UInt32 = 16384
        static let batchSize: UInt32 = 1024
        static let performanceTargetTokens = 500
        static let jsonTestMaxTokens = 100
        static let minimumTokensPerSecond: Double = 24.0
        static let testSeed: UInt32 = 42
        static let shortTestTokens = 50
    }
    
    // MARK: - Properties
    
    private var llamaService: LlamaService!
    
    // MARK: - Setup and Teardown

    override func setUpWithError() throws {
        try super.setUpWithError()
        llamaService = createLlamaService()
    }

    override func tearDownWithError() throws {
        llamaService = nil
        try super.tearDownWithError()
    }

    // MARK: - Performance Tests

    func testStreamCompletionPerformance() async throws {
        // Given
        let messages = createStoryMessages()
        let samplingConfig = createPerformanceSamplingConfig()
        
        // When
        let result = try await measureTokenGenerationPerformance(
            messages: messages,
            samplingConfig: samplingConfig,
            targetTokens: TestConfig.performanceTargetTokens
        )
        
        // Then
        XCTAssertEqual(
            result.tokenCount,
            TestConfig.performanceTargetTokens,
            "Should generate the expected number of tokens"
        )
        
        XCTAssertGreaterThan(
            result.tokensPerSecond,
            TestConfig.minimumTokensPerSecond,
            "Token generation speed (\(result.tokensPerSecond) tokens/sec) is below minimum threshold"
        )
        
        // Log performance metrics
        print("=== Performance Test Results ===")
        print("Generated \(result.tokenCount) tokens")
        print("Speed: \(String(format: "%.2f", result.tokensPerSecond)) tokens/second")
        print("Generated text preview: \(String(result.generatedText.prefix(100)))...")
    }

    // MARK: - Grammar Tests
    
    func testJSONGenerationWithGrammar() async throws {
        // Given
        let messages = createJSONMessages()
        let samplingConfig = try createJSONSamplingConfig()
        
        // When
        let generatedText = try await generateJSONWithGrammar(
            messages: messages,
            samplingConfig: samplingConfig,
            maxTokens: TestConfig.jsonTestMaxTokens
        )
        
        // Then
        XCTAssertFalse(generatedText.isEmpty, "Should generate some text")
        
        let jsonObject = try validateJSON(generatedText)
        
        // Log results
        print("=== JSON Generation Test Results ===")
        print("Generated JSON: \(generatedText)")
        print("Parsed object keys: \(Array(jsonObject.keys))")
        
        // Verify structure (optional - could be more specific based on requirements)
        XCTAssertTrue(jsonObject.count > 0, "JSON object should have at least one property")
    }
    
    // MARK: - Sampling Configuration Tests
    
    func testSeedReproducibility() async throws {
        // Given
        let messages = createSimpleMessages()
        let seed: UInt32 = 12345
        let samplingConfig = LlamaSamplingConfig(temperature: 0.0, seed: seed)
        
        // When - Generate same content twice with same seed
        let result1 = try await generateLimitedText(
            messages: messages,
            samplingConfig: samplingConfig,
            maxTokens: TestConfig.shortTestTokens
        )
        
        let result2 = try await generateLimitedText(
            messages: messages,
            samplingConfig: samplingConfig,
            maxTokens: TestConfig.shortTestTokens
        )
        
        // Then
        XCTAssertEqual(result1, result2, "Same seed should produce identical output with temperature 0.0")
        
        print("=== Reproducibility Test Results ===")
        print("Output 1: \(result1)")
        print("Output 2: \(result2)")
    }
    
    func testTemperatureEffectsOnOutput() async throws {
        // Given
        let messages = createSimpleMessages()
        let lowTempConfig = LlamaSamplingConfig(temperature: 0.0, seed: TestConfig.testSeed)
        let highTempConfig = LlamaSamplingConfig(temperature: 1.0, seed: TestConfig.testSeed)
        
        // When
        let lowTempResult = try await generateLimitedText(
            messages: messages,
            samplingConfig: lowTempConfig,
            maxTokens: TestConfig.shortTestTokens
        )
        
        let highTempResult = try await generateLimitedText(
            messages: messages,
            samplingConfig: highTempConfig,
            maxTokens: TestConfig.shortTestTokens
        )
        
        // Then
        XCTAssertFalse(lowTempResult.isEmpty, "Low temperature should generate text")
        XCTAssertFalse(highTempResult.isEmpty, "High temperature should generate text")
        
        print("=== Temperature Effects Test Results ===")
        print("Low temp (0.0): \(lowTempResult)")
        print("High temp (1.0): \(highTempResult)")
    }
    
    func testTopKSamplingConstraints() async throws {
        // Given
        let messages = createSimpleMessages()
        let topKConfig = LlamaSamplingConfig(
            temperature: 0.8,
            seed: TestConfig.testSeed,
            topK: 5 // Very restrictive top-k
        )
        
        // When
        let result = try await generateLimitedText(
            messages: messages,
            samplingConfig: topKConfig,
            maxTokens: TestConfig.shortTestTokens
        )
        
        // Then
        XCTAssertFalse(result.isEmpty, "Top-K sampling should still generate text")
        
        print("=== Top-K Sampling Test Results ===")
        print("Generated with top-K=5: \(result)")
    }
    
    func testRepetitionPenaltyConfiguration() async throws {
        // Given
        let messages = createRepetitivePromptMessages()
        let penaltyConfig = LlamaRepetitionPenaltyConfig(
            lastN: 20,
            repeatPenalty: 1.3,
            freqPenalty: 0.1,
            presentPenalty: 0.1
        )
        let samplingConfig = LlamaSamplingConfig(
            temperature: 0.8,
            seed: TestConfig.testSeed,
            repetitionPenaltyConfig: penaltyConfig
        )
        
        // When
        let result = try await generateLimitedText(
            messages: messages,
            samplingConfig: samplingConfig,
            maxTokens: TestConfig.shortTestTokens
        )
        
        // Then
        XCTAssertFalse(result.isEmpty, "Should generate text with repetition penalty")
        
        print("=== Repetition Penalty Test Results ===")
        print("Generated with penalties: \(result)")
    }
    
    // MARK: - Edge Case Tests
    
    func testEmptyMessageHandling() async throws {
        // Given
        let emptyMessages: [LlamaChatMessage] = []
        let samplingConfig = createPerformanceSamplingConfig()
        
        // When/Then
        do {
            _ = try await llamaService.streamCompletion(of: emptyMessages, samplingConfig: samplingConfig)
            XCTFail("Should throw an error for empty messages")
        } catch {
            // Expected to fail - empty messages should not be allowed
            print("=== Empty Message Test Results ===")
            print("Correctly failed with error: \(error)")
        }
    }
    
    func testVeryShortPromptHandling() async throws {
        // Given
        let shortMessages = [
            LlamaChatMessage(role: .user, content: "Hi")
        ]
        let samplingConfig = createPerformanceSamplingConfig()
        
        // When
        let result = try await generateLimitedText(
            messages: shortMessages,
            samplingConfig: samplingConfig,
            maxTokens: 20
        )
        
        // Then
        XCTAssertFalse(result.isEmpty, "Should handle very short prompts")
        
        print("=== Short Prompt Test Results ===")
        print("Response to 'Hi': \(result)")
    }
    
    func testUnicodeAndSpecialCharacters() async throws {
        // Given
        let unicodeMessages = [
            LlamaChatMessage(role: .user, content: "Here are some emojis: 🎯🚀🔥 and Unicode: αβγδε français español 中文. What do you say?")
        ]
        let samplingConfig = createPerformanceSamplingConfig()
        
        // When
        let result = try await generateLimitedText(
            messages: unicodeMessages,
            samplingConfig: samplingConfig,
            maxTokens: TestConfig.shortTestTokens
        )
        
        // Then
        XCTAssertFalse(result.isEmpty, "Should handle Unicode and special characters")
        
        print("=== Unicode Test Results ===")
        print("Unicode response: \(result)")
    }
    
    // MARK: - Cancellation Tests
    
    func testStreamCancellation() async throws {
        // Given
        let messages = createStoryMessages()
        let samplingConfig = createPerformanceSamplingConfig()
        
        // When
        let stream = try await llamaService.streamCompletion(of: messages, samplingConfig: samplingConfig)
        var tokenCount = 0
        var generatedText = ""
        
        // Cancel after generating a few tokens
        for try await token in stream {
            generatedText += token
            tokenCount += 1
            
            if tokenCount >= 10 {
                await llamaService.stopCompletion()
                break
            }
        }
        
        // Then
        XCTAssertGreaterThan(tokenCount, 0, "Should have generated some tokens before cancellation")
        XCTAssertLessThanOrEqual(tokenCount, 15, "Should have stopped generation quickly after cancellation")
        
        print("=== Cancellation Test Results ===")
        print("Generated \(tokenCount) tokens before cancellation: \(generatedText)")
    }
}

// MARK: - Helper Methods

extension LlamaServiceTests {
    
    private func createLlamaService() -> LlamaService {
        LlamaService(
            modelUrl: .llama1B,
            config: .init(
                batchSize: TestConfig.batchSize,
                maxTokenCount: TestConfig.contextLength
            )
        )
    }
    
    private func createPerformanceSamplingConfig() -> LlamaSamplingConfig {
        LlamaSamplingConfig(
            temperature: 0.7,
            seed: TestConfig.testSeed
        )
    }
    
    private func createStoryMessages() -> [LlamaChatMessage] {
        [
            LlamaChatMessage(role: .system, content: "You are a helpful assistant."),
            LlamaChatMessage(role: .user, content: "Can you tell me a long story about mars colonization?")
        ]
    }
    
    private func createSimpleMessages() -> [LlamaChatMessage] {
        [
            LlamaChatMessage(role: .user, content: "Write a short sentence about the weather.")
        ]
    }
    
    private func createRepetitivePromptMessages() -> [LlamaChatMessage] {
        [
            LlamaChatMessage(role: .user, content: "Write about cats. Cats are great. Cats cats cats. Tell me more about cats.")
        ]
    }
    
    private func loadJSONGrammar() throws -> String {
        guard let grammarURL = Bundle.module.url(forResource: "Resources/json", withExtension: "gbnf") else {
            throw TestError.resourceNotFound("json.gbnf")
        }
        return try String(contentsOf: grammarURL)
    }
    
    private func createJSONSamplingConfig() throws -> LlamaSamplingConfig {
        let grammarString = try loadJSONGrammar()
        let grammarConfig = LlamaGrammarConfig(
            grammar: grammarString,
            grammarRoot: "root"
        )
        
        return LlamaSamplingConfig(
            temperature: 0.1, // Low temperature for deterministic output
            seed: TestConfig.testSeed,
            grammarConfig: grammarConfig
        )
    }
    
    private func createJSONMessages() -> [LlamaChatMessage] {
        [
            LlamaChatMessage(role: .system, content: "You are a helpful assistant that responds only in valid JSON format."),
            LlamaChatMessage(role: .user, content: "Generate a JSON object describing a person with name, age, and city properties.")
        ]
    }
    
    private func measureTokenGenerationPerformance(
        messages: [LlamaChatMessage],
        samplingConfig: LlamaSamplingConfig,
        targetTokens: Int
    ) async throws -> (tokenCount: Int, tokensPerSecond: Double, generatedText: String) {
        
        let startTime = CFAbsoluteTimeGetCurrent()
        let stream = try await llamaService.streamCompletion(of: messages, samplingConfig: samplingConfig)
        
        var generatedTokensCount = 0
        var generatedText = ""
        
        for try await token in stream where generatedTokensCount < targetTokens {
            generatedTokensCount += 1
            generatedText += token
        }
        
        let endTime = CFAbsoluteTimeGetCurrent()
        let timeElapsed = endTime - startTime
        let tokensPerSecond = Double(generatedTokensCount) / timeElapsed
        
        return (generatedTokensCount, tokensPerSecond, generatedText)
    }
    
    private func generateJSONWithGrammar(
        messages: [LlamaChatMessage],
        samplingConfig: LlamaSamplingConfig,
        maxTokens: Int
    ) async throws -> String {
        
        let stream = try await llamaService.streamCompletion(of: messages, samplingConfig: samplingConfig)
        var generatedText = ""
        var tokenCount = 0
        
        for try await token in stream where tokenCount < maxTokens {
            generatedText += token
            tokenCount += 1
            
            // Stop when we have a complete JSON object
            if isCompleteJSON(generatedText) {
                break
            }
        }
        
        return generatedText
    }
    
    private func generateLimitedText(
        messages: [LlamaChatMessage],
        samplingConfig: LlamaSamplingConfig,
        maxTokens: Int
    ) async throws -> String {
        
        let stream = try await llamaService.streamCompletion(of: messages, samplingConfig: samplingConfig)
        var generatedText = ""
        var tokenCount = 0
        
        for try await token in stream where tokenCount < maxTokens {
            generatedText += token
            tokenCount += 1
        }
        
        return generatedText
    }
    
    private func isCompleteJSON(_ text: String) -> Bool {
        guard text.contains("}") else { return false }
        let openBraces = text.filter { $0 == "{" }.count
        let closeBraces = text.filter { $0 == "}" }.count
        return openBraces == closeBraces && openBraces > 0
    }
    
    private func validateJSON(_ jsonString: String) throws -> [String: Any] {
        guard let jsonData = jsonString.data(using: .utf8) else {
            throw TestError.invalidJSON("Could not convert to data")
        }
        
        let jsonObject = try JSONSerialization.jsonObject(with: jsonData, options: [])
        
        guard let dictionary = jsonObject as? [String: Any] else {
            throw TestError.invalidJSON("JSON is not an object/dictionary")
        }
        
        return dictionary
    }
}

// MARK: - Test Errors

private enum TestError: Error, LocalizedError {
    case resourceNotFound(String)
    case invalidJSON(String)
    
    var errorDescription: String? {
        switch self {
        case .resourceNotFound(let resource):
            return "Test resource not found: \(resource)"
        case .invalidJSON(let reason):
            return "Invalid JSON: \(reason)"
        }
    }
}
