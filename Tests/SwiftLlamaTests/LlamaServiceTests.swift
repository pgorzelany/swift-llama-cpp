//
//  LlamaServiceTests.swift
//  LlamaSwift
//
//  Created by Piotr Gorzelany on 22/10/2024.
//

import XCTest
import OSLog
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
        static let grammarPerformanceTokens = 100
        static let grammarMinimumTokensPerSecond: Double = 12.0
    }
    
    // MARK: - Properties
    
    private var llamaService: LlamaService!
    private let logger = Logger(subsystem: "SwiftLlamaTests", category: "LlamaServiceTests")
    
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
        logger.info("=== Performance Test Results ===")
        logger.info("Generated \(result.tokenCount) tokens")
        logger.info("Speed: \(String(format: "%.2f", result.tokensPerSecond), privacy: .public) tokens/second")
        logger.info("Generated text preview: \(String(result.generatedText.prefix(100)), privacy: .public)...")
    }
    
    func testGrammarGenerationPerformance() async throws {
        // Given
        let messages = createJSONMessages()
        let samplingConfig = try createJSONSamplingConfig()
        
        // When
        let result = try await measureTokenGenerationPerformance(
            messages: messages,
            samplingConfig: samplingConfig,
            targetTokens: TestConfig.grammarPerformanceTokens
        )
        
        // Then
        XCTAssertEqual(
            result.tokenCount,
            TestConfig.grammarPerformanceTokens,
            "Grammar generation should complete target tokens"
        )
        
        XCTAssertGreaterThan(
            result.tokensPerSecond,
            TestConfig.grammarMinimumTokensPerSecond,
            "Grammar generation should maintain minimum performance"
        )
        
        // Verify grammar output is actually valid JSON
        _ = try validateJSON(result.generatedText)
        
        // Log performance metrics
        logger.info("=== Grammar Generation Performance ===")
        logger.info("Generated \(result.tokenCount) tokens")
        logger.info("Speed: \(String(format: "%.2f", result.tokensPerSecond), privacy: .public) tokens/second")
        logger.info("Valid JSON generated: \(result.generatedText.prefix(100), privacy: .public)...")
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
        logger.info("=== JSON Generation Test Results ===")
        logger.info("Generated JSON: \(generatedText, privacy: .public)")
        logger.info("Parsed object keys: \(Array(jsonObject.keys), privacy: .public)")
        
        // Verify structure (optional - could be more specific based on requirements)
        XCTAssertTrue(jsonObject.count > 0, "JSON object should have at least one property")
    }
    
    func testJSONArrayGenerationWithGrammar() async throws {
        // Given
        let messages = createJSONArrayMessages()
        let samplingConfig = try createJSONArraySamplingConfig()
        
        // When
        let generatedText = try await generateJSONWithGrammar(
            messages: messages,
            samplingConfig: samplingConfig,
            maxTokens: TestConfig.jsonTestMaxTokens
        )
        
        // Then
        XCTAssertFalse(generatedText.isEmpty, "Should generate some text")
        
        let jsonArray = try validateJSONArray(generatedText)
        
        // Log results
        logger.info("=== JSON Array Generation Test Results ===")
        logger.info("Generated JSON Array: \(generatedText, privacy: .public)")
        logger.info("Array contains \(jsonArray.count) elements")
        if !jsonArray.isEmpty {
            logger.info("First element type: \(String(describing: type(of: jsonArray[0])), privacy: .public)")
        }
        
        // Verify structure
        XCTAssertTrue(jsonArray.count >= 0, "JSON array should be valid (can be empty)")
    }
    
    // MARK: - Sampling Configuration Tests
    
    func testSeedReproducibility() async throws {
        // Given
        let messages = createSimpleMessages()
        let seed: UInt32 = 12345
        let samplingConfig = LlamaSamplingConfig(temperature: 0.1, seed: seed)

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
        XCTAssertEqual(result1, result2, "Same seed should produce identical output")
        
        logger.info("=== Reproducibility Test Results ===")
        logger.info("Output 1: \(result1, privacy: .public)")
        logger.info("Output 2: \(result2, privacy: .public)")
    }
    
    func testTemperatureEffectsOnOutput() async throws {
        // Given
        let messages = createSimpleMessages()
        #warning("When changing temperature tokens stop being generated. There seems to be an issue in the sampling logic.")
        let temperatureValues: [Float] = Array(stride(from: 0.0, through: 2.0, by: 0.1))
        var results: [(temperature: Float, output: String, success: Bool)] = []
        
        logger.info("=== Temperature Effects Test Results ===")
        logger.info("Testing temperatures from 0.0 to 2.0 in 0.1 intervals...")
        
        // When - Test each temperature value
        for temperature in temperatureValues {
            let samplingConfig = LlamaSamplingConfig(
                temperature: temperature,
                seed: TestConfig.testSeed
            )
            
            do {
                let result = try await generateLimitedText(
                    messages: messages,
                    samplingConfig: samplingConfig,
                    maxTokens: 20 // Shorter for this test
                )
                
                let success = !result.isEmpty
                results.append((temperature: temperature, output: result, success: success))
                
                // Log each result
                let formattedTemp = String(format: "%.1f", temperature)
                let preview = result.prefix(30).replacingOccurrences(of: "\n", with: "\\n")
                logger.info("Temp \(formattedTemp, privacy: .public): ✅ \"\(preview, privacy: .public)\"")
                
                // Then - Verify output was generated
                XCTAssertFalse(result.isEmpty, "Temperature \(formattedTemp) should generate text")
                
            } catch {
                results.append((temperature: temperature, output: "", success: false))
                let formattedTemp = String(format: "%.1f", temperature)
                logger.error("Temp \(formattedTemp, privacy: .public): ❌ Error: \(error)")
                XCTFail("Temperature \(formattedTemp) failed with error: \(error)")
            }
        }
        
        // Summary assertions
        let successfulTests = results.filter { $0.success }
        let failedTests = results.filter { !$0.success }
        
        logger.info("=== Summary ===")
        logger.info("Successful temperatures: \(successfulTests.count)/\(results.count)")
        logger.info("Failed temperatures: \(failedTests.count)")
        
        if !failedTests.isEmpty {
            let failedTemps = failedTests.map { String(format: "%.1f", $0.temperature) }
            logger.error("Failed at temperatures: \(failedTemps.joined(separator: ", "), privacy: .public)")
        }
        
        // Assert that most temperatures work (allowing for some edge case failures)
        let successRate = Double(successfulTests.count) / Double(results.count)
        XCTAssertGreaterThan(successRate, 0.9, "At least 90% of temperature values should work")
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
        
        logger.info("=== Top-K Sampling Test Results ===")
        logger.info("Generated with top-K=5: \(result, privacy: .public)")
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
        
        logger.info("=== Repetition Penalty Test Results ===")
        logger.info("Generated with penalties: \(result, privacy: .public)")
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
            logger.info("=== Empty Message Test Results ===")
            logger.info("Correctly failed with error: \(error)")
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
        
        logger.info("=== Short Prompt Test Results ===")
        logger.info("Response to 'Hi': \(result, privacy: .public)")
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
        
        logger.info("=== Unicode Test Results ===")
        logger.info("Unicode response: \(result, privacy: .public)")
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
        
        logger.info("=== Cancellation Test Results ===")
        logger.info("Generated \(tokenCount) tokens before cancellation: \(generatedText, privacy: .public)")
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
    
    private func createJSONArrayMessages() -> [LlamaChatMessage] {
        [
            LlamaChatMessage(role: .system, content: "You are a helpful assistant that responds only in valid JSON array format."),
            LlamaChatMessage(role: .user, content: "Generate a JSON array containing 3 different fruits as strings.")
        ]
    }
    
    private func loadJSONArrayGrammar() throws -> String {
        guard let grammarURL = Bundle.module.url(forResource: "Resources/json_array", withExtension: "gbnf") else {
            throw TestError.resourceNotFound("json_array.gbnf")
        }
        return try String(contentsOf: grammarURL)
    }
    
    private func createJSONArraySamplingConfig() throws -> LlamaSamplingConfig {
        let grammarString = try loadJSONArrayGrammar()
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
        // Check for complete JSON object
        if text.contains("}") {
            let openBraces = text.filter { $0 == "{" }.count
            let closeBraces = text.filter { $0 == "}" }.count
            if openBraces == closeBraces && openBraces > 0 {
                return true
            }
        }
        
        // Check for complete JSON array
        if text.contains("]") {
            let openBrackets = text.filter { $0 == "[" }.count
            let closeBrackets = text.filter { $0 == "]" }.count
            if openBrackets == closeBrackets && openBrackets > 0 {
                return true
            }
        }
        
        return false
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
    
    private func validateJSONArray(_ jsonString: String) throws -> [Any] {
        guard let jsonData = jsonString.data(using: .utf8) else {
            throw TestError.invalidJSON("Could not convert to data")
        }
        
        let jsonObject = try JSONSerialization.jsonObject(with: jsonData, options: [])
        
        guard let array = jsonObject as? [Any] else {
            throw TestError.invalidJSON("JSON is not an array")
        }
        
        return array
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
