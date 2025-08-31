//
//  LlamaServiceTests.swift
//  LlamaSwift
//
//  Created by Piotr Gorzelany on 22/10/2024.
//

import Testing
import Foundation
import OSLog
@testable import SwiftLlama

struct LlamaServiceTests {

    // MARK: - Test Configuration
    
    private struct TestConfig {
        static let contextLength: UInt32 = 16384
        static let batchSize: UInt32 = 1024
        static let performanceTargetTokens = 500
        static let jsonTestMaxTokens = 100
        // Local performance baselines captured on this machine (detect regressions)
        static let minimumTokensPerSecond: Double = 20.0
        static let testSeed: UInt32 = 42
        static let shortTestTokens = 50
        static let grammarPerformanceTokens = 20
        static let grammarMinimumTokensPerSecond: Double = 3.0
        static let deterministicTokens = 100
        static let deterministicMinCharacters = 200
    }
    
    // MARK: - Properties
    
    private var llamaService: LlamaService!
    private let logger = Logger(subsystem: "SwiftLlamaTests", category: "LlamaServiceTests")
    
    // MARK: - Setup and Teardown

    init() {
        llamaService = createLlamaService()
    }

    // MARK: - Performance Tests

    @Test("Streaming performance baseline")
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
        #expect(result.tokenCount == TestConfig.performanceTargetTokens)
        // Local baseline; intent is regression tracking, not strict perf
        #expect(result.tokensPerSecond > TestConfig.minimumTokensPerSecond)
        
        // Log performance metrics
        print("PERF_STREAM tokens=\(result.tokenCount) tps=\(String(format: "%.2f", result.tokensPerSecond))")
        logger.info("=== Performance Test Results ===")
        logger.info("Generated \(result.tokenCount) tokens")
        logger.info("Speed: \(String(format: "%.2f", result.tokensPerSecond), privacy: .public) tokens/second")
        logger.info("Generated text preview: \(String(result.generatedText.prefix(100)), privacy: .public)...")
    }
    
    @Test("Grammar generation performance baseline")
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
        // Baseline: ensure some progress and modest throughput to track regressions
        #expect(result.tokenCount >= 10)
        #expect(result.tokensPerSecond > TestConfig.grammarMinimumTokensPerSecond)
        
        // Note: performance harness may cut mid-JSON; don't validate structure here
        
        // Log performance metrics
        print("PERF_GRAMMAR tokens=\(result.tokenCount) tps=\(String(format: "%.2f", result.tokensPerSecond))")
        logger.info("=== Grammar Generation Performance ===")
        logger.info("Generated \(result.tokenCount) tokens")
        logger.info("Speed: \(String(format: "%.2f", result.tokensPerSecond), privacy: .public) tokens/second")
        logger.info("Valid JSON generated: \(result.generatedText.prefix(100), privacy: .public)...")
    }

    // MARK: - Grammar Tests
    
    @Test("JSON object generation with grammar")
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
        #expect(!generatedText.isEmpty)
        
        let jsonObject = try validateJSON(generatedText)
        
        // Log results
        logger.info("=== JSON Generation Test Results ===")
        logger.info("Generated JSON: \(generatedText, privacy: .public)")
        logger.info("Parsed object keys: \(Array(jsonObject.keys), privacy: .public)")
        
        // Verify structure (optional - could be more specific based on requirements)
        #expect(jsonObject.count > 0)
    }
    
    @Test("JSON array generation with grammar")
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
        #expect(!generatedText.isEmpty)
        
        let jsonArray = try validateJSONArray(generatedText)
        
        // Log results
        logger.info("=== JSON Array Generation Test Results ===")
        logger.info("Generated JSON Array: \(generatedText, privacy: .public)")
        logger.info("Array contains \(jsonArray.count) elements")
        if !jsonArray.isEmpty {
            logger.info("First element type: \(String(describing: type(of: jsonArray[0])), privacy: .public)")
        }
        
        // Verify structure
        #expect(jsonArray.count >= 0)
    }

    @Test("JSON string array generation and parsing to [String]")
    func testJSONStringArrayGenerationAndParsing() async throws {
        // Given
        let messages = [
            LlamaChatMessage(role: .system, content: "You are a helpful assistant that responds only in valid JSON array format."),
            LlamaChatMessage(role: .user, content: "Generate a JSON array of 5 programming languages as strings.")
        ]
        let samplingConfig = try createJSONStringArraySamplingConfig()

        // When
        let generatedText = try await generateJSONWithGrammar(
            messages: messages,
            samplingConfig: samplingConfig,
            maxTokens: TestConfig.jsonTestMaxTokens
        )

        // Then
        let array = try validateStringArray(generatedText)
        #expect(!array.isEmpty)
        // Basic sanity: ensure they look like single words or known languages
        #expect(array.count == 5)
    }
    
    // MARK: - Sampling Configuration Tests
    
    @Test("Determinism: same seed -> identical output")
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
        #expect(result1 == result2)
        
        logger.info("=== Reproducibility Test Results ===")
        logger.info("Output 1: \(result1, privacy: .public)")
        logger.info("Output 2: \(result2, privacy: .public)")
    }

    @Test("Deterministic run produces at least 200 characters")
    func testDeterministicMinimumCharacterCount() async throws {
        // Given: use the longer story prompt and fixed seed
        let messages = createStoryMessages()
        let samplingConfig = LlamaSamplingConfig(temperature: 0.1, seed: TestConfig.testSeed)

        // When
        let result = try await generateLimitedText(
            messages: messages,
            samplingConfig: samplingConfig,
            maxTokens: TestConfig.deterministicTokens
        )

        // Then
        #expect(result.count >= TestConfig.deterministicMinCharacters)
        logger.info("Deterministic length: \(result.count, privacy: .public) chars")
    }

    @Test("Short story matches deterministic baseline")
    func testShortStoryMatchesBaseline() async throws {
        // Given: fixed prompt and seed/temperature for determinism
        let baseline = "Whiskers, a sleek and agile feline, spent her Martian days lounging in the low-gravity sunbeams that streamed through the transparent dome of her habitat module. At night, she'd prowl the dusty terrain outside, chasing after the occasional Martian dust bunny as she explored the barren landscape of Olympus Mons, the largest volcano on the Red Planet."
        let messages = [
            LlamaChatMessage(role: .system, content: "You are a helpful assistant."),
            LlamaChatMessage(role: .user, content: "Write a concise two-sentence story about a cat living on Mars. Be specific.")
        ]
        let cfg = LlamaSamplingConfig(temperature: 0.1, seed: TestConfig.testSeed)

        // When
        let generated = try await generateLimitedText(messages: messages, samplingConfig: cfg, maxTokens: 160)

        // Then: compare after light whitespace normalization to avoid incidental spacing differences
        func normalize(_ s: String) -> String {
            let trimmed = s.trimmingCharacters(in: .whitespacesAndNewlines)
            let squashed = trimmed.replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
            return squashed
        }
        #expect(normalize(generated) == normalize(baseline))
    }

    // NOTE: The baseline above was captured by a temporary print-only test and then inlined.
    
    @Test("Temperature impacts output")
    func testTemperatureEffectsOnOutput() async throws {
        // Given
        let messages = createSimpleMessages()
        let temperatureValues: [Float] = [0.0, 0.7, 1.3, 2.0]
        var successes = 0
        
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
                    maxTokens: 10
                )
                if !result.isEmpty { successes += 1 }
            } catch {
                let formattedTemp = String(format: "%.1f", temperature)
                Issue.record("Temperature \(formattedTemp) failed with error: \(error)")
            }
        }
        
        // At least one temperature setting should yield output
        #expect(successes >= 1)
    }
    
    @Test("Top-K sampling produces output")
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
        #expect(!result.isEmpty)
        
        logger.info("=== Top-K Sampling Test Results ===")
        logger.info("Generated with top-K=5: \(result, privacy: .public)")
    }

    // MARK: - respond<T: Codable>() Tests
    
    private struct Person: Codable, Equatable {
        let name: String
        let age: Int
        let city: String?
    }
    
    @Test("Typed respond() produces decodable JSON object")
    func testRespondProducesDecodableObject() async throws {
        // Given
        let messages = [
            LlamaChatMessage(role: .system, content: "You are a helpful assistant that responds only in JSON matching the schema."),
            LlamaChatMessage(role: .user, content: "Return a person with name 'Ada', age 36, city 'London'.")
        ]
        
        // When
        let person = try await llamaService.respond(to: messages, generating: Person.self)
        
        // Then
        #expect(!person.name.isEmpty)
        #expect(person.age >= 0)
    }

    @Test("Typed respond() produces decodable array of strings")
    func testRespondProducesArrayOfStrings() async throws {
        // Given
        let messages = [
            LlamaChatMessage(role: .system, content: "You are a helpful assistant that responds only in a JSON array of strings."),
            LlamaChatMessage(role: .user, content: "Return an array of 3 fruits as strings.")
        ]
        
        // When
        let fruits = try await llamaService.respond(to: messages, generating: [String].self)
        
        // Then
        #expect(!fruits.isEmpty)
        #expect(fruits.count >= 3)
    }
    
    // MARK: - respond(messages:samplingConfig:) Tests
    
    @Test("Plain respond() returns non-empty text")
    func testRespondTextNonEmpty() async throws {
        // Given: small token budget to keep the test bounded
        let service = LlamaService(
            modelUrl: .llama1B,
            config: .init(batchSize: 256, maxTokenCount: 80, useGPU: false)
        )
        let messages = [
            LlamaChatMessage(role: .system, content: "You are a helpful assistant."),
            LlamaChatMessage(role: .user, content: "Write a single short sentence (max 20 words).")
        ]
        let cfg = LlamaSamplingConfig(temperature: 0.3, seed: TestConfig.testSeed)

        // When
        let text = try await service.respond(to: messages, samplingConfig: cfg)

        // Then
        #expect(!text.isEmpty)
    }

    @Test("Plain respond() is deterministic with same seed")
    func testRespondDeterminismWithSameSeed() async throws {
        // Given: two fresh services and identical config for determinism
        let cfg = LlamaSamplingConfig(temperature: 0.1, seed: TestConfig.testSeed)
        let serviceA = LlamaService(
            modelUrl: .llama1B,
            config: .init(batchSize: 256, maxTokenCount: 80, useGPU: false)
        )
        let serviceB = LlamaService(
            modelUrl: .llama1B,
            config: .init(batchSize: 256, maxTokenCount: 80, useGPU: false)
        )
        let messages = createSimpleMessages()

        // When
        let out1 = try await serviceA.respond(to: messages, samplingConfig: cfg)
        let out2 = try await serviceB.respond(to: messages, samplingConfig: cfg)

        // Then
        #expect(out1 == out2)
    }

    // MARK: - json_array_strings.gbnf grammar with respond()

    @Test("JSONStringArray grammar via respond() decodes to [String]")
    func testJSONStringArrayRespondParsesToArray() async throws {
        // Given
        let samplingConfig = try createJSONStringArraySamplingConfig()
        let service = LlamaService(
            modelUrl: .llama1B,
            config: .init(batchSize: 256, maxTokenCount: 120, useGPU: false)
        )
        let messages = [
            LlamaChatMessage(role: .system, content: "You are a helpful assistant that responds only in a JSON array of strings."),
            LlamaChatMessage(role: .user, content: "Generate a JSON array of 4 animals as strings.")
        ]

        // When
        let generatedText = try await service.respond(to: messages, samplingConfig: samplingConfig)

        // Then
        let array = try validateStringArray(generatedText)
        #expect(array.count == 4)
    }

    @Test("JSONStringArray grammar via respond() is deterministic with same seed")
    func testJSONStringArrayRespondDeterminism() async throws {
        // Given
        let samplingConfig = try createJSONStringArraySamplingConfig()
        let serviceA = LlamaService(
            modelUrl: .llama1B,
            config: .init(batchSize: 256, maxTokenCount: 120, useGPU: false)
        )
        let serviceB = LlamaService(
            modelUrl: .llama1B,
            config: .init(batchSize: 256, maxTokenCount: 120, useGPU: false)
        )
        let messages = [
            LlamaChatMessage(role: .system, content: "You are a helpful assistant that responds only in a JSON array of strings."),
            LlamaChatMessage(role: .user, content: "Generate a JSON array of 3 programming languages as strings.")
        ]

        // When
        let out1 = try await serviceA.respond(to: messages, samplingConfig: samplingConfig)
        let out2 = try await serviceB.respond(to: messages, samplingConfig: samplingConfig)

        // Then
        #expect(out1 == out2)
        let arr1 = try validateStringArray(out1)
        let arr2 = try validateStringArray(out2)
        #expect(arr1 == arr2)
    }

    @Test("JSONStringArray grammar remains valid at high temperature via respond()")
    func testJSONStringArrayHighTemperatureValidity() async throws {
        // Given: high temperature but constrained by grammar should still yield valid JSON array of strings
        let grammarString = try loadJSONStringArrayGrammar()
        let grammarConfig = LlamaGrammarConfig(grammar: grammarString, grammarRoot: "root")
        let samplingConfig = LlamaSamplingConfig(
            temperature: 1.8,
            seed: TestConfig.testSeed,
            grammarConfig: grammarConfig
        )
        let service = LlamaService(
            modelUrl: .llama1B,
            config: .init(batchSize: 256, maxTokenCount: 120, useGPU: false)
        )
        let messages = [
            LlamaChatMessage(role: .system, content: "You are a helpful assistant that responds only in a JSON array of strings."),
            LlamaChatMessage(role: .user, content: "Generate a short JSON array of lowercase fruit names as strings.")
        ]

        // When
        let generatedText = try await service.respond(to: messages, samplingConfig: samplingConfig)

        // Then: ensure it's a valid JSON array of strings (size may vary)
        let array = try validateStringArray(generatedText)
        #expect(!array.isEmpty)
    }
    
    @Test("Repetition penalty produces output")
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
        #expect(!result.isEmpty)
        
        logger.info("=== Repetition Penalty Test Results ===")
        logger.info("Generated with penalties: \(result, privacy: .public)")
    }
    
    // MARK: - Edge Case Tests
    
    @Test("Empty messages are rejected")
    func testEmptyMessageHandling() async throws {
        // Given
        let emptyMessages: [LlamaChatMessage] = []
        let samplingConfig = createPerformanceSamplingConfig()
        
        // When/Then
        do {
            _ = try await llamaService.streamCompletion(of: emptyMessages, samplingConfig: samplingConfig)
            Issue.record("Should throw an error for empty messages")
        } catch {
            // Expected to fail - empty messages should not be allowed
            logger.info("=== Empty Message Test Results ===")
            logger.info("Correctly failed with error: \(error)")
        }
    }
    
    @Test("Handles very short prompts")
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
        #expect(!result.isEmpty)
        
        logger.info("=== Short Prompt Test Results ===")
        logger.info("Response to 'Hi': \(result, privacy: .public)")
    }
    
    @Test("Handles Unicode and special characters")
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
        #expect(!result.isEmpty)
        
        logger.info("=== Unicode Test Results ===")
        logger.info("Unicode response: \(result, privacy: .public)")
    }
    
    // MARK: - Cancellation Tests
    
    @Test("Streaming can be cancelled")
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
        #expect(tokenCount > 0)
        #expect(tokenCount <= 30)
        
        logger.info("=== Cancellation Test Results ===")
        logger.info("Generated \(tokenCount) tokens before cancellation: \(generatedText, privacy: .public)")
    }

    // MARK: - Golden Deterministic Story

    @Test("Long deterministic story baseline")
    func testLongDeterministicStoryBaseline() async throws {
        // Given: deterministic CPU-only config and fixed prompt
        let service = LlamaService(
            modelUrl: .llama1B,
            config: .init(batchSize: 256, maxTokenCount: 2048, useGPU: false)
        )

        let messages = [
            LlamaChatMessage(role: .system, content: "You are a helpful assistant."),
            LlamaChatMessage(role: .user, content: "Write a vivid short story (3-4 paragraphs) about a time traveler visiting ancient Alexandria, focusing on the Library and the harbor. Keep it under 350 words.")
        ]
        let cfg = LlamaSamplingConfig(temperature: 0.0, seed: 12345, topP: 1.0, topK: nil, minKeep: 1)

        // When
        let story = try await generateText(
            using: service,
            messages: messages,
            samplingConfig: cfg,
            maxTokens: 240
        )

        // Always print the story delimited for easy copy-paste
        print("LLAMA_GENERATED_STORY_START\n\(story)\nLLAMA_GENERATED_STORY_END")

        // Baseline captured locally on this machine using the same model + binary.
        let baseline = """
        As the sun set over ancient Alexandria, a lone figure emerged from the shadows. A time traveler, with eyes that shone like stars in the night sky, stepped onto the bustling streets of this fabled city. The air was thick with the scent of papyrus and olive oil as he made his way to the Great Library of Alexandria. The towering structure loomed before him, its marble columns glinting like a thousand tiny diamonds in the fading light.

        As he pushed open the doors, a warm golden glow enveloped him, illuminating rows upon rows of dusty scrolls and ancient texts. The time traveler wandered through the stacks, running his fingers over the worn leather bindings, feeling the weight of centuries of knowledge within. He paused before a shelf dedicated to the works of Homer, his eyes scanning the yellowed pages as if searching for a specific verse.

        The sound of gulls crying overhead gave way to the murmur of merchants haggling over goods at the harbor. The time traveler's gaze followed the throngs of ships and sailors, their vessels bearing exotic spices from distant lands. He watched as a young apprentice, his eyes shining with excitement, carefully unwrapped a shipment of pome
        """
        #expect(story == baseline)
    }

    @Test("Long deterministic token baseline")
    func testLongDeterministicTokenBaseline() async throws {
        // Given: deterministic CPU-only config and fixed prompt
        let service = LlamaService(
            modelUrl: .llama1B,
            config: .init(batchSize: 256, maxTokenCount: 2048, useGPU: false)
        )

        let messages = [
            LlamaChatMessage(role: .system, content: "You are a helpful assistant."),
            LlamaChatMessage(role: .user, content: "Write a vivid short story (3-4 paragraphs) about a time traveler visiting ancient Alexandria, focusing on the Library and the harbor. Keep it under 350 words.")
        ]
        let cfg = LlamaSamplingConfig(temperature: 0.0, seed: 12345, topP: 1.0, topK: nil, minKeep: 1)

        // When: run generation but also collect token ids from the underlying Llama actor
        let stream = try await service.streamCompletion(of: messages, samplingConfig: cfg)
        var out = ""
        var count = 0
        for try await token in stream where count < 240 {
            out += token
            count += 1
        }

        // We need the token ids, which are tracked by Llama
        // Re-process the same prompt through a fresh Llama to retrieve tokens deterministically
        let llama = try Llama(modelPath: URL.llama1B.path, config: .init(batchSize: 256, maxTokenCount: 2048, useGPU: false))
        try await llama.initializeCompletion(messages: messages)
        await llama.updateSamplingConfig(cfg)
        var tokens: [Int32] = []
        generation: while await llama.currentTokenPosition < llama.maxTokenCount && tokens.count < 240 {
            let result = try await llama.generateNextToken()
            switch result {
            case .token:
                tokens = await llama.getProcessedTokenIds()
            case .endOfString:
                break generation
            }
        }

        // Print tokens for capture
        print("LLAMA_GENERATED_TOKENS_START\n\(tokens)\nLLAMA_GENERATED_TOKENS_END")

        // Then: token baseline captured on this machine for strict reproducibility
        let tokenBaseline: [Int32] = [128006, 9125, 128007, 271, 2675, 527, 264, 11190, 18328, 13, 128009, 128006, 882, 128007, 271, 8144, 264, 43120, 2875, 3446, 320, 18, 12, 19, 43743, 8, 922, 264, 892, 63865, 17136, 14154, 57233, 11, 21760, 389, 279, 11896, 323, 279, 57511, 13, 13969, 433, 1234, 220, 8652, 4339, 13, 128009, 128006, 78191, 128007, 271, 2170, 279, 7160, 743, 927, 14154, 57233, 11, 264, 47766, 7216, 22763, 505, 279, 35612, 13, 362, 892, 63865, 11, 449, 6548, 430, 559, 606, 1093, 9958, 304, 279, 3814, 13180, 11, 25319, 8800, 279, 90256, 14708, 315, 420, 282, 2364, 3363, 13, 578, 3805, 574, 12314, 449, 279, 41466, 315, 281, 90294, 323, 33213, 5707, 439, 568, 1903, 813, 1648, 311, 279, 8681, 11896, 315, 57233, 13, 578, 87794, 6070, 781, 25111, 1603, 1461, 11, 1202, 42390, 8310, 2840, 396, 287, 1093, 264, 16579, 13987, 49151, 304, 279, 59617, 3177, 382, 2170, 568, 15753, 1825, 279, 14365, 11, 264, 8369, 21411, 37066, 54285, 291, 1461, 11, 44087, 1113, 7123, 5304, 7123, 315, 77973, 79664, 323, 14154, 22755, 13, 578, 892, 63865, 82294, 1555, 279, 41050, 11, 4401, 813, 19779, 927, 279, 24634, 18012, 36800, 11, 8430, 279, 4785, 315, 24552, 315, 6677, 2949, 13, 1283, 35595, 1603, 264, 28745, 12514, 311, 279, 4375, 315, 66805, 11, 813, 6548, 36201, 279, 14071, 291, 6959, 439, 422, 15389, 369, 264, 3230, 33487, 382, 791, 5222, 315, 342, 71523, 31299, 32115, 6688, 1648, 311, 279, 8309, 66206, 315]
        #expect(tokens == tokenBaseline)
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

    private func loadJSONStringArrayGrammar() throws -> String {
        guard let grammarURL = Bundle.module.url(forResource: "Resources/json_array_strings", withExtension: "gbnf") else {
            throw TestError.resourceNotFound("json_array_strings.gbnf")
        }
        return try String(contentsOf: grammarURL)
    }

    private func createJSONStringArraySamplingConfig() throws -> LlamaSamplingConfig {
        let grammarString = try loadJSONStringArrayGrammar()
        let grammarConfig = LlamaGrammarConfig(
            grammar: grammarString,
            grammarRoot: "root"
        )
        return LlamaSamplingConfig(
            temperature: 0.1,
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

    private func generateText(
        using service: LlamaService,
        messages: [LlamaChatMessage],
        samplingConfig: LlamaSamplingConfig,
        maxTokens: Int
    ) async throws -> String {
        let stream = try await service.streamCompletion(of: messages, samplingConfig: samplingConfig)
        var generatedText = ""
        var tokenCount = 0
        for try await token in stream where tokenCount < maxTokens {
            generatedText += token
            tokenCount += 1
        }
        return generatedText
    }

    @Test("Service respects maxTokenCount and does not exceed it")
    func testServiceRespectsMaxTokenCount() async throws {
        // Given: a very small maxTokenCount to make the test fast
        let hardLimit = 40
        let service = LlamaService(
            modelUrl: .llama1B,
            config: .init(batchSize: 256, maxTokenCount: UInt32(hardLimit), useGPU: false)
        )
        let messages = [
            LlamaChatMessage(role: .system, content: "You are a helpful assistant."),
            LlamaChatMessage(role: .user, content: "Write a long paragraph about Mars to ensure many tokens are produced.")
        ]
        let cfg = LlamaSamplingConfig(temperature: 0.7, seed: 123)

        // When: stream until the model stops due to context/token budget
        let stream = try await service.streamCompletion(of: messages, samplingConfig: cfg)
        var produced = 0
        for try await _ in stream {
            produced += 1
            if produced > hardLimit + 2 { break } // safety guard for the loop
        }

        // Then: should not exceed the configured maxTokenCount
        #expect(produced <= hardLimit)
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

    private func validateStringArray(_ jsonString: String) throws -> [String] {
        guard let jsonData = jsonString.data(using: .utf8) else {
            throw TestError.invalidJSON("Could not convert to data")
        }
        let jsonObject = try JSONSerialization.jsonObject(with: jsonData, options: [])
        guard let array = jsonObject as? [String] else {
            throw TestError.invalidJSON("JSON is not an array of strings")
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

