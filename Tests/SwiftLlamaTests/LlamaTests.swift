import XCTest
@testable import SwiftLlama

final class LlamaPerformanceTests: XCTestCase {

    // Properties to hold the Llama actor and any necessary configuration
    var sut: Llama!
    // Model Path using Bundle.module
    let modelPath = URL.llama1B.path
    let maxTokenCount: UInt32 = 2048
    let batchSize: UInt32 = 256
    let temperature: Float = 0.7
    let initialPrompt = "Tell me a very long story about mars colonization"

    override func setUpWithError() throws {
        try super.setUpWithError()
        // Initialize the Llama actor
        sut = try Llama(
            modelPath: modelPath,
            config: .init(batchSize: batchSize, maxTokenCount: maxTokenCount)
        )
    }

    override func tearDownWithError() throws {
        // Clean up
        sut = nil
        try super.tearDownWithError()
    }

    func testTokenGenerationSpeed() async throws {
        try await sut.initializeCompletion(messages: [LlamaChatMessage(role: .system, content: initialPrompt)])
        await sut.updateSamplingConfig(.init(temperature: temperature, seed: 0))

        // Define the number of tokens you want to generate
        let numberOfTokensToGenerate = 1000
        var tokensGenerated = 0
        var generatedText = ""

        // Measure the time taken to generate tokens
        let startTime = CFAbsoluteTimeGetCurrent()

        for _ in 0..<numberOfTokensToGenerate {
            let nextToken = try await sut.generateNextToken()
            switch nextToken {
            case .token(let token):
                tokensGenerated += 1
                generatedText += token
            case .endOfString:
                // If end of string is reached before generating all tokens
                break
            }
        }

        let endTime = CFAbsoluteTimeGetCurrent()
        let timeElapsed = endTime - startTime
        let tokensPerSecond = Double(tokensGenerated) / timeElapsed

        // Assert if needed or print the results
        print("Generated \(tokensGenerated) tokens in \(timeElapsed) seconds.")
        print("Speed: \(tokensPerSecond) tokens/second.")
        print("Generated text:\n\(generatedText)")

        // Example assertion (adjust threshold as needed)
        XCTAssert(tokensPerSecond > 50, "Token generation is too slow: \(tokensPerSecond) tokens/second.")
        XCTAssertEqual(generatedText, expectedCompletion)
    }
}

extension LlamaPerformanceTests {
    var expectedCompletion: String {
    """
    In the year 2154, humanity had finally achieved the impossible: establishing a thriving colony on Mars. The Red Planet, once considered too harsh and inhospitable for human habitation, had been transformed into a paradise of sorts. The Martian settlers, known as the Redbranders, had worked tirelessly to create a new society that would thrive on the unforgiving alien landscape.

    The initial colonization effort was led by a brilliant scientist and engineer named Dr. Sofia Patel, who had spent years studying the Martian environment and developing innovative technologies to sustain life on the red planet. The Redbranders' first settlement, named Nova Terra, was a sprawling city that stretched across the Martian surface, with towering skyscrapers and advanced life support systems.

    As the years passed, the Redbranders continued to expand their colony, establishing new settlements and habitats throughout the Martian terrain. They built state-of-the-art research facilities, conducting cutting-edge studies on Martian geology, atmosphere, and ecology. The Redbranders were also pioneers in terraforming, using advanced technologies to alter the Martian environment to make it more hospitable for human life.

    One of the most significant innovations was the development of a revolutionary new technology called the "Atmospheric Respiration System" (ARS). This system, which used advanced nanotechnology and artificial intelligence to clean the Martian atmosphere of toxic gases, allowed humans to breathe easily on the red planet. The ARS had also enabled the Redbranders to grow a variety of crops, using advanced hydroponics and aeroponics systems.

    As the colony grew, so did its population. The Redbranders were attracted to Mars by its potential for scientific discovery and exploration. Many of them came from all over the world, drawn by the promise of a new home on the red planet. The Martian society was a melting pot of cultures and backgrounds, with people from all walks of life coming together to build a new community.

    The government of Nova Terra was led by a council of leaders who were chosen for their wisdom, integrity, and dedication to the colony. The council was advised by a team of scientists and engineers who had spent years studying the Martian environment and developing strategies for its long-term sustainability.

    One of the most important figures in the council was Dr. Liam Chen, a brilliant astrobiologist who had spent years searching for signs of life on Mars. He had discovered evidence of ancient microbial life on the Martian surface, and his findings had sparked a new wave of scientific research into the planet's potential for supporting life.

    As the years passed, the Redbranders continued to explore Mars, discovering new wonders and marvels on every turn. They built a network of advanced research facilities, which included state-of-the-art laboratories, observatories, and libraries. The Redbranders were also pioneers in the field of astrobiology, using advanced technologies to search for signs of life on other planets.

    One of the most significant discoveries made by the Redbranders was the detection of evidence of ancient Martian civilization. The team led by Dr. Sofia Patel had discovered a massive, abandoned city on the Martian surface, hidden beneath layers of Martian regolith and debris.

    The city was estimated to be over 2 million years old, and it was a testament to the ingenuity and technological prowess of the Martian civilization that had once flourished on the red planet. The Redbranders were fascinated by this discovery, and they quickly set about studying the city and its artifacts in detail.

    As they delved deeper into the mystery of the ancient Martian civilization, the Redbranders began to uncover a complex web of technological innovations and scientific achievements. They discovered evidence of advanced propulsion systems, exotic energy sources, and even ancient artifacts that defied explanation.

    The discovery sparked a new wave of research and exploration, as the Redbranders sought to understand the secrets of the ancient Martian civilization. They built advanced databases and archives, storing vast amounts of information about the city and its artifacts.

    The discovery also sparked a new wave of scientific inquiry into the Martian environment. The Redbranders began to study the planet's geology, atmosphere, and ecology in greater detail than ever before. They discovered new species of plants and animals that had adapted to the Martian environment, and they began to develop advanced technologies for terraforming and engineering.

    As the years passed, the Redbranders continued to explore Mars, always seeking new discoveries and insights. They became a presence on the Martian surface that was felt by all who lived there, a testament to their hard work and dedication.

    In 2175, the council of leaders announced plans for a new era of growth and development on Mars. The Redbranders had proven themselves capable of thriving in the harsh Martian environment, and they were eager to take their colony to the next level.

    The council announced a new initiative called "Mars Terraforming Project," which would aim to transform the Martian surface into a habitable environment for human life. The project was led by Dr.
    """
    }
}
