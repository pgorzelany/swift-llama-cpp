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
    In the year 2154, humanity had finally achieved the impossible: establishing a thriving colony on Mars. The Red Planet, once considered too harsh and inhospitable for human habitation, had been transformed into a paradise of sorts. The Martian settlers, known as the Redbranders, had worked tirelessly to create a new society that would thrive in the unforgiving environment.

    The initial landing site was chosen carefully, selecting a region with minimal radiation exposure and stable temperatures. The first wave of settlers arrived on Mars in 2157, accompanied by state-of-the-art spacecraft designed for long-term habitation. These ships, dubbed "Mars Colonies," were equipped with cutting-edge life support systems, advanced robotics, and sustainable energy sources.

    As the years passed, the Redbranders continued to expand their colony, establishing a thriving agricultural sector, harnessing the Martian soil's mineral-rich resources, and developing innovative infrastructure. The colony was initially called Nova Terra, but it later became known as Mars Colony, or MC for short.

    The early settlers were a diverse group of scientists, engineers, and entrepreneurs who had left Earth to escape poverty, environmental degradation, and social unrest. They brought with them advanced technology, knowledge of sustainable living, and a deep understanding of the Martian environment. The colony's founders also established a rigorous testing program to ensure that their society would be self-sufficient and capable of adapting to any challenges that might arise.

    One of the most significant breakthroughs during this period was the development of a closed-loop life support system (CLSS). This marvel of engineering allowed the colony to recycle air, water, and waste, minimizing the need for resupply missions from Earth. The CLSS also enabled the colony to generate its own energy through solar panels, wind turbines, and advanced nuclear reactors.

    As the years went by, Mars Colony continued to grow and prosper. The settlers developed innovative technologies that harnessed the Martian environment's unique properties, such as the planet's low gravity, which allowed for the creation of massive 3D printing facilities. These facilities produced everything from consumer goods to complex infrastructure components, reducing reliance on Earth-based supply chains.

    The Martian economy flourished, driven by a thriving tourism industry catering to space enthusiasts and adventure-seekers. The colony established its own currency, Mars dollars (MD), which was pegged to the value of Earth dollars. The Redbranders also developed a robust healthcare system, built around cutting-edge medical research and access to advanced life support systems.

    As the 22nd century dawned, Mars Colony had become a beacon of hope for humanity's future. The settlers had created a self-sustaining society that was capable of thriving in the Martian environment, and they were eager to expand their knowledge and capabilities into the cosmos.

    However, not everyone was pleased with the progress made on Mars. A small but vocal group of Earth-based scientists and policymakers began to raise concerns about the long-term sustainability of the colony. They argued that Mars Colony's reliance on Earth supplies, lack of natural resources, and potential risks to global stability were serious concerns that needed to be addressed.

    The debate between those who supported continued expansion and those who advocated for caution was a contentious one, with each side presenting compelling arguments. Ultimately, the decision to continue growing and developing Mars Colony was made by the Martian government, which prioritized its citizens' well-being and security above all else.

    In 2178, the Martian Parliament approved the proposal to establish a permanent human settlement on Mars, paving the way for the colony's further growth and development. The Redbranders continued to thrive, building a new society that would be shaped by their experiences on the Red Planet.

    Years later, in 2203, the first interplanetary space station was launched into orbit around Mars, marking a significant milestone in human history. The Martian settlers had created a thriving community of entrepreneurs, scientists, and explorers who were ready to take on the challenges of exploring the cosmos.

    As we look back at the story of Mars Colony, it's clear that the Redbranders' determination, innovation, and resilience have inspired generations of space-faring humans. Their legacy serves as a testament to humanity's capacity for growth, adaptation, and progress in the face of uncertainty.

    But what happens next? The Martian government has announced plans to establish a permanent human settlement on the Moon by 2225, followed by Mars Colony itself, which will be operational by 2230. The Redbranders are preparing for their final mission, but they know that there will be no Planet B, only new frontiers waiting to be explored.

    The story of Mars Colony continues, with endless possibilities and challenges ahead. As humanity ventures further into the cosmos, the Redbranders' legacy serves as a beacon of hope and inspiration for generations to come. The universe is full of mysteries, but one thing is certain: the future belongs to those who are brave, determined, and willing to explore
    """
    }
}
