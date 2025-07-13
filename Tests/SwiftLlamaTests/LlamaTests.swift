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
    In the year 2154, humanity had finally achieved the impossible: establishing a thriving, self-sustaining colony on the red planet of Mars. The initial landing site, dubbed Nova Terra, was a small, cratered region in the Martian equatorial region. The first waves of settlers, led by the brilliant and driven Dr. Sofia Patel, began to arrive on the planet about 20 years ago.

    The early colonists faced numerous challenges, from the harsh Martian environment to the psychological strain of being isolated from Earth for so long. However, as the years went by, the settlers began to adapt, and the colony grew. New technologies and innovations were developed to overcome the difficulties, and the Martian settlers proved to be resourceful and resilient.

    The first generation of settlers established a thriving agricultural community, cultivating the Martian soil to produce enough food for the colony. The second generation focused on establishing a robust energy infrastructure, harnessing the planet's limited sunlight to generate power. The third generation, however, laid the groundwork for the colonization of Mars itself.

    A group of scientists, led by Dr. Liam Chen, were chosen to explore the possibility of terraforming Mars. They traveled to the Martian poles, where they discovered vast reserves of water ice and a thin atmosphere. The team developed a revolutionary new technology, dubbed "Mars-Dry," which used advanced nanotechnology to extract water from the Martian soil and condense it into usable liquid.

    The Mars-Dry technology was met with skepticism at first, but as the settlers began to notice the effects of this new resource, they realized its potential. The technology enabled the creation of a stable and sustainable atmosphere, allowing the colonists to establish a reliable food supply.

    The next step in the colonization process was the establishment of a robust infrastructure. The settlers built massive solar farms, harnessing the energy of the sun to power the colony. They also developed advanced water recycling systems, reducing the need for Earth-based supplies.

    As the colony grew, the settlers began to explore the Martian surface. The first expedition, led by Dr. Sofia Patel, reached the Martian equatorial region and discovered a vast, ancient riverbed. The team used advanced geological instruments to study the riverbed, learning about the planet's geological history.

    The findings of the expedition sparked a new wave of research and exploration. The settlers began to map the Martian surface, identifying vast areas of unexplored terrain. They also discovered evidence of ancient civilizations, dating back millions of years.

    One of the most significant discoveries was the presence of a massive, ancient city. The city, known as Aridos, was hidden deep within the Martian terrain, protected by towering mountain ranges and treacherous canyons. The team used advanced scanning technology to map the city's layout, revealing a sprawling metropolis with towering spires and grand architecture.

    The discovery of Aridos sent shockwaves throughout the Martian colony. The settlers realized that they had stumbled upon a major archaeological site, one that could hold the secrets of human civilization's past.

    As the years went by, the Martian colony continued to thrive. The settlers established a new society, one that was deeply connected to the planet and its unique environment. They developed advanced technologies to harness the Martian resources, and the planet became a thriving, self-sustaining ecosystem.

    The Martian colony became a beacon of hope for humanity, a symbol of what could be achieved through determination and innovation. The settlers continued to explore, discovering new resources and pushing the boundaries of what was thought possible.

    However, as the Martian colony continued to grow, new challenges arose. The settlers faced the ever-present threat of asteroid impacts, and the need to establish a reliable food supply became increasingly pressing.

    The fourth generation of settlers, led by Dr. Sofia Patel's own son, Dr. Rohan Patel, was tasked with addressing these challenges. They developed advanced technologies to deflect asteroids, and established a thriving aquaculture industry, using the Martian soil to produce fresh seafood.

    The fifth generation, led by Dr. Liam Chen's own daughter, Dr. Ava Chen, continued to push the boundaries of what was possible. They developed advanced terraforming technologies, using the Martian atmosphere to create a breathable air supply.

    The sixth generation, led by Dr. Sofia Patel's own granddaughter, Dr. Aria Patel, took the reins of the colony. They established a new era of exploration, using advanced technologies to search for new sources of resources and new worlds to colonize.

    As the years went by, the Martian colony continued to thrive. The settlers became a diverse, multigenerational society, each contributing their unique perspective and expertise to the colony's growth.

    But just as the colony seemed to be nearing its goal of self-sufficiency, a new challenge arose. A massive, long-lost civilization on Mars, known as the N'Tari, was discovered using advanced technologies to harness the planet's energy.

    The N'Tari civilization had been wiped out by a catastrophic event
    """
    }
}
