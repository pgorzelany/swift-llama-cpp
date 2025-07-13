# swift-llama-cpp

Run any LLM locally on iOS or MacOS. Powered by [llama.cpp](https://github.com/ggml-org/llama.cpp) 

## Usage

Here is a quick example of how to use `SwiftLlama` to generate text from a model.

First, make sure you have a GGUF model file accessible in your project. You can download models from sources like [Hugging Face](https://huggingface.co/models?search=gguf).

```swift
import SwiftLlama
import Foundation

func generateText() async {
    // 1. Get the model URL
    // Make sure to add a GGUF model to your project and get its URL.
    guard let modelUrl = Bundle.main.url(forResource: "your-model-name", withExtension: "gguf") else {
        print("Model file not found")
        return
    }

    // 2. Initialize the LlamaService
    // This service manages the model and context.
    let llamaService = LlamaService(modelUrl: modelUrl, config: .init())

    // 3. Prepare your messages
    // The conversation history can be provided as an array of messages.
    let messages = [
        LlamaChatMessage(role: .system, content: "You are a helpful assistant."),
        LlamaChatMessage(role: .user, content: "Tell me a short story."),
    ]

    // 4. Generate text
    // The `streamCompletion` method returns an `AsyncThrowingStream` of tokens.
    do {
        let stream = try await llamaService.streamCompletion(of: messages)
        var generatedText = ""
        print("Generated text: ")
        for try await token in stream {
            // As tokens are generated, they are appended to the generatedText variable
            // and printed to the console.
            generatedText += token
            print(token, terminator: "")
        }
    } catch {
        print("Error generating text: \(error.localizedDescription)")
    }
}
``` 
