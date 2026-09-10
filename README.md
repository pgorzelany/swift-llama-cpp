# swift-llama-cpp

Run any LLM locally on iOS or MacOS. Powered by [llama.cpp](https://github.com/ggml-org/llama.cpp)

To browse upstream C/C++ source at the same revision as the pinned xcframework, see [Reference/README.md](Reference/README.md).

## Running model-backed tests

The test suite runs natively on macOS and uses an ignored 808 MB Llama 3.2 GGUF fixture:

```bash
./Scripts/download-test-model.sh
swift test --no-parallel
```

The download script verifies the fixture's SHA-256 before installing it. To run the optional Gemma 4 compatibility test against a local GGUF file:

```bash
GEMMA4_GGUF_PATH=/absolute/path/to/gemma-4.gguf swift test --filter GemmaCompatibilityTests
```

## Coverage

This wrapper covers:
- Model loading (single file and splits), save, metadata, size, params, encoder/decoder flags
- Vocab API (token text, score, attrs, special tokens), tokenize/detokenize
- Context creation/free, threads, embeddings/attention/warmup toggles
- Memory API (sequence remove/copy/keep/add/div/min/max/canShift)
- Encode/decode, logits/embeddings getters, synchronize
- State/session save-load, per-sequence state
- Chat templates (apply and list built-ins)
- Sampler chain (grammar, top-k, top-p, temp, penalties, dist), sample/accept/reset/clone
- LoRA adapter load/apply/remove/clear, control vectors
- Backend init/free, capability queries, system info, logging hook

## Apple sessions (iOS 27 / macOS 27)

The existing `SwiftLlama` product also exposes `LlamaLanguageModel` and `LlamaLanguageModelExecutor`. These APIs require Swift 6.4 and iOS/macOS 27; older compilers and deployment targets retain the existing `LlamaService` API without a package-wide deployment bump.

```swift
import Foundation
import FoundationModels
import SwiftLlama

@available(iOS 27.0, macOS 27.0, *)
func answer(using modelURL: URL) async throws -> String {
    let model = LlamaLanguageModel(
        modelURL: modelURL,
        configuration: .init(batchSize: 256, maxTokenCount: 4096)
    )
    let session = LanguageModelSession(model: model, instructions: "Be helpful.")
    do {
        try await model.prewarm(transcript: session.transcript)
        let response = try await session.respond(to: "What is the capital of France?")
        await model.unload()
        return response.content
    } catch {
        await model.unload()
        throw error
    }
}
```

The executor drives the internal `Llama` actor directly: no `LlamaService` or intermediate completion stream. Create a separate model instance for each conversation owner; copies share the same execution resources. Concurrent generations on one model are rejected. For Stop, cancel the consuming task and await `model.cancelAndWait()` before reusing the model. `unload()` also awaits warmup/inference before releasing weights and context. Cancellation is checked between decoded tokens and prompt batches; it cannot interrupt a synchronous C decode already in progress.

Text generation supports temperature, seed, greedy/top-k/top-p sampling, output limits, prompt-cache reuse, tagged thinking and exact raw replay. Apple transcript reasoning entries expose the tagged thinking separately from answer text. `LlamaLanguageModel.Metadata` contains raw output, measured token totals and timing. The reasoning-token split and cached-token count are not measured; native usage uses zero for those required fields, and metadata marks reasoning counts as unknown. Do not present those zeros as measured counts.

Vision, tool calling, guided generation and reasoning-effort controls are not advertised; requesting them fails explicitly. This does not remove the legacy service's grammar APIs.

The `LlamaLanguageModelTests` suite exercises real Apple sessions, lifecycle/cancellation, reasoning/replay, sampling, token limits and synthetic overhead, plus the ignored GGUF fixture. Run it on an iOS 27 simulator or macOS 27 host; a newer SDK alone cannot run these APIs on macOS 26. Run simulator suites sequentially, not concurrently against the same simulator.

```bash
DEVELOPER_DIR=/Applications/Xcode-27.0.0-Beta.5.app/Contents/Developer \
  xcodebuild -scheme swift-llama-cpp \
  -destination 'platform=iOS Simulator,OS=27.0,name=iPhone 17 Pro' \
  -derivedDataPath /tmp/SwiftLlamaSessionTests \
  -only-testing:SwiftLlamaTests/LlamaLanguageModelTests \
  -parallel-testing-enabled NO CODE_SIGNING_ALLOWED=NO test
```

## Legacy service usage

Here is a quick example of how to use `SwiftLlama` to generate text from a model.

First, make sure you have a GGUF model file accessible in your project. You can download models from sources like [Hugging Face](https://huggingface.co/models?search=gguf).

```swift
import SwiftLlama
import Foundation

// 1. Get the model URL
// Make sure to add a GGUF model to your project and get its URL.
guard let modelUrl = Bundle.main.url(forResource: "your-model-name", withExtension: "gguf") else {
    print("Model file not found")
    return
}

// 2. Initialize the LlamaService
// This service manages the model and context.
let llamaService = LlamaService(modelUrl: modelUrl, config: .init(batchSize: 256, maxTokenCount: 4096, useGPU: true))

// 3. Prepare your messages
// The conversation history can be provided as an array of messages.
let messages = [
    LlamaChatMessage(role: .system, content: "You are a helpful assistant."),
    LlamaChatMessage(role: .user, content: "Tell me a short story."),
]

// 4. Generate text
// The `streamCompletion` method returns an `AsyncThrowingStream` of tokens.
do {
    let stream = try await llamaService.streamCompletion(of: messages, samplingConfig: .init(temperature: 0.8, seed: 42))
    var generatedText = ""
    for try await token in stream {
        generatedText += token
        print("Generated token: \(token)")
    }
    print("Generated text: \(generatedText)")
} catch {
    print("Error generating text: \(error.localizedDescription)")
}
``` 
