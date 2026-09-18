# swift-llama-cpp

Run any LLM locally on iOS or MacOS. Powered by [llama.cpp](https://github.com/ggml-org/llama.cpp)

The package pins llama.cpp v0.4.1 through the `b10964` Apple XCFramework, with its release SHA-256 verified by SwiftPM.

## iOS simulator setup

Upstream [stopped including simulator slices in release binaries](https://github.com/ggml-org/llama.cpp/pull/27252). Before building for an iOS simulator, install CMake and run this once with Xcode selected:

```bash
bash Scripts/prepare-apple-framework.sh
```

The script verifies the official release archive and the pinned source commit, builds the arm64/x86_64 simulator slice, and combines it with the unchanged official macOS/iOS device binaries and debug symbols. SwiftPM automatically uses the resulting ignored `Artifacts/llama-b10964.xcframework`. Run the script before opening Xcode or resolving packages. If the project has already resolved the remote binary, use Xcode's **File → Packages → Reset Package Caches**, then resolve again so its cached manifest picks up the local artifact. Repeat preparation after changing the runtime pin. A checkout without this local artifact uses the official binary and supports macOS and iOS devices only.

After a runtime upgrade, clean SwiftPM and Xcode build products to avoid stale C API layouts. When updating the pin, update `llamaVersion`, `llamaChecksum`, and `llamaRevision` together in `Package.swift`.

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

Text generation supports temperature, seed, greedy/top-k/top-p sampling, output limits, prompt-cache reuse, tagged thinking and exact raw replay. The parser separates both `<think>…</think>` blocks and Gemma 4's `<|channel>thought\n…<channel|>` blocks into Apple transcript reasoning entries, retaining exact raw output for replay. `LlamaLanguageModel.Metadata` contains raw output, measured token totals and timing. The reasoning-token split and cached-token count are not measured; native usage uses zero for those required fields, and metadata marks reasoning counts as unknown. Do not present those zeros as measured counts.

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

## Foundation Models tool calling (iOS / macOS 27)

`LlamaLanguageModel` defaults to text-only capability. Opt in to the tested
LFM2.5-1.2B-Instruct-QAD-Q4_0 profile explicitly:

```swift
let model = LlamaLanguageModel(
    modelURL: modelURL,
    configuration: .init(batchSize: 512, maxTokenCount: 4096),
    capabilityProfile: .lfm2_5InstructQ4_0
)
let session = LanguageModelSession(model: model, tools: [myTool])
let answer = try await session.respond(to: "Look up my record")
```

The profile pins the shipped GGUF SHA-256 and llama.cpp b10964. Loading fails if
its bytes do not match; filenames and embedded metadata do not grant tool
capability. Requalify this profile when changing the model, framework revision,
renderer, or parser. The app's downloaded models retain the default text profile.

The C chat API cannot supply tool schemas. This profile implements the shipped
LFM ChatML template's text/tool branches with `preserve_thinking=true`, renders
Foundation Models schemas as the model's `List of tools`, and preserves raw
assistant output in transcript metadata. It supports Pythonic keyword arguments,
JSON containers, multiple calls in one batch, and text before/after a call. It
validates the whole batch against enabled tool names and argument schemas before
sending native `toolCalls` events. Apple's session executes tools and requests
continuation; the wrapper does not run a separate agent loop. Tool failures thrown
by `Tool.call` follow Apple's session error policy. Tools may also return error
content for model-led recovery.

Allowed (default) and disallowed tool modes are supported. Required tool choice
is rejected explicitly; this profile does not force a call with a grammar.
Tool calling is a model capability, not a promise that every prompt selects the
right tool. The integration fixture uses opaque record values to prove the final
answer came from tool execution, while deterministic tests use a calculator.

Tests run entirely offline once the existing GGUF/framework fixtures are present:

```sh
swift test --filter 'LlamaToolCallingTests|LlamaLanguageModelTests'
ENCLAVE_GGUF_TEST_MODEL=/absolute/path/LFM2.5-1.2B-Instruct-QAD-Q4_0.gguf \
  swift test --filter LlamaToolCallingIntegrationTests
TEST_RUNNER_ENCLAVE_GGUF_TEST_MODEL=/absolute/path/LFM2.5-1.2B-Instruct-QAD-Q4_0.gguf \
  xcodebuild -scheme swift-llama-cpp -destination 'platform=iOS Simulator,name=iPhone 18 Pro' \
  -only-testing:SwiftLlamaTests/LlamaToolCallingTests \
  -only-testing:SwiftLlamaTests/LlamaToolCallingIntegrationTests \
  -parallel-testing-enabled NO CODE_SIGNING_ALLOWED=NO test
```

The real-model suite is opt-in through `ENCLAVE_GGUF_TEST_MODEL`. An explicitly
provided missing or incorrect model fails, rather than skipping. Ordinary parser
and native-session tests do not require the shipped LFM binary.
