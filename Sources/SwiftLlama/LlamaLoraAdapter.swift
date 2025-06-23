import Foundation
import llama

public enum LlamaLoraError: Error {
    case couldNotLoadAdapter
}

/// Wraps a `llama_adapter_lora` pointer, managing its lifecycle.
///
/// A LoRA (Low-Rank Adaptation) adapter is loaded from a file against a base `LlamaModel`.
/// It can then be applied to a `LlamaContext` to modify the model's behavior for inference.
public final class LlamaLoraAdapter {

    // MARK: - Properties

    let adapterPointer: OpaquePointer

    // MARK: - Lifecycle

    /// Loads a LoRA adapter from a file.
    /// - Parameters:
    ///   - model: The base `LlamaModel` to load the adapter against.
    ///   - path: The file path to the LoRA adapter.
    /// - Throws: `LlamaLoraError.couldNotLoadAdapter` if the adapter cannot be loaded.
    public init(model: LlamaModel, path: String) throws {
        guard let adapterPointer = llama_adapter_lora_init(model.modelPointer, path) else {
            throw LlamaLoraError.couldNotLoadAdapter
        }
        self.adapterPointer = adapterPointer
    }

    deinit {
        llama_adapter_lora_free(adapterPointer)
    }
} 