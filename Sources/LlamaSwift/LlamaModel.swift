//
//  LlamaModel.swift
//  PrivateAI
//
//  Created by Piotr Gorzelany on 12/02/2024.
//

import Foundation
import llama

enum LlamaModelError: Error {
    case initializationError
}

final class LlamaModel {

    // MARK: - Properties

    let modelPointer: OpaquePointer
    let vocabPointer: OpaquePointer

    // MARK: - Lifecycle

    init?(path: String, parameters: llama_model_params = llama_model_default_params()) {
        guard let modelPointer = llama_model_load_from_file(path, parameters), let vocabPointer = llama_model_get_vocab(modelPointer) else {
            return nil
        }
        self.modelPointer = modelPointer
        self.vocabPointer = vocabPointer
    }

    deinit {
        llama_model_free(modelPointer)
    }

    // MARK: - Methods

    func trainedContextSize() -> Int32 {
        llama_model_n_ctx_train(modelPointer)
    }

    func description() -> String {
        let bufferSize = 1024
        var buffer = [CChar](repeating: 0, count: bufferSize)
        let descriptionBufferSize = llama_model_desc(modelPointer, &buffer, bufferSize)
        guard descriptionBufferSize > 0 else {
            fatalError("Something went wrong")
        }
        return String(cString: buffer)
    }

    func string(from token: llama_token) -> String {
        guard let results = llama_vocab_get_text(vocabPointer, token) else {
            return ""
        }
        return String(cString: results, encoding: .utf8) ?? ""
    }

    func piece(from token: llama_token) -> String {
        let bufferSize: Int32 = 32
        var buffer = [CChar](repeating: 0, count: Int(bufferSize))
        let charCount = llama_token_to_piece(vocabPointer, token, &buffer, bufferSize, 0, false)
        let chars = Array(buffer.prefix(upTo: Int(charCount))) + [0]
        return String(cString: chars, encoding: .utf8) ?? ""
    }

    func bosToken() -> llama_token {
        llama_vocab_bos(vocabPointer)
    }

    func shouldAddBos() -> Bool {
        let addBos = llama_vocab_get_add_bos(vocabPointer)
        if addBos {
            return llama_vocab_type(vocabPointer) == LLAMA_VOCAB_TYPE_SPM
        }
        return addBos
    }

    func eosToken() -> llama_token {
        llama_vocab_eos(modelPointer)
    }

    func isEogToken(_ token: llama_token) -> Bool {
        llama_vocab_is_eog(vocabPointer, token)
    }

    func tokenize(text: String, addBos: Bool, special: Bool) -> [llama_token] {
        guard !text.isEmpty else {
            return []
        }
        let utf8Count = text.utf8.count
        let maxTokens = trainedContextSize()
        let tokenBufferSize = utf8Count + (addBos ? 1 : 0) + 1
        var tokensBuffer = [llama_token](repeating: llama_token(), count: Int(tokenBufferSize))
        let tokenCount = llama_tokenize(vocabPointer, text, Int32(utf8Count), &tokensBuffer, maxTokens, addBos, special)
        return Array(tokensBuffer.prefix(upTo: Int(tokenCount)))
    }

    func vocabularySize() -> Int32 {
        llama_vocab_n_tokens(vocabPointer)
    }

    func applyChatTemplate(to messages: [LlamaChatMessage], addAssistant: Bool? = nil) -> String {
        let cTemplatePointer = llama_model_chat_template(modelPointer, nil)

        // Convert Swift messages to C messages
        var cMessages = messages.map { message -> llama_chat_message in
           let roleCString = strdup(message.role.rawValue)
           let contentCString = strdup(message.content)
           return llama_chat_message(role: roleCString, content: contentCString)
        }

        // Initial buffer size
        let bufferSizeMultiplier = 3
        var bufferSize = bufferSizeMultiplier * messages.reduce(0) { $0 + $1.content.count }
        var buffer = [CChar](repeating: 0, count: bufferSize)

        var resultSize: Int32 = 0
        repeat {
           // If the buffer was too small, increase the buffer size
           if resultSize >= Int32(bufferSize) {
               bufferSize = Int(resultSize + 1) // the buffer has to be null (0) terminated
               buffer = [CChar](repeating: 0, count: bufferSize)
           }

           resultSize = llama_chat_apply_template(
               cTemplatePointer,
               &cMessages,
               messages.count,
               addAssistant ?? (messages.last?.role != .assistant),
               &buffer,
               Int32(bufferSize)
           )
        } while resultSize >= Int32(bufferSize)

        // Free the allocated C strings
        for message in cMessages {
           free(UnsafeMutablePointer(mutating: message.role))
           free(UnsafeMutablePointer(mutating: message.content))
        }

        // Convert the C string buffer to a Swift string
        return String(cString: buffer)
    }

    func numberOfParameters() -> UInt64 {
        return llama_model_n_params(modelPointer)
    }
}
