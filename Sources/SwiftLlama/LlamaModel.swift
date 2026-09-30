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

public final class LlamaModel {

    // MARK: - Properties

    let modelPointer: OpaquePointer
    let vocabPointer: OpaquePointer

    // MARK: - Lifecycle

    public init?(path: String, parameters: llama_model_params = llama_model_default_params()) {
        LlamaBackend.acquireModel()
        guard let modelPointer = llama_model_load_from_file(path, parameters) else {
            LlamaBackend.releaseModel()
            return nil
        }
        guard let vocabPointer = llama_model_get_vocab(modelPointer) else {
            llama_model_free(modelPointer)
            LlamaBackend.releaseModel()
            return nil
        }
        self.modelPointer = modelPointer
        self.vocabPointer = vocabPointer
    }

    /// Initializes a model from multiple GGUF split files.
    /// The `paths` must be ordered correctly.
    public init?(paths: [String], parameters: llama_model_params = llama_model_default_params()) {
        guard !paths.isEmpty else { return nil }
        LlamaBackend.acquireModel()
        var cStrings: [UnsafeMutablePointer<CChar>?] = paths.map { strdup($0) }
        defer { cStrings.forEach { if let p = $0 { free(UnsafeMutablePointer(mutating: p)) } } }
        guard cStrings.allSatisfy({ $0 != nil }) else {
            LlamaBackend.releaseModel()
            return nil
        }
        let count = cStrings.count
        let result = cStrings.withUnsafeMutableBufferPointer { buf in
            buf.baseAddress!.withMemoryRebound(to: UnsafePointer<CChar>?.self, capacity: count) { reboundPtr in
                llama_model_load_from_splits(reboundPtr, size_t(count), parameters)
            }
        }
        guard let modelPointer = result else {
            LlamaBackend.releaseModel()
            return nil
        }
        guard let vocabPointer = llama_model_get_vocab(modelPointer) else {
            llama_model_free(modelPointer)
            LlamaBackend.releaseModel()
            return nil
        }
        self.modelPointer = modelPointer
        self.vocabPointer = vocabPointer
    }

    deinit {
        llama_model_free(modelPointer)
        LlamaBackend.releaseModel()
    }

    // MARK: - Methods

    // Helper to convert a null-terminated CChar buffer into Swift String without deprecation warnings
    private static func stringFromNullTerminated(_ buffer: [CChar]) -> String {
        let units: [UInt8] = buffer.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }
        return String(decoding: units, as: UTF8.self)
    }

    private static func readString(_ getter: (UnsafeMutablePointer<CChar>, Int) -> Int32) -> String? {
        var buffer = [CChar](repeating: 0, count: 512)
        var required = getter(&buffer, buffer.count)
        if required >= buffer.count {
            buffer = [CChar](repeating: 0, count: Int(required) + 1)
            required = getter(&buffer, buffer.count)
        }
        guard required >= 0, required < buffer.count else { return nil }
        return String(decoding: buffer.prefix(Int(required)).map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }

    /// Text context size used during training.
    public func trainedContextSize() -> Int32 {
        llama_model_n_ctx_train(modelPointer)
    }

    /// A string describing the model type.
    public func description() -> String {
        Self.readString { llama_model_desc(modelPointer, $0, $1) } ?? ""
    }

    /// Render token text for a token id.
    public func string(from token: llama_token) -> String {
        guard token >= 0, token < vocabularySize(), let results = llama_vocab_get_text(vocabPointer, token) else {
            return ""
        }
        return String(cString: results, encoding: .utf8) ?? ""
    }

    /// Returns the exact bytes of a token; pieces can end inside a UTF-8 character.
    public func pieceBytes(from token: llama_token, renderSpecial: Bool = false, lstrip: Int32 = 0) -> [UInt8] {
        guard token >= 0, token < vocabularySize(), lstrip >= 0 else { return [] }
        var buffer = [CChar](repeating: 0, count: 64)
        var written = llama_token_to_piece(vocabPointer, token, &buffer, Int32(buffer.count), lstrip, renderSpecial)
        if written < 0 {
            guard written != Int32.min else { return [] }
            buffer = [CChar](repeating: 0, count: Int(-written))
            written = llama_token_to_piece(vocabPointer, token, &buffer, Int32(buffer.count), lstrip, renderSpecial)
        }
        guard written >= 0, written <= buffer.count else { return [] }
        return buffer.prefix(Int(written)).map { UInt8(bitPattern: $0) }
    }

    /// Renders one piece, replacing incomplete UTF-8. Use pieceBytes with a streaming decoder for generation.
    public func piece(from token: llama_token, renderSpecial: Bool = false, lstrip: Int32 = 0) -> String {
        String(decoding: pieceBytes(from: token, renderSpecial: renderSpecial, lstrip: lstrip), as: UTF8.self)
    }

    /// Beginning-of-sentence token id.
    public func bosToken() -> llama_token {
        llama_vocab_bos(vocabPointer)
    }

    /// Whether a BOS token should be added automatically.
    public func shouldAddBos() -> Bool {
        llama_vocab_get_add_bos(vocabPointer)
    }

    /// End-of-sentence token id.
    public func eosToken() -> llama_token {
        llama_vocab_eos(vocabPointer)
    }

    /// Whether the token is an end-of-generation token (e.g. EOS/EOT).
    public func isEogToken(_ token: llama_token) -> Bool {
        llama_vocab_is_eog(vocabPointer, token)
    }

    /// Convert the provided text into tokens.
    /// - Parameters:
    ///   - addBos: Allow to add BOS/EOS if model is configured so.
    ///   - special: Allow tokenizing special/control tokens.
    public func tokenize(text: String, addBos: Bool, special: Bool) -> [llama_token] {
        guard !text.isEmpty else {
            return []
        }
        guard let utf8Count = Int32(exactly: text.utf8.count) else { return [] }
        let required = llama_tokenize(vocabPointer, text, utf8Count, nil, 0, addBos, special)
        guard required < 0, required != Int32.min else { return [] }
        var tokensBuffer = [llama_token](repeating: 0, count: Int(-required))
        let written = llama_tokenize(vocabPointer, text, utf8Count, &tokensBuffer, -required, addBos, special)
        guard written >= 0, written <= tokensBuffer.count else { return [] }
        return Array(tokensBuffer.prefix(Int(written)))
    }

    /// Convert tokens back to text (inverse of tokenize)
    /// Convert tokens back to text (inverse of tokenize()).
    public func detokenize(tokens: [llama_token], removeSpecial: Bool = true, unparseSpecial: Bool = false) -> String {
        guard !tokens.isEmpty else { return "" }
        // Heuristic buffer: tokens * avg 4 bytes + 16
        guard tokens.count <= Int(Int32.max), tokens.allSatisfy({ $0 >= 0 && $0 < vocabularySize() }) else { return "" }
        var bufSize: Int32 = 64
        var buffer = [CChar](repeating: 0, count: Int(bufSize))
        var written: Int32 = -1
        repeat {
            written = tokens.withUnsafeBufferPointer { ptr in
                llama_detokenize(vocabPointer, ptr.baseAddress, Int32(tokens.count), &buffer, bufSize, removeSpecial, unparseSpecial)
            }
            if written < 0 { // need bigger buffer
                guard written != Int32.min else { return "" }
                bufSize = -written
                buffer = [CChar](repeating: 0, count: Int(bufSize))
            }
        } while written < 0
        return String(decoding: buffer.prefix(Int(written)).map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }

    /// Number of tokens in the vocabulary.
    public func vocabularySize() -> Int32 {
        llama_vocab_n_tokens(vocabPointer)
    }

    /// Apply chat template using the default model template (or custom by name).
    public func applyChatTemplate(to messages: [LlamaChatMessage], addAssistant: Bool? = nil) -> String {
        guard !messages.isEmpty, messages.allSatisfy({ !$0.content.contains("\u{0}") }) else { return "" }
        if messages.contains(where: \.usesLFMToolTemplate) {
            // The verified LFM template uses ChatML with preserve_thinking=true. The C API
            // cannot render its tools parameter, so the mapper supplies the exact tool preamble.
            return Self.renderLFMToolPrompt(messages, bos: string(from: bosToken()), addAssistant: addAssistant ?? (messages.last?.role != .assistant))
        }
        let cTemplatePointer = llama_model_chat_template(modelPointer, nil)
        let shouldAddAssistant = addAssistant ?? (messages.last?.role != .assistant)

        // Convert Swift messages to C messages
        var cMessages = messages.map { message -> llama_chat_message in
           let roleCString = strdup(message.role.rawValue)
           let contentCString = strdup(message.content)
           return llama_chat_message(role: roleCString, content: contentCString)
        }

        // Initial buffer size
        let bufferSizeMultiplier = 3
        var bufferSize = max(1, bufferSizeMultiplier * messages.reduce(0) { $0 + $1.content.count })
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
               shouldAddAssistant,
               &buffer,
               Int32(bufferSize)
           )
        } while resultSize >= Int32(bufferSize)

        // Free the allocated C strings
        for message in cMessages {
           free(UnsafeMutablePointer(mutating: message.role))
           free(UnsafeMutablePointer(mutating: message.content))
        }

        let prompt = resultSize >= 0 ? String(decoding: buffer.prefix(Int(resultSize)).map { UInt8(bitPattern: $0) }, as: UTF8.self) : ""
        if prompt.isEmpty, metaValue(forKey: "general.architecture") == "gemma4" {
            return applyGemma4ChatTemplate(to: messages, addAssistant: shouldAddAssistant)
        }
        return prompt
    }

    static func renderLFMToolPrompt(_ messages: [LlamaChatMessage], bos: String, addAssistant: Bool) -> String {
        let turns = messages.first?.role == .system && messages.first?.content.isEmpty == true ? messages.dropFirst() : messages[...]
        var prompt = bos
        prompt += turns.map { "<|im_start|>\($0.role.rawValue)\n\($0.content)<|im_end|>\n" }.joined()
        if addAssistant { prompt += "<|im_start|>assistant\n" }
        return prompt
    }

    private func applyGemma4ChatTemplate(
        to messages: [LlamaChatMessage],
        addAssistant: Bool
    ) -> String {
        var prompt = string(from: bosToken())
        prompt += messages.map { message in
            let role = message.role == .assistant ? "model" : message.role.rawValue
            let content = message.content.trimmingCharacters(in: .whitespacesAndNewlines)
            return "<|turn>\(role)\n\(content)<turn|>\n"
        }.joined()

        if addAssistant {
            prompt += "<|turn>model\n"
        }
        return prompt
    }

    /// Apply chat template by template name found in the model.
    public func applyChatTemplate(name: String, to messages: [LlamaChatMessage], addAssistant: Bool? = nil) -> String {
        guard !messages.isEmpty, messages.allSatisfy({ !$0.content.contains("\u{0}") }) else { return "" }
        let cTemplatePointer = name.withCString { cname in
            llama_model_chat_template(modelPointer, cname)
        }
        guard cTemplatePointer != nil else { return "" }
        // Convert Swift messages to C messages
        var cMessages = messages.map { message -> llama_chat_message in
           let roleCString = strdup(message.role.rawValue)
           let contentCString = strdup(message.content)
           return llama_chat_message(role: roleCString, content: contentCString)
        }
        let bufferSizeMultiplier = 3
        var bufferSize = max(1, bufferSizeMultiplier * messages.reduce(0) { $0 + $1.content.utf8.count })
        var buffer = [CChar](repeating: 0, count: bufferSize)
        var resultSize: Int32 = 0
        repeat {
            if resultSize >= Int32(bufferSize) {
                bufferSize = Int(resultSize + 1)
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
        for message in cMessages {
            free(UnsafeMutablePointer(mutating: message.role))
            free(UnsafeMutablePointer(mutating: message.content))
        }
        guard resultSize >= 0 else { return "" }
        return String(decoding: buffer.prefix(Int(resultSize)).map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }

    /// Total number of parameters in the model.
    public func numberOfParameters() -> UInt64 {
        return llama_model_n_params(modelPointer)
    }

    // MARK: - Model & Vocab Introspection

    /// Model and vocab introspection helpers.
    public func ropeType() -> llama_rope_type { llama_model_rope_type(modelPointer) }
    public func nEmbed() -> Int32 { llama_model_n_embd(modelPointer) }
    public func nEmbedOutput() -> Int32 { llama_model_n_embd_out(modelPointer) }
    public func nLayer() -> Int32 { llama_model_n_layer(modelPointer) }
    public func nHead() -> Int32 { llama_model_n_head(modelPointer) }
    public func nHeadKV() -> Int32 { llama_model_n_head_kv(modelPointer) }
    public func nSWA() -> Int32 { llama_model_n_swa(modelPointer) }
    public func ropeFreqScaleTrain() -> Float { llama_model_rope_freq_scale_train(modelPointer) }
    public func nClassifierOutputs() -> UInt32 { llama_model_n_cls_out(modelPointer) }
    public func classifierLabel(at index: UInt32) -> String? {
        guard let cstr = llama_model_cls_label(modelPointer, index) else { return nil }
        return String(cString: cstr)
    }
    public func modelSizeBytes() -> UInt64 { llama_model_size(modelPointer) }
    public func hasEncoder() -> Bool { llama_model_has_encoder(modelPointer) }
    public func hasDecoder() -> Bool { llama_model_has_decoder(modelPointer) }
    public func decoderStartToken() -> llama_token { llama_model_decoder_start_token(modelPointer) }
    public func isRecurrent() -> Bool { llama_model_is_recurrent(modelPointer) }
    public func isDiffusion() -> Bool { llama_model_is_diffusion(modelPointer) }

    // Vocab helpers
    public func vocabType() -> llama_vocab_type { llama_vocab_type(vocabPointer) }
    public func vocabScore(for token: llama_token) -> Float { llama_vocab_get_score(vocabPointer, token) }
    public func vocabAttr(for token: llama_token) -> llama_token_attr { llama_vocab_get_attr(vocabPointer, token) }
    public func isControl(token: llama_token) -> Bool { llama_vocab_is_control(vocabPointer, token) }
    public func sepToken() -> llama_token { llama_vocab_sep(vocabPointer) }
    public func nlToken() -> llama_token { llama_vocab_nl(vocabPointer) }
    public func padToken() -> llama_token { llama_vocab_pad(vocabPointer) }
    public func maskToken() -> llama_token { llama_vocab_mask(vocabPointer) }
    public func addEos() -> Bool { llama_vocab_get_add_eos(vocabPointer) }
    public func addSep() -> Bool { llama_vocab_get_add_sep(vocabPointer) }
    public func fimPre() -> llama_token { llama_vocab_fim_pre(vocabPointer) }
    public func fimSuf() -> llama_token { llama_vocab_fim_suf(vocabPointer) }
    public func fimMid() -> llama_token { llama_vocab_fim_mid(vocabPointer) }
    public func fimPad() -> llama_token { llama_vocab_fim_pad(vocabPointer) }
    public func fimRep() -> llama_token { llama_vocab_fim_rep(vocabPointer) }
    public func fimSep() -> llama_token { llama_vocab_fim_sep(vocabPointer) }

    // Metadata
    /// Read model GGUF metadata value by key as string.
    public func metaValue(forKey key: String) -> String? {
        Self.readString { llama_model_meta_val_str(modelPointer, key, $0, $1) }
    }
    /// Number of model GGUF metadata key/value pairs.
    public func metaCount() -> Int32 { llama_model_meta_count(modelPointer) }
    /// Read metadata key name by index.
    public func metaKey(at index: Int32) -> String? {
        Self.readString { llama_model_meta_key_by_index(modelPointer, index, $0, $1) }
    }
    /// Read metadata value as a string by index.
    public func metaValue(at index: Int32) -> String? {
        Self.readString { llama_model_meta_val_str_by_index(modelPointer, index, $0, $1) }
    }

    // Save model
    /// Save the model to a file.
    public func save(to path: String) {
        llama_model_save_to_file(modelPointer, path)
    }

    // Built-in chat templates
    /// Get list of built-in chat templates.
    public func builtinChatTemplates(maxCount: Int = 64) -> [String] {
        guard maxCount > 0 else { return [] }
        var result: [String] = []
        var ptrs = Array<UnsafePointer<CChar>?>(repeating: nil, count: maxCount)
        let n = ptrs.withUnsafeMutableBufferPointer { buf in
            llama_chat_builtin_templates(buf.baseAddress, size_t(maxCount))
        }
        if n > 0 {
            for i in 0..<min(Int(n), maxCount) {
                if let p = ptrs[i] { result.append(String(cString: p)) }
            }
        }
        return result
    }

    // Quantize helper (wraps the C quantize function)
    @discardableResult
    /// Quantize a model file.
    public static func quantizeModel(inputPath: String, outputPath: String, params: inout llama_model_quantize_params) -> UInt32 {
        llama_model_quantize(inputPath, outputPath, &params)
    }

    /// Default quantization parameters.
    public static func defaultQuantizeParams() -> llama_model_quantize_params {
        llama_model_quantize_default_params()
    }

    // MARK: - Split utilities

    /// Build a split GGUF final path for this chunk.
    public static func splitPath(pathPrefix: String, splitNo: Int32, splitCount: Int32) -> String {
        guard splitNo >= 0, splitNo < splitCount else { return "" }
        var buf = [CChar](repeating: 0, count: pathPrefix.utf8.count + 64)
        _ = pathPrefix.withCString { c in
            llama_split_path(&buf, buf.count, c, splitNo, splitCount)
        }
        return Self.stringFromNullTerminated(buf)
    }

    /// Extract the path prefix from a split path if and only if the split_no and split_count match.
    public static func splitPrefix(splitPath: String, splitNo: Int32, splitCount: Int32) -> String? {
        guard splitNo >= 0, splitNo < splitCount else { return nil }
        var buf = [CChar](repeating: 0, count: splitPath.utf8.count + 1)
        let n = splitPath.withCString { c in
            llama_split_prefix(&buf, buf.count, c, splitNo, splitCount)
        }
        if n <= 0 { return nil }
        return Self.stringFromNullTerminated(buf)
    }
}
