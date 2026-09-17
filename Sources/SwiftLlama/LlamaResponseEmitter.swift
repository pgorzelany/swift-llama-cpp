#if compiler(>=6.4) && canImport(FoundationModels)
import Foundation
import FoundationModels

@available(iOS 27.0, macOS 27.0, *)
struct LlamaResponseEmitter {
    let requestID: UUID
    let inputTokens: Int
    let start: ContinuousClock.Instant
    let channel: LanguageModelExecutorGenerationChannel
    private(set) var tokenCount = 0
    private var firstTokenTime: Double?
    private var parser = LlamaReasoningParser()
    private var hasAnswer = false
    private var answerID: String { "\(requestID)-answer" }

    mutating func append(_ text: String) async {
        tokenCount += 1
        if firstTokenTime == nil { firstTokenTime = start.duration(to: .now).llamaSeconds }
        await send(parser.append(text))
        await publishMetadata(finished: false)
    }

    mutating func finish() async {
        await send(parser.finish())
        await channel.send(.response(entryID: answerID, action: .updateUsage(
            input: .init(totalTokenCount: inputTokens, cachedTokenCount: 0),
            output: .init(totalTokenCount: tokenCount, reasoningTokenCount: 0)
        )))
        await publishMetadata(finished: true)
    }

    private mutating func send(_ fragments: [LlamaReasoningParser.Fragment]) async {
        for fragment in fragments {
            switch fragment {
            case .answer(let text):
                if !hasAnswer {
                    await channel.send(.response(entryID: answerID, action: .updateMetadata(metadata(finished: false))))
                    hasAnswer = true
                }
                await channel.send(.response(entryID: answerID, action: .appendText(text, segmentID: "answer", tokenCount: 0)))
            case .reasoningStarted(let index):
                let id = "\(requestID)-reasoning-\(index)"
                await channel.send(.reasoning(entryID: id, action: .updateMetadata(metadata(finished: false))))
                await channel.send(.reasoning(entryID: id, action: .appendText("", segmentID: "thought", tokenCount: 0)))
            case .reasoning(let index, let text):
                await channel.send(.reasoning(entryID: "\(requestID)-reasoning-\(index)", action: .appendText(text, segmentID: "thought", tokenCount: 0)))
            case .reasoningFinished: break
            }
        }
    }

    private func publishMetadata(finished: Bool) async {
        guard finished || hasAnswer || !parser.reasoning.isEmpty else { return }
        if !finished, !parser.reasoning.isEmpty, parser.isReasoning || !hasAnswer {
            await channel.send(.reasoning(entryID: "\(requestID)-reasoning-\(parser.reasoning.count - 1)", action: .updateMetadata(metadata(finished: false))))
        } else {
            await channel.send(.response(entryID: answerID, action: .updateMetadata(metadata(finished: finished))))
        }
    }

    private func metadata(finished: Bool) -> [String: any ConvertibleToGeneratedContent] {
        typealias Key = LlamaLanguageModel.Metadata
        var values: [String: any ConvertibleToGeneratedContent] = [
            Key.requestID: requestID.uuidString,
            Key.rawOutput: parser.rawText,
            Key.isReasoning: parser.isReasoning,
            Key.finished: finished,
            Key.usageReported: true,
            Key.inputTokensKnown: true,
            Key.inputTokens: inputTokens,
            Key.outputTokens: tokenCount,
            // Tagged text does not give us an exact tokenizer-level reasoning split.
            Key.reasoningTokensKnown: false
        ]
        if let firstTokenTime { values[Key.timeToFirstToken] = firstTokenTime }
        let elapsed = start.duration(to: .now).llamaSeconds
        if finished, elapsed > 0 { values[Key.tokensPerSecond] = Double(tokenCount) / elapsed }
        return values
    }
}

private extension Duration {
    var llamaSeconds: Double {
        Double(components.seconds) + Double(components.attoseconds) / 1e18
    }
}

/// Keeps delimiter fragments out of visible answers and retains exact generated text for replay.
struct LlamaReasoningParser: Sendable {
    enum Fragment: Equatable, Sendable {
        case answer(String)
        case reasoningStarted(Int)
        case reasoning(Int, String)
        case reasoningFinished(Int)
    }

    private(set) var rawText = ""
    private(set) var answer = ""
    private(set) var reasoning: [String] = []
    private(set) var isReasoning = false
    private var closingDelimiter = "</think>"
    private var pending = ""

    mutating func append(_ text: String) -> [Fragment] {
        rawText += text
        pending += text
        var fragments: [Fragment] = []
        while !pending.isEmpty {
            let delimiters = isReasoning ? [closingDelimiter] : ["<think>", "<|channel>thought\n"]
            let match = delimiters.compactMap { delimiter in
                pending.range(of: delimiter).map { (delimiter: delimiter, range: $0) }
            }.min { $0.range.lowerBound < $1.range.lowerBound }
            if let match {
                let range = match.range
                emit(String(pending[..<range.lowerBound]), into: &fragments)
                pending = String(pending[range.upperBound...])
                if isReasoning {
                    fragments.append(.reasoningFinished(reasoning.count - 1))
                } else {
                    closingDelimiter = match.delimiter == "<think>" ? "</think>" : "<channel|>"
                    reasoning.append("")
                    fragments.append(.reasoningStarted(reasoning.count - 1))
                }
                isReasoning.toggle()
            } else {
                let held = delimiters.map { delimiter in
                    (1..<delimiter.count).reversed().first { pending.hasSuffix(delimiter.prefix($0)) } ?? 0
                }.max() ?? 0
                emit(String(pending.dropLast(held)), into: &fragments)
                pending = String(pending.suffix(held))
                break
            }
        }
        return fragments
    }

    mutating func finish() -> [Fragment] {
        var fragments: [Fragment] = []
        emit(pending, into: &fragments)
        pending = ""
        if isReasoning { fragments.append(.reasoningFinished(reasoning.count - 1)) }
        isReasoning = false
        return fragments
    }

    private mutating func emit(_ text: String, into fragments: inout [Fragment]) {
        guard !text.isEmpty else { return }
        if isReasoning {
            reasoning[reasoning.count - 1] += text
            fragments.append(.reasoning(reasoning.count - 1, text))
        } else {
            answer += text
            fragments.append(.answer(text))
        }
    }
}
#endif
