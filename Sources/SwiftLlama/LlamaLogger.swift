//
//  LlamaLoger.swift
//  swift-llama-cpp
//
//  Created by Piotr Gorzelany on 22/07/2025.
//

import OSLog
import llama
import os

extension Logger {
    static func llama(category: String) -> Logger {
        return Logger(subsystem: "swift-llama-ccp", category: category)
    }
}

public enum LlamaLog {
    private static let sink = LlamaLogSink()

    /// Captures llama.cpp diagnostics and optionally forwards them to Unified Logging.
    public static func setLogger(_ logger: Logger?) {
        sink.setLogger(logger)
        installDiagnosticCapture()
    }

    static func installDiagnosticCapture() {
        sink.installIfNeeded {
            llama_log_set({ level, message, userData in
                guard let message, let userData else { return }
                let sink = Unmanaged<LlamaLogSink>.fromOpaque(userData).takeUnretainedValue()
                sink.record(level: level.rawValue, message: String(cString: message))
            }, Unmanaged.passUnretained(sink).toOpaque())
        }
    }

    static func marker() -> UInt64 {
        sink.marker()
    }

    static func diagnostics(since marker: UInt64) -> String? {
        sink.diagnostics(since: marker)
    }

    static func recordForTesting(level: UInt32, message: String) {
        sink.record(level: level, message: message)
    }

    static func resetForTesting() {
        sink.reset()
    }
}

private final class LlamaLogSink: Sendable {
    private struct Entry: Sendable {
        let sequence: UInt64
        let level: UInt32
        let message: String
    }

    private struct State: Sendable {
        var isInstalled = false
        var nextSequence: UInt64 = 0
        var lastLevel = GGML_LOG_LEVEL_INFO.rawValue
        var entries: [Entry] = []
        var logger: Logger?
    }

    private static let maximumEntryCount = 200
    private static let maximumDiagnosticCount = 12
    private static let maximumDiagnosticLength = 1_600

    private let state = OSAllocatedUnfairLock(initialState: State())

    func installIfNeeded(_ install: @Sendable () -> Void) {
        state.withLock { state in
            guard !state.isInstalled else { return }
            install()
            state.isInstalled = true
        }
    }

    func setLogger(_ logger: Logger?) {
        state.withLock { $0.logger = logger }
    }

    func marker() -> UInt64 {
        state.withLock { $0.nextSequence }
    }

    func record(level: UInt32, message: String) {
        let lines = message
            .split(whereSeparator: \Character.isNewline)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        guard !lines.isEmpty else { return }

        let logger = state.withLock { state -> Logger? in
            let effectiveLevel: UInt32
            if level == GGML_LOG_LEVEL_CONT.rawValue {
                effectiveLevel = state.lastLevel
            } else {
                effectiveLevel = level
                state.lastLevel = level
            }
            for line in lines {
                state.entries.append(Entry(sequence: state.nextSequence, level: effectiveLevel, message: line))
                state.nextSequence &+= 1
            }
            if state.entries.count > Self.maximumEntryCount {
                state.entries.removeFirst(state.entries.count - Self.maximumEntryCount)
            }
            return state.logger
        }

        guard let logger else { return }
        for line in lines {
            switch level {
            case GGML_LOG_LEVEL_DEBUG.rawValue:
                logger.debug("\(line)")
            case GGML_LOG_LEVEL_INFO.rawValue, GGML_LOG_LEVEL_CONT.rawValue:
                logger.info("\(line)")
            case GGML_LOG_LEVEL_WARN.rawValue:
                logger.warning("\(line)")
            case GGML_LOG_LEVEL_ERROR.rawValue:
                logger.error("\(line)")
            default:
                logger.log("\(line)")
            }
        }
    }

    func diagnostics(since marker: UInt64) -> String? {
        let entries = state.withLock { state in
            Array(state.entries.lazy.filter { $0.sequence >= marker })
        }
        guard !entries.isEmpty else { return nil }

        let importantEntries = entries.filter {
            $0.level == GGML_LOG_LEVEL_WARN.rawValue || $0.level == GGML_LOG_LEVEL_ERROR.rawValue
        }
        let selectedEntries = importantEntries.isEmpty ? entries : importantEntries
        let combined = selectedEntries
            .suffix(Self.maximumDiagnosticCount)
            .map(\.message)
            .joined(separator: "\n")
        guard combined.count > Self.maximumDiagnosticLength else { return combined }
        return String(combined.suffix(Self.maximumDiagnosticLength))
    }

    func reset() {
        state.withLock { state in
            state.entries.removeAll(keepingCapacity: true)
            state.nextSequence = 0
            state.lastLevel = GGML_LOG_LEVEL_INFO.rawValue
        }
    }
}
